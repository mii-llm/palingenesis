"""Optimizer, LR scheduler, layer-wise LR decay, critical batch size."""

import logging
import math
import re

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ==============================================================================
# LAYER-WISE LEARNING RATE DECAY (LLRD)
# Paper: "One LR Doesn't Fit All" (arxiv:2605.22297, NeurIPS 2025)
# Effect: 1.5x speed + less catastrophic forgetting
#
# Early layers (general knowledge) get LOWER LR -> preserve pretrained features
# Late layers (task-specific) get HIGHER LR -> adapt faster to new task
# Formula: layer_lr = base_lr * decay^(num_layers - layer_idx)
# ==============================================================================


OPTIMIZERS = ("adamw", "muon", "adamw8bit", "lion8bit", "paged_adamw8bit")


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    llrd_decay: float = 1.0,
    use_muon: bool = False,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    """Build optimizer with optional LLRD, Muon, or 8-bit support.

    Options:
      - "adamw": standard AdamW (16 bytes/param)
      - "muon": hybrid Muon + AdamW (8 bytes/param, 1.5× convergence)
      - "adamw8bit": bitsandbytes 8-bit AdamW (6 bytes/param)
      - "lion8bit": bitsandbytes 8-bit Lion (4 bytes/param, sign-based)
      - "paged_adamw8bit": 8-bit AdamW whose states page to CPU memory under pressure
    """
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"train.optimizer={optimizer_name!r} is not one of {', '.join(OPTIMIZERS)}.")
    if use_muon or optimizer_name == "muon":
        return _build_muon_optimizer(model, lr, weight_decay)

    if optimizer_name in ("adamw8bit", "lion8bit", "paged_adamw8bit"):
        return _build_bnb_optimizer(model, lr, weight_decay, optimizer_name)

    if llrd_decay >= 1.0:
        return _build_simple_optimizer(model, lr, weight_decay)

    param_groups = _build_llrd_groups(model, lr, weight_decay, llrd_decay)

    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    use_fused = not is_distributed

    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        fused=use_fused,
        foreach=not use_fused,
    )

    num_groups = len(param_groups)
    min_lr = param_groups[0]["lr"]
    max_lr = param_groups[-1]["lr"]
    logger.info(f"LLRD optimizer: {num_groups} groups, LR range [{min_lr:.2e}, {max_lr:.2e}], decay={llrd_decay}")

    return optimizer


def _build_bnb_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    name: str,
) -> torch.optim.Optimizer:
    """8-bit optimizers via bitsandbytes. Massive memory savings.

    adamw8bit: 6 bytes/param (vs 16 for fp32 AdamW) — 62% savings
    lion8bit:  4 bytes/param (vs 16 for fp32 AdamW) — 75% savings

    Lion is sign-based (like Muon but simpler). Less memory than even Muon.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("bitsandbytes not installed. pip install bitsandbytes")

    decay_params = []
    no_decay_params = []

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in pname.lower() for k in ("bias", "norm", "embed")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"AdamW 8-bit (bitsandbytes): {total/1e6:.1f}M params, ~{total*6/1e9:.1f} GB optimizer memory")
    elif name == "lion8bit":
        # NO hidden LR scaling: config learning_rate is what the optimizer gets.
        # (A silent lr*3 here used to stack with configs that already adjusted
        # for Lion, and desynced AdamC/scheduler which read the config value.
        # Note the Lion paper recommends 3-10x SMALLER LR than AdamW, not larger.)
        optimizer = bnb.optim.Lion8bit(param_groups, lr=lr, betas=(0.95, 0.98))
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Lion 8-bit (bitsandbytes): {total/1e6:.1f}M params, lr={lr:.2e}, "
            f"~{total*4/1e9:.1f} GB optimizer memory"
        )
    elif name == "paged_adamw8bit":
        # Paged: automatically offloads optimizer states to CPU when GPU OOMs
        optimizer = bnb.optim.PagedAdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Paged AdamW 8-bit (bitsandbytes): {total/1e6:.1f}M params, auto-pages to CPU on OOM")
    else:
        raise ValueError(f"Unknown bnb optimizer: {name}")

    return optimizer


def _build_muon_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Hybrid Muon + AdamW optimizer.

    Muon for 2D weight matrices (1.5-2× faster, 50% less memory).
    AdamW for 1D parameters (embeddings, norms, biases).

    Returns a _HybridMuonAdamW wrapper that steps both optimizers.
    """
    from torch.optim import Muon

    muon_params = []
    adam_decay = []
    adam_no_decay = []

    muon_count = 0
    adam_count = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Muon orthogonalises hidden 2-D matrices; embeddings and the output head stay
        # on AdamW (as in the Muon paper), and so do stacked 3-D expert weights, which
        # torch's Muon does not accept.
        if p.ndim == 2 and not any(k in name.lower() for k in ("embed", "lm_head")):
            muon_params.append(p)
            muon_count += p.numel()
        elif any(k in name.lower() for k in ("bias", "norm")):
            adam_no_decay.append(p)
            adam_count += p.numel()
        else:
            adam_decay.append(p)
            adam_count += p.numel()

    # Muon LR is typically 10× AdamW LR for same convergence speed
    muon_lr = lr * 10

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=weight_decay)

    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})

    adam_opt = AdamW(adam_groups, lr=lr, betas=(0.9, 0.95)) if adam_groups else None

    total = muon_count + adam_count
    logger.info(
        f"Muon hybrid: {muon_count/1e6:.1f}M params (Muon, lr={muon_lr:.2e}) + "
        f"{adam_count/1e6:.1f}M params (AdamW, lr={lr:.2e}) = {total/1e6:.1f}M total"
    )

    return _HybridOptimizer(muon_opt, adam_opt)


class _HybridOptimizer(torch.optim.Optimizer):
    """Wraps two optimizers (Muon + AdamW) into one interface.

    Steps both optimizers, exposes unified param_groups for scheduler compatibility.
    """

    def __init__(self, primary: torch.optim.Optimizer, secondary: torch.optim.Optimizer | None):
        self._primary = primary
        self._secondary = secondary
        # Expose combined param_groups for LR scheduler
        self.param_groups = list(primary.param_groups)
        if secondary:
            self.param_groups.extend(secondary.param_groups)

    def step(self, closure=None):
        self._primary.step(closure)
        if self._secondary:
            self._secondary.step(closure)

    def zero_grad(self, set_to_none=True):
        self._primary.zero_grad(set_to_none=set_to_none)
        if self._secondary:
            self._secondary.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "primary": self._primary.state_dict(),
            "secondary": self._secondary.state_dict() if self._secondary else None,
        }

    def load_state_dict(self, state_dict):
        self._primary.load_state_dict(state_dict["primary"])
        if self._secondary and state_dict.get("secondary"):
            self._secondary.load_state_dict(state_dict["secondary"])


def _build_simple_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """Standard two-group optimizer (no LLRD)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name.lower() for k in ("bias", "norm", "embed")):
            no_decay.append(p)
        else:
            decay.append(p)

    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    use_fused = not is_distributed

    optimizer = AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        fused=use_fused,
        foreach=not use_fused,
    )

    # NOTE: Do NOT compile optimizer.step here — it breaks LambdaLR scheduler.
    # torch.compile is applied at the training-loop level instead.

    return optimizer


def _build_llrd_groups(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    decay: float,
) -> list[dict]:
    """Build per-layer param groups with decayed learning rates.

    Groups:
      - embed_tokens: lr * decay^(num_layers+1)
      - layer 0: lr * decay^num_layers
      - layer 1: lr * decay^(num_layers-1)
      - ...
      - layer N-1: lr * decay^1
      - lm_head + norm: lr (full rate)
    """
    # Discover layer count
    num_layers = 0
    for name, _ in model.named_parameters():
        m = re.search(r"layers?\.(\d+)\.", name)
        if m:
            num_layers = max(num_layers, int(m.group(1)) + 1)

    if num_layers == 0:
        # Can't determine layers, fall back to simple
        logger.warning("LLRD: could not detect transformer layers, using uniform LR")
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

    # Assign each param to its layer depth
    layer_params: dict[int, list] = {}  # depth -> list of (name, param)
    embed_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"layers?\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            layer_params.setdefault(layer_idx, []).append((name, p))
        elif "embed" in name.lower():
            embed_params.append((name, p))
        else:
            head_params.append((name, p))  # lm_head, final norm, etc.

    # Build groups sorted by depth (shallowest LR first)
    groups = []

    # Embedding: deepest, lowest LR
    if embed_params:
        embed_lr = lr * (decay ** (num_layers + 1))
        decay_p = [p for n, p in embed_params if p.ndim > 1]
        no_decay_p = [p for n, p in embed_params if p.ndim <= 1]
        if decay_p:
            groups.append({"params": decay_p, "lr": embed_lr, "weight_decay": weight_decay})
        if no_decay_p:
            groups.append({"params": no_decay_p, "lr": embed_lr, "weight_decay": 0.0})

    # Transformer layers: progressively higher LR
    for layer_idx in sorted(layer_params.keys()):
        layer_lr = lr * (decay ** (num_layers - layer_idx))
        params = layer_params[layer_idx]
        decay_p = [p for n, p in params if p.ndim > 1 and not any(k in n.lower() for k in ("bias", "norm"))]
        no_decay_p = [p for n, p in params if p.ndim <= 1 or any(k in n.lower() for k in ("bias", "norm"))]
        if decay_p:
            groups.append({"params": decay_p, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_p:
            groups.append({"params": no_decay_p, "lr": layer_lr, "weight_decay": 0.0})

    # lm_head + final norm: highest LR (full rate)
    if head_params:
        decay_p = [p for n, p in head_params if p.ndim > 1 and not any(k in n.lower() for k in ("bias", "norm"))]
        no_decay_p = [p for n, p in head_params if p.ndim <= 1 or any(k in n.lower() for k in ("bias", "norm"))]
        if decay_p:
            groups.append({"params": decay_p, "lr": lr, "weight_decay": weight_decay})
        if no_decay_p:
            groups.append({"params": no_decay_p, "lr": lr, "weight_decay": 0.0})

    return groups


# ==============================================================================
# CRITICAL BATCH SIZE CHECK
# Paper: "Revisiting Critical Batch Size" (AllenAI, 2025)
# The batch size beyond which you get diminishing returns.
# ==============================================================================


def check_critical_batch_size(
    model: torch.nn.Module,
    current_batch_tokens: int,
) -> dict:
    """Estimate whether current batch size exceeds the critical batch size.

    The critical batch size (CBS) is where increasing batch no longer helps.
    Above CBS: you're wasting compute (more tokens per step, same convergence).
    Below CBS: larger batch would help (more gradient signal per step).

    Approximation: CBS ~ trace(grad_covariance) / ||mean_grad||^2
    Since we can't compute full covariance, we use the simpler heuristic:
    CBS ~ (grad_norm_variance / mean_grad_norm^2) * current_batch

    Call this after a few training steps when gradient statistics are available.

    Args:
        model: Model with accumulated gradient statistics
        current_batch_tokens: Current effective tokens per optimizer step

    Returns:
        Dict with estimated CBS, ratio, and recommendation
    """
    # Collect per-param gradient norms
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.float().norm().item())

    if len(norms) < 10:
        return {"estimated_cbs": None, "ratio": None, "recommendation": "insufficient_data"}

    mean_norm = sum(norms) / len(norms)
    var_norm = sum((n - mean_norm) ** 2 for n in norms) / len(norms)

    if mean_norm < 1e-10:
        return {"estimated_cbs": None, "ratio": None, "recommendation": "zero_gradients"}

    # Noise-to-signal ratio: higher = noisier gradients = benefit more from larger batch
    noise_ratio = var_norm / (mean_norm**2)

    # Rough CBS estimate (tokens):
    # If noise_ratio is high, CBS is large (need big batch to average out noise)
    # If noise_ratio is low, CBS is small (already clean signal)
    estimated_cbs = int(current_batch_tokens * noise_ratio * 0.5)
    estimated_cbs = max(1024, min(estimated_cbs, 10_000_000))  # sanity clamp

    ratio = current_batch_tokens / max(estimated_cbs, 1)

    if ratio > 2.0:
        recommendation = "batch_too_large"
    elif ratio < 0.3:
        recommendation = "batch_too_small"
    else:
        recommendation = "batch_optimal"

    return {
        "estimated_cbs_tokens": estimated_cbs,
        "current_batch_tokens": current_batch_tokens,
        "ratio": round(ratio, 2),
        "noise_ratio": round(noise_ratio, 4),
        "recommendation": recommendation,
    }


# ==============================================================================
# LR SCHEDULER
# ==============================================================================


def build_scheduler(
    optimizer, scheduler_type: str, num_steps: int, warmup_ratio: float, min_lr_ratio: float
) -> LambdaLR:
    """LR scheduler with linear warmup + cosine/linear/power decay.

    With LLRD: the scheduler multiplier is applied to ALL groups uniformly,
    so the relative layer-wise ratios are preserved throughout training.

    Scheduler types:
      - "cosine": standard cosine annealing (γ=2, capacity saturates at β>3)
      - "linear": linear decay to min_lr
      - "constant": no decay after warmup
      - "power_decay": η(z) = η_peak · (1-z/N)^γ where γ=4 by default
        From arxiv:2602.06797: optimal for "easy tasks" (s ≥ 1-1/β).
        Power-decay with γ=2β-1 ≈ 4-5 avoids cosine's capacity saturation.
      - "wsd": warmup-stable-decay (optimal for "hard tasks", s < 1-1/β)
        Maintains peak LR for 80% of training, then decays with power profile.
    """
    warmup_steps = int(num_steps * warmup_ratio)

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if scheduler_type == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        progress = min(progress, 1.0)
        if scheduler_type == "cosine":
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if scheduler_type == "power_decay":
            # Power-decay: η(z) = η_peak · (1 - progress)^γ
            # γ = 4 is near-optimal for typical LLMs (β ≈ 2.5-3.0, so 2β-1 ≈ 4-5)
            # This avoids cosine's capacity saturation phenomenon.
            gamma = 4.0
            decay = (1.0 - progress) ** gamma
            return min_lr_ratio + (1 - min_lr_ratio) * decay
        if scheduler_type == "wsd":
            # Warmup-Stable-Decay: keep peak LR for 80% post-warmup, then power decay
            # From arxiv:2602.06797: optimal for hard tasks where signal is slow to learn
            stable_fraction = 0.8
            if progress < stable_fraction:
                return 1.0
            decay_progress = (progress - stable_fraction) / (1.0 - stable_fraction)
            gamma = 4.0
            decay = (1.0 - decay_progress) ** gamma
            return min_lr_ratio + (1 - min_lr_ratio) * decay
        # Default: linear
        return max(min_lr_ratio, 1 - progress * (1 - min_lr_ratio))

    return LambdaLR(optimizer, lr_fn)


# ==============================================================================
# HYPERBALL: NORM-CONSTRAINED OPTIMIZATION
# Paper: "Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball
# Optimization" (arXiv:2606.16899), Algorithm 1, over ANY base optimizer.
#
# For each constrained matrix (attention / MLP weights), with R = ||W_0||_F and
# u_t the base optimizer's update direction (Adam: m_hat/(sqrt(v_hat)+eps), Lion:
# sign(...), Muon: orthogonalised momentum):
#     W_{t+1} = R * Normalize(W_t - eta_t * R * Normalize(u_t))
# eta_t is an ANGULAR step (fraction of the norm moved per update); the norm never
# changes, which replaces weight decay. Embeddings, norms, biases and the head
# stay on the base optimizer, as in the paper.
# ==============================================================================


def is_hyperball_param(name: str, p: torch.Tensor) -> bool:
    """Attention / MLP weight matrices (the paper's constrained set)."""
    return p.ndim == 2 and not any(k in name.lower() for k in ("embed", "norm", "bias", "lm_head"))


def _math_dtype(p: torch.Tensor) -> torch.dtype:
    """At least float32 for the update math; never lower than the weight's own dtype."""
    return torch.promote_types(p.dtype, torch.float32)


class Hyperball:
    """Hyperball over an arbitrary base optimizer (call `.step()` instead of the
    base optimizer's).

    The base optimizer's direction for a matrix is recovered from its own step:
    with weight decay off, it moves W by -lr * u, so Normalize(u) = -dW / ||dW||
    exactly, whatever lr is. Constrained matrices are therefore stepped in buckets
    of at most `snapshot_bytes`: snapshot, let the base optimizer step only that
    bucket (parameters without .grad are skipped by every torch/bitsandbytes
    optimizer), then apply the Hyperball update from the snapshot. Memory: one
    bucket, not a copy of the model.

    angular_lr > 0: the paper's global angular step eta, scaled by the LR schedule
      (the ratio of each group's current lr to its initial lr).
    angular_lr = 0: per-matrix eta calibrated at the first step to the relative
      update the base optimizer made there (for Adam and Lion exactly
      lr / rms(W)), then kept, scaled by the schedule.
    R is each matrix's norm at the first step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, constrained_params: list[torch.Tensor],
                 angular_lr: float = 0.0, snapshot_bytes: int = 1 << 30):
        self.optimizer = optimizer
        self.angular_lr = angular_lr
        ids = {id(p) for p in constrained_params}
        self._constrained = [p for g in optimizer.param_groups for p in g["params"] if id(p) in ids]
        self._group_of = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        self._buckets, bucket, size = [], [], 0
        for p in self._constrained:
            nbytes = p.numel() * p.element_size()
            if bucket and size + nbytes > snapshot_bytes:
                self._buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(p)
            size += nbytes
        if bucket:
            self._buckets.append(bucket)
        self._radius: dict[int, torch.Tensor] = {}
        self._eta: dict[int, float] = {}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"base": self.optimizer.state_dict(),
                "radius": [self._radius.get(id(p)) for p in self._constrained],
                "eta": [self._eta.get(id(p)) for p in self._constrained]}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["base"])
        for p, r, e in zip(self._constrained, state.get("radius", []), state.get("eta", [])):
            if r is not None:
                self._radius[id(p)] = r.to(p.device) if torch.is_tensor(r) else r
            if e is not None:
                self._eta[id(p)] = e

    def _schedule(self, p) -> float:
        group = self._group_of[id(p)]
        return group["lr"] / group.get("initial_lr", group["lr"]) if group.get("initial_lr") else 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        grads = {id(p): p.grad for g in self.optimizer.param_groups for p in g["params"]}
        constrained = {id(p) for p in self._constrained}
        # 1) everything else: a plain base step
        for g in self.optimizer.param_groups:
            for p in g["params"]:
                if id(p) in constrained:
                    p.grad = None
        self.optimizer.step()
        # 2) constrained matrices, bucket by bucket
        decay = [g.get("weight_decay", 0.0) for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            if "weight_decay" in g:
                g["weight_decay"] = 0.0
        try:
            for bucket in self._buckets:
                live = [p for p in bucket if grads[id(p)] is not None]
                if not live:
                    continue
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        p.grad = None
                for p in live:
                    p.grad = grads[id(p)]
                before = [p.detach().to(_math_dtype(p)).clone() for p in live]
                self.optimizer.step()
                for p, w0 in zip(live, before):
                    self._project(p, w0)
        finally:
            for g, d in zip(self.optimizer.param_groups, decay):
                if "weight_decay" in g:
                    g["weight_decay"] = d
            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.grad = grads[id(p)]
        return loss

    def _project(self, p: torch.Tensor, before: torch.Tensor) -> None:
        key = id(p)
        if key not in self._radius:
            self._radius[key] = before.norm()
        radius = self._radius[key]
        delta = p.detach().to(before.dtype) - before  # = -lr * u
        step_norm = delta.norm()
        if key not in self._eta:
            if self.angular_lr:
                self._eta[key] = self.angular_lr
            elif self._schedule(p) <= 0 or float(step_norm) == 0.0:
                # Nothing to calibrate against yet: the first warmup step has lr 0
                # (the base optimizer did not move W). Calibrating here would fix
                # eta at 0 and freeze the matrix for the whole run.
                p.copy_(before)
                return
            else:
                self._eta[key] = float(step_norm / radius.clamp_min(1e-30)) / self._schedule(p)
        eta = self._eta[key] * self._schedule(p)
        trial = before.add_(delta, alpha=float(eta * radius / step_norm.clamp_min(1e-30)))
        p.copy_(trial.mul_(radius / trial.norm().clamp_min(1e-30)))


# ==============================================================================
# MONA: MUON + NESTEROV ACCELERATION
# Paper: "MONA: Muon Optimizer with Nesterov Acceleration" (Meituan, 2026)
# arxiv:2605.26842
#
# Key insight: Muon lacks curvature awareness — it can get trapped in sharp minima.
# MONA adds an acceleration term (EMA of gradient differences) BEFORE
# orthogonalization. The gradient difference g_k - g_{k-1} ≈ H·Δθ implicitly
# encodes curvature: sharp directions get large acceleration → escape.
#
# Result: Consistently beats Muon and AdamW at 1B, 6B, 68B MoE scales.
# MONA-Lite: bf16 buffers + streaming computation → 75% memory overhead reduction.
#
# Implementation: Applied as a pre-processing step on gradients before any
# optimizer (Muon, Adam, etc.) processes them.
# ==============================================================================


class MONAAcceleration:
    """MONA curvature-aware acceleration — augments gradients before optimizer.

    Before the optimizer processes gradients, MONA modifies them:
        D_k = G_k - G_{k-1}           (gradient difference)
        A_k = β_a · A_{k-1} + (1-β_a) · D_k  (EMA of differences)
        G̃_k = G_k + α · A_k          (accelerated gradient)

    The acceleration term points away from sharp minima, biasing optimization
    toward flatter solutions.

    Usage:
        mona = MONAAcceleration(model, beta_a=0.975, alpha=-10.0)
        for step in training:
            optimizer.zero_grad()
            loss.backward()
            mona.apply()       # Augment gradients in-place
            optimizer.step()   # Optimizer sees augmented gradients

    Args:
        model: The model whose gradients to augment.
        beta_a: EMA decay for acceleration buffer (0.975 for 68B, 0.99 for 1B).
        alpha: Acceleration coefficient. Rule: α = -1 / (2*(1-β_a)).
        lite: If True, store buffers in bf16 (MONA-Lite, 75% overhead reduction).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        beta_a: float = 0.975,
        alpha: float | None = None,
        lite: bool = True,
    ):
        self.model = model
        self.beta_a = beta_a
        # Default alpha from paper: α = -1 / (2*(1-β_a))
        self.alpha = alpha if alpha is not None else -1.0 / (2.0 * (1.0 - beta_a))
        self.lite = lite
        self._step_count = 0

        # Buffers: store previous gradient and acceleration EMA
        # Using streaming: compute diff in-place, overwrite prev_grad with current
        store_dtype = torch.bfloat16 if lite else torch.float32
        self._prev_gradients: dict[int, torch.Tensor] = {}
        self._acceleration_buffers: dict[int, torch.Tensor] = {}

        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                pid = id(p)
                self._prev_gradients[pid] = torch.zeros_like(p.data, dtype=store_dtype)
                self._acceleration_buffers[pid] = torch.zeros_like(p.data, dtype=store_dtype)

    def apply(self):
        """Augment model gradients in-place with MONA acceleration.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        self._step_count += 1

        for p in self.model.parameters():
            if p.grad is None or id(p) not in self._acceleration_buffers:
                continue

            pid = id(p)
            grad = p.grad.data
            prev_grad = self._prev_gradients[pid]
            accel_buf = self._acceleration_buffers[pid]

            if self._step_count > 1:
                # D_k = G_k - G_{k-1} (gradient difference)
                diff = grad.float() - prev_grad.float()

                # A_k = β_a · A_{k-1} + (1-β_a) · D_k
                accel_buf.mul_(self.beta_a).add_(diff.to(accel_buf.dtype), alpha=(1.0 - self.beta_a))

                # G̃_k = G_k + α · A_k (augment gradient in-place)
                grad.add_(accel_buf.to(grad.dtype), alpha=self.alpha)

            # Store current gradient for next step (streaming: overwrite prev)
            self._prev_gradients[pid].copy_(grad.to(self._prev_gradients[pid].dtype))


# ==============================================================================
# SAGE: SIGN ADAPTIVE GRADIENT FOR EMBEDDINGS
# Paper: "SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization"
# arxiv:2604.07663
#
# Key insight: Embedding layers have sparse, high-variance gradients due to
# Zipfian token frequency. Sign-based optimizers (Lion, Muon) fail here because
# they apply uniform magnitude updates. SAGE adds an O(d) adaptive damper:
# - Compute per-dimension mean absolute gradient (L1 norm across vocab)
# - Compare each dimension's "loudness" to the layer RMS
# - Damp high-variance dimensions (scale < 1.0), pass through quiet ones (scale = 1.0)
#
# Result: Outperforms AdamW at 1.3B (24.33 PPL vs 27.81) with 50% less memory.
# ==============================================================================


class SAGE(torch.optim.Optimizer):
    """SAGE: Sign Adaptive GradiEnt optimizer for embedding layers.

    Combines Lion-style sign updates with an O(d) adaptive damper that
    stabilizes high-variance embedding dimensions. The damper is provably
    bounded by 1.0, making SAGE a strictly safer generalization of Lion.

    Memory: O(V*d) for momentum + O(d) for adaptive state = same as Lion.
    (vs AdamW: O(V*d) momentum + O(V*d) second moment = 2× Lion)

    Args:
        params: Embedding parameters to optimize.
        lr: Learning rate (can be more aggressive than Lion due to damping).
        beta1: Momentum decay (0.9 default, same as Lion).
        beta2: Adaptive state EMA decay (0.99 default).
        weight_decay: Decoupled weight decay coefficient.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)
                    # Adaptive state: O(d) — per-dimension mean absolute gradient
                    if p.ndim == 2:
                        # Embedding: track per-dimension (column) statistics
                        state["s_ema"] = torch.zeros(p.shape[1], device=p.device, dtype=p.dtype)
                    else:
                        # 1D params: per-element
                        state["s_ema"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]
                momentum = state["momentum"]
                s_ema = state["s_ema"]

                # Decoupled weight decay
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Compute per-dimension scale (s_t)
                if p.ndim == 2:
                    # Embedding: s_t = mean(|grad|) along vocab dimension
                    s_t = grad.abs().mean(dim=0)  # shape: (d,)
                else:
                    s_t = grad.abs()

                # Update EMA of absolute gradient scale
                s_ema.mul_(beta2).add_(s_t, alpha=(1.0 - beta2))

                # Bias correction
                s_hat = s_ema / (1.0 - beta2**t)

                # Compute adaptive damper H_t (bounded by 1.0)
                # σ_rms = RMS(ŝ) — layer-wide reference "loudness"
                sigma_rms = (s_hat.pow(2).mean()).sqrt()
                # γ_rms = RMS(s_t) — instantaneous reference
                gamma_rms = (s_t.pow(2).mean()).sqrt()

                # D_ema = σ_rms / (ŝ_j + ε) — EMA-based damper
                d_ema = sigma_rms / (s_hat + eps)
                # D_inst = γ_rms / (s_t_j + ε) — instantaneous damper
                d_inst = gamma_rms / (s_t + eps)

                # H_t = min(D_ema, D_inst, 1.0) — bounded adaptive scale
                h_t = torch.minimum(d_ema, d_inst)
                h_t = torch.clamp(h_t, max=1.0)

                # Lion-style update direction: sign(β1 * momentum + (1-β1) * grad)
                update_direction = torch.sign(beta1 * momentum + (1.0 - beta1) * grad)

                # Apply adaptive scale
                if p.ndim == 2:
                    # Broadcast h_t (d,) across vocab dimension
                    update = update_direction * h_t.unsqueeze(0)
                else:
                    update = update_direction * h_t

                # Apply update
                p.data.add_(update, alpha=-lr)

                # Update momentum (after computing update direction, like Lion)
                momentum.mul_(beta2).add_(grad, alpha=(1.0 - beta2))

        return loss

    @torch.no_grad()
    def get_adaptive_scales(self) -> dict[str, torch.Tensor]:
        """Return current adaptive scales for testing/debugging."""
        scales = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                state = self.state.get(p, {})
                if "s_ema" not in state or state.get("step", 0) == 0:
                    continue
                s_ema = state["s_ema"]
                t = state["step"]
                beta2 = group["beta2"]
                eps = group["eps"]

                s_hat = s_ema / (1.0 - beta2**t)
                sigma_rms = (s_hat.pow(2).mean()).sqrt()
                d_ema = sigma_rms / (s_hat + eps)
                h_t = torch.clamp(d_ema, max=1.0)
                scales[f"group{i}_param{j}"] = h_t
        return scales


# ==============================================================================
# ADAMC: CORRECTED WEIGHT DECAY FOR NORMALIZED LAYERS
# Paper: "Why Gradients Rapidly Increase Near the End of Training" (Defazio, 2025)
# arxiv:2506.02285
#
# Problem: With cosine LR decay, weight decay forces gradient norms to follow
#          ||g||/||w|| = sqrt(2λ/γ_t). As γ_t → 0, this → ∞. Gradient explosion.
#
# Fix: For layers followed by normalization (which have the property g⊥w),
#      scale weight decay by (γ_t / γ_max), making the steady-state constant.
#
# Implementation: After each scheduler step, adjust weight_decay for param
#      groups tagged as 'normalized'. This is a zero-cost correction.
# ==============================================================================


class AdamCCorrection:
    """Corrects weight decay for normalized layers to prevent end-of-training gradient explosion.

    Standard AdamW with cosine LR decay causes gradient norms to explode near the
    end of training for layers followed by normalization (LayerNorm/RMSNorm).

    The fix: multiply weight_decay by (current_lr / peak_lr) for normalized layers.
    This keeps the gradient-to-weight ratio constant throughout training.

    With LLRD (layer-wise LR decay): each param group has a different peak LR.
    The correction uses each group's OWN initial LR as its peak, not the global config value.

    Usage:
        correction = AdamCCorrection(optimizer, peak_lr)
        for step in training:
            ...
            scheduler.step()
            correction.step()  # adjust WD after LR changes
    """

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float):
        self.optimizer = optimizer
        self.peak_lr = peak_lr  # global peak (fallback)
        # Store per-group peak LR (the initial LR at construction time).
        # With LLRD, each group's initial LR IS its peak.
        self._group_peak_lrs: list[float] = []
        self._base_weight_decays: list[float] = []
        for group in optimizer.param_groups:
            self._group_peak_lrs.append(group.get("lr", peak_lr))
            self._base_weight_decays.append(group.get("weight_decay", 0.0))

    def step(self):
        """Adjust weight decay based on current LR ratio. Call after scheduler.step()."""
        for i, group in enumerate(self.optimizer.param_groups):
            base_wd = self._base_weight_decays[i]
            if base_wd <= 0:
                continue  # no WD for this group, nothing to correct
            # Use THIS GROUP's peak LR (not the global one)
            group_peak = self._group_peak_lrs[i]
            current_lr = group["lr"]
            # Correction: wd_effective = wd_base * (current_lr / group_peak_lr)
            ratio = current_lr / group_peak if group_peak > 0 else 1.0
            group["weight_decay"] = base_wd * ratio
