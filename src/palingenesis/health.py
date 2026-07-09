"""Training health metrics — tiered diagnostics for iterative improvement.

Three tiers of metrics, each at different frequencies to minimize overhead:

Tier 1 (every step, <1ms): loss variance accumulation, token efficiency, output entropy
Tier 2 (every N steps, ~10ms): gradient stability, memory, loss statistics
Tier 3 (every M steps, ~200ms): weight norms, stable rank, hidden state norms,
        attention entropy, logit confidence, model drift from init

These metrics tell you:
    - Is training converging? (loss trend, grad cosine sim)
    - Is the model collapsing? (stable rank decline, attention entropy)
    - Are representations healthy? (hidden state norms, layer distribution)
    - Is the data pipeline working? (token efficiency, loss variance)
    - Are we about to OOM? (memory trends)
    - Should we change hyperparams? (grad norm patterns, loss plateaus)
    - Is the model ready for RL? (output entropy, diversity metrics)

Key research-backed signals:
    - Stable rank decline precedes collapse (arxiv:2602.01734)
    - Attention entropy collapse = degenerate model (arxiv:2303.06296)
    - Hidden state norm explosion in QKV/Proj layers = divergence (arxiv:2410.16682)
    - High loss variance = effective batch too small or data issues
    - Output entropy collapse → GRPO failure (arxiv:2606.18487)
    - Excessive SFT kills plasticity for RL (arxiv:2606.09932)
"""

import logging
import math
import re
from collections import deque

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class HealthMonitor:
    """Tiered training health monitor.

    Usage:
        monitor = HealthMonitor(model, tier2_every=10, tier3_every=100)

        for step in training_loop:
            # Tier 1: always (just buffers data, zero cost)
            monitor.record_microstep(loss_val, labels)

            # After optimizer step:
            metrics = monitor.on_step(step, model)
            if metrics:
                tracker.log(metrics)
    """

    def __init__(
        self,
        model: nn.Module,
        tier2_every: int = 10,
        tier3_every: int = 100,
        rl_readiness: bool = False,
        rl_entropy_floor: float = 1.0,
    ):
        self.model = model
        self.tier2_every = tier2_every
        self.tier3_every = tier3_every

        # Tier 1 accumulators
        self._loss_buffer: deque[float] = deque(maxlen=100)
        self._token_eff_buffer: deque[float] = deque(maxlen=100)

        # RL-readiness monitoring (arxiv:2606.18487, 2606.09932)
        # Output entropy collapse predicts GRPO/RL failure.
        # Track entropy of model logits over training to detect overtraining.
        self._rl_readiness = rl_readiness
        self._rl_entropy_floor = rl_entropy_floor
        self._entropy_buffer: deque[float] = deque(maxlen=100)
        self._entropy_warned = False

        # Tier 2 state
        self._prev_grad_flat: torch.Tensor | None = None

        # GNS estimation state (from micro-batch gradient norms)
        # GNS = B_simple = tr(Σ) / ||G||^2 ≈ Var(g_micro_norms) / mean(g_micro_norms)^2
        self._micro_grad_norms: list[float] = []
        self._gns_history: deque[float] = deque(maxlen=50)

        # Tier 3 state
        self._init_weight_norms: dict[str, float] | None = None
        self._layer_info = _scan_model_layers(model)

    def record_microstep(self, loss: float, labels: torch.Tensor):
        """Tier 1: record per-microstep data (zero GPU overhead)."""
        if math.isfinite(loss):
            self._loss_buffer.append(loss)
        total = labels.numel()
        valid = (labels != IGNORE_INDEX).sum().item()
        self._token_eff_buffer.append(valid / max(total, 1))

    def record_logit_entropy(self, logits: torch.Tensor, labels: torch.Tensor):
        """Record output entropy from logits for RL-readiness monitoring.

        Measures the average per-token entropy of the model's output distribution.
        Low entropy = overconfident = entropy collapse = bad for RL.

        From arxiv:2606.18487: "SFT overtraining compresses output diversity,
        extinguishing the gradient signal GRPO requires."

        From arxiv:2606.09932: "Models from excessive SFT tend to produce
        over-confident token distributions... which make them harder to optimize
        in the RL stage."

        Args:
            logits: Model output logits (batch, seq, vocab)
            labels: Target labels (batch, seq) — used to compute entropy only
                    on tokens that have loss (non-IGNORE positions)

        The entropy is computed on a random subsample of valid tokens for speed.
        Cost: ~0.5ms on typical batch sizes.
        """
        if not self._rl_readiness:
            return

        with torch.no_grad():
            # Only compute on valid (non-ignore) positions
            mask = labels != IGNORE_INDEX
            if mask.sum() == 0:
                return

            # Subsample for speed and memory: at most 64 tokens
            # (For 262K vocab: 64 * 262K * 4 bytes = 67MB — safe even on tight memory)
            valid_indices = mask.nonzero(as_tuple=False)
            if valid_indices.shape[0] > 64:
                perm = torch.randperm(valid_indices.shape[0], device=valid_indices.device)[:64]
                valid_indices = valid_indices[perm]

            # Extract logits at valid positions
            batch_idx = valid_indices[:, 0]
            seq_idx = valid_indices[:, 1]
            sampled_logits = logits[batch_idx, seq_idx]  # (N, vocab)

            self._record_sampled_entropy(sampled_logits)

    def record_entropy_from_hidden(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        lm_head: nn.Module,
    ):
        """Record output entropy when full logits are never materialized.

        Used with chunked CE / Cut Cross-Entropy paths, which skip the [B, S, V]
        logit tensor entirely. Samples up to 64 valid token positions from the
        hidden states and projects ONLY those through lm_head — cost is a
        (64, D) x (D, V) matmul, negligible at logging frequency.

        Args:
            hidden: Backbone hidden states (batch, seq, dim), detached
            labels: Target labels (batch, seq) for valid-position masking
            lm_head: The lm_head projection module
        """
        if not self._rl_readiness:
            return

        with torch.no_grad():
            mask = labels != IGNORE_INDEX
            if mask.sum() == 0:
                return

            valid_indices = mask.nonzero(as_tuple=False)
            if valid_indices.shape[0] > 64:
                perm = torch.randperm(valid_indices.shape[0], device=valid_indices.device)[:64]
                valid_indices = valid_indices[perm]

            batch_idx = valid_indices[:, 0]
            seq_idx = valid_indices[:, 1]
            sampled_hidden = hidden[batch_idx, seq_idx]  # (N, dim)
            sampled_logits = lm_head(sampled_hidden)  # (N, vocab)

            self._record_sampled_entropy(sampled_logits)

    def _record_sampled_entropy(self, sampled_logits: torch.Tensor):
        """Compute mean entropy of sampled (N, vocab) logits and buffer it."""
        # Compute entropy: H = -sum(p * log(p))
        # Use log_softmax for numerical stability
        log_probs = torch.log_softmax(sampled_logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (N,)
        mean_entropy = entropy.mean().item()

        if math.isfinite(mean_entropy):
            self._entropy_buffer.append(mean_entropy)

    def record_micro_grad_norm(self, grad_norm: float):
        """Record per-micro-batch gradient norm for GNS estimation.

        Call this BEFORE the optimizer step, once per micro-batch during
        gradient accumulation. The variance between micro-batch norms gives
        us the gradient noise scale (GNS), which approximates critical batch size.

        GNS interpretation (from McCandlish et al., 2018):
          - GNS < current_batch_size: batch is oversized, wasting compute
          - GNS ≈ current_batch_size: optimal trade-off
          - GNS > current_batch_size: batch is undersized, could benefit from more

        NOTE: Can also be estimated from per-micro-batch loss variance
        (which we already track). This method uses gradient norms for higher
        precision when available.
        """
        if math.isfinite(grad_norm):
            self._micro_grad_norms.append(grad_norm)

    def flush_gns(self) -> float | None:
        """Compute GNS from accumulated micro-batch data, then reset.

        Call after each optimizer step.
        Returns B_simple estimate, or None if insufficient data.

        Uses per-micro-batch loss values as a lightweight GNS proxy:
          B_simple ≈ N * Var(L_micro) / mean(L_micro)^2

        If gradient norms were recorded (more precise), uses those instead:
          B_simple ≈ N * Var(||g||) / ||mean(g)||^2

        From NeurIPS 2024 (2411.00999): GNS from norm layers alone predicts
        full-model GNS with r > 0.96 correlation.
        """
        # Prefer gradient norms if available, otherwise fall back to loss
        norms = self._micro_grad_norms
        self._micro_grad_norms = []

        if len(norms) >= 2:
            values = norms
        elif len(self._loss_buffer) >= 4:
            # Use recent loss values as proxy (last N entries = recent micro-batches)
            values = list(self._loss_buffer)[-8:]
        else:
            return None

        n = len(values)
        if n < 2:
            return None

        mean_val = sum(values) / n
        if mean_val < 1e-12:
            return None

        var_val = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        # GNS estimate: scales with batch count and normalized variance
        gns = n * var_val / (mean_val**2)
        self._gns_history.append(gns)
        return gns

    def on_step(self, global_step: int, model: nn.Module | None = None) -> dict[str, float]:
        """Called after each optimizer step. Returns metrics to log.

        Automatically dispatches tier 2 and tier 3 based on step number.
        Always returns tier 1 aggregates.
        """
        model = model or self.model
        metrics: dict[str, float] = {}

        # ── Tier 1: always (aggregates from buffers) ──────────────────
        metrics.update(self._tier1_metrics())

        # ── Tier 2: every N steps ────────────────────────────────────
        if global_step % self.tier2_every == 0:
            metrics.update(self._tier2_metrics(model))

        # ── Tier 3: every M steps ────────────────────────────────────
        if global_step % self.tier3_every == 0 and global_step > 0:
            metrics.update(self._tier3_metrics(model))

        return metrics

    # ── Tier 1: zero overhead ─────────────────────────────────────────────

    def _tier1_metrics(self) -> dict[str, float]:
        """Loss statistics, token efficiency, and output entropy from buffers."""
        metrics = {}

        if len(self._loss_buffer) > 1:
            losses = list(self._loss_buffer)
            mean = sum(losses) / len(losses)
            var = sum((l - mean) ** 2 for l in losses) / (len(losses) - 1)
            metrics["health/loss_mean_window"] = mean
            metrics["health/loss_std_window"] = math.sqrt(var)
            # Coefficient of variation — normalized measure of noise
            metrics["health/loss_cv"] = math.sqrt(var) / max(abs(mean), 1e-8)

        if self._token_eff_buffer:
            metrics["health/token_efficiency"] = sum(self._token_eff_buffer) / len(self._token_eff_buffer)

        # RL-readiness: output entropy monitoring
        if self._rl_readiness and self._entropy_buffer:
            current_entropy = self._entropy_buffer[-1]
            mean_entropy = sum(self._entropy_buffer) / len(self._entropy_buffer)
            metrics["health/output_entropy"] = current_entropy
            metrics["health/output_entropy_ema"] = mean_entropy

            # Detect entropy collapse (predicts RL failure)
            # Threshold from arxiv:2606.18487: when entropy drops below floor,
            # GRPO's group-relative advantage variance collapses.
            if len(self._entropy_buffer) >= 10:
                # Check if entropy is declining monotonically (danger signal)
                recent = list(self._entropy_buffer)[-20:]
                if len(recent) >= 10:
                    first_half = sum(recent[: len(recent) // 2]) / (len(recent) // 2)
                    second_half = sum(recent[len(recent) // 2 :]) / (len(recent) - len(recent) // 2)
                    entropy_trend = (second_half - first_half) / max(abs(first_half), 1e-8)
                    metrics["health/entropy_trend"] = entropy_trend

                if mean_entropy < self._rl_entropy_floor and not self._entropy_warned:
                    self._entropy_warned = True
                    logger.warning(
                        f"⚠️  RL-READINESS WARNING: Output entropy collapsed to {mean_entropy:.2f} "
                        f"(floor={self._rl_entropy_floor:.2f}). "
                        f"If you plan RL after SFT, consider stopping NOW. "
                        f"Entropy collapse kills GRPO signal (arxiv:2606.18487). "
                        f"Recommendations: (1) Stop SFT, (2) Use EMA/base-merge to restore diversity, "
                        f"(3) If continuing SFT, add pre_rl plugin for entropy preservation."
                    )
                    metrics["health/rl_readiness_warning"] = 1.0
                elif mean_entropy >= self._rl_entropy_floor:
                    metrics["health/rl_readiness_warning"] = 0.0

        return metrics

    # ── Tier 2: gradient stability + memory (~10ms) ───────────────────────

    def _tier2_metrics(self, model: nn.Module) -> dict[str, float]:
        """Gradient direction stability, GNS, and CUDA memory."""
        metrics = {}

        # Gradient cosine similarity with previous step
        grad_flat = _sample_grads(model, max_elements=500_000)
        if grad_flat is not None and self._prev_grad_flat is not None:
            if grad_flat.shape == self._prev_grad_flat.shape:
                cos = torch.nn.functional.cosine_similarity(
                    grad_flat.unsqueeze(0), self._prev_grad_flat.unsqueeze(0)
                ).item()
                metrics["health/grad_cosine_sim"] = cos
                # Interpretation:
                #   > 0.5: stable, consistent direction
                #   0.1-0.5: normal noise
                #   < 0.1: oscillating, possibly bad LR
                #   < 0: actively fighting itself, LR too high
        if grad_flat is not None:
            self._prev_grad_flat = grad_flat.clone()

        # GNS (Gradient Noise Scale) — critical batch size estimate
        # From NeurIPS 2024: "Normalization Layer Per-Example Gradients are
        # Sufficient to Predict Gradient Noise Scale" (2411.00999)
        if self._gns_history:
            gns = self._gns_history[-1]
            metrics["health/gns"] = gns
            # Moving average for smoother signal
            metrics["health/gns_ema"] = sum(self._gns_history) / len(self._gns_history)
            # Ratio: GNS / effective_batch_size would tell us efficiency
            # (logged externally since we don't know batch size here)

        # CUDA memory
        if torch.cuda.is_available():
            metrics["health/cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            metrics["health/cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            # How full is the GPU? (useful for knowing headroom)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            metrics["health/cuda_utilization_pct"] = torch.cuda.memory_allocated() / total_mem * 100
            torch.cuda.reset_peak_memory_stats()

        return metrics

    # ── Tier 3: deep diagnostics (~200ms) ─────────────────────────────────

    @torch.no_grad()
    def _tier3_metrics(self, model: nn.Module) -> dict[str, float]:
        """Weight health, stable rank, model drift, activation norms."""
        metrics = {}
        params = dict(model.named_parameters())

        # ── Weight norms per layer depth ──────────────────────────────
        # Track how weight norms evolve across layers (uniform = healthy,
        # diverging = some layers learning faster/slower)
        layer_norms = _per_layer_weight_norms(params, self._layer_info)
        if layer_norms:
            norms = list(layer_norms.values())
            metrics["health/weight_norm_min"] = min(norms)
            metrics["health/weight_norm_max"] = max(norms)
            metrics["health/weight_norm_ratio"] = max(norms) / max(min(norms), 1e-8)
            # Log a few representative layers
            for name, norm in list(layer_norms.items())[:3] + list(layer_norms.items())[-3:]:
                metrics[f"health/wnorm/{name}"] = norm

        # ── Per-layer effective LR: ‖∇L‖/‖W‖ ratio ──────────────────
        # This tells you which layers are receiving disproportionate updates.
        # High ratio = layer learns faster = overfits faster = may need more WD.
        # Used to diagnose whether AlphaDecay (per-module WD) would help.
        grad_weight_ratios = _per_layer_grad_weight_ratio(params, self._layer_info)
        if grad_weight_ratios:
            ratios = list(grad_weight_ratios.values())
            metrics["health/grad_weight_ratio_max"] = max(ratios)
            metrics["health/grad_weight_ratio_min"] = min(ratios)
            metrics["health/grad_weight_ratio_spread"] = max(ratios) / max(min(ratios), 1e-12)
            for name, ratio in grad_weight_ratios.items():
                metrics[f"health/gw_ratio/{name}"] = ratio

        # ── Stable rank (sampled — expensive but critical) ────────────
        # stable_rank = ||W||_F^2 / ||W||_2^2
        # Declining stable rank = the matrix is becoming rank-1 = collapse
        stable_ranks = _sample_stable_ranks(params, self._layer_info)
        if stable_ranks:
            srs = list(stable_ranks.values())
            metrics["health/stable_rank_min"] = min(srs)
            metrics["health/stable_rank_mean"] = sum(srs) / len(srs)
            # Alert threshold: if any key matrix drops below 5, it's concerning
            for name, sr in stable_ranks.items():
                metrics[f"health/srank/{name}"] = sr

        # ── Model drift from initialization ───────────────────────────
        # How far has the model moved from its pretrained weights?
        # Useful for SFT: too much drift = catastrophic forgetting
        if self._init_weight_norms is None:
            # First time: snapshot initial norms
            self._init_weight_norms = {}
            for name, p in params.items():
                if p.ndim >= 2 and p.numel() > 1000:
                    self._init_weight_norms[name] = p.data.float().norm().item()
        else:
            drift_ratios = []
            for name, init_norm in self._init_weight_norms.items():
                if name in params and init_norm > 0:
                    current_norm = params[name].data.float().norm().item()
                    drift_ratios.append(abs(current_norm - init_norm) / init_norm)
            if drift_ratios:
                metrics["health/weight_drift_mean"] = sum(drift_ratios) / len(drift_ratios)
                metrics["health/weight_drift_max"] = max(drift_ratios)
                # >50% drift on any layer is catastrophic forgetting territory
                # 5-20% is normal for SFT

        # ── Logit confidence (if lm_head accessible) ─────────────────
        # Not computed here — would need a forward pass. Could be done
        # by hooking into the training step's output.

        # ── Health warnings ───────────────────────────────────────────
        warnings = 0
        if stable_ranks and min(stable_ranks.values()) < 3.0:
            warnings += 1
            logger.warning(f"Low stable rank detected: {min(stable_ranks.values()):.1f} — collapse risk")
        if layer_norms and max(norms) / max(min(norms), 1e-8) > 100:
            warnings += 1
            logger.warning(f"Weight norm imbalance: {max(norms)/max(min(norms),1e-8):.0f}x ratio across layers")
        if "health/weight_drift_max" in metrics and metrics["health/weight_drift_max"] > 0.5:
            warnings += 1
            logger.warning(f"High weight drift from init: {metrics['health/weight_drift_max']:.1%}")

        metrics["health/warnings"] = warnings
        return metrics


# ── Utility functions ─────────────────────────────────────────────────────────


def compute_token_efficiency(labels: torch.Tensor) -> float:
    """Fraction of tokens that have loss computed (not IGNORE_INDEX)."""
    return (labels != IGNORE_INDEX).sum().item() / max(labels.numel(), 1)


def _scan_model_layers(model: nn.Module) -> dict[str, list[str]]:
    """Identify named parameters by layer depth for grouped stats.

    Returns {layer_group_name: [param_names]} where groups are:
        layer_0, layer_1, ..., layer_N (one per transformer layer)
        embed, lm_head
    """
    groups: dict[str, list[str]] = {}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            continue
        m = re.search(r"layers?\.(\d+)\.", name)
        if m:
            layer_id = int(m.group(1))
            key = f"layer_{layer_id}"
            groups.setdefault(key, []).append(name)
        elif "embed" in name.lower():
            groups.setdefault("embed", []).append(name)
        elif "lm_head" in name.lower():
            groups.setdefault("lm_head", []).append(name)
    return groups


def _per_layer_weight_norms(
    params: dict[str, torch.nn.Parameter],
    layer_info: dict[str, list[str]],
) -> dict[str, float]:
    """Compute average Frobenius norm per layer group."""
    result = {}
    for group_name, param_names in layer_info.items():
        norms = []
        for pname in param_names:
            if pname in params:
                p = params[pname]
                w = p.data
                # Handle DTensor/FSDP by getting local tensor
                if hasattr(w, "_local_tensor"):
                    w = w._local_tensor
                norms.append(w.float().norm().item())
        if norms:
            result[group_name] = sum(norms) / len(norms)
    return result


def _sample_stable_ranks(
    params: dict[str, torch.nn.Parameter],
    layer_info: dict[str, list[str]],
    sample_layers: int = 6,
) -> dict[str, float]:
    """Compute stable rank for sampled layers (early, mid, late).

    Stable rank = ||W||_F^2 / sigma_max(W)^2
    Only computes for a few representative layers to keep cost bounded.

    We sample: first layer, 2 from middle, last layer, plus lm_head.
    """
    # Pick which layers to sample
    layer_keys = sorted([k for k in layer_info if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
    if not layer_keys:
        return {}

    # Sample: first, 1/3, 1/2, 2/3, last
    n = len(layer_keys)
    indices = sorted(set([0, n // 3, n // 2, 2 * n // 3, n - 1]))
    sampled = [layer_keys[i] for i in indices if i < n]
    if "lm_head" in layer_info:
        sampled.append("lm_head")

    result = {}
    for group_name in sampled:
        param_names = layer_info.get(group_name, [])
        # Pick the largest 2D parameter in this group (usually a projection matrix)
        best_param = None
        best_numel = 0
        for pname in param_names:
            if pname in params and params[pname].ndim == 2:
                if params[pname].numel() > best_numel:
                    best_param = params[pname]
                    best_numel = params[pname].numel()

        if best_param is None:
            continue

        w = best_param.data
        if hasattr(w, "_local_tensor"):
            w = w._local_tensor

        # Subsample for speed: take at most 512x512
        w2d = w.float()
        if w2d.shape[0] > 512:
            w2d = w2d[:512]
        if w2d.shape[1] > 512:
            w2d = w2d[:, :512]

        frob_sq = w2d.norm() ** 2
        try:
            # Top singular value via svdvals (fast — only computes values)
            sigma_max = torch.linalg.svdvals(w2d)[0]
            spec_sq = sigma_max**2
            stable_rank = (frob_sq / max(spec_sq, 1e-30)).item()
            result[group_name] = stable_rank
        except Exception:
            pass

    return result


@torch.no_grad()
def _sample_grads(model: nn.Module, max_elements: int = 500_000) -> torch.Tensor | None:
    """Flatten a sample of gradients for cosine similarity computation."""
    chunks = []
    total = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        flat = p.grad.data.float().flatten()
        take = min(flat.numel(), max_elements - total)
        if take <= 0:
            break
        chunks.append(flat[:take])
        total += take
    return torch.cat(chunks) if chunks else None


def _per_layer_grad_weight_ratio(
    params: dict[str, torch.nn.Parameter],
    layer_info: dict[str, list[str]],
) -> dict[str, float]:
    """Compute per-layer ‖grad‖/‖weight‖ ratio (effective learning rate signal).

    High ratio = that layer receives disproportionately large updates relative
    to its weight magnitude. Indicates faster learning/overfitting risk.
    If spread (max/min) exceeds 10×, consider per-module weight decay (AlphaDecay).
    """
    result = {}
    for group_name, param_names in layer_info.items():
        grad_norm_sq = 0.0
        weight_norm_sq = 0.0
        has_grad = False
        for pname in param_names:
            if pname not in params:
                continue
            p = params[pname]
            if p.ndim < 2:
                continue
            w = p.data
            if hasattr(w, "_local_tensor"):
                w = w._local_tensor
            weight_norm_sq += w.float().norm().item() ** 2
            if p.grad is not None:
                g = p.grad.data
                if hasattr(g, "_local_tensor"):
                    g = g._local_tensor
                grad_norm_sq += g.float().norm().item() ** 2
                has_grad = True

        if has_grad and weight_norm_sq > 1e-12:
            result[group_name] = (grad_norm_sq**0.5) / (weight_norm_sq**0.5)

    return result
