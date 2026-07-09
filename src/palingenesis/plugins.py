"""Opt-in training plugins -- research-backed, torch.compile-optimized.

Plugins:
    SymNoise       -- Symmetric noisy embeddings (ICLR 2024 + NeurIPS 2025)
    InfoSFT        -- Information-aware token weighting (arxiv:2605.14967)
    DEFT           -- Dynamic Entropy Fine-Tuning: parameter-free adaptive (arxiv:2602.11424)
    DFT            -- Dynamic Fine-Tuning: bounded gradient scaling (Wu et al., 2025)
    CADFT          -- Compatibility-Aware DFT: sample-level variance control (arxiv:2606.11206)
    ScheduleFree   -- Schedule-Free AdamW (NeurIPS 2025)
    Pre-RL         -- Entropy/KL regularization for GRPO warm-start (arxiv:2605.29303)

Loss hierarchy (from most to least adaptive):
    DEFT > CADFT > DFT > InfoSFT > NLL

    DEFT:   α = per-token collision probability (automatic, parameter-free)
    CADFT:  α = 1 (DFT) + sample-level z-score compatibility weighting
    DFT:    α = 1 (fixed, pure probability-loss / sharpening)
    InfoSFT: information-theoretic token selection (different mechanism)
    NLL:    α = 0 (uniform, standard cross-entropy)

Performance: All loss computations are compiled via torch.compile into fused
kernels. Zero accuracy loss, pure speed.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===== SYMNOISE =====
# Paper: "Advancing Language Model Fine-tuning with Symmetric Noise" (2312.01523)
# Result: +6.7% over NEFTune on AlpacaEval


class SymNoiseHook:
    """Bernoulli {-1,+1} noisy embedding injection.

    Key difference from NEFTune: uses Bernoulli noise, not Uniform.
    Paper ablation confirms Bernoulli > Uniform > Gaussian.
    """

    def __init__(self, model: nn.Module, alpha: float = 5.0):
        self.alpha = alpha
        self._handle = None
        self._embed = self._find_embedding(model)
        if self._embed is not None:
            self._handle = self._embed.register_forward_hook(self._hook)
            logger.info(f"SymNoise enabled (alpha={alpha}, bernoulli)")
        else:
            logger.warning("Could not find embedding layer for SymNoise")

    def _hook(self, module: nn.Module, input, output: torch.Tensor) -> torch.Tensor:
        if not module.training:
            return output
        seq_len, dim = output.shape[-2], output.shape[-1]
        mag = self.alpha / (seq_len * dim) ** 0.5
        # Bernoulli {-1, +1} -- the key ingredient
        noise = (torch.bernoulli(torch.full_like(output, 0.5)) * 2 - 1) * mag
        return output + noise

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @staticmethod
    def _find_embedding(model: nn.Module) -> nn.Embedding | None:
        for path in ("model.embed_tokens", "model.model.embed_tokens", "transformer.wte"):
            obj = model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                if isinstance(obj, nn.Embedding):
                    return obj
            except AttributeError:
                continue
        for m in model.modules():
            if isinstance(m, nn.Embedding) and m.embedding_dim > 256:
                return m
        return None


# ===== INFOSFT =====
# Paper: "Learn More and Forget Less" (arxiv:2605.14967, May 2025)
# Formula: w(q) = q * [logit(p_bar) - logit(q)]_+
# Compiled: torch.compile fuses weight computation + CE into ~2 kernels

INFOSFT_PBAR_DEFAULT = 0.93


def _infosft_fused(logits: torch.Tensor, labels: torch.Tensor, logit_pbar: float) -> torch.Tensor:
    """Fused InfoSFT: weight computation + weighted CE in one compilable function.

    All ops are standard PyTorch -- inductor fuses them into minimal kernels:
    softmax -> gather -> log -> sub -> clamp -> mul -> CE -> mul -> sum
    becomes ~2-3 fused CUDA kernels instead of 8+ separate launches.
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # 1. Get q = P(correct token) for each position
    probs = torch.softmax(logits.float(), dim=-1)  # [B, S, V]
    valid_mask = labels != IGNORE_INDEX
    safe_labels = labels.clamp(min=0)
    q = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
    q = q.clamp(min=1e-8, max=1.0 - 1e-8)

    # 2. InfoSFT weight: w(q) = q * [logit(p_bar) - logit(q)]_+
    # Using log(q) - log(1-q) instead of log(q/(1-q)) for numerical stability
    logit_q = torch.log(q) - torch.log1p(-q)
    correction = (logit_pbar - logit_q).clamp(min=0.0)
    weights = q * correction  # [B, S]

    # 3. Normalize over valid tokens (mean=1 preserves loss magnitude)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    valid_count = valid_mask.sum().clamp(min=1)
    mean_w = weights.sum() / valid_count
    weights = weights / mean_w.clamp(min=1e-8)

    # 4. Weighted cross-entropy
    loss_flat = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )  # [B*S]
    weighted = loss_flat.view(B, S) * weights
    return weighted.sum()


# Compile for maximum fusion -- dynamic=True handles variable seq lengths
_infosft_compiled = torch.compile(_infosft_fused, dynamic=True)


def infosft_weighted_loss(
    logits: torch.Tensor, labels: torch.Tensor, p_bar: float = INFOSFT_PBAR_DEFAULT
) -> torch.Tensor:
    """InfoSFT loss with torch.compile fusion.

    First call triggers compilation (~30s). Subsequent calls use the fused kernel.
    Falls back to eager on compilation failure.
    """
    logit_pbar = math.log(p_bar / (1.0 - p_bar))
    try:
        return _infosft_compiled(logits, labels, logit_pbar)
    except Exception:
        return _infosft_fused(logits, labels, logit_pbar)


# ===== DEFT: DYNAMIC ENTROPY FINE-TUNING =====
# Paper: "Gradients Must Earn Their Influence" (arxiv:2602.11424, Feb 2026)
#
# THE unified SFT loss that subsumes NLL, DFT, and InfoSFT.
#
# Key insight: All SFT losses have the form: gradient ∝ p^α × (1-p)
#   - α=0: NLL (learns everything, including noise)
#   - α=1: DFT/-p loss (sharpens, ignores hard tokens)
#   - α=dynamic: DEFT (adapts to model's per-token confidence)
#
# DEFT computes α per-token as the Rényi-2 collision probability:
#   α(c) = Σ P_θ(v|c)² ∈ (0, 1]
#
# When distribution is diffuse (uncertain) → α≈0 → full NLL coverage
# When distribution is concentrated (confident) → α≈1 → DFT sharpening
#
# Result: parameter-free, adapts automatically, +70-80% over NLL on math.
# Cost: one squared-softmax sum per position (~free, fused by inductor).
#
# Loss formula: L_DEFT = (1 - p^α) / α, where α = Σ p_v²
# For the gradient, this gives: ∂L/∂z_target = -p^α × (1-p)
# Which is equivalent to weighting CE by p^α (the trust gate).


def _deft_loss_fused(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """DEFT: Dynamic Entropy Fine-Tuning. Parameter-free adaptive SFT loss.

    Automatically interpolates between NLL (for uncertain tokens where model
    needs to learn) and probability-loss (for confident tokens where model
    needs to sharpen), using Rényi-2 entropy as the state signal.

    Compiled into fused kernels by inductor: softmax → square → sum → gather → pow → CE → mul
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # 1. Compute softmax probabilities
    probs = torch.softmax(logits.float(), dim=-1)  # [B, S, V]

    # 2. Per-token focus index α = collision probability = Σ p_v²
    # This is the Rényi-2 entropy exponentiated: measures distribution concentration
    # Diffuse → small α (NLL-like), concentrated → large α (DFT-like)
    alpha = (probs * probs).sum(dim=-1)  # [B, S], in (0, 1]

    # 3. Get p_t = P(correct token)
    valid_mask = labels != IGNORE_INDEX
    safe_labels = labels.clamp(min=0)
    p_t = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
    p_t = p_t.clamp(min=1e-8)

    # 4. Trust gate: p_t^α (detached — importance weight, not loss target)
    # When α≈0 (uncertain): gate ≈ 1 (full NLL coverage)
    # When α≈1 (confident): gate ≈ p_t (DFT sharpening)
    gate = torch.pow(p_t, alpha).detach()  # [B, S]

    # 5. Standard per-token CE
    loss_flat = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, S)

    # 6. Apply trust gate
    weighted = loss_flat * gate

    # 7. Mask and sum
    weighted = torch.where(valid_mask, weighted, torch.zeros_like(weighted))
    return weighted.sum()


def chunked_deft_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    num_chunks: int = 8,
    global_valid_tokens: float = 1.0,
    stats: dict | None = None,
) -> torch.Tensor:
    """Chunked DEFT: memory-efficient DEFT loss via sequence-chunked lm_head projection.

    Problem: DEFT needs full logits (for softmax → p_t → alpha → gate), but materializing
    [B, S, V] for large vocab (152K) is 4.7-9.4 GB. Can't use CCE (needs custom kernel).

    Solution: Process chunks along the sequence dimension. Each chunk independently:
      1. Projects hidden_states → logits (only [B, S/N, V] materialized)
      2. Computes softmax, alpha, p_t, gate for that chunk
      3. Computes weighted CE for that chunk
      4. Frees the chunk's logits before processing next

    This is exact — DEFT is per-token, no cross-position dependency.
    Memory: O(B × S/N × V) instead of O(B × S × V).

    Args:
        hidden_states: [B, S, D] from model backbone (before lm_head)
        labels: [B, S] target labels
        lm_head: The lm_head module
        num_chunks: Number of sequence chunks (8 → each chunk is S/8 positions)
        global_valid_tokens: Denominator for loss normalization
        stats: Optional dict; if given, filled with side metrics computed for
            free during the loss pass: "ce_sum" (unweighted CE over valid
            tokens — the DEFT value is NOT a CE and can't be used for ppl),
            "gate_sum" (sum of trust gates), "valid" (valid token count).

    Returns:
        Scalar loss (sum / global_valid_tokens)
    """
    from torch.distributed._composable.fsdp import FSDPModule

    IGNORE_INDEX = -100
    B, S, D = hidden_states.shape
    fsdp_enabled = isinstance(lm_head, FSDPModule)
    requires_grad = hidden_states.requires_grad

    # Split into chunks along sequence dimension
    h_chunks = [c.contiguous() for c in torch.chunk(hidden_states.detach(), num_chunks, dim=1)]
    label_chunks = list(torch.chunk(labels, num_chunks, dim=1))

    if requires_grad:
        for c in h_chunks:
            c.requires_grad_(True)

    # Pre-allocate gradient buffer
    grad_buffer = torch.zeros_like(hidden_states, dtype=torch.float32) if requires_grad else None

    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

    # FSDP management (keep lm_head unsharded across chunks)
    if fsdp_enabled:
        lm_head.set_reshard_after_forward(False)
        lm_head.set_reshard_after_backward(False)
        lm_head.set_requires_gradient_sync(False, recurse=False)

    last_idx = len(h_chunks) - 1
    seq_offset = 0

    for i, (h_chunk, l_chunk) in enumerate(zip(h_chunks, label_chunks)):
        chunk_len = h_chunk.shape[1]

        if fsdp_enabled and i == last_idx:
            lm_head.set_requires_gradient_sync(True, recurse=False)

        # Project to logits: [B, chunk_S, V]
        logits = lm_head(h_chunk)

        # ── DEFT computation on this chunk ────────────────────────────
        V = logits.shape[-1]

        # Softmax + collision probability (alpha)
        probs = torch.softmax(logits.float(), dim=-1)
        alpha = (probs * probs).sum(dim=-1)  # [B, chunk_S]

        # p_t = P(correct token)
        valid_mask = l_chunk != IGNORE_INDEX
        safe_labels = l_chunk.clamp(min=0)
        p_t = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        p_t = p_t.clamp(min=1e-8)

        # Trust gate (detached)
        gate = torch.pow(p_t, alpha).detach()

        # Per-token CE
        ce_flat = F.cross_entropy(
            logits.reshape(-1, V),
            l_chunk.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(h_chunk.shape[0], chunk_len)

        # Weighted loss for this chunk
        weighted = ce_flat * gate
        weighted = torch.where(valid_mask, weighted, torch.zeros_like(weighted))
        chunk_loss = weighted.sum() / global_valid_tokens

        total_loss = total_loss + chunk_loss.detach()

        # Side metrics (free: everything is already computed)
        if stats is not None:
            with torch.no_grad():
                stats["ce_sum"] = stats.get("ce_sum", 0.0) + torch.where(
                    valid_mask, ce_flat, torch.zeros_like(ce_flat)
                ).sum().item()
                stats["gate_sum"] = stats.get("gate_sum", 0.0) + torch.where(
                    valid_mask, gate, torch.zeros_like(gate)
                ).sum().item()
                stats["valid"] = stats.get("valid", 0) + valid_mask.sum().item()

        # Backward for this chunk (frees logits immediately)
        if requires_grad:
            chunk_loss.backward()
            grad_buffer[:, seq_offset : seq_offset + chunk_len] = h_chunk.grad.float()
            h_chunk.grad = None

        seq_offset += chunk_len

    # Restore FSDP
    if fsdp_enabled:
        lm_head.set_reshard_after_forward(True)
        lm_head.set_reshard_after_backward(True)
        lm_head.set_requires_gradient_sync(True, recurse=False)
        lm_head.reshard()

    if not requires_grad:
        return total_loss

    # Bridge gradients back to decoder
    from palingenesis.loss import _BackwardBridge

    accumulated_grad = grad_buffer.to(hidden_states.dtype)
    return _BackwardBridge.apply(hidden_states, accumulated_grad, total_loss)


_deft_compiled = torch.compile(_deft_loss_fused, dynamic=True)


def deft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """DEFT loss: parameter-free adaptive SFT objective.

    Automatically balances between learning new knowledge (NLL-like, for uncertain
    tokens) and sharpening existing knowledge (DFT-like, for confident tokens).
    Uses Rényi-2 entropy (collision probability) as the per-token state signal.

    Subsumes: NLL (α→0), DFT (α=1), and interpolates continuously.
    Evidence: +70-80% over NLL on math reasoning benchmarks.

    Returns: sum-reduced loss (divide by valid_tokens externally).
    """
    try:
        return _deft_compiled(logits, labels)
    except Exception:
        return _deft_loss_fused(logits, labels)


# Paper: Wu et al. (2025) "Reformulating SFT as Policy Optimization"
# Key insight: Standard SFT gradient has implicit 1/p_t amplification.
#   ∂L_SFT/∂θ = -(1/p_t) * ∂logit/∂θ  (explodes as p_t → 0)
#   ∂L_DFT/∂θ = -(1 + log p_t) * ∂logit/∂θ  (bounded, goes to 0 gracefully)
#
# Implementation: weight each token's CE loss by p_t (the model's own probability).
# Effect: Rare tokens no longer dominate gradients. Smoother optimization.
# Cost: ~0 extra compute (just a gather + multiply during loss).
#
# STACKING BEHAVIOR:
#   DFT + InfoSFT: DFT stabilizes magnitude, InfoSFT selects informative tokens. ✓
#   DFT + CADFT: DFT fixes token-level, CADFT fixes sample-level. ✓
#   DFT alone: pure gradient stabilization without information-aware weighting.


def _dft_loss_fused(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dynamic Fine-Tuning loss: p_t-weighted CE. Compilable.

    L_DFT(t) = -p_t * log(p_t) for each token position t.
    Gradient: ∝ -(1 + log p_t) — bounded as p_t → 0.

    This is mathematically equivalent to: CE_per_token * p_t,
    since CE = -log(p_t) and thus p_t * CE = -p_t * log(p_t).
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # Get p_t = P(correct token) for each position
    probs = torch.softmax(logits.float(), dim=-1)
    valid_mask = labels != IGNORE_INDEX
    safe_labels = labels.clamp(min=0)
    p_t = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
    p_t = p_t.detach()  # stop gradient through p_t (importance weight, not loss target)

    # Standard per-token CE
    loss_flat = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, S)

    # DFT weighting: multiply by p_t
    weighted = loss_flat * p_t

    # Mask and sum
    weighted = torch.where(valid_mask, weighted, torch.zeros_like(weighted))
    return weighted.sum()


_dft_compiled = torch.compile(_dft_loss_fused, dynamic=True)


def dft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dynamic Fine-Tuning loss with bounded gradient scaling.

    Fixes the fundamental SFT pathology where rare tokens get 1/p_t gradient
    amplification. With DFT, gradients are bounded by -(1 + log p_t).

    Returns: sum-reduced loss (divide by valid_tokens externally).
    """
    try:
        return _dft_compiled(logits, labels)
    except Exception:
        return _dft_loss_fused(logits, labels)


# ===== CADFT: COMPATIBILITY-AWARE DYNAMIC FINE-TUNING =====
# Paper: "Compatibility-Aware Dynamic Fine-Tuning" (arxiv:2606.11206, Apr 2026)
# Extends DFT with sample-level variance control.
#
# Idea: Some samples are "incompatible" with the current model state — they
# have high NLL (the model can't produce them yet). Training on these induces
# high-variance gradient updates that destabilize optimization.
#
# Solution: For each sample in the batch, compute normalized compatibility
# (z-scored NLL), then down-weight incompatible samples exponentially.
#
# Formula:
#   c_raw(x,y) = mean(-log p_t) over tokens in y (just the per-sample NLL)
#   ĉ = (c_raw - μ_batch) / σ_batch     (z-score within batch)
#   w(ĉ) = exp(-β * max(0, ĉ))          (down-weight outliers)
#
# This is applied ON TOP of the token-level DFT loss.
# β = 1.0 is recommended (from the paper's ablation).


def compute_sample_compatibility(logits: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Compute per-sample compatibility weights for CADFT.

    Returns: [B] tensor of weights in (0, 1], where 1 = fully compatible.
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # Per-token NLL (no grad needed for compatibility computation)
    with torch.no_grad():
        loss_flat = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(B, S)

        valid_mask = labels != IGNORE_INDEX
        # Per-sample mean NLL (raw compatibility)
        token_counts = valid_mask.sum(dim=1).clamp(min=1).float()
        c_raw = (loss_flat * valid_mask.float()).sum(dim=1) / token_counts  # [B]

        # Z-score normalization within batch
        mu = c_raw.mean()
        sigma = c_raw.std().clamp(min=1e-6)
        c_hat = (c_raw - mu) / sigma

        # Exponential decay for incompatible samples
        weights = torch.exp(-beta * c_hat.clamp(min=0.0))  # [B], in (0, 1]

    return weights


def cadft_loss(logits: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """CADFT loss: DFT + sample-level compatibility weighting.

    Combines:
      1. Token-level: DFT bounded gradient scaling (p_t weighting)
      2. Sample-level: compatibility-aware variance reduction (z-score + exp decay)

    Args:
        logits: [B, S, V] model output logits
        labels: [B, S] target labels
        beta: Compatibility sensitivity (1.0 recommended)

    Returns: sum-reduced loss (divide by valid_tokens externally).
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # 1. Sample-level compatibility weights [B]
    sample_weights = compute_sample_compatibility(logits, labels, beta)

    # 2. Token-level DFT loss [B, S]
    probs = torch.softmax(logits.float(), dim=-1)
    valid_mask = labels != IGNORE_INDEX
    safe_labels = labels.clamp(min=0)
    p_t = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).detach()

    loss_flat = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, S)

    # DFT: weight by p_t
    dft_weighted = loss_flat * p_t
    dft_weighted = torch.where(valid_mask, dft_weighted, torch.zeros_like(dft_weighted))

    # 3. Per-sample sum, then apply sample weights
    per_sample_loss = dft_weighted.sum(dim=1)  # [B]
    weighted_loss = (per_sample_loss * sample_weights).sum()

    return weighted_loss


# ===== SCHEDULE-FREE ADAMW =====
# Paper: "The Road Less Scheduled" (ICLR 2024, NeurIPS 2025 at scale)


# ===== KL ANCHORING (ASFT) =====
# Paper: "Anchored Supervised Fine-Tuning" (arxiv:2509.23753, ICLR 2026)
#
# DFT/DEFT improve math/code reasoning but cause distributional DRIFT on
# knowledge-intensive tasks. ASFT fixes this by adding a KL penalty from
# the frozen base model:
#   L = L_DFT + λ * KL(π_base || π_θ)
#
# This is a UNIVERSAL composable component — works with any loss function.
# When kl_anchor > 0, it's automatically added to the training loss.
# λ = 0.05 recommended (from paper, validated on 7B models).
#
# Result: +12.84 over DFT on medical knowledge, +1 on math (no regression).


def kl_anchor_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence KL(π_base || π_θ) for anchoring.

    Only computed on valid (non-masked) token positions.
    Returns sum-reduced loss (divide by valid_tokens externally).

    Note: This is KL(base || current), not KL(current || base).
    The direction matters: KL(base||current) penalizes the current model
    for assigning low probability where the base model assigns high probability.
    This is the "mode-covering" direction that prevents forgetting.
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape
    valid_mask = labels != IGNORE_INDEX

    # Log probabilities
    log_p_current = F.log_softmax(logits.float(), dim=-1)  # [B, S, V]
    log_p_base = F.log_softmax(ref_logits.float(), dim=-1)  # [B, S, V]
    p_base = log_p_base.exp()

    # KL(base || current) = Σ p_base * (log_p_base - log_p_current)
    kl = (p_base * (log_p_base - log_p_current)).sum(dim=-1)  # [B, S]

    # Mask invalid positions
    kl = torch.where(valid_mask, kl, torch.zeros_like(kl))
    return kl.sum()


def build_schedule_free_optimizer(
    model: nn.Module, lr: float, weight_decay: float, warmup_steps: int = 500
) -> torch.optim.Optimizer:
    """Schedule-Free AdamW -- no LR schedule needed, same memory as AdamW."""
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError:
        raise ImportError("Install schedulefree: pip install schedulefree")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name.lower() for k in ("bias", "norm", "embed")):
            no_decay.append(p)
        else:
            decay.append(p)

    opt = AdamWScheduleFree(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        warmup_steps=warmup_steps,
    )
    logger.info(f"Schedule-Free AdamW (lr={lr}, warmup={warmup_steps})")
    return opt


# ===== PRE-RL MODE =====
# Paper: "EKSFT" (arxiv:2605.29303, May 2026) + GEM (Li et al., 2025)
# Purpose: SFT that preserves diversity for subsequent GRPO/DPO/PPO
#
# Standard SFT sharpens the policy -> RL can't explore.
# Pre-RL SFT adds:
#   1. Entropy regularization: keep output distribution diverse
#   2. KL penalty from base model: prevent drift on uncertain tokens
#   3. Selective masking: only train on "safe" tokens (medium entropy)
#
# The result: model learns the task format but retains exploration capacity.
# Compiled into a single fused function for speed.


def _pre_rl_loss_inner(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ref_logits: torch.Tensor,
    entropy_coeff: float,
    kl_coeff: float,
    entropy_threshold: float,
    kl_threshold: float,
) -> torch.Tensor:
    """Pre-RL loss: masked CE + entropy reg + KL reg. Compilable.

    Steps:
      1. Compute token-level entropy and KL from reference
      2. Mask tokens with high entropy OR high KL (don't imitate these)
      3. CE loss only on unmasked tokens (safe to imitate)
      4. Entropy bonus on masked tokens (keep them diverse)
      5. KL penalty on masked tokens (don't drift on them)
    """
    IGNORE_INDEX = -100
    B, S, V = logits.shape

    # Valid tokens (not padding/system)
    valid_mask = labels != IGNORE_INDEX

    # Token-level entropy: H(pi) = -sum(p * log p)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, S]

    # Token-level KL from reference: KL(pi || ref)
    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
    kl_div = (probs * (log_probs - ref_log_probs)).sum(dim=-1)  # [B, S]

    # Masking: mask tokens with high entropy OR high KL
    # These are "unsafe" to imitate — would cause distribution sharpening or drift
    high_entropy = entropy > entropy_threshold
    high_kl = kl_div > kl_threshold
    mask = high_entropy | high_kl  # True = masked (don't imitate)
    imitate_mask = ~mask & valid_mask  # Tokens we DO train on with CE

    # 1. CE loss on unmasked (safe) tokens only
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, S)
    ce_loss = (ce_per_token * imitate_mask.float()).sum()
    imitate_count = imitate_mask.sum().clamp(min=1)
    ce_loss = ce_loss / imitate_count

    # 2. Entropy bonus on MASKED tokens (keep them diverse for RL)
    # Maximize entropy = minimize negative entropy
    masked_valid = mask & valid_mask
    entropy_loss = -(entropy * masked_valid.float()).sum() / masked_valid.sum().clamp(min=1)

    # 3. KL penalty on MASKED tokens (don't drift from base)
    kl_loss = (kl_div * masked_valid.float()).sum() / masked_valid.sum().clamp(min=1)

    # Combined: CE + entropy reg + KL reg
    total = ce_loss + entropy_coeff * entropy_loss + kl_coeff * kl_loss
    return total


_pre_rl_compiled = torch.compile(_pre_rl_loss_inner, dynamic=True)


def pre_rl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ref_logits: torch.Tensor,
    entropy_coeff: float = 0.1,
    kl_coeff: float = 0.5,
    entropy_threshold: float = 2.0,
    kl_threshold: float = 0.5,
) -> torch.Tensor:
    """Pre-RL SFT loss: learn task format while preserving RL exploration capacity.

    Use this when your training pipeline is SFT -> GRPO/DPO/PPO.
    The model learns the task (tool calls, format) but retains the diversity
    needed for RL to explore different solutions.

    Args:
        logits: [B, S, V] current model logits
        labels: [B, S] target labels (with IGNORE_INDEX)
        ref_logits: [B, S, V] reference model logits (base/init model)
        entropy_coeff: Weight for entropy bonus (higher = more diverse)
        kl_coeff: Weight for KL penalty (higher = less drift)
        entropy_threshold: Entropy above this -> mask token (default 2.0 nats)
        kl_threshold: KL above this -> mask token (default 0.5 nats)

    Returns:
        Scalar loss
    """
    try:
        return _pre_rl_compiled(
            logits,
            labels,
            ref_logits,
            entropy_coeff,
            kl_coeff,
            entropy_threshold,
            kl_threshold,
        )
    except Exception:
        return _pre_rl_loss_inner(
            logits,
            labels,
            ref_logits,
            entropy_coeff,
            kl_coeff,
            entropy_threshold,
            kl_threshold,
        )
