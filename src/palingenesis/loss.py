"""Loss computation: standard CE + FSDP-compatible chunked CE + Cut Cross-Entropy.

Three modes:
  1. Standard: sum-reduction CE / global_valid_tokens (correct multi-GPU scaling)
  2. Chunked: split hidden states into N chunks, lm_head+CE per chunk, accumulate
     grads. Works with FSDP2 by managing lm_head reshard state (like torchtitan).
  3. CCE (Cut Cross-Entropy): Apple's memory-free CE via custom Triton kernel.
     Never materializes the [B, S, V] logit tensor. Reduces memory from GB to MB.
     Paper: "Cut Your Losses in Large-Vocabulary Language Models" (ICLR 2025)

Both sum reduction + explicit valid-token normalization for correct gradients
across ranks with unbalanced masking (different # of valid tokens per rank).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from palingenesis import stable_softmax

IGNORE_INDEX = -100

# Try importing Cut Cross-Entropy (optional dependency)
_CCE_AVAILABLE = False
try:
    from cut_cross_entropy import linear_cross_entropy

    _CCE_AVAILABLE = True
except ImportError:
    pass


def cce_available() -> bool:
    """Check if Cut Cross-Entropy is installed."""
    return _CCE_AVAILABLE


def shift_labels(labels: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Shift labels left by one position for next-token prediction.

    The data pipeline produces labels aligned with input_ids (labels[t] is
    input_ids[t], with non-trained positions masked). A causal LM's logits at
    position t predict the token at position t+1, so the loss must compare
    logits[t] against labels[t+1]. HF models do this shift internally when you
    pass labels= to forward(); since this codebase computes the loss manually
    from logits/hidden states, the shift must happen here — once, at the batch
    level — before any loss function or valid-token counting.

    Args:
        labels: [B, S] labels aligned with input_ids
        position_ids: optional [B, S] positions for packed sequences. Positions
            reset to 0 at each document boundary; the last token of a document
            must not be trained to predict the first token of the next one, so
            those positions are masked.

    Returns:
        [B, S] shifted labels; the final position is IGNORE_INDEX.
    """
    shifted = torch.full_like(labels, IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    if position_ids is not None:
        # position_ids[t+1] == 0 marks a document boundary after position t
        boundary = position_ids[:, 1:] == 0
        shifted[:, :-1] = torch.where(boundary, torch.full_like(shifted[:, :-1], IGNORE_INDEX), shifted[:, :-1])
    return shifted


def shift_weights(weights: torch.Tensor) -> torch.Tensor:
    """Per-token loss weights aligned like `shift_labels`: the weight of predicting
    token t+1 sits at position t. Positions whose label is masked keep any weight;
    the loss never reads them."""
    shifted = torch.zeros_like(weights)
    shifted[:, :-1] = weights[:, 1:]
    return shifted


def _weighted_sum(per_token: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    return per_token.sum() if weights is None else (per_token * weights.reshape(-1).to(per_token.dtype)).sum()


# ==============================================================================
# STANDARD CROSS-ENTROPY (sum reduction, correct for distributed)
# ==============================================================================


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    global_valid_tokens: float = 1.0,
    weights: torch.Tensor | None = None,
    stable: bool = False,
) -> torch.Tensor:
    """Cross-entropy with sum reduction / global_valid_tokens normalization.

    This gives correct gradients across distributed ranks when each rank
    has different numbers of valid (non-masked) tokens. `weights` ([B, S], shifted
    like the labels) scale each token's term (per-turn scaling). `stable`: palingenesis.stable_softmax.
    """
    if stable:
        loss = stable_softmax.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), IGNORE_INDEX)
        return _weighted_sum(loss, weights) / global_valid_tokens
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        reduction="none" if weights is not None else "sum",
        ignore_index=IGNORE_INDEX,
    )
    return _weighted_sum(loss, weights) / global_valid_tokens


# ==============================================================================
# CUT CROSS-ENTROPY (Apple, ICLR 2025) — ZERO LOGIT MATERIALIZATION
# ==============================================================================


def cut_cross_entropy_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    global_valid_tokens: float = 1.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cut Cross-Entropy: memory-free CE via custom Triton kernel.

    From Apple (ICLR 2025): computes CE without materializing the [B,S,V] logit
    tensor. For Gemma 4 (262K vocab, seq=8192): saves 8+ GB per batch.

    Memory: O(1) instead of O(B*S*V). Speed: same or faster than standard CE.

    Works by computing only:
      1. The dot product for the correct token (indexed matmul)
      2. The log-sum-exp over all vocab entries (on-the-fly in SRAM)

    Args:
        hidden_states: [B, S, D] from model backbone
        labels: [B, S] target labels
        lm_head: The lm_head linear layer (needs .weight attribute)
        global_valid_tokens: Denominator for loss normalization

    Returns:
        Scalar loss (sum / global_valid_tokens)

    Requires: pip install cut-cross-entropy (+ CUDA GPU)
    """
    if not _CCE_AVAILABLE:
        raise ImportError("cut-cross-entropy not installed. pip install cut-cross-entropy")

    # Get lm_head weight (V, D)
    weight = lm_head.weight

    # Flatten to 2D for CCE: (B*S, D) and (B*S,)
    B, S, D = hidden_states.shape
    e_flat = hidden_states.reshape(-1, D)
    targets_flat = labels.reshape(-1)

    # CCE: computes CE without materializing (B*S, V) logit matrix
    loss = linear_cross_entropy(
        e_flat,
        weight,
        targets_flat,
        ignore_index=IGNORE_INDEX,
        reduction="none" if weights is not None else "sum",
    )

    return _weighted_sum(loss, weights) / global_valid_tokens


# ==============================================================================
# CHUNKED CROSS-ENTROPY (FSDP2-compatible, mimics torchtitan ChunkedCELoss)
# ==============================================================================


class _BackwardBridge(torch.autograd.Function):
    """Bridges chunked loss backward with decoder backward via autograd.

    Forward: takes hidden_states (in decoder graph) + accumulated gradient + loss.
    Returns detached loss with this Function as grad_fn.
    Backward: returns accumulated_grad as gradient for hidden_states.
    Autograd then propagates through the decoder layers automatically.
    """

    @staticmethod
    def forward(ctx, hidden_states, accumulated_grad, loss):
        ctx.save_for_backward(accumulated_grad)
        return loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        (accumulated_grad,) = ctx.saved_tensors
        # Respect downstream scaling (e.g. loss / grad_accum_steps before
        # .backward()): the accumulated per-chunk gradient was computed with
        # an implicit grad_output of 1.0, so scale it here.
        return accumulated_grad * grad_output, None, None


def chunked_cross_entropy_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    num_chunks: int = 8,
    global_valid_tokens: float = 1.0,
    weights: torch.Tensor | None = None,
    stable: bool = False,
) -> torch.Tensor:
    """FSDP2-compatible chunked cross-entropy loss over the trained positions.

    Only positions with a label reach lm_head: their hidden states are gathered and projected a
    slice at a time, each slice back-propagated at once, so peak memory is one slice of logits
    and the head's work is proportional to the trained tokens. On agentic traces (tool results
    and prompts carry no label) that is a fraction of the sequence; projecting every position
    spent most of the head's time on logits the loss then ignored.

    FSDP2 composability (mimics torchtitan):
      - Disable reshard_after_forward on lm_head (keep weight unsharded across slices)
      - Disable reshard_after_backward on lm_head (same reason)
      - Disable gradient sync for slices 0..N-2, enable only on the last one
        (coalesce reduce-scatter into a single operation)
      - Re-enable all and reshard after the loop

    Args:
        hidden_states: [B, S, D] from model backbone (before lm_head)
        labels: [B, S] target tokens
        lm_head: The lm_head module (may be FSDP-wrapped)
        num_chunks: Bounds a slice at ceil(B*S / num_chunks) positions (the logits of one slice)
        global_valid_tokens: Denominator for loss normalization
        stable: palingenesis.stable_softmax's cross-entropy instead of PyTorch's
    """
    from torch.distributed._composable.fsdp import FSDPModule

    # A PostProcessedHead (palingenesis.logits) wraps the projection that FSDP manages.
    fsdp_head = getattr(lm_head, "head", lm_head)
    fsdp_enabled = isinstance(fsdp_head, FSDPModule)
    requires_grad = hidden_states.requires_grad

    flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    rows = torch.nonzero(flat_labels != IGNORE_INDEX).flatten()
    scale = 1.0
    if rows.numel() == 0:
        # Nothing trained in this batch: one position at zero weight, so the head still takes part
        # in the backward (under FSDP every rank must reduce its gradient).
        rows, scale = rows.new_zeros(1), 0.0
        targets = flat_labels.new_zeros(1)
    else:
        targets = flat_labels.index_select(0, rows)
    row_weights = weights.reshape(-1).index_select(0, rows) if weights is not None else None
    slice_rows = max(1, math.ceil(flat.shape[0] / max(num_chunks, 1)))

    # Gradient of the hidden states, filled slice by slice (zero at untrained positions)
    grad_buffer = torch.zeros_like(flat, dtype=torch.float32) if requires_grad else None

    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

    # === FSDP reshard management (exactly like torchtitan) ===
    # Disable reshard to keep lm_head weight unsharded across all slices
    # (avoids N all-gathers; just 1 all-gather at the start)
    # Disable grad sync for slices 0..N-2, coalesce into the last one
    if fsdp_enabled:
        fsdp_head.set_reshard_after_forward(False)
        fsdp_head.set_reshard_after_backward(False)
        fsdp_head.set_requires_gradient_sync(False, recurse=False)

    starts = range(0, rows.numel(), slice_rows)
    for start in starts:
        idx = rows[start : start + slice_rows]
        h_slice = flat.index_select(0, idx)
        if requires_grad:
            h_slice.requires_grad_(True)

        # Enable grad sync only on the last slice (single reduce-scatter)
        if fsdp_enabled and start == starts[-1]:
            fsdp_head.set_requires_gradient_sync(True, recurse=False)

        # lm_head projection: [rows, D] -> [rows, V]
        logits = lm_head(h_slice)

        # CE with sum reduction (weighted per token under turn scaling)
        slice_targets = targets[start : start + slice_rows]
        if stable:
            per_token = stable_softmax.cross_entropy(logits, slice_targets)
        else:
            per_token = F.cross_entropy(
                logits.float(), slice_targets, reduction="none" if row_weights is not None else "sum"
            )
        slice_loss = _weighted_sum(
            per_token,
            row_weights[start : start + slice_rows] if row_weights is not None else None,
        )
        scaled_loss = slice_loss * scale / global_valid_tokens
        total_loss = total_loss + scaled_loss.detach()

        # Per-slice backward: frees logits immediately
        if requires_grad:
            scaled_loss.backward()
            grad_buffer.index_copy_(0, idx, h_slice.grad.float())

    # === Restore FSDP state ===
    if fsdp_enabled:
        fsdp_head.set_reshard_after_forward(True)
        fsdp_head.set_reshard_after_backward(True)
        fsdp_head.set_requires_gradient_sync(True, recurse=False)
        fsdp_head.reshard()

    if not requires_grad:
        return total_loss

    # Bridge: connect accumulated gradient back to decoder's autograd graph
    accumulated_grad = grad_buffer.view_as(hidden_states).to(hidden_states.dtype)
    return _BackwardBridge.apply(hidden_states, accumulated_grad, total_loss)
