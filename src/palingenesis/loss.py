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

import torch
import torch.nn as nn
import torch.nn.functional as F

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
) -> torch.Tensor:
    """Cross-entropy with sum reduction / global_valid_tokens normalization.

    This gives correct gradients across distributed ranks when each rank
    has different numbers of valid (non-masked) tokens. `weights` ([B, S], shifted
    like the labels) scale each token's term (per-turn scaling).
    """
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
) -> torch.Tensor:
    """FSDP2-compatible chunked cross-entropy loss.

    Splits hidden states into N chunks along the sequence dimension and computes
    lm_head + CE per chunk sequentially. Reduces peak memory from O(B*S*V) to
    O(B*S/N*V).

    FSDP2 composability (mimics torchtitan):
      - Disable reshard_after_forward on lm_head (keep weight unsharded across chunks)
      - Disable reshard_after_backward on lm_head (same reason)
      - Disable gradient sync for chunks 0..N-2, enable only on last chunk
        (coalesce reduce-scatter into a single operation)
      - Re-enable all and reshard after the loop

    Args:
        hidden_states: [B, S, D] from model backbone (before lm_head)
        labels: [B, S] target tokens
        lm_head: The lm_head module (may be FSDP-wrapped)
        num_chunks: Number of sequence chunks
        global_valid_tokens: Denominator for loss normalization
    """
    from torch.distributed._composable.fsdp import FSDPModule

    # A PostProcessedHead (palingenesis.logits) wraps the projection that FSDP manages.
    fsdp_head = getattr(lm_head, "head", lm_head)
    fsdp_enabled = isinstance(fsdp_head, FSDPModule)
    requires_grad = hidden_states.requires_grad

    # Split into contiguous chunks along sequence dim
    h_chunks = [c.contiguous() for c in torch.chunk(hidden_states.detach(), num_chunks, dim=1)]
    label_chunks = list(torch.chunk(labels, num_chunks, dim=1))
    weight_chunks = list(torch.chunk(weights, num_chunks, dim=1)) if weights is not None else [None] * len(h_chunks)

    # Make each chunk a leaf for gradient accumulation
    if requires_grad:
        for c in h_chunks:
            c.requires_grad_(True)

    # Pre-allocate gradient buffer
    grad_buffer = torch.zeros_like(hidden_states, dtype=torch.float32) if requires_grad else None

    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

    # === FSDP reshard management (exactly like torchtitan) ===
    # Disable reshard to keep lm_head weight unsharded across all chunks
    # (avoids N all-gathers; just 1 all-gather at the start)
    # Disable grad sync for chunks 0..N-2, coalesce into last chunk
    if fsdp_enabled:
        fsdp_head.set_reshard_after_forward(False)
        fsdp_head.set_reshard_after_backward(False)
        fsdp_head.set_requires_gradient_sync(False, recurse=False)

    last_idx = len(h_chunks) - 1
    seq_offset = 0

    for i, (h_chunk, l_chunk, w_chunk) in enumerate(zip(h_chunks, label_chunks, weight_chunks)):
        chunk_len = h_chunk.shape[1]

        # Enable grad sync only on the last chunk (single reduce-scatter)
        if fsdp_enabled and i == last_idx:
            fsdp_head.set_requires_gradient_sync(True, recurse=False)

        # lm_head projection: [B, chunk_S, D] -> [B, chunk_S, V]
        logits = lm_head(h_chunk)

        # CE with sum reduction (weighted per token under turn scaling)
        chunk_loss = _weighted_sum(F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            l_chunk.reshape(-1),
            reduction="none" if w_chunk is not None else "sum",
            ignore_index=IGNORE_INDEX,
        ), w_chunk)
        scaled_loss = chunk_loss / global_valid_tokens
        total_loss = total_loss + scaled_loss.detach()

        # Per-chunk backward: frees logits immediately
        if requires_grad:
            scaled_loss.backward()
            grad_buffer[:, seq_offset : seq_offset + chunk_len] = h_chunk.grad.float()
            h_chunk.grad = None

        seq_offset += chunk_len

    # === Restore FSDP state ===
    if fsdp_enabled:
        fsdp_head.set_reshard_after_forward(True)
        fsdp_head.set_reshard_after_backward(True)
        fsdp_head.set_requires_gradient_sync(True, recurse=False)
        fsdp_head.reshard()

    if not requires_grad:
        return total_loss

    # Bridge: connect accumulated gradient back to decoder's autograd graph
    accumulated_grad = grad_buffer.to(hidden_states.dtype)
    return _BackwardBridge.apply(hidden_states, accumulated_grad, total_loss)
