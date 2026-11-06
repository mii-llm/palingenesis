"""Losses over the output head's logits, a slice of rows at a time.

`chunked_rows_loss` projects the student's hidden states at the selected rows (trained positions)
to logits a slice at a time and differentiates each slice immediately, so neither the student's nor
a teacher's [tokens, vocabulary] logits ever exist in full: peak memory is one slice of each. The
head's gradients accumulate through ordinary autograd (FSDP2-safe, as palingenesis.loss's chunked
cross-entropy: the head stays unsharded across slices and its gradient is reduced once), and the
gradient of the hidden states reaches the backbone through a bridge node at the caller's backward.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from palingenesis.loss import _BackwardBridge

# fp32 elements of logits per slice and per side: 2^27 = 512 MB
SLICE_ELEMENTS = 2**27


def rows_per_slice(vocab: int, configured: int = 0) -> int:
    return configured if configured > 0 else max(1, SLICE_ELEMENTS // max(vocab, 1))


def chunked_rows_loss(
    hidden: Tensor,
    rows: Tensor,
    head: nn.Module,
    fn: Callable[[Tensor, int, int], tuple[Tensor, dict]],
    denom: float,
    slice_rows: int,
    stats: dict | None = None,
) -> Tensor:
    """sum over slices of fn(head(h[rows[a:b]]) in fp32, a, b) / denom, as a differentiable scalar.

    hidden [..., H] (any leading shape; flattened), rows [N] flat indices into it. `fn` returns the
    slice's summed loss (a scalar tensor) and a dict of floats to accumulate into `stats`. With
    gradient enabled, the slices are back-propagated as they are computed; the returned value carries
    the accumulated hidden-state gradient to the caller's backward.
    """
    from torch.distributed.fsdp import FSDPModule

    stats = {} if stats is None else stats
    flat = hidden.reshape(-1, hidden.shape[-1])
    n = rows.numel()
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    if n == 0:
        # still a graph node, so the caller's backward runs (no-op gradient)
        return total + hidden.sum() * 0.0 if hidden.requires_grad else total
    requires_grad = hidden.requires_grad and torch.is_grad_enabled()
    fsdp_head = getattr(head, "head", head)
    fsdp = isinstance(fsdp_head, FSDPModule)
    grad = torch.zeros_like(flat, dtype=torch.float32) if requires_grad else None
    source = flat.detach()
    bounds = [(a, min(a + slice_rows, n)) for a in range(0, n, slice_rows)]
    if fsdp:
        fsdp_head.set_reshard_after_forward(False)
        fsdp_head.set_reshard_after_backward(False)
        fsdp_head.set_requires_gradient_sync(False, recurse=False)
    try:
        for i, (a, b) in enumerate(bounds):
            if fsdp and i == len(bounds) - 1:
                fsdp_head.set_requires_gradient_sync(True, recurse=False)
            idx = rows[a:b]
            h = source.index_select(0, idx)
            if requires_grad:
                h.requires_grad_(True)
                with torch.enable_grad():
                    loss, slice_stats = fn(head(h).float(), a, b)
                    scaled = loss / denom
                if scaled.requires_grad:
                    scaled.backward()
                    if h.grad is not None:
                        grad.index_add_(0, idx, h.grad.float())
            else:
                with torch.no_grad():
                    loss, slice_stats = fn(head(h).float(), a, b)
                    scaled = loss / denom
            total = total + scaled.detach().float()
            for k, v in slice_stats.items():
                stats[k] = stats.get(k, 0.0) + float(v)
    finally:
        if fsdp:
            fsdp_head.set_reshard_after_forward(True)
            fsdp_head.set_reshard_after_backward(True)
            fsdp_head.set_requires_gradient_sync(True, recurse=False)
            fsdp_head.reshard()
    if not requires_grad:
        return total
    return _BackwardBridge.apply(hidden, grad.view_as(hidden).to(hidden.dtype), total)
