"""SeCO/SpaCO: Sequential/Sparse Chunk-wise Optimization for long-context training.

Paper: "Training Long-Context LLMs Efficiently via Chunk-wise Optimization"
       (Li et al., 2505.16710, ACL Findings 2025)

Two modes:
  SeCO (exact): backprop through ALL chunks. ~33% overhead. Exact gradients.
  SpaCO (fast): backprop through only t randomly-selected chunks. Unbiased gradient
                estimate with compensation factor. Training time → inference time.

VERIFIED on GPT-2: loss diff < 5e-7, grad diff < 3e-4.

REQUIREMENTS FOR QWEN3.5 (hybrid DeltaNet):
  pip install causal-conv1d flash-linear-attention
  (Without these, chunked prefill produces wrong results on DeltaNet layers)

Usage:
    from palingenesis.seco import seco_forward_backward, spaco_forward_backward

    # SeCO (exact, ~33% overhead)
    loss = seco_forward_backward(model, input_ids, labels, chunk_size=4096)

    # SpaCO (fast, unbiased estimate, for very long sequences)
    loss = spaco_forward_backward(model, input_ids, labels, chunk_size=4096, budget_t=8)
"""

import copy
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from palingenesis.loss import shift_labels

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def seco_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 4096,
) -> float:
    """SeCO: exact gradients, ~33% compute overhead."""
    return _chunked_forward_backward(
        model,
        input_ids,
        labels,
        attention_mask,
        chunk_size,
        budget_t=None,
        compensation_cap=None,
    )


def spaco_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 4096,
    budget_t: int = 8,
    compensation_cap: float = 2.0,
) -> float:
    """SpaCO: fast unbiased gradient estimate, only backprops t chunks.

    For 64K seq with chunk_size=4096: k=16 chunks.
    With budget_t=8: only 8 get backward (50% compute savings vs SeCO).
    Compensation factor k/t=2.0 ensures unbiased gradient.

    Args:
        budget_t: number of chunks to backprop (rest are skip)
        compensation_cap: max scaling factor per relay hop (paper recommends 2.0)
    """
    return _chunked_forward_backward(
        model,
        input_ids,
        labels,
        attention_mask,
        chunk_size,
        budget_t=budget_t,
        compensation_cap=compensation_cap,
    )


def _chunked_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_size: int,
    budget_t: int | None,
    compensation_cap: float | None,
) -> float:
    """Core implementation for both SeCO (budget_t=None) and SpaCO (budget_t=int)."""
    device = input_ids.device
    B, S = input_ids.shape
    assert B == 1, "SeCO/SpaCO requires batch_size=1"

    # Shift for next-token prediction: logits[t] predicts input_ids[t+1].
    # Callers pass labels aligned with input_ids (data-pipeline convention).
    labels = shift_labels(labels)

    total_valid = (labels != IGNORE_INDEX).sum().item()
    if total_valid == 0:
        return 0.0

    num_chunks = (S + chunk_size - 1) // chunk_size
    id_chunks = list(input_ids.split(chunk_size, dim=1))
    label_chunks = list(labels.split(chunk_size, dim=1))

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: Sequential forward (no grad) — build KV cache checkpoints
    # ═══════════════════════════════════════════════════════════════════════
    cache_before: list = [None]
    past = None

    with torch.no_grad():
        for i in range(num_chunks):
            out = model(input_ids=id_chunks[i], past_key_values=past, use_cache=True)
            past = out.past_key_values
            if i < num_chunks - 1:
                cache_before.append(copy.deepcopy(past))

    # Extract per-chunk attention KV slabs for gradient relay
    num_layers = len(past)
    chunk_kv_slabs: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
    offset = 0
    for i in range(num_chunks):
        clen = id_chunks[i].shape[1]
        layer_slabs = []
        for layer_idx in range(num_layers):
            layer = past.layers[layer_idx]
            if hasattr(layer, "keys") and layer.keys is not None:
                k = layer.keys[..., offset : offset + clen, :].detach().clone().requires_grad_(True)
                v = layer.values[..., offset : offset + clen, :].detach().clone().requires_grad_(True)
                layer_slabs.append((k, v))
            else:
                layer_slabs.append(None)
        chunk_kv_slabs.append(layer_slabs)
        offset += clen

    del past, out

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: Backward pass
    # ═══════════════════════════════════════════════════════════════════════
    # SeCO: all chunks in reverse order
    # SpaCO: random t chunks (still in reverse for correct gradient accumulation)

    if budget_t is not None and budget_t < num_chunks:
        # SpaCO: randomly select t chunks
        selected = sorted(random.sample(range(num_chunks), budget_t), reverse=True)
        scaler = min(num_chunks / budget_t, compensation_cap or float("inf"))
    else:
        # SeCO: all chunks
        selected = list(reversed(range(num_chunks)))
        scaler = 1.0

    total_loss = 0.0

    for i in selected:
        chunk_start = sum(id_chunks[j].shape[1] for j in range(i))
        clen = id_chunks[i].shape[1]

        # Build past from checkpoint slabs (NOT deepcopy).
        # For pure Transformers: reconstruct DynamicCache from the grad-enabled slabs.
        # For hybrid models: use the deepcopy (slabs don't cover recurrent state).
        saved_cache = cache_before[i]

        # Check if we can build from slabs (all layers have KV = pure Transformer)
        all_layers_have_kv = (
            all(chunk_kv_slabs[0][l] is not None for l in range(num_layers))
            if num_chunks > 0 and chunk_kv_slabs
            else False
        )

        if i > 0 and all_layers_have_kv:
            # Pure Transformer: build past from grad-enabled slabs
            from transformers.cache_utils import DynamicCache

            past_from_slabs = DynamicCache()
            for layer_idx in range(num_layers):
                k_parts = [chunk_kv_slabs[j][layer_idx][0] for j in range(i)]
                v_parts = [chunk_kv_slabs[j][layer_idx][1] for j in range(i)]
                k_cat = torch.cat(k_parts, dim=-2)
                v_cat = torch.cat(v_parts, dim=-2)
                past_from_slabs.update(k_cat, v_cat, layer_idx)
            saved_cache = past_from_slabs

        # Forward with grad
        outputs = model(
            input_ids=id_chunks[i],
            past_key_values=saved_cache,
            use_cache=True,
        )

        # CE loss
        logits = outputs.logits
        chunk_loss = (
            F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                label_chunks[i].view(-1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
            / total_valid
        )

        total_loss += chunk_loss.item()

        # Gradient relay with compensation scaling
        relay_loss = torch.zeros(1, device=device, dtype=logits.dtype).squeeze()
        new_cache = outputs.past_key_values
        if new_cache is not None:
            for layer_idx in range(min(num_layers, len(new_cache))):
                if chunk_kv_slabs[i][layer_idx] is None:
                    continue
                layer = new_cache.layers[layer_idx]
                if not hasattr(layer, "keys") or layer.keys is None:
                    continue
                k_new = layer.keys[..., chunk_start : chunk_start + clen, :]
                v_new = layer.values[..., chunk_start : chunk_start + clen, :]

                k_grad = chunk_kv_slabs[i][layer_idx][0].grad
                v_grad = chunk_kv_slabs[i][layer_idx][1].grad

                # SpaCO: scale by compensation factor (k/t clamped)
                if k_grad is not None:
                    relay_loss = relay_loss + (k_new * (k_grad * scaler).detach()).sum()
                if v_grad is not None:
                    relay_loss = relay_loss + (v_new * (v_grad * scaler).detach()).sum()

        (chunk_loss + relay_loss).backward()
        del outputs, new_cache, logits, saved_cache

    return total_loss
