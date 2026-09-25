"""SeCO / SpaCO: chunk-wise training on sequences of any length.

Idea from "Training Long-Context LLMs Efficiently via Chunk-wise Optimization"
(Li et al., arXiv:2505.16710). The sequence is split into k chunks; only ONE
chunk's activations are alive at a time, so activation memory is set by the
chunk size, not by the sequence length. What grows with the sequence is only
the cache (attention K/V) and, during the backward sweep, its gradient.

  Stage 1  no grad: run the chunks in order through the model with a cache and
           keep, per chunk, its checkpoint: the K/V it appends to full-attention
           layers, and the state every bounded-state layer starts it from.
  Stage 2  for i = k..1: rebuild chunk i's cache from grad-carrying leaves of
           those checkpoints, recompute the chunk with grad, and backpropagate its
           loss plus the relay terms <out, leaf.grad>: the gradient that later
           chunks sent into what this chunk produced. Chunks run in reverse, so
           every such gradient is complete when its chunk is recomputed.

Every cache layer is one of two kinds, whatever the architecture:

  append-only    full attention (`DynamicLayer`): the K/V grows by one slab per
                 chunk. Slabs are stored once and their gradients accumulate.
  bounded state  sliding-window / chunked attention, linear attention (Gated
                 DeltaNet, ...), short convolutions (LFM2, ...): the layer's
                 whole state at each chunk boundary is snapshotted and treated as
                 a Markov state, with the gradient relayed boundary to boundary.
  (hybrid layers have both parts and get both treatments.)

This follows the installed transformers' own cache classes, so every model
that uses them (Llama, Mistral, Qwen2/3, Qwen3.5, Qwen3-Next, Gemma 2/3, LFM2,
GPT-OSS, ...) is covered without per-model code. Static, quantized and indexed
caches raise.

SeCO visits every chunk: exact gradients (tested equal to full backprop) for
one extra no-grad forward. SpaCO visits t = `budget` random chunks: relayed
gradients are scaled by k/t as in the paper, and so is each visited chunk's
loss. The paper's pseudocode leaves the loss unscaled, which we measured to
give E[g] = (t/k) * grad; with it scaled, E[g] ~= grad (measured scale
0.92-0.99, cosine 0.98-0.9999 at k=5, t=1-4). SpaCO is an estimate, not exact.

The one assumption is that the model's cached, chunked forward equals its full
forward. That holds for most transformers implementations but not all (we
measured Bamba, Jamba, Mamba and RecurrentGemma to differ), so
`verify_chunked_forward` checks it on the actual model; the trainer runs it
before the first step. Randomness (dropout, router jitter) is replayed: each
chunk's RNG state is saved in stage 1 and restored for its recomputation, so
gradients stay exact in train mode.

Rows may be right-padded (padding only follows real tokens, and causality keeps
it inert). Packed sequences are not supported.
"""

from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import cache_utils as cu

from palingenesis.logits import output_head, verify_output_head
from palingenesis.loss import chunked_cross_entropy_loss, shift_labels
from palingenesis.seco_attention import KVStore, check_host_budget, chunk_attention, reset_host_budget

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# transformers >= 5.15 keeps linear-attention states in dicts keyed by state
# index and passes only the NEW tokens to update_conv_state; earlier 5.x keeps
# one tensor per state and passes the already-assembled conv window.
_DICT_STATES = isinstance(cu.LinearAttentionLayer().conv_states, dict)

# Cache layer classes this module understands (any subclass of these that
# overrides their update logic is rejected: its semantics are unknown).
_APPEND_ONLY = (cu.DynamicLayer,)
_BOUNDED = tuple(c for c in (
    getattr(cu, "DynamicSlidingWindowLayer", None),
    cu.LinearAttentionLayer,
    getattr(cu, "LinearAttentionAndSlidingWindowAttentionLayer", None),
) if c is not None)
_HYBRID = tuple(c for c in (getattr(cu, "LinearAttentionAndFullAttentionLayer", None),) if c is not None)
_KV = ("keys", "values")
# Per-chunk bookkeeping set on cache layers by this module, not model state.
_TRANSIENT = ("_store_ctx", "_chunk_kv")


# ── attention for chunk-vs-prefix queries ──────────────────────────────────────
# A chunk's queries attend to a longer prefix (q_len < kv_len). HF's sdpa path
# then builds an explicit [q, kv] mask, and with a mask it expands grouped K/V
# heads to all query heads (repeat_kv) and saves the copies for backward: O(kv x
# query heads) per layer, which dominated SeCO's memory at long lengths. While
# SeCO runs, models configured for sdpa use this path instead: no mask for plain
# causal layers, lower-right causal bias and native GQA in SDPA's efficient
# kernels. Anything else (sliding windows, sinks, padding masks, position bias)
# goes through HF's own sdpa unchanged.

_ATTN = "palingenesis_seco_sdpa"
_REGISTERED = False
# Lower-right causal biases, created per chunk OUTSIDE the layers: the CausalBias
# tensor subclass cannot be constructed inside selective activation
# checkpointing (its dispatch mode intercepts the construction).
_BIASES: dict[tuple[int, int], object] = {}
# Full-attention layers served from a KVStore in the current forward:
# layer_idx -> (store, prefix length, block size). Filled by the layers' update().
_ACTIVE: dict[int, tuple[KVStore, int, int]] = {}
KV_BLOCK = 8192        # prefix tokens per attention block (bounds per-block memory and transfers)


def _mask(*args, **kwargs):
    from transformers.masking_utils import causal_mask_function, sdpa_mask

    plain_causal = (kwargs.get("mask_function") is causal_mask_function and kwargs.get("attention_mask") is None
                    and not kwargs.get("kv_offset") and kwargs.get("local_size") is None
                    and kwargs.get("allow_is_causal_skip", True))
    return None if plain_causal else sdpa_mask(*args, **kwargs)


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    from torch.nn.attention.bias import causal_lower_right
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    q_len, kv_len = query.shape[2], key.shape[2]
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    special = any(kwargs.get(k) is not None for k in ("position_bias", "s_aux", "sinks"))
    entry = _ACTIVE.get(getattr(module, "layer_idx", None))
    if entry is not None:
        store, prefix, block = entry
        if query.is_cuda and torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
            query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
        if attention_mask is None and causal and not special and not dropout:
            out = chunk_attention(query, key, value, store, prefix, scaling, block)
            return out.transpose(1, 2).contiguous(), None
        # Anything else: materialise the prefix (exactly the previous behaviour),
        # routing its gradient into the store's buffer.
        if prefix:
            pk, pv = store.load(0, prefix)
            if torch.is_grad_enabled() and store.grad_k is not None:
                pk, pv = pk.requires_grad_(True), pv.requires_grad_(True)
                pk.register_hook(lambda g: store.grad_k[..., :prefix, :].add_(g))
                pv.register_hook(lambda g: store.grad_v[..., :prefix, :].add_(g))
            key, value = torch.cat([pk, key], dim=-2), torch.cat([pv, value], dim=-2)
            kv_len = key.shape[2]
    if attention_mask is not None or not causal or special or q_len == 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    # Under autocast, RoPE's fp32 cos/sin promote q/k to fp32 while v stays bf16.
    # The CausalBias dispatcher picks its kernel BEFORE autocast casts anything, and
    # with fp32 inputs it rules out flash and materialises the [q, kv] scores. Cast
    # to the autocast dtype here so the flash kernel (lower-right causal + GQA) runs.
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    flash_capable = (query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)
                     and key.shape[-1] == value.shape[-1] <= 256)
    extra = {}
    if key.shape[1] != query.shape[1]:
        if flash_capable:
            extra["enable_gqa"] = True
        else:                                 # the other kernels need matching head counts
            key = repeat_kv(key, query.shape[1] // key.shape[1])
            value = repeat_kv(value, query.shape[1] // value.shape[1])
    bias = None
    if q_len != kv_len:
        bias = _BIASES.get((q_len, kv_len))
        if bias is None:
            try:
                bias = causal_lower_right(q_len, kv_len)
            except RuntimeError:              # e.g. inside selective checkpointing: explicit mask
                bias = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(kv_len - q_len)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling, is_causal=q_len == kv_len, **extra)
    return out.transpose(1, 2).contiguous(), None


@contextmanager
def _chunk_attention(model: nn.Module):
    """Route sdpa-configured attention through `_attention` for the duration."""
    global _REGISTERED
    from transformers import AttentionInterface, PretrainedConfig
    from transformers.masking_utils import AttentionMaskInterface

    if not _REGISTERED:
        AttentionInterface.register(_ATTN, _attention)
        AttentionMaskInterface.register(_ATTN, _mask)
        _REGISTERED = True
    configs = {id(m.config): m.config for m in model.modules()
               if isinstance(getattr(m, "config", None), PretrainedConfig)}
    original = {key: cfg._attn_implementation for key, cfg in configs.items()}
    try:
        for key, cfg in configs.items():
            if original[key] == "sdpa":
                cfg._attn_implementation_internal = _ATTN
        yield
    finally:
        for key, cfg in configs.items():
            cfg._attn_implementation_internal = original[key]


@dataclass
class ChunkwiseResult:
    """`loss` is the summed CE over all scored tokens divided by the loss
    denominator — the value a full forward with that denominator would give."""

    loss: float
    num_chunks: int
    backpropagated: int


def seco_forward_backward(model, input_ids, labels, *, chunk_size: int = 4096, **kwargs) -> ChunkwiseResult:
    """SeCO: exact gradients, activations of one chunk at a time."""
    return chunkwise_forward_backward(model, input_ids, labels, chunk_size=chunk_size, budget=None, **kwargs)


def spaco_forward_backward(model, input_ids, labels, *, chunk_size: int = 4096, budget: int = 8,
                           **kwargs) -> ChunkwiseResult:
    """SpaCO: backpropagate `budget` random chunks (a stochastic gradient estimate)."""
    return chunkwise_forward_backward(model, input_ids, labels, chunk_size=chunk_size, budget=budget, **kwargs)


def chunkwise_forward_backward(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> ChunkwiseResult:
    """Accumulate d(loss)/d(params) into `.grad`, chunk by chunk; the caller must
    NOT call `.backward()` afterwards. See `_chunkwise` for the arguments."""
    with _chunk_attention(model), differentiable_decode(model):
        return _chunkwise(model, input_ids, labels, **kwargs)


@contextmanager
def differentiable_decode(model: nn.Module):
    """One-token cached forwards through kernels that backpropagate, for the duration.

    transformers' linear-attention layers (Gated DeltaNet: Qwen3.5, Qwen3-Next, ...)
    switch to decode kernels when a cached forward gets a single token: fla's
    fused_recurrent (its backward raises) and causal-conv1d's update (no autograd:
    gradients would silently stop there). A chunk of one token happens in chunk-wise
    training (a sequence one token past a multiple of the chunk size, adjacent
    branch points). Route them to the layer's chunk kernel, which handles any length,
    and to the model's own torch convolution update."""
    import sys

    patched = []
    for module in model.modules():
        if "recurrent_gated_delta_rule" in module.__dict__ and "chunk_gated_delta_rule" in module.__dict__:
            patched.append((module, "recurrent_gated_delta_rule", module.recurrent_gated_delta_rule))
            module.recurrent_gated_delta_rule = module.chunk_gated_delta_rule
        torch_update = getattr(sys.modules.get(type(module).__module__), "torch_causal_conv1d_update", None)
        if "causal_conv1d_update" in module.__dict__ and torch_update is not None:
            patched.append((module, "causal_conv1d_update", module.causal_conv1d_update))
            module.causal_conv1d_update = torch_update
    try:
        yield
    finally:
        for module, name, original in patched:
            setattr(module, name, original)


def _chunkwise(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
    budget: int | None = None,
    loss_denom: float | None = None,
    num_loss_chunks=None,
    rng: random.Random | None = None,
    kv_offload: bool = False,
) -> ChunkwiseResult:
    """Accumulate d(loss)/d(params) into `.grad`, chunk by chunk; the caller must
    NOT call `.backward()` afterwards.

    input_ids:  [B, S]; rows may be right-padded.
    labels:     [B, S] aligned with input_ids (the data-pipeline convention);
                shifted here for next-token prediction.
    budget:     None = SeCO (exact); an int < number of chunks = SpaCO.
    loss_denom: normaliser of the summed CE (default: number of scored tokens);
                the trainer passes global valid tokens x grad-accum steps.
    num_loss_chunks: tokens -> number of lm_head chunks (bounds logits memory).
    kv_offload: keep full-attention K/V (and recurrent start states) in pinned CPU
                memory, streamed to the GPU block by block. Exact; costs transfers.
    """
    backbone, lm_head = _backbone(model), _lm_head(model)
    if getattr(backbone, "gradient_checkpointing", False) and backbone.training:
        raise RuntimeError(
            "SeCO needs the cache during training, which HF gradient checkpointing disables. "
            "Use palingenesis activation checkpointing (train.gradient_checkpointing) instead."
        )
    if budget is not None and budget < 1:
        raise ValueError(f"SpaCO budget must be >= 1 (got {budget})")
    shifted = shift_labels(labels)
    denom = float(loss_denom if loss_denom is not None else max((shifted != IGNORE_INDEX).sum().item(), 1))
    seq_len = input_ids.shape[1]
    bounds = [(lo, min(lo + chunk_size, seq_len)) for lo in range(0, seq_len, chunk_size)]
    k = len(bounds)
    sparse = budget is not None and budget < k
    loss_chunks = num_loss_chunks or (lambda n: 1)

    def chunk_loss(hidden, lo, hi, weight=1.0):
        # The weight goes into the denominator: chunked CE computes lm_head's
        # gradients inside this call, so scaling the returned loss would miss them.
        return chunked_cross_entropy_loss(hidden, shifted[:, lo:hi], lm_head,
                                          num_chunks=loss_chunks(hidden.shape[0] * hidden.shape[1]),
                                          global_valid_tokens=denom / weight)

    # K/V are cached in the autocast dtype: attention consumes them in that dtype
    # anyway (autocast casts its inputs), so this matches the full forward exactly
    # while halving the cache and its gradient (RoPE otherwise promotes k to fp32).
    cast = torch.get_autocast_dtype("cuda") if input_ids.is_cuda and torch.is_autocast_enabled("cuda") else None

    # Full-attention K/V go to a KVStore per layer (no prefix copies, optional CPU
    # offload) whenever attention runs through `_attention`, i.e. the model is
    # configured for sdpa. Otherwise (e.g. eager-only models) the previous path.
    use_stores = _uses_chunk_attention(backbone)
    if kv_offload and not use_stores:
        raise NotImplementedError("memory.seco_kv_offload needs a model configured for sdpa attention")
    if kv_offload:
        # Check the WHOLE offloaded size before allocating any of it: pinning tens
        # of GiB and only then failing already brings the host to its knees.
        check_host_budget(_offloaded_bytes(backbone, input_ids.shape[0], seq_len, cast))
    stores: dict[int, KVStore] = {}
    reset_host_budget()

    # ── Stage 1: no-grad forward, keep the per-chunk checkpoints ────────────
    cache = _new_cache(backbone, cast)
    append = [_is_append_only(layer) for layer in cache.layers]
    in_store = [app and use_stores and _storable(layer) for layer, app in zip(cache.layers, append)]
    starts: list[list[dict]] = []       # starts[i][layer]: layer state chunk i starts from (bounded part)
    rng_states = []                     # RNG state each chunk starts from (dropout replay)
    devices = [input_ids.device] if input_ids.device.type == "cuda" else []
    logged = 0.0
    with torch.no_grad():
        for lo, hi in bounds:
            starts.append([_offload(_state(layer, app, clone=True), kv_offload)
                           for layer, app in zip(cache.layers, append)])
            rng_states.append(_rng_state(devices))
            _attach_stores(cache, in_store, stores, seq_len, lo, write=True, offload=kv_offload)
            hidden = _run(backbone, input_ids[:, lo:hi], cache)
            if sparse:   # skipped chunks are not recomputed; report their loss from here
                logged += float(chunk_loss(hidden, lo, hi))
            del hidden
    # Append-only checkpoints: ONE leaf per layer holding the whole sequence's K/V.
    # A chunk's prefix is a view of it (no copy), and every chunk's gradient
    # accumulates into a single buffer of the same size.
    kv = [tuple(getattr(layer, a).detach().requires_grad_(True) for a in _KV) if app and not stored else None
          for layer, app, stored in zip(cache.layers, append, in_store)]
    del cache
    for store in stores.values():
        store.start_gradients()

    # ── Stage 2: reverse sweep with gradient relay ─────────────────────────
    if sparse:
        order = sorted((rng or random).sample(range(k), budget), reverse=True)
        scale = k / budget
    else:
        order, scale = list(reversed(range(k))), 1.0

    next_leaves, next_chunk = None, None        # state leaves of the chunk processed last
    for i in order:
        lo, hi = bounds[i]
        cache = _new_cache(backbone, cast)
        _attach_stores(cache, in_store, stores, seq_len, lo, write=False, offload=kv_offload)
        leaves = []
        for idx, (layer, app) in enumerate(zip(cache.layers, append)):
            state = _leafify(_onto(starts[i][idx], input_ids.device)) if i > 0 else {}
            # clones of the leaves (the gradient still reaches them): a one-token chunk updates the
            # conv state in place (the decode path), which a leaf does not allow
            _apply_state(layer, {path: (v.clone() if torch.is_tensor(v) and v.requires_grad else v)
                                 for path, v in state.items()})
            if app and i > 0 and not in_store[idx]:
                _seed_kv(layer, [t[..., :lo, :] for t in kv[idx]])
            leaves.append(state)
        before = [_snapshot(layer) for layer in cache.layers]

        with torch.random.fork_rng(devices=devices):
            _set_rng_state(rng_states[i], devices)          # same dropout masks as stage 1
            hidden = _run(backbone, input_ids[:, lo:hi], cache)
        loss = chunk_loss(hidden, lo, hi, weight=scale)   # SpaCO: J_i scaled by k/t (module doc)
        # The relay, d<out, scale * grad>/dθ for every output later chunks read, passed to
        # autograd as (root, incoming gradient) pairs next to the loss (no scalar to build).
        roots, grads = ([loss], [torch.ones_like(loss)]) if loss.requires_grad else ([], [])
        for idx, (layer, app) in enumerate(zip(cache.layers, append)):
            pairs = []
            if in_store[idx]:
                store = stores[idx]
                chunk_k, chunk_v = layer._chunk_kv
                pairs += [(chunk_k, store.grad_k[..., lo:hi, :]), (chunk_v, store.grad_v[..., lo:hi, :])]
            elif app:
                pairs += [(getattr(layer, a)[..., lo:, :], None if leaf.grad is None else leaf.grad[..., lo:hi, :])
                          for a, leaf in zip(_KV, kv[idx])]
            if next_chunk == i + 1:
                after = _state(layer, app, clone=False)
                pairs += [(after[path], leaf.grad) for path, leaf in _tensors(next_leaves[idx])]
            for out, grad in pairs:
                if grad is not None and out.requires_grad:
                    roots.append(out)
                    grads.append((grad * scale if scale != 1.0 else grad).to(out.dtype))
        # Restore the pre-forward cache: activation checkpointing re-runs layer
        # forwards during backward and must see exactly the same inputs.
        for layer, state in zip(cache.layers, before):
            _restore(layer, state)
        if roots:
            torch.autograd.backward(roots, grads)
        if not sparse:
            logged += float(loss.detach())

        next_leaves, next_chunk = leaves, i
        del cache, hidden, loss, roots, grads, before
    _ACTIVE.clear()
    return ChunkwiseResult(loss=logged, num_chunks=k, backpropagated=len(order))


# ── cache plumbing ─────────────────────────────────────────────────────────────


def _backbone(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "base_model", None)
    if backbone is None or backbone is model:
        raise RuntimeError("SeCO needs a Hugging Face causal LM (a backbone at model.base_model and an lm_head).")
    return backbone


def _lm_head(model: nn.Module) -> nn.Module:
    head = output_head(model)            # lm_head + the model's logit transform, if any
    if head is None:
        raise RuntimeError("SeCO needs the model's output projection (get_output_embeddings()).")
    return head


def _rng_state(devices):
    return torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]


def _set_rng_state(state, devices) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    for d, s in zip(devices, cuda):
        torch.cuda.set_rng_state(s, d)


@torch.no_grad()
def verify_chunked_forward(model: nn.Module, chunk_size: int = 64, num_chunks: int = 3,
                           tolerance: float | None = None) -> float:
    """Check the one assumption SeCO rests on: running the model chunk by chunk
    with a cache gives the same logits as one full forward (through the same
    attention path SeCO uses). Returns the relative max difference; raises if it
    exceeds `tolerance` (default: 1e-4 for fp32/fp64 weights, 2e-2 otherwise).
    Random tokens, eval mode; the model's mode is restored."""
    backbone, head = _backbone(model), _lm_head(model)
    verify_output_head(model, head, backbone)
    dtype = next(model.parameters()).dtype
    if tolerance is None:
        tolerance = 1e-4 if dtype in (torch.float32, torch.float64) else 2e-2
        if tolerance < 1e-3 and _uses_fla_kernels(model):
            # flash-linear-attention's Triton kernels do not reproduce fp32 exactly
            # (their chunked and full forwards differ by ~6e-4 on Qwen3.5-0.8B); SeCO's
            # gradients then match full backprop to cosine 0.99998 instead of bit-exactly.
            tolerance = 2e-3
    device = next(model.parameters()).device
    vocab = head.weight.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(0, vocab, (1, chunk_size * num_chunks), generator=g).to(device)
    was_training = model.training
    model.eval()
    # Check the implementation, not the numerics: TF32 (float32_matmul_precision
    # "high", the trainer default) alone makes chunked and full forwards differ by
    # ~1e-3 in fp32, which would hide nothing and reject every model.
    matmul_precision, cudnn_tf32 = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        with _chunk_attention(model):
            full = head(_run(backbone, ids, None)).float()
            cache = _new_cache(backbone)
            in_store = [_is_append_only(layer) and _uses_chunk_attention(backbone) and _storable(layer)
                        for layer in cache.layers]
            stores: dict[int, KVStore] = {}
            try:
                outs = []
                for lo in range(0, ids.shape[1], chunk_size):
                    _attach_stores(cache, in_store, stores, ids.shape[1], lo, write=True, offload=False)
                    outs.append(head(_run(backbone, ids[:, lo:lo + chunk_size], cache)).float())
                chunked = torch.cat(outs, dim=1)
            except NotImplementedError:
                raise
            except Exception as exc:
                raise NotImplementedError(
                    f"SeCO: this model cannot run chunk by chunk with a cache ({type(exc).__name__}: {exc})"
                ) from exc
    finally:
        _ACTIVE.clear()
        model.train(was_training)
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    diff = float((chunked - full).abs().max() / full.abs().max().clamp(min=1e-30))
    if diff > tolerance:
        raise NotImplementedError(
            f"SeCO: this model's cached, chunked forward differs from its full forward (relative difference "
            f"{diff:.1e} > {tolerance:.0e}), so chunk-wise training would not compute its gradient. This is a "
            f"limitation of the model implementation in transformers {__import__('transformers').__version__} "
            "(measured for e.g. Bamba, Jamba, Mamba, RecurrentGemma)."
        )
    return diff


def _uses_fla_kernels(model: nn.Module) -> bool:
    """Linear-attention layers running flash-linear-attention's Triton kernels."""
    from palingenesis.packing import has_linear_attention

    if not has_linear_attention(model):
        return False
    from transformers.utils.import_utils import is_flash_linear_attention_available

    return is_flash_linear_attention_available()


def _run(backbone: nn.Module, input_ids: torch.Tensor, cache, position_ids: torch.Tensor | None = None) -> torch.Tensor:
    # attention_mask=None: rows are right-padded at most, so padding only follows
    # real tokens and causal attention keeps it from influencing them.
    from torch.nn.attention.bias import causal_lower_right

    q_len = input_ids.shape[1]
    prefix = _prefix_length(cache)
    _BIASES.clear()
    _ACTIVE.clear()
    if prefix:
        _BIASES[(q_len, prefix + q_len)] = causal_lower_right(q_len, prefix + q_len)
    extra = {"position_ids": position_ids} if position_ids is not None else {}
    out = backbone(input_ids=input_ids, past_key_values=cache, use_cache=cache is not None, **extra)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _prefix_length(cache) -> int:
    """Tokens already in the full-attention layers of the cache (0 if none)."""
    if cache is None:
        return 0
    return max((layer.keys.shape[-2] for layer in cache.layers
                if _is_append_only(layer) and "_store_ctx" not in layer.__dict__
                and getattr(layer, "is_initialized", False) and layer.keys.numel()), default=0)


class _StoreCtx:
    """Where a full-attention cache layer's K/V live during one chunk."""

    def __init__(self, stores, idx, seq_len, lo, write, offload):
        self.stores, self.idx, self.seq_len, self.lo, self.write, self.offload = stores, idx, seq_len, lo, write, offload


def _offloaded_bytes(backbone: nn.Module, batch: int, seq_len: int, cast: torch.dtype | None) -> int:
    """Host memory the K/V stores will need: every full-attention layer's K and V
    for the whole sequence."""
    config = backbone.config.get_text_config() if hasattr(backbone.config, "get_text_config") else backbone.config
    layers = getattr(config, "layer_types", None)
    count = sum(1 for t in layers if t in ("full_attention", "attention")) if layers else config.num_hidden_layers
    heads = getattr(config, "num_attention_heads", 1)
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // max(heads, 1)
    dtype = cast or next(backbone.parameters()).dtype
    return 2 * count * batch * kv_heads * head_dim * seq_len * (torch.finfo(dtype).bits // 8)


def _uses_chunk_attention(backbone: nn.Module) -> bool:
    return getattr(backbone.config, "_attn_implementation", None) == _ATTN


def _storable(layer) -> bool:
    """Plain full-attention layers (hybrid layers also carry linear state and keep
    the previous path)."""
    return not isinstance(layer, cu.LinearAttentionCacheLayerMixin)


def _attach_stores(cache, in_store, stores, seq_len, lo, write, offload) -> None:
    for idx, layer in enumerate(cache.layers):
        if in_store[idx]:
            layer._store_ctx = _StoreCtx(stores, idx, seq_len, lo, write, offload)
            layer.__dict__.pop("_chunk_kv", None)    # the previous chunk's (stage 1 reuses the cache)


def _offload(state: dict, offload: bool) -> dict:
    """Move a snapshotted state to pinned CPU memory (stage 1, with kv_offload)."""
    if not offload:
        return state
    out = {}
    for path, v in state.items():
        if torch.is_tensor(v) and v.is_cuda:
            host = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
            host.copy_(v, non_blocking=True)
            v = host
        out[path] = v
    return out


def _onto(state: dict, device: torch.device) -> dict:
    return {path: (v.to(device, non_blocking=True) if torch.is_tensor(v) and v.device != device else v)
            for path, v in state.items()}


def _new_cache(backbone: nn.Module, cast: torch.dtype | None = None):
    """A DynamicCache for the model whose layers update functionally (see
    `_functional`); K/V cast to `cast` when given."""
    cache = cu.DynamicCache(config=backbone.config)
    if not cache.layers:
        raise NotImplementedError("SeCO needs a model whose config describes its layers (DynamicCache(config=...)).")
    for idx, layer in enumerate(cache.layers):
        cls = type(layer)
        if cls not in _APPEND_ONLY + _BOUNDED + _HYBRID and cls.__name__ not in _FUNCTIONAL_NAMES:
            raise NotImplementedError(
                f"SeCO does not support the {cls.__name__} cache layer (layer {idx}); supported: full, "
                "sliding-window/chunked and linear attention, short convolutions, and their hybrids."
            )
        cache.layers[idx] = _functional(layer, cast)
    return cache


def _is_append_only(layer) -> bool:
    return isinstance(layer, _APPEND_ONLY) and not getattr(layer, "is_sliding", False)


def _state(layer, append_only: bool, clone: bool) -> dict:
    """The layer's bounded state: every attribute except an append-only K/V, with
    dicts flattened to (name, key) paths. Tensors cloned in stage 1 (the stock
    layers mutate some of them in place)."""
    out = {}
    for name, value in layer.__dict__.items():
        if (append_only and name in _KV) or name in _TRANSIENT:
            continue
        if isinstance(value, dict):
            for key, item in value.items():
                out[(name, key)] = item.clone() if clone and torch.is_tensor(item) else item
        else:
            out[(name,)] = value.clone() if clone and torch.is_tensor(value) else value
    return out


def _leafify(state: dict) -> dict:
    return {path: (v.detach().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v)
            for path, v in state.items()}


def _tensors(state: dict):
    return [(path, v) for path, v in state.items() if torch.is_tensor(v) and v.requires_grad]


def _apply_state(layer, state: dict) -> None:
    for path, value in state.items():
        if len(path) == 1:
            setattr(layer, path[0], value)
        else:
            getattr(layer, path[0])[path[1]] = value


def _seed_kv(layer, kv) -> None:
    layer.keys, layer.values = kv
    layer.dtype, layer.device = kv[0].dtype, kv[0].device
    layer.is_initialized = True


def _snapshot(layer) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in layer.__dict__.items()}


def _restore(layer, snapshot: dict) -> None:
    layer.__dict__.clear()
    layer.__dict__.update(_snapshot_copy(snapshot))


def _snapshot_copy(snapshot: dict) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in snapshot.items()}


# ── functional linear-attention updates ────────────────────────────────────────
# The stock linear-attention layers copy new states into static buffers in
# place. That breaks autograd once a state carries a graph (or was saved for
# backward by the op that read it), so stage 2 swaps in a subclass that
# reassigns instead, with otherwise identical semantics.


def _update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None,
                       **kwargs) -> torch.Tensor:
    self.dtype, self.device = conv_states.dtype, conv_states.device
    if not _DICT_STATES:
        # Called with the conv window already padded/cropped to the kernel size
        # (cached context prepended): it IS the new state.
        kernel = conv_states.shape[-1] if self.conv_states is None else self.conv_states.shape[-1]
        self.conv_states = conv_states[..., -kernel:]
        self.is_conv_states_initialized = self.has_previous_state = True
        return self.conv_states
    # Called with the new tokens only; returns the window the causal conv needs
    # (cached context + new tokens) and keeps the last `kernel` columns.
    s = state_idx
    kernel = self.conv_kernel_size[s] or conv_kernel_size or conv_states.shape[-1]
    if self.has_previous_state[s]:
        full = torch.cat([self.conv_states[s], conv_states], dim=-1)
    else:
        full = F.pad(conv_states, (max(kernel - conv_states.shape[-1], 0), 0))
        self.has_previous_state[s] = True
    self.conv_kernel_size[s] = kernel
    self.conv_states[s] = full[..., -kernel:]
    self.is_conv_states_initialized[s] = True
    return full


def _update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> torch.Tensor:
    if not _DICT_STATES:
        self.recurrent_states = recurrent_states
        self.is_recurrent_states_initialized = True
    else:
        self.recurrent_states[state_idx] = recurrent_states
        self.is_recurrent_states_initialized[state_idx] = True
    return recurrent_states


_FUNCTIONAL: dict[type, type] = {}
_FUNCTIONAL_NAMES: set[str] = set()


def _functional(layer, cast: torch.dtype | None = None):
    """The same layer, with functional linear-attention updates and K/V cast to
    `cast` on the way into the cache."""
    cls = type(layer)
    if cls.__name__ in _FUNCTIONAL_NAMES:
        layer._cast = cast
        return layer
    if cls not in _FUNCTIONAL:
        methods = {}
        if isinstance(layer, cu.LinearAttentionCacheLayerMixin):
            methods.update(update_conv_state=_update_conv_state, update_recurrent_state=_update_recurrent_state)
        if isinstance(layer, cu.DynamicLayer):
            base_update = cls.update

            base_seq_length = cls.get_seq_length

            def update(self, key_states, value_states, *args, **kwargs):
                dtype = self.__dict__.get("_cast")
                if dtype is not None:
                    key_states, value_states = key_states.to(dtype), value_states.to(dtype)
                ctx = self.__dict__.get("_store_ctx")
                if ctx is None:
                    return base_update(self, key_states, value_states, *args, **kwargs)
                # Store layer: the prefix stays in the KVStore; attention (see
                # `_attention`) reads it block by block. Return the chunk only.
                store = ctx.stores.get(ctx.idx)
                if store is None:
                    store = ctx.stores[ctx.idx] = KVStore(key_states, ctx.seq_len, ctx.offload)
                if ctx.write:
                    store.write(ctx.lo, key_states, value_states)
                self._chunk_kv = (key_states, value_states)
                _ACTIVE[ctx.idx] = (store, ctx.lo, KV_BLOCK)
                return key_states, value_states

            def get_seq_length(self, *args, **kwargs):
                ctx = self.__dict__.get("_store_ctx")
                if ctx is None:
                    return base_seq_length(self, *args, **kwargs)
                chunk = self.__dict__.get("_chunk_kv")
                # rows continuing the store from different positions (seco_tree) pass their own
                # position ids; the cache then reports no shared past
                lo = ctx.lo if isinstance(ctx.lo, int) else 0
                return lo + (chunk[0].shape[-2] if chunk is not None else 0)

            methods["update"] = update
            methods["get_seq_length"] = get_seq_length
        sub = type(f"Functional{cls.__name__}", (cls,), methods)
        _FUNCTIONAL[cls] = sub
        _FUNCTIONAL_NAMES.add(sub.__name__)
    new = object.__new__(_FUNCTIONAL[cls])
    new.__dict__.update(_snapshot(layer))
    new._cast = cast
    return new
