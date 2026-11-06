"""Context parallelism: one sequence split over the ranks of a group, attention exact across them.

Each rank of a context-parallel (CP) group of size N holds 1/N of every sequence and runs the
model on it: embeddings, MLPs, norms and the loss are per token, so they need nothing from the
other ranks. Attention is the one layer that does. Every attention layer here gathers the K/V
of the whole sequence from the group (one all-gather per layer; grouped-query attention keeps
K/V small) and attends its own queries to it, causally, with their global positions; the
backward sends each K/V gradient back to the rank that owns those tokens (a reduce-scatter).
This is the all-gather form of context parallelism (as in the Llama 3 training recipe) and is
exact: the loss and every gradient equal those of the unsplit sequence (tests check it on the
CPU with several processes).

Layout. Under causal attention a token's cost grows with its position, so contiguous slices
would leave the last rank with most of the work. The sequence is cut into 2N chunks and rank r
holds chunks r and 2N-1-r ("head-tail" load balancing): every rank does the same amount of
attention. Sequences are right-padded to a multiple of 2N first (padding follows the real
tokens, cannot reach them under causal attention, and carries no label). Labels are shifted on
the whole sequence before it is split, so the token that predicts across a chunk boundary is
scored on whichever rank holds it, and position ids are the tokens' global positions.

What is supported: the model's softmax attention layers routed through transformers' attention
interface, full causal and sliding-window (Gemma 2/3, Qwen3 / Mistral with sliding windows).
What is refused with a clear error, before training: recurrent and linear-attention layers
(Qwen3.5 / Qwen3-Next Gated DeltaNet, Mamba, short convolutions: their state flows along the
sequence, from one rank's chunk into the next), chunked attention (Llama 4), attention logit
soft-capping and attention sinks. A runtime check backs this up: under context parallelism every
decoder layer must have gone through this attention in each forward, or the step raises.
"""

from __future__ import annotations

import logging
import sys

import torch
import torch.distributed as dist
from torch import nn

from palingenesis.config import ConfigError
from palingenesis.seco_attention import _block_backward, _block_forward, _flash_ok, _merge

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Per-layer types (config.layer_types) this module computes exactly.
SUPPORTED_LAYER_TYPES = frozenset({"full_attention", "sliding_attention", "attention"})
# Submodule names of recurrent / linear-attention / convolutional sequence mixers, for models
# without `layer_types` in their config.
_RECURRENT_NAMES = ("linear_attn", "mamba", "ssm", "delta_net", "deltanet", "recurrent", "short_conv")
# Attention keyword arguments this module does not implement: a non-None value raises.
_UNSUPPORTED_KWARGS = ("softcap", "s_aux", "sinks", "position_bias", "alibi_slopes")


class ContextParallelError(RuntimeError):
    """A forward under context parallelism met something it cannot compute exactly."""


# ── layout: which tokens a rank holds ──────────────────────────────────────────


def padded_length(seq_len: int, cp_size: int) -> int:
    """`seq_len` rounded up to a multiple of 2 x cp_size (the head-tail layout's chunk count)."""
    multiple = 2 * cp_size
    return -(-seq_len // multiple) * multiple


def load_balanced_indices(seq_len: int, cp_size: int, cp_rank: int, device=None) -> torch.Tensor:
    """Global positions of the tokens rank `cp_rank` holds, in its local order.

    The (padded) sequence is cut into 2 x cp_size chunks; rank r holds chunk r, then chunk
    2 x cp_size - 1 - r. Over all ranks the indices are a permutation of range(seq_len)."""
    if seq_len % (2 * cp_size):
        raise ValueError(f"sequence length {seq_len} is not a multiple of 2 x cp_size = {2 * cp_size}")
    chunk = seq_len // (2 * cp_size)
    head = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    tail_start = (2 * cp_size - 1 - cp_rank) * chunk
    tail = torch.arange(tail_start, tail_start + chunk, device=device)
    return torch.cat([head, tail])


def to_global_order(parts: torch.Tensor) -> torch.Tensor:
    """[N, B, H, L, D] per-rank blocks (rank order, head-tail layout) -> [B, H, N*L, D] in sequence order."""
    n, b, h, length, d = parts.shape
    halves = parts.reshape(n, b, h, 2, length // 2, d)
    chunks = torch.cat([halves[:, :, :, 0], halves[:, :, :, 1].flip(0)], dim=0)  # [2N, B, H, C, D]
    return chunks.permute(1, 2, 0, 3, 4).reshape(b, h, n * length, d)


def to_rank_order(full: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Inverse of `to_global_order`: [B, H, T, D] in sequence order -> [N, B, H, T/N, D] per rank."""
    b, h, t, d = full.shape
    chunks = full.reshape(b, h, 2 * cp_size, t // (2 * cp_size), d).permute(2, 0, 1, 3, 4)  # [2N, B, H, C, D]
    halves = torch.stack([chunks[:cp_size], chunks[cp_size:].flip(0)], dim=3)  # [N, B, H, 2, C, D]
    return halves.reshape(cp_size, b, h, t // cp_size, d)


# ── collectives (autograd-aware) ───────────────────────────────────────────────


def _all_gather(x: torch.Tensor, group, size: int) -> torch.Tensor:
    """[N, *x.shape]: every rank's `x`, in rank order."""
    x = x.contiguous()
    if size == 1:
        return x.unsqueeze(0)
    if dist.get_backend(group) == "nccl":
        out = x.new_empty((size, *x.shape))
        dist.all_gather_into_tensor(out, x, group=group)
        return out
    parts = [torch.empty_like(x) for _ in range(size)]
    dist.all_gather(parts, x, group=group)
    return torch.stack(parts)


def _reduce_scatter(parts: torch.Tensor, group, rank: int) -> torch.Tensor:
    """parts [N, ...] on every rank -> the sum over ranks of parts[rank]."""
    parts = parts.contiguous()
    if parts.shape[0] == 1:
        return parts[0]
    if dist.get_backend(group) == "nccl":
        out = parts.new_empty(parts.shape[1:])
        dist.reduce_scatter_tensor(out, parts, group=group)
        return out
    dist.all_reduce(parts, group=group)  # gloo has no reduce-scatter
    return parts[rank].clone()


class _GatherSequence(torch.autograd.Function):
    """[B, H, L, D] local K or V (head-tail layout) -> [B, H, N*L, D], the whole sequence in order.

    Backward: the gradient of the whole sequence's K/V is summed over the group (each rank's
    queries contribute to every key) and each rank keeps its own tokens'. Reduced in fp32."""

    @staticmethod
    def forward(ctx, x, cp: ContextParallel):
        ctx.cp = cp
        return cp.gather(x)

    @staticmethod
    def backward(ctx, grad):
        return ctx.cp.scatter_gradient(grad.float()).to(grad.dtype), None


# Full causal layers use _GatheredCausalAttention (False: SDPA under autograd for every layer; tests compare both).
FUSED = True


class _GatheredCausalAttention(torch.autograd.Function):
    """Causal attention of this rank's two query chunks over the whole sequence's K/V.

    q [B, H, L, D], k / v [B, Hkv, L, D]: this rank's tokens (head-tail layout). Per query chunk
    the keys are the chunks before it (full attention) and the chunk itself (causal): two kernel
    calls merged through their log-sum-exp, the way flash and ring attention combine blocks. The
    backward runs the kernels' backward from the merged output and log-sum-exp, so it is exact.

    Memory: the gathered K/V live only inside forward and backward (the backward gathers them
    again); the saved tensors are this rank's q, k, v, output and log-sum-exp. Saving the gathered
    sequence instead would keep 2 x T x Hkv x D values per layer until the backward (4 GiB per
    layer for 8 K/V heads of 128 at a million tokens in bf16). The cost is one more all-gather per
    layer in the backward, which the reduce-scatter of the K/V gradients needs anyway.

    Kernels (palingenesis.seco_attention): flash on CUDA for bf16 / fp16 (grouped K/V heads
    natively), an exact math path otherwise (fp32 and fp64, the CPU)."""

    @staticmethod
    def forward(ctx, q, k, v, cp: ContextParallel, scale: float):
        chunk = q.shape[2] // 2
        keys, values = cp.gather(k), cp.gather(v)
        outs, lses, auxes = [], [], []
        for half, (_, hi) in enumerate(cp._key_ranges(chunk, None)):
            qh = q[:, :, half * chunk : (half + 1) * chunk]
            out, lse, aux = _block_forward(qh, keys[:, :, hi - chunk : hi], values[:, :, hi - chunk : hi], True, scale)
            prefix_aux = None
            if hi > chunk:
                o, l_, prefix_aux = _block_forward(
                    qh, keys[:, :, : hi - chunk], values[:, :, : hi - chunk], False, scale
                )
                out, lse = _merge(*_merge(None, None, out, lse), o, l_)
            outs.append(out.to(q.dtype))
            lses.append(lse.float())
            auxes.append((aux, prefix_aux))
        out = torch.cat(outs, dim=2)
        ctx.save_for_backward(q, k, v, out, torch.cat(lses, dim=2))
        ctx.cp, ctx.scale, ctx.auxes = cp, scale, auxes
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        cp, scale = ctx.cp, ctx.scale
        chunk = q.shape[2] // 2
        dout = dout.to(q.dtype).contiguous()
        keys, values = cp.gather(k), cp.gather(v)
        dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        dkeys = torch.zeros(keys.shape, dtype=torch.float32, device=k.device)
        dvalues = torch.zeros(values.shape, dtype=torch.float32, device=v.device)
        for half, ((_, hi), (aux, prefix_aux)) in enumerate(zip(cp._key_ranges(chunk, None), ctx.auxes)):
            rows = slice(half * chunk, (half + 1) * chunk)
            args = (dout[:, :, rows], q[:, :, rows].contiguous())
            merged = (out[:, :, rows].contiguous(), lse[:, :, rows].contiguous())
            diag = slice(hi - chunk, hi)
            dqh, dkh, dvh = _block_backward(
                args[0].contiguous(), args[1], keys[:, :, diag], values[:, :, diag], *merged, True, scale, aux
            )
            dq[:, :, rows] = dqh
            dkeys[:, :, diag] += dkh
            dvalues[:, :, diag] += dvh
            if prefix_aux is not None:
                prefix = slice(0, hi - chunk)
                dqh, dkh, dvh = _block_backward(
                    args[0].contiguous(),
                    args[1],
                    keys[:, :, prefix],
                    values[:, :, prefix],
                    *merged,
                    False,
                    scale,
                    prefix_aux,
                )
                dq[:, :, rows] += dqh
                dkeys[:, :, prefix] += dkh
                dvalues[:, :, prefix] += dvh
        dk = cp.scatter_gradient(dkeys).to(k.dtype)
        dv = cp.scatter_gradient(dvalues).to(v.dtype)
        return dq.to(q.dtype), dk, dv, None, None


# ── the context-parallel group and its attention ───────────────────────────────


class ContextParallel:
    """One context-parallel group: the ranks that split a sequence, and this rank's place in it."""

    def __init__(self, group, size: int, rank: int):
        self.group, self.size, self.rank = group, size, rank
        self._indices: dict[tuple[int, str], torch.Tensor] = {}
        self._masks: dict[tuple, torch.Tensor] = {}

    @classmethod
    def from_mesh(cls, cp_mesh) -> ContextParallel:
        return cls(cp_mesh.get_group(), cp_mesh.size(), cp_mesh.get_local_rank())

    # -- inputs

    def padded_length(self, seq_len: int) -> int:
        return padded_length(seq_len, self.size)

    def indices(self, seq_len: int, device) -> torch.Tensor:
        """Global positions of this rank's tokens for a batch of length `seq_len` (padded first)."""
        total = self.padded_length(seq_len)
        key = (total, str(device))
        if key not in self._indices:
            self._indices[key] = load_balanced_indices(total, self.size, self.rank, device)
        return self._indices[key]

    def shard(self, t: torch.Tensor, pad_value=0) -> torch.Tensor:
        """This rank's tokens of a per-token batch field [B, T, ...]: right-padded with `pad_value`
        to a multiple of 2 x cp_size, then the head-tail chunks of this rank."""
        seq_len = t.shape[1]
        total = self.padded_length(seq_len)
        if total != seq_len:
            pad = t.new_full((t.shape[0], total - seq_len, *t.shape[2:]), pad_value)
            t = torch.cat([t, pad], dim=1)
        return t.index_select(1, self.indices(seq_len, t.device)).contiguous()

    def position_ids(self, seq_len: int, batch_size: int, device) -> torch.Tensor:
        """[B, T_local] global positions of this rank's tokens (rotary embeddings need them)."""
        return self.indices(seq_len, device).unsqueeze(0).expand(batch_size, -1)

    # -- communication

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, D] this rank's tokens -> [B, H, N*L, D], the group's in sequence order."""
        return to_global_order(_all_gather(x, self.group, self.size))

    def scatter_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Inverse of `gather` for gradients: [B, H, T, D] from every rank, summed over the group, ->
        this rank's [B, H, T/N, D]."""
        return _reduce_scatter(to_rank_order(grad, self.size), self.group, self.rank)

    # -- attention

    def attention(self, module, query, key, value, scaling, dropout: float, window: int | None) -> torch.Tensor:
        """Causal attention of this rank's queries [B, H, L, D] over the whole sequence's K/V.

        Returns [B, L, H, D] (the layout transformers' attention functions return)."""
        from palingenesis import seco

        local = query.shape[2]
        if local % 2:
            raise ContextParallelError(
                f"context parallel: a local sequence of {local} tokens is not two equal chunks (inputs must be "
                "split with ContextParallel.shard)."
            )
        chunk = local // 2
        if query.is_cuda and torch.is_autocast_enabled("cuda"):
            # one dtype before the gather: RoPE's fp32 cos/sin promote q/k to fp32 under autocast, which would
            # double the communication and rule out the fused kernels
            dtype = torch.get_autocast_dtype("cuda")
            query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
        if window is None and not dropout and (_flash_ok(query, key) or not query.is_cuda) and FUSED:
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            out = _GatheredCausalAttention.apply(query, key, value, self, scale)
            return out.transpose(1, 2).contiguous()
        # Sliding windows, dropout, kernels flash does not cover: SDPA under autograd, on the keys each chunk
        # needs. The slices are copied when they are a window's worth, so the backward keeps only those (and
        # not the gathered sequence) alive.
        keys = _GatherSequence.apply(key, self)
        values = _GatherSequence.apply(value, self)
        outs = []
        for half, (lo, hi) in enumerate(self._key_ranges(chunk, window)):
            q = query[:, :, half * chunk : (half + 1) * chunk]
            k, v = keys[:, :, lo:hi], values[:, :, lo:hi]
            if window is not None:
                k, v = k.clone(), v.clone()
            mask = None
            if window is not None and hi - 1 - lo >= window:  # some query/key pair falls outside the window
                mask = self._window_mask(chunk, lo, hi, window, query.device)
            # seco._attention: lower-right causal (queries are the last `chunk` of the keys) on SDPA's fused
            # kernels, grouped K/V heads natively; an explicit mask goes to transformers' sdpa
            out, _ = seco._attention(module, q, k, v, mask, dropout=dropout, scaling=scaling)
            outs.append(out)
        return torch.cat(outs, dim=1)

    def _key_ranges(self, chunk: int, window: int | None) -> list[tuple[int, int]]:
        """Per local chunk (head, tail): the global key range [lo, hi) its queries attend to. The chunk's
        queries are the last `chunk` positions of the range (lower-right causal alignment)."""
        ranges = []
        for block in (self.rank, 2 * self.size - 1 - self.rank):
            hi = (block + 1) * chunk
            lo = 0 if window is None else max(0, hi - chunk - window + 1)
            ranges.append((lo, hi))
        return ranges

    def _window_mask(self, chunk: int, lo: int, hi: int, window: int, device) -> torch.Tensor:
        """[1, 1, chunk, hi - lo] boolean: query i (position hi - chunk + i) sees key j (position lo + j)
        iff 0 <= distance < window (transformers' sliding-window convention)."""
        key = (chunk, lo, hi, window, str(device))
        mask = self._masks.get(key)
        if mask is None:
            q_pos = torch.arange(hi - chunk, hi, device=device)
            k_pos = torch.arange(lo, hi, device=device)
            dist_ = q_pos[:, None] - k_pos[None, :]
            mask = ((dist_ >= 0) & (dist_ < window))[None, None]
            if len(self._masks) > 8:
                self._masks.clear()
            self._masks[key] = mask
        return mask

    def prepare(self, seq_len: int, device) -> None:
        """Before a forward on a batch of length `seq_len`: build the lower-right causal biases the
        attention will use, outside the layers (selective activation checkpointing cannot construct
        them inside)."""
        from torch.nn.attention.bias import causal_lower_right

        from palingenesis import seco

        chunk = self.padded_length(seq_len) // (2 * self.size)
        for block in (self.rank, 2 * self.size - 1 - self.rank):
            hi = (block + 1) * chunk
            if hi != chunk and (chunk, hi) not in seco._BIASES:
                seco._BIASES[(chunk, hi)] = causal_lower_right(chunk, hi)

    # -- scope

    def activate(self, seq_len: int | None = None, device=None) -> None:
        """Route the rerouted models' attention through this group until `deactivate` (forward AND
        backward: activation checkpointing recomputes attention during the backward)."""
        global _ACTIVE
        if seq_len is not None:
            self.prepare(seq_len, device)
        _ACTIVE = self

    def deactivate(self) -> None:
        global _ACTIVE
        _ACTIVE = None

    def all_reduce_sum(self, values: list[float], device) -> list[float]:
        t = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(t, group=self.group)
        return t.tolist()


# Set by ContextParallel.activate. A plain global, not a context variable: the autograd engine runs
# the backward (and with it activation checkpointing's recompute) on its own threads.
_ACTIVE: ContextParallel | None = None
# Attention modules that ran under context parallelism in the current backbone forward.
_SEEN: set[int] = set()


# ── routing a model's attention ────────────────────────────────────────────────

# original attention implementation -> the name registered for it here
_NAMES: dict[str, str] = {}


def _fallback(original: str, module):
    """The attention function `original` names, for forwards outside context parallelism."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if original != "eager" and original in ALL_ATTENTION_FUNCTIONS:
        return ALL_ATTENTION_FUNCTIONS[original]
    eager = getattr(sys.modules.get(type(module).__module__), "eager_attention_forward", None)
    if eager is None:
        raise ContextParallelError(f"no attention function {original!r} for {type(module).__name__}")
    return eager


def _make_attention(original: str):
    @torch.compiler.disable  # collectives and a Python-side check: run eagerly between compiled regions
    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        cp = _ACTIVE
        if cp is None:
            return _fallback(original, module)(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                **kwargs,
            )
        name = type(module).__name__
        if attention_mask is not None:
            raise ContextParallelError(
                f"context parallel: {name} received an attention mask. Only right-padded, unpacked rows are "
                "supported (causal attention keeps the padding inert); left padding, packing and custom masks "
                "are not."
            )
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        if not causal:
            raise ContextParallelError(
                f"context parallel: {name} is bidirectional; only causal attention is supported."
            )
        for arg in _UNSUPPORTED_KWARGS:
            if kwargs.get(arg) is not None:
                raise ContextParallelError(f"context parallel: {name} uses `{arg}`, which is not supported.")
        window = kwargs["sliding_window"] if "sliding_window" in kwargs else getattr(module, "sliding_window", None)
        _SEEN.add(id(module))
        return cp.attention(module, query, key, value, scaling, dropout, window), None

    return attention


def _mask_function(original: str):
    def mask(*args, **kwargs):
        if _ACTIVE is not None:
            return None  # the attention computes causality and windows from global positions
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

        fn = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.get(original, eager_mask)
        return fn(*args, **kwargs)

    return mask


def _registered_name(original: str) -> str:
    """The attention / mask implementation name that wraps `original` (registered once)."""
    if original not in _NAMES:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface

        # no "flash" / "sdpa" in the name: transformers branches on those substrings
        name = f"palingenesis_context_parallel_{len(_NAMES)}"
        AttentionInterface.register(name, _make_attention(original))
        AttentionMaskInterface.register(name, _mask_function(original))
        _NAMES[original] = name
    return _NAMES[original]


def check_model_support(model: nn.Module, label: str = "the model") -> None:
    """Raise ConfigError if `model` has layers context parallelism cannot compute exactly."""
    from transformers import PretrainedConfig

    configs = [model.config] if isinstance(getattr(model, "config", None), PretrainedConfig) else []
    configs += [m.config for m in model.modules() if isinstance(getattr(m, "config", None), PretrainedConfig)]
    seen = set()
    for cfg in configs:
        if id(cfg) in seen:
            continue
        seen.add(id(cfg))
        text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        types = set(getattr(text, "layer_types", None) or [])
        other = sorted(types - SUPPORTED_LAYER_TYPES)
        if other:
            raise ConfigError(
                f"parallel.context_parallel: {label} has {other} layers. Context parallelism splits each sequence "
                "over the ranks and computes softmax attention exactly across them; a recurrent or linear-attention "
                "layer (Gated DeltaNet, Mamba, short convolutions) carries a state along the sequence from one "
                "rank's chunk into the next, which is not implemented. Use memory.seco for long sequences on one "
                "GPU, or FSDP without context parallelism."
            )
        if getattr(text, "attention_chunk_size", None):
            raise ConfigError(f"parallel.context_parallel: {label} uses chunked attention, which is not supported.")
        if getattr(text, "attn_logit_softcapping", None) is not None:
            raise ConfigError(
                f"parallel.context_parallel: {label} uses attention logit soft-capping "
                f"(attn_logit_softcapping={text.attn_logit_softcapping}), which the fused attention kernels do not "
                "compute. Not supported."
            )
    recurrent = sorted({n for n, _ in model.named_modules() if n.rsplit(".", 1)[-1].lower() in _RECURRENT_NAMES})
    if recurrent:
        raise ConfigError(
            f"parallel.context_parallel: {label} has recurrent / convolutional sequence mixers ({recurrent[0]}, ...), "
            "whose state would have to pass from one rank's chunk to the next. Not supported."
        )


def route_attention(model: nn.Module, layers: list[nn.Module]) -> int:
    """Send the attention of `model`'s decoder `layers` through context parallelism (when active;
    their original attention otherwise), and check after every forward under it that each layer
    did. Returns the number of attention configs rerouted."""
    from transformers import PretrainedConfig

    from palingenesis.logits import backbone_of

    configs = {}
    for layer in layers:
        for m in layer.modules():
            cfg = getattr(m, "config", None)
            if isinstance(cfg, PretrainedConfig):
                configs[id(cfg)] = cfg
    for cfg in configs.values():
        original = cfg._attn_implementation or "eager"
        if not original.startswith("palingenesis_context_parallel"):
            cfg._attn_implementation_internal = _registered_name(original)
    if not configs:
        raise ConfigError("parallel.context_parallel: cannot find the attention configuration of the decoder layers.")

    layer_modules = [{id(m) for m in layer.modules()} for layer in layers]
    backbone = backbone_of(model)

    def start(module, args, kwargs=None):
        _SEEN.clear()

    def check(module, args, output):
        if _ACTIVE is None:
            return
        missing = [i for i, ids in enumerate(layer_modules) if not ids & _SEEN]
        _SEEN.clear()
        if missing:
            raise ContextParallelError(
                f"context parallel: decoder layer(s) {missing} did not run their attention through context "
                "parallelism (a layer type that does not use transformers' attention interface). Their output "
                "would be computed on this rank's slice of the sequence only. Not supported."
            )

    backbone.register_forward_pre_hook(start)
    backbone.register_forward_hook(check)
    return len(configs)
