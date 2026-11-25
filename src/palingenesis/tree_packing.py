"""Padding-free packing of many sequences into one forward row: end to end (varlen) or as a prefix tree.

A micro-batch of right-padded rows spends compute on padding, and every rollout of a group computes the
group's prompt again. Packing lays the sequences out in one row [1, N] instead:

  varlen  the sequences end to end, each attending only to itself: N = sum of their lengths, no padding.
  tree    the sequences folded into a prefix trie: a token that several sequences share with the same
          prefix (a group's prompt, a common system prompt, a conversation before its turns branch) is
          forwarded once. N = the trie's unique tokens.

Layout. The trie is compressed into segments: maximal runs of tokens with one predecessor and one
successor in the trie, ending where sequences branch or end. A segment's context is the path from the root
to it. Segments are laid out level by level (all roots, then their children, ...): every level is one
contiguous stretch of the row, which is what the recurrent layers need. Each token keeps its position in its
sequences (its depth in the trie) as position id, so rotary embeddings see what an unpacked forward sees.
varlen is the tree whose segments are all roots.

Layers.
  * softmax attention (palingenesis.seco's attention route, models configured for sdpa): each segment's
    queries attend to the K/V of its root-to-segment path, gathered by index, in one variable-length flash
    call; the causal mask is aligned bottom-right, so a query sees every ancestor and its own causal past.
    Without a flash kernel (CPU, fp32) an explicit [N, N] ancestor mask gives the same result (a reference
    path: memory grows with N^2).
  * Gated DeltaNet (Qwen3.5, Qwen3-Next): the short causal convolution reads each segment's last
    kernel-1 inputs from its ancestors; the chunked delta rule runs once per level over the level's segments
    (the layer's own kernel, variable-length), each segment starting from its parent's final state. The
    state's gradient flows back to the parent through autograd, so the gradients are exact.

Gradients reach every shared token from all the sequences that read it, as in the unpacked forward:
per-token log-probs and gradients equal the padded path's up to floating-point reordering.

The trie and the attention over gathered paths follow TRL's AsyncGRPOTrainer (TreeRow, tree_attention,
patch_tree_recurrent; Apache-2.0, https://github.com/huggingface/trl). The recurrent layers here reuse the
layer's own chunked kernel per tree level instead of TRL's dedicated Triton tree kernels.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PACKED_KWARG = "pgs_packed"  # the forward keyword argument carrying a PackedRow down to the layers
PACKING_MODES = ("padded", "varlen", "tree")
_ATTENTION_LAYER_TYPES = {"full_attention", "attention"}
_RECURRENT_NAMES = ("linear_attn", "mamba", "ssm", "delta_net", "deltanet", "recurrent", "short_conv")
# Explicit-mask attention (no flash kernel) builds an [N, N] mask per layer: past this many tokens, warn once.
MATH_ATTENTION_WARN_TOKENS = 8192


class PackingError(ValueError):
    """The model or the batch cannot be packed exactly."""


# ── the trie ───────────────────────────────────────────────────────────────────


class _Node:
    """A segment: the tokens sequences[seq][lo:hi], at positions lo..hi-1 of every sequence through it."""

    __slots__ = ("seq", "lo", "hi", "children", "parent", "index", "start")

    def __init__(self, seq: int, lo: int, hi: int, parent: _Node | None):
        self.seq, self.lo, self.hi, self.parent = seq, lo, hi, parent
        self.children: dict[int, _Node] = {}
        self.index = -1  # segment index in the layout
        self.start = -1  # first packed position


def _common(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the common prefix of two int arrays."""
    m = min(len(a), len(b))
    diff = np.flatnonzero(a[:m] != b[:m])
    return int(diff[0]) if diff.size else m


class _Trie:
    """A radix trie over token sequences (numpy int64 arrays), built incrementally."""

    def __init__(self, sequences: list[np.ndarray], share: bool = True):
        self.sequences = sequences
        self.share = share
        self.roots: dict[int, _Node] = {}  # first token -> root (share); sequence index -> root (no sharing)
        self.unique = 0  # tokens in the trie

    def match(self, seq: np.ndarray) -> int:
        """How many leading tokens of `seq` the trie already holds."""
        if not self.share:
            return 0
        pos, children = 0, self.roots
        while pos < len(seq):
            node = children.get(int(seq[pos]))
            if node is None:
                return pos
            tokens = self.sequences[node.seq][node.lo : node.hi]
            common = _common(tokens, seq[pos:])
            pos += common
            if common < len(tokens):
                return pos
            children = node.children
        return pos

    def insert(self, s: int) -> None:
        seq = self.sequences[s]
        if not self.share:
            self.roots[s] = _Node(s, 0, len(seq), None)
            self.unique += len(seq)
            return
        pos, children, parent = 0, self.roots, None
        while pos < len(seq):
            node = children.get(int(seq[pos]))
            if node is None:
                children[int(seq[pos])] = _Node(s, pos, len(seq), parent)
                self.unique += len(seq) - pos
                return
            tokens = self.sequences[node.seq][node.lo : node.hi]
            common = _common(tokens, seq[pos:])
            if common == len(tokens):
                pos, children, parent = pos + common, node.children, node
                continue
            if pos + common == len(seq):  # the sequence ends inside the segment: nothing new
                return
            # the sequence leaves the segment after `common` tokens: split it there
            upper = _Node(node.seq, node.lo, node.lo + common, parent)
            children[int(seq[pos])] = upper
            node.lo += common
            node.parent = upper
            upper.children[int(self.sequences[node.seq][node.lo])] = node
            upper.children[int(seq[pos + common])] = _Node(s, pos + common, len(seq), upper)
            self.unique += len(seq) - pos - common
            return

    def path(self, s: int) -> list[tuple[_Node, int]]:
        """The segments sequence s runs through, with how many of each segment's tokens it takes."""
        seq = self.sequences[s]
        if not self.share:
            node = self.roots[s]
            return [(node, len(seq))]
        out, pos, children = [], 0, self.roots
        while pos < len(seq):
            node = children[int(seq[pos])]
            take = min(node.hi - node.lo, len(seq) - pos)
            out.append((node, take))
            pos += take
            children = node.children
        return out


# ── the packed row ─────────────────────────────────────────────────────────────


@dataclass
class _Level:
    lo: int  # packed token range of the level
    hi: int
    lengths: list[int]  # its segments' lengths, in order
    parent_slot: np.ndarray | None  # per segment: its parent's index within the previous level (None: roots)


class PackedRow:
    """Sequences packed into one row: what the forward needs, and where each sequence's tokens went.

    Build with PackedRow.build(sequences, share); `positions(i)` maps sequence i's token indices to packed
    positions (the hidden state at packed position p is the unpacked forward's at that token).

    A plain class on purpose: the row travels down the forward as a keyword argument, and FSDP2 rebuilds the
    dataclasses, lists and sets it finds in a forward's arguments (casting their tensors); the layers must see
    this very object."""

    def __init__(
        self,
        tokens: np.ndarray,
        position_ids: np.ndarray,
        seg_starts: np.ndarray,
        seg_parents: np.ndarray,
        paths: list[np.ndarray],
        levels: list[_Level],
        sequence_positions: list[np.ndarray],
    ):
        self.tokens = tokens  # [N] token ids
        self.position_ids = position_ids  # [N] each token's position in its sequences
        self.seg_starts = seg_starts  # [S + 1] segment boundaries in the row (level-major)
        self.seg_parents = seg_parents  # [S] parent segment, -1 for a root
        self.paths = paths  # per segment: packed positions of its root-to-segment path (its own last)
        self.levels = levels
        self.sequence_positions = sequence_positions  # per input sequence: packed position of each of its tokens
        self._device_cache: dict = {}
        # layers that served this row in the current forward (checked by the caller: packing must not be ignored)
        self.attention_layers: set = set()
        self.linear_layers: set = set()

    def __repr__(self) -> str:
        return f"PackedRow(tokens={self.num_tokens}, segments={self.num_segments}, levels={len(self.levels)})"

    @classmethod
    def build(cls, sequences: list[list[int]] | list[np.ndarray], share: bool) -> PackedRow:
        arrays = [np.asarray(s, dtype=np.int64) for s in sequences]
        if not arrays or any(len(a) == 0 for a in arrays):
            raise PackingError("packing needs at least one sequence, each of at least one token")
        trie = _Trie(arrays, share)
        for i in range(len(arrays)):
            trie.insert(i)
        return cls._layout(trie)

    @classmethod
    def _layout(cls, trie: _Trie) -> PackedRow:
        arrays = trie.sequences
        levels_nodes: list[list[_Node]] = [list(trie.roots.values())]
        while True:
            nxt = [child for node in levels_nodes[-1] for child in node.children.values()]
            if not nxt:
                break
            levels_nodes.append(nxt)
        tokens, positions, starts, parents, paths, levels = [], [], [0], [], [], []
        cursor, index = 0, 0
        for depth, nodes in enumerate(levels_nodes):
            lo = cursor
            slot_of = {id(n): i for i, n in enumerate(levels_nodes[depth - 1])} if depth else {}
            lengths, slots = [], []
            for node in nodes:
                node.index, node.start = index, cursor
                n = node.hi - node.lo
                tokens.append(arrays[node.seq][node.lo : node.hi])
                positions.append(np.arange(node.lo, node.hi, dtype=np.int64))
                own = np.arange(cursor, cursor + n, dtype=np.int64)
                if node.parent is None:
                    parents.append(-1)
                    paths.append(own)
                else:
                    parents.append(node.parent.index)
                    paths.append(np.concatenate([paths[node.parent.index], own]))
                    slots.append(slot_of[id(node.parent)])
                cursor += n
                starts.append(cursor)
                lengths.append(n)
                index += 1
            levels.append(_Level(lo, cursor, lengths, np.asarray(slots, dtype=np.int64) if depth else None))
        sequence_positions = []
        for s in range(len(arrays)):
            parts = [np.arange(node.start, node.start + take, dtype=np.int64) for node, take in trie.path(s)]
            sequence_positions.append(np.concatenate(parts))
        return cls(
            tokens=np.concatenate(tokens),
            position_ids=np.concatenate(positions),
            seg_starts=np.asarray(starts, dtype=np.int64),
            seg_parents=np.asarray(parents, dtype=np.int64),
            paths=paths,
            levels=levels,
            sequence_positions=sequence_positions,
        )

    # ---------------------------------------------------------------- sizes

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_segments(self) -> int:
        return len(self.seg_parents)

    @property
    def flat(self) -> bool:
        """Every segment a root (varlen, or a tree that shares nothing): no ancestors to read."""
        return len(self.levels) == 1

    @property
    def gathered_tokens(self) -> int:
        """K/V rows the attention reads: every segment's whole path."""
        return int(sum(len(p) for p in self.paths))

    def positions(self, sequence: int) -> np.ndarray:
        return self.sequence_positions[sequence]

    # ---------------------------------------------------------------- device tensors

    def _cached(self, key, device, make):
        full = (key, str(device))
        value = self._device_cache.get(full)
        if value is None:
            value = make()
            self._device_cache[full] = value
        return value

    def _tensor(self, key: str, array: np.ndarray, device, dtype=torch.long) -> torch.Tensor:
        return self._cached(key, device, lambda: torch.from_numpy(np.ascontiguousarray(array)).to(device, dtype))

    def forward_kwargs(self, device) -> dict:
        """input_ids [1, N], position_ids [1, N] and this row (PACKED_KWARG) for the model's forward (no
        cache: the row's tokens are not one sequence)."""
        return {
            "input_ids": self._tensor("tokens", self.tokens, device)[None],
            "position_ids": self._tensor("position_ids", self.position_ids, device)[None],
            "use_cache": False,
            PACKED_KWARG: self,
        }

    def cu_seqlens_q(self, device) -> torch.Tensor:
        return self._tensor("cu_q", self.seg_starts, device, torch.int32)

    def kv_index(self, device) -> torch.Tensor:
        return self._cached(
            "kv_index",
            device,
            lambda: torch.from_numpy(np.concatenate(self.paths)).to(device),
        )

    def cu_seqlens_k(self, device) -> torch.Tensor:
        lengths = np.asarray([len(p) for p in self.paths], dtype=np.int64)
        return self._tensor("cu_k", np.concatenate([[0], np.cumsum(lengths)]), device, torch.int32)

    def max_seqlen_q(self) -> int:
        return int(np.diff(self.seg_starts).max())

    def max_seqlen_k(self) -> int:
        return max(len(p) for p in self.paths)

    def seq_idx(self, device) -> torch.Tensor:
        """[1, N] int32: the segment of each token (causal-conv1d's document index; flat rows)."""
        lengths = np.diff(self.seg_starts)
        return self._tensor("seq_idx", np.repeat(np.arange(len(lengths)), lengths)[None], device, torch.int32)

    def attention_mask(self, device) -> torch.Tensor:
        """[N, N] bool: token i may attend to token j (the reference path; N^2 memory)."""

        def make():
            n = self.num_tokens
            segment = np.repeat(np.arange(self.num_segments), np.diff(self.seg_starts))
            ancestor = np.zeros((self.num_segments, self.num_segments), dtype=bool)  # [s, a]: a strictly above s
            for s, parent in enumerate(self.seg_parents):
                if parent >= 0:
                    ancestor[s] = ancestor[parent]
                    ancestor[s, parent] = True
            idx = np.arange(n)
            same = (segment[:, None] == segment[None, :]) & (idx[None, :] <= idx[:, None])
            allowed = same | ancestor[segment[:, None], segment[None, :]]
            return torch.from_numpy(allowed).to(device)

        return self._cached("mask", device, make)

    def conv_index(self, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """For a causal convolution of `width` taps: indices into [x ; 0] (N + 1 columns) that put each
        segment's last width-1 ancestor inputs (zeros at the root) before its own tokens, and where the
        segments' own tokens land in that extended sequence."""

        def make():
            halo = width - 1
            n = self.num_tokens
            src, real = [], []
            cursor = 0
            for s in range(self.num_segments):
                lo, hi = int(self.seg_starts[s]), int(self.seg_starts[s + 1])
                parent = int(self.seg_parents[s])
                before = self.paths[parent][-halo:] if parent >= 0 and halo else np.empty(0, dtype=np.int64)
                if halo:
                    src.append(np.full(halo - len(before), n, dtype=np.int64))
                    src.append(before)
                src.append(np.arange(lo, hi, dtype=np.int64))
                real.append(np.arange(cursor + halo, cursor + halo + hi - lo, dtype=np.int64))
                cursor += halo + hi - lo
            return (
                torch.from_numpy(np.concatenate(src)).to(device),
                torch.from_numpy(np.concatenate(real)).to(device),
            )

        return self._cached(("conv", width), device, make)

    def level_tensors(self, device) -> list[tuple[int, int, torch.Tensor, torch.Tensor | None, list[int]]]:
        """Per level: token range, cu_seqlens of its segments (int32, level-relative), parent slots, lengths."""

        def make():
            out = []
            for level in self.levels:
                cu = np.concatenate([[0], np.cumsum(level.lengths)])
                slots = None if level.parent_slot is None else torch.from_numpy(level.parent_slot).to(device)
                out.append((level.lo, level.hi, torch.from_numpy(cu).to(device, torch.int32), slots, level.lengths))
            return out

        return self._cached("levels", device, make)

    def padded_level_index(self, level: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """For kernels without variable-length support: [S, L] indices into the level's tokens (+ one zero row
        at the end for padding) and the flat positions of the real tokens in the [S, L] layout."""

        def make():
            lengths = self.levels[level].lengths
            width = max(lengths)
            total = sum(lengths)
            index = np.full((len(lengths), width), total, dtype=np.int64)
            real = []
            cursor = 0
            for r, n in enumerate(lengths):
                index[r, :n] = np.arange(cursor, cursor + n)
                real.append(np.arange(r * width, r * width + n))
                cursor += n
            return torch.from_numpy(index).to(device), torch.from_numpy(np.concatenate(real)).to(device)

        return self._cached(("padded", level), device, make)

    def routing(self, recorded: list, shape) -> torch.Tensor:
        """MoE routing replay: [1, N, *shape] expert indices from each sequence's recorded routing (the first
        sequence that recorded a shared position wins), -1 where none."""
        out = np.full((self.num_tokens, *shape), -1, dtype=np.int64)
        filled = np.zeros(self.num_tokens, dtype=bool)
        for s, r in enumerate(recorded):
            if r is None or not len(r) or tuple(r.shape[1:]) != tuple(shape):
                continue
            pos = self.sequence_positions[s][: len(r)]
            take = ~filled[pos]
            out[pos[take]] = np.asarray(r[: len(pos)])[take].astype(np.int64)
            filled[pos] = True
        return torch.from_numpy(out)[None]

    def begin_forward(self) -> None:
        self.attention_layers.clear()
        self.linear_layers.clear()


# ── planning micro-batches ─────────────────────────────────────────────────────


def plan(
    sequences: list[list[int]] | list[np.ndarray],
    groups: list,
    trained: list[int],
    budget: int,
    share: bool,
) -> list[list[int]]:
    """Micro-batches (lists of sequence indices) of at most `budget` packed tokens and `budget` loss terms.

    varlen (share False): first-fit decreasing on the sequences' lengths.
    tree (share True): a group at a time, longest groups first; each sequence goes to the micro-batch whose trie
    already holds the longest prefix of it, among those it fits (its new tokens counted, not its length).
    A sequence longer than the budget gets a micro-batch of its own."""
    arrays = [np.asarray(s, dtype=np.int64) for s in sequences]
    if not share:
        order = sorted(range(len(arrays)), key=lambda i: len(arrays[i]), reverse=True)
        bins: list[list[int]] = []
        loads: list[int] = []
        for i in order:
            n = len(arrays[i])
            for b, load in enumerate(loads):
                if load + n <= budget:
                    bins[b].append(i)
                    loads[b] += n
                    break
            else:
                bins.append([i])
                loads.append(n)
        return bins
    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    ordered = sorted(by_group.values(), key=lambda members: max(len(arrays[i]) for i in members), reverse=True)
    tries: list[_Trie] = []
    terms: list[int] = []
    bins = []
    for members in ordered:
        for i in sorted(members, key=lambda i: len(arrays[i]), reverse=True):
            best, best_match = -1, -1
            for b, trie in enumerate(tries):
                match = trie.match(arrays[i])
                if trie.unique + len(arrays[i]) - match <= budget and terms[b] + trained[i] <= budget:
                    if match > best_match:
                        best, best_match = b, match
            if best < 0:
                tries.append(_Trie(arrays, share=True))
                terms.append(0)
                bins.append([])
                best = len(tries) - 1
            tries[best].insert(i)
            terms[best] += trained[i]
            bins[best].append(i)
    return bins


# ── attention ──────────────────────────────────────────────────────────────────


def _flash_capable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] % 8 == 0
        and q.shape[-1] <= 256
        and k.shape[-1] == v.shape[-1] == q.shape[-1]
    )


class TreeFlashAttention(torch.autograd.Function):
    """Variable-length flash attention of each segment's queries over its gathered path, causal bottom-right.
    q [N, H, D], k / v [N, Hkv, D]. Saves the ungathered K/V (N rows, not the gathered paths) and gathers
    again in the backward; the K/V gradients of shared tokens are summed in fp32."""

    @staticmethod
    def forward(ctx, q, k, v, kv_index, cu_q, cu_k, max_q: int, max_k: int, scale: float):
        kg, vg = k.index_select(0, kv_index), v.index_select(0, kv_index)
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            q, kg, vg, cu_q, cu_k, max_q, max_k, 0.0, True, False, scale=scale
        )
        ctx.save_for_backward(q, k, v, out, lse, rng, unused, kv_index, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.scale = max_q, max_k, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, rng, unused, kv_index, cu_q, cu_k = ctx.saved_tensors
        kg, vg = k.index_select(0, kv_index), v.index_select(0, kv_index)
        dq, dkg, dvg = torch.ops.aten._flash_attention_backward(
            dout.contiguous(),
            q,
            kg,
            vg,
            out,
            lse,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            0.0,
            True,
            rng,
            unused,
            scale=ctx.scale,
        )
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device).index_add_(0, kv_index, dkg.float())
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device).index_add_(0, kv_index, dvg.float())
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None


def _math_attention(q, k, v, mask, scale):
    """q [N, H, D], k / v [N, Hkv, D], mask [N, N] -> [N, H, D] (at least fp32 inside)."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    dtype = torch.promote_types(q.dtype, torch.float32)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).to(dtype),
        k.transpose(0, 1).to(dtype),
        v.transpose(0, 1).to(dtype),
        attn_mask=mask,
        scale=scale,
    )
    return out.transpose(0, 1).to(q.dtype)


_VERIFIED: dict = {}  # (device, dtype) -> the self-check's max errors
_WARNED = threading.Event()


def _device_key(device) -> str:
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


def verify_flash_tree(device, dtype=torch.bfloat16) -> None:
    """The flash tree path against the math path on a small random tree, forward and backward; once per device
    and dtype, outside any model forward. The causal alignment of the variable-length kernel (bottom-right) is
    what makes a segment see its ancestors: a kernel that aligned it top-left would silently compute something
    else."""
    key = (_device_key(device), dtype)
    if key in _VERIFIED:
        return
    gen = torch.Generator().manual_seed(0)
    seqs = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 10, 11, 12], [1, 2, 3, 4, 10, 13], [20, 21, 22]]
    row = PackedRow.build(seqs, share=True)
    n = row.num_tokens
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        q = torch.randn(n, 4, 64, generator=gen).to(device, dtype).requires_grad_(True)
        k = torch.randn(n, 2, 64, generator=gen).to(device, dtype).requires_grad_(True)
        v = torch.randn(n, 2, 64, generator=gen).to(device, dtype).requires_grad_(True)
        grad = torch.randn(n, 4, 64, generator=gen).to(device, dtype)
        scale = 0.125
        out = TreeFlashAttention.apply(
            q, k, v, row.kv_index(device), row.cu_seqlens_q(device), row.cu_seqlens_k(device), row.max_seqlen_q(),
            row.max_seqlen_k(), scale,
        )  # fmt: skip
        got = torch.autograd.grad(out, (q, k, v), grad)
        qf, kf, vf = (t.detach().double().requires_grad_(True) for t in (q, k, v))
        ref = _math_attention(qf, kf, vf, row.attention_mask(device), scale)
        want = torch.autograd.grad(ref, (qf, kf, vf), grad.double())
    errors = [(out.double() - ref).abs().max().item()]
    errors += [(a.double() - b).abs().max().item() for a, b in zip(got, want)]
    if max(errors) > 5e-2:
        raise PackingError(
            f"the flash tree attention disagrees with the reference on {device} ({dtype}): max errors {errors}. "
            "The variable-length flash kernel behind torch must align its causal mask bottom-right (FlashAttention "
            ">= 2.1). Use train.packing padded or varlen."
        )
    _VERIFIED[key] = errors


def packed_attention(module, query, key, value, row: PackedRow, scaling, dropout, kwargs) -> torch.Tensor:
    """Attention over a packed row: query [1, H, N, D], key / value [1, Hkv, N, D] -> [1, N, H, D]."""
    if dropout:
        raise PackingError("packed attention does not apply attention dropout (RL trains without it)")
    for name in ("sliding_window", "softcap", "s_aux", "sinks", "position_bias"):
        if kwargs.get(name) is not None:
            raise PackingError(f"packed attention does not support {name} (layer {type(module).__name__})")
    if query.shape[0] != 1:
        raise PackingError(f"a packed row is one row, got a batch of {query.shape[0]}")
    row.attention_layers.add(id(module))
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    q, k, v = (t[0].transpose(0, 1).contiguous() for t in (query, key, value))  # [N, heads, D]
    scale = scaling if scaling is not None else q.shape[-1] ** -0.5
    device = q.device
    if _flash_capable(q, k, v):
        if row.flat:  # every segment a root: plain variable-length attention, nothing gathered
            cu = row.cu_seqlens_q(device)
            m = row.max_seqlen_q()
            out = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, m, m, 0.0, True, False, scale=scale)[0]
        else:
            if (_device_key(device), q.dtype) not in _VERIFIED:
                raise PackingError(
                    "the flash tree attention runs after its self-check: call prepare_model(model, device) first"
                )
            out = TreeFlashAttention.apply(
                q,
                k,
                v,
                row.kv_index(device),
                row.cu_seqlens_q(device),
                row.cu_seqlens_k(device),
                row.max_seqlen_q(),
                row.max_seqlen_k(),
                scale,
            )
        return out.unsqueeze(0)
    if row.num_tokens > MATH_ATTENTION_WARN_TOKENS and not _WARNED.is_set():
        _WARNED.set()
        logger.warning(
            "packed attention without a flash kernel (%s, %s, head dim %d): an explicit [N, N] mask over %d tokens. "
            "This is the reference path; on a GPU use bf16 autocast.",
            device,
            q.dtype,
            q.shape[-1],
            row.num_tokens,
        )
    return _math_attention(q, k, v, row.attention_mask(device), scale).unsqueeze(0)


# ── Gated DeltaNet ─────────────────────────────────────────────────────────────

_LOCAL = threading.local()  # the row the current linear-attention forward runs on (set by the layer's wrapper)


def _current() -> PackedRow | None:
    return getattr(_LOCAL, "row", None)


def is_gated_delta_net(module: nn.Module) -> bool:
    from palingenesis.context_parallel import is_gated_delta_net as check

    return check(module)


def supports_varlen(kernel) -> bool:
    """Whether a chunked gated-delta-rule kernel takes `cu_seqlens` with one initial state per sequence:
    flash-linear-attention's does; transformers' torch fallback ignores `cu_seqlens` (its levels then run as
    right-padded rows)."""
    module = getattr(kernel, "__module__", "") or ""
    name = getattr(kernel, "__name__", "") or ""
    return module.startswith("fla") or (name == "chunk_gated_delta_rule" and not module.startswith("transformers"))


class _PackedLinearAttention:
    """The kernels of one Gated DeltaNet layer, following a packed row when the forward carries one and the
    layer's own otherwise. transformers calls them through instance attributes (`chunk_gated_delta_rule`,
    `causal_conv1d_fn`), which are replaced by this route's methods; the layer's forward is wrapped to tell
    them the row (forward keyword arguments reach the layer, not its kernels), recomputations under
    activation checkpointing included."""

    def __init__(self, module: nn.Module):
        self.layer = id(module)
        self.delta_rule = module.chunk_gated_delta_rule
        self.conv = module.causal_conv1d_fn
        self.forward = module.forward
        self.varlen_kernel = supports_varlen(self.delta_rule)
        module.chunk_gated_delta_rule = self.chunk_gated_delta_rule
        module.causal_conv1d_fn = self.causal_conv1d_fn
        module.forward = self.wrapped_forward
        module._pgs_packed_route = self

    @classmethod
    def of(cls, module: nn.Module) -> _PackedLinearAttention:
        route = module.__dict__.get("_pgs_packed_route")
        return route if isinstance(route, cls) else cls(module)

    def wrapped_forward(self, *args, **kwargs):
        row = kwargs.get(PACKED_KWARG)
        previous = _current()
        _LOCAL.row = row
        try:
            if row is not None and kwargs.get("cache_params") is not None:
                raise PackingError("a packed forward runs without a cache (training)")
            return self.forward(*args, **kwargs)
        finally:
            _LOCAL.row = previous

    def _local_conv(self, x, weight, bias, activation, **kwargs):
        if self.conv is not None:
            return self.conv(x=x, weight=weight, bias=bias, activation=activation, **kwargs)
        # transformers' own fallback without causal-conv1d: a depthwise causal conv1d and SiLU
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
        return F.silu(out[..., : x.shape[-1]])

    def causal_conv1d_fn(self, x, weight, bias=None, activation=None, seq_idx=None, **kwargs):
        row = _current()
        if row is None:
            return self._local_conv(x, weight, bias, activation, seq_idx=seq_idx, **kwargs)
        row.linear_layers.add(self.layer)
        if x.shape[0] != 1 or kwargs.get("initial_states") is not None:
            raise PackingError("a packed linear-attention convolution takes one row and no cached state")
        if row.flat and self.conv is not None:  # causal-conv1d resets at document boundaries itself
            return self.conv(x=x, weight=weight, bias=bias, activation=activation, seq_idx=row.seq_idx(x.device))
        src, real = row.conv_index(weight.shape[-1], x.device)
        extended = torch.cat([x, x.new_zeros(x.shape[0], x.shape[1], 1)], dim=-1).index_select(-1, src)
        return self._local_conv(extended.contiguous(), weight, bias, activation).index_select(-1, real)

    def chunk_gated_delta_rule(self, *args, **kwargs):
        row = _current()
        if row is None:
            return self.delta_rule(*args, **kwargs)
        return self._packed(row, *args, **kwargs)

    def _packed(
        self,
        row: PackedRow,
        query,
        key,
        value,
        g=None,
        beta=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        **kwargs,
    ):
        if initial_state is not None or query.shape[0] != 1:
            raise PackingError("a packed delta rule takes one row and no initial state")
        row.linear_layers.add(self.layer)
        device = query.device
        outs, finals = [], None
        for level, (lo, hi, cu, slots, lengths) in enumerate(row.level_tensors(device)):
            start = None if slots is None else finals.index_select(0, slots)
            part = [t[:, lo:hi] for t in (query, key, value, g, beta)]
            if self.varlen_kernel:
                out, finals = self.delta_rule(
                    *part[:3],
                    g=part[3],
                    beta=part[4],
                    initial_state=start,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu,
                    **kwargs,
                )
            else:
                out, finals = self._padded_level(row, level, part, start, use_qk_l2norm_in_kernel, kwargs)
            outs.append(out)
        return torch.cat(outs, dim=1), None

    def _padded_level(self, row, level, part, start, l2norm, kwargs):
        """One level through a kernel without variable-length support: its segments as right-padded rows. A
        padded step has zero key, value, gate and beta, which leaves the state unchanged, so each row's final
        state is its segment's."""
        index, real = row.padded_level_index(level, part[0].device)

        def rows(t):
            flat = t[0]
            flat = torch.cat([flat, flat.new_zeros(1, *flat.shape[1:])])
            return flat[index]  # [S, L, ...]

        q, k, v, g, beta = (rows(t) for t in part)
        out, final = self.delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=start,
            output_final_state=True,
            use_qk_l2norm_in_kernel=l2norm,
            **kwargs,
        )
        out = out.reshape(-1, *out.shape[2:]).index_select(0, real)
        return out[None], final


# ── preparing a model ──────────────────────────────────────────────────────────


@dataclass
class PackingSupport:
    """What a prepared model's packed forward must touch: checked after every packed forward."""

    attention: set  # ids of the softmax-attention modules
    linear: set  # ids of the Gated DeltaNet modules

    def check(self, row: PackedRow) -> None:
        missing_attention = self.attention - row.attention_layers
        missing_linear = self.linear - row.linear_layers
        if missing_attention or missing_linear:
            raise PackingError(
                f"packed forward: {len(missing_attention)} attention and {len(missing_linear)} linear-attention "
                "layers ran without the packed layout (they would mix the packed sequences). The model's attention "
                "must be configured for sdpa (rerouted by palingenesis), and its recurrent layers must be "
                "transformers' Gated DeltaNet."
            )


def _attention_modules(model: nn.Module, linear: set) -> set:
    return {
        id(m)
        for m in model.modules()
        if id(m) not in linear
        and hasattr(m, "layer_idx")
        and hasattr(m, "is_causal")
        and type(m).__name__.endswith("Attention")
    }


def prepare_model(model: nn.Module, device=None) -> PackingSupport:
    """Check that `model` can run packed rows exactly and route its layers; raise PackingError otherwise.
    Softmax attention runs through palingenesis.seco's attention route (sdpa models are rerouted here). On a
    CUDA `device`, the flash tree attention is checked against the reference once (verify_flash_tree)."""
    from transformers import PretrainedConfig

    from palingenesis.seco import _ATTN, use_chunk_attention

    use_chunk_attention(model)
    seen = set()
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if not isinstance(cfg, PretrainedConfig) or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        types = set(getattr(text, "layer_types", None) or [])
        other = sorted(types - _ATTENTION_LAYER_TYPES - {"linear_attention"})
        if other:
            raise PackingError(
                f"packing: the model has {other} layers; packed rows support full softmax attention and Gated "
                "DeltaNet layers only (sliding windows and other recurrent layers are not implemented)."
            )
        # a sliding window applies when the config has one and does not turn it off (Qwen2 / Qwen3:
        # use_sliding_window false; Mistral has no switch)
        if getattr(text, "sliding_window", None) and getattr(text, "use_sliding_window", True):
            raise PackingError("packing: the model uses sliding-window attention, which packed rows do not implement.")
        for name in ("attention_chunk_size", "attn_logit_softcapping"):
            if getattr(text, name, None):
                raise PackingError(f"packing: the model sets {name}, which packed attention does not implement.")
        implementation = cfg._attn_implementation
        if implementation not in (_ATTN, None) and (types & _ATTENTION_LAYER_TYPES or not types):
            raise PackingError(
                f"packing needs the model's attention configured for sdpa (palingenesis routes it), got "
                f"{implementation!r}."
            )
    linear = {id(m) for m in model.modules() if is_gated_delta_net(m)}
    recurrent = sorted(
        name
        for name, m in model.named_modules()
        if id(m) not in linear and name.rsplit(".", 1)[-1].lower() in _RECURRENT_NAMES
    )
    if recurrent:
        raise PackingError(
            f"packing: {recurrent[0]} is a recurrent layer other than transformers' Gated DeltaNet; its state would "
            "carry from one packed sequence into the next."
        )
    for m in model.modules():
        if id(m) in linear:
            _PackedLinearAttention.of(m)
    if device is not None and torch.device(device).type == "cuda":
        for dtype in (torch.bfloat16, torch.float16):
            verify_flash_tree(device, dtype)
    return PackingSupport(attention=_attention_modules(model, linear), linear=linear)
