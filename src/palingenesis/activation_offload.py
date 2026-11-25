"""Activation offload: the activations autograd saves for the backward wait in pinned host memory instead of GPU
memory. They are copied out during the forward and back during the backward on a side CUDA stream, overlapped with
the compute of other layers.

Design (after torchtune's OffloadActivations, BSD-3, and the layer-wise CPU offload of Megatron-LM / Transformer
Engine, Apache-2.0; re-implemented, no code copied):

- torch.autograd.graph.saved_tensors_hooks: every tensor autograd saves in the forward goes through `pack`, which
  starts a device-to-host copy of a large activation on the side stream and returns a handle; `unpack` returns the
  tensor on the GPU in the backward.
- The transformer layer is the unit of scheduling, as in Megatron: forward pre-hooks tag each saved tensor with its
  layer. In the forward, a layer's copies may stay in flight while the next `in_flight` layers compute; then the
  compute stream waits for them and their GPU memory is released. In the backward, reaching layer i fetches the
  saved tensors of layers i-1 ... i-`prefetch` on the side stream while layer i's backward runs.
- Not offloaded: the last `keep_last` layers (their backward comes first: a copy would only add a stall), tensors
  saved outside the layers (embeddings, final norm, loss), parameters and their casts, expanded (stride-0) views,
  tensor subclasses, and tensors smaller than `min_bytes`.
- With activation checkpointing autograd saves only each layer's inputs (full), or its inputs plus the outputs that
  selective checkpointing keeps (selective, on torch versions whose checkpoint routes them through saved-tensor
  hooks: `respect_saved_tensors_hooks`): those are what moves. `boundaries_only` moves the layer inputs alone.
- Host memory comes from a pinned arena of 256 MiB chunks that every step reuses (torch's caching host allocator
  rounds each request up to a power of two, up to twice the memory), bounded by a share of the machine's RAM and
  by what the machine has available when a chunk is added (a tenth of its RAM stays free): past either, an error.
- A tensor saved by several ops (a layer input read by a norm and by the residual) is copied once.

The values come back bit-identical, so the gradients are those of a run without offload.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import weakref
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MiB = 1 << 20
ARENA_CHUNK = 256 * MiB
_ARENA_ALIGN = 512  # bytes: each offloaded tensor starts at an address that is a multiple of 512


@dataclass(frozen=True)
class OffloadOptions:
    """min_bytes: smallest tensor moved (smaller ones stay on the GPU: the copy costs more than it frees).
    in_flight: layers whose device-to-host copies may still run while a later layer computes (>= 1); each one keeps
               its activations on the GPU until its copies finish.
    prefetch: layers fetched ahead of the backward (0: a layer's tensors are fetched when its backward needs them).
    boundaries_only: offload only the tensors a checkpointed layer saves at its boundary (its inputs); what
               selective checkpointing keeps stays on the GPU.
    keep_last: final layers never offloaded."""

    min_bytes: int = MiB
    in_flight: int = 1
    prefetch: int = 1
    boundaries_only: bool = False
    keep_last: int = 1

    def __post_init__(self):
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes must be >= 0, got {self.min_bytes}")
        if self.in_flight < 1:
            raise ValueError(f"in_flight must be >= 1, got {self.in_flight}")
        if self.prefetch < 0:
            raise ValueError(f"prefetch must be >= 0, got {self.prefetch}")
        if self.keep_last < 0:
            raise ValueError(f"keep_last must be >= 0, got {self.keep_last}")


def _in_backward() -> bool:
    """Whether this thread is running an autograd backward (where checkpointing recomputes layer forwards)."""
    task = getattr(torch._C, "_current_graph_task_id", None)
    return task is not None and task() != -1


def host_limit_bytes() -> int | None:
    """Pinned host memory this process may hold for offloaded activations: the share of RAM SeCO's K/V offload may
    use (palingenesis.seco_attention.HOST_BUDGET_FRACTION), split between the processes of this node."""
    from palingenesis.seco_attention import HOST_BUDGET_FRACTION, _host_bytes

    total = _host_bytes("SC_PHYS_PAGES")
    if not total:
        return None
    ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    return int(HOST_BUDGET_FRACTION * total / ranks)


HOST_HEADROOM_FRACTION = 0.1  # of the machine's RAM: what must stay available after a chunk is pinned


def host_headroom_bytes() -> int | None:
    """Host memory that must stay available (the kernel's MemAvailable) after the arena pins another chunk. The
    share-of-RAM limit alone does not see what other processes, tmpfs files or this process's own model copy hold:
    pinning past what is available gets the process killed by the kernel instead of an error."""
    from palingenesis.seco_attention import _host_bytes

    total = _host_bytes("SC_PHYS_PAGES")
    return int(HOST_HEADROOM_FRACTION * total) if total else None


def _host_available_bytes() -> int | None:
    from palingenesis.accelerator import host_available_bytes

    return host_available_bytes()


# ----------------------------------------------------------------------------------------------------------------
# Host memory
# ----------------------------------------------------------------------------------------------------------------


class _Chunk:
    """`size` usable bytes of `buf` from its first byte at an aligned address (`start`); `offset` is the bump
    pointer, an index into `buf`."""

    __slots__ = ("buf", "size", "start", "offset", "live")

    def __init__(self, buf: torch.Tensor, size: int):
        start = -buf.data_ptr() % _ARENA_ALIGN
        if start + size > buf.numel():
            raise ValueError(f"a buffer of {buf.numel()} bytes has no {size} bytes from an aligned address")
        self.buf, self.size, self.start, self.offset, self.live = buf, size, start, start, 0

    @property
    def free(self) -> int:
        return self.start + self.size - self.offset


class HostArena:
    """Host buffers for offloaded tensors, carved from large (pinned) chunks with a bump pointer. A chunk's pointer
    rewinds when the last tensor carved from it is released, so a training step reuses the previous step's memory."""

    def __init__(
        self,
        pin: bool,
        limit_bytes: int | None = None,
        chunk_bytes: int = ARENA_CHUNK,
        headroom_bytes: int | None = None,
        available=_host_available_bytes,
    ):
        """limit_bytes: the most the arena may reserve. headroom_bytes: host memory that must stay available
        (`available()`, bytes or None when unknown) after a new chunk; None: not checked."""
        self.pin = pin
        self.limit_bytes = limit_bytes
        self.chunk_bytes = chunk_bytes
        self.headroom_bytes = headroom_bytes
        self._available = available
        self._chunks: list[_Chunk] = []
        # Releases come from the autograd engine's threads, and from the garbage collector, which may free a graph
        # (its handles) inside take() on the thread that holds the lock: reentrant.
        self._lock = threading.RLock()

    @property
    def reserved_bytes(self) -> int:
        return sum(c.size for c in self._chunks)

    def _allocate(self, size: int) -> torch.Tensor:
        """A uint8 buffer with `size` bytes from an aligned address. Pinned (page-locked) memory starts on a page;
        pageable memory from the C allocator need not start on a multiple of the alignment (glibc: 64 bytes), and
        then the buffer is taken one alignment longer, so the chunk can start at its first aligned byte."""
        buf = torch.empty(size, dtype=torch.uint8, pin_memory=self.pin)
        if buf.data_ptr() % _ARENA_ALIGN:
            del buf
            buf = torch.empty(size + _ARENA_ALIGN, dtype=torch.uint8, pin_memory=self.pin)
        return buf

    def take(self, nbytes: int) -> tuple[torch.Tensor, _Chunk]:
        need = -(-nbytes // _ARENA_ALIGN) * _ARENA_ALIGN
        with self._lock:
            chunk = next((c for c in self._chunks if c.free >= need), None)
            if chunk is None:
                size = self.chunk_bytes if need <= self.chunk_bytes else 1 << (need - 1).bit_length()
                if self.limit_bytes is not None and self.reserved_bytes + size > self.limit_bytes:
                    raise RuntimeError(
                        f"memory.activation_offload needs more than {(self.reserved_bytes + size) / 2**30:.2f} GiB of "
                        f"pinned host memory for one micro-batch's activations, over the {self.limit_bytes / 2**30:.2f}"
                        " GiB this process may hold (a share of the machine's RAM, split between the GPUs of the "
                        "node). Use gradient checkpointing (full keeps only each layer's input), "
                        "memory.activation_offload_boundaries_only, a shorter sequence or a smaller batch."
                    )
                available = self._available() if self.headroom_bytes is not None else None
                if available is not None and available - size < self.headroom_bytes:
                    raise RuntimeError(
                        f"memory.activation_offload needs another {size / 2**30:.2f} GiB of pinned host memory "
                        f"(it holds {self.reserved_bytes / 2**30:.2f} GiB), but the machine has "
                        f"{available / 2**30:.2f} GiB available and {self.headroom_bytes / 2**30:.2f} GiB must stay "
                        "free. Free host memory, or move less: gradient checkpointing (full keeps only each "
                        "layer's input), memory.activation_offload_boundaries_only, a shorter sequence or a "
                        "smaller batch."
                    )
                buf = self._allocate(size)
                # the allocation may have run the garbage collector, and with it release(): look again
                chunk = next((c for c in self._chunks if c.free >= need), None)
                if chunk is None:
                    chunk = _Chunk(buf, size)
                    self._chunks.append(chunk)
            # book the range before anything else allocates: a release() of this chunk's other tensors that the
            # garbage collector runs from here on finds it live and does not rewind it under the new view
            start = chunk.offset
            chunk.offset += need
            chunk.live += 1
            return chunk.buf[start : start + nbytes], chunk

    def release(self, chunk: _Chunk) -> None:
        with self._lock:
            chunk.live -= 1
            if chunk.live <= 0:
                chunk.live, chunk.offset = 0, chunk.start

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()


# ----------------------------------------------------------------------------------------------------------------
# Copies: a side CUDA stream, or synchronous copies (CPU; tests)
# ----------------------------------------------------------------------------------------------------------------


class StreamCopies:
    """Device <-> host copies on a side CUDA stream, ordered after the compute stream's work so far."""

    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.cuda.Stream(device)

    def to_host(self, src: torch.Tensor, dst: torch.Tensor):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            dst.copy_(src, non_blocking=True)
        return self.stream.record_event()

    def to_device(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for host, dev in pairs:
                dev.copy_(host, non_blocking=True)
        for _, dev in pairs:
            # dev belongs to the compute stream's pool: if it is freed before anything waited for the copy (a
            # prefetched tensor whose graph is dropped), the allocator must not reuse it while the copy runs
            dev.record_stream(self.stream)
        return self.stream.record_event()

    def wait(self, event) -> None:
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)


class SyncCopies:
    """Synchronous copies (no streams): the CPU path, and what the tests run."""

    def to_host(self, src: torch.Tensor, dst: torch.Tensor):
        dst.copy_(src)
        return None

    def to_device(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]):
        for host, dev in pairs:
            dev.copy_(host)
        return None

    def wait(self, event) -> None:
        return None


# ----------------------------------------------------------------------------------------------------------------
# The offloader
# ----------------------------------------------------------------------------------------------------------------


class _Saved:
    """An offloaded tensor: its host copy, and its device copy while fetched."""

    __slots__ = ("group", "host", "chunk", "arena", "dtype", "shape", "stride", "device", "dev", "event", "refs")
    __slots__ += ("__weakref__",)

    def __init__(self, group, host, chunk, arena, dtype, shape, stride, device):
        self.group, self.host, self.chunk, self.arena = group, host, chunk, arena
        self.dtype, self.shape, self.stride, self.device = dtype, shape, stride, device
        self.dev, self.event, self.refs = None, None, 1

    def __del__(self):
        arena = getattr(self, "arena", None)
        if arena is not None:
            try:
                arena.release(self.chunk)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass


def _layout(t: torch.Tensor) -> tuple[int, ...]:
    """The strides a copy of `t` keeps: its own when it is dense and non-overlapping (a permuted tensor: the copies
    are plain memcpys and the backward sees the layout the forward saved), contiguous otherwise."""
    if t.is_contiguous():
        return torch.empty(t.shape, device="meta").stride()
    kept = torch.empty_like(t, device="meta").stride()  # preserve_format keeps the strides of dense tensors only
    return tuple(t.stride()) if tuple(kept) == tuple(t.stride()) else torch.empty(t.shape, device="meta").stride()


def _tensors(*trees) -> list[torch.Tensor]:
    from torch.utils._pytree import tree_leaves

    return [x for x in tree_leaves(trees) if isinstance(x, torch.Tensor)]


class ActivationOffloader:
    """Moves the activations saved inside `layers` to host memory while their backward is not running (module doc).

    Use `saving()` around the forward (it may also cover the backward: SeCO runs both per chunk); the hooks on the
    layers stay until `close()`."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        device: torch.device | str,
        options: OffloadOptions | None = None,
        *,
        params: Iterable[torch.Tensor] = (),
        copies=None,
        arena: HostArena | None = None,
    ):
        self.options = options or OffloadOptions()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:  # tensors report an indexed device
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.layers = list(layers)
        self._index = {id(m): i for i, m in enumerate(self.layers)}
        self._first_kept = len(self.layers) - self.options.keep_last
        cuda = self.device.type == "cuda"
        self.copies = copies if copies is not None else (StreamCopies(self.device) if cuda else SyncCopies())
        self.arena = (
            arena
            if arena is not None
            else HostArena(
                pin=cuda, limit_bytes=host_limit_bytes(), headroom_bytes=host_headroom_bytes() if cuda else None
            )
        )
        # 2-D trained weights, by shape: a cast copy of one (fp32 masters under autocast, inside a compiled graph)
        # is not an activation
        shapes = {tuple(p.shape) for p in params if p.ndim == 2 and p.requires_grad}
        self._weight_shapes = shapes | {s[::-1] for s in shapes}
        self._group: int | None = None
        self._inputs: set[int] = set()
        self._phase = "idle"
        self._groups: dict[int, list] = {}
        self._pending: deque = deque()  # [group, [GPU tensors being copied], last event], oldest first
        self._seen: dict[int, tuple] = {}
        self._entered: set[int] = set()
        self.tensors = 0  # offloaded in the current pass
        self.bytes = 0
        self.last_pass_tensors = 0
        self.last_pass_bytes = 0
        enter, leave = self._enter, self._leave
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
            # compiled layers (model.compile) run these hooks eagerly: a graph break before and after each layer
            enter, leave = torch.compiler.disable(enter), torch.compiler.disable(leave)
        self._hooks = [m.register_forward_pre_hook(enter, with_kwargs=True) for m in self.layers]
        self._hooks += [m.register_forward_hook(leave) for m in self.layers]

    # -- context ---------------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def saving(self):
        """Offload what the forward inside this context saves."""
        self._new_pass()
        self._group = None  # a forward that raised inside a layer never reached the layer's exit hook
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield self

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._finalize(None)
        self.arena.clear()

    def stats(self) -> dict:
        return {
            "tensors": self.last_pass_tensors,
            "gb": self.last_pass_bytes / 2**30,
            "host_reserved_gb": self.arena.reserved_bytes / 2**30,
        }

    # -- layer boundaries --------------------------------------------------------------------------------------------

    def _enter(self, module, args, kwargs):
        if _in_backward() or not torch.is_grad_enabled():
            return None  # checkpointing's recompute, evaluation
        group = self._index.get(id(module))
        self._group = group
        if group is None:
            return None
        if self.options.boundaries_only:
            self._inputs = {t.untyped_storage().data_ptr() for t in _tensors(args, kwargs)}
        self._finalize(group - self.options.in_flight)
        return None

    def _leave(self, module, args, output):
        if not _in_backward():
            self._group = None
        return None

    # -- pack / unpack ---------------------------------------------------------------------------------------------

    def _new_pass(self) -> None:
        self._finalize(None)  # a pass that ended in an exception may have left copies in flight
        self.tensors = self.bytes = 0
        self._groups.clear()
        self._seen.clear()
        self._entered.clear()
        self._phase = "forward"

    def _finalize(self, before: int | None) -> None:
        """The compute stream waits for the copies of the layers before `before` (all: None); their GPU tensors are
        released."""
        while self._pending and (before is None or self._pending[0][0] < before):
            _, _, event = self._pending.popleft()
            self.copies.wait(event)

    def _is_weight(self, t: torch.Tensor) -> bool:
        base = t._base if t._base is not None else t
        if base.is_leaf:
            return base.requires_grad  # a parameter (or another trained leaf)
        fn = base.grad_fn
        if fn is not None and type(fn).__name__ == "ToCopyBackward0":  # an autocast copy of a parameter
            nxt = fn.next_functions
            if nxt and nxt[0][0] is not None and type(nxt[0][0]).__name__ == "AccumulateGrad":
                return True
        return not t.requires_grad and t.ndim == 2 and tuple(t.shape) in self._weight_shapes

    def _offloadable(self, t: torch.Tensor) -> bool:
        if type(t) is not torch.Tensor or t.layout != torch.strided or t.device != self.device:
            return False
        nbytes = t.numel() * t.element_size()
        if nbytes < max(self.options.min_bytes, 1) or nbytes > t.untyped_storage().nbytes():
            return False  # small, or an expanded view (its copy would be larger than what it frees)
        if self._is_weight(t):
            return False
        return not self.options.boundaries_only or t.untyped_storage().data_ptr() in self._inputs

    def _pack(self, t: torch.Tensor):
        if _in_backward():
            return t  # recompute or a custom backward that saves tensors: not part of the forward's schedule
        if self._phase == "backward":
            self._new_pass()  # the next forward after a backward (SeCO chunks, gradient accumulation)
        group = self._group
        if group is None or group >= self._first_kept or not self._offloadable(t):
            return t
        seen = self._seen.get(id(t))
        if seen is not None and seen[0]() is t:
            seen[1].refs += 1
            return seen[1]
        nbytes = t.numel() * t.element_size()
        buf, chunk = self.arena.take(nbytes)
        stride = _layout(t)
        host = buf.view(t.dtype).as_strided(t.shape, stride)
        event = self.copies.to_host(t, host)
        handle = _Saved(group, host, chunk, self.arena, t.dtype, t.shape, stride, t.device)
        if self._pending and self._pending[-1][0] == group:
            self._pending[-1][1].append(t)
            self._pending[-1][2] = event
        else:
            self._pending.append([group, [t], event])
        self._groups.setdefault(group, []).append(weakref.ref(handle))
        self._seen[id(t)] = (weakref.ref(t), handle)
        self.tensors += 1
        self.bytes += nbytes
        return handle

    def _fetch(self, handles: list[_Saved]) -> None:
        pairs = [(h.host, torch.empty_strided(h.shape, h.stride, dtype=h.dtype, device=h.device)) for h in handles]
        event = self.copies.to_device(pairs)
        for h, (_, dev) in zip(handles, pairs):
            h.dev, h.event = dev, event

    def _fetch_group(self, group: int) -> None:
        handles = [h for h in (r() for r in self._groups.get(group, ())) if h is not None and h.dev is None]
        if handles:
            self._fetch(handles)

    def _unpack(self, handle):
        if not isinstance(handle, _Saved):
            return handle
        if self._phase != "backward":
            self._finalize(None)  # the backward reads what the forward copied out
            self._phase = "backward"
            self._entered.clear()
            self._seen.clear()
            self.last_pass_tensors, self.last_pass_bytes = self.tensors, self.bytes
        group = handle.group
        if group not in self._entered:
            self._entered.add(group)
            for k in range(self.options.prefetch + 1):  # this layer, then the next ones the backward reaches
                self._fetch_group(group - k)
        if handle.dev is None:  # not in a fetched group, or unpacked again (retain_graph)
            self._fetch([handle])
        self.copies.wait(handle.event)
        dev = handle.dev
        handle.refs -= 1
        if handle.refs <= 0:
            handle.dev, handle.event = None, None  # the host copy stays until the graph is freed
        return dev


def describe(options: OffloadOptions, checkpointing: str, sac_hooks: bool) -> str:
    """One line for the log: what is offloaded under this checkpointing mode."""
    if checkpointing == "full":
        what = "each layer's input (full checkpointing saves nothing else)"
    elif checkpointing == "selective":
        what = (
            "each layer's input"
            if options.boundaries_only or not sac_hooks
            else "each layer's input and the outputs selective checkpointing keeps"
        )
    else:
        what = "every activation the layers save"
    return (
        f"activation offload: {what}, tensors >= {options.min_bytes / MiB:g} MiB; {options.in_flight} layer(s) in "
        f"flight, {options.prefetch} prefetched; the last {options.keep_last} layer(s) stay on the GPU"
    )
