"""Weights as checksummed buckets, and the link that carries them between GPUs.

A verified push (rollout.push_verify, and every push over rollout.push_link nccl / gloo) packs the weights into
byte buckets of about rollout.push_bucket_mb: each tensor's bytes (cast to the wire dtype, rollout.push_dtype) at an
8-byte-aligned offset of one bucket, described by a manifest [(name, dtype, shape, bucket, offset)]. Every bucket
carries a checksum of its bytes, computed where the bucket lives (on the GPU for a GPU bucket) and compared by the
engine worker after the bucket has arrived, before anything is applied. The worker then reads each tensor as a view of
its bucket (no copy) and applies them all at once (palingenesis.rl.pool.worker).

Buckets travel inside a frame of the TCP stream (from pinned host memory), or over a Link: a process group of two
ranks, the trainer (rank 0) and one engine worker (rank 1), built on torch.distributed's NCCL backend (GPU to GPU:
NVLink, shared memory, InfiniBand / RoCE, without a copy through the trainer's host memory) or its Gloo backend (CPU
tensors: the protocol's tests). One group per engine, never one for all: a collective over a group fails for every
rank when one rank dies, so a lost engine would end every other engine's pushes. The group is built from the
backend's own classes on a store the trainer hosts, next to (not instead of) the default process group torchrun gave
the trainer's ranks.

In-flight weight updates over NCCL are what TRL's AsyncGRPOTrainer does through vLLM's weight-transfer engine
(vllm/distributed/weight_transfer, packed broadcasts); this is an independent implementation for this library's
engine workers, with the checks and the fallback the pool needs. The NCCL path has not run on a GPU yet
(scripts/gpu_checks/async_rl.sh covers the protocol over Gloo and TCP; two GPUs are needed for NCCL).
"""

from __future__ import annotations

import datetime
import logging
import uuid
from dataclasses import dataclass, field

import torch

from palingenesis.rl.pool.protocol import DTYPES

logger = logging.getLogger(__name__)

ALIGN = 8  # every tensor starts at a multiple of this in its bucket (a view of any dtype needs an aligned offset)
_SUM_WORDS = 1 << 22  # int64 words checksummed at a time (32 MiB of temporaries)
_MAGIC = 0x7067732D6C696E6B  # the handshake word of a new link


def checksum(buffer: torch.Tensor) -> torch.Tensor:
    """An order-sensitive checksum of a 1-d uint8 buffer as an int64 tensor [2] on its device (no host sync):
    the wrapping sum of its bytes read as int64 words, and the wrapping sum of word x (1 + its index). Two buffers
    that differ in one bit, or hold the same words in another order, differ in it."""
    words = buffer.numel() // 8
    total = torch.zeros(2, dtype=torch.int64, device=buffer.device)
    body = buffer[: words * 8].view(torch.int64)
    for start in range(0, words, _SUM_WORDS):
        part = body[start : start + _SUM_WORDS]
        index = torch.arange(start + 1, start + 1 + part.numel(), dtype=torch.int64, device=buffer.device)
        total[0] += part.sum()
        total[1] += (part * index).sum()
    tail = buffer[words * 8 :]
    if tail.numel():
        total[0] += tail.to(torch.int64).sum()
        total[1] += (tail.to(torch.int64) * torch.arange(1, tail.numel() + 1, device=buffer.device)).sum() * (words + 1)
    return total


@dataclass
class Buckets:
    """A snapshot of the weights as buckets. buffers: 1-d uint8 tensors (host or device); manifest: per tensor
    [name, dtype name, shape, bucket, offset]; sums: per bucket, the int64 [2] checksum (on the bucket's device)."""

    buffers: list[torch.Tensor] = field(default_factory=list)
    manifest: list[list] = field(default_factory=list)
    sums: list[torch.Tensor] = field(default_factory=list)
    _host: list[torch.Tensor] | None = None

    def spec(self, via: str) -> dict:
        """What the worker needs to receive and check the snapshot (one host sync for the checksums)."""
        sums = torch.stack(self.sums).tolist() if self.sums else []
        return {"sizes": [b.numel() for b in self.buffers], "sums": sums, "manifest": self.manifest, "via": via}

    def host(self) -> dict[str, torch.Tensor]:
        """The buckets as host tensors named bucket/<i> (copied once if they live on a device)."""
        if self._host is None:
            self._host = [b if b.device.type == "cpu" else b.cpu() for b in self.buffers]
        return {f"bucket/{i}": b for i, b in enumerate(self._host)}

    @property
    def nbytes(self) -> int:
        return sum(b.numel() for b in self.buffers)


def pack(named_tensors, bucket_bytes: int, allocate, wire_dtype: torch.dtype | None = None) -> Buckets:
    """Pack (name, tensor) pairs into buckets of about `bucket_bytes` (a larger tensor gets a bucket of its own).
    `allocate(index, nbytes)` returns the 1-d uint8 buffer of bucket `index` (reused between pushes by the caller;
    on the device the buckets are sent from). Floating-point tensors are cast to `wire_dtype` as they are copied
    (one copy: cast and device transfer together). The pairs may be a generator of gathered parameters: at most
    one bucket's tensors are held at a time."""
    out = Buckets()
    pending: list[tuple[str, torch.Tensor, torch.dtype, int]] = []
    used = 0

    def flush() -> None:
        nonlocal pending, used
        if not pending:
            return
        index = len(out.buffers)
        buffer = allocate(index, used)
        for name, tensor, dtype, offset in pending:
            nbytes = tensor.numel() * dtype.itemsize
            if nbytes:
                buffer[offset : offset + nbytes].view(dtype).view(tensor.shape).copy_(tensor.detach())
            out.manifest.append([name, str(dtype).replace("torch.", ""), list(tensor.shape), index, offset])
        out.buffers.append(buffer)
        out.sums.append(checksum(buffer))
        pending, used = [], 0

    for name, tensor in named_tensors:
        dtype = wire_dtype if wire_dtype is not None and tensor.is_floating_point() else tensor.dtype
        if str(dtype).replace("torch.", "") not in DTYPES:
            raise ValueError(f"{name}: tensors of dtype {dtype} cannot be sent")
        nbytes = -(-tensor.numel() * dtype.itemsize // ALIGN) * ALIGN
        if pending and used + nbytes > bucket_bytes:
            flush()
        pending.append((name, tensor, dtype, used))
        used += nbytes
    flush()
    return out


def unpack(spec: dict, buffers: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors of a snapshot as views of its buckets."""
    out = {}
    for name, kind, shape, index, offset in spec["manifest"]:
        dtype = DTYPES[kind]
        count = 1
        for n in shape:
            count *= n
        nbytes = count * dtype.itemsize
        if nbytes:
            out[name] = buffers[index][offset : offset + nbytes].view(dtype).view(shape)
        else:
            out[name] = torch.empty(shape, dtype=dtype)
    return out


def verify(spec: dict, buffers: list[torch.Tensor]) -> str:
    """ "" when every bucket has the size and the checksum the sender announced, else what differs."""
    if len(buffers) != len(spec["sizes"]):
        return f"{len(buffers)} buckets arrived, {len(spec['sizes'])} were announced"
    for i, (buffer, size, expected) in enumerate(zip(buffers, spec["sizes"], spec["sums"])):
        if buffer.numel() != size:
            return f"bucket {i} has {buffer.numel()} bytes, {size} were announced"
        if checksum(buffer).tolist() != list(expected):
            return f"bucket {i} of {len(buffers)} arrived with another checksum than it was sent with"
    return ""


# ---------------------------------------------------------------------- links


class LinkError(RuntimeError):
    """A link could not be built, or a transfer over it failed: the push goes over TCP instead."""


class Link:
    """One end of a two-rank group. `backend`: "nccl" or "gloo"; `store`: (host, port) of the trainer's store, or
    None on the trainer's side (which hosts it: `Link.host_store`); `prefix`: this link's own key space in it.
    Building blocks until the other end has joined (Gloo), or until the handshake (NCCL builds its communicator at
    the first transfer): build both ends at the same time."""

    def __init__(self, backend: str, rank: int, store, prefix: str, timeout: float = 120.0, device: str = ""):
        import torch.distributed as dist
        from torch._C import _distributed_c10d as c10d

        self.backend, self.rank, self.timeout = backend, rank, timeout
        self.device = torch.device(device or ("cuda" if backend == "nccl" else "cpu"))
        self.scratch: torch.Tensor | None = None  # (nccl) a device buffer host tensors are received through
        limit = datetime.timedelta(seconds=timeout)
        try:
            if not isinstance(store, dist.Store):
                store = dist.TCPStore(store[0], int(store[1]), None, False, timeout=limit)
            scoped = dist.PrefixStore(prefix, store)
            if backend == "gloo":
                group = c10d.ProcessGroupGloo(scoped, rank, 2, limit)
                kind = c10d.ProcessGroup.BackendType.GLOO
            elif backend == "nccl":
                try:
                    options = c10d.ProcessGroupNCCL.Options()
                    options._timeout = limit
                    group = c10d.ProcessGroupNCCL(scoped, rank, 2, options)
                except TypeError:  # older torch: the timeout itself
                    group = c10d.ProcessGroupNCCL(scoped, rank, 2, limit)
                kind = c10d.ProcessGroup.BackendType.NCCL
            else:
                raise ValueError(f"unknown link backend {backend!r}")
            self.group = c10d.ProcessGroup(scoped, rank, 2)
            self.group._set_default_backend(kind)
            self.group._register_backend(self.device, kind, group)
            self._backend, self._store = group, store  # (kept alive with the link)
        except Exception as error:  # noqa: BLE001 — every way a backend refuses is a reason to use TCP
            raise LinkError(
                f"{backend} link (rank {rank}) could not be built: {type(error).__name__}: {error}"
            ) from error

    @staticmethod
    def host_store(timeout: float = 120.0):
        """The store the trainer hosts for its links (any free port; several links share it under their prefixes)."""
        import torch.distributed as dist

        limit = datetime.timedelta(seconds=timeout)
        return dist.TCPStore("0.0.0.0", 0, None, True, timeout=limit, wait_for_workers=False, multi_tenant=True)

    @staticmethod
    def new_prefix() -> str:
        return f"pgs-link/{uuid.uuid4().hex}/"

    def _transfer(self, buffer: torch.Tensor) -> None:
        try:
            work = self.group.broadcast(buffer, 0)
            if not work.wait(datetime.timedelta(seconds=self.timeout)):
                raise TimeoutError(f"no answer in {self.timeout:g} s")
        except Exception as error:  # noqa: BLE001
            raise LinkError(f"{self.backend} link transfer failed: {type(error).__name__}: {error}") from error

    def send(self, buffer: torch.Tensor) -> None:
        """Rank 0: send a 1-d uint8 buffer (on the link's device) to the other end."""
        self._transfer(buffer if buffer.device == self.device else buffer.to(self.device))

    def receive(self, buffer: torch.Tensor) -> None:
        """Rank 1: fill `buffer` (1-d uint8). A host buffer on an NCCL link is received through a device buffer of
        its size (kept between buckets) and copied down."""
        if buffer.device.type == self.device.type:
            self._transfer(buffer)
            return
        if self.scratch is None or self.scratch.numel() < buffer.numel():
            self.scratch = torch.empty(buffer.numel(), dtype=torch.uint8, device=self.device)
        through = self.scratch[: buffer.numel()]
        self._transfer(through)
        buffer.copy_(through)

    def handshake(self) -> None:
        """One word each way round: builds the communicator and proves both ends are on the same link."""
        word = torch.tensor([_MAGIC if self.rank == 0 else 0], dtype=torch.int64, device=self.device)
        self._transfer(word.view(torch.uint8))
        if int(word.item()) != _MAGIC:
            raise LinkError(f"{self.backend} link handshake failed")

    def close(self) -> None:
        self.scratch = None
        for target in (getattr(self, "_backend", None),):
            shutdown = getattr(target, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 — best effort
                    pass
        self.group = self._backend = None
