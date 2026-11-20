"""The wire format between the trainer and an engine worker: one TCP stream of frames.

A frame is a 4-byte little-endian length, a JSON header of that length, then the raw bytes of the tensors the
header lists under "tensors" ([{name, dtype, shape, nbytes}], in order). Tensor bytes are the host's own
representation (native byte order: both ends must be of the same endianness, which every platform this runs on
is). Tensors are sent straight from host memory (`pack` exposes each tensor's storage as a buffer, a chunk at a
time) and received into the buffers of a Staging area, which keeps them (pinned, with a GPU) from one push to
the next.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Awaitable, Callable
from typing import Any

import torch

_NAMES = (
    "bfloat16 float16 float32 float64 float8_e4m3fn float8_e5m2 int8 uint8 int16 uint16 int32 uint32 int64 uint64 bool"
).split()
DTYPES = {name: getattr(torch, name) for name in _NAMES if hasattr(torch, name)}
CHUNK = 1 << 22  # bytes handed to, or read from, the socket at a time


def pack(tensors: dict[str, torch.Tensor]) -> tuple[list[dict], list[memoryview]]:
    """(the header entries, the buffers) of host tensors, without copying contiguous ones."""
    entries, buffers = [], []
    for name, tensor in tensors.items():
        kind = str(tensor.dtype).replace("torch.", "")
        if kind not in DTYPES:
            raise ValueError(f"{name}: tensors of dtype {kind} cannot be sent (supported: {', '.join(DTYPES)})")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name}: stage tensors on the host before sending them")
        flat = tensor.detach().contiguous().reshape(-1)
        raw = flat.view(torch.uint8) if flat.numel() else flat.new_empty(0, dtype=torch.uint8)
        entries.append({"name": name, "dtype": kind, "shape": list(tensor.shape), "nbytes": raw.numel()})
        buffers.append(memoryview(raw.numpy()))
    return entries, buffers


def model_signature(config) -> dict:
    """What identifies a model's architecture to both ends of a pool, from its transformers config (the language
    model's, for a model with towers): two processes that disagree here cannot exchange weights by name."""
    inner = getattr(config, "text_config", None) or config
    keys = ("model_type", "vocab_size", "hidden_size", "num_hidden_layers")
    return {key: getattr(inner, key, None) for key in keys}


async def send(writer: asyncio.StreamWriter, header: dict, tensors: dict[str, torch.Tensor] | None = None) -> int:
    """Write one frame; returns the payload bytes."""
    entries, buffers = pack(tensors or {})
    head = json.dumps({**header, "tensors": entries}).encode()
    writer.write(struct.pack("<I", len(head)) + head)
    total = 0
    for buffer in buffers:
        for start in range(0, len(buffer), CHUNK):
            writer.write(buffer[start : start + CHUNK])
            await writer.drain()  # back-pressure: at most a chunk of a tensor is ever copied into the socket's buffer
        total += len(buffer)
    await writer.drain()
    return total


class Staging:
    """Host buffers a push is received into (or staged in before it is sent), kept between pushes so that a
    tensor of the same name and size reuses its memory. With `pin` the memory is page-locked, which is what makes
    the copy to or from a GPU fast and asynchronous."""

    def __init__(self, pin: bool = False):
        self.pin = pin and torch.cuda.is_available()
        self.buffers: dict[str, torch.Tensor] = {}

    def buffer(self, name: str, dtype: torch.dtype, shape: list[int] | torch.Size) -> torch.Tensor:
        have = self.buffers.get(name)
        if have is None or have.dtype != dtype or tuple(have.shape) != tuple(shape):
            have = torch.empty(tuple(shape), dtype=dtype, pin_memory=self.pin)
            self.buffers[name] = have
        return have

    def stage(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Copies of `tensors` in this staging area's buffers (a snapshot the sender owns)."""
        out = {}
        for name, tensor in tensors.items():
            buffer = self.buffer(name, tensor.dtype, tensor.shape)
            buffer.copy_(tensor.detach(), non_blocking=self.pin)
            out[name] = buffer
        if self.pin:
            torch.cuda.synchronize()
        return out

    def release(self, names) -> None:
        """Forget the buffers of `names` (their memory is freed once nothing else holds them)."""
        for name in names:
            self.buffers.pop(name, None)


async def receive(
    reader: asyncio.StreamReader,
    staging: Staging | None = None,
    gate: Callable[[dict], Awaitable[None]] | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read one frame: (header, tensors). Tensors land in `staging`'s buffers when given (they are then valid
    until a frame with tensors of the same names is received into it). `gate(header)` is awaited before each
    chunk of a tensor is read (tests hold a push there). Raises asyncio.IncompleteReadError when the stream ends;
    inside a frame, the error carries the frame's header as `.frame` ({} when the header itself was cut)."""
    start = await reader.readexactly(4)
    try:
        header = json.loads(await reader.readexactly(struct.unpack("<I", start)[0]))
    except asyncio.IncompleteReadError as error:
        error.frame = {}  # the stream ended inside a frame's header
        raise
    tensors: dict[str, Any] = {}
    for entry in header.pop("tensors", []):
        dtype = DTYPES[entry["dtype"]]
        if staging is not None:
            tensor = staging.buffer(entry["name"], dtype, entry["shape"])
        else:
            tensor = torch.empty(tuple(entry["shape"]), dtype=dtype)
        view = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()) if entry["nbytes"] else memoryview(b"")
        done = 0
        while done < entry["nbytes"]:
            step = min(CHUNK, entry["nbytes"] - done)
            try:
                if gate is not None:
                    await gate(header)
                data = await reader.readexactly(step)
            except asyncio.IncompleteReadError as error:
                error.frame = header  # the stream ended inside this frame
                raise
            view[done : done + step] = data
            done += step
        tensors[entry["name"]] = tensor
    return header, tensors
