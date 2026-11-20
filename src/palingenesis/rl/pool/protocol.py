"""The wire format between the trainer and an engine worker: one TCP stream of frames.

A frame is a 4-byte little-endian length, a JSON header of that length, then the raw bytes of the tensors the
header lists under "tensors" ([{name, dtype, shape, nbytes}], in order). Tensors are sent straight from host
memory (`pack` exposes each tensor's own storage as a buffer) and received into the buffers of a Staging area,
which keeps them (pinned, with a GPU) from one push to the next.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

import torch

DTYPES = {str(d).replace("torch.", ""): d for d in (torch.bfloat16, torch.float16, torch.float32, torch.float64)}
DTYPES.update(
    {
        str(d).replace("torch.", ""): d
        for d in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool)
    }
)
CHUNK = 1 << 20  # bytes per read when the stream is throttled (tests: a slow link)
SEND_CHUNK = 1 << 22  # bytes handed to the socket at a time


def pack(tensors: dict[str, torch.Tensor]) -> tuple[list[dict], list[memoryview]]:
    """(the header entries, the buffers) of host tensors, without copying them."""
    entries, buffers = [], []
    for name, tensor in tensors.items():
        tensor = tensor.detach().contiguous()
        if tensor.device.type != "cpu":
            raise ValueError(f"{name}: stage tensors on the host before sending them")
        raw = tensor.reshape(-1).view(torch.uint8) if tensor.numel() else tensor.reshape(-1).to(torch.uint8)
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "nbytes": raw.numel(),
            }
        )
        buffers.append(memoryview(raw.numpy()))
    return entries, buffers


async def send(writer: asyncio.StreamWriter, header: dict, tensors: dict[str, torch.Tensor] | None = None) -> int:
    """Write one frame; returns the payload bytes."""
    entries, buffers = pack(tensors or {})
    head = json.dumps({**header, "tensors": entries}).encode()
    writer.write(struct.pack("<I", len(head)) + head)
    total = 0
    for buffer in buffers:
        for start in range(0, len(buffer), SEND_CHUNK):
            writer.write(buffer[start : start + SEND_CHUNK])
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


async def receive(
    reader: asyncio.StreamReader, staging: Staging | None = None, throttle: float = 0.0
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read one frame: (header, tensors). Tensors land in `staging`'s buffers when given (they are then valid
    until the next frame received into it). `throttle`: bytes per second at most (0: unlimited). Raises
    asyncio.IncompleteReadError when the stream ends, also in the middle of a frame."""
    (size,) = struct.unpack("<I", await reader.readexactly(4))
    header = json.loads(await reader.readexactly(size))
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
                data = await reader.readexactly(step)
            except asyncio.IncompleteReadError as error:
                error.frame = header  # the stream ended inside this frame
                raise
            view[done : done + step] = data
            done += step
            if throttle:
                await asyncio.sleep(step / throttle)
        tensors[entry["name"]] = tensor
    return header, tensors
