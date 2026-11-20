"""The trainer's side of an engine pool: generation requests spread over the engines, weights pushed to them
without waiting.

RemotePool is the trainer's rollout engine (the RolloutEngine protocol of palingenesis.opd.rollout) when the
engines are other processes. `update_weights` copies what has to travel into host staging buffers, which is the
only part done under the trainer's weights lock, and returns; the bytes go out on the pool's own thread while the
trainer runs its next step and the engines keep generating with the weights they have. Each engine applies a push
when all of it has arrived and tags what it generates from then on with the new version, so the version on every
turn is the version that sampled it, and rollout.max_staleness decides what is still trained. With
max_staleness 0 a push waits for every engine instead (on-policy).

Two staging areas alternate: a new snapshot is written to the one no push is sending from. When both are still
being sent, the snapshot is skipped and the next sync tries again with newer weights (a push that could not start
is never the reason a trainer waits). An engine that is still receiving version v when v+1 is staged receives v+1
next: a push is never interrupted, and the newest staged version is the one that follows.

A request goes to the engine with the fewest requests in flight. There is no prefix affinity: every push resets
an engine's prefix cache, and an engine's own batch is where the turns of a session meet their prefix.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import threading
import time

from palingenesis.rl.generation import Generation
from palingenesis.rl.pool.protocol import Staging, receive, send

logger = logging.getLogger(__name__)


class _Engine:
    """One worker: a connection for requests and one for weights (a long push must not hold up requests)."""

    def __init__(self, address: str):
        host, _, port = address.rpartition(":")
        self.address, self.host, self.port = address, host or "127.0.0.1", int(port)
        self.reader = self.writer = self.weights_reader = self.weights_writer = None
        self.waiting: dict[int, asyncio.Future] = {}
        self.write_lock: asyncio.Lock | None = None
        self.push_lock: asyncio.Lock | None = None
        self.applied = -1  # the newest version the worker confirmed
        self.queued: tuple | None = None  # the push that follows the one being sent (the newest staged)
        self.listener: asyncio.Task | None = None

    @property
    def load(self) -> int:
        return len(self.waiting)


class RemotePool:
    """`addresses`: "host:port" of every engine worker this process generates with. `push`: whether this process
    sends the weights (rank 0 of a multi-rank trainer does, for every rank's engines: `push_to`). `blocking`: a
    weight update returns only when every engine has applied it."""

    def __init__(
        self,
        addresses: list[str],
        push_to: list[str] | None = None,
        blocking: bool = False,
        connect_timeout: float = 600.0,
        pin: bool = True,
    ):
        if not addresses:
            raise ValueError("rollout.engines is empty: list the engine workers as host:port")
        self.engines = [_Engine(a) for a in addresses]
        known = {e.address: e for e in self.engines}
        self.targets = [known.get(a) or _Engine(a) for a in (addresses if push_to is None else push_to)]
        self.blocking = blocking
        # the newest version handed to the pool (PublishedWeights compares with it). -1: nothing yet, so the first
        # sync always sends: a worker may hold another run's weights
        self.version = -1
        self.payload = None  # callable -> (header fields, tensors) of the current weights (LoRA pushes), else full
        self.first = True  # the next push is this trainer's first: engines get everything they need to keep
        self.staging = [Staging(pin), Staging(pin)]
        self.sending = [0, 0]  # pushes still being sent from each staging area
        self.ids = itertools.count()
        self.stats = {"pushes": 0, "skipped": 0, "bytes": 0, "seconds": 0.0, "stage_seconds": 0.0}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="pgs-engine-pool", daemon=True)
        self.thread.start()
        self._call(self._connect(connect_timeout))

    def _call(self, coroutine, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    async def _connect(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        everyone = {id(e): e for e in [*self.engines, *self.targets]}.values()
        for engine in everyone:
            while True:
                try:
                    if engine in self.engines:
                        engine.reader, engine.writer = await asyncio.open_connection(engine.host, engine.port)
                    if engine in self.targets:
                        engine.weights_reader, engine.weights_writer = await asyncio.open_connection(
                            engine.host, engine.port
                        )
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise ConnectionError(
                            f"engine worker {engine.address} did not answer in {timeout:g} s"
                        ) from None
                    await asyncio.sleep(0.5)
            engine.write_lock, engine.push_lock = asyncio.Lock(), asyncio.Lock()
            if engine.reader is not None:
                engine.listener = asyncio.ensure_future(self._listen(engine))

    async def _listen(self, engine: _Engine) -> None:
        try:
            while True:
                header, _ = await receive(engine.reader)
                future = engine.waiting.pop(header.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(header)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError) as error:
            for future in engine.waiting.values():
                if not future.done():
                    future.set_exception(ConnectionError(f"engine worker {engine.address} went away ({error!r})"))
            engine.waiting.clear()

    # ------------------------------------------------------------- generation

    def make_client(self, merge: bool = False, adapter: str = "") -> "RemoteClient":
        return RemoteClient(self, adapter)

    async def _request(self, prompt_ids, max_new_tokens, temperature, adapter: str) -> Generation:
        engine = min(self.engines, key=lambda e: e.load)
        number = next(self.ids)
        future = self.loop.create_future()
        engine.waiting[number] = future
        header = {
            "op": "generate",
            "id": number,
            "prompt": list(prompt_ids),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "adapter": adapter,
        }
        async with engine.write_lock:
            await send(engine.writer, header)
        reply = await future
        if "error" in reply:
            raise RuntimeError(f"engine worker {engine.address}: {reply['error']}")
        return Generation(reply["ids"], reply["logprobs"], reply["finish"], reply["version"], reply["first_version"])

    # ---------------------------------------------------------------- weights

    def update_weights(self, named_tensors, version: int) -> None:
        """Stage the weights of `version` and start sending them to every target engine; returns at once (see
        the module docstring), or when every engine applied them if the pool is blocking. A process that does not
        push (ranks other than 0) only records the version."""
        if not self.targets:
            for _ in named_tensors or ():  # (gathered parameters: every rank must iterate them)
                pass
            self.version = version
            return
        free = [i for i in (0, 1) if not self.sending[i]]
        if not free:
            for _ in named_tensors or ():
                pass
            self.stats["skipped"] += 1  # both snapshots still travelling: the next sync brings newer weights
            return
        start = time.perf_counter()
        if self.payload is not None:  # LoRA: [(header fields, tensors), ...], the base first on the first push
            for _ in named_tensors or ():
                pass
            frames = self.payload(self.first)
        else:
            frames = [({"mode": "full"}, dict(named_tensors))]
        area = self.staging[free[0]]
        frames = [({"op": "weights", "version": version, **fields}, area.stage(tensors)) for fields, tensors in frames]
        self.stats["stage_seconds"] += time.perf_counter() - start
        self.first = False
        if not self.blocking:
            self.version = version
        self.sending[free[0]] += len(self.targets)
        futures = [
            asyncio.run_coroutine_threadsafe(self._push(engine, free[0], frames), self.loop) for engine in self.targets
        ]
        if self.blocking:
            for future in futures:
                future.result()
            self.version = version  # (after the engines have it: pool/lag reads 0 between updates)

    async def _push(self, engine: _Engine, area: int, frames: list) -> None:
        if engine.push_lock.locked():  # a push is on its way to this engine: this one follows it, unless a newer
            previous, engine.queued = engine.queued, (area, frames)  # one is staged before it starts
            if previous is not None:
                self.sending[previous[0]] -= 1
            return
        async with engine.push_lock:
            job: tuple | None = (area, frames)
            while job is not None:
                area, frames = job
                start = time.perf_counter()
                try:
                    sent = 0
                    for header, staged in frames:
                        sent += await send(engine.weights_writer, header, staged)
                        reply, _ = await receive(engine.weights_reader)
                    engine.applied = max(engine.applied, int(reply["version"]))
                    self.stats["pushes"] += 1
                    self.stats["bytes"] += sent
                    self.stats["seconds"] += time.perf_counter() - start
                except (ConnectionError, asyncio.IncompleteReadError) as error:
                    logger.warning("weights for %s were not delivered (%r)", engine.address, error)
                finally:
                    self.sending[area] -= 1
                job, engine.queued = engine.queued, None

    def applied(self) -> list[int]:
        """The newest version each target engine confirmed."""
        return [engine.applied for engine in self.targets]

    def metrics(self) -> dict[str, float]:
        """pool/lag: versions the slowest engine is behind the newest staged one; pool/push_s and pool/push_mb:
        means per delivered push; pool/skipped: snapshots not staged because two were still travelling."""
        pushes = max(self.stats["pushes"], 1)
        out = {
            "pool/push_s": self.stats["seconds"] / pushes,
            "pool/push_mb": self.stats["bytes"] / pushes / 2**20,
            "pool/skipped": float(self.stats["skipped"]),
        }
        if self.targets:
            out["pool/lag"] = float(self.version - min(self.applied()))
        return out

    def state(self) -> list[dict]:
        """Each generating engine's own state (version, pushes applied, pushes dropped)."""
        return self._call(self._states())

    async def _states(self) -> list[dict]:
        out = []
        for engine in self.engines:
            future = self.loop.create_future()
            engine.waiting[None] = future
            async with engine.write_lock:
                await send(engine.writer, {"op": "state"})
            out.append(await future)
        return out

    def sleep(self) -> None:
        pass

    def wake(self) -> None:
        pass

    def close(self, shutdown: bool = False) -> None:
        async def finish():
            for engine in {id(e): e for e in [*self.engines, *self.targets]}.values():
                if engine.listener is not None:
                    engine.listener.cancel()
                for writer in (engine.writer, engine.weights_writer):
                    if writer is not None:
                        if shutdown and writer is engine.writer:
                            with contextlib.suppress(Exception):
                                await send(writer, {"op": "shutdown"})
                        writer.close()

        with contextlib.suppress(Exception):
            self._call(finish(), 10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


class RemoteClient:
    """What the rollout pipeline generates through (the interface of generation.GenerationClient): one request per
    assistant turn, each resolved when its engine finishes it. `adapter`: the LoRA adapter the turns use."""

    streaming = True

    def __init__(self, pool: RemotePool, adapter: str = ""):
        self.pool, self.adapter = pool, adapter
        self.merge = False
        self.calls = 0
        self.active = 0
        self.timeline: list[tuple[float, int, int]] = []
        self.engine_lock = threading.Lock()  # nothing to pause: the engines apply weights between their own steps

    def enter(self) -> None:
        self.active += 1

    def exit(self) -> None:
        self.active -= 1

    def paused(self):
        return self.engine_lock

    async def generate(self, prompt_ids, max_new_tokens, temperature, images: list | None = None) -> Generation:
        if images:
            raise NotImplementedError("remote engines do not take images yet")
        self.calls += 1
        coroutine = self.pool._request(prompt_ids, max_new_tokens, temperature, self.adapter)
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, self.pool.loop))

    def close(self) -> None:
        pass
