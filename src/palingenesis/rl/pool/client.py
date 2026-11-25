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

The first update of a run waits for the engines whatever max_staleness is, and its `hello` marks what a worker
held before as another run's (palingenesis.rl.pool.worker): a worker that outlived a run samples nothing that the
new run would train.

Losing an engine. A worker whose connection ends is not sent anything more. Requests it was running are sent
again to the other engines of this process, so no rollout waits for it (a rank whose only engine is gone fails
its rollouts at once); the run itself ends at the next weight update with an error that names the worker (an engine that missed updates would otherwise keep sampling an old policy, and a
restarted one holds none of this run's weights). Restart the worker and resume from the last checkpoint.

Keeping the run (rollout.min_engines n > 0). While at least n engines are alive, losing one does not end the run:
it is sent nothing more, its requests go to the others, and every rollout.reconnect_s seconds the pool tries its
address again. A worker that answers (restarted, or back after a network cut) is greeted as a new one: it gets
no request until a push has been applied, which the next update sends.

Verified pushes (rollout.push_verify, rollout.push_link; full weights only). The snapshot is packed into checksummed
buckets (palingenesis.rl.pool.buckets), cast to rollout.push_dtype as they are copied. The engine applies a push
only when every bucket arrived intact, and restores its weights if applying fails (palingenesis.rl.pool.worker). A
rejected push is sent once more; an engine that rejects three in a row ends the run, otherwise it simply stays on
the version it has, and the staleness bound decides what is trained. Over a link (nccl: GPU to GPU) the buckets
live on the trainer's device (one snapshot, 2 bytes per parameter in bf16) and are sent to each engine over its own
two-rank group; an engine whose link fails is served over TCP from then on, starting with the push that failed.

A request goes to the engine with the fewest requests in flight. There is no prefix affinity: every push resets
an engine's prefix cache, and an engine's own batch is where the turns of a session meet their prefix. With
rollout.concurrency kv (`concurrency`), each engine has a cap on its requests in flight, moved with the load its
worker reports in `state` (palingenesis.rl.concurrency), and a request goes to the engine with the most room,
waiting while every engine is at its cap.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import itertools
import logging
import threading
import time

import torch

from palingenesis.opd.rollout import Candidates
from palingenesis.rl.concurrency import AdmissionGate, ConcurrencySettings, EngineLoad
from palingenesis.rl.generation import Generation
from palingenesis.rl.pool import buckets as bucketed
from palingenesis.rl.pool.protocol import Staging, receive, send

# identity settings a worker may exceed (it serves adapters up to its LoRA rank, log-probs up to its maximum)
_AT_LEAST = ("lora_rank", "max_logprobs")

logger = logging.getLogger(__name__)


class EngineLost(ConnectionError):
    """The connection to an engine worker ended."""


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
        self.dead = ""  # why nothing more is sent to this worker
        self.confirmed = False  # the worker holds weights this run sent it
        self.link: bucketed.Link | None = None  # rollout.push_link nccl / gloo: this engine's own two-rank group
        self.rejections = 0  # verified pushes this engine rejected in a row
        self.retry: asyncio.Task | None = None  # rollout.min_engines: reconnecting

    @property
    def load(self) -> int:
        return len(self.waiting)


class RemotePool:
    """`addresses`: "host:port" of every engine worker this process generates with. `push_to`: the workers this
    process sends the weights to (rank 0 of a multi-rank trainer: every rank's engines; the other ranks: none).
    `blocking`: a weight update returns only when the engines have applied it (the ones this process sends to,
    and the ones it generates with), or fails after `push_timeout` seconds. `identity`: this trainer's model
    signature, stop tokens, chat-template arguments, context length and head precision; a worker that was started
    with other values is refused. `concurrency`: rollout.concurrency kv's settings (None: no cap)."""

    def __init__(
        self,
        addresses: list[str],
        push_to: list[str] | None = None,
        blocking: bool = False,
        connect_timeout: float = 600.0,
        pin: bool = True,
        identity: dict | None = None,
        push_timeout: float = 1800.0,
        concurrency: ConcurrencySettings | None = None,
        link: str = "tcp",
        verify: bool = False,
        bucket_mb: int = 512,
        wire_dtype: str = "",
        min_engines: int = 0,
        reconnect_s: float = 10.0,
        link_timeout: float = 120.0,
    ):
        if not addresses:
            raise ValueError("rollout.engines is empty: list the engine workers as host:port")
        self.engines = [_Engine(a) for a in addresses]
        known = {e.address: e for e in self.engines}
        self.targets = [known.get(a) or _Engine(a) for a in (addresses if push_to is None else push_to)]
        self.blocking, self.push_timeout = blocking, push_timeout
        self.identity = dict(identity or {})
        # the newest version handed to the pool (PublishedWeights compares with it). -1: nothing yet, so the first
        # sync always sends: a worker may hold another run's weights
        self.version = -1
        self.payload = None  # callable(first) -> [(header fields, tensors), ...] (LoRA pushes), else full weights
        self.first = True  # the next push is this trainer's first: engines get everything they need to keep
        self.started = False  # the first update was confirmed by the engines (it always waits for them)
        self.staging = [Staging(pin), Staging(pin)]
        self.sending = [0, 0]  # pushes still being sent from each staging area (under `counts`)
        self.once: list[list[str]] = [[], []]  # tensors of each area that are sent once (the base): freed after
        self.counts = threading.Lock()
        self.failure = ""  # a push that was not applied, or a worker that went away: the next update raises
        self.ids = itertools.count()
        self.stats = {"pushes": 0, "skipped": 0, "bytes": 0, "seconds": 0.0, "stage_seconds": 0.0}
        self.stats.update({"rejected": 0, "link_failures": 0, "link_pushes": 0, "rejoined": 0, "lost": 0})
        self.link_backend, self.link_timeout = link, link_timeout
        self.verify = verify or link != "tcp"  # full pushes travel as checksummed buckets
        self.bucket_bytes = bucket_mb * 2**20
        self.wire_dtype = getattr(torch, wire_dtype) if wire_dtype else None
        self.min_engines, self.reconnect_s = min_engines, reconnect_s
        self.store = None  # the links' rendezvous store (hosted here, built with the first link)
        self.device_buckets: dict[int, torch.Tensor] = {}  # link pushes: the snapshot's buckets on the device
        self.closing = False
        self.gate: AdmissionGate | None = None
        if concurrency is not None:
            self.gate = AdmissionGate(concurrency, [e.address for e in self.engines], self._load, self._capacity)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="pgs-engine-pool", daemon=True)
        self.thread.start()
        try:
            self._call(self._connect(connect_timeout))
        except BaseException:
            self.close()
            raise

    def _call(self, coroutine, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    async def _connect(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        everyone = {id(e): e for e in [*self.engines, *self.targets]}.values()
        for engine in everyone:
            while True:
                try:
                    await self._open(engine)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise ConnectionError(
                            f"engine worker {engine.address} did not answer in {timeout:g} s"
                        ) from None
                    await asyncio.sleep(0.5)

    async def _open(self, engine: _Engine) -> None:
        """Connect to a worker, greet it, build its link (if any) and listen for its answers."""
        if engine in self.engines:
            engine.reader, engine.writer = await asyncio.open_connection(engine.host, engine.port)
        if engine in self.targets:
            engine.weights_reader, engine.weights_writer = await asyncio.open_connection(engine.host, engine.port)
        engine.write_lock, engine.push_lock = asyncio.Lock(), asyncio.Lock()
        await self._hello(engine)
        if engine in self.targets and self.link_backend != "tcp":
            await self._link(engine)
        if engine.reader is not None:
            engine.listener = asyncio.ensure_future(self._listen(engine))

    async def _link(self, engine: _Engine) -> None:
        """Build the two-rank group with this engine's worker; on any failure the engine is served over TCP."""
        loop = asyncio.get_running_loop()
        try:
            if self.store is None:
                self.store = bucketed.Link.host_store(self.link_timeout)
            prefix = bucketed.Link.new_prefix()
            host = engine.weights_writer.get_extra_info("sockname")[0]  # this node, as the worker reaches it
            offer = {
                "op": "link",
                "backend": self.link_backend,
                "store": [host, self.store.port],
                "prefix": prefix,
                "timeout": self.link_timeout,
            }
            await send(engine.weights_writer, offer)

            def build():
                link = bucketed.Link(self.link_backend, 0, self.store, prefix, self.link_timeout)
                link.handshake()
                return link

            mine = loop.run_in_executor(None, build)
            reply, _ = await receive(engine.weights_reader)
            if "error" in reply:
                mine.add_done_callback(lambda f: f.exception())  # (its end times out by itself)
                raise bucketed.LinkError(f"the worker could not join: {reply['error']}")
            engine.link = await mine
            logger.info("engine worker %s: %s link ready", engine.address, self.link_backend)
        except (bucketed.LinkError, RuntimeError, ValueError) as error:
            self.stats["link_failures"] += 1
            engine.link = None
            logger.warning(
                "engine worker %s: no %s link (%s): its pushes go over TCP", engine.address, self.link_backend, error
            )

    async def _hello(self, engine: _Engine) -> None:
        """Refuse a worker that was started for another model or with other generation settings."""
        reader, writer = (
            (engine.reader, engine.writer)
            if engine.reader is not None
            else (engine.weights_reader, engine.weights_writer)
        )
        await send(writer, {"op": "hello", "push": engine in self.targets})
        reply, _ = await receive(reader)
        theirs = reply.get("identity") or {}
        differ = [
            f"{key}: the trainer has {mine!r}, the worker {theirs[key]!r}"
            for key, mine in self.identity.items()
            # (the worker serves adapters up to its rank and log-probs up to its maximum; every other setting must
            # be the same)
            if key in theirs and (theirs[key] < mine if key in _AT_LEAST else theirs[key] != mine)
        ]
        if differ:
            raise ValueError(
                f"engine worker {engine.address} was started with other settings than this run's "
                f"({'; '.join(differ)}). Start it with the flags `pgs scenario plan --engine-gpus` prints "
                "(or the worker's --model, --stop-tokens, --chat-template-kwargs, --max-model-len, --fp32-head, "
                "--lora-rank, --max-logprobs, --inflight)."
            )

    def _lose(self, engine: _Engine, reason: str, closing: bool = False) -> None:
        if closing:
            engine.dead = engine.dead or reason
        if not engine.dead:
            engine.dead = reason
            logger.error("engine worker %s is gone (%s)", engine.address, reason)
            self.stats["lost"] += 1
            if engine.link is not None:
                engine.link.close()
                engine.link = None
            alive = sum(not e.dead for e in self.engines) if self.engines else sum(not e.dead for e in self.targets)
            if self.min_engines and alive >= self.min_engines and not self.closing:
                # the run goes on without it; its address is tried again until a worker answers there
                logger.warning("continuing with %d engine(s) (rollout.min_engines %d)", alive, self.min_engines)
                if engine.retry is None or engine.retry.done():
                    engine.retry = asyncio.run_coroutine_threadsafe(self._rejoin(engine), self.loop)
            elif engine in self.targets or self.min_engines:
                self.failure = self.failure or f"engine worker {engine.address} went away ({reason})"
        for future in engine.waiting.values():
            if not future.done():
                future.set_exception(EngineLost(f"engine worker {engine.address} went away ({reason})"))
        engine.waiting.clear()

    async def _rejoin(self, engine: _Engine) -> None:
        """rollout.min_engines: try a lost worker's address until one answers, then greet it as a new worker. It
        generates again at once; what it samples is trained after the next full push reached it."""
        while not self.closing:
            await asyncio.sleep(self.reconnect_s)
            if engine.listener is not None:
                engine.listener.cancel()
            for writer in (engine.writer, engine.weights_writer):
                if writer is not None:
                    writer.close()
            try:
                engine.applied, engine.confirmed, engine.queued, engine.rejections = -1, False, None, 0
                await self._open(engine)
            except (OSError, asyncio.IncompleteReadError, ValueError) as error:
                logger.debug("engine worker %s is still away (%r)", engine.address, error)
                continue
            if self.payload is not None:
                self.first = True  # adapters: the next push carries the frozen base again
            engine.dead = ""
            self.stats["rejoined"] += 1
            logger.warning("engine worker %s is back: it joins after the next weight update", engine.address)
            while engine not in self.targets and not engine.confirmed and not engine.dead and not self.closing:
                # (another rank pushes to it: ask until it holds this run's weights)
                await asyncio.sleep(self.reconnect_s)
                with contextlib.suppress(EngineLost):
                    state = await self._ask(engine, {"op": "state"})
                    if state.get("confirmed"):
                        engine.confirmed, engine.applied = True, int(state["version"])
            return

    async def _listen(self, engine: _Engine) -> None:
        try:
            while True:
                header, tensors = await receive(engine.reader)
                if tensors:  # a generation's candidates (opd.rollout.Candidates)
                    header["tensors"] = tensors
                future = engine.waiting.pop(header.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(header)
        except asyncio.CancelledError:
            self._lose(engine, "the pool was closed", closing=True)
            raise
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as error:
            self._lose(engine, repr(error))

    async def _ask(self, engine: _Engine, header: dict) -> dict:
        """Send a request frame on the request connection and wait for the reply with its id."""
        if engine.dead:
            raise EngineLost(f"engine worker {engine.address} went away ({engine.dead})")
        number = next(self.ids)
        future = self.loop.create_future()
        engine.waiting[number] = future
        try:
            async with engine.write_lock:
                await send(engine.writer, {**header, "id": number})
        except (ConnectionError, OSError) as error:
            engine.waiting.pop(number, None)
            self._lose(engine, repr(error))
            raise EngineLost(f"engine worker {engine.address} went away ({error!r})") from None
        try:
            return await future
        finally:
            engine.waiting.pop(number, None)  # (a cancelled request no longer counts as this engine's load)

    # ------------------------------------------------------------- generation

    def make_client(self, merge: bool = False, adapter: str = "") -> "RemoteClient":
        return RemoteClient(self, adapter)

    async def _request(
        self, prompt_ids, max_new_tokens, temperature, adapter: str, continuing: bool = False
    ) -> Generation:
        if dataclasses.is_dataclass(temperature):  # opd.rollout.Sampling
            temperature = dataclasses.asdict(temperature)
        else:
            temperature = float(temperature)
        header = {
            "op": "generate",
            "prompt": list(prompt_ids),
            "max_new_tokens": int(max_new_tokens),
            "temperature": temperature,
            "adapter": adapter,
        }
        while True:
            alive = [e for e in self.engines if not e.dead]
            if not alive:
                gone = "; ".join(f"{e.address}: {e.dead}" for e in self.engines)
                raise ConnectionError(f"no engine worker is left to generate with ({gone})")
            if self.started:  # a worker that rejoined holds none of this run's weights until a push reached it
                alive = [e for e in alive if e.confirmed] or alive
            try:
                if self.gate is not None:  # rollout.concurrency kv: the engine with the most room under its cap
                    address = await self.gate.acquire([e.address for e in alive], priority=continuing)
                    engine = next(e for e in alive if e.address == address)
                    # The slot is held until the worker answers (or is lost): a caller that is cancelled leaves
                    # its request generating on the worker, which has no abort, so the slot stays taken.
                    ask = asyncio.ensure_future(self._ask(engine, header))
                    ask.add_done_callback(functools.partial(self._answered, address, len(header["prompt"])))
                    reply = await asyncio.shield(ask)
                else:
                    engine = min(alive, key=lambda e: e.load)
                    reply = await self._ask(engine, header)
            except EngineLost:
                continue  # the request starts again on an engine that is still there
            if "error" in reply:
                raise RuntimeError(f"engine worker {engine.address}: {reply['error']}")
            return Generation(
                reply["ids"],
                reply["logprobs"],
                reply["finish"],
                reply["version"],
                reply["first_version"],
                candidates=Candidates.from_tensors(reply.get("tensors") or {}),
                versions=[tuple(run) for run in reply["versions"]] if reply.get("versions") else None,
            )

    def _answered(self, address: str, prompt_tokens: int, ask: asyncio.Future) -> None:
        """A gated request was answered (or its engine lost): free its slot, telling the gate how many tokens the
        request held, and take the exception a cancelled caller will not read."""
        reply = None if ask.cancelled() or ask.exception() is not None else ask.result()
        answered = isinstance(reply, dict) and "ids" in reply
        self.gate.release(address, prompt_tokens + len(reply["ids"]) if answered else 0)

    async def _load(self, address: str) -> EngineLoad | None:
        """An engine's load as its worker reports it in `state` (None: the worker's engine reports none)."""
        engine = next(e for e in self.engines if e.address == address)
        if engine.dead:
            return None
        try:
            load = (await self._ask(engine, {"op": "state"})).get("load")
        except EngineLost:
            return None
        if not isinstance(load, dict):
            return None
        fields = ("kv_usage", "waiting", "running", "preemptions", "kv_tokens")
        return EngineLoad(**{key: load.get(key) for key in fields})

    async def _capacity(self, address: str) -> tuple[int | None, int | None]:
        load = await self._load(address)
        return (load.kv_tokens if load is not None else None), self.identity.get("max_model_len")

    # ---------------------------------------------------------------- weights

    def check(self) -> None:
        """Raise if a worker went away or a push was not applied (see the module docstring)."""
        if self.failure:
            raise RuntimeError(
                f"rollout engines: {self.failure}. The engines no longer hold this run's weights: restart the "
                "worker and resume from the last checkpoint."
            )

    def update_weights(self, named_tensors, version: int) -> None:
        """Stage the weights of `version` and start sending them to every target engine; returns at once (see
        the module docstring), or when every engine applied them if the pool is blocking. A process that does not
        push (ranks other than 0) records the version, and if the pool is blocking waits until its own engines
        report it. The first update of a run always waits: no rollout is sampled from what a worker held before."""
        self.check()
        blocking = self.blocking or not self.started
        if not self.targets:
            for _ in named_tensors or ():  # (gathered parameters: every rank must iterate them)
                pass
            if blocking:
                self._wait(self._reached(version), f"version {version} from rank 0")
            self.version, self.started = version, True
            return
        on_link = self.link_backend != "tcp" and self.payload is None
        with self.counts:
            # (a link's snapshot lives on the trainer's device: one of them, not two)
            free = [i for i in ((0,) if on_link else (0, 1)) if not self.sending[i]]
            if free:
                self.sending[free[0]] += len(self.targets)
        if not free:
            for _ in named_tensors or ():
                pass
            self.stats["skipped"] += 1  # both snapshots still travelling: the next sync brings newer weights
            return
        try:
            start = time.perf_counter()
            if self.payload is not None:  # LoRA: [(header fields, tensors), ...], the base first on the first push
                for _ in named_tensors or ():
                    pass
                frames = self.payload(self.first)
            elif self.verify:  # checksummed buckets: on the device for a link, in pinned host memory for TCP
                snapshot = self._snapshot(named_tensors, self.staging[free[0]], on_link)
                self.stats["stage_seconds"] += time.perf_counter() - start
                frames = [({"op": "weights", "version": version, "mode": "full"}, snapshot)]
                named_tensors = None
            else:
                frames = [({"mode": "full"}, self._wire(named_tensors))]
            area = self.staging[free[0]]
            self.once[free[0]] = [
                name for fields, tensors in frames if fields.get("mode") == "base" for name in tensors
            ]
            if named_tensors is not None or self.payload is not None:
                frames = [
                    ({"op": "weights", "version": version, **fields}, area.stage(tensors)) for fields, tensors in frames
                ]
                self.stats["stage_seconds"] += time.perf_counter() - start
        except BaseException:
            with self.counts:
                self.sending[free[0]] -= len(self.targets)
            raise
        self.first = False
        if not blocking:
            self.version = version
        futures = [
            asyncio.run_coroutine_threadsafe(self._push(engine, free[0], frames), self.loop) for engine in self.targets
        ]
        if blocking:
            self._wait(self._all(futures), f"version {version}")
            self.check()  # (a push the worker could not apply is raised now, not after a wait for its version)
            self._wait(self._reached(version), f"version {version}")  # (engines of this rank it does not push to)
            self.check()
            self.version, self.started = version, True  # (set after the engines have it: pool/lag reads 0)

    def _wire(self, named_tensors) -> dict[str, torch.Tensor]:
        """Full weights as they are staged: float tensors cast to rollout.push_dtype on their own device first."""
        if self.wire_dtype is None:
            return dict(named_tensors)
        return {
            name: tensor.detach().to(self.wire_dtype) if tensor.is_floating_point() else tensor
            for name, tensor in named_tensors
        }

    def _snapshot(self, named_tensors, area: Staging, on_link: bool) -> bucketed.Buckets:
        """The weights packed into checksummed buckets, reusing the last snapshot's buffers."""
        named = list(named_tensors) if on_link else named_tensors
        device = named[0][1].device if on_link and named else None

        def allocate(index: int, nbytes: int) -> torch.Tensor:
            if not on_link:
                return area.buffer(f"bucket/{index}", torch.uint8, [nbytes])
            have = self.device_buckets.get(index)
            if have is None or have.numel() != nbytes or have.device != device:
                have = self.device_buckets[index] = torch.empty(nbytes, dtype=torch.uint8, device=device)
            return have

        snapshot = bucketed.pack(named, self.bucket_bytes, allocate, self.wire_dtype)
        if not on_link and area.pin:
            torch.cuda.synchronize()
        return snapshot

    async def _all(self, futures) -> None:
        await asyncio.gather(*(asyncio.wrap_future(f) for f in futures))

    def _wait(self, coroutine, what: str) -> None:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            future.result(self.push_timeout)
        except TimeoutError:
            future.cancel()
            behind = [f"{e.address} (at version {e.applied})" for e in {*self.targets, *self.engines}]
            raise RuntimeError(
                f"rollout engines did not confirm {what} in {self.push_timeout:g} s (rollout.max_staleness 0 waits "
                f"for them): {', '.join(sorted(behind))}. Check the workers and the link between the nodes."
            ) from None

    async def _reached(self, version: int) -> None:
        """Until every engine this process generates with holds this run's weights, at `version` or newer (or a
        push failed: the caller raises it)."""

        def behind() -> list[_Engine]:
            return [e for e in self.engines if not e.dead and not (e.confirmed and e.applied >= version)]

        while behind() and not self.failure:
            for engine in behind():
                with contextlib.suppress(EngineLost):
                    state = await self._ask(engine, {"op": "state"})
                    if state.get("confirmed", True):
                        engine.confirmed, engine.applied = True, max(engine.applied, int(state["version"]))
            if behind():
                await asyncio.sleep(0.02)

    async def _push(self, engine: _Engine, area: int, frames: list) -> None:
        if engine.dead:
            self._sent(area)
            return
        if engine.push_lock.locked():  # a push is on its way to this engine: this one follows it, unless a newer
            previous, engine.queued = engine.queued, (area, frames)  # one is staged before it starts
            if previous is not None:
                self._sent(previous[0])
            return
        async with engine.push_lock:
            job: tuple | None = (area, frames)
            while job is not None:
                area, frames = job
                start = time.perf_counter()
                try:
                    sent = 0
                    applied = True
                    for header, staged in frames:
                        if isinstance(staged, bucketed.Buckets):
                            reply, sent = await self._push_buckets(engine, header, staged)
                            applied = reply is not None
                            continue
                        sent += await send(engine.weights_writer, header, staged)
                        reply, _ = await receive(engine.weights_reader)
                        if "error" in reply:
                            raise RuntimeError(f"the worker could not apply the update: {reply['error']}")
                    if applied:
                        engine.applied, engine.confirmed = max(engine.applied, int(reply["version"])), True
                        self.stats["pushes"] += 1
                        self.stats["bytes"] += sent
                        self.stats["seconds"] += time.perf_counter() - start
                except (ConnectionError, OSError, asyncio.IncompleteReadError) as error:
                    self._lose(engine, repr(error))
                except Exception as error:  # noqa: BLE001 — the trainer raises it at its next update
                    logger.exception("weights for %s were not applied", engine.address)
                    self.failure = self.failure or f"{engine.address}: {error}"
                finally:
                    self._sent(area)
                job, engine.queued = engine.queued, None
                if job is not None and engine.dead:
                    self._sent(job[0])
                    job = None

    async def _push_buckets(self, engine: _Engine, header: dict, snapshot: bucketed.Buckets):
        """Send a verified push: over the engine's link when it has one (TCP from then on if the link fails), in
        the frame otherwise; once more if the engine rejects it. (the worker's reply, bytes sent), or (None, 0)
        when the engine did not apply it and stays on its version; raises when it rejected three pushes in a row,
        or holds no whole weights and the pool waits for it."""
        loop = asyncio.get_running_loop()
        reply: dict = {}
        for attempt in range(2):
            if engine.link is not None:
                await send(engine.weights_writer, {**header, "buckets": snapshot.spec("link")})
                link = engine.link

                def transfer(link=link):
                    for buffer in snapshot.buffers:
                        link.send(buffer)

                failed = ""
                try:
                    await loop.run_in_executor(None, transfer)
                except bucketed.LinkError as error:
                    failed = str(error)
                reply, _ = await receive(engine.weights_reader)
                if failed or reply.get("link") is False:
                    self.stats["link_failures"] += 1
                    link.close()
                    engine.link = None
                    logger.warning(
                        "engine worker %s: the %s link failed (%s): its pushes go over TCP from now on",
                        engine.address,
                        self.link_backend,
                        failed or reply.get("error"),
                    )
                    if "error" not in reply:  # (it arrived whole all the same)
                        return reply, snapshot.nbytes
                    reply = {}
                    continue  # this push again, in the frame
                if "error" not in reply:
                    self.stats["link_pushes"] += 1
            else:
                host = await loop.run_in_executor(None, snapshot.host)
                await send(engine.weights_writer, {**header, "buckets": snapshot.spec("tcp")}, host)
                reply, _ = await receive(engine.weights_reader)
            if "error" not in reply:
                engine.rejections = 0
                return reply, snapshot.nbytes
            if not reply.get("rejected"):
                raise RuntimeError(f"the worker could not apply the update: {reply['error']}")
            self.stats["rejected"] += 1
            logger.warning(
                "engine worker %s rejected version %s: %s", engine.address, header["version"], reply["error"]
            )
            if not reply.get("restored", True):
                engine.confirmed = False  # it tags what it samples as untrainable until a push is applied
        engine.rejections += 1
        if engine.rejections >= 3 or self.blocking or not self.started:
            raise RuntimeError(f"verified push rejected {engine.rejections} time(s) in a row: {reply.get('error')}")
        return None, 0

    def _sent(self, area: int) -> None:
        """One push from `area` is over (delivered or not); its one-time tensors are freed with the last."""
        with self.counts:
            self.sending[area] -= 1
            if not self.sending[area] and self.once[area]:
                self.staging[area].release(self.once[area])
                self.once[area] = []

    def applied(self) -> list[int]:
        """The newest version each target engine confirmed."""
        return [engine.applied for engine in self.targets]

    def metrics(self) -> dict[str, float]:
        """pool/lag: versions the slowest engine is behind the newest staged one; pool/push_s and pool/push_mb:
        means per delivered push; pool/skipped: snapshots not staged because two were still travelling;
        pool/engines: workers still generating."""
        pushes = max(self.stats["pushes"], 1)
        out = {
            "pool/push_s": self.stats["seconds"] / pushes,
            "pool/push_mb": self.stats["bytes"] / pushes / 2**20,
            "pool/skipped": float(self.stats["skipped"]),
            "pool/engines": float(sum(not e.dead for e in self.engines)),
        }
        alive = [engine.applied for engine in self.targets if not engine.dead]
        if alive:
            out["pool/lag"] = float(self.version - min(alive))
        seconds = self.stats["seconds"]
        if seconds > 0:  # measured streaming: what the pushes moved per second of travel and apply
            out["pool/push_mb_s"] = self.stats["bytes"] / 2**20 / seconds
        out["pool/stage_s"] = self.stats["stage_seconds"] / max(self.stats["pushes"] + self.stats["skipped"], 1)
        if self.verify:
            out["pool/rejected"] = float(self.stats["rejected"])
        if self.link_backend != "tcp":
            out["pool/link_pushes"] = float(self.stats["link_pushes"])
            out["pool/link_failures"] = float(self.stats["link_failures"])
        if self.min_engines:
            out["pool/lost"], out["pool/rejoined"] = float(self.stats["lost"]), float(self.stats["rejoined"])
        return out

    def state(self) -> list[dict]:
        """Each generating engine's own state (version, pushes applied, pushes dropped, receiving)."""
        return self._call(self._states())

    async def _states(self) -> list[dict]:
        return list(await asyncio.gather(*(self._ask(engine, {"op": "state"}) for engine in self.engines)))

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

        if not self.thread.is_alive():
            return
        self.closing = True
        for engine in {id(e): e for e in [*self.engines, *self.targets]}.values():
            if engine.retry is not None:
                engine.retry.cancel()
            if engine.link is not None:
                engine.link.close()
                engine.link = None
        if self.gate is not None:
            self.gate.close()
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
        self.gate = pool.gate  # (its metrics; requests acquire it in the pool)
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

    async def generate(
        self, prompt_ids, max_new_tokens, temperature, images: list | None = None, continuing: bool = False
    ) -> Generation:
        """`continuing`: a later turn of a running episode (rollout.concurrency kv admits it first)."""
        if images:
            raise NotImplementedError("remote engines do not take images yet")
        self.calls += 1
        coroutine = self.pool._request(prompt_ids, max_new_tokens, temperature, self.adapter, continuing)
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, self.pool.loop))

    def close(self) -> None:
        pass
