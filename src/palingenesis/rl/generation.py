"""Async generation on top of a rollout engine.

Rollouts are coroutines (one per trajectory) that ask for one assistant turn at a time.

Streaming engines (the colocated vLLM engine): one engine thread owns the engine. Every
request is added to the running batch the moment it arrives and resolved the step it
finishes, so a multi-turn trajectory whose turn ended runs its tools and comes back while
the rest keep decoding: continuous batching across turns. Requests that arrive together
with the same prompt and budget (a group's first turn) become one request with n samples,
so their prompt is prefilled once.

Batch engines (HF generate, a vLLM server): the requests pending at once are gathered into
one engine.generate call on the engine's thread, as soon as every live trajectory is waiting
for a turn, or after `window` seconds when some are still busy with tools or rewards. A call
returns when its longest sequence ends, so multi-turn trajectories advance in rounds there.
"""

import asyncio
import itertools
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from palingenesis.opd.rollout import RolloutEngine


@dataclass
class Generation:
    ids: list[int]
    logprobs: list[float]
    finish: str  # stop | length
    version: int


def _generation(r) -> Generation:
    return Generation(list(r.completion_ids), list(r.logprobs), r.finish_reason, r.policy_version)


class GenerationClient:
    def __init__(self, engine: RolloutEngine, window: float = 0.02):
        self.engine = engine
        self.window = window
        self.calls = 0  # engine calls (batch) or requests added (streaming)
        self.active = 0  # live trajectories (each may ask for another turn)
        self.streaming = hasattr(engine, "stream_add")
        if self.streaming:
            self.inbox: queue.Queue = queue.Queue()
            self.ids = itertools.count()
            self.thread = threading.Thread(target=self._engine_loop, name="pgs-engine", daemon=True)
            self.thread.start()
        else:
            self.executor = ThreadPoolExecutor(1, thread_name_prefix="pgs-generate")  # one engine call at a time
            self.pending: list[tuple[list[int], int, float, asyncio.Future]] = []
            self._arrived: asyncio.Event | None = None
            self._flusher: asyncio.Task | None = None

    def enter(self) -> None:
        self.active += 1

    def exit(self) -> None:
        self.active -= 1
        if not self.streaming:
            self._signal()  # one fewer to wait for

    async def generate(self, prompt_ids: list[int], max_new_tokens: int, temperature: float) -> Generation:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self.streaming:
            self.inbox.put((prompt_ids, max_new_tokens, temperature, future, loop))
            return await future
        self.pending.append((prompt_ids, max_new_tokens, temperature, future))
        if self._arrived is None:
            self._arrived = asyncio.Event()
        self._signal()
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush())
        return await future

    # -------------------------------------------------------------- streaming

    def _engine_loop(self) -> None:
        """The engine thread: add what arrived, step, resolve what finished. It touches the
        engine only while requests are in flight, so weight updates and sleep/wake (between
        batches, with nothing in flight) never race with it."""
        inflight: dict[str, list[tuple[asyncio.Future, asyncio.AbstractEventLoop]]] = {}
        while True:
            items = []
            try:
                items.append(self.inbox.get(timeout=None if not inflight else 0))
            except queue.Empty:
                pass
            while True:
                try:
                    items.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            if any(item is None for item in items):
                return
            groups: dict[tuple, list] = {}
            for prompt, budget, temperature, future, loop in items:
                groups.setdefault((tuple(prompt), budget, temperature), []).append((future, loop))
            for (prompt, budget, temperature), waiters in groups.items():
                request_id = str(next(self.ids))
                try:
                    self.engine.stream_add(request_id, list(prompt), budget, temperature, len(waiters))
                except BaseException as e:  # noqa: BLE001 — the waiting trajectories see the failure
                    self._resolve(waiters, error=e)
                    continue
                inflight[request_id] = waiters
                self.calls += 1
            if not inflight:
                continue
            try:
                finished = self.engine.stream_step()
            except BaseException as e:  # noqa: BLE001 — every trajectory in flight sees the failure
                for waiters in inflight.values():
                    self._resolve(waiters, error=e)
                inflight.clear()
                continue
            for request_id, rollouts in finished:
                self._resolve(inflight.pop(request_id), rollouts)

    @staticmethod
    def _resolve(waiters, rollouts=None, error: BaseException | None = None) -> None:
        for i, (future, loop) in enumerate(waiters):
            value = error if error is not None else _generation(rollouts[i])

            def settle(future=future, value=value):
                if future.done():
                    return
                if isinstance(value, BaseException):
                    future.set_exception(value)
                else:
                    future.set_result(value)

            loop.call_soon_threadsafe(settle)

    def close(self) -> None:
        if self.streaming and self.thread.is_alive():
            self.inbox.put(None)
            self.thread.join(timeout=10)

    # ------------------------------------------------------------------ batch

    def _signal(self) -> None:
        if self._arrived is not None:
            self._arrived.set()

    async def _flush(self) -> None:
        loop = asyncio.get_running_loop()
        while self.pending:
            deadline = loop.time() + self.window
            while len(self.pending) < self.active and (remaining := deadline - loop.time()) > 0:
                self._arrived.clear()
                try:
                    await asyncio.wait_for(self._arrived.wait(), remaining)
                except asyncio.TimeoutError:
                    break
            batch, self.pending = self.pending, []
            for temperature in sorted({b[2] for b in batch}):
                part = [b for b in batch if b[2] == temperature]
                try:
                    rollouts = await loop.run_in_executor(
                        self.executor,
                        self.engine.generate,
                        [b[0] for b in part],
                        [b[1] for b in part],
                        temperature,
                    )
                except BaseException as e:  # noqa: BLE001 — every waiting trajectory sees the failure
                    for b in part:
                        if not b[3].done():
                            b[3].set_exception(e)
                    continue
                self.calls += 1
                for b, r in zip(part, rollouts):
                    if not b[3].done():
                        b[3].set_result(_generation(r))
