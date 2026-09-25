"""Async generation on top of a batch rollout engine.

Rollouts are coroutines (one per trajectory) that ask for one assistant turn at a time.
GenerationClient gathers the requests pending at once into a single engine.generate call
on the engine's own thread: as soon as every live trajectory is waiting for a turn, or
after `window` seconds when some are still busy with tools or rewards. A single-turn
batch is then exactly one engine call (vLLM schedules it continuously inside), and
multi-turn trajectories never wait for the slowest tool of a lock-stepped batch.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from palingenesis.opd.rollout import RolloutEngine


@dataclass
class Generation:
    ids: list[int]
    logprobs: list[float]
    finish: str  # stop | length
    version: int


class GenerationClient:
    def __init__(self, engine: RolloutEngine, window: float = 0.02):
        self.engine = engine
        self.window = window
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="pgs-generate")  # one engine call at a time
        self.pending: list[tuple[list[int], int, float, asyncio.Future]] = []
        self.active = 0  # live trajectories (each may ask for another turn)
        self.calls = 0
        self._arrived: asyncio.Event | None = None
        self._flusher: asyncio.Task | None = None

    def enter(self) -> None:
        self.active += 1

    def exit(self) -> None:
        self.active -= 1
        self._signal()  # one fewer to wait for

    def _signal(self) -> None:
        if self._arrived is not None:
            self._arrived.set()

    async def generate(self, prompt_ids: list[int], max_new_tokens: int, temperature: float) -> Generation:
        future = asyncio.get_running_loop().create_future()
        self.pending.append((prompt_ids, max_new_tokens, temperature, future))
        if self._arrived is None:
            self._arrived = asyncio.Event()
        self._signal()
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush())
        return await future

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
                        b[3].set_result(
                            Generation(
                                list(r.completion_ids),
                                list(r.logprobs),
                                r.finish_reason,
                                r.policy_version,
                            )
                        )
