"""Batched verification in a process pool.

Verifiers that burn CPU (symbolic equivalence, anything that parses or searches) must not run on the event loop that
drives the rollouts, and threads do not help pure-Python work. verify() sends the check to a shared pool of worker
processes; calls made within the same loop iteration (a group's samples finishing together, a batch of episodes)
travel as one batch per worker, so the cost per check is a list element, not a round trip.

    equal = await verify(answers_equal, prediction, gold, timeout=2.0)

`fn` is a module-level function (it is pickled by reference). Each item has its own time limit inside the worker
(an alarm): a check that hangs returns VerifyTimeout for that item and the worker goes on with the batch. Workers
start on first use ("spawn": safe next to threads and CUDA) and stay up; PGS_VERIFY_WORKERS sets their number
(default: up to 4), 0 runs the checks in threads of this process (debugging, or platforms without spawn).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

MAX_BATCH = 64


class VerifyTimeout(Exception):
    """The check did not finish within its time limit."""


def _run_batch(fn: Callable, items: list[tuple], timeout: float) -> list[tuple[bool, Any]]:
    """In the worker: fn(*args) for every item, each under its own alarm. (ok, value | error text) per item."""
    import signal

    use_alarm = timeout > 0 and hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()

    def expired(signum, frame):
        raise VerifyTimeout()

    previous = signal.signal(signal.SIGALRM, expired) if use_alarm else None
    out: list[tuple[bool, Any]] = []
    try:
        for args in items:
            try:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, timeout)
                out.append((True, fn(*args)))
            except VerifyTimeout:
                out.append((False, "timeout"))
            except BaseException as error:  # noqa: BLE001 — recursion limits, memory: the item fails, not the worker
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                out.append((False, f"{type(error).__name__}: {error}"))
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        if use_alarm:
            signal.signal(signal.SIGALRM, previous)
    return out


class VerifierPool:
    def __init__(self, workers: int | None = None):
        if workers is None:
            workers = int(os.environ.get("PGS_VERIFY_WORKERS", min(4, os.cpu_count() or 1)))
        self.workers = max(0, workers)
        self.executor: ProcessPoolExecutor | None = None
        self.lock = threading.Lock()
        self.pending: dict[Any, list] = {}  # loop -> [(fn, timeout, args, future)]
        self.batches = self.items = 0

    def _executor(self) -> ProcessPoolExecutor | None:
        if self.workers == 0:
            return None
        with self.lock:
            if self.executor is None:
                import multiprocessing

                self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))
            return self.executor

    async def verify(self, fn: Callable, *args, timeout: float = 5.0) -> Any:
        """fn(*args) in a worker process; raises VerifyTimeout, or RuntimeError with the worker's error."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        queue = self.pending.setdefault(loop, [])
        queue.append((fn, timeout, args, future))
        if len(queue) == 1:
            loop.call_soon(self._flush, loop)  # after every coroutine that is ready now has queued its check
        ok, value = await future
        if ok:
            return value
        if value == "timeout":
            raise VerifyTimeout(f"{getattr(fn, '__name__', fn)} did not finish within {timeout:g} s")
        raise RuntimeError(value)

    def _flush(self, loop) -> None:
        queue = self.pending.pop(loop, [])
        groups: dict[tuple, list] = {}
        for fn, timeout, args, future in queue:
            groups.setdefault((fn, timeout), []).append((args, future))
        executor = self._executor()
        for (fn, timeout), members in groups.items():
            # one batch per worker, at most MAX_BATCH items each
            size = max(1, min(MAX_BATCH, -(-len(members) // max(1, self.workers))))
            for start in range(0, len(members), size):
                chunk = members[start : start + size]
                self.batches += 1
                self.items += len(chunk)
                items = [args for args, _ in chunk]
                task = loop.run_in_executor(executor, _run_batch, fn, items, timeout)
                task.add_done_callback(lambda done, chunk=chunk: self._deliver(done, chunk))

    def _deliver(self, done, chunk: list) -> None:
        try:
            results = done.result()
        except BaseException as error:  # noqa: BLE001 — a worker died (out of memory, killed): the pool restarts
            with self.lock:
                if self.executor is not None:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                    self.executor = None
            results = [(False, f"verifier worker failed: {type(error).__name__}: {error}")] * len(chunk)
        for (_, future), result in zip(chunk, results):
            if not future.done():
                future.set_result(result)

    def shutdown(self) -> None:
        with self.lock:
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None


_shared: VerifierPool | None = None


def shared_pool() -> VerifierPool:
    global _shared
    if _shared is None:
        _shared = VerifierPool()
        atexit.register(_shared.shutdown)
    return _shared


async def verify(fn: Callable, *args, timeout: float = 5.0) -> Any:
    """fn(*args) in the shared verifier pool (see the module docstring)."""
    return await shared_pool().verify(fn, *args, timeout=timeout)
