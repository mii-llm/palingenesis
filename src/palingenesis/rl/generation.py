"""Async generation on top of a rollout engine.

Rollouts are coroutines (one per trajectory) that ask for one assistant turn at a time.

Streaming engines (the colocated vLLM engine): one engine thread owns the engine. Every
request is added to the running batch the moment it arrives and resolved the step it
finishes, so a multi-turn trajectory whose turn ended runs its tools and comes back while
the rest keep decoding: continuous batching across turns. Requests that arrive together
with the same prompt and budget (a group's first turn) become one request with n samples
when `merge` (single-turn: a group needs all its answers anyway), so their prompt is
prefilled once. Multi-turn requests stay separate: vLLM returns an n-sample request when
its last sample ends, which would hold every trajectory's first tool call until the group's
longest first turn (measured: 83-token turns waiting 7.9 s); the prefix cache shares the
prompt instead.

Batch engines (HF generate, a vLLM server): the requests pending at once are gathered into
one engine.generate call on the engine's thread, as soon as every live trajectory is waiting
for a turn, or after `window` seconds when some are still busy with tools or rewards. A call
returns when its longest sequence ends, so multi-turn trajectories advance in rounds there.
"""

import asyncio
import contextlib
import functools
import itertools
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from palingenesis.accelerator import on_current_device
from palingenesis.opd.rollout import RolloutEngine, samplings


@dataclass
class Generation:
    ids: list[int]
    logprobs: list[float]
    finish: str  # stop | length
    version: int  # the policy version that finished the turn
    first_version: int | None = None  # the version that sampled its first token (None: `version`)
    routed_experts: object | None = None  # MoE routing replay: see opd.rollout.Rollout
    candidates: object | None = None  # opd.rollout.Candidates (Sampling.candidates > 0), else None
    # per-token policy versions as runs [(first token index, version), ...] (opd.rollout.Rollout.versions), or None
    versions: list | None = None


def _generation(r) -> Generation:
    first = getattr(r, "first_version", None)
    routed = getattr(r, "routed_experts", None)
    top = getattr(r, "candidates", None)
    runs = getattr(r, "versions", None)
    return Generation(
        list(r.completion_ids), list(r.logprobs), r.finish_reason, r.policy_version, first, routed, top, runs
    )


def _joined(prefix, rest):
    """A completion resumed after a weight update (rollout.inflight resume): `prefix`, what was sampled before
    it (a Rollout), followed by `rest`, the request that continued from it. Log-probs, candidates and per-token
    versions are each token's own; the routing record is dropped (the continuation re-prefilled the prefix)."""
    from palingenesis.opd.rollout import Candidates, Rollout

    n = len(prefix.completion_ids)
    before = prefix.versions or [(0, prefix.policy_version if prefix.first_version is None else prefix.first_version)]
    after = rest.versions or [(0, rest.policy_version)]
    runs = [r for r in before if r[0] < n or r[0] == 0] if n else []
    for start, version in after:
        if runs and runs[-1][1] == version:
            continue
        runs.append((n + start, version))
    candidates = None
    if prefix.candidates is not None or rest.candidates is not None:
        candidates = Candidates.join(
            [
                prefix.candidates if prefix.candidates is not None else Candidates.empty(n),
                rest.candidates if rest.candidates is not None else Candidates.empty(len(rest.completion_ids)),
            ]
        )
    return Rollout(
        list(prefix.completion_ids) + list(rest.completion_ids),
        list(prefix.logprobs) + list(rest.logprobs),
        rest.finish_reason,
        rest.policy_version,
        runs[0][1] if runs else rest.policy_version,
        None,
        candidates,
        runs,
    )


def _release(gate, key: str, prompt_tokens: int, future: asyncio.Future) -> None:
    """A gated request's future is done: free its slot, telling the gate how many tokens the request held (and
    take its exception, which a caller that was cancelled will not read)."""
    failed = future.cancelled() or future.exception() is not None
    gate.release(key, 0 if failed else prompt_tokens + len(future.result().ids))


class GenerationClient:
    def __init__(
        self, engine: RolloutEngine, window: float = 0.02, merge: bool = True, gate=None, inflight: str = "continue"
    ):
        """`gate`: a palingenesis.rl.concurrency.AdmissionGate every request acquires first (rollout.concurrency
        kv: requests in flight under the engine's KV-sized cap); None: every request goes out at once.
        `inflight` (rollout.inflight): what requests in flight do when weights are loaded under paused().
        "continue": go on; "tag": go on, the engine recording each token's version (engines with track_tokens);
        "resume": each is aborted and continued from its tokens so far as a new request (engines with
        stream_abort; the others continue)."""
        self.engine = engine
        self.inflight_mode = inflight
        self.resumed = 0  # generations aborted at a weight update and continued (rollout.inflight resume)
        if inflight != "continue" and hasattr(engine, "track_tokens"):
            engine.track_tokens = True
        self.inflight: dict[str, list[tuple[asyncio.Future, asyncio.AbstractEventLoop]]] = {}
        self.requests: dict[str, tuple] = {}  # request id -> (prompt, budget, temperature, images, adapter)
        self.prefixes: dict[str, object] = {}  # resumed request id -> the Rollout sampled before the update
        self.merge = merge
        self.window = window
        self.gate = gate
        self.calls = 0  # engine calls (batch) or requests added (streaming)
        self.active = 0  # live trajectories (each may ask for another turn)
        self.streaming = hasattr(engine, "stream_add")
        self.timeline: list[tuple[float, int, int]] = []  # per engine step: (time, sequences in flight, finished)
        # held by the engine thread for each add + step; weight updates take it between steps,
        # so requests in flight keep their state and continue with the new weights
        self.engine_lock = threading.Lock()
        if self.streaming:
            self.inbox: queue.Queue = queue.Queue()
            self.ids = itertools.count()
            self.thread = threading.Thread(target=on_current_device(self._engine_loop), name="pgs-engine", daemon=True)
            self.thread.start()
        else:
            # one engine call at a time, on this rank's GPU
            self.executor = ThreadPoolExecutor(
                1, thread_name_prefix="pgs-generate", initializer=on_current_device(lambda: None)
            )
            self.pending: list[tuple[list[int], int, float, asyncio.Future, list | None, str]] = []
            self._arrived: asyncio.Event | None = None
            self._flusher: asyncio.Task | None = None

    def enter(self) -> None:
        self.active += 1

    def exit(self) -> None:
        self.active -= 1
        if not self.streaming:
            self._signal()  # one fewer to wait for

    async def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        images: list | None = None,
        adapter: str = "",
        continuing: bool = False,
    ) -> Generation:
        """`images`: the prompt's images, one per placeholder token of `prompt_ids`, in order
        (a multimodal policy; the engine expands the placeholders). `adapter`: the LoRA adapter of the engine to
        generate with (engines that serve several; requests of different adapters share the running batch).
        `continuing`: a later turn of an episode that is already running (with a gate: admitted before requests
        that start an episode)."""
        if self.gate is None:
            return await self._submit(prompt_ids, max_new_tokens, temperature, images, adapter)
        key = await self.gate.acquire(priority=continuing)
        try:
            future = self._submit(prompt_ids, max_new_tokens, temperature, images, adapter)
        except BaseException:
            self.gate.release(key)
            raise
        # The slot is the engine's work: it is released when the engine resolves the request, not when the caller
        # stops waiting. Engines have no abort, so a cancelled rollout's request keeps generating (and keeps its
        # slot) until it finishes; releasing at cancellation would let the gate send more than the cap.
        future.add_done_callback(functools.partial(_release, self.gate, key, len(prompt_ids)))
        return await asyncio.shield(future)

    def _submit(self, prompt_ids, max_new_tokens, temperature, images, adapter) -> asyncio.Future:
        """Hand a request to the engine: the future the engine resolves with its Generation."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        images = list(images) if images else None
        if self.streaming:
            self.inbox.put((prompt_ids, max_new_tokens, temperature, future, loop, images, adapter))
            return future
        self.pending.append((prompt_ids, max_new_tokens, temperature, future, images, adapter))
        if self._arrived is None:
            self._arrived = asyncio.Event()
        self._signal()
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush())
        return future

    # -------------------------------------------------------------- streaming

    def _engine_loop(self) -> None:
        """The engine thread: add what arrived, step, resolve what finished. Each add + step
        holds engine_lock, which weight updates take between steps (see paused())."""
        inflight = self.inflight
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
            with self.engine_lock:
                finished = self._add_and_step(items, inflight)
                if finished is None:
                    continue
                resolved = []
                for request_id, rollouts in finished:
                    self.requests.pop(request_id, None)
                    prefix = self.prefixes.pop(request_id, None)
                    if prefix is not None:
                        rollouts = [_joined(prefix, rollouts[0])]
                    resolved.append((inflight.pop(request_id), rollouts))
                live = sum(len(w) for w in inflight.values())
            for waiters, rollouts in resolved:
                self._resolve(waiters, rollouts)
            self.timeline.append((time.monotonic(), live, len(finished)))

    def _add_and_step(self, items, inflight):
        """Add the arrived requests, then one engine step: its finished requests (None when
        nothing is in flight)."""
        groups: dict[tuple, list] = {}
        pictures: dict[tuple, list] = {}
        adapters: dict[tuple, str] = {}
        for i, (prompt, budget, temperature, future, loop, *rest) in enumerate(items):
            images = rest[0] if rest else None  # a text request may come without the fields
            adapter = rest[1] if len(rest) > 1 else ""
            # greedy requests stay separate: vLLM refuses n > 1 without sampling
            merge = self.merge and samplings(temperature, 1)[0].temperature > 0
            # the same tokens with other pictures are another prompt (a group's rollouts share the objects)
            seen = tuple(id(image) for image in images or ())
            key = (tuple(prompt), budget, temperature, seen, adapter) + (() if merge else (i,))
            groups.setdefault(key, []).append((future, loop))
            if images:
                pictures[key] = images
            if adapter:
                adapters[key] = adapter
        for key, waiters in groups.items():
            prompt, budget, temperature = key[:3]
            request_id = str(next(self.ids))
            extra = {"images": pictures[key]} if key in pictures else {}
            if key in adapters:
                extra["adapter"] = adapters[key]
            try:
                self.engine.stream_add(request_id, list(prompt), budget, temperature, len(waiters), **extra)
            except BaseException as e:  # noqa: BLE001 — the waiting trajectories see the failure
                self._resolve(waiters, error=e)
                continue
            inflight[request_id] = waiters
            self.requests[request_id] = (list(prompt), budget, temperature, pictures.get(key), adapters.get(key, ""))
            self.calls += 1
        if not inflight:
            return None
        try:
            return self.engine.stream_step()
        except BaseException as e:  # noqa: BLE001 — every trajectory in flight sees the failure
            for waiters in inflight.values():
                self._resolve(waiters, error=e)
            inflight.clear()
            self.requests.clear()
            self.prefixes.clear()
            return None

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

    def paused(self):
        """A context in which the engine does not step: load weights while requests are in
        flight (they continue with the new weights; their tokens keep the log-probs they were
        sampled with, which is what the importance ratio uses). With rollout.inflight resume, requests in
        flight when the context loaded new weights are aborted and continued from their tokens so far."""
        if self.inflight_mode != "resume" or not self.streaming or not hasattr(self.engine, "stream_abort"):
            return self.engine_lock
        return self._resuming()

    @contextlib.contextmanager
    def _resuming(self):
        with self.engine_lock:
            before = self.engine.version
            yield
            if self.engine.version != before and before >= 0:
                self._resume()

    def _resume(self) -> None:
        """Under engine_lock, after new weights were loaded: abort every request in flight and add, for each of its
        samples that had not ended, a request that continues from the tokens it has (the new weights prefill them,
        so what follows is sampled by one policy on its own context). A sample that had ended is answered; one
        without budget left ends as "length"."""
        for request_id in list(self.inflight):
            try:
                partial = self.engine.stream_abort(request_id)
            except Exception as e:  # noqa: BLE001 — the request's trajectories see the failure
                self._resolve(self.inflight.pop(request_id), error=e)
                continue
            if partial is None:  # the engine does not follow this request: it continues under the new weights
                continue
            waiters = self.inflight.pop(request_id)
            prompt, budget, temperature, images, adapter = self.requests.pop(request_id)
            before = self.prefixes.pop(request_id, None)
            for waiter, part in zip(waiters, partial):
                left = budget - len(part.completion_ids)
                continued = prompt + list(part.completion_ids)
                if before is not None:  # resumed before: what it had sampled until then comes first
                    part = _joined(before, part)
                if part.finish_reason != "abort" or left <= 0:
                    if part.finish_reason == "abort":
                        part.finish_reason = "length"
                    self._resolve([waiter], [part])
                    continue
                new_id = str(next(self.ids))
                extra = {"images": images} if images else {}
                if adapter:
                    extra["adapter"] = adapter
                try:
                    self.engine.stream_add(new_id, continued, left, temperature, 1, **extra)
                except BaseException as e:  # noqa: BLE001
                    self._resolve([waiter], error=e)
                    continue
                self.inflight[new_id] = [waiter]
                self.requests[new_id] = (continued, left, temperature, images, adapter)
                self.prefixes[new_id] = part
                self.resumed += 1

    def close(self) -> None:
        if self.gate is not None:
            self.gate.close()
        if self.streaming and self.thread.is_alive():
            self.inbox.put(None)
            self.thread.join(timeout=10)
        if not self.streaming:
            self.executor.shutdown(wait=False, cancel_futures=True)  # its thread exits after the call in flight

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
            for temperature, adapter in sorted({(b[2], b[5]) for b in batch}, key=repr):  # (floats and Samplings)
                part = [b for b in batch if b[2] == temperature and b[5] == adapter]
                call = self.engine.generate
                if any(b[4] for b in part):  # only multimodal engines take images
                    call = functools.partial(call, images=[b[4] for b in part])
                if adapter:  # engines that serve several LoRA adapters
                    call = functools.partial(call, adapter=adapter)
                try:
                    rollouts = await loop.run_in_executor(
                        self.executor,
                        call,
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
