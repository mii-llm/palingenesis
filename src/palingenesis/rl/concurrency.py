"""How many generation requests a rollout engine gets at once, sized from its KV cache (rollout.concurrency: kv).

A vLLM engine admits every request it is sent and runs as many as its KV cache holds; when the running sequences
grow past it, vLLM preempts some and recomputes them later. Sending too few leaves the engine underused, sending
too many wastes its time on preemptions. The controller here keeps a cap on the requests in flight to one engine
and moves it with the engine's own load, read from its metrics every `poll_s` seconds:

  start  KV-cache tokens / max_model_len: as many requests as fit at full length (or the configured maximum when
         the engine does not report its KV capacity).
  cut    KV-cache usage above `high` (or requests waiting in the engine's queue on two polls in a row): the cap
         x 0.7, at most once until the requests in flight have drained below the cap (running requests keep
         going; new ones wait).
  raise  usage below `low`, nothing waiting, the cap reached since the last poll (requests waited at the gate),
         and a cap's worth of requests finished since the last change: the cap x 1.25.

The cap stays in [minimum, maximum]. A poll whose metrics are missing changes nothing (a renamed metric would
otherwise read as an idle engine and raise the cap). After TRL's AsyncGRPOTrainer (`_init_kv_cap`,
`_control_loop`, https://github.com/huggingface/trl), whose thresholds and factors these are.

`AdmissionGate` is what requests acquire, from one asyncio event loop: one controller per engine (a vLLM server,
or each worker of an engine pool), and a request goes to the engine with the most room. Requests wait first come,
first served, the later turns of running episodes before requests that start one, and a slot is held until the
engine is done with its request (a cancelled caller does not free it early). `TokenAllowance` bounds,
in tokens, how far rollout generation may run ahead of what a batch needs (rollout.token_allowance).

Nothing here imports torch or vLLM: configuration checks reach this module.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CUT = 0.7  # the cap's factor when the engine is overloaded
GROW = 1.25  # its factor when the engine has room
WAITING_POLLS = 2  # polls in a row with requests waiting in the engine's queue that count as overload

# vLLM's metric names (V1 first, the older V0 spelling after)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")  # a fraction, 1 = full, despite the name
WAITING = ("vllm:num_requests_waiting",)
RUNNING = ("vllm:num_requests_running",)
PREEMPTIONS = ("vllm:num_preemptions_total", "vllm:num_preemptions")
CACHE_INFO = "vllm:cache_config_info"

_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


# ── reading an engine's load ───────────────────────────────────────────────────


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Samples of a Prometheus text exposition: {name: [(labels, value), ...]}. Comments, blank lines and lines
    that do not parse are skipped; values that are not numbers too."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name, _, labels, value = match.groups()
        try:
            number = float(value)
        except ValueError:
            continue
        pairs = {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(labels or "")}
        out.setdefault(name, []).append((pairs, number))
    return out


@dataclass
class EngineLoad:
    """One reading of an engine's load. None: the engine did not report it."""

    kv_usage: float | None = None  # KV-cache blocks in use, a fraction (the highest over the engine's replicas)
    waiting: float | None = None  # requests in the engine's queue, not yet scheduled
    running: float | None = None  # requests being decoded
    preemptions: float | None = None  # cumulative preemptions
    kv_tokens: int | None = None  # the KV cache's capacity in tokens


def _first(samples: dict, names: tuple[str, ...]) -> list[float] | None:
    for name in names:
        if name in samples:
            return [value for _, value in samples[name] if math.isfinite(value)]
    return None


def kv_capacity(samples: dict) -> int | None:
    """KV-cache tokens from vLLM's cache_config_info labels (kv_cache_size_tokens, else num_gpu_blocks x
    block_size), summed over the engine's replicas."""
    total = 0
    for labels, _ in samples.get(CACHE_INFO, []):
        try:
            if labels.get("kv_cache_size_tokens", "").isdigit():
                total += int(labels["kv_cache_size_tokens"])
            elif labels.get("num_gpu_blocks", "").isdigit() and labels.get("block_size", "").isdigit():
                total += int(labels["num_gpu_blocks"]) * int(labels["block_size"])
        except ValueError:
            continue
    return total or None


def load_from_prometheus(text: str) -> EngineLoad:
    """An engine's load from its /metrics text (vLLM's names; several engines behind one server: the highest
    usage, the summed queues)."""
    samples = parse_prometheus(text)
    usage, waiting = _first(samples, KV_USAGE), _first(samples, WAITING)
    running, preempted = _first(samples, RUNNING), _first(samples, PREEMPTIONS)
    return EngineLoad(
        kv_usage=max(usage) if usage else None,
        waiting=sum(waiting) if waiting is not None else None,
        running=sum(running) if running is not None else None,
        preemptions=sum(preempted) if preempted is not None else None,
        kv_tokens=kv_capacity(samples),
    )


class ServerMetrics:
    """The load of a vLLM server, read over HTTP (blocking: run it off the event loop)."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(f"{self.url}{path}", timeout=self.timeout) as response:
            return response.read()

    def load(self) -> EngineLoad | None:
        try:
            return load_from_prometheus(self._get("/metrics").decode("utf-8", errors="replace"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            logger.warning(
                "rollout concurrency: %s/metrics could not be read (%r); the cap is unchanged", self.url, error
            )
            return None

    def max_model_len(self) -> int | None:
        try:
            models = json.loads(self._get("/v1/models"))
            lengths = [int(m["max_model_len"]) for m in models.get("data", []) if m.get("max_model_len")]
            return max(lengths) if lengths else None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError, KeyError, TypeError):
            return None


# ── the controller ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConcurrencySettings:
    minimum: int = 4
    maximum: int = 256
    low: float = 0.6  # raise below this KV-cache usage
    high: float = 0.8  # cut above it
    poll_s: float = 5.0
    max_model_len: int = 8192  # what one request may reach (the start: KV tokens / this)


def settings_from(config) -> ConcurrencySettings | None:
    """The controller's settings from an RLConfig (None: rollout.concurrency fixed)."""
    r = config.rollout
    if r.concurrency != "kv":
        return None
    from palingenesis.rl.config import max_num_seqs

    maximum = r.concurrency_max or max_num_seqs(config)
    return ConcurrencySettings(
        minimum=min(r.concurrency_min, maximum),
        maximum=maximum,
        low=r.kv_usage_low,
        high=r.kv_usage_high,
        poll_s=r.concurrency_poll_s,
        max_model_len=r.max_model_len,
    )


class KVController:
    """The cap on one engine's requests in flight (see the module docstring). Thread-safe."""

    def __init__(self, settings: ConcurrencySettings, name: str = ""):
        self.settings, self.name = settings, name
        self.lock = threading.Lock()
        self.cap = settings.maximum
        self.started = False
        self.at_change = 0  # requests finished when the cap last changed
        self.waiting_polls = 0
        self.cuts = self.raises = 0
        self.missing = 0  # polls without the metrics the controller needs
        self.warned = False
        self.last = EngineLoad()
        self.preemptions_seen: float | None = None
        self.preempted = 0.0  # preemptions since the start (as the engine counts them)

    def _clamp(self, value: float) -> int:
        return int(min(self.settings.maximum, max(self.settings.minimum, value)))

    def start(self, kv_tokens: int | None, max_model_len: int | None = None) -> int:
        """The first cap: KV tokens / max_model_len (as many requests as fit at full length), else the maximum."""
        with self.lock:
            length = max_model_len or self.settings.max_model_len
            if kv_tokens and length:
                self.cap = self._clamp(kv_tokens // length)
            else:
                self.cap = self.settings.maximum
                logger.warning(
                    "rollout concurrency%s: the engine does not report its KV-cache size; starting at "
                    "rollout.concurrency_max (%d)",
                    f" ({self.name})" if self.name else "",
                    self.cap,
                )
            self.started = True
            return self.cap

    def observe(self, load: EngineLoad | None, in_flight: int, finished: int, saturated: bool) -> int:
        """Move the cap after a poll. `in_flight`: requests out to the engine now; `finished`: requests it has
        answered since the start; `saturated`: requests waited at the gate since the last poll. Returns the cap."""
        with self.lock:
            if load is None or load.kv_usage is None or load.waiting is None:
                self.missing += 1
                if load is not None and not self.warned:  # (None: the reading failed, and its reader said why)
                    self.warned = True
                    logger.warning(
                        "rollout concurrency%s: the engine's metrics have no KV-cache usage or queue length; the cap "
                        "stays at %d",
                        f" ({self.name})" if self.name else "",
                        self.cap,
                    )
                return self.cap
            self.last = load
            if load.preemptions is not None:
                if self.preemptions_seen is not None and load.preemptions >= self.preemptions_seen:
                    self.preempted += load.preemptions - self.preemptions_seen
                self.preemptions_seen = load.preemptions
            self.waiting_polls = self.waiting_polls + 1 if load.waiting > 0 else 0
            overloaded = load.kv_usage > self.settings.high or self.waiting_polls >= WAITING_POLLS
            if overloaded and in_flight <= self.cap:  # once per drain: cuts do not compound while still above
                cap = self._clamp(int(self.cap * CUT))
                if cap != self.cap:
                    self.cap, self.at_change, self.cuts = cap, finished, self.cuts + 1
                    logger.info(
                        "rollout concurrency%s: cap cut to %d (KV-cache usage %.2f, %d waiting)",
                        f" ({self.name})" if self.name else "",
                        cap,
                        load.kv_usage,
                        int(load.waiting),
                    )
                self.waiting_polls = 0
            elif (
                load.kv_usage < self.settings.low
                and load.waiting == 0
                and saturated
                and finished - self.at_change >= self.cap
            ):
                cap = self._clamp(math.ceil(self.cap * GROW))
                if cap != self.cap:
                    self.cap, self.at_change, self.raises = cap, finished, self.raises + 1
            return self.cap

    def snapshot(self, reset: bool = False) -> dict:
        """The cap, the last load, the cuts and raises since the start and the preemptions since the last reset,
        read together (`reset`: start counting preemptions again)."""
        with self.lock:
            state = {
                "cap": self.cap,
                "last": self.last,
                "cuts": self.cuts,
                "raises": self.raises,
                "preempted": self.preempted,
            }
            if reset:
                self.preempted = 0.0
            return state


# ── the gate requests acquire ──────────────────────────────────────────────────


@dataclass
class _Slot:
    controller: KVController
    in_flight: int = 0
    finished: int = 0
    saturated: bool = False  # a request waited for this engine since the last poll
    waiting: int = 0  # requests waiting at the gate that may go to this engine


@dataclass
class _Totals:
    waits: int = 0  # requests that waited at the gate
    wait_s: float = 0.0
    admitted: int = 0
    peak: int = 0


@dataclass(eq=False)
class _Waiter:
    future: asyncio.Future  # resolved with the engine key the request goes to
    keys: list[str]  # the engines it may go to


class AdmissionGate:
    """Requests in flight to one or more engines, each under its controller's cap. Used from one event loop:
    `acquire(keys)` returns the engine key a request goes to (the one with the most room among `keys`), waiting
    while every one of them is at its cap; `release(key)` when the engine is done with the request.

    Order. Waiting requests are admitted first come, first served, per engine: a slot that frees on an engine
    goes to the request that has waited longest among those that may use it. Requests with `priority` (the later
    turns of an episode that is already running) go before all the others: an episode in flight holds an
    environment, its engine's prefix cache and a place in its batch, so finishing it comes before starting a new
    one, and its turns are not starved by first turns when the cap is cut.

    Release. A slot is the engine's work, not the caller's wait: a caller that gives up (a cancelled rollout)
    must release only when the engine has finished or aborted the request, or the cap would undercount what the
    engine is running (GenerationClient and the engine pool release from the request's own completion).

    The wait at the gate is part of the caller's time: a timeout around an episode (scenario.timeout) counts it.

    `fetch(key)`: an engine's load (a coroutine; None when it could not be read). `capacity(key)`: (KV-cache tokens,
    max_model_len) for the first cap, read on the first acquire. The poller runs on the gate's loop from the first
    acquire until `close()`. `metrics()` may be called from any thread."""

    def __init__(
        self,
        settings: ConcurrencySettings,
        keys: list[str],
        fetch: Callable[[str], Awaitable[EngineLoad | None]],
        capacity: Callable[[str], Awaitable[tuple[int | None, int | None]]] | None = None,
    ):
        self.settings = settings
        self.slots = {key: _Slot(KVController(settings, key)) for key in keys}
        self.fetch, self.capacity = fetch, capacity
        self.poller: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.starting: asyncio.Task | None = None
        # requests waiting for a slot, oldest first: continuing episodes (priority), then the others
        self.waiters: tuple[collections.deque[_Waiter], collections.deque[_Waiter]] = (
            collections.deque(),
            collections.deque(),
        )
        self.totals = _Totals()
        self.lock = threading.Lock()  # the slots' counts and the totals: the loop writes, metrics() reads and resets
        self.closed = False

    # -------------------------------------------------------------- admission

    async def _start(self) -> None:
        async def one(key: str, slot: _Slot) -> None:
            kv_tokens = length = None
            try:
                if self.capacity is not None:
                    kv_tokens, length = await self.capacity(key)
                else:
                    load = await self.fetch(key)
                    kv_tokens = load.kv_tokens if load is not None else None
            except Exception:  # noqa: BLE001 — the controller starts at its maximum instead
                logger.warning("rollout concurrency: the KV capacity of %s could not be read", key, exc_info=True)
            slot.controller.start(kv_tokens, length)

        await asyncio.gather(*(one(key, slot) for key, slot in self.slots.items()))
        logger.info(
            "rollout concurrency: caps start at %s",
            ", ".join(f"{key or 'engine'} {slot.controller.cap}" for key, slot in self.slots.items()),
        )

    async def acquire(self, keys: list[str] | None = None, priority: bool = False) -> str:
        """The engine key a request goes to, once one of `keys` (None: any engine) has room under its cap.
        `priority`: a later turn of a running episode, admitted before requests that start one."""
        if self.starting is None:
            self.loop = asyncio.get_running_loop()
            self.starting = asyncio.ensure_future(self._start())
            self.poller = asyncio.ensure_future(self._poll())
        await asyncio.shield(self.starting)
        candidates = [k for k in (keys if keys is not None else self.slots) if k in self.slots]
        if not candidates:
            raise ValueError("no engine to send the request to")
        waiter = _Waiter(asyncio.get_running_loop().create_future(), candidates)
        queue = self.waiters[0 if priority else 1]
        queue.append(waiter)
        for k in candidates:
            self.slots[k].waiting += 1
        self._grant()
        if waiter.future.done():  # admitted at once
            return waiter.future.result()
        for k in candidates:
            self.slots[k].saturated = True
        start = time.perf_counter()
        try:
            await waiter.future
        except BaseException:  # cancelled while waiting: leave the queue, give back a slot granted meanwhile
            if waiter.future.done() and not waiter.future.cancelled():
                self._give_back(waiter.future.result())
            else:
                self._leave(queue, waiter)
            raise
        key = waiter.future.result()
        with self.lock:
            self.totals.waits += 1
            self.totals.wait_s += time.perf_counter() - start
        return key

    def _leave(self, queue: collections.deque, waiter: _Waiter) -> None:
        queue.remove(waiter)
        for k in waiter.keys:
            self.slots[k].waiting -= 1

    def _grant(self) -> None:
        """Admit the waiting requests that have room, oldest first, continuing episodes before new ones. A request
        whose engines are all at their cap keeps its place; one behind it that may use another engine goes ahead
        (it takes nothing the first could have used). Stops as soon as no engine with waiting requests has room,
        so a call costs the requests it admits (plus those it passes over for another engine)."""
        for queue in self.waiters:
            for waiter in list(queue) if self._room() else ():
                free = [k for k in waiter.keys if self.slots[k].in_flight < self.slots[k].controller.cap]
                if not free:
                    continue
                key = min(free, key=lambda k: self.slots[k].in_flight / max(self.slots[k].controller.cap, 1))
                self._leave(queue, waiter)
                with self.lock:
                    self.slots[key].in_flight += 1
                    self.totals.admitted += 1
                    self.totals.peak = max(self.totals.peak, sum(s.in_flight for s in self.slots.values()))
                waiter.future.set_result(key)
                if not self._room():
                    return

    def _room(self) -> bool:
        """Whether an engine that requests wait for has room under its cap."""
        return any(s.waiting and s.in_flight < s.controller.cap for s in self.slots.values())

    def _give_back(self, key: str) -> None:
        """A slot granted to a request that was cancelled before it used it."""
        with self.lock:
            self.slots[key].in_flight -= 1
            self.totals.admitted -= 1
        self._grant()

    def release(self, key: str) -> None:
        """The engine is done with a request admitted to `key` (finished, failed or aborted). On the gate's loop."""
        with self.lock:
            slot = self.slots[key]
            slot.in_flight -= 1
            slot.finished += 1
        self._grant()

    # ---------------------------------------------------------------- polling

    async def _poll(self) -> None:
        await asyncio.shield(self.starting)
        while not self.closed:
            await asyncio.sleep(self.settings.poll_s)
            loads = await asyncio.gather(*(self.fetch(key) for key in self.slots), return_exceptions=True)
            for (key, slot), load in zip(self.slots.items(), loads):
                if isinstance(load, BaseException):
                    logger.warning("rollout concurrency: reading the load of %s failed: %r", key or "the engine", load)
                    load = None
                saturated = slot.saturated or slot.waiting > 0
                slot.controller.observe(load, slot.in_flight, slot.finished, saturated)
                slot.saturated = False
            self._grant()  # a raised cap admits waiting requests

    def close(self) -> None:
        """Stop polling (from any thread)."""
        self.closed = True
        tasks = [t for t in (self.poller, self.starting) if t is not None]
        if self.loop is not None and not self.loop.is_closed():
            for task in tasks:
                self.loop.call_soon_threadsafe(task.cancel)

    # ---------------------------------------------------------------- metrics

    def metrics(self) -> dict[str, float]:
        """rollout/concurrency_cap (summed over engines), rollout/concurrency (requests in flight now),
        rollout/concurrency_peak, rollout/kv_usage (the highest), rollout/requests_waiting and
        rollout/requests_running (summed), rollout/preemptions (since the last call), rollout/gate_waits and
        rollout/gate_wait_s (requests that waited at the gate, and their total wait, since the last call),
        rollout/cap_cuts and rollout/cap_raises (since the start). From any thread: the counters are read and
        reset under the locks the loop updates them with."""
        states = [slot.controller.snapshot(reset=True) for slot in self.slots.values()]
        with self.lock:
            in_flight = sum(s.in_flight for s in self.slots.values())
            out = {
                "rollout/concurrency_cap": float(sum(state["cap"] for state in states)),
                "rollout/concurrency": float(in_flight),
                "rollout/concurrency_peak": float(max(self.totals.peak, in_flight)),
                "rollout/gate_waits": float(self.totals.waits),
                "rollout/gate_wait_s": self.totals.wait_s,
            }
            self.totals.waits, self.totals.wait_s = 0, 0.0
            self.totals.peak = in_flight
        out["rollout/cap_cuts"] = float(sum(state["cuts"] for state in states))
        out["rollout/cap_raises"] = float(sum(state["raises"] for state in states))
        out["rollout/preemptions"] = float(sum(state["preempted"] for state in states))
        loads = [state["last"] for state in states]
        usage = [x.kv_usage for x in loads if x.kv_usage is not None]
        if usage:
            out["rollout/kv_usage"] = max(usage)
        for name, values in (("waiting", [x.waiting for x in loads]), ("running", [x.running for x in loads])):
            known = [v for v in values if v is not None]
            if known:
                out[f"rollout/requests_{name}"] = float(sum(known))
        return out


def merge_metrics(parts: list[dict[str, float]]) -> dict[str, float]:
    """The metrics of several gates (one per engine kind of a scenario) as one: the highest KV-cache usage, the rest
    summed."""
    out: dict[str, float] = {}
    for part in parts:
        for key, value in part.items():
            out[key] = max(out.get(key, value), value) if key == "rollout/kv_usage" else out.get(key, 0.0) + value
    return out


def server_gate(settings: ConcurrencySettings, url: str) -> AdmissionGate:
    """The gate of one vLLM server (rollout.backend vllm_server), polled at `url`/metrics."""
    server = ServerMetrics(url)

    async def fetch(key: str) -> EngineLoad | None:
        return await asyncio.get_running_loop().run_in_executor(None, server.load)

    async def capacity(key: str) -> tuple[int | None, int | None]:
        loop = asyncio.get_running_loop()
        load = await loop.run_in_executor(None, server.load)
        length = await loop.run_in_executor(None, server.max_model_len)
        return (load.kv_tokens if load is not None else None), length

    return AdmissionGate(settings, [""], fetch, capacity)


def gate_for(config, engine) -> AdmissionGate | None:
    """The gate of a rollout engine under rollout.concurrency kv: a vLLM server's (the engine pool of
    rollout.backend remote has its own, palingenesis.rl.pool.client); None otherwise."""
    settings = settings_from(config)
    server = getattr(engine, "server", None)
    if settings is None or server is None or not getattr(server, "url", ""):
        return None
    return server_gate(settings, server.url)


# ── the token allowance ────────────────────────────────────────────────────────


class TokenAllowance:
    """How far generation may run ahead of the batch it collects, in tokens (rollout.token_allowance).

    While a batch of `target` groups is collected, the pipeline holds tokens that will train in it or later: its
    kept groups, groups carried over from earlier batches, and the groups in flight (estimated at group_size x the
    mean context tokens, prompt and everything after it, of the rollouts finished so far). A group launched beyond
    what the batch still needs (the speculative launches that cover groups expected to have zero variance, and
    with them what will be carried into later batches or trained after more weight updates) starts only while

        held + in-flight estimate + its own estimate <= (1 + allowance) x target x group_size x mean tokens

    so the tokens generated ahead of a batch stay within `allowance` batches' worth, and with them the share of
    off-policy tokens a partial rollout or a carried group brings into later batches. Launches a batch needs are
    never held back (a batch always fills). Until a rollout has finished there is no mean, and only the needed
    launches start. After TRL AsyncGRPOTrainer's token allowance (`_fits_token_allowance`)."""

    def __init__(self, allowance: float, group_size: int, smoothing: float = 0.2):
        self.allowance, self.group_size, self.smoothing = allowance, group_size, smoothing
        self.tokens_per_rollout = 0.0
        self.deferred = 0  # launches held back since the last metrics call
        self.ahead = 0.0  # the held + in-flight estimate at the last decision

    def observe(self, tokens: list[int]) -> None:
        """Context tokens of a finished group's rollouts."""
        if not tokens:
            return
        mean = sum(tokens) / len(tokens)
        if not self.tokens_per_rollout:
            self.tokens_per_rollout = mean
        else:
            self.tokens_per_rollout += self.smoothing * (mean - self.tokens_per_rollout)

    def launches(self, needed: int, wanted: int, target: int, held_tokens: float, in_flight_groups: int) -> int:
        """Groups to launch now: `needed` (what the batch still lacks) always, up to `wanted` while within the
        allowance. `held_tokens`: tokens of the finished groups the pipeline holds (kept and carried)."""
        needed, wanted = max(0, needed), max(0, wanted)
        group = self.group_size * self.tokens_per_rollout
        self.ahead = held_tokens + in_flight_groups * group
        if wanted <= needed:
            return wanted
        if not group:
            self.deferred += wanted - needed
            return needed
        limit = (1.0 + self.allowance) * target * group
        fit = int(math.floor((limit - self.ahead) / group + 1e-9)) - needed
        n = needed + max(0, min(wanted - needed, fit))
        self.deferred += wanted - n
        return n

    def metrics(self) -> dict[str, float]:
        out = {
            "rollout/tokens_per_rollout": self.tokens_per_rollout,
            "rollout/ahead_tokens": self.ahead,
            "rollout/launches_deferred": float(self.deferred),
        }
        self.deferred = 0
        return out
