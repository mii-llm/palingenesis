"""Rollout concurrency (palingenesis.rl.concurrency, rollout.concurrency kv) and the token allowance
(rollout.token_allowance): reading vLLM's metrics, the controller's start / cut / raise / bounds, the gate's
admission against a fake metrics feed, a vLLM server's metrics over HTTP, engine workers that report their load,
and how many groups the pipeline launches ahead of a batch."""

import asyncio
import http.server
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from palingenesis.rl.concurrency import (
    AdmissionGate,
    ConcurrencySettings,
    EngineLoad,
    KVController,
    ServerMetrics,
    TokenAllowance,
    load_from_prometheus,
    merge_metrics,
    parse_prometheus,
    server_gate,
    settings_from,
)
from palingenesis.rl.config import RLConfig, RLConfigError

# A vLLM V1 server's /metrics, abridged (two engines behind one API server: data parallel 2)
VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-0.6B"} 37.0
vllm:num_requests_running{engine="1",model_name="Qwen/Qwen3-0.6B"} 35.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-0.6B"} 0.0
vllm:num_requests_waiting{engine="1",model_name="Qwen/Qwen3-0.6B"} 3.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-0.6B"} 0.4123
vllm:kv_cache_usage_perc{engine="1",model_name="Qwen/Qwen3-0.6B"} 0.8551
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 12.0
vllm:num_preemptions_total{engine="1",model_name="Qwen/Qwen3-0.6B"} 3.0
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.3",num_gpu_blocks="20480",num_cpu_blocks="None",prefix_caching_hash_algo="sha256"} 1.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",engine="1",gpu_memory_utilization="0.3",num_gpu_blocks="20480",num_cpu_blocks="None",prefix_caching_hash_algo="sha256"} 1.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 1.234567e+06
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="Qwen/Qwen3-0.6B"} 4021.0
python_gc_objects_collected_total{generation="0"} 2913.0
"""


def settings(**overrides) -> ConcurrencySettings:
    base = {"minimum": 2, "maximum": 64, "low": 0.6, "high": 0.8, "poll_s": 0.01, "max_model_len": 1000}
    return ConcurrencySettings(**{**base, **overrides})


# ------------------------------------------------------------------- metrics


def test_vllm_metrics_are_read_from_a_real_exposition():
    samples = parse_prometheus(VLLM_METRICS)
    assert samples["vllm:num_requests_running"][1] == ({"engine": "1", "model_name": "Qwen/Qwen3-0.6B"}, 35.0)
    assert samples["vllm:e2e_request_latency_seconds_bucket"][0][0]["le"] == "+Inf"
    assert samples["vllm:prompt_tokens_total"][0][1] == 1234567.0
    load = load_from_prometheus(VLLM_METRICS)
    # the highest usage of the server's engines, the summed queues, KV tokens from blocks x block size
    assert load == EngineLoad(kv_usage=0.8551, waiting=3.0, running=72.0, preemptions=15.0, kv_tokens=2 * 20480 * 16)


def test_older_names_explicit_kv_size_and_odd_lines():
    text = "\n".join(
        [
            'vllm:gpu_cache_usage_perc{model_name="m"} 0.25',  # vLLM V0's name
            'vllm:num_requests_waiting{model_name="m"} 0',
            'vllm:num_preemptions_total{model_name="m"} NaN',  # not finite: not counted
            'vllm:cache_config_info{kv_cache_size_tokens="123456",block_size="16",num_gpu_blocks="9"} 1',
            'weird{label="with \\"quotes\\", and commas"} 1.0',
            "a line that is not a sample",
            "",
        ]
    )
    samples = parse_prometheus(text)
    assert samples["weird"][0][0]["label"] == 'with "quotes", and commas'
    load = load_from_prometheus(text)
    assert (load.kv_usage, load.waiting, load.running, load.kv_tokens) == (0.25, 0.0, None, 123456)
    assert load.preemptions == 0.0


def test_metrics_without_the_controller_s_inputs_are_none():
    load = load_from_prometheus("# nothing here\nprocess_cpu_seconds_total 3.0\n")
    assert load == EngineLoad()


# ---------------------------------------------------------------- controller


def test_the_cap_starts_at_kv_tokens_over_max_model_len_within_bounds():
    c = KVController(settings())
    assert c.start(kv_tokens=20_000, max_model_len=1000) == 20
    assert c.start(kv_tokens=500, max_model_len=1000) == 2  # the minimum
    assert c.start(kv_tokens=10**9, max_model_len=1000) == 64  # the maximum
    assert c.start(kv_tokens=20_000, max_model_len=None) == 20  # the configured max_model_len
    assert c.start(kv_tokens=None) == 64  # unknown capacity: the maximum


def test_high_usage_cuts_once_per_drain_and_never_below_the_minimum():
    c = KVController(settings())
    c.start(20_000)
    high = EngineLoad(kv_usage=0.9, waiting=0.0)
    assert c.observe(high, in_flight=20, finished=0, saturated=True) == 14  # 20 x 0.7
    # still 20 in flight, above the new cap: no second cut until they drain below it
    assert c.observe(high, in_flight=20, finished=1, saturated=True) == 14
    assert c.observe(high, in_flight=14, finished=7, saturated=True) == 9
    for _ in range(10):
        c.observe(high, in_flight=0, finished=7, saturated=False)
    assert c.cap == 2 and c.cuts == 5


def test_requests_waiting_on_two_polls_in_a_row_cut():
    c = KVController(settings())
    c.start(20_000)
    waiting = EngineLoad(kv_usage=0.5, waiting=4.0)
    assert c.observe(waiting, 20, 0, True) == 20  # one poll may be a burst
    assert c.observe(EngineLoad(kv_usage=0.5, waiting=0.0), 20, 0, True) == 20  # (the streak is broken)
    assert c.observe(waiting, 20, 0, True) == 20
    assert c.observe(waiting, 20, 0, True) == 14


def test_low_usage_raises_only_when_the_cap_binds_and_after_a_turnover():
    c = KVController(settings())
    c.start(10_000)  # 10
    low = EngineLoad(kv_usage=0.3, waiting=0.0)
    assert c.observe(low, 10, finished=4, saturated=True) == 10  # fewer than half a cap's worth finished
    assert c.observe(low, 3, finished=50, saturated=False) == 10  # nobody waited at the gate: no demand
    assert c.observe(EngineLoad(kv_usage=0.3, waiting=1.0), 10, 50, True) == 10  # the engine queues
    assert c.observe(EngineLoad(kv_usage=0.7, waiting=0.0), 10, 50, True) == 10  # between low and high: hold
    assert c.observe(low, 10, finished=50, saturated=True) == 13  # ceil(10 x 1.25)
    assert c.observe(low, 13, finished=55, saturated=True) == 13  # the turnover counts from the change
    assert c.observe(low, 13, finished=56, saturated=True) == 17
    idle = EngineLoad(kv_usage=0.2, waiting=0.0)  # under half of `low`: x 1.5
    assert c.observe(idle, 17, finished=70, saturated=True) == 26
    for finished in range(100, 10_000, 100):
        c.observe(low, c.cap, finished, True)
    assert c.cap == 64 and c.raises >= 3  # the maximum


def test_finished_requests_size_the_cap_without_waiting_for_a_turnover():
    """KV tokens / max_model_len is the worst case: once requests have finished, their mean length gives what fits
    under `high` even if every request in flight reached it at once."""
    c = KVController(settings(maximum=512))
    assert c.start(74_704, 4096) == 18
    low = EngineLoad(kv_usage=0.1, waiting=0.0)
    for _ in range(7):
        c.note_finished(1152)
    assert c.observe(low, 18, finished=7, saturated=True) == 18  # too few measured, no turnover yet
    c.note_finished(1152)
    c.note_finished(0)  # a failed request tells nothing
    assert c.observe(low, 18, finished=8, saturated=True) == 51 and c.raised  # 0.8 x 74704 / 1152
    assert c.observe(low, 51, finished=20, saturated=True) == 51 and not c.raised  # beyond it: a turnover first
    assert c.observe(EngineLoad(kv_usage=0.5, waiting=0.0), 51, finished=34, saturated=True) == 64
    for _ in range(200):  # the requests get longer: the bound follows, the cap is not pulled down by it
        c.note_finished(4000)
    assert c.observe(low, 64, finished=40, saturated=True) == 64
    # the expected length given up front: the cap starts there
    assert KVController(settings(maximum=512, request_tokens=1152)).start(74_704, 4096) == 51


def test_preemptions_cut_the_cap():
    c = KVController(settings())
    c.start(20_000)  # 20
    assert c.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=5.0), 20, 0, True) == 20  # the first reading
    assert c.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=5.0), 20, 0, True) == 20
    assert c.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=8.0), 20, 0, True) == 14 and c.cuts == 1
    assert c.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=9.0), 20, 0, True) == 14  # once per drain


def test_the_gate_polls_sooner_while_it_raises_and_learns_lengths_from_releases():
    feed = FakeFeed([EngineLoad(kv_usage=0.1, waiting=0.0)], capacity=(40_000, 4000))
    gate = AdmissionGate(settings(minimum=1, maximum=64, poll_s=0.05), [""], feed.fetch, feed.capacity)

    async def run():
        async def request():
            key = await gate.acquire()
            await asyncio.sleep(0.002)
            gate.release(key, tokens=1000)

        await asyncio.gather(*(request() for _ in range(300)))
        gate.close()

    asyncio.run(run())
    controller = gate.slots[""].controller
    assert controller.requests_measured == 300 and controller.request_tokens == pytest.approx(1000)
    assert controller.cap >= 32  # from 10 (40000 / 4000) to 0.8 x 40000 / 1000 in one step, then on


def test_missing_metrics_leave_the_cap_alone(caplog):
    c = KVController(settings())
    c.start(10_000)
    for load in (None, EngineLoad(waiting=0.0), EngineLoad(kv_usage=0.0)):  # no usage reads as an idle engine
        assert c.observe(load, 10, 1000, True) == 10
    assert c.missing == 3
    assert "no KV-cache usage or queue length" in caplog.text


def test_preemptions_are_counted_from_the_first_reading():
    c = KVController(settings())
    c.start(10_000)
    for total in (100.0, 104.0, 104.0, 110.0):
        c.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=total), 0, 0, False)
    assert c.preempted == 10.0


# ----------------------------------------------------------------------- gate


class FakeFeed:
    """Engine loads served in order (the last repeats); a key's capacity is (kv_tokens, max_model_len)."""

    def __init__(self, loads, capacity=(4000, 1000)):
        self.loads, self.capacity_value, self.calls = list(loads), capacity, 0

    async def fetch(self, key):
        self.calls += 1
        load = self.loads[0] if len(self.loads) == 1 else self.loads.pop(0)
        if isinstance(load, BaseException):
            raise load
        return load

    async def capacity(self, key):
        return self.capacity_value


def test_the_gate_holds_requests_at_the_cap_and_raises_it_when_the_engine_has_room():
    feed = FakeFeed([EngineLoad(kv_usage=0.1, waiting=0.0)])
    gate = AdmissionGate(settings(minimum=1, maximum=8, poll_s=0.02), [""], feed.fetch, feed.capacity)

    async def run():
        live, peak = 0, 0

        async def request():
            nonlocal live, peak
            key = await gate.acquire()
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.005)
            live -= 1
            gate.release(key)

        first = await gate.acquire()  # the cap starts at 4000 / 1000
        assert gate.slots[""].controller.cap == 4
        gate.release(first)
        await asyncio.gather(*(request() for _ in range(200)))
        gate.close()
        return peak

    peak = asyncio.run(run())
    controller = gate.slots[""].controller
    assert controller.cap > 4 and controller.raises >= 1  # the engine had room and requests waited
    assert peak <= controller.cap <= 8
    metrics = gate.metrics()
    assert metrics["rollout/concurrency_cap"] == controller.cap and metrics["rollout/kv_usage"] == 0.1
    assert metrics["rollout/gate_waits"] > 0 and metrics["rollout/concurrency"] == 0


def test_an_overloaded_engine_gets_fewer_requests_and_a_failing_feed_changes_nothing():
    feed = FakeFeed([RuntimeError("down"), EngineLoad(kv_usage=0.95, waiting=0.0), EngineLoad(kv_usage=0.7, waiting=0)])
    gate = AdmissionGate(settings(minimum=1, maximum=8, poll_s=0.01), [""], feed.fetch, feed.capacity)

    async def run():
        keys = [await gate.acquire() for _ in range(4)]  # the starting cap, all taken
        waiter = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(
            0.1
        )  # polls: a failure (no change), usage 0.95 with 4 <= cap in flight (a cut), then 0.7 (hold)
        assert not waiter.done() and gate.slots[""].controller.cap == 2
        for key in keys[:2]:
            gate.release(key)
        await asyncio.sleep(0.05)
        assert not waiter.done()  # 2 in flight, cap 2
        gate.release(keys[2])
        key = await asyncio.wait_for(waiter, 5)
        gate.release(key)
        gate.release(keys[3])
        gate.close()

    asyncio.run(run())


def test_a_request_goes_to_the_engine_with_the_most_room():
    feeds = {"a": FakeFeed([EngineLoad(kv_usage=0.7, waiting=0.0)], (2000, 1000))}
    feeds["b"] = FakeFeed([EngineLoad(kv_usage=0.7, waiting=0.0)], (6000, 1000))
    gate = AdmissionGate(
        settings(minimum=1, maximum=16, poll_s=10.0),
        ["a", "b"],
        lambda key: feeds[key].fetch(key),
        lambda key: feeds[key].capacity(key),
    )

    async def run():
        keys = [await gate.acquire() for _ in range(8)]
        assert keys.count("a") == 2 and keys.count("b") == 6  # caps 2 and 6, filled in proportion
        waiter = asyncio.ensure_future(gate.acquire(["a"]))  # only "a" may serve it (e.g. "b" went away)
        gate.release("b")
        await asyncio.sleep(0.05)
        assert not waiter.done()
        gate.release("a")
        assert await asyncio.wait_for(waiter, 5) == "a"
        gate.close()
        return gate.metrics()

    metrics = asyncio.run(run())
    assert metrics["rollout/concurrency_cap"] == 8 and metrics["rollout/concurrency"] == 7


def _still_gate(cap: int, keys=("",), maximum: int = 16) -> AdmissionGate:
    """A gate whose caps stay at `cap` (no poll within the test)."""
    feed = FakeFeed([EngineLoad(kv_usage=0.7, waiting=0.0)], (1000 * cap, 1000))
    return AdmissionGate(settings(minimum=1, maximum=maximum, poll_s=60.0), list(keys), feed.fetch, feed.capacity)


def test_waiting_requests_are_admitted_first_come_first_served():
    gate = _still_gate(1)

    async def run():
        order = []

        async def request(i):
            key = await gate.acquire()
            order.append(i)
            await asyncio.sleep(0.001)
            gate.release(key)

        held = await gate.acquire()
        tasks = []
        for i in range(12):
            tasks.append(asyncio.ensure_future(request(i)))
            await asyncio.sleep(0)  # each reaches the gate before the next starts
        await asyncio.sleep(0.01)
        late = asyncio.ensure_future(request("late"))  # arrives while the others wait: it does not barge
        gate.release(held)
        await asyncio.gather(*tasks, late)
        gate.close()
        return order

    assert asyncio.run(run()) == [*range(12), "late"]


def test_continuing_episodes_go_before_new_ones_and_engines_do_not_block_each_other():
    gate = _still_gate(1, keys=("a", "b"))

    async def run():
        order = []

        async def request(name, keys=None, priority=False):
            key = await gate.acquire(keys, priority=priority)
            order.append((name, key))

        assert sorted([await gate.acquire(), await gate.acquire()]) == ["a", "b"]
        tasks = [asyncio.ensure_future(request("new-1", ["a"]))]
        await asyncio.sleep(0)
        tasks.append(asyncio.ensure_future(request("new-2")))
        await asyncio.sleep(0)
        tasks.append(asyncio.ensure_future(request("turn-2", ["a"], priority=True)))
        await asyncio.sleep(0.01)
        gate.release("b")  # the oldest waiter cannot use "b": the next one that can takes it
        await asyncio.sleep(0.01)
        assert order == [("new-2", "b")]
        gate.release("a")  # the continuing episode goes before the older first turn
        await asyncio.sleep(0.01)
        assert order == [("new-2", "b"), ("turn-2", "a")]
        gate.release("a")
        await asyncio.gather(*tasks)
        gate.close()
        return order

    assert asyncio.run(run())[-1] == ("new-1", "a")


def test_a_cancelled_waiter_leaves_the_queue_and_takes_no_slot():
    gate = _still_gate(1)

    async def run():
        held = await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        first.cancel()
        await asyncio.sleep(0.01)
        assert not any(gate.waiters[0]) and len(gate.waiters[1]) == 1
        gate.release(held)
        assert await asyncio.wait_for(second, 5) == ""
        # granted and cancelled before it ran again: the slot goes back, to the next in line
        third = asyncio.ensure_future(gate.acquire())
        fourth = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        gate.release("")  # grants `third` (its future is set; its task has not resumed)
        third.cancel()
        assert await asyncio.wait_for(fourth, 5) == ""
        assert gate.slots[""].in_flight == 1
        with pytest.raises(ValueError, match="no engine"):
            await gate.acquire(["gone"])
        assert gate.slots[""].waiting == 0
        gate.close()
        return gate.metrics()

    metrics = asyncio.run(run())
    assert metrics["rollout/concurrency"] == 1 and metrics["rollout/gate_waits"] == 2


def test_a_slot_freed_before_a_cancelled_waiter_resumes_goes_to_the_next_in_line():
    """A waiter cancelled (its future is cancelled at once) whose task has not run again yet is still in the queue
    when a slot frees: the slot must go to the next waiter, not to the cancelled one."""
    gate = _still_gate(1)

    async def run():
        held = await gate.acquire()
        first = asyncio.ensure_future(gate.acquire())
        second = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0.01)
        first.cancel()
        gate.release(held)  # before `first` has seen its cancellation
        assert await asyncio.wait_for(second, 5) == ""
        with pytest.raises(asyncio.CancelledError):
            await first
        assert gate.slots[""].in_flight == 1 and gate.slots[""].waiting == 0
        assert not any(gate.waiters)
        gate.close()

    asyncio.run(run())


def test_a_cancelled_rollout_keeps_its_slot_until_the_engine_is_done():
    """The engine has no abort: a request whose caller was cancelled still generates, so its slot is released when
    the engine resolves it, and a failed request's slot when it fails."""
    from palingenesis.opd.rollout import Rollout
    from palingenesis.rl.generation import GenerationClient

    class SlowEngine:
        def __init__(self):
            self.go, self.calls = threading.Event(), 0

        def generate(self, prompts, max_new_tokens, temperature):
            self.calls += 1
            if prompts[0] == [13]:
                raise RuntimeError("engine failure")
            assert self.go.wait(10)
            return [Rollout([1], [-0.1], "stop", 0) for _ in prompts]

    engine, gate = SlowEngine(), _still_gate(1)
    client = GenerationClient(engine, gate=gate)

    async def run():
        client.enter()
        first = asyncio.ensure_future(client.generate([1], 4, 1.0))
        await asyncio.sleep(0.2)  # in the engine
        second = asyncio.ensure_future(client.generate([2], 4, 1.0, continuing=True))
        await asyncio.sleep(0.05)
        first.cancel()
        await asyncio.sleep(0.2)
        assert first.cancelled() and gate.slots[""].in_flight == 1  # still the engine's
        assert not second.done() and engine.calls == 1  # the second request was not sent on top of it
        engine.go.set()
        assert (await asyncio.wait_for(second, 10)).ids == [1]
        assert gate.slots[""].finished == 2
        with pytest.raises(RuntimeError, match="engine failure"):
            await client.generate([13], 4, 1.0)
        await asyncio.sleep(0)
        assert gate.slots[""].in_flight == 0 and gate.slots[""].finished == 3

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        client.close()
        loop.close()


def test_metrics_are_read_and_reset_under_the_locks_of_their_writers():
    gate = _still_gate(4)
    controller = gate.slots[""].controller
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            time.sleep(0.0005)
            try:
                gate.metrics()
            except Exception as error:  # noqa: BLE001
                errors.append(error)

    async def run():
        async def request():
            key = await gate.acquire()
            await asyncio.sleep(0)
            gate.release(key)

        await asyncio.gather(*(request() for _ in range(500)))
        gate.close()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(200):  # preemptions arrive while another thread reads and resets them
            controller.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=float(i)), 0, 0, False)
        asyncio.run(run())
    finally:
        stop.set()
        thread.join()
    assert not errors and gate.slots[""].in_flight == 0 and gate.slots[""].finished == 500
    assert gate.totals.admitted == 500
    # every preemption was reported exactly once: none lost between a read and its reset
    controller.observe(EngineLoad(kv_usage=0.7, waiting=0.0, preemptions=250.0), 0, 0, False)
    snapshot = controller.snapshot(reset=True)
    assert snapshot["preempted"] <= 250.0 - 0.0 and controller.snapshot()["preempted"] == 0.0
    assert gate.metrics()["rollout/preemptions"] == 0.0


def test_merged_metrics_take_the_highest_usage_and_sum_the_rest():
    merged = merge_metrics(
        [
            {"rollout/kv_usage": 0.2, "rollout/concurrency_cap": 8.0},
            {"rollout/kv_usage": 0.7, "rollout/concurrency_cap": 4.0},
        ]
    )
    assert merged == {"rollout/kv_usage": 0.7, "rollout/concurrency_cap": 12.0}


# ------------------------------------------------------- a server over HTTP


class _Handler(http.server.BaseHTTPRequestHandler):
    metrics = VLLM_METRICS

    def do_GET(self):  # noqa: N802 — http.server's name
        if self.path == "/metrics":
            body = self.metrics.encode()
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "Qwen/Qwen3-0.6B", "max_model_len": 4096}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_a_vllm_server_s_metrics_over_http_start_the_cap(server):
    assert ServerMetrics(server).max_model_len() == 4096
    assert ServerMetrics(server).load().kv_tokens == 655360
    assert ServerMetrics("http://127.0.0.1:9").load() is None  # nothing listens there: no reading, no error
    gate = server_gate(settings(maximum=1024, poll_s=60.0), server)

    async def run():
        key = await gate.acquire()
        gate.release(key)
        gate.close()

    asyncio.run(run())
    assert gate.slots[""].controller.cap == 655360 // 4096  # 160 requests of 4096 tokens fit


def test_the_generation_client_keeps_a_server_s_requests_under_the_cap():
    from palingenesis.opd.rollout import Rollout
    from palingenesis.rl.generation import GenerationClient

    class BatchEngine:
        def __init__(self):
            self.batches = []

        def generate(self, prompts, max_new_tokens, temperature):
            self.batches.append(len(prompts))
            time.sleep(0.01)
            return [Rollout([1], [-0.1], "stop", 0) for _ in prompts]

    engine = BatchEngine()
    feed = FakeFeed([EngineLoad(kv_usage=0.5, waiting=0.0)], capacity=(3000, 1000))
    gate = AdmissionGate(settings(minimum=1, maximum=3, poll_s=60.0), [""], feed.fetch, feed.capacity)
    client = GenerationClient(engine, gate=gate)

    async def run():
        for _ in range(10):
            client.enter()
        out = await asyncio.gather(*(client.generate([i], 4, 1.0) for i in range(10)))
        return out

    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(run())
    finally:
        client.close()
        loop.close()
    assert len(out) == 10 and max(engine.batches) <= 3 and sum(engine.batches) == 10


# ------------------------------------------------------ remote engine workers

ROOT = Path(__file__).parent.parent


class LoadedEngine:
    """A batch engine for the pool test that reports a load (2 requests of max_model_len fit its KV cache) and
    records how many requests it holds at once; generate calls wait while the PGS_TEST_HOLD file exists."""

    def __init__(self):
        self.version, self.hold, self.seen = 0, os.environ.get("PGS_TEST_HOLD", ""), []

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter=""):
        from palingenesis.opd.rollout import Rollout

        self.seen.append(len(prompts))
        while self.hold and os.path.exists(self.hold):
            time.sleep(0.01)
        return [Rollout([len(self.seen), max(self.seen)], [-0.5, -0.5], "stop", -1) for _ in prompts]

    def load_stats(self):
        return {"kv_usage": 0.7, "waiting": 0.0, "running": float(self.seen[-1] if self.seen else 0), "kv_tokens": 64}

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def loaded_engine():
    return LoadedEngine()


def test_engine_workers_report_their_load_and_the_pool_keeps_each_under_its_cap(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_rl_pool import start_worker

    from palingenesis.rl.pool.client import RemotePool

    hold = tmp_path / "hold"
    hold.write_text("")
    process, address = start_worker(
        "--engine", f"{__file__}:loaded_engine", env={"PGS_TEST_HOLD": str(hold), "PYTHONPATH": f"{ROOT / 'src'}"}
    )
    pool = None
    try:
        pool = RemotePool(
            [address],
            identity={"max_model_len": 32},  # (the scripted worker checks nothing)
            concurrency=settings(minimum=1, maximum=16, poll_s=60.0, max_model_len=32),
        )
        futures = [asyncio.run_coroutine_threadsafe(pool._request([1, i], 4, 1.0, ""), pool.loop) for i in range(5)]
        time.sleep(0.5)
        assert pool.gate.slots[address].controller.cap == 2  # 64 KV tokens / max_model_len 32
        assert pool.gate.slots[address].in_flight == 2 and not any(f.done() for f in futures)
        # a caller that gives up leaves its request on the worker (no abort): the slot stays taken
        running = [f for f in futures if not f.done()]
        futures[0].cancel()
        time.sleep(0.3)
        assert pool.gate.slots[address].in_flight == 2
        hold.unlink()
        replies = [f.result(30) for f in futures[1:]]
        assert len(running) == 5
        assert max(ids[1] for ids in (r.ids for r in replies)) <= 2  # the engine never held more than 2 at once
        deadline = time.time() + 10
        while pool.gate.slots[address].in_flight and time.time() < deadline:
            time.sleep(0.01)
        assert pool.gate.slots[address].in_flight == 0
        state = pool.state()[0]
        assert state["load"]["kv_tokens"] == 64 and state["load"]["kv_usage"] == 0.7
    finally:
        if pool is not None:
            pool.close()
        process.kill()
        process.wait()


# ---------------------------------------------------------- token allowance


def test_the_allowance_lets_needed_launches_through_and_bounds_the_speculative_ones():
    a = TokenAllowance(allowance=0.5, group_size=4)
    assert a.launches(needed=3, wanted=6, target=4, held_tokens=0, in_flight_groups=0) == 3  # no mean yet
    assert a.metrics()["rollout/launches_deferred"] == 3
    a.observe([100, 100, 100, 100])  # 400 tokens a group, 1600 a batch of 4, 2400 with the allowance
    assert a.launches(needed=0, wanted=4, target=4, held_tokens=800, in_flight_groups=2) == 2  # 800 + 800 + 2 x 400
    assert a.launches(needed=1, wanted=4, target=4, held_tokens=800, in_flight_groups=3) == 1  # full: only the need
    assert a.launches(needed=4, wanted=8, target=4, held_tokens=0, in_flight_groups=0) == 6
    assert a.launches(needed=2, wanted=2, target=4, held_tokens=10**9, in_flight_groups=9) == 2  # never below need
    a.observe([300, 300])
    assert a.tokens_per_rollout == pytest.approx(100 + 0.2 * 200)
    assert a.metrics()["rollout/ahead_tokens"] == 10**9 + 9 * 4 * 100


def _pipeline(tmp_path, allowance: float):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_rl import ScriptedEngine, tokenizer

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.pipeline import RLPipeline
    from palingenesis.rl.rewards import resolve_rewards

    torch = pytest.importorskip("torch")
    tok = tokenizer("Qwen/Qwen3-0.6B")
    kwargs = {"enable_thinking": False}
    eot = end_of_turn_id(tok, kwargs)
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": kwargs,
        "rollout.group_size": 2,
        "rollout.batch_prompts": 4,
        "rollout.max_new_tokens": 16,
        "rollout.max_model_len": 512,
        "rollout.token_allowance": allowance,
    }.items():
        config.set(key, value)
    counter = itertools.count()

    def alternating(completion):  # every group disagrees: informative
        return float(next(counter) % 2)

    rows = [{"prompt": f"Say something about {i}."} for i in range(64)]
    return RLPipeline(
        tok,
        ChatFormat(tok, eot, kwargs),
        ScriptedEngine(tok, eot, ["Something."]),
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        resolve_rewards({}, [alternating]),
        None,
        None,
        (eot,),
    )


@pytest.mark.parametrize("allowance", [0.0, 0.25])
def test_the_pipeline_launches_ahead_only_within_the_allowance(tmp_path, allowance):
    pipeline = _pipeline(tmp_path, allowance)
    try:
        pipeline.drop_rate = 0.5  # as if half the groups had zero variance: 4 needed, 8 launched speculatively
        first = pipeline.run(4, 1.0)
        assert len(first.samples) == 4
        launched = first.stats["groups/launched"]
        if not allowance:
            assert launched == 8 and "rollout/launches_deferred" not in first.stats
            return
        # no rollout has finished yet: only the 4 groups the batch needs start
        assert launched == 4 and first.stats["rollout/launches_deferred"] == 4
        tokens = first.stats["rollout/tokens_per_rollout"]
        assert tokens > 0
        pipeline.drop_rate = 0.5
        second = pipeline.run(4, 1.0)
        # 4 needed, one more within 1.25 x the batch's tokens
        assert second.stats["groups/launched"] == 5 and second.stats["rollout/launches_deferred"] == 3
    finally:
        pipeline.close()


# --------------------------------------------------------------- configuration


def test_concurrency_settings_and_defaults():
    config = RLConfig()
    assert config.rollout.concurrency == "fixed" and settings_from(config) is None
    config.set("rollout.concurrency", "kv")
    config.set("rollout.max_num_seqs", 300)
    s = settings_from(config)
    assert (s.minimum, s.maximum, s.low, s.high, s.poll_s, s.max_model_len) == (4, 300, 0.6, 0.8, 5.0, 8192)
    config.set("rollout.concurrency_max", 2)
    assert settings_from(config).minimum == 2  # never above the maximum


def _config(**overrides) -> RLConfig:
    config = RLConfig()
    base = {"model.policy": "p", "data.dataset": "d", "rewards.correct.fn": "math", "rollout.backend": "vllm_server"}
    for key, value in {**base, **overrides}.items():
        config.set(key, value)
    return config


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout.concurrency": "adaptive"}, "rollout.concurrency must be one of"),
        ({"rollout.concurrency": "kv", "rollout.backend": "vllm"}, "admits and preempts by itself"),
        ({"rollout.concurrency": "kv", "rollout.backend": "hf"}, "generates with the trainer's model"),
        ({"rollout.concurrency": "kv", "rollout.backend": "sglang"}, "SGLang admits by itself"),
        ({"rollout.concurrency": "kv", "rollout.backend": "sglang_server"}, "SGLang's metrics are not vLLM's"),
        ({"rollout.concurrency": "kv", "rollout.concurrency_min": 0}, "concurrency_min must be an integer >= 1"),
        ({"rollout.concurrency": "kv", "rollout.concurrency_max": -1}, "concurrency_max must be an integer >= 0"),
        (
            {"rollout.concurrency": "kv", "rollout.concurrency_min": 8, "rollout.concurrency_max": 4},
            "must be >= rollout.concurrency_min",
        ),
        ({"rollout.concurrency": "kv", "rollout.kv_usage_low": 0.9}, "0 < low < high <= 1"),
        ({"rollout.concurrency": "kv", "rollout.kv_usage_high": 1.5}, "0 < low < high <= 1"),
        ({"rollout.concurrency": "kv", "rollout.concurrency_poll_s": 0}, "concurrency_poll_s must be > 0"),
        ({"rollout.kv_usage_high": 0.9}, "only used with rollout.concurrency kv"),
        ({"rollout.concurrency_max": 64}, "only used with rollout.concurrency kv"),
        ({"rollout.concurrency_request_tokens": 512}, "only used with rollout.concurrency kv"),
        ({"rollout.concurrency": "kv", "rollout.concurrency_request_tokens": -1}, "must be an integer >= 0"),
        (
            {"rollout.concurrency": "kv", "rollout.concurrency_request_tokens": 10**7},
            "is above rollout.max_model_len",
        ),
        ({"rollout.token_allowance": -1.0}, "token_allowance must be a finite number >= 0"),
        ({"rollout.token_allowance": float("inf")}, "token_allowance must be a finite number >= 0"),
    ],
)
def test_concurrency_configuration_errors(overrides, match):
    with pytest.raises(RLConfigError, match=match):
        _config(**overrides).validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout.concurrency": "kv"},
        {"rollout.concurrency": "kv", "rollout.backend": "remote", "rollout.engines": ["h:1"]},
        {"rollout.concurrency": "kv", "rollout.concurrency_min": 1, "rollout.concurrency_max": 512},
        {"rollout.token_allowance": 0.5, "rollout.backend": "vllm"},
    ],
)
def test_valid_concurrency_configurations(overrides):
    _config(**overrides).validate()


def test_concurrency_options_from_yaml(tmp_path):
    path = tmp_path / "rl.yaml"
    path.write_text(
        "model: {policy: p}\ndata: {dataset: d}\nrewards: {correct: math}\n"
        "rollout: {backend: vllm_server, concurrency: kv, kv_usage_high: 0.85, token_allowance: 1}\n"
    )
    config = RLConfig.from_yaml(path)
    config.validate()
    assert config.rollout.token_allowance == 1.0 and isinstance(config.rollout.token_allowance, float)
    assert settings_from(config).high == 0.85
