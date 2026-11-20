"""Rollout engines in other processes (palingenesis.rl.pool, rollout.backend remote): versions across processes,
weight pushes that overlap generation and training, pushes dropped whole when their sender dies, the staleness
bound on what is trained, LoRA adapters sent as factors (folded by the engine, or served as named adapters of one
engine), workers that go away, the first update's base, resume, and trainer ranks with a replicated base sharing
an engine pool. Pushes are held where a test needs them with the worker's --gate file, not with timing."""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy  # noqa: E402
from test_rl_scenarios import CHAT, Chat  # noqa: E402

from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402
from palingenesis.rl.pool.client import RemotePool  # noqa: E402
from palingenesis.rl.pool.protocol import Staging, pack, receive, send  # noqa: E402
from palingenesis.rl.pool.worker import EngineWorker  # noqa: E402

ROOT = Path(__file__).parent.parent


class WireEngine:
    """A scripted engine for the protocol tests: a completion is [its version, the sum of its weights, the
    adapter's first character] (and the top_k of a Sampling), so a reply says exactly which weights produced it.
    While the file named by PGS_TEST_HOLD exists, generate calls wait."""

    def __init__(self):
        self.version, self.total = 0, 0.0
        self.hold = os.environ.get("PGS_TEST_HOLD", "")
        self.calls = []

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter=""):
        from palingenesis.opd.rollout import Rollout, Sampling

        version, total = self.version, self.total
        self.calls.append((adapter, [list(p) for p in prompts]))
        while self.hold and os.path.exists(self.hold):
            time.sleep(0.01)
        ids = [version, int(round(total)), ord(adapter[0]) if adapter else 0]
        if isinstance(temperature, Sampling):
            ids.append(temperature.top_k)
        return [Rollout(list(ids), [-0.5] * len(ids), "stop", -1) for _ in prompts]

    def update_weights(self, named, version):
        self.total = float(sum(tensor.double().sum() for _, tensor in named))
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def scripted_engine():
    return WireEngine()


def start_worker(*args, env=None):
    """An engine worker in its own process: (process, "127.0.0.1:port")."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "palingenesis.rl.pool.worker",
            *(() if "--port" in args else ("--port", "0")),
            *args,
        ],
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}", **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        found = re.search(r"listening on [\w.]+:(\d+)", line)
        if found:
            threading.Thread(target=process.stdout.read, daemon=True).start()  # (its log must not fill the pipe)
            return process, f"127.0.0.1:{found.group(1)}"
        if not line and process.poll() is not None:
            break
    process.kill()
    raise RuntimeError("the engine worker did not start")


@pytest.fixture
def workers():
    started = []

    def start(*args, env=None):
        process, address = start_worker(*args, env=env)
        started.append(process)
        start.processes[address] = process
        return address

    start.processes = {}
    yield start
    for process in started:
        process.kill()
        process.wait()


class Gate:
    """The worker's --gate file: while it is closed, a push to that worker waits where it is."""

    def __init__(self, path):
        self.path = Path(path)

    def close(self):
        self.path.write_text("")

    def open(self):
        self.path.unlink(missing_ok=True)


@pytest.fixture
def gate(tmp_path):
    gate = Gate(tmp_path / "gate")
    yield gate
    gate.open()


def generate(pool, adapter="", temperature=1.0):
    out = asyncio.run_coroutine_threadsafe(pool._request([1, 2, 3], 4, temperature, adapter), pool.loop).result(30)
    return out.ids, out.version


def wait_for(condition, seconds=30):
    """Poll until `condition()` (a bound on a hang, never a measurement)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def in_thread(call, *args):
    """Run `call` on a thread; .error holds what it raised."""

    def run():
        try:
            call(*args)
        except BaseException as error:  # noqa: BLE001
            thread.error = error

    thread = threading.Thread(target=run, daemon=True)
    thread.error = None
    thread.start()
    return thread


SCRIPTED = ("--engine", f"{__file__}:scripted_engine")
MB = 2**20


# ------------------------------------------------------------------- the wire


class Pipe:
    """A writer whose bytes a StreamReader reads back."""

    def __init__(self, limit=None):
        self.data, self.limit = bytearray(), limit

    def write(self, data):
        self.data += bytes(data)

    async def drain(self):
        pass

    def reader(self):
        reader = asyncio.StreamReader()
        reader.feed_data(bytes(self.data if self.limit is None else self.data[: self.limit]))
        reader.feed_eof()
        return reader


def test_frames_carry_every_supported_dtype_and_shape_bit_for_bit():
    tensors = {
        "bf16": torch.randn(3, 5).to(torch.bfloat16),
        "f32": torch.randn(7),
        "f64": torch.randn(2, 2, 2, dtype=torch.float64),
        "i64": torch.arange(6).reshape(2, 3),
        "u8": torch.arange(4, dtype=torch.uint8),
        "flag": torch.tensor([True, False, True]),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty(0, 4),
        "strided": torch.arange(12.0).reshape(3, 4).t(),  # not contiguous
    }
    for name in ("uint16", "uint32", "float8_e4m3fn"):
        if hasattr(torch, name):
            tensors[name] = torch.arange(5, dtype=torch.float32).to(getattr(torch, name))

    async def roundtrip():
        pipe = Pipe()
        sent = await send(pipe, {"op": "weights", "version": 3}, tensors)
        header, got = await receive(pipe.reader())
        return sent, header, got

    sent, header, got = asyncio.run(roundtrip())
    assert header == {"op": "weights", "version": 3} and list(got) == list(tensors)
    assert sent == sum(t.numel() * t.element_size() for t in tensors.values())
    for name, tensor in tensors.items():
        assert got[name].dtype == tensor.dtype and got[name].shape == tensor.shape
        assert got[name].contiguous().view(-1).view(torch.uint8).tolist() == (
            tensor.contiguous().view(-1).view(torch.uint8).tolist()
        ), name


def test_a_tensor_that_cannot_travel_is_refused_by_name_before_anything_is_sent():
    with pytest.raises(ValueError, match="w.*complex64.*supported"):
        pack({"w": torch.zeros(2, dtype=torch.complex64)})


def test_a_stream_that_ends_inside_a_frame_says_which_frame():
    async def cut(limit):
        pipe = Pipe()
        await send(pipe, {"op": "weights", "version": 1}, {"w": torch.ones(64)})
        pipe.limit = limit if limit >= 0 else len(pipe.data) + limit
        return await receive(pipe.reader())

    with pytest.raises(asyncio.IncompleteReadError) as payload:
        asyncio.run(cut(-8))
    assert payload.value.frame == {"op": "weights", "version": 1}
    with pytest.raises(asyncio.IncompleteReadError) as header:
        asyncio.run(cut(10))
    assert header.value.frame == {}
    with pytest.raises(asyncio.IncompleteReadError) as between:
        asyncio.run(cut(0))
    assert not hasattr(between.value, "frame")  # the peer left between two frames


def test_staging_reuses_a_buffer_of_the_same_name_and_shape():
    staging = Staging()
    source = torch.ones(4, 2)
    first = staging.stage({"w": source})["w"]
    source.mul_(5)
    assert first.sum() == 8  # a snapshot, not a view
    assert staging.stage({"w": source})["w"] is first and first.sum() == 40
    assert staging.stage({"w": torch.ones(3)})["w"] is not first
    staging.release(["w"])
    assert not staging.buffers


# ------------------------------------------------------------- pool and worker


def test_versions_follow_the_weights_across_processes(workers):
    from palingenesis.opd.rollout import Sampling

    pool = RemotePool([workers(*SCRIPTED), workers(*SCRIPTED)])
    try:
        assert [generate(pool) for _ in range(4)] == [([0, 0, 0], 0)] * 4 and pool.version == -1
        weights = {"a.weight": torch.full((4, 8), 0.5, dtype=torch.bfloat16), "b.bias": torch.ones(8)}
        pool.update_weights(weights.items(), 1)
        assert pool.version == 1  # handed over; the engines confirm when it is theirs
        assert wait_for(lambda: pool.applied() == [1, 1])
        assert {tuple(generate(pool)[0]) for _ in range(6)} == {(1, 24, 0)}  # both engines, the new weights' sum
        weights["a.weight"] += 1
        pool.update_weights(weights.items(), 2)
        assert wait_for(lambda: pool.applied() == [2, 2]) and generate(pool) == ([2, 56, 0], 2)
        assert generate(pool, "verifier")[0][2] == ord("v")  # a request names its adapter
        assert generate(pool, temperature=Sampling(0.7, top_k=11))[0] == [2, 56, 0, 11]  # sampling settings travel
        threads = [in_thread(lambda: states.append(pool.state())) for states in [[]] for _ in range(8)]
        for thread in threads:
            thread.join(30)
        assert all(thread.error is None and not thread.is_alive() for thread in threads)  # (replies keyed by id)
        assert [s["pushes"] for s in pool.state()] == [2, 2] and all(s["dropped"] == 0 for s in pool.state())
        metrics = pool.metrics()
        assert metrics["pool/lag"] == 0 and metrics["pool/push_mb"] > 0 and metrics["pool/skipped"] == 0
        assert metrics["pool/engines"] == 2
    finally:
        pool.close(shutdown=True)


def test_a_push_travels_while_the_engine_generates_and_the_trainer_goes_on(workers, gate):
    """A push held on its way: the update has returned, requests are answered meanwhile by the weights the engine
    has, nothing of the push shows before all of it arrived, and newer snapshots do not pile up."""
    pool = RemotePool([workers(*SCRIPTED, "--gate", str(gate.path))], pin=False)
    try:
        weights = {"w": torch.ones(MB, dtype=torch.float32)}
        gate.close()
        pool.update_weights(weights.items(), 1)  # returns with the push staged, not delivered
        assert pool.version == 1 and wait_for(lambda: pool.state()[0]["receiving"])
        weights["w"].mul_(3)  # the trainer's next step changes its weights: the push carries the staged snapshot
        assert [generate(pool) for _ in range(5)] == [([0, 0, 0], 0)] * 5  # the old weights, whole
        assert pool.applied() == [-1] and pool.metrics()["pool/lag"] == 2
        pool.update_weights(weights.items(), 2)  # staged in the other area: it follows the push in flight
        weights["w"].mul_(2)
        pool.update_weights(weights.items(), 3)  # both areas busy: skipped, the next sync brings newer weights
        assert pool.version == 2 and pool.stats["skipped"] == 1 and pool.sending == [1, 1]
        assert generate(pool) == ([0, 0, 0], 0)
        gate.open()
        assert wait_for(lambda: pool.applied() == [2]) and pool.sending == [0, 0]
        assert generate(pool) == ([2, 3 * MB, 0], 2)  # the snapshot taken at version 2
        assert pool.state()[0]["pushes"] == 2  # version 1 (the snapshot of ones) was applied whole before it
        pool.update_weights(weights.items(), 4)  # both areas free again
        assert pool.version == 4 and wait_for(lambda: pool.applied() == [4])
        assert generate(pool) == ([4, 6 * MB, 0], 4)
    finally:
        pool.close(shutdown=True)


def test_a_push_cut_short_is_dropped_whole_and_a_new_trainer_takes_over(workers, gate):
    address = workers(*SCRIPTED, "--gate", str(gate.path))
    first = RemotePool([address], pin=False)
    gate.close()
    first.update_weights({"w": torch.ones(16 * MB)}.items(), 1)  # 64 MB: more than the sockets buffer
    assert wait_for(lambda: first.state()[0]["receiving"])
    first.loop.call_soon_threadsafe(first.targets[0].weights_writer.transport.abort)  # the trainer dies mid-push
    first.close()
    gate.open()
    second = RemotePool([address], pin=False)
    try:
        assert wait_for(lambda: second.state()[0]["dropped"] == 1)
        state = second.state()[0]
        assert (state["version"], state["pushes"], state["dropped"], state["receiving"]) == (0, 0, 1, False)
        assert generate(second) == ([0, 0, 0], 0)  # nothing of the cut push reached the engine's weights
        second.update_weights({"w": torch.full((64,), 2.0)}.items(), 7)  # the resumed trainer's first update
        assert wait_for(lambda: second.applied() == [7]) and generate(second) == ([7, 128, 0], 7)
    finally:
        second.close(shutdown=True)


def test_a_blocking_pool_waits_for_every_engine_and_says_which_one_did_not_confirm(workers, gate):
    address = workers(*SCRIPTED, "--gate", str(gate.path))
    pool = RemotePool([address], blocking=True, pin=False)
    try:
        gate.close()
        update = in_thread(pool.update_weights, {"w": torch.ones(64)}.items(), 1)
        assert wait_for(lambda: pool.state()[0]["receiving"])
        assert update.is_alive() and pool.version == -1  # held with the push
        gate.open()
        update.join(30)
        assert not update.is_alive() and update.error is None
        assert pool.version == 1 and pool.applied() == [1] and generate(pool)[1] == 1
    finally:
        pool.close()
    impatient = RemotePool([address], blocking=True, pin=False, push_timeout=0.3)
    try:
        gate.close()
        with pytest.raises(RuntimeError, match=f"did not confirm version 2 in 0.3 s.*{address}"):
            impatient.update_weights({"w": torch.ones(64)}.items(), 2)
    finally:
        gate.open()
        impatient.close(shutdown=True)


def test_every_rank_of_a_blocking_pool_waits_for_its_own_engines(workers, gate):
    """max_staleness 0 with two trainer ranks: rank 0 sends to both engines; rank 1 does not go on to collect
    rows before its own engine has the version (rows of the old one would be dropped as stale)."""
    near, far = workers(*SCRIPTED), workers(*SCRIPTED, "--gate", str(gate.path))
    rank0 = RemotePool([near], push_to=[near, far], blocking=True, pin=False)
    rank1 = RemotePool([far], push_to=[], blocking=True, pin=False)
    try:
        gate.close()
        pushing = in_thread(rank0.update_weights, {"w": torch.ones(8)}.items(), 1)
        waiting = in_thread(rank1.update_weights, (), 1)
        assert wait_for(lambda: rank1.state()[0]["receiving"]) and wait_for(lambda: rank0.applied()[0] == 1)
        assert pushing.is_alive() and waiting.is_alive() and rank1.version == -1
        assert generate(rank1) == ([0, 0, 0], 0)  # (what rank 1 would have collected)
        gate.open()
        pushing.join(30), waiting.join(30)
        assert pushing.error is None and waiting.error is None and not waiting.is_alive()
        assert rank0.version == rank1.version == 1 and generate(rank1) == ([1, 8, 0], 1)
    finally:
        rank1.close()
        rank0.close(shutdown=True)


def test_a_worker_that_dies_loses_no_request_and_ends_the_run_at_the_next_update(workers, tmp_path):
    hold = tmp_path / "hold"
    doomed, other = workers(*SCRIPTED, env={"PGS_TEST_HOLD": str(hold)}), workers(*SCRIPTED)
    pool = RemotePool([doomed, other], pin=False)
    try:
        pool.update_weights({"w": torch.ones(4)}.items(), 1)
        assert wait_for(lambda: pool.applied() == [1, 1])
        hold.write_text("")
        running = [asyncio.run_coroutine_threadsafe(pool._request([1], 4, 1.0, ""), pool.loop) for _ in range(4)]
        assert wait_for(lambda: pool.engines[0].load == 2)  # two of them are on the worker that holds its calls
        workers.processes[doomed].kill()
        assert [f.result(30).version for f in running] == [1] * 4  # sent again to the engine that is left
        assert pool.engines[0].dead and pool.metrics()["pool/engines"] == 1
        assert [generate(pool) for _ in range(3)] == [([1, 4, 0], 1)] * 3  # nothing more goes to the dead one
        with pytest.raises(RuntimeError, match=f"{doomed} went away.*resume from the last checkpoint"):
            pool.update_weights({"w": torch.ones(4)}.items(), 2)
        assert pool.sending == [0, 0]
        hold.unlink()
        workers("--port", doomed.rpartition(":")[2], *SCRIPTED)  # a restarted worker: empty, at version 0
        with pytest.raises(RuntimeError, match="went away"):  # this pool does not take it back silently
            pool.update_weights({"w": torch.ones(4)}.items(), 2)
        assert [generate(pool)[1] for _ in range(3)] == [1] * 3
        workers.processes[other].kill()
        assert wait_for(lambda: pool.engines[1].dead)
        with pytest.raises(ConnectionError, match="no engine worker is left"):
            generate(pool)
    finally:
        pool.close()
    again = RemotePool([doomed], pin=False)  # the resumed run: its first update brings the worker to its version
    try:
        again.update_weights({"w": torch.ones(4)}.items(), 5)
        assert wait_for(lambda: again.applied() == [5]) and generate(again) == ([5, 4, 0], 5)
    finally:
        again.close(shutdown=True)


def test_a_weights_connection_that_ends_fails_the_next_update(workers):
    pool = RemotePool([workers(*SCRIPTED)], pin=False)
    try:
        pool.loop.call_soon_threadsafe(pool.targets[0].weights_writer.transport.abort)
        pool.update_weights({"w": torch.ones(4)}.items(), 1)  # starts sending: the connection is found gone
        assert wait_for(lambda: bool(pool.failure)) and pool.sending == [0, 0]
        with pytest.raises(RuntimeError, match="went away"):
            pool.update_weights({"w": torch.ones(4)}.items(), 2)
    finally:
        pool.close(shutdown=True)


def test_an_update_that_fails_releases_its_staging_area_and_is_raised(workers):
    """What goes wrong while a push is staged or applied never leaves a staging area counted as busy."""
    pool = RemotePool([workers(*SCRIPTED)], pin=False)
    try:

        def broken(first):
            raise KeyError("no such weight")

        pool.payload = broken
        with pytest.raises(KeyError):
            pool.update_weights((), 1)
        assert pool.sending == [0, 0] and pool.version == -1
        pool.payload = lambda first: [({"mode": "full"}, {"w": torch.zeros(2, dtype=torch.complex64)})]
        pool.update_weights((), 1)  # refused where it is packed for the wire, on the pool's thread
        assert wait_for(lambda: bool(pool.failure)) and pool.sending == [0, 0]
        with pytest.raises(RuntimeError, match="complex64"):
            pool.check()
    finally:
        pool.close(shutdown=True)
    pool = RemotePool([workers(*SCRIPTED)], pin=False)
    try:
        # the worker cannot apply it (factors without the base they fold into): it says so, the pool raises
        pool.payload = lambda first: [({"mode": "fold", "scaling": {}}, {})]
        pool.update_weights((), 1)
        assert wait_for(lambda: bool(pool.failure)) and pool.sending == [0, 0]
        with pytest.raises(RuntimeError, match="fold push came before the base"):
            pool.update_weights((), 2)
        assert pool.state()[0]["version"] == 0 and pool.state()[0]["receiving"] is False
    finally:
        pool.close(shutdown=True)


def test_the_base_of_a_first_update_is_applied_with_what_follows_it_and_not_before():
    """Between the base and the factors of a first update the engine still has its own weights and version:
    a completion is never sampled from a base that is tagged as the trained policy."""
    engine = WireEngine()
    worker = EngineWorker(engine)
    base = {"layer.weight": torch.zeros(2, 3), "layer.bias": torch.ones(2)}
    assert worker._apply({"version": 4, "mode": "base"}, base) == 0.0
    assert (engine.version, engine.total, worker.pushes) == (0, 0.0, 0) and worker.held is not None
    factors = {"layer.weight|A": torch.ones(1, 3), "layer.weight|B": torch.ones(2, 1)}
    worker._apply({"version": 4, "mode": "fold", "scaling": {"layer.weight": 2.0}}, factors)
    assert (engine.version, engine.total, worker.pushes) == (4, 14.0, 1)  # 2 (bias) + 2 * six ones, together
    assert worker.held is None and worker.base["layer.weight"].sum() == 0  # the base itself is kept unfolded
    worker._apply({"version": 5, "mode": "fold", "scaling": {"layer.weight": 1.0}}, factors)
    assert (engine.version, engine.total) == (5, 8.0)
    with pytest.raises(RuntimeError, match="before the base"):
        EngineWorker(WireEngine())._apply({"version": 1, "mode": "fold", "scaling": {}}, {})
    worker.client.close()


class Slow(WireEngine):
    """Its generate call waits in the middle for the test (a batch engine: one call at a time)."""

    def __init__(self):
        super().__init__()
        self.inside, self.proceed = threading.Event(), threading.Event()

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter=""):
        from palingenesis.opd.rollout import Rollout

        before = self.total
        self.inside.set()
        assert self.proceed.wait(30)
        return [Rollout([int(before), int(self.total)], [-0.5] * 2, "stop", -1) for _ in prompts]


def test_weights_are_never_copied_into_a_batch_engine_during_a_generate_call():
    engine = Slow()
    worker = EngineWorker(engine)
    replies = []

    async def reply(message):
        replies.append(message)

    header = {"op": "generate", "id": 1, "prompt": [1], "max_new_tokens": 2, "temperature": 1.0}
    request = in_thread(asyncio.run, worker._generate(header, reply))
    assert engine.inside.wait(30)
    update = in_thread(worker._apply, {"version": 3, "mode": "full"}, {"w": torch.ones(5)})
    update.join(0.3)
    assert update.is_alive() and (engine.version, engine.total) == (0, 0.0)  # it waits for the call to end
    engine.proceed.set()
    request.join(30), update.join(30)
    assert request.error is None and update.error is None and engine.version == 3
    # the call saw one set of weights from start to end, and is tagged with that set's version
    assert replies[0]["ids"] == [0, 0] and replies[0]["version"] == 0
    worker.client.close()


class NativeEngine(WireEngine):
    """An engine that serves LoRA adapters from directories (vLLM's interface)."""

    def __init__(self):
        super().__init__()
        self.loaded = {}

    def load_adapter(self, name, path, version):
        self.loaded[name] = (Path(path), version)


def test_the_default_adapter_of_a_native_push_is_what_requests_without_an_adapter_sample(tmp_path):
    """With adapters served next to the base (vLLM), a request that names no adapter must sample the trained
    policy, not the bare base; "base" still names the base. Old adapter directories are deleted."""
    from safetensors.torch import load_file

    engine = NativeEngine()
    worker = EngineWorker(engine, adapter_dir=str(tmp_path))
    lora = {"rank": 2, "alpha": 4}
    replies = []

    async def reply(message):
        replies.append(message)

    async def ask(adapter):
        header = {"op": "generate", "id": 1, "prompt": [1], "max_new_tokens": 2, "temperature": 1.0, "adapter": adapter}
        await worker._generate(header, reply)
        return engine.calls[-1][0]

    async def run():
        assert await ask("") == ""  # nothing loaded yet: the engine's own weights
        for version in (1, 2, 3, 4):
            tensors = {
                "|model.layers.0.q_proj|lora_A": torch.full((2, 3), float(version)),
                "|model.layers.0.q_proj|lora_B": torch.ones(4, 2),
                "critic|model.layers.0.q_proj|lora_A": torch.zeros(2, 3),
                "critic|model.layers.0.q_proj|lora_B": torch.zeros(4, 2),
            }
            base = {"model.w": torch.ones(3)}
            if version == 1:  # the first update: the base, held, then the adapters
                worker._apply({"version": 1, "mode": "base"}, base)
                assert engine.version == 0 and not engine.loaded
            worker._apply({"version": version, "mode": "native", "lora": lora}, tensors)
            assert engine.version == version and engine.total == 3.0
        return [await ask(""), await ask("critic"), await ask("base")]

    assert asyncio.run(run()) == ["policy", "critic", "base"]
    path, version = engine.loaded["policy"]
    assert version == 4 and json.loads((path / "adapter_config.json").read_text())["r"] == 2
    state = load_file(str(path / "adapter_model.safetensors"))
    assert state["base_model.model.model.layers.0.q_proj.lora_A.weight"].unique().tolist() == [4.0]
    for name in ("policy", "critic"):  # this version and the one before stay (a request may still read it)
        assert sorted(p.name for p in path.parents[1].joinpath(name).iterdir()) == ["v3", "v4"]
    worker.client.close()


def test_requests_of_different_adapters_in_one_batch_are_grouped_and_answered_in_place():
    from palingenesis.rl.generation import GenerationClient

    engine = WireEngine()
    client = GenerationClient(engine, merge=False)
    order = ["x", "", "y", "x", "", "x"]

    async def run():
        for _ in order:
            client.enter()
        return await asyncio.gather(*(client.generate([i], 4, 1.0, adapter=a) for i, a in enumerate(order)))

    out = asyncio.run(run())
    assert [o.ids[2] for o in out] == [ord(a) if a else 0 for a in order]  # each caller, its own adapter's reply
    assert sorted(engine.calls) == [("", [[1], [4]]), ("x", [[0], [3], [5]]), ("y", [[2]])]  # one call per adapter
    client.close()


def test_a_worker_started_with_other_settings_is_refused(tmp_path, workers):
    from transformers import AutoConfig

    from palingenesis.rl.pool.protocol import model_signature

    policy = tiny_policy(tmp_path)
    address = workers("--model", policy, "--backend", "hf")
    mine = {"model": model_signature(AutoConfig.from_pretrained(policy)), "chat_template_kwargs": {}}
    RemotePool([address], identity=mine).close()
    with pytest.raises(ValueError, match="other settings.*chat_template_kwargs.*enable_thinking"):
        RemotePool([address], identity={**mine, "chat_template_kwargs": {"enable_thinking": False}})
    with pytest.raises(ValueError, match="other settings.*model.*hidden_size"):
        RemotePool([address], identity={**mine, "model": {**mine["model"], "hidden_size": 4096}})
    with pytest.raises(ValueError, match="stop_ids"):
        RemotePool([address], identity={**mine, "stop_ids": [0]})


# ----------------------------------------------------------- with the trainer


KWARGS = ("--chat-template-kwargs", '{"enable_thinking": false}')  # what the trainers below render with


def _config(tmp_path, engines, **overrides):
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in Chat.generate(24)))
    config = RLConfig()
    for key, value in {
        "model.policy": str(tmp_path / "policy"),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": str(tmp_path / "rows.jsonl"),
        "rollout.backend": "remote",
        "rollout.engines": engines,
        "rollout.batch_prompts": 2,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "rollout.max_staleness": 1,
        "scenario.type": CHAT,
        "loss.advantage_std": "role",
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 3,
        "train.learning_rate": 1e-2,
        "train.warmup_steps": 1,
        **overrides,
    }.items():
        config.set(key, value)
    return config


def _train(tmp_path, engines, each_step=None, watch=None, **overrides):
    from palingenesis.rl.trainer import RLTrainer

    trainer = RLTrainer(_config(tmp_path, engines, **overrides))
    logged = []
    if watch is not None:
        in_thread(watch, trainer)

    def log(kind, metrics, step):
        logged.append((kind, dict(metrics)))
        if kind == "step" and each_step is not None:
            each_step(dict(metrics))

    trainer._log = log
    trainer.train()
    return trainer, [m for kind, m in logged if kind == "step"]


def test_adapters_travel_as_factors_and_the_engines_sample_the_trained_policy(tmp_path, workers):
    """LoRA with remote hf engines ("fold"): megabytes of factors instead of the model, folded by each engine into
    the base it was sent once. The trainer's log-probs of the sampled tokens match the engines' (the fold on the
    other side is the trained policy), with max_staleness 0 every batch is on-policy, and a resumed run takes the
    engines over."""
    policy = tiny_policy(tmp_path)
    engines = [
        workers("--model", policy, "--backend", "hf", *KWARGS),
        workers("--model", policy, "--backend", "hf", *KWARGS),
    ]
    options = {
        "lora.rank": 4,
        "lora.alpha": 8,
        "rollout.max_staleness": 0,
        "train.save_steps": 3,
        "loss.kl_ref": "start",
    }
    trainer, steps = _train(tmp_path, engines, **options)
    assert trainer.push_mode == "fold" and len(steps) == 3
    assert all(m["staleness"] == 0 and m["pool/lag"] == 0 for m in steps), [
        (m["staleness"], m["pool/lag"]) for m in steps
    ]
    # the engines' folded weights are the trained policy up to float32 rounding in another process, while the
    # policy is far from the base it started from
    assert all(m["policy/mismatch_k3"] < 1e-3 for m in steps), [m["policy/mismatch_k3"] for m in steps]
    assert steps[-1]["policy/kl_ref"] > 20 * steps[-1]["policy/mismatch_k3"]
    model_mb = 4 * sum(p.numel() for p in trainer.model.parameters()) / MB
    assert steps[-1]["pool/push_mb"] < model_mb  # (the mean includes the one push of the base)
    assert all(m["role/a/policy/mismatch_k3"] < 1e-3 for m in steps)

    resumed, more = _train(tmp_path, engines, **options, **{"train.steps": 4, "train.resume_from": "auto"})
    assert resumed.start_step == 3 and len(more) == 1 and more[0]["policy/mismatch_k3"] < 1e-3


def test_two_roles_adapters_share_one_engine_as_named_adapters(tmp_path, workers):
    policy = tiny_policy(tmp_path)
    engines = [workers("--model", policy, "--backend", "hf", *KWARGS)]
    options = {
        "lora.rank": 4,
        "lora.alpha": 8,
        "roles.a.adapter": "alpha",
        "roles.b.adapter": "beta",
        "rollout.max_staleness": 0,
    }
    trainer, steps = _train(tmp_path, engines, **options)
    assert trainer.push_mode == "native" and set(trainer.pipeline.clients) == {"", "alpha", "beta"}
    for m in steps:  # each role's turns were sampled with its own adapter, in the one engine
        assert m["role/a/policy/mismatch_k3"] < 1e-3 and m["role/b/policy/mismatch_k3"] < 1e-3
    assert steps[-1]["grad_norm/alpha"] > 0 and steps[-1]["grad_norm/beta"] > 0
    model_mb = 4 * sum(p.numel() for p in trainer.model.parameters()) / MB
    assert steps[-1]["pool/push_mb"] < model_mb / 2  # the base once, then the adapters alone


def test_full_weights_overlap_training_and_rows_past_the_staleness_bound_are_dropped(tmp_path, workers, gate):
    """Full weights that do not arrive: the trainer never waits for a push, the engine samples with what it has,
    and rows whose turns are more than max_staleness versions old are left out. Once the pushes arrive, the rows
    are trained again."""
    policy = tiny_policy(tmp_path)
    held = [workers("--model", policy, "--backend", "hf", *KWARGS, "--gate", str(gate.path))]
    while_held = []

    def each_step(metrics):
        if gate.path.exists():
            while_held.append(metrics)

    def watch(trainer):  # the trainer has dropped rows as stale and waits for newer ones: let the pushes through
        if wait_for(lambda: trainer.orchestrator.replaced + trainer.stale_groups > 0, 120):
            gate.open()

    gate.close()
    trainer, steps = _train(tmp_path, held, each_step, watch, **{"train.steps": 5})
    assert trainer.push_mode == "full" and len(steps) == 5
    # steps were taken while every push was still on its way, on rows of the version the engine had
    assert len(while_held) >= 2 and while_held[0]["trained_sequences"] > 0
    assert while_held[-1]["pool/lag"] >= 2 and all(m["staleness"] <= 1 for m in steps)
    # the engine was past the bound until the pushes arrived: its batches were dropped, whole or by group
    assert trainer.orchestrator.replaced + steps[-1]["stale_groups_dropped"] > 0
    assert any(m.get("trained_sequences", 0) > 0 for m in steps[len(while_held) :])  # and trains again after

    fast = [workers("--model", policy, "--backend", "hf", *KWARGS)]
    trainer, steps = _train(tmp_path, fast, **{"train.steps": 4, "train.output_dir": str(tmp_path / "fast")})
    assert sum(m.get("trained_sequences", 0) > 0 for m in steps) >= 3 and all(m["staleness"] <= 1 for m in steps)
    current = [m["policy/mismatch_k3/current"] for m in steps if "policy/mismatch_k3/current" in m]
    # turns sampled at the trained version: the pushed weights are the trained ones
    assert current and max(current) < 1e-3


def test_a_role_on_the_frozen_base_makes_the_engines_serve_adapters(tmp_path, workers):
    """partner: base with remote engines: an engine that folded the adapters into its weights would have the
    partner sample the trained policy. The engines keep the base and serve the policy as an adapter next to it."""
    policy = tiny_policy(tmp_path)
    engines = [workers("--model", policy, "--backend", "hf", *KWARGS)]
    options = {
        "lora.rank": 4,
        "lora.alpha": 8,
        "roles.b.partner": "base",
        "rollout.max_staleness": 0,
        "loss.kl_ref": "start",
    }
    trainer, steps = _train(tmp_path, engines, **options)
    assert trainer.push_mode == "native" and "base" in trainer.pipeline.clients
    assert all(m["role/a/policy/mismatch_k3"] < 1e-3 for m in steps) and "role/b/policy/mismatch_k3" not in steps[-1]
    assert steps[-1]["policy/kl_ref"] > 0  # the policy moved; its turns still match the engine's adapter


def test_two_trainer_ranks_share_a_pool_and_only_rank_0_pushes(tmp_path, workers):
    policy = tiny_policy(tmp_path)
    engines = [
        workers("--model", policy, "--backend", "hf", *KWARGS),
        workers("--model", policy, "--backend", "hf", *KWARGS),
    ]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in Chat.generate(24)))
    (tmp_path / "rl.yaml").write_text(
        f"model: {{policy: {policy}, use_liger_kernel: false, chat_template_kwargs: {{enable_thinking: false}}}}\n"
        f"data: {{dataset: {tmp_path}/rows.jsonl}}\n"
        f"rollout: {{backend: remote, engines: {json.dumps(engines)}, batch_prompts: 4, group_size: 3, "
        "max_new_tokens: 8, max_staleness: 0}\n"
        f'scenario: {{type: "{CHAT}"}}\n'
        "lora: {rank: 4, alpha: 8}\n"
        "loss: {advantage_std: role}\n"
        f"train: {{output_dir: {tmp_path}/run, steps: 3, learning_rate: 1.0e-2, warmup_steps: 1, "
        "replicate_base: true}\n"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    environment = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
    }
    ranks = [
        subprocess.Popen(
            [sys.executable, "-m", "palingenesis.rl.trainer", "--config", str(tmp_path / "rl.yaml")],
            env={**environment, "RANK": str(rank), "LOCAL_RANK": str(rank)},
            stdout=open(tmp_path / f"rank{rank}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        for rank in range(2)
    ]
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and all(p.poll() in (None, 0) for p in ranks):
            if all(p.poll() == 0 for p in ranks):
                break
            time.sleep(0.2)
    finally:
        for process in ranks:
            process.kill()
            process.wait()
    log = (tmp_path / "rank0.log").read_text()
    assert [p.returncode for p in ranks] == [0, 0], log[-3000:] + (tmp_path / "rank1.log").read_text()[-3000:]
    last = [line for line in log.splitlines() if "step 3 |" in line][0]
    mismatch = float(re.search(r"policy/mismatch_k3=(\S+)", last).group(1))
    assert mismatch < 1e-3  # both ranks' engines sampled with the adapters rank 0 sent them
    assert "pool/push_mb" in last and (tmp_path / "run" / "final").exists()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rollout.engines": []}, "needs rollout.engines"),
        ({"rollout.backend": "hf"}, "only used with rollout.backend remote"),
        ({"rollout.push": "sometimes"}, "rollout.push must be"),
        ({"rollout.push": "fold"}, "it needs lora.rank"),
        ({"rollout.push": "native", "lora.rank": 4, "train.fsdp": True}, "not supported with FSDP"),
        ({"model.multimodal": True}, "does not support model.multimodal"),
        ({"rollout.push": "fold", "lora.rank": 4, "roles.a.adapter": "x"}, "rollout.push must be native"),
        ({"rollout.push": "fold", "lora.rank": 4, "roles.b.partner": "base"}, "rollout.push must be native"),
        ({"rollout.push": "native", "lora.rank": 4, "lora.dora": True}, "lora.dora magnitudes"),
        ({"lora.rank": 4, "lora.dora": True, "roles.b.partner": "base"}, "has no lora.dora"),
    ],
)
def test_configuration_errors(overrides, match):
    tiny = {
        "model.policy": "p",
        "data.dataset": "d",
        "rollout.backend": "remote",
        "rollout.engines": ["h:1"],
        "scenario.type": CHAT,
    }
    config = RLConfig()
    for key, value in {**tiny, **overrides}.items():
        config.set(key, value)
    with pytest.raises(RLConfigError, match=match):
        config.validate()
