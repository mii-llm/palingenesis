"""eval_tasks placement async, transport push: the trainer's weights over TCP to `pgs eval serve` (CPU, tiny random
models, the hf engine, localhost)."""

import asyncio
import json
import threading
import time

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from palingenesis.eval import handoff, push  # noqa: E402
from palingenesis.eval.async_eval import AsyncEval, RunInfo  # noqa: E402
from palingenesis.eval.config import EvalTasksConfig  # noqa: E402
from tests.test_eval_training import _task  # noqa: E402
from tests.test_rl import tiny_policy  # noqa: E402


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(push, "RECONNECT_SECONDS", 0.05)


def _spec(tmp_path, **options):
    base = {"placement": "async", "service": "auto", "every": 2, "limit": 3, "modes": ["plain"]}
    base.update(options)
    base.setdefault("engine", {"backend": "hf"})
    return EvalTasksConfig(tasks=[_task(tmp_path)], **base)


def _trainer(tmp_path, spec, model, steps=6):
    info = RunInfo(trainer="rl", model=model, output_dir=str(tmp_path / "run"), every=spec.every, total_steps=steps,
                   chat_template_kwargs={"enable_thinking": False})  # fmt: skip
    evaluator = AsyncEval(spec, info, training_gpus=[{"uuid": "", "label": "rank 0"}])
    evaluator.start()
    return evaluator


def _weights(model_path, scale=1.0):
    from palingenesis.opd.rollout import checkpoint_named_parameters

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    return list(checkpoint_named_parameters(model))


def _serve(tmp_path):
    codes = []
    thread = threading.Thread(
        target=lambda: codes.append(push.serve(tmp_path / "run", "127.0.0.1", 0, idle_timeout=60)), daemon=True
    )
    thread.start()
    return thread, codes


def _connect(evaluator, timeout=30):
    deadline = time.monotonic() + timeout
    while not evaluator.client.ready():
        assert time.monotonic() < deadline, "the trainer never connected"
        time.sleep(0.05)


def _wait_result(run, step, timeout=120):
    deadline = time.monotonic() + timeout
    while step not in handoff.results(run):
        assert time.monotonic() < deadline, f"no result for step {step}"
        time.sleep(0.05)
    return handoff.results(run)[step]


class _Corrupt:
    pass


def test_receiver_keeps_the_newest_ready_snapshot():
    from palingenesis.rl.pool import buckets as bucketed

    receiver = push.Receiver()

    def frame(step, value):
        snapshot = bucketed.pack(
            [("w", torch.full((4,), float(value)))], 1 << 20, lambda i, n: torch.empty(n, dtype=torch.uint8)
        )
        header = {"op": "weights", "step": step, "buckets": snapshot.spec("tcp")}
        area = receiver.area(header)
        tensors = {}
        for name, buffer in snapshot.host().items():
            tensors[name] = area.buffer(name, torch.uint8, list(buffer.shape))
            tensors[name].copy_(buffer)
        return header, tensors

    assert receiver.arrived(*frame(2, 2.0)) == ""
    assert receiver.arrived(*frame(4, 4.0)) == ""  # replaces step 2, not started yet
    assert receiver.skipped == [2]
    step, weights = receiver.take(timeout=0.1)
    assert step == 4 and torch.equal(dict(weights())["w"], torch.full((4,), 4.0))
    # while step 4 loads, step 6 arrives in the other area, then step 8 replaces it
    assert receiver.arrived(*frame(6, 6.0)) == "" and receiver.arrived(*frame(8, 8.0)) == ""
    assert receiver.skipped == [2, 6]
    assert torch.equal(dict(weights())["w"], torch.full((4,), 4.0))  # the loading area was left alone
    receiver.loaded()
    assert receiver.take(timeout=0.1)[0] == 8
    header, tensors = frame(10, 10.0)
    tensors["bucket/0"][0] ^= 1
    assert "checksum" in receiver.arrived(header, tensors)
    assert receiver.take(timeout=0.05) is None


def test_push_end_to_end(tmp_path):
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    evaluator = _trainer(tmp_path, _spec(tmp_path), model)
    thread, codes = _serve(tmp_path)
    _connect(evaluator)
    assert evaluator.wants_push(2)
    seconds = evaluator.push(2, iter(_weights(model)))
    assert seconds >= 0 and evaluator.metrics()["eval/async/snapshot_seconds"] == seconds
    assert _wait_result(run, 2)["status"] == "done"
    logged = evaluator.poll(3, force=True)
    assert [s for s, _ in logged] == [2] and "tasks/plus/plain/score" in logged[0][1]
    assert (run / "eval_tasks" / "step_2" / "plus.plain.jsonl").exists()
    # the final step: sent once whatever is in flight, waited for, then the evaluator ends
    final = evaluator.finish(6, named=iter(_weights(model, 0.5)))
    thread.join(timeout=120)
    assert codes == [0] and [s for s, _ in final] == [6]
    assert json.loads((run / "eval_tasks" / "service.json").read_text())["state"] == "ended"
    assert evaluator.counts["done"] == 2


class _SilentEvaluator:
    """An evaluator that answers hello and then never acknowledges a push (stuck, or very slow)."""

    def __init__(self, run):
        self.loop = asyncio.new_event_loop()
        self.writers = []
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.port = asyncio.run_coroutine_threadsafe(self._start(), self.loop).result()
        handoff.write_json(handoff.root(run) / handoff.SERVICE, {"mode": "serve", "state": "listening",
                                                                 "host": "127.0.0.1", "port": self.port})  # fmt: skip

    async def _start(self):
        self.server = await asyncio.start_server(self._connection, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _connection(self, reader, writer):
        from palingenesis.rl.pool.protocol import receive, send

        self.writers.append(writer)
        await receive(reader)
        await send(writer, {"op": "hello", "ok": True})
        await asyncio.sleep(3600)

    def drop(self):
        async def close():
            self.server.close()
            for w in self.writers:
                w.close()

        asyncio.run_coroutine_threadsafe(close(), self.loop).result()


def test_a_stuck_or_dead_evaluator_never_blocks_training(tmp_path):
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    evaluator = _trainer(tmp_path, _spec(tmp_path), model, steps=100)
    silent = _SilentEvaluator(run)
    _connect(evaluator)
    weights = _weights(model)
    start = time.perf_counter()
    assert evaluator.wants_push(2)
    evaluator.push(2, iter(weights))
    staged = time.perf_counter() - start
    # the next due steps find the snapshot still unacknowledged: skipped before anything is copied
    times = []
    for step in (4, 6, 8, 10):
        start = time.perf_counter()
        assert not evaluator.wants_push(step)
        times.append(time.perf_counter() - start)
    assert max(times) < 0.05 and staged < 5.0
    assert evaluator.counts["push_skipped"] == 4
    silent.drop()  # the evaluator dies: the connection is lost, training goes on
    deadline = time.monotonic() + 10
    while evaluator.client.connected and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not evaluator.client.connected and not evaluator.wants_push(12)
    # a new evaluator comes up: the trainer connects again and the next due step is evaluated
    thread, codes = _serve(tmp_path)
    _connect(evaluator)
    assert evaluator.wants_push(14)
    evaluator.push(14, iter(weights))
    assert _wait_result(run, 14)["status"] == "done"
    evaluator.finish(100, named=None)
    thread.join(timeout=60)


def test_a_corrupted_push_is_dropped(tmp_path, monkeypatch):
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    flag = tmp_path / "corrupt"
    flag.write_text("1")
    monkeypatch.setenv("PGS_TEST_CORRUPT", str(flag))
    evaluator = _trainer(tmp_path, _spec(tmp_path), model)
    thread, codes = _serve(tmp_path)
    _connect(evaluator)
    evaluator.push(2, iter(_weights(model)))
    deadline = time.monotonic() + 30
    while evaluator.client.stats["push_failures"] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert evaluator.client.stats["push_failures"] == 1 and 2 not in handoff.results(run)
    flag.unlink()
    _connect(evaluator)
    evaluator.push(4, iter(_weights(model)))
    assert _wait_result(run, 4)["status"] == "done"
    evaluator.finish(6, named=None)
    thread.join(timeout=60)


def test_final_wait_is_bounded(tmp_path):
    model = tiny_policy(tmp_path)
    evaluator = _trainer(tmp_path, _spec(tmp_path, final_timeout_s=1), model)
    start = time.monotonic()
    assert evaluator.finish(6, named=iter(_weights(model))) == []  # no evaluator at all
    assert time.monotonic() - start < 10
    evaluator = _trainer(tmp_path, _spec(tmp_path, final_timeout_s=0), model)
    start = time.monotonic()
    evaluator.finish(6, named=iter(_weights(model)))
    assert time.monotonic() - start < 2


def test_the_evaluator_refuses_a_gpu_that_trains(tmp_path, monkeypatch):
    import palingenesis.eval.service as service

    model = tiny_policy(tmp_path)
    monkeypatch.setattr(service, "own_gpus", lambda: [{"uuid": "GPU-aaaa", "label": "GPU 0"}])
    spec = _spec(tmp_path)
    info = RunInfo(trainer="rl", model=model, output_dir=str(tmp_path / "run"), every=2, total_steps=6)
    evaluator = AsyncEval(spec, info, training_gpus=[{"uuid": "GPU-aaaa", "label": "rank 0"}])
    evaluator.start()
    thread, codes = _serve(tmp_path)
    deadline = time.monotonic() + 30
    while not evaluator.client.refused and time.monotonic() < deadline:
        evaluator.client.ready()
        time.sleep(0.05)
    thread.join(timeout=30)
    assert "GPU-aaaa" in evaluator.client.refused and codes == [1]
    assert not evaluator.wants_push(2)  # training goes on without it
    assert json.loads((tmp_path / "run" / "eval_tasks" / "service.json").read_text())["state"] == "refused"


def test_host_memory_check_at_start(tmp_path, monkeypatch):
    import palingenesis.eval.placement as placement
    from palingenesis.eval.config import EvalConfigError

    monkeypatch.setattr(placement, "available_host_memory", lambda: 100 * 2**30)
    info = RunInfo(trainer="train", model="m", output_dir=str(tmp_path / "run"), every=2, total_steps=6,
                   weights_bytes=140 * 2**30)  # fmt: skip
    with pytest.raises(EvalConfigError, match="transport checkpoint"):
        AsyncEval(_spec(tmp_path), info).start()
    AsyncEval(_spec(tmp_path, transport="checkpoint"), info).start()  # no staging: accepted


def test_rl_trainer_with_a_push_evaluator(tmp_path):
    """pgs rl, eval_tasks async with transport push on an evaluator the trainer starts itself."""
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.trainer import RLTrainer

    rows = [{"prompt": f"What is {i} plus {i + 1}?", "answer": str(2 * i + 1)} for i in range(20)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = RLConfig()
    for key, value in {
        "model.policy": tiny_policy(tmp_path),
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "model.gradient_checkpointing": "none",
        "data.dataset": str(tmp_path / "data.jsonl"),
        "rollout.backend": "hf",
        "rollout.batch_prompts": 3,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 4,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "eval_tasks.tasks": [_task(tmp_path)],
        "eval_tasks.every": 2,
        "eval_tasks.limit": 3,
        "eval_tasks.modes": ["plain"],
        "eval_tasks.placement": "async",
        "eval_tasks.devices": [0],
        "eval_tasks.engine": {"backend": "hf"},
        "eval_tasks.final_timeout_s": 300,
    }.items():
        config.set(key, value)
    config.validate(python_rewards=True)
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40])
    assert trainer.task_eval is None
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = {step: m for kind, step, m in logged if kind == "eval" and any(k.startswith("tasks/") for k in m)}
    statuses = {s: r["status"] for s, r in handoff.results(tmp_path / "run").items()}
    assert statuses.get(4) == "done", statuses  # the final step always
    assert sorted(evals) == sorted(s for s, v in statuses.items() if v == "done")
    steps = [m for kind, _, m in logged if kind == "step"]
    assert all("eval/async/alive" in m for m in steps)
    log = (tmp_path / "run" / "eval_tasks" / "service.log").read_text()
    assert "Traceback" not in log, log


def test_a_staging_failure_still_runs_every_gather(monkeypatch):
    """Under FSDP the weights' generator gathers: when staging fails on rank 0 the other ranks still go through
    every parameter's collective, so rank 0 must finish the generator too, or training hangs."""
    import threading
    import types

    from palingenesis.eval.push import PushClient

    visited = []

    def named():
        for i in range(5):
            visited.append(i)
            yield f"layer.{i}.weight", torch.ones(4, 4)

    def allocate(index, nbytes):
        raise RuntimeError("cannot pin host memory")

    def pack(named, bucket_bytes, allocate, dtype, checksums=False):  # fails after the first bucket, mid-way
        for i, _ in enumerate(named):
            if i == 1:
                allocate(0, 1)

    from palingenesis.rl.pool import buckets

    monkeypatch.setattr(buckets, "pack", pack)

    client = types.SimpleNamespace(
        lock=threading.Lock(), connected=True, sending=False, refused=False,
        spec=types.SimpleNamespace(final_timeout_s=0), staging=types.SimpleNamespace(pin=False),
        stats={"push_failures": 0}, _allocate=allocate,
    )
    assert PushClient.submit(client, 10, named()) is False
    assert visited == [0, 1, 2, 3, 4]
    assert client.sending is False and client.stats["push_failures"] == 1
