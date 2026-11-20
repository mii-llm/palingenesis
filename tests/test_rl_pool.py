"""Rollout engines in other processes (palingenesis.rl.pool, rollout.backend remote): versions across processes,
weight pushes that overlap generation and training, pushes dropped whole when their sender dies, the staleness
bound on what is trained, LoRA adapters sent as factors (folded by the engine, or served as named adapters of one
engine), resume, and trainer ranks with a replicated base sharing an engine pool."""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy  # noqa: E402
from test_rl_scenarios import CHAT, Chat  # noqa: E402

from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402
from palingenesis.rl.pool.client import RemotePool  # noqa: E402

ROOT = Path(__file__).parent.parent


class WireEngine:
    """A scripted engine for the protocol tests: a completion is [its version, the sum of its weights, the
    adapter's first character], so a reply says exactly which weights produced it."""

    def __init__(self):
        self.version, self.total = 0, 0.0

    def generate(self, prompts, max_new_tokens, temperature, images=None, adapter=""):
        from palingenesis.opd.rollout import Rollout

        ids = [self.version, int(round(self.total)), ord(adapter[0]) if adapter else 0]
        return [Rollout(list(ids), [-0.5] * 3, "stop", self.version) for _ in prompts]

    def update_weights(self, named, version):
        self.total = float(sum(tensor.double().sum() for _, tensor in named))
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def scripted_engine():
    return WireEngine()


def start_worker(*args):
    """An engine worker in its own process: (process, "127.0.0.1:port")."""
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "palingenesis.rl.pool.worker", "--port", "0", *args],
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        found = re.search(r"listening on [\w.]+:(\d+)", line)
        if found:
            return process, f"127.0.0.1:{found.group(1)}"
        if not line and process.poll() is not None:
            break
    process.kill()
    raise RuntimeError("the engine worker did not start")


@pytest.fixture
def workers():
    started = []

    def start(*args):
        process, address = start_worker(*args)
        started.append(process)
        return address

    yield start
    for process in started:
        process.kill()
        process.wait()


def generate(pool, adapter=""):
    out = asyncio.run_coroutine_threadsafe(pool._request([1, 2, 3], 4, 1.0, adapter), pool.loop).result(30)
    return out.ids, out.version


def wait_for(condition, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


SCRIPTED = ("--engine", f"{__file__}:scripted_engine")
MB = 2**20


def test_versions_follow_the_weights_across_processes(workers):
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
        assert [s["pushes"] for s in pool.state()] == [2, 2] and all(s["dropped"] == 0 for s in pool.state())
        metrics = pool.metrics()
        assert metrics["pool/lag"] == 0 and metrics["pool/push_mb"] > 0 and metrics["pool/skipped"] == 0
    finally:
        pool.close(shutdown=True)


def test_a_push_travels_while_the_engine_generates_and_the_trainer_goes_on(workers):
    """A slow link (4 MB at 2 MB/s): the update returns at once, requests are answered meanwhile by the weights
    the engine has, nothing of the push shows before all of it arrived, and newer snapshots do not pile up."""
    pool = RemotePool([workers(*SCRIPTED, "--throttle", str(2 * MB))], pin=False)
    try:
        big = {"w": torch.ones(MB, dtype=torch.float32)}  # 4 MB: about 2 s
        start = time.monotonic()
        pool.update_weights(big.items(), 1)
        assert time.monotonic() - start < 0.5 and pool.version == 1  # staged, not delivered
        big["w"].mul_(3)  # the trainer's next step changes its weights: the push carries the staged snapshot
        during = [generate(pool) for _ in range(5)]
        assert all(ids == [0, 0, 0] and version == 0 for ids, version in during)  # the old weights, whole
        assert time.monotonic() - start < 1.5 and pool.applied() == [-1] and pool.metrics()["pool/lag"] == 2
        pool.update_weights(big.items(), 2)  # staged in the other area, follows the push in flight
        pool.update_weights(big.items(), 3)  # both areas busy: skipped, the next sync brings newer weights
        assert pool.version == 2 and pool.stats["skipped"] == 1
        assert wait_for(lambda: pool.applied() == [1])
        assert generate(pool) == ([1, MB, 0], 1)  # version 1 is the snapshot taken at version 1
        assert wait_for(lambda: pool.applied() == [2]) and generate(pool) == ([2, 3 * MB, 0], 2)
        pool.update_weights(big.items(), 4)  # both areas free again
        assert pool.version == 4 and wait_for(lambda: pool.applied() == [4])
    finally:
        pool.close(shutdown=True)


def test_a_push_cut_short_is_dropped_whole_and_a_new_trainer_takes_over(workers):
    address = workers(*SCRIPTED, "--throttle", str(MB))
    first = RemotePool([address], pin=False)
    first.update_weights({"w": torch.ones(MB)}.items(), 1)  # 4 s on this link
    time.sleep(0.5)
    first.loop.call_soon_threadsafe(first.targets[0].weights_writer.transport.abort)  # the trainer dies mid-push
    first.close()
    second = RemotePool([address], pin=False)
    try:
        assert wait_for(lambda: second.state()[0]["dropped"] == 1)
        assert second.state()[0] == {"op": "state", "version": 0, "pushes": 0, "dropped": 1}
        assert generate(second) == ([0, 0, 0], 0)  # nothing of the cut push reached the engine's weights
        second.update_weights({"w": torch.full((64,), 2.0)}.items(), 7)  # the resumed trainer's first update
        assert wait_for(lambda: second.applied() == [7]) and generate(second) == ([7, 128, 0], 7)
    finally:
        second.close(shutdown=True)


def test_a_blocking_pool_waits_for_every_engine(workers):
    pool = RemotePool([workers(*SCRIPTED, "--throttle", str(MB))], blocking=True, pin=False)
    try:
        start = time.monotonic()
        pool.update_weights({"w": torch.ones(MB // 2)}.items(), 1)  # 2 MB at 1 MB/s
        assert time.monotonic() - start > 1.0 and pool.applied() == [1] and generate(pool)[1] == 1
    finally:
        pool.close(shutdown=True)


# ----------------------------------------------------------- with the trainer


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


def _train(tmp_path, engines, **overrides):
    from palingenesis.rl.trainer import RLTrainer

    trainer = RLTrainer(_config(tmp_path, engines, **overrides))
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, dict(metrics)))
    trainer.train()
    return trainer, [m for kind, m in logged if kind == "step"]


def test_adapters_travel_as_factors_and_the_engines_sample_the_trained_policy(tmp_path, workers):
    """LoRA with remote hf engines ("fold"): megabytes of factors instead of the model, folded by each engine into
    the base it was sent once. The trainer's log-probs of the sampled tokens match the engines' (the fold on the
    other side is the trained policy), with max_staleness 0 every batch is on-policy, and a resumed run takes the
    engines over."""
    policy = tiny_policy(tmp_path)
    engines = [workers("--model", policy, "--backend", "hf"), workers("--model", policy, "--backend", "hf")]
    options = {
        "lora.rank": 4,
        "lora.alpha": 8,
        "rollout.max_staleness": 0,
        "train.save_steps": 3,
        "loss.kl_ref": "start",
    }
    trainer, steps = _train(tmp_path, engines, **options)
    assert trainer.push_mode == "fold" and len(steps) == 3
    assert all(m["staleness"] == 0 and m["pool/lag"] == 0 for m in steps), [(m["staleness"], m["pool/lag"]) for m in steps]
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
    engines = [workers("--model", policy, "--backend", "hf")]
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
    assert steps[-1]["grad_norm/alpha"] > 0 and steps[-1]["grad_norm/beta"] > 0 and steps[-1]["pool/push_mb"] < 1


def test_full_weights_overlap_training_and_rows_past_the_staleness_bound_are_dropped(tmp_path, workers):
    """Full weights over a slow link: the trainer never waits for a push, the engines sample with what they have,
    and rows whose turns are more than max_staleness versions old are left out."""
    policy = tiny_policy(tmp_path)
    slow = [workers("--model", policy, "--backend", "hf", "--throttle", str(2 * MB))]  # ~19 MB: ~10 s a push
    trainer, steps = _train(tmp_path, slow, **{"train.steps": 4})
    assert trainer.push_mode == "full" and len(steps) == 4
    assert steps[-1]["stale_groups_dropped"] > 0  # the engine was still at an old version
    assert max(m["pool/lag"] for m in steps) >= 2 and all(m["time/sync"] < 2.0 for m in steps)
    assert steps[0]["trained_sequences"] > 0  # (the first batch was sampled at the version it trained)

    fast = [workers("--model", policy, "--backend", "hf")]
    trainer, steps = _train(tmp_path, fast, **{"train.steps": 4, "train.output_dir": str(tmp_path / "fast")})
    assert sum(m.get("trained_sequences", 0) > 0 for m in steps) >= 3 and all(m["staleness"] <= 1 for m in steps)
    current = [m["policy/mismatch_k3/current"] for m in steps if "policy/mismatch_k3/current" in m]
    assert (
        current and max(current) < 1e-3
    )  # turns sampled at the trained version: the pushed weights are the trained ones


def test_two_trainer_ranks_share_a_pool_and_only_rank_0_pushes(tmp_path, workers):
    policy = tiny_policy(tmp_path)
    engines = [workers("--model", policy, "--backend", "hf"), workers("--model", policy, "--backend", "hf")]
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
