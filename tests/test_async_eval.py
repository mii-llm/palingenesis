"""eval_tasks placement async, transport checkpoint: the evaluator reads the run's checkpoints (CPU, tiny random
models, the hf engine)."""

import json
import threading
import time

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from palingenesis.eval import handoff  # noqa: E402
from palingenesis.eval.async_eval import AsyncEval, RunInfo  # noqa: E402
from palingenesis.eval.config import EvalTasksConfig  # noqa: E402
from palingenesis.eval.service import Watcher, checkpoint_format, checkpoints, consolidate_dcp  # noqa: E402
from tests.test_eval_training import _task  # noqa: E402
from tests.test_rl import tiny_policy  # noqa: E402


def _section(tmp_path, **options):
    base = {"placement": "async", "service": "auto", "transport": "checkpoint", "every": 2, "limit": 3}
    base.update(options)
    base.setdefault("modes", ["plain"])
    base.setdefault("engine", {"backend": "hf"})
    return EvalTasksConfig(tasks=[_task(tmp_path)], **base)


def _trainer_side(tmp_path, spec, model, prefix="step_", marker="trainer_state.pt", base=True, steps=6):
    info = RunInfo(
        trainer="rl", model=model, output_dir=str(tmp_path / "run"), every=spec.every, total_steps=steps,
        chat_template_kwargs={"enable_thinking": False}, base_step=base,
        checkpoints={"prefix": prefix, "complete": marker},
    )  # fmt: skip
    evaluator = AsyncEval(spec, info, training_gpus=[{"uuid": "", "label": "rank 0"}])
    evaluator.start()
    return evaluator


def _save(model_path, directory, marker="trainer_state.pt", scale=None):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    if scale is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
    model.save_pretrained(directory)
    (directory / marker).write_text("x")
    return directory


def _statuses(run):
    return {step: r["status"] for step, r in handoff.results(run).items()}


def test_spec_is_written_with_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _section(tmp_path)
    spec.tasks[0]["dataset"] = "eval.jsonl"  # relative to the trainer's directory
    _trainer_side(tmp_path, spec, tiny_policy(tmp_path))
    written = json.loads((tmp_path / "run" / "eval_tasks" / "spec.json").read_text())
    assert written["eval_tasks"]["tasks"][0]["dataset"] == str((tmp_path / "eval.jsonl").resolve())
    assert written["base_step"] == 0 and written["every"] == 2 and written["transport"] == "checkpoint"
    assert written["version"] == handoff.SPEC_VERSION


def test_dcp_checkpoint_is_consolidated(tmp_path):
    import torch.distributed.checkpoint as dcp

    path = tiny_policy(tmp_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(path)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    step = tmp_path / "run" / "step-4"
    optimizer = {"state": {"0": {"exp_avg": torch.ones(7)}}}
    dcp.save({"model": state, "optimizer": optimizer}, checkpoint_id=str(step / "dcp"), no_dist=True)
    assert checkpoint_format(step) == "dcp"
    weights = consolidate_dcp(step / "dcp", path)
    assert set(weights) == set(state)  # the model's tensors only, under their names
    assert all(torch.equal(weights[k], state[k]) for k in state)


def test_hf_layouts():
    from pathlib import Path

    assert checkpoint_format(Path("/nonexistent")) == "missing"


def test_purge_keeps_leased_and_best_checkpoints(tmp_path):
    from palingenesis.checkpoint import TRAINING_META_FILE, _purge_old_checkpoints
    from palingenesis.opd.trainer import prunable_checkpoints

    run = tmp_path / "run"
    for n in (1, 2, 3, 4):
        (run / f"step-{n}").mkdir(parents=True)
        (run / f"step-{n}" / TRAINING_META_FILE).write_text("{}")
    handoff.write_lease(run / "step-1", 1)
    handoff.write_json(run / "step-2" / handoff.LEASE, {"step": 2, "since": time.time() - 3 * 3600})  # expired
    _purge_old_checkpoints(str(run), 2)
    assert sorted(p.name for p in run.iterdir() if p.name.startswith("step-")) == ["step-1", "step-3", "step-4"]
    handoff.release(run / "step-1")
    handoff.write_json(handoff.root(run) / handoff.BEST, {"step": 1, "value": 1.0, "path": str(run / "step-1")})
    _purge_old_checkpoints(str(run), 1)
    assert sorted(p.name for p in run.iterdir() if p.name.startswith("step-")) == ["step-1", "step-4"]

    rl = tmp_path / "rl"
    for n in (2, 4, 6):
        (rl / f"step_{n}").mkdir(parents=True)
        (rl / f"step_{n}" / "trainer_state.pt").write_text("x")
    handoff.write_lease(rl / "step_2", 2)
    assert prunable_checkpoints(str(rl), 1) == [str(rl / "step_4")]


def test_watch_backlog_latest_and_all(tmp_path):
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    for backlog in ("latest", "all"):
        if run.exists():
            import shutil

            shutil.rmtree(run)
        spec = _section(tmp_path, backlog=backlog, select="plus/plain/score")
        trainer = _trainer_side(tmp_path, spec, model)
        _save(model, run / "step_2")
        _save(model, run / "step_4", scale=0.5)
        handoff.write_lease(run / "step_2", 2)
        handoff.write_lease(run / "step_4", 4)
        (run / "step_6").mkdir()  # an interrupted save (no trainer_state.pt): not evaluated
        assert sorted(checkpoints(run, json.loads((run / "eval_tasks" / "spec.json").read_text()))) == [0, 2, 4]
        assert Watcher(run, once=True, poll=0.01).run() == 0
        expected = {0: "skipped", 2: "skipped", 4: "done"} if backlog == "latest" else {0: "done", 2: "done", 4: "done"}
        assert _statuses(run) == expected
        assert not (run / "step_2" / handoff.LEASE).exists() and not (run / "step_4" / handoff.LEASE).exists()
        files = sorted(p.name for p in (run / "eval_tasks" / "step_4").iterdir())
        assert "plus.plain.jsonl" in files and "result.json" in files
        best = json.loads((run / "eval_tasks" / handoff.BEST).read_text())
        assert best["metric"] == "plus/plain/score" and best["step"] in (0, 2, 4)
        # the trainer's side: the metrics of the done steps, at their own steps, once
        logged = trainer.poll(10, force=True)
        assert [s for s, _ in logged] == sorted(s for s, v in expected.items() if v == "done")
        assert all("tasks/plus/plain/score" in m for _, m in logged)
        assert trainer.poll(11, force=True) == []
        counts = trainer.metrics()
        assert counts["eval/async/skipped"] == list(expected.values()).count("skipped")
        assert counts["eval/async/lag_steps"] == 6


def test_watch_follows_checkpoints_as_they_appear(tmp_path):
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    spec = _section(tmp_path, backlog="all")
    trainer = _trainer_side(tmp_path, spec, model, prefix="step-", marker="training_meta.json", base=False)
    watcher = Watcher(run, poll=0.05)
    code = []
    thread = threading.Thread(target=lambda: code.append(watcher.run()), daemon=True)
    thread.start()
    time.sleep(0.5)
    step_dir = run / "step-2"
    step_dir.mkdir(parents=True)
    _save(model, step_dir / "model", marker="unused")  # pgs train, one process: <step-N>/model
    (step_dir / "training_meta.json").write_text("{}")
    deadline = time.monotonic() + 120
    while 2 not in handoff.results(run) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _statuses(run) == {2: "done"}
    _save(model, run / "final", marker="unused")
    logged = trainer.finish(5, run / "final")  # the final step (not a multiple of every): the final model
    thread.join(timeout=120)
    assert code == [0]
    assert _statuses(run) == {2: "done", 5: "done"}
    assert [s for s, _ in logged] == [2, 5]
    assert json.loads((run / "eval_tasks" / "service.json").read_text())["state"] == "ended"


def test_watch_refuses_a_gpu_that_trains(tmp_path, monkeypatch):
    import palingenesis.eval.service as service

    model = tiny_policy(tmp_path)
    spec = _section(tmp_path)
    info = RunInfo(trainer="rl", model=model, output_dir=str(tmp_path / "run"), every=2, total_steps=4)
    AsyncEval(spec, info, training_gpus=[{"uuid": "GPU-aaaa", "label": "rank 0"}]).start()
    monkeypatch.setattr(service, "own_gpus", lambda: [{"uuid": "GPU-aaaa", "label": "GPU 0"}])
    monkeypatch.setattr(service, "EvalService", lambda *a, **k: pytest.fail("loaded a model after a refusal"))
    assert Watcher(tmp_path / "run", once=True).run() == 1
    state = json.loads((tmp_path / "run" / "eval_tasks" / "service.json").read_text())
    assert state["state"] == "refused" and "GPU-aaaa" in state["error"]


@pytest.mark.parametrize("mode", ["once", "always"])
def test_evaluator_out_of_memory(tmp_path, monkeypatch, mode):
    """An OOM is retried with a smaller engine; one that persists fails the step, and three in a row stop the
    evaluator: none of it reaches the trainer, which only reads the files."""
    model = tiny_policy(tmp_path)
    run = tmp_path / "run"
    spec = _section(tmp_path, backlog="all")
    _trainer_side(tmp_path, spec, model)
    for n in (2, 4, 6):
        _save(model, run / f"step_{n}")
    monkeypatch.setenv("PGS_TEST_EVAL_OOM", mode)
    code = Watcher(run, once=True, poll=0.01).run()
    statuses = _statuses(run)
    if mode == "once":
        assert code == 0 and set(statuses.values()) == {"done"}
    else:
        assert code == 1 and list(statuses.values()) == ["failed"] * 3
        assert "out-of-memory" in handoff.results(run)[0]["error"]
        assert json.loads((run / "eval_tasks" / "service.json").read_text())["state"] == "broken"


def test_rl_trainer_with_a_checkpoint_evaluator(tmp_path, monkeypatch):
    """pgs rl, eval_tasks async on its own process (started by the trainer): the trainer never runs the tasks
    itself, and the results come back at the steps they belong to."""
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
        "train.save_steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "eval_tasks.tasks": [_task(tmp_path)],
        "eval_tasks.every": 2,
        "eval_tasks.limit": 3,
        "eval_tasks.modes": ["plain"],
        "eval_tasks.placement": "async",
        "eval_tasks.devices": [0],
        "eval_tasks.transport": "checkpoint",
        "eval_tasks.backlog": "all",
        "eval_tasks.engine": {"backend": "hf"},
        "eval_tasks.final_timeout_s": 300,
    }.items():
        config.set(key, value)
    assert config.validate(python_rewards=True) is not None
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40])
    assert trainer.task_eval is None and trainer.async_eval is not None
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = {step: m for kind, step, m in logged if kind == "eval" and any(k.startswith("tasks/") for k in m)}
    run = tmp_path / "run"
    statuses = _statuses(run)
    assert statuses[4] == "done" and set(statuses) <= {0, 2, 4}
    assert sorted(evals) == sorted(s for s, v in statuses.items() if v == "done")
    assert "tasks/plus/plain/score" in evals[4]
    assert (run / "eval_tasks" / "step_4" / "plus.plain.jsonl").exists()
    log = (run / "eval_tasks" / "service.log").read_text()
    assert "Traceback" not in log, log


def test_opd_trainer_with_a_checkpoint_evaluator(tmp_path):
    from palingenesis.opd.trainer import OPDTrainer
    from tests.test_opd_trainer import make_config, tiny_model

    models = {
        "student": tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "student", 0),
        "same": tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "teacher_same", 1),
        "other": tiny_model(tmp_path, "Qwen/Qwen3.5-0.8B", "teacher_other", 2),
    }
    config = make_config(tmp_path, models)
    for key, value in {
        "eval_tasks.tasks": [_task(tmp_path)],
        "eval_tasks.limit": 3,
        "eval_tasks.placement": "async",
        "eval_tasks.devices": [0],
        "eval_tasks.transport": "checkpoint",
        "eval_tasks.backlog": "all",
        "eval_tasks.engine": {"backend": "hf"},
        "train.eval_every": 2,
        "train.save_steps": 2,
    }.items():
        config.set(key, value, "test")
    config.validate()
    trainer = OPDTrainer(config)
    assert trainer.task_eval is None
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = {step: m for kind, step, m in logged if kind == "eval" and any(k.startswith("tasks/") for k in m)}
    assert _statuses(tmp_path / "run") == {0: "done", 2: "done"}
    assert sorted(evals) == [0, 2] and "tasks/plus/default/score" in evals[2]
