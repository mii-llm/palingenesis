"""eval_tasks: evaluation tasks inside pgs rl / pgs distill / pgs train, on tiny random models (CPU, the hf engine)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.test_rl import tiny_policy  # noqa: E402


def _task(tmp_path):
    rows = [{"qid": i, "question": f"What is {i} plus 1?", "answer": str(i + 1)} for i in range(6)]
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "name": "plus",
        "dataset": str(tmp_path / "eval.jsonl"),
        "id_field": "qid",
        "prompt": "{{question}}",
        "scorer": "numeric",
        "sampling": {"max_tokens": 6, "temperature": 0.0},
    }


def _results(directory):
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*.jsonl"))


def test_rl_trainer_runs_eval_tasks(tmp_path):
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
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
        "train.save_final": False,
        "eval_tasks.tasks": [_task(tmp_path)],
        "eval_tasks.every": 1,
        "eval_tasks.limit": 4,
        "eval_tasks.modes": ["plain", "think"],
    }.items():
        config.set(key, value)
    assert config.validate(python_rewards=True) is not None
    trainer = RLTrainer(config, rewards=[lambda completion: len(completion) / 40])
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = {step: m for kind, step, m in logged if kind == "eval" and any(k.startswith("tasks/") for k in m)}
    assert sorted(evals) == [0, 1, 2]  # before training, every step, the last step once
    assert {"tasks/plus/plain/score", "tasks/plus/think/score", "tasks/plus/plain/tokens"} <= set(evals[0])
    assert all(0.0 <= m["tasks/plus/plain/score"] <= 1.0 for m in evals.values())
    files = _results(tmp_path / "run" / "eval_tasks")
    assert files == [f"step_{s}/plus.{mode}.jsonl" for s in (0, 1, 2) for mode in ("plain", "think")]
    rows = [json.loads(line) for line in (tmp_path / "run" / "eval_tasks" / "step_2" / "plus.plain.jsonl").open()]
    assert len(rows) == 4 and all(len(r["scores"]) == 1 for r in rows)
    assert not trainer.task_eval.disabled


def test_opd_trainer_runs_eval_tasks(tmp_path):
    from palingenesis.opd.trainer import OPDTrainer
    from tests.test_opd_trainer import make_config, tiny_model

    models = {
        "student": tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "student", 0),
        "same": tiny_model(tmp_path, "Qwen/Qwen3-0.6B", "teacher_same", 1),
        "other": tiny_model(tmp_path, "Qwen/Qwen3.5-0.8B", "teacher_other", 2),
    }
    config = make_config(tmp_path, models)
    config.set("eval_tasks.tasks", [_task(tmp_path)], "test")
    config.set("eval_tasks.limit", 3, "test")
    config.set("train.eval_every", 2, "test")
    config.validate()
    trainer = OPDTrainer(config)
    logged = []
    trainer._log = lambda kind, metrics, step: logged.append((kind, step, metrics))
    trainer.train()
    evals = {step: m for kind, step, m in logged if kind == "eval" and any(k.startswith("tasks/") for k in m)}
    assert sorted(evals) == [0, 2]  # the trainer's eval_every, and the last step once
    assert {"tasks/plus/default/score", "tasks/plus/default/truncated"} <= set(evals[2])
    assert _results(tmp_path / "run" / "eval_tasks") == ["step_0/plus.default.jsonl", "step_2/plus.default.jsonl"]
    assert not trainer.task_eval.disabled


def test_eval_on_the_model_in_training(tmp_path):
    """pgs train: the tasks generate with the model itself, which goes back to training mode."""
    import transformers

    from palingenesis.eval.config import EvalTasksConfig
    from palingenesis.eval.training import TrainingEval

    path = tiny_policy(tmp_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(path).train()
    tok = transformers.AutoTokenizer.from_pretrained(path)
    spec = EvalTasksConfig(tasks=[_task(tmp_path)], limit=3, modes=["plain"], every=5)
    hook = TrainingEval(spec, path, tok, tmp_path / "run", engine_options={"max_model_len": 512})
    metrics = hook.run_on_model(model, 5)
    assert model.training and not hook.disabled
    assert set(metrics) == {f"tasks/plus/plain/{name}" for name in ("score", "truncated", "loops", "tokens")}
    assert 0 < metrics["tasks/plus/plain/tokens"] <= 6
    summary = json.loads((tmp_path / "run" / "eval_tasks" / "step_5" / "plus.plain.summary.json").read_text())
    assert summary["items"] == 3 and summary["sampling"]["temperature"] == 0.0
    assert summary["chat_template_kwargs"] == {"enable_thinking": False}
