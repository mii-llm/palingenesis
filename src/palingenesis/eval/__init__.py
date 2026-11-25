"""Evaluation through the training stack (`pgs eval`).

A task is a YAML file: a dataset, a field mapping, a prompt, a scorer (any RL reward, or a built-in of
palingenesis.eval.scorers), k samples, sampling, thinking modes and, for agentic tasks, an environment. Rollouts go
through the RL pipeline (palingenesis.rl.pipeline), so prompts are rendered, tool calls parsed, environments stepped
and rewards computed by the code a training run uses. Results are written in the `pgs compare` format.

    pgs eval --model runs/sft/final --tasks gsm8k,math500 --k 4 --modes plain,think --out eval/sft
    pgs compare eval/base/gsm8k.plain.jsonl eval/sft/gsm8k.plain.jsonl --require not_worse

    from palingenesis.eval import Evaluator, load_task
    summary = Evaluator(tokenizer, engine, "runs/sft/final").run(load_task("gsm8k"), "plain", out_dir="eval/sft")

Modules: config (the typed task / run configuration), tasks (rows, field mapping, prompts), scorers (built-in
scorers, the validated judge), ifeval (instruction checks), runner (the Evaluator, summaries, files), training
(the eval_tasks: hook of pgs train / rl / distill), cli.
"""

from __future__ import annotations

_EXPORTS = {
    "EvalConfig": "palingenesis.eval.config",
    "EvalConfigError": "palingenesis.eval.config",
    "EvalTasksConfig": "palingenesis.eval.config",
    "TaskConfig": "palingenesis.eval.config",
    "builtin_tasks": "palingenesis.eval.config",
    "load_task": "palingenesis.eval.config",
    "build_items": "palingenesis.eval.tasks",
    "Evaluator": "palingenesis.eval.runner",
    "summarize": "palingenesis.eval.runner",
    "TrainingEval": "palingenesis.eval.training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
