"""Evaluation tasks inside a training run: the `eval_tasks:` section of pgs train / rl / distill.

    eval_tasks:
      tasks: [gsm8k, my_task.yaml]     # built-in names, task files or inline tasks
      every: 200                       # steps (0: the trainer's eval_every)
      limit: 200                       # items per task: a fixed sample, the same at every evaluation
      modes: [plain, think]            # empty: the trainer's own chat template arguments
      k: 1

The trainer hands its rollout engine over at each evaluation (pgs rl and pgs distill: the engine that samples
training rollouts, holding the newest weights; pgs train: the model itself, single process only). Metrics are
logged as eval/tasks/<task>/<mode>/score, pass@k, truncated, loops and tokens; the result files of each step go
to <output_dir>/eval_tasks/step_<n>/, in the `pgs compare` format, so two steps compare directly:

    pgs compare out/eval_tasks/step_0/gsm8k.plain.jsonl out/eval_tasks/step_400/gsm8k.plain.jsonl

An evaluation that fails is logged and switched off; it never stops the training run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from palingenesis.eval.config import EvalConfigError, EvalTasksConfig, _sampling, load_task, resolve_modes

logger = logging.getLogger(__name__)


class TrainingEval:
    def __init__(
        self,
        spec: EvalTasksConfig,
        model: str,
        tokenizer,
        output_dir: str | Path,
        *,
        every: int = 0,
        engine_options: dict | None = None,
        chat_template_kwargs: dict | None = None,
        stop_tokens: list | tuple = (),
        seed: int = 0,
    ):
        """model: the checkpoint the run started from (its generation config and stop tokens); every: the
        trainer's eval_every, used when the section sets none. Raises EvalConfigError on a bad section; the
        datasets are loaded at the first evaluation."""
        problems = spec.errors()
        if problems:
            raise EvalConfigError("\n".join(problems))
        self.spec, self.model, self.tokenizer = spec, model, tokenizer
        self.output_dir = Path(output_dir) / "eval_tasks"
        self.every = spec.every or every
        self.engine_options = dict(engine_options or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.stop_tokens = list(stop_tokens)
        self.seed = seed
        self.tasks = [load_task(ref) for ref in spec.tasks]
        self.sampling = _sampling(spec.sampling, "eval_tasks.sampling")
        resolve_modes(spec.modes or ["default"], {}, "eval_tasks.modes")
        self.items: dict[str, list] | None = None
        self.disabled = False
        self.last_step = -1

    def due(self, step: int, total: int = 0) -> bool:
        """Whether the tasks run at `step` (every `every` steps, at step 0 when every is set, at the last step)."""
        if self.disabled or step == self.last_step:
            return False
        if total and step >= total:
            return self.spec.final
        return bool(self.every) and step % self.every == 0

    def run(self, engine, step: int, client=None) -> dict[str, float]:
        """Evaluate every task and mode with `engine` (holding the weights to evaluate) and return the metrics.
        client: the trainer pipeline's GenerationClient, when it already drives the engine."""
        if self.disabled:
            return {}
        self.last_step = step
        try:
            return self._run(engine, step, client)
        except Exception:  # noqa: BLE001 — an auxiliary evaluation must not stop a training run
            logger.exception("eval_tasks failed at step %d: switched off for the rest of the run", step)
            self.disabled = True
            return {}

    def run_on_model(self, model, step: int, micro_seqs: int = 8) -> dict[str, float]:
        """Evaluate with the model being trained itself (pgs train, one process): batched model.generate on it, in
        eval mode, through opd.rollout.HFRollout."""
        if self.disabled:
            return {}
        try:
            from palingenesis.opd.rollout import HFRollout
            from palingenesis.opd.trainer import student_stop_ids

            stop_ids = student_stop_ids(self.tokenizer, self.model, self.stop_tokens, self.chat_template_kwargs)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else stop_ids[0]
            engine = HFRollout(getattr(model, "_orig_mod", model), stop_ids, pad_id, micro_seqs)
        except Exception:  # noqa: BLE001 — as in run(): never stop the training run
            logger.exception("eval_tasks: no generation engine for this model: switched off")
            self.disabled = True
            return {}
        return self.run(engine, step)

    def _run(self, engine, step: int, client) -> dict[str, float]:
        from palingenesis.eval.runner import Evaluator, format_summary
        from palingenesis.eval.tasks import build_items

        if self.items is None:
            self.items = {task.name: build_items(task, self.spec.limit) for task in self.tasks}
        # a streaming client resolves each request on its caller's loop, so two pipelines can share it; a batching
        # client belongs to one event loop (the evaluation gets its own, the trainer's lock keeps them apart)
        client = client if getattr(client, "streaming", False) else None
        evaluator = Evaluator(
            self.tokenizer,
            engine,
            self.model,
            engine_options=self.engine_options,
            stop_tokens=self.stop_tokens,
            chat_template_kwargs=self.chat_template_kwargs,
            sampling=self.sampling,
            client=client,
            resamples=1000,
            seed=self.seed,
        )
        metrics: dict[str, float] = {}
        out_dir = self.output_dir / f"step_{step}"
        for task in self.tasks:
            for mode in task.modes or self.spec.modes or ["default"]:
                summary = evaluator.run(
                    task,
                    mode,
                    k=self.spec.k if not task.k else 0,
                    items=self.items[task.name],
                    out_dir=out_dir,
                    save_samples=self.spec.save_samples,
                )
                logger.info("eval_tasks step %d | %s", step, format_summary(summary))
                prefix = f"tasks/{task.name}/{mode}"
                if summary.get("mean_score"):
                    metrics[f"{prefix}/score"] = summary["mean_score"]["estimate"]
                    k = summary["k"]
                    if k > 1 and f"pass@{k}" in summary:
                        metrics[f"{prefix}/pass@{k}"] = summary[f"pass@{k}"]["estimate"]
                metrics[f"{prefix}/truncated"] = summary["truncated_share"]
                metrics[f"{prefix}/loops"] = summary["loop_share"]
                metrics[f"{prefix}/tokens"] = summary["completion_tokens"]["mean"]
        return metrics
