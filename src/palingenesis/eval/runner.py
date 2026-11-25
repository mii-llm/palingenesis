"""Run evaluation tasks through the RL pipeline and write `pgs compare` results.

    evaluator = Evaluator(tokenizer, engine, "runs/sft/final", engine_options={"max_model_len": 8192})
    summary = evaluator.run(load_task("gsm8k"), "think", k=4, out_dir="eval/sft")

One (task, mode) run builds an RLConfig from the task (its prompt column, system prompt, environment, sandbox and
budgets), the mode's chat template arguments and the sampling, then rolls every item out k times with
RLPipeline._eval: prompts are encoded, tool calls parsed, environments stepped and rewards computed exactly as in
`pgs rl`, `pgs screen` and rl.episodes.run_episodes.

Files in out_dir, per task and mode:
    <task>.<mode>.jsonl           one row per item, the `pgs compare` format: {"id", "scores": [k numbers],
                                  "cluster"?} plus per-sample "finish", "tokens" and "loop" lists
    <task>.<mode>.samples.jsonl   one row per sample: score, reward components, finish reason, tokens, the
                                  completion and its reasoning, the prompt (first sample) and the turns of an episode
    <task>.<mode>.summary.json    the summary below
Summary: items, mean score and unbiased pass@k with bootstrap intervals over items (clusters), the Wilson interval
of the first sample's pass rate, completion tokens, finish-reason shares (truncated_share = "length"), loop_share
(the rollout dumps' loop detector), unscored samples and items, seconds.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from palingenesis.eval.config import (
    DEFAULT_MAX_TOKENS,
    ENGINE_DEFAULTS,
    EvalConfigError,
    SamplingConfig,
    TaskConfig,
    _sampling,
    merge_sampling,
    resolve_modes,
)
from palingenesis.eval.tasks import EvalItem, build_items, prompt_field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- sampling


def model_sampling(model: str) -> SamplingConfig:
    """The sampling a server would use for `model` by default: its generation_config.json (do_sample false, or no
    file: nothing)."""
    try:
        from transformers import GenerationConfig

        g = GenerationConfig.from_pretrained(model)
    except Exception:  # noqa: BLE001 — no generation config (a bare checkpoint, offline)
        return SamplingConfig()
    if not getattr(g, "do_sample", False):
        return SamplingConfig()
    return SamplingConfig(
        temperature=float(g.temperature) if g.temperature is not None else None,
        top_p=float(g.top_p) if g.top_p is not None else None,
        top_k=int(g.top_k) if g.top_k is not None else None,
    )


def resolve_sampling(model: str, k: int, *layers: SamplingConfig) -> tuple[SamplingConfig, str]:
    """The sampling of one run and where its temperature came from: the layers merged (later over earlier), then,
    with no temperature set, the model's generation config, then greedy (k = 1) or temperature 1 (k > 1)."""
    s = merge_sampling(*layers)
    source = "config"
    if s.temperature is None:
        own = model_sampling(model) if model else SamplingConfig()
        if own.temperature is not None:
            source = "generation_config"
            s.temperature = own.temperature
            s.top_p = own.top_p if s.top_p is None else s.top_p
            s.top_k = own.top_k if s.top_k is None else s.top_k
        else:
            source = "default"
            s.temperature = 0.0 if k == 1 else 1.0
    s.top_p = 1.0 if s.top_p is None else s.top_p
    s.top_k = 0 if s.top_k is None else s.top_k
    s.presence_penalty = 0.0 if s.presence_penalty is None else s.presence_penalty
    s.max_tokens = DEFAULT_MAX_TOKENS if s.max_tokens is None else s.max_tokens
    return s, source


def engine_sampling(s: SamplingConfig):
    """What the rollout engine takes: a float for plain temperature sampling, else an opd.rollout.Sampling."""
    if s.top_p >= 1.0 and s.top_k == 0 and not s.presence_penalty:
        return float(s.temperature)
    from palingenesis.opd.rollout import Sampling

    return Sampling(float(s.temperature), float(s.top_p), int(s.top_k), float(s.presence_penalty))


# ------------------------------------------------------------------ RL config


def rl_config(
    model: str,
    task: TaskConfig,
    chat_template_kwargs: dict,
    sampling: SamplingConfig,
    engine_options: dict | None = None,
    stop_tokens: list | tuple = (),
    seed: int = 0,
):
    """The RLConfig a task's rollouts run under: the task's env: and sandbox: sections and the engine options are
    RL config options, checked by the RL config itself."""
    from palingenesis.config import ConfigError
    from palingenesis.rl.config import RLConfig

    config = RLConfig()
    options: dict[str, Any] = {
        "model.policy": model or "eval",
        "model.chat_template_kwargs": dict(chat_template_kwargs),
        "model.stop_tokens": list(stop_tokens),
        "data.prompt_field": prompt_field(task),
        "data.system_prompt": task.system_prompt,
        "data.success_threshold": task.threshold,
        "train.seed": seed,
    }
    options.update({f"rollout.{k}": v for k, v in {**ENGINE_DEFAULTS, **(engine_options or {})}.items()})
    options.update({f"env.{k}": v for k, v in task.env.items()})
    options.update({f"sandbox.{k}": v for k, v in task.sandbox.items()})
    try:
        for key, value in options.items():
            config.set(key, value, where=f"task {task.name}")
    except ConfigError as e:
        raise EvalConfigError(str(e).replace("rollout.", "engine.")) from None
    agentic = config.env.type != "single_turn"
    config.rollout.max_new_tokens = int(sampling.max_tokens)
    config.rollout.max_completion_tokens = int(
        sampling.max_completion_tokens or (sampling.max_tokens * config.env.max_turns if agentic else 0)
    )
    config.rollout.temperature = float(sampling.temperature)
    return config


def build_engine(config, model: str, tokenizer, stop_ids):
    """The rollout engine of an RL config (vllm and sglang in-process, or hf), as `pgs screen` builds it."""
    from palingenesis.rl.screen import _engine

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
    return _engine(config, model, stop_ids, pad_id)


# -------------------------------------------------------------------- records


def sample_record(trajectory) -> dict[str, Any]:
    """One rollout as a plain record: its score (None when it could not be scored), how it ended, its texts."""
    from palingenesis.rollouts import loop_flags

    t = trajectory
    prompt = t.info.get("prompt")
    if prompt is None:  # failed before the prompt was rendered
        first = next((i for i, m in enumerate(t.messages) if m.get("role") == "assistant"), len(t.messages))
        prompt = t.messages[:first]
    turns = t.messages[len(prompt) :]
    answers = [m for m in turns if m.get("role") == "assistant"]
    calls = [
        json.dumps(c.get("function"), sort_keys=True, default=str) for m in answers for c in m.get("tool_calls") or []
    ]
    last = answers[-1] if answers else {}
    record = {
        "score": float(t.reward) if t.scored else None,
        "rewards": dict(t.rewards),
        "finish": t.finish or "unknown",
        "tokens": t.sampled_tokens,
        "loop": bool(loop_flags([m["content"] for m in answers if isinstance(m.get("content"), str)], calls)["loop"]),
        "turns": t.turns,
        "tool_calls": t.tool_calls,
        "tool_errors": t.tool_errors,
        "completion": last.get("content") or "",
        "reasoning": last.get("reasoning_content") or "",
        "prompt": prompt,
    }
    if len(turns) > 1:
        record["messages"] = turns
    if t.info.get("error"):
        record["error"] = t.info["error"]
    return record


def item_row(item: EvalItem, samples: list[dict], task: str, mode: str) -> dict[str, Any]:
    """An item's `pgs compare` row: the scores of its scored samples, and per-sample outcome lists."""
    row: dict[str, Any] = {
        "id": item.id,
        "scores": [s["score"] for s in samples if s["score"] is not None],
        "task": task,
        "mode": mode,
        "finish": [s["finish"] for s in samples],
        "tokens": [s["tokens"] for s in samples],
        "loop": [s["loop"] for s in samples],
    }
    if item.cluster is not None:
        row["cluster"] = item.cluster
    return row


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    return float(sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))])


def summarize(
    rows: list[dict],
    threshold: float = 0.5,
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """The summary of one result file's rows (item_row): scores with intervals through `pgs compare`'s own
    statistics (mean per item first, unbiased pass@k, bootstrap over items or clusters), and the outcome shares
    over all samples."""
    from palingenesis.agent_tooling.compare import summarize as score_summary
    from palingenesis.evalstats import wilson_interval

    scored = {r["id"]: {"scores": r["scores"], "cluster": r.get("cluster")} for r in rows if r["scores"]}
    finishes = Counter(f for r in rows for f in r.get("finish", []))
    samples = sum(finishes.values()) or sum(len(r["scores"]) for r in rows)
    tokens = sorted(float(t) for r in rows for t in r.get("tokens", []))
    loops = sum(bool(x) for r in rows for x in r.get("loop", []))
    out: dict[str, Any] = {
        "items": len(scored),
        "unscored_items": len(rows) - len(scored),
        "samples": samples,
        "unscored_samples": samples - sum(len(r["scores"]) for r in rows),
    }
    if scored:
        k = min(len(v["scores"]) for v in scored.values())
        ks = sorted({1, k, *(2**i for i in range(1, 12) if 2**i <= k)})
        stats = score_summary(scored, ks, threshold, resamples, confidence, seed)
        out.update({key: value for key, value in stats.items() if key not in ("items",)})
        first = [v["scores"][0] >= threshold for v in scored.values()]
        lo, hi = wilson_interval(sum(first), len(first), confidence)
        out["wilson"] = {
            "successes": sum(first),
            "n": len(first),
            "estimate": sum(first) / len(first),
            "lo": lo,
            "hi": hi,
        }
    out["completion_tokens"] = {
        "mean": sum(tokens) / len(tokens) if tokens else 0.0,
        "p50": _percentile(tokens, 0.5),
        "p95": _percentile(tokens, 0.95),
        "max": tokens[-1] if tokens else 0.0,
    }
    out["finish"] = {name: count / samples for name, count in sorted(finishes.items())} if samples else {}
    out["truncated_share"] = finishes.get("length", 0) / samples if samples else 0.0
    out["turn_limit_share"] = finishes.get("turns", 0) / samples if samples else 0.0
    out["error_share"] = finishes.get("error", 0) / samples if samples else 0.0
    out["loop_share"] = loops / samples if samples else 0.0
    return out


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def result_paths(out_dir: str | Path, task: str, mode: str) -> dict[str, Path]:
    base = Path(out_dir)
    return {
        "results": base / f"{task}.{mode}.jsonl",
        "samples": base / f"{task}.{mode}.samples.jsonl",
        "summary": base / f"{task}.{mode}.summary.json",
    }


# ------------------------------------------------------------------ evaluator


class Evaluator:
    """Evaluation tasks on one model through one rollout engine.

    engine: a rollout engine (palingenesis.opd.rollout: VLLMColocateRollout, HFRollout; opd.sglang_rollout; or
    anything with generate(prompts, max_new_tokens, temperature)). engine_options: the RL rollout options the runs
    assume (max_model_len, ...). chat_template_kwargs: under every mode's own. modes: custom mode definitions.
    client: a GenerationClient to share (evaluation inside a trainer whose pipeline already drives the engine)."""

    def __init__(
        self,
        tokenizer,
        engine,
        model: str,
        *,
        engine_options: dict | None = None,
        stop_tokens: list | tuple = (),
        chat_template_kwargs: dict | None = None,
        sampling: SamplingConfig | None = None,
        modes: dict | None = None,
        client=None,
        chunk: int = 512,
        resamples: int = 10000,
        confidence: float = 0.95,
        seed: int = 0,
    ):
        self.tokenizer, self.engine, self.model = tokenizer, engine, model
        self.engine_options = dict(engine_options or {})
        self.stop_tokens = list(stop_tokens)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.sampling = sampling or SamplingConfig()
        self.modes = dict(modes or {})
        self.client = client
        self.chunk, self.resamples, self.confidence, self.seed = chunk, resamples, confidence, seed
        self._judges: dict[str, dict] = {}

    def mode(self, name: str) -> dict:
        return resolve_modes([name], self.modes, "mode")[name]

    def stop_ids(self, chat_template_kwargs: dict) -> tuple[int, ...]:
        from palingenesis.opd.trainer import student_stop_ids

        return student_stop_ids(self.tokenizer, self.model, self.stop_tokens, chat_template_kwargs)

    def _chat(self, kwargs: dict):
        from palingenesis.opd.teachers import end_of_turn_id
        from palingenesis.rl.chat import ChatFormat

        eot = end_of_turn_id(self.tokenizer, kwargs)
        try:
            from palingenesis.rl.trainer import _think_tags

            return ChatFormat(self.tokenizer, eot, kwargs, _think_tags(self.tokenizer, kwargs))
        except Exception:  # noqa: BLE001 — a template without think tags
            return ChatFormat(self.tokenizer, eot, kwargs)

    def _pipeline(self, config, items: list[EvalItem], rewards: list, kwargs: dict):
        import torch

        from palingenesis.opd.orchestrator import PublishedWeights
        from palingenesis.rl.data import PromptSampler, check_row_limits
        from palingenesis.rl.pipeline import RLPipeline
        from palingenesis.rl.sandbox import make_sandbox
        from palingenesis.rl.screen import _env_pool, _needs_sandbox

        rows = [item.row for item in items]
        check_row_limits(rows, f"task rows ({config.model.policy})", config.completion_budget)
        sandbox = make_sandbox(config.sandbox) if _needs_sandbox(config, rewards) else None
        env_pool = _env_pool(config, sandbox)
        if not rewards and not (env_pool is not None and env_pool.has_reward):
            raise EvalConfigError("no score: the task has no scorer and its environment has no get_reward")
        return RLPipeline(
            self.tokenizer,
            self._chat(kwargs),
            self.engine,
            PublishedWeights(torch.nn.Linear(1, 1)),
            config,
            PromptSampler(rows),
            rewards,
            env_pool,
            sandbox,
            self.stop_ids(kwargs),
            client=self.client,
        )

    def run(
        self,
        task: TaskConfig,
        mode: str = "default",
        *,
        k: int = 0,
        sampling: SamplingConfig | None = None,
        limit: int = 0,
        items: list[EvalItem] | None = None,
        out_dir: str | Path | None = None,
        save_samples: bool = True,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Evaluate one task in one mode and return its summary. k: samples per item (0: the task's, else 1);
        sampling: options over the task's and the mode's (a command line's); items: prebuilt items (else
        build_items(task, limit)); out_dir: where the files go (None: nothing is written)."""
        from palingenesis.eval.scorers import is_judge, resolve_scorers, validate_judge

        spec = self.mode(mode)
        kwargs = {**self.chat_template_kwargs, **spec["chat_template_kwargs"]}
        k = k or task.k or 1
        resolved, source = resolve_sampling(
            self.model,
            k,
            self.sampling,
            _sampling(task.sampling, f"task {task.name}: sampling"),
            spec["sampling"],
            sampling or SamplingConfig(),
        )
        rewards = resolve_scorers(task)
        judge = None
        if task.judge_validation or any(is_judge(r) for r in rewards):
            if task.name not in self._judges:  # once per task, whatever the mode
                self._judges[task.name] = validate_judge(rewards[0], task.judge_validation, task.name)
            judge = self._judges[task.name]
        config = rl_config(self.model, task, kwargs, resolved, self.engine_options, self.stop_tokens, self.seed)
        items = build_items(task, limit) if items is None else items
        paths = result_paths(out_dir, task.name, mode) if out_dir is not None else None
        rows: list[dict] = []
        if paths is not None:
            paths["results"].parent.mkdir(parents=True, exist_ok=True)
            if resume and paths["results"].exists():
                wanted = {item.id for item in items}
                rows = [r for r in read_rows(paths["results"]) if r["id"] in wanted]
            else:
                for name in ("results", "samples"):
                    paths[name].unlink(missing_ok=True)
        done = {r["id"] for r in rows}
        todo = [item for item in items if item.id not in done]
        logger.info(
            "eval %s [%s]: %d items x k=%d (%d already done), temperature %s (%s), max_tokens %d",
            task.name, mode, len(todo), k, len(done), resolved.temperature, source, resolved.max_tokens,
        )  # fmt: skip
        seconds, generated = 0.0, 0
        if todo:
            pipeline = self._pipeline(config, todo, rewards, kwargs)
            results = paths["results"].open("a") if paths else None
            samples_file = paths["samples"].open("a") if paths and save_samples else None
            try:
                for start in range(0, len(todo), self.chunk):
                    part = todo[start : start + self.chunk]
                    t0 = time.perf_counter()
                    groups = pipeline._await(pipeline._eval([item.row for item in part], engine_sampling(resolved), k))
                    seconds += time.perf_counter() - t0
                    for item, group in zip(part, groups):
                        samples = [sample_record(t) for t in group]
                        generated += sum(s["tokens"] for s in samples)
                        row = item_row(item, samples, task.name, mode)
                        rows.append(row)
                        if results is not None and row["scores"]:  # `pgs compare` needs a score in every row
                            results.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        if samples_file is not None:
                            for index, sample in enumerate(samples):
                                record = {"id": item.id, "sample": index, **sample}
                                if index:
                                    del record["prompt"]
                                samples_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    for handle in (results, samples_file):
                        if handle is not None:
                            handle.flush()
                    logger.info(
                        "eval %s [%s]: %d / %d items", task.name, mode, min(start + self.chunk, len(todo)), len(todo)
                    )
            finally:
                for handle in (results, samples_file):
                    if handle is not None:
                        handle.close()
                pipeline.close()
        order = {item.id: i for i, item in enumerate(items)}
        rows.sort(key=lambda r: order[r["id"]])
        summary: dict[str, Any] = {
            "task": task.name,
            "mode": mode,
            "model": self.model,
            "k": k,
            "threshold": task.threshold,
            "chat_template_kwargs": kwargs,
            "sampling": {
                "temperature": resolved.temperature,
                "top_p": resolved.top_p,
                "top_k": resolved.top_k,
                "presence_penalty": resolved.presence_penalty,
                "max_tokens": resolved.max_tokens,
                "max_completion_tokens": config.completion_budget,
                "temperature_source": source,
            },
            **summarize(rows, task.threshold, self.resamples, self.confidence, self.seed),
            "seconds": seconds,
            "tokens_per_second": generated / seconds if seconds > 0 else 0.0,
        }
        if judge is not None:
            summary["judge_validation"] = judge
        if paths is not None:
            summary["results"] = str(paths["results"])
            if save_samples:
                summary["samples_file"] = str(paths["samples"])
            paths["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
        if summary["unscored_samples"]:
            logger.warning(
                "eval %s [%s]: %d of %d samples could not be scored (%d items without any score): see the samples "
                "file's `error` and `rewards`",
                task.name, mode, summary["unscored_samples"], summary["samples"], summary["unscored_items"],
            )  # fmt: skip
        return summary


def format_summary(summary: dict) -> str:
    """One line of a results table."""
    mean = summary.get("mean_score")
    score = f"{mean['estimate']:.4f} [{mean['lo']:.4f}, {mean['hi']:.4f}]" if mean else "no scored items"
    k = summary["k"]
    passk = f"  pass@{k} {summary[f'pass@{k}']['estimate']:.4f}" if k > 1 and f"pass@{k}" in summary else ""
    return (
        f"{summary['task']:<16} {summary['mode']:<8} items {summary['items']:<6} score {score}{passk}  "
        f"tokens {summary['completion_tokens']['mean']:.0f}  truncated {summary['truncated_share']:.1%}  "
        f"loops {summary['loop_share']:.1%}  {summary['seconds']:.0f}s"
    )
