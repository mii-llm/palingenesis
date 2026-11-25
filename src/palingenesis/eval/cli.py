"""pgs eval: evaluation tasks on a checkpoint, through the training stack.

    pgs eval --model runs/sft/final --tasks gsm8k,math500 --k 4 --modes plain,think --out eval/sft
    pgs eval --config eval.yaml                       # model, engine, run, sampling, modes, tasks
    pgs eval --list                                   # the built-in tasks

Every (task, mode) pair is one result: <out>/<task>.<mode>.jsonl in the `pgs compare` format (plus
.samples.jsonl and .summary.json), and <out>/summary.json for the run. Compare two checkpoints, or two modes:

    pgs compare eval/base/gsm8k.plain.jsonl eval/sft/gsm8k.plain.jsonl --require not_worse --margin 0.02

Tasks are built-in names (configs/eval/), task YAML files, or entries of the config's `tasks:`. Sampling left unset
is the model's generation_config.json (greedy at k = 1 without one). Other --section.option value arguments set
config options (model.*, run.*, sampling.*, engine.*). Exit code 0, or 1 on a bad config or when nothing was scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgs eval",
        description="Evaluate a checkpoint on tasks (YAML) through the RL pipeline; results in the pgs compare "
        "format. Other --section.option value arguments set config options (model, run, sampling, engine).",
    )
    parser.add_argument("--config", help="an evaluation config (model, engine, run, sampling, modes, tasks)")
    parser.add_argument("--model", help="the checkpoint to evaluate (a Hugging Face id or a path)")
    parser.add_argument("--tasks", help="comma-separated built-in task names or task YAML files")
    parser.add_argument("--k", type=int, help="samples per item (default 1)")
    parser.add_argument("--modes", help="comma-separated thinking modes: default, plain, think or the config's own")
    parser.add_argument("--out", help="the output directory (default: run.output_dir, eval_results)")
    parser.add_argument("--limit", type=int, help="items per task: a fixed-seed sample (0 = all)")
    parser.add_argument("--temperature", type=float, help="over every task's and mode's (0 = greedy)")
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--max_tokens", type=int, help="generated tokens per assistant turn, thinking included")
    parser.add_argument("--backend", choices=("vllm", "sglang", "hf"), help="the rollout engine (default vllm)")
    parser.add_argument("--max_model_len", type=int, help="the engine's context (prompt + completion tokens)")
    parser.add_argument("--resume", action="store_true", help="keep the items already in the result files")
    parser.add_argument("--no_samples", action="store_true", help="do not write the .samples.jsonl files")
    parser.add_argument("--list", action="store_true", help="list the built-in tasks and exit")
    parser.add_argument("--json", action="store_true", help="print the run summary as JSON")
    return parser


def _list_tasks() -> int:
    from palingenesis.eval.config import builtin_tasks, load_task

    tasks = builtin_tasks()
    if not tasks:
        print("no built-in tasks found")
        return 1
    print("Built-in tasks (pgs eval --tasks <name>,...):")
    for name in tasks:
        task = load_task(name)
        print(f"  {name:<14} {task.dataset} [{task.split}]  scorer {task.scorer or 'rewards'}  {task.description}")
    return 0


def build_config(args, overrides: list[str]):
    """The EvalConfig of a command line: --config, then the flags, then --section.option overrides."""
    from palingenesis.eval.config import EvalConfig, EvalConfigError, SamplingConfig, load_task

    config = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    cli_sampling = SamplingConfig()
    for name in ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens"):
        if getattr(args, name) is not None:
            setattr(cli_sampling, name, getattr(args, name))
    if args.model:
        config.model.path = args.model
    if args.tasks:
        config.tasks = [load_task(ref.strip()) for ref in args.tasks.split(",") if ref.strip()]
    if args.k is not None:
        config.run.k = args.k
    if args.modes:
        config.run.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.out:
        config.run.output_dir = args.out
    if args.limit is not None:
        config.run.limit = args.limit
    if args.backend:
        config.engine["backend"] = args.backend
    if args.max_model_len is not None:
        config.engine["max_model_len"] = args.max_model_len
    config.run.resume = config.run.resume or args.resume
    config.run.save_samples = config.run.save_samples and not args.no_samples
    i = 0
    while i < len(overrides):
        if not overrides[i].startswith("--") or i + 1 >= len(overrides):
            raise EvalConfigError(
                f"command line: unexpected argument {overrides[i]!r} (overrides are --section.option value)."
            )
        config.set(overrides[i][2:], overrides[i + 1])
        i += 2
    return config, cli_sampling


def run(config, cli_sampling=None, engine=None, tokenizer=None) -> dict:
    """Run a validated EvalConfig and return the run summary ({"model", "results": [...]}). engine / tokenizer:
    use these instead of loading the model (tests, notebooks)."""
    from palingenesis.eval.config import SamplingConfig, TaskConfig, sampling_errors
    from palingenesis.eval.runner import Evaluator, build_engine, format_summary, rl_config
    from palingenesis.eval.tasks import build_items

    r = config.run
    cli_sampling = cli_sampling or SamplingConfig()
    problems = sampling_errors(cli_sampling, "command line")
    if problems:
        from palingenesis.eval.config import EvalConfigError

        raise EvalConfigError("\n".join(problems))
    # the data first: a missing column or an offline dataset fails before the engine takes the GPU
    items = {task.name: build_items(task, r.limit) for task in config.tasks}
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    evaluator = Evaluator(
        tokenizer,
        engine,
        config.model.path,
        engine_options=config.engine,
        stop_tokens=config.model.stop_tokens,
        chat_template_kwargs=config.model.chat_template_kwargs,
        sampling=config.sampling,
        modes=config.modes,
        chunk=r.chunk,
        resamples=r.resamples,
        confidence=r.confidence,
        seed=r.seed,
    )
    owned = engine is None
    if owned:
        first = config.tasks[0]
        kwargs = {
            **config.model.chat_template_kwargs,
            **evaluator.mode((first.modes or r.modes)[0])["chat_template_kwargs"],
        }
        engine_config = rl_config(
            config.model.path,
            TaskConfig(name="engine"),
            kwargs,
            SamplingConfig(temperature=0.0, max_tokens=1),
            config.engine,
            config.model.stop_tokens,
            r.seed,
        )
        evaluator.engine = engine = build_engine(
            engine_config, config.model.path, tokenizer, evaluator.stop_ids(kwargs)
        )
    summaries = []
    try:
        for task in config.tasks:
            for mode in task.modes or r.modes:
                summary = evaluator.run(
                    task,
                    mode,
                    k=task.k or r.k,
                    sampling=cli_sampling,
                    items=items[task.name],
                    out_dir=r.output_dir,
                    save_samples=r.save_samples,
                    resume=r.resume,
                )
                logger.info(format_summary(summary))
                summaries.append(summary)
    finally:
        if owned and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception:  # noqa: BLE001 — best effort at shutdown
                logger.warning("engine close failed", exc_info=True)
    report = {"model": config.model.path, "output_dir": r.output_dir, "results": summaries}
    Path(r.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(r.output_dir) / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    from palingenesis.eval.config import EvalConfigError
    from palingenesis.eval.runner import format_summary
    from palingenesis.logging import setup_logging

    args, overrides = _parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        return _list_tasks()
    setup_logging(rank=0)
    try:
        config, cli_sampling = build_config(args, overrides)
        for warning in config.validate():
            logger.warning(warning)
        report = run(config, cli_sampling)
    except EvalConfigError as e:
        print(f"pgs eval: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for summary in report["results"]:
            print(format_summary(summary))
        print(f"results in {report['output_dir']} (summary.json; compare runs with `pgs compare a.jsonl b.jsonl`)")
    return 0 if any(s["items"] for s in report["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
