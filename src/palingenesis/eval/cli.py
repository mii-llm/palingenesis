"""pgs eval: evaluation tasks on a checkpoint, through the training stack.

    pgs eval --model runs/sft/final --suite core                       # the curated suite, one engine for all of it
    pgs eval --model runs/sft/final --tasks gsm8k,math500 --k 4 --modes plain,think --out eval/sft
    pgs eval --model M --tasks path/to/lm_eval/tasks/gsm8k/gsm8k.yaml  # a task of lm-evaluation-harness, as it is
    pgs eval --config eval.yaml                       # model, engine, run, sampling, modes, tasks
    pgs eval --list                                   # the built-in tasks and suites

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
    parser.add_argument(
        "--tasks", help="comma-separated built-in task names or task YAML files (ours or lm-eval's; globs expand)"
    )
    parser.add_argument("--suite", help="a built-in suite (core, quick, ...; see --list) or a suite YAML file")
    parser.add_argument("--template", choices=("chat", "completion"), help="every task through the chat template, or as raw text")
    parser.add_argument("--num_fewshot", type=int, help="lm-eval tasks: few-shot examples (default: the task's)")
    parser.add_argument("--replicas", type=int, help="engines per GPU (data parallel; small models on a large GPU)")
    parser.add_argument("--devices", help="comma-separated GPU indices, one engine each (default: all visible)")
    parser.add_argument("--score_workers", type=int, help="scorer processes (-1 automatic, 0: threads)")
    parser.add_argument("--cache_dir", help="the generation cache (default ~/.cache/palingenesis/eval)")
    parser.add_argument("--no_cache", action="store_true", help="generate everything again")
    parser.add_argument("--contamination", help="comma-separated training data files / directories: n-gram overlap")
    parser.add_argument("--categories", action="store_true", help="per-category lines in the table")
    parser.add_argument("--push_to_hub", help="a dataset repository for the results (created private)")
    parser.add_argument("--k", type=int, help="samples per item (default 1)")
    parser.add_argument("--modes", help="comma-separated thinking modes: default, plain, think or the config's own")
    parser.add_argument("--out", help="the output directory (default: run.output_dir, eval_results)")
    parser.add_argument("--limit", type=int, help="items per task: a fixed-seed sample (0 = all)")
    parser.add_argument("--head", action="store_true", help="--limit keeps the first items (as lm-eval's --limit)")
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
    from palingenesis.eval.config import builtin_suites, builtin_tasks, load_suite, load_task

    tasks = builtin_tasks()
    if not tasks:
        print("no built-in tasks found")
        return 1
    print("Built-in tasks (pgs eval --tasks <name>,...):")
    for name in tasks:
        task = load_task(name)
        kind = "loglikelihood" if task.output == "choices" else f"scorer {task.scorer or ', '.join(task.rewards)}"
        print(f"  {name:<16} {task.dataset} [{task.split}]  {kind}  {task.description}")
    print("Suites (pgs eval --suite <name>):")
    for name in builtin_suites():
        print(f"  {name:<16} {', '.join(t.name for t in load_suite(name))}")
    return 0


def _task_refs(spec: str) -> list[str]:
    """--tasks entries, with glob patterns expanded (a directory of lm-eval task files)."""
    import glob

    refs: list[str] = []
    for ref in (r.strip() for r in spec.split(",")):
        if any(c in ref for c in "*?["):
            found = sorted(glob.glob(ref))
            if not found:
                from palingenesis.eval.config import EvalConfigError

                raise EvalConfigError(f"--tasks: no file matches {ref!r}")
            refs += found
        elif ref:
            refs.append(ref)
    return refs


def build_config(args, overrides: list[str]):
    """The EvalConfig of a command line: --config, then the flags, then --section.option overrides."""
    from palingenesis.eval.config import EvalConfig, EvalConfigError, SamplingConfig, load_suite, load_task

    config = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    cli_sampling = SamplingConfig()
    for name in ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens"):
        if getattr(args, name) is not None:
            setattr(cli_sampling, name, getattr(args, name))
    if args.model:
        config.model.path = args.model
    if args.suite or args.tasks:
        config.tasks = (load_suite(args.suite) if args.suite else []) + [
            load_task(ref) for ref in _task_refs(args.tasks or "")
        ]
    if args.template:
        config.run.template = args.template
        for task in config.tasks:
            task.template = args.template
    for name in ("num_fewshot", "score_workers"):
        if getattr(args, name) is not None:
            setattr(config.run, name, getattr(args, name))
    if args.replicas is not None:
        config.engine["replicas"] = args.replicas
    if args.devices:
        config.engine["devices"] = [int(d) for d in args.devices.split(",") if d.strip()]
    if args.no_cache:
        config.run.cache_dir = ""
    elif args.cache_dir:
        config.run.cache_dir = args.cache_dir
    elif not config.run.cache_dir and not args.config:
        import os

        config.run.cache_dir = os.environ.get("PGS_EVAL_CACHE") or str(Path.home() / ".cache/palingenesis/eval")
    if args.contamination:
        config.run.contamination = [p.strip() for p in args.contamination.split(",") if p.strip()]
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
    config.run.head = config.run.head or args.head
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
    """Run a validated EvalConfig and return the run report ({"model", "results": [...], "overall": {...}}).
    engine / tokenizer: use these instead of loading the model (tests, notebooks)."""
    import time

    from palingenesis.eval import report as reporting
    from palingenesis.eval.config import SamplingConfig, TaskConfig, sampling_errors
    from palingenesis.eval.engines import engine_layout, start_engines
    from palingenesis.eval.runner import Evaluator, format_summary, read_rows, result_paths, rl_config
    from palingenesis.eval.tasks import build_items, prompt_text

    started = time.perf_counter()
    r = config.run
    cli_sampling = cli_sampling or SamplingConfig()
    problems = sampling_errors(cli_sampling, "command line")
    if problems:
        from palingenesis.eval.config import EvalConfigError

        raise EvalConfigError("\n".join(problems))
    # the data first: a missing column or an offline dataset fails before the engine takes the GPU
    items = {task.name: build_items(task, r.limit, num_fewshot=r.num_fewshot, head=r.head) for task in config.tasks}
    prepare_seconds = time.perf_counter() - started
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    options, layout = engine_layout(config.engine)
    evaluator = Evaluator(
        tokenizer,
        engine,
        config.model.path,
        engine_options=options,
        stop_tokens=config.model.stop_tokens,
        chat_template_kwargs=config.model.chat_template_kwargs,
        sampling=config.sampling,
        modes=config.modes,
        chunk=r.chunk,
        resamples=r.resamples,
        confidence=r.confidence,
        seed=r.seed,
        template=r.template,
        cache_dir=r.cache_dir,
        score_workers=r.score_workers,
        logprob_top=layout.max_logprobs if engine is None else 20,
    )
    owned = engine is None
    # scorer processes start while the engine loads
    evaluator.warm(sum(len(items[t.name]) * max(1, t.k or r.k) * len(t.modes or r.modes) for t in config.tasks))
    summaries = []
    engine_seconds = 0.0
    try:
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
                options,
                config.model.stop_tokens,
                r.seed,
            )
            t0 = time.perf_counter()
            evaluator.engine = engine = start_engines(
                config.model.path,
                engine_config,
                tokenizer,
                evaluator.stop_ids(kwargs),
                layout,
                kwargs,
                config.model.stop_tokens,
            )
            engine_seconds = time.perf_counter() - t0
            logger.info("engine ready in %.1fs (%d engine%s)", engine_seconds, layout.engines, "s"[: layout.engines > 1])
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
                if r.contamination:
                    from palingenesis.eval.contamination import task_report

                    summary["contamination"] = task_report(
                        items[task.name],
                        read_rows(Path(summary["results"])),
                        r.contamination,
                        r.contamination_ngram,
                        lambda item, task=task: prompt_text(item.row, task),
                    )
                    result_paths(r.output_dir, task.name, mode)["summary"].write_text(
                        json.dumps(summary, indent=2, default=str) + "\n"
                    )
                logger.info(format_summary(summary))
                summaries.append(summary)
    finally:
        evaluator.close()
        if owned and engine is not None and hasattr(engine, "close"):
            try:
                engine.close()
            except Exception:  # noqa: BLE001 — best effort at shutdown
                logger.warning("engine close failed", exc_info=True)
    report = {
        "model": config.model.path,
        "output_dir": r.output_dir,
        "engine": {"backend": options.get("backend"), "engines": layout.engines, "start_seconds": engine_seconds},
        "prepare_seconds": prepare_seconds,  # datasets read, prompts built
        "wall_seconds": time.perf_counter() - started,
        "overall": reporting.overall(summaries),
        "results": summaries,
    }
    Path(r.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(r.output_dir) / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (Path(r.output_dir) / "report.md").write_text(reporting.markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    from palingenesis.eval.config import EvalConfigError
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
    if args.push_to_hub:
        from palingenesis.eval.report import push_to_hub

        try:
            report["hub"] = push_to_hub(report["output_dir"], args.push_to_hub)
        except Exception as e:  # noqa: BLE001 — the results are on disk either way
            print(f"pgs eval: --push_to_hub {args.push_to_hub}: {type(e).__name__}: {e}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        from palingenesis.eval.report import table

        print(table(report["results"], categories=args.categories))
        print(
            f"results in {report['output_dir']} (summary.json, report.md; compare runs with "
            "`pgs compare a.jsonl b.jsonl`)" + (f"; pushed to {report['hub']}" if report.get("hub") else "")
        )
    return 0 if any(s["items"] for s in report["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
