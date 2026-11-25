"""pgs env: the environment catalogue from the command line.

    pgs env list [--family F] [--trainable | --eval-only] [--permissive] [--json]
    pgs env info <name> [--json]
    pgs env test <name>... | --all [--n 32] [--seed 0] [--dataset D] [--json]
    pgs env run  <name> --model M [--k 4] [--n 64] [--dataset D] [--output results.jsonl] [--json]
    pgs env rows <name> --output rows.jsonl [--n 1000] [--seed 0] [--dataset D] [--option key=value]
    pgs env new  <name> [--dir .] [--family other]

<name> is a registered environment, or "module:SPEC" / "path/to/file.py:SPEC" for one under development.

test    the audit (palingenesis.rl.catalog.audit): the oracle must be paid on every row, the degenerate policies
        on none (beyond the entry's chance level), and the oracle again on reused instances. Exit 1 on a failure;
        reward-hacking holes are listed with the policy and the first row it passes.
run     k episodes per row with a model, through the RL rollout loop (palingenesis.rl.episodes.run_episodes: the
        same prompt rendering, tool-call parsing, limits and environment rewards as training). Reports pass@1 (mean
        over rows of the share of passing samples), pass@k (rows with at least one passing sample), the mean
        reward, how episodes ended, tool errors and episodes/s; --output writes one line per row
        ({"id", "scores": [0 / 1 per sample], "rewards", "finishes"}), which `pgs compare` reads.
        Environments that execute model-written code need --sandbox docker (the default for them) or an explicit
        --sandbox process (resource limits, no isolation).
rows    rows of the environment for `pgs rl`: data.dataset: <output>, env.type: the entry's class (`pgs env info`
        prints the config lines). Evaluation-only entries are refused unless --allow-eval-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from typing import Any


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _options(pairs: list[str]) -> dict[str, Any]:
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--option {pair!r}: expected key=value")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _spec(name: str):
    from palingenesis.rl.catalog import registry
    from palingenesis.rl.catalog.spec import SpecError

    try:
        return registry.get(name)
    except (registry.UnknownEnvironment, SpecError, ImportError, AttributeError, ValueError) as error:
        raise SystemExit(f"pgs env: {error}") from None


# ----------------------------------------------------------------------- list


def cmd_list(args) -> int:
    from palingenesis.rl.catalog import registry

    trainable = True if args.trainable else False if args.eval_only else None
    found = registry.specs(args.family, trainable, True if args.permissive else None)
    errors = registry.load_errors()
    if args.json:
        _print_json({"environments": [spec.describe() for spec in found], "errors": errors})
        return 0
    if not found:
        print("no environments match")
    width = max((len(spec.name) for spec in found), default=4)
    family = ""
    for spec in found:
        if spec.family != family:
            family = spec.family
            print(f"\n{family}")
        flags = [spec.sandbox if spec.sandbox != "none" else "", "eval-only" if spec.eval_only else "",
                 "" if spec.permissive else f"licence {spec.license}", spec.source if spec.source != "builtin" else ""]  # fmt: skip
        tail = f"  [{', '.join(f for f in flags if f)}]" if any(flags) else ""
        print(f"  {spec.name:<{width}}  {spec.version:<7} {spec.summary}{tail}")
    for error in errors:
        print(f"\nWARN  not loaded: {error}")
    print(f"\n{len(found)} environments. pgs env info <name> for the entry, pgs env test <name> for its audit.")
    return 0


# ----------------------------------------------------------------------- info


def rl_config(spec, dataset: str = "rows.jsonl") -> str:
    """The lines of an RL config that train on the environment."""
    lines = ["env:", f"  type: {spec.env_ref()}", f"  max_turns: {spec.max_turns}"]
    plain = {k: v for k, v in spec.env_args.items() if isinstance(v, (str, int, float, bool, list, dict))}
    if plain:
        lines.append(f"  args: {json.dumps(plain)}")
    lines += ["data:", f"  dataset: {dataset}", f"  success_threshold: {spec.success_threshold:g}"]
    lines += ["rollout:", f"  max_new_tokens: {spec.max_new_tokens}"]
    if spec.max_turns > 1:
        lines.append(f"  max_completion_tokens: {spec.max_new_tokens * min(spec.max_turns, 8)}")
    if spec.sandbox == "process":
        lines += ["sandbox:", "  backend: docker        # model-written code runs here"]
    return "\n".join(lines)


def cmd_info(args) -> int:
    spec = _spec(args.name)
    entry = spec.describe()
    row, tools, problem = None, [], ""
    try:
        from palingenesis.rl.catalog.runner import load_rows

        if spec.datasets[spec.default_dataset].kind == "generator" and not spec.missing_requirements():
            row = load_rows(spec, n=1)[0]
            tools = spec.tool_schemas(row)
    except Exception as error:  # noqa: BLE001 — the entry is still worth showing
        problem = f"{type(error).__name__}: {error}"
    if args.json:
        _print_json({**entry, "tools": tools, "example_row": row, "rl_config": rl_config(spec), "problem": problem})
        return 0
    print(f"{spec.name} {spec.version}  ({spec.family})")
    print(f"  {spec.summary}")
    print(f"\n  environment   {entry['env']}")
    print(f"  verifier      {spec.verifier}")
    print(f"  sandbox       {spec.sandbox}")
    print(f"  use           {'evaluation only' if spec.eval_only else 'training and evaluation'}")
    print(f"  licence       {spec.license}{'' if spec.permissive else '  (not permissive, or a dataset is not)'}")
    print(f"  limits        max_turns {spec.max_turns}, max_new_tokens {spec.max_new_tokens}, pass at reward >= {spec.success_threshold:g}")
    if spec.difficulty:
        print(f"  difficulty    {json.dumps(spec.difficulty)}")
    if spec.citation:
        print(f"  citation      {spec.citation}")
    if spec.requires:
        missing = spec.missing_requirements()
        print(f"  requires      {', '.join(spec.requires)}" + (f"  (missing: {', '.join(missing)})" if missing else ""))
    print("\n  datasets")
    for key, dataset in spec.datasets.items():
        where = dataset.hub_id and f"hub {dataset.hub_id}" + (f" [{dataset.config}]" if dataset.config else "") + f" ({dataset.split})"
        where = where or ("generated" if dataset.kind == "generator" else dataset.path or "your rows (--dataset <path>)")
        print(f"    {key:<12} {where}; licence {dataset.license or '-'}")
        if dataset.description:
            print(f"                 {dataset.description}")
        if dataset.citation:
            print(f"                 {dataset.citation}")
    print("\n  tools         " + (", ".join(t["function"]["name"] for t in tools) or "none (the reply is the answer)"))
    for tool in tools:
        function = tool["function"]
        parameters = ", ".join((function.get("parameters") or {}).get("properties") or {})
        print(f"    {function['name']}({parameters})  {(function.get('description') or '').splitlines()[0] if function.get('description') else ''}")
    policies = entry["policies"]
    print(f"\n  audit         oracle: {'yes' if policies['oracle'] else 'none'}; degenerate: "
          f"{', '.join(policies['degenerate']) or '-'} (+ empty, refusal, echo_prompt, tool_spam); chance {spec.chance:g}")  # fmt: skip
    if spec.reference:
        print("\n  reference results")
        for model, metrics in spec.reference.items():
            print(f"    {model}: " + ", ".join(f"{k} {v:g}" for k, v in metrics.items()))
    else:
        print("\n  reference results: none recorded (scripts/gpu_checks/envs.sh measures them)")
    if spec.notes:
        print(f"\n  notes         {spec.notes}")
    if row is not None:
        from palingenesis.rl.data import prompt_messages

        print("\n  example prompt")
        for message in prompt_messages(row):
            text = str(message.get("content"))
            print("    " + (text if len(text) <= 900 else text[:900] + " ...").replace("\n", "\n    "))
    if problem:
        print(f"\n  WARN  could not build an example row: {problem}")
    if spec.trainable:
        print(f"\n  train on it (pgs env rows {spec.name} --n 2000 --output rows.jsonl):\n")
        print("    " + rl_config(spec).replace("\n", "\n    "))
    return 0


# ----------------------------------------------------------------------- test


def cmd_test(args) -> int:
    from palingenesis.rl.catalog import registry
    from palingenesis.rl.catalog.audit import audit, render

    names = registry.names() if args.all else args.names
    if not names:
        raise SystemExit("pgs env test: name an environment or pass --all")
    reports, skipped = [], []
    for name in names:
        spec = _spec(name)
        if args.all and spec.datasets[spec.default_dataset].kind == "file" and not args.dataset:
            skipped.append(name)  # no built-in rows (harbor): audited with --dataset
            continue
        report = audit(spec, n=args.n, seed=args.seed, dataset=args.dataset, concurrency=args.concurrency)
        reports.append(report)
        if not args.json:
            print(render(report))
    if args.json:
        _print_json(reports if len(reports) != 1 else reports[0])
    else:
        for name in skipped:
            print(f"{name}: skipped (no built-in rows; pgs env test {name} --dataset <path>)")
        if len(reports) > 1:
            failed = [r["env"] for r in reports if not r["ok"]]
            print(f"\n{len(reports) - len(failed)} of {len(reports)} audits passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 0 if all(r["ok"] for r in reports) else 1


# ----------------------------------------------------------------------- rows


def cmd_rows(args) -> int:
    from palingenesis.rl.catalog.runner import load_rows

    spec = _spec(args.name)
    if spec.eval_only and not args.allow_eval_only:
        raise SystemExit(
            f"pgs env rows: {spec.name} is evaluation-only ({spec.notes or 'its verifier is not safe to optimize'}); "
            "--allow-eval-only writes the rows anyway"
        )
    rows = load_rows(spec, args.dataset, args.n, args.seed, **_options(args.option))
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"{len(rows)} rows -> {args.output}\n")
    print(rl_config(spec, args.output))
    return 0


# ------------------------------------------------------------------------ run


def summarize(spec, rows: list[dict], groups: list[list], seconds: float, k: int) -> tuple[dict, list[dict]]:
    """(summary, per-row records) of k trajectories per row (t.reward, t.scored, t.finish, t.tool_calls, ...)."""
    threshold = spec.success_threshold
    records, rates, any_pass, rewards = [], [], [], []
    finishes: Counter = Counter()
    tool_calls = tool_errors = tokens = episodes = unscored = 0
    for row, group in zip(rows, groups):
        scored = [t for t in group if t.scored]
        scores = [int(t.reward >= threshold) for t in scored]
        episodes += len(group)
        unscored += len(group) - len(scored)
        finishes.update(t.finish or "unknown" for t in group)
        tool_calls += sum(t.tool_calls for t in group)
        tool_errors += sum(t.tool_errors for t in group)
        tokens += sum(getattr(t, "sampled_tokens", 0) for t in group)
        records.append(
            {
                "id": row.get("id"),
                "scores": scores,
                "rewards": [t.reward for t in scored],
                "finishes": [t.finish for t in group],
            }
        )
        if scored:
            rates.append(sum(scores) / len(scores))
            any_pass.append(int(any(scores)))
            rewards.append(sum(t.reward for t in scored) / len(scored))
    summary = {
        "env": spec.name,
        "version": spec.version,
        "rows": len(rows),
        "k": k,
        "episodes": episodes,
        "unscored": unscored,
        "pass@1": sum(rates) / len(rates) if rates else None,
        f"pass@{k}": sum(any_pass) / len(any_pass) if any_pass else None,
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "finishes": {name: count / episodes for name, count in sorted(finishes.items())} if episodes else {},
        "tool_calls_per_episode": tool_calls / episodes if episodes else 0.0,
        "tool_error_rate": tool_errors / tool_calls if tool_calls else 0.0,
        "completion_tokens": tokens / episodes if episodes else 0.0,
        "seconds": seconds,
        "episodes_per_s": episodes / seconds if seconds > 0 else 0.0,
    }
    return summary, records


def evaluate(spec, engine, tokenizer, rows: list[dict], k: int = 4, env_args: dict | None = None, **episode_args):
    """k episodes per row with a rollout engine: (summary, per-row records). episode_args go to run_episodes
    (temperature, chat_template_kwargs, model, max_turns, max_new_tokens, max_model_len, max_concurrent, ...)."""
    from palingenesis.rl.episodes import run_episodes

    episode_args.setdefault("max_turns", spec.max_turns)
    episode_args.setdefault("max_new_tokens", spec.max_new_tokens)
    start = time.perf_counter()
    groups = run_episodes(engine, tokenizer, rows, spec.env_class(), {**spec.env_args, **(env_args or {})}, k=k, **episode_args)
    return summarize(spec, rows, groups, time.perf_counter() - start, k)


def _sandbox(spec, choice: str):
    """The sandbox for environments that run model-written code (None for the others)."""
    if spec.sandbox != "process":
        return None
    choice = choice or "docker"
    if choice == "process":
        from palingenesis.rl.catalog.envs.code import process_sandbox

        print(
            "WARN  --sandbox process: model-written code runs on this machine with resource limits and no isolation",
            file=sys.stderr,
        )
        return process_sandbox()
    from palingenesis.rl.config import SandboxConfig
    from palingenesis.rl.sandbox import make_sandbox

    config = SandboxConfig()
    config.backend = choice
    return make_sandbox(config)


def cmd_run(args) -> int:
    from palingenesis.logging import setup_logging
    from palingenesis.rl.catalog.runner import load_rows

    setup_logging(rank=0)
    spec = _spec(args.name)
    missing = spec.missing_requirements()
    if missing:
        raise SystemExit(f"pgs env run: {spec.name} needs the Python modules {', '.join(missing)}")
    rows = load_rows(spec, args.dataset, args.n, args.seed, **_options(args.option))
    if not rows:
        raise SystemExit("pgs env run: no rows")
    env_args = json.loads(args.env_args) if args.env_args else {}
    sandbox = _sandbox(spec, args.sandbox)
    if sandbox is not None:
        env_args["sandbox"] = sandbox

    from transformers import AutoTokenizer

    from palingenesis.opd.trainer import student_stop_ids
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.screen import _engine

    kwargs = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = student_stop_ids(tokenizer, args.model, [], kwargs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
    config = RLConfig()
    max_turns = args.max_turns or spec.max_turns
    max_new_tokens = args.max_new_tokens or spec.max_new_tokens
    for key, value in {
        "model.policy": args.model,
        "rollout.backend": args.backend,
        "rollout.max_model_len": args.max_model_len,
        "rollout.gpu_memory_utilization": args.gpu_memory_utilization,
        "rollout.max_num_seqs": max(256, min(2048, len(rows) * args.k)),
        "train.seed": args.seed,
    }.items():
        config.set(key, value)
    temperature: Any = args.temperature
    if args.top_p < 1 or args.top_k > 0:
        from palingenesis.opd.rollout import Sampling

        temperature = Sampling(args.temperature, args.top_p, args.top_k)
    engine = _engine(config, args.model, stop_ids, pad_id)
    summary, records = evaluate(
        spec,
        engine,
        tokenizer,
        rows,
        args.k,
        env_args,
        temperature=temperature,
        chat_template_kwargs=kwargs,
        model=args.model,
        max_turns=max_turns,
        max_new_tokens=max_new_tokens,
        max_model_len=args.max_model_len,
        max_concurrent=args.max_concurrent,
        tool_timeout=args.tool_timeout,
    )
    summary.update(model=args.model, dataset=args.dataset or spec.default_dataset, temperature=args.temperature,
                   thinking=args.thinking, backend=args.backend)  # fmt: skip
    if args.output:
        with open(args.output, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        summary["output"] = args.output
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, default=str))
    else:
        def rate(value) -> str:
            return "-" if value is None else f"{value:.3f}"

        print(
            f"{spec.name} {spec.version}  {summary['model']}  {summary['rows']} rows x {args.k}: "
            f"pass@1 {rate(summary['pass@1'])}  pass@{args.k} {rate(summary[f'pass@{args.k}'])}  "
            f"mean reward {rate(summary['mean_reward'])}  {summary['episodes_per_s']:.2f} episodes/s"
        )
        print("  finishes: " + ", ".join(f"{name} {share:.0%}" for name, share in summary["finishes"].items())
              + f"; tool error rate {summary['tool_error_rate']:.1%}; unscored {summary['unscored']}")  # fmt: skip
    return 0


# ------------------------------------------------------------------------ new


def cmd_new(args) -> int:
    from palingenesis.rl.catalog.scaffold import scaffold

    try:
        root = scaffold(args.name, args.dir, args.family)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"pgs env new: {error}") from None
    module = next(root.glob("*.py")).stem
    print(f"{root}/\n  {module}.py              the environment, its generator, policies and SPEC\n"
          f"  tests/test_{module}.py   the audit as a test\n  pyproject.toml           registers SPEC under palingenesis.environments\n")  # fmt: skip
    print(f"  pgs env test {root}/{module}.py:SPEC     audit it without installing\n"
          f"  pip install -e {root}             then it is in `pgs env list` as {args.name}")  # fmt: skip
    return 0


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    from palingenesis.rl.catalog.spec import FAMILIES

    parser = argparse.ArgumentParser(prog="pgs env", description="The environment catalogue: find, audit, run, write.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="the registered environments")
    p.add_argument("--family", default="", choices=("", *FAMILIES))
    p.add_argument("--trainable", action="store_true", help="only environments safe to train on")
    p.add_argument("--eval-only", action="store_true", help="only evaluation-only environments")
    p.add_argument("--permissive", action="store_true", help="only permissively licensed code and data")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_list)

    p = sub.add_parser("info", help="one environment's catalogue entry")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_info)

    p = sub.add_parser("test", help="audit: the oracle must pass, degenerate policies must not")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true", help="every registered environment")
    p.add_argument("--n", type=int, default=32, help="rows to audit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="", help="a dataset of the entry, or a path (default: the entry's first)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_test)

    p = sub.add_parser("rows", help="write rows for pgs rl")
    p.add_argument("name")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="")
    p.add_argument("--option", action="append", default=[], help="a generator option, key=value (repeatable)")
    p.add_argument("--allow-eval-only", action="store_true")
    p.set_defaults(run=cmd_rows)

    p = sub.add_parser("run", help="k episodes per row with a model: pass@1, pass@k, episodes/s")
    p.add_argument("name")
    p.add_argument("--model", required=True, help="a checkpoint directory or a Hugging Face model id")
    p.add_argument("--k", type=int, default=4, help="episodes per row")
    p.add_argument("--n", type=int, default=64, help="rows (0: the whole dataset)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="")
    p.add_argument("--option", action="append", default=[], help="a generator option, key=value (repeatable)")
    p.add_argument("--backend", default="vllm", choices=("vllm", "hf", "sglang"), help="the rollout engine")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--thinking", default="default", choices=("default", "on", "off"), help="enable_thinking of the chat template")
    p.add_argument("--max_turns", type=int, default=0, help="default: the entry's")
    p.add_argument("--max_new_tokens", type=int, default=0, help="per turn; default: the entry's")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_concurrent", type=int, default=0, help="environments alive at once (0: no limit)")
    p.add_argument("--tool_timeout", type=float, default=120.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--sandbox", default="", choices=("", "docker", "process", "agent_sandbox"), help="for code environments (default: docker)")
    p.add_argument("--env_args", default="", help="JSON arguments for the environment class, over the entry's")
    p.add_argument("--output", "-o", default="", help="per-row results (JSONL), readable by pgs compare")
    p.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    p.set_defaults(run=cmd_run)

    p = sub.add_parser("new", help="scaffold an environment package with a test")
    p.add_argument("name")
    p.add_argument("--dir", default=".")
    p.add_argument("--family", default="other", choices=FAMILIES)
    p.set_defaults(run=cmd_new)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if getattr(args, "k", 1) < 1:
        parser.error("--k must be >= 1")
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
