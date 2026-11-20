"""`pgs scenario`: data, audits, exports and layouts for multi-role RL (palingenesis.rl.scenarios).

    pgs scenario patterns                             # patterns over your own RL dataset: roles, row fields, options
    pgs scenario check --config rl.yaml [--n 200]     # your rows against the config's pattern, then the audit on them
    pgs scenario list
    pgs scenario generate take_away --n 2000 --seed 0 --out data/take_away.jsonl
    pgs scenario audit take_away [--n 500]            # oracle partners must earn the reward, degenerate ones must not
    pgs scenario export runs/x/rollouts --format sft --out sft.jsonl [--roles lead --min-reward 1]
    pgs scenario export runs/x/rollouts --format dpo --out pairs.jsonl [--margin 0.5]
    pgs scenario plan --config rl.yaml --params-b 0.8 --nodes 2 --gpus-per-node 8 --gpu-memory-gb 80
    pgs scenario plan --config rl.yaml --params-b 4 --nodes 2 --gpus-per-node 8 --engine-gpus 8 --hosts n1,n2

Training itself is `pgs rl --config <a config with a scenario: section>`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def describe_patterns() -> list[dict]:
    """The patterns (scenarios over a dataset you bring): roles, row fields, options and scripted partners."""
    import inspect

    from palingenesis.rl.scenarios.base import PATTERNS, Scenario, build_scenario, scenario_class

    out = []
    for name in PATTERNS:
        cls = scenario_class(name)
        options: dict[str, object] = {}
        for klass in cls.__mro__:
            if klass in (Scenario, object):
                break
            for key, parameter in inspect.signature(klass.__init__).parameters.items():
                hidden = ("weights", *getattr(cls, "hidden_options", ()))
                if parameter.default is not inspect.Parameter.empty and key not in hidden:
                    options.setdefault(key, parameter.default)
        scenario = build_scenario(name)
        out.append(
            {
                "pattern": name,
                "summary": (inspect.getdoc(inspect.getmodule(cls)) or "").split("\n")[0].split(": ", 1)[-1],
                "roles": list(scenario.roles),
                "outcome": scenario.outcome,
                "scored_by": "the rewards: section" if scenario.uses_rewards else "the environment's tests",
                "fields": dict(scenario.fields),
                "options": options,
                "weights": scenario.weights,
                "partners": {role: list(table) for role, table in scenario.partners().items()},
            }
        )
    return out


def opening(session) -> list[dict]:
    """The messages a session started from (everything before its first assistant message)."""
    out = []
    for message in session.messages:
        if message.get("role") == "assistant":
            break
        out.append(message)
    return out


def check(path: str, n: int = 200, seed: int = 0, as_json: bool = False) -> int:
    """Validate the dataset of an RL config against its scenario: the fields each row needs, the rewards, one
    rendered prompt per role (and per language), then the audit on the rows themselves. 1 on any failure."""
    import asyncio
    import random

    from palingenesis.rl.config import RLConfig, RLConfigError
    from palingenesis.rl.scenarios.base import build_scenario
    from palingenesis.rl.scenarios.runtime import play

    report: dict = {"config": path, "failures": [], "warnings": [], "prompts": [], "audit": None}
    failures, say = report["failures"], (lambda *a: None) if as_json else print

    def done() -> int:
        if as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        else:
            for warning in report["warnings"]:
                print(f"WARN  {warning}")
            for failure in failures:
                print(f"FAIL  {failure}")
            print("scenario check:", "FAILED" if failures else "passed")
        return 1 if failures else 0

    config = RLConfig.from_yaml(path)
    if not config.scenario.type:
        failures.append("the config has no scenario.type (see `pgs scenario patterns`)")
        return done()
    try:
        config.validate()
    except RLConfigError as error:
        failures.append(f"config: {error}")
        return done()
    scenario = build_scenario(config.scenario.type, config.scenario.args)
    sandbox = None
    try:
        from palingenesis.rl.rewards import code_reward, resolve_rewards

        rewards = resolve_rewards(config.rewards)
        if any(r.fn is code_reward or "sandbox" in r.accepts for r in rewards):
            from palingenesis.rl.sandbox import make_sandbox

            sandbox = make_sandbox(config.sandbox)
    except Exception as error:  # noqa: BLE001 — an unknown built-in, a module that does not import
        failures.append(f"rewards: {type(error).__name__}: {error}")
        return done()
    scenario.bind(rewards, sandbox, config)
    report.update(scenario=scenario.name, roles=list(scenario.roles), rewards=[r.name for r in rewards])
    say(f"scenario    {scenario.name}: roles {', '.join(scenario.roles)}; outcome {scenario.outcome}")
    if scenario.uses_rewards:
        say("rewards     " + ", ".join(f"{r.name} ({r.kind or 'custom'}, x{r.weight:g})" for r in rewards))

    from palingenesis.rl.data import load_rows
    from palingenesis.rl.formats import convert

    try:
        rows = convert(load_rows(config.data.dataset, config.data.split), config.data.format)
    except Exception as error:  # noqa: BLE001 — a missing file, an unreadable dataset
        failures.append(f"data.dataset {config.data.dataset!r}: {type(error).__name__}: {error}")
        return done()
    picked = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    sample = [rows[i] for i in picked]
    problems = {i: scenario.check_row(rows[i]) for i in picked}
    valid = [rows[i] for i in picked if not problems[i]]
    bad = [(i, p[0]) for i, p in problems.items() if p]
    report.update(rows=len(rows), checked=len(sample), valid=len(valid))
    say(f"rows        {len(rows)} in {config.data.dataset}; {len(sample)} checked, {len(valid)} fit the scenario")
    if bad:
        failures.append(f"{len(bad)} of {len(sample)} rows do not fit; row {bad[0][0]}: {bad[0][1]}")
    report["warnings"] += scenario.advice(valid)
    if not valid:
        return done()

    table = scenario.partners()
    oracles = {role: table.get(role, {}).get("oracle") for role in scenario.roles}
    if not all(oracles.values()):
        failures.append(f"no oracle partner for {[r for r, p in oracles.items() if p is None]}")
        return done()
    try:
        scenario.prepare(valid)
        shown: dict[str, dict] = {}
        for row in valid:  # one row per language the prompts are written in
            shown.setdefault(getattr(scenario, "language", lambda r: "")(row), row)
        for language, row in shown.items():
            episode = play(scenario, [row], oracles, seed)[0][0]
            first: dict[str, list] = {}
            for session in episode.sessions:
                first.setdefault(session.role, opening(session))
            for role in scenario.roles:
                report["prompts"].append({"role": role, "language": language, "messages": first.get(role)})
                say(f"\n--- prompt of {role}" + (f" ({language})" if language else "") + " ---")
                for message in first.get(role) or [
                    {"role": "-", "content": "(no session of this role in the episode)"}
                ]:
                    say(f"[{message['role']}] {message.get('content')}")
        say("")
        audit = scenario.audit(seed=seed, rows=valid)
    except Exception as error:  # noqa: BLE001 — a reward reading a missing column, an environment that fails
        failures.append(
            f"the episodes could not be played: {type(error).__name__}: {error} (a reward or the scenario read "
            "something the rows do not have: check rewards.<name>.args and scenario.args)"
        )
        return done()
    finally:
        if sandbox is not None:
            try:
                asyncio.run(sandbox.close())
            except Exception:  # noqa: BLE001 — nothing depends on it
                pass
    report["audit"] = audit
    for role, value in audit["oracle"].items():
        say(f"oracle      {role:<24} {value:+.3f}")
    for name, value in audit["degenerate"].items():
        say(f"degenerate  {name:<24} {value:+.3f}")
    failures += audit["failures"]
    if any(f.startswith("oracle") for f in audit["failures"]) and scenario.uses_rewards:
        report["warnings"].append(
            "an oracle answers with the gold column: when it is not paid, the gold text does not pass your rewards "
            "as a final answer (a worked solution in the column, a reward that wants another format), a channel "
            "limit cuts it, or the rewards also pay wrong answers. Check scenario.args.gold_field, answer_tag and "
            "rewards.<name>.args."
        )
    return done()


def main(argv: list[str] | None = None) -> int:
    from palingenesis.rl.scenarios.base import BUILTIN, GENERATED, PATTERNS, build_scenario

    parser = argparse.ArgumentParser(prog="pgs scenario", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the built-in scenarios and their roles")
    pt = sub.add_parser("patterns", help="the patterns over your own RL dataset: roles, row fields, options")
    pt.add_argument("--json", action="store_true")
    c = sub.add_parser("check", help="validate a config's dataset against its scenario, then audit it on those rows")
    c.add_argument("--config", required=True, help="the RL config (data, rewards, scenario)")
    c.add_argument("--n", type=int, default=200, help="rows to check and to audit on")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--json", action="store_true")
    g = sub.add_parser("generate", help="write a scenario's rows (seeded)")
    g.add_argument("scenario")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    g.add_argument("--params", default="{}", help="JSON: the generator's own options")
    a = sub.add_parser("audit", help="play the scripted partners through the verifier")
    a.add_argument("scenario")
    a.add_argument("--n", type=int, default=500)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--args", default="{}", help="JSON: the scenario's options (scenario.args)")
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("export", help="rollout dumps of a scenario run as SFT / DPO / distillation rows")
    e.add_argument("directory", help="<output_dir>/rollouts")
    e.add_argument("--format", choices=("sft", "dpo", "traces"), default="sft")
    e.add_argument("--out", required=True)
    e.add_argument("--roles", nargs="*", default=None)
    e.add_argument("--min-reward", type=float, default=1.0)
    e.add_argument("--margin", type=float, default=0.5)
    p = sub.add_parser("plan", help="a cluster and a config turned into a layout")
    p.add_argument("--config", required=True)
    p.add_argument("--params-b", type=float, required=True, help="the policy's parameters, in billions")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--gpu-memory-gb", type=float, default=80.0)
    p.add_argument(
        "--engine-gpus", type=int, default=0, help="GPUs given to engine workers (0: engines share the trainers' GPUs)"
    )
    p.add_argument("--engine-tp", type=int, default=1, help="GPUs per engine (tensor parallelism)")
    p.add_argument("--hosts", default="", help="comma-separated node names, in order")
    p.add_argument("--link-gbps", type=float, default=25.0, help="the network between nodes")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in BUILTIN:
            scenario = build_scenario(name)
            kind = "generator" if name in GENERATED else "pattern over your dataset"
            print(f"{name} ({kind}): roles {', '.join(scenario.roles)}; outcome {scenario.outcome}")
            for role, table in scenario.partners().items():
                print(f"    {role} partners: {', '.join(table)}")
        return 0
    if args.command == "patterns":
        patterns = describe_patterns()
        if args.json:
            print(json.dumps(patterns, indent=2))
            return 0
        for p in patterns:
            print(f"{p['pattern']}: {p['summary']}")
            print(f"    roles       {', '.join(p['roles'])} (outcome: {p['outcome']}); scored by {p['scored_by']}")
            print("    row fields")
            for name, what in p["fields"].items():
                print(f"        {name:<40} {what}")
            print("    options     " + ", ".join(f"{k}={v!r}" for k, v in p["options"].items()))
            for role, names in p["partners"].items():
                print(f"    partners    {role}: {', '.join(names)}")
        print(
            "\nUse one with scenario: {type: <pattern>, args: {...}} in an RL config, then pgs scenario check --config."
        )
        return 0
    if args.command == "check":
        return check(args.config, args.n, args.seed, args.json)
    if args.command in ("audit", "generate") and args.scenario in PATTERNS:
        print(
            f"{args.scenario} is a pattern over your own dataset: it has no generator, and it is audited on your "
            "rows with `pgs scenario check --config <rl.yaml>`."
        )
        return 1
    if args.command == "generate":
        rows = build_scenario(args.scenario).generate(args.n, args.seed, **json.loads(args.params))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"{len(rows)} rows of {args.scenario} (seed {args.seed}) -> {args.out}")
        return 0
    if args.command == "audit":
        report = build_scenario(args.scenario, json.loads(args.args)).audit(args.n, args.seed)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for role, value in report["oracle"].items():
                print(f"oracle      {role:<24} {value:+.3f}")
            for name, value in report["degenerate"].items():
                print(f"degenerate  {name:<24} {value:+.3f}")
            for failure in report["failures"]:
                print(f"FAIL  {failure}")
            print("verifier audit:", "FAILED" if report["failures"] else "passed")
        return 1 if report["failures"] else 0
    if args.command == "export":
        from palingenesis.rl.scenarios.export import dpo_rows, sft_rows
        from palingenesis.rollouts import read_records

        records = read_records(args.directory)
        if args.format == "dpo":
            rows = dpo_rows(records, args.roles, args.margin)
        else:
            rows = sft_rows(records, args.roles, None if args.format == "traces" else args.min_reward)
        count = 0
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        print(f"{count} {args.format} rows -> {args.out}")
        return 0 if count else 1
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.scenarios.placement import Cluster, plan, workload_of

    config = RLConfig.from_yaml(args.config)
    if args.engine_gpus:
        from palingenesis.rl.scenarios.config import BASE, adapter_names
        from palingenesis.rl.scenarios.placement import plan_pool, worker_flags

        pool = plan_pool(
            Cluster(args.nodes, args.gpus_per_node, args.gpu_memory_gb),
            workload_of(config, args.params_b),
            args.engine_gpus,
            args.config,
            hosts=args.hosts.split(",") if args.hosts else None,
            model=config.model.policy,
            engine_tp=args.engine_tp,
            link_gbps=args.link_gbps,
            adapters=len(adapter_names(config)),
            flags=worker_flags(config),
            base_partner=any(role.partner == BASE for role in config.roles.values()),
        )
        print(json.dumps(pool.to_dict(), indent=2))
        return 1 if pool.errors else 0
    layout = plan(
        Cluster(args.nodes, args.gpus_per_node, args.gpu_memory_gb), workload_of(config, args.params_b), args.config
    )
    print(json.dumps(layout.to_dict(), indent=2))
    return 1 if layout.errors else 0


if __name__ == "__main__":
    sys.exit(main())
