"""`pgs scenario`: data, audits, exports and layouts for multi-role RL (palingenesis.rl.scenarios).

    pgs scenario list
    pgs scenario generate take_away --n 2000 --seed 0 --out data/take_away.jsonl
    pgs scenario audit take_away [--n 500]            # oracle partners must earn the reward, degenerate ones must not
    pgs scenario export runs/x/rollouts --format sft --out sft.jsonl [--roles lead --min-reward 1]
    pgs scenario export runs/x/rollouts --format dpo --out pairs.jsonl [--margin 0.5]
    pgs scenario plan --config rl.yaml --params-b 0.8 --nodes 2 --gpus-per-node 8 --gpu-memory-gb 80

Training itself is `pgs rl --config <a config with a scenario: section>`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from palingenesis.rl.scenarios.base import BUILTIN, build_scenario

    parser = argparse.ArgumentParser(prog="pgs scenario", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the built-in scenarios and their roles")
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
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in BUILTIN:
            scenario = build_scenario(name)
            print(f"{name}: roles {', '.join(scenario.roles)}; outcome {scenario.outcome}")
            for role, table in scenario.partners().items():
                print(f"    {role} partners: {', '.join(table)}")
        return 0
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
    layout = plan(
        Cluster(args.nodes, args.gpus_per_node, args.gpu_memory_gb), workload_of(config, args.params_b), args.config
    )
    print(json.dumps(layout.to_dict(), indent=2))
    return 1 if layout.errors else 0


if __name__ == "__main__":
    sys.exit(main())
