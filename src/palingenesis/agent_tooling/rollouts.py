#!/usr/bin/env python3
"""Read the rollout dumps of an RL or on-policy distillation run (logging.dump_trajectories).

Reports what the policy does, not what the loss says:
    - how episodes end: stop / length (token budget) / turns (turn limit) / env_done / error
    - rewards and their components, zero-variance groups (RL); the sampled KL to the teacher (OPD)
    - tool calls: error rate and latency per tool
    - repetition loops (the text repeats itself, or one call is made 3+ times)
    - the slowest turns and tool calls, the worst trajectories (lowest reward, longest, highest KL)
    - alarms when a share crosses its threshold (exit code 1)

Usage:
    pgs rollouts --dir outputs/rl                     # the run's output_dir or its rollouts/ directory
    pgs rollouts --dir outputs/rl --since 200 --json  # dumped steps >= 200, as JSON
    pgs rollouts --dir outputs/rl --threshold loop_share=0.05
"""

import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Alarm thresholds: a share above its value is an issue (--threshold name=value overrides one).
THRESHOLDS = {
    "length_share": 0.2,  # episodes cut by the token budget (rollout.max_completion_tokens, max_model_len)
    "turns_share": 0.2,  # episodes cut by the turn limit (env.max_turns)
    "error_share": 0.02,  # rollouts that failed (left out of training)
    "loop_share": 0.1,  # trajectories with a repetition loop
    "tool_error_rate": 0.3,  # failed tool calls (bad names or arguments, timeouts, exceptions) per call
    "zero_variance_share": 0.5,  # RL: dumped groups whose rollouts all got the same reward (no signal)
}
EXPLAIN = {
    "length_share": "episodes hit the token budget: raise the budget, or penalise length (loss.overlong_*)",
    "turns_share": "episodes hit the turn limit: the policy does not finish; read the worst trajectories",
    "error_share": "rollouts failed (environment, sandbox or parser errors): see info.error in the dump",
    "loop_share": "repetition loops: see the loop rows of the AGENTS.md decision tree",
    "tool_error_rate": "tool calls fail: check the per-tool table (schemas, arguments, timeouts)",
    "zero_variance_share": "most groups carry no signal: tasks too easy or too hard for the policy",
}


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _last_text(record: dict) -> str | None:
    for turn in reversed(record.get("turns") or []):
        if isinstance(turn.get("content"), str) and turn["content"]:
            text = turn["content"].strip()
            return text if len(text) <= 300 else text[:150] + " ... " + text[-150:]
    return None


class _Top:
    """The k records with the largest key, kept as small summaries."""

    def __init__(self, k: int):
        self.k, self.heap, self.count = k, [], 0

    def add(self, key: float, item: dict) -> None:
        self.count += 1
        entry = (key, self.count, item)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        elif key > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def items(self) -> list[dict]:
        return [item for _, _, item in sorted(self.heap, key=lambda e: (-e[0], e[1]))]


def analyze(records, top: int = 5, thresholds: dict | None = None) -> dict:
    """The report of an iterable of trajectory records (palingenesis.rollouts.read_records)."""
    limits = {**THRESHOLDS, **(thresholds or {})}
    n = 0
    steps, trainers = set(), Counter()
    finishes: Counter = Counter()
    rewards: list[float] = []
    components: dict[str, list[float]] = defaultdict(list)
    groups: dict[tuple, bool] = {}  # (step, rank, group) -> kept (RL)
    loops = 0
    calls = errors = 0
    tools: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "seconds": []})
    output_tokens: list[float] = []
    turn_seconds: list[float] = []
    k1: list[float] = []
    slow_turns, slow_calls = _Top(top), _Top(top)
    lowest, longest, highest_k1 = _Top(top), _Top(top), _Top(top)
    for r in records:
        n += 1
        step = r.get("step")
        steps.add(step)
        trainers[r.get("trainer", "rl")] += 1
        finish = r.get("finish") or "unknown"
        finishes[finish] += 1
        ident = {"step": step, "trajectory_id": r.get("trajectory_id"), "finish": finish}
        tokens = r.get("gen_ai.usage.output_tokens") or 0
        output_tokens.append(tokens)
        if "group" in r and "kept" in r:
            groups[(step, r.get("rank", 0), r["group"])] = bool(r["kept"])
        if r.get("loop"):
            loops += 1
        reward = r.get("reward")
        if isinstance(reward, int | float) and r.get("scored", True) and finish != "error":
            rewards.append(reward)
            lowest.add(-reward, {**ident, "reward": reward, "tokens": tokens, "text": _last_text(r)})
        for name, value in (r.get("rewards") or {}).items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                components[name].append(float(value))
        if isinstance(r.get("k1_mean"), int | float):
            k1.append(r["k1_mean"])
            highest_k1.add(r["k1_mean"], {**ident, "k1_mean": r["k1_mean"], "tokens": tokens, "text": _last_text(r)})
        longest.add(tokens, {**ident, "tokens": tokens, "reward": reward, "text": _last_text(r)})
        calls += r.get("tool_calls") or 0
        errors += r.get("tool_errors") or 0
        for i, turn in enumerate(r.get("turns") or []):
            seconds = turn.get("duration_s")
            if isinstance(seconds, int | float):
                turn_seconds.append(seconds)
                slow_turns.add(
                    seconds, {**ident, "turn": i, "seconds": seconds, "tokens": turn.get("gen_ai.usage.output_tokens")}
                )
            for call in turn.get("tool_calls") or []:
                name = call.get("gen_ai.tool.name") or "(unnamed)"
                tool = tools[name]
                tool["calls"] += 1
                tool["errors"] += bool(call.get("error"))
                if isinstance(call.get("duration_s"), int | float):
                    tool["seconds"].append(call["duration_s"])
                    slow_calls.add(
                        call["duration_s"], {**ident, "turn": i, "tool": name, "seconds": call["duration_s"]}
                    )
    if not n:
        return {"status": "NO_DATA", "trajectories": 0, "alarms": ["no trajectory records found"]}

    shares = {f"{k}_share": v / n for k, v in finishes.items()}
    report: dict = {
        "trajectories": n,
        "steps": [min(s for s in steps if s is not None), max(s for s in steps if s is not None)]
        if any(s is not None for s in steps)
        else None,
        "dumped_steps": len(steps),
        "trainer": dict(trainers),
        "finish": {k: round(v / n, 4) for k, v in finishes.most_common()},
        "loop_share": round(loops / n, 4),
        "output_tokens": _summary(output_tokens),
    }
    values = {**shares, "loop_share": loops / n}
    if rewards:
        report["reward"] = _summary(rewards)
        report["reward_components"] = {k: _summary(v)["mean"] for k, v in sorted(components.items())}
    if groups:
        zero = sum(not kept for kept in groups.values()) / len(groups)
        report["groups"] = len(groups)
        report["zero_variance_share"] = round(zero, 4)
        values["zero_variance_share"] = zero
    if k1:
        report["k1_mean"] = _summary(k1)
    if calls or tools:
        report["tool_calls"] = calls or sum(t["calls"] for t in tools.values())
        rate = errors / max(report["tool_calls"], 1)
        report["tool_error_rate"] = round(rate, 4)
        values["tool_error_rate"] = rate
        report["tools"] = {
            name: {
                "calls": t["calls"],
                "error_rate": round(t["errors"] / t["calls"], 4),
                **({"seconds": _summary(t["seconds"])} if t["seconds"] else {}),
            }
            for name, t in sorted(tools.items(), key=lambda kv: -kv[1]["calls"])
        }
    if turn_seconds:
        report["turn_seconds"] = _summary(turn_seconds)
    report["slowest_turns"] = slow_turns.items()
    report["slowest_tool_calls"] = slow_calls.items()
    report["worst"] = {
        "lowest_reward": lowest.items(),
        "longest": longest.items(),
        **({"highest_k1": highest_k1.items()} if k1 else {}),
    }
    alarms = [
        f"{name} {values[name]:.3f} > {limit:g}: {EXPLAIN[name]}"
        for name, limit in limits.items()
        if name in values and values[name] > limit
    ]
    report["alarms"] = alarms
    report["status"] = "ISSUES" if alarms else "HEALTHY"
    return report


def _fmt(summary: dict, unit: str = "") -> str:
    if not summary:
        return "-"
    return (
        f"mean {summary['mean']:.4g}{unit}  p50 {summary['p50']:.4g}{unit}  p95 {summary['p95']:.4g}{unit}  "
        f"max {summary['max']:.4g}{unit}  (n={summary['n']})"
    )


def print_report(report: dict) -> None:
    print("=" * 70)
    print("ROLLOUTS")
    print("=" * 70)
    if report["status"] == "NO_DATA":
        print(f"  {report['alarms'][0]}\n")
        return
    span = f"steps {report['steps'][0]}..{report['steps'][1]}, " if report.get("steps") else ""
    print(f"  {report['trajectories']} trajectories, {span}{report['dumped_steps']} dumped step(s)")
    print("  Finish: " + "  ".join(f"{k} {v:.1%}" for k, v in report["finish"].items()))
    print(f"  Loops: {report['loop_share']:.1%}")
    print(f"  Output tokens: {_fmt(report['output_tokens'])}")
    if "reward" in report:
        print(f"  Reward: {_fmt(report['reward'])}")
        for name, mean in report["reward_components"].items():
            print(f"    {name}: {mean:.4g}")
    if "zero_variance_share" in report:
        print(f"  Zero-variance groups: {report['zero_variance_share']:.1%} of {report['groups']}")
    if "k1_mean" in report:
        print(f"  Sampled KL to the teacher (per token): {_fmt(report['k1_mean'])}")
    if "turn_seconds" in report:
        print(f"  Generation per turn: {_fmt(report['turn_seconds'], 's')}")
    if "tools" in report:
        print(f"  Tool calls: {report['tool_calls']}, errors per call {report['tool_error_rate']:.1%}")
        for name, t in list(report["tools"].items())[:15]:
            seconds = t.get("seconds")
            timing = (
                f"  p50 {seconds['p50']:.3g}s  p95 {seconds['p95']:.3g}s  max {seconds['max']:.3g}s" if seconds else ""
            )
            print(f"    {name:<28} {t['calls']:>7} calls  {t['error_rate']:>6.1%} errors{timing}")
    if report["slowest_turns"]:
        print("  Slowest turns:")
        for t in report["slowest_turns"]:
            print(f"    {t['seconds']:.2f}s  {t['trajectory_id']} turn {t['turn']} ({t['tokens']} tokens)")
    if report["slowest_tool_calls"]:
        print("  Slowest tool calls:")
        for t in report["slowest_tool_calls"]:
            print(f"    {t['seconds']:.2f}s  {t['tool']}  {t['trajectory_id']} turn {t['turn']}")
    for title, key, field in (
        ("Lowest reward", "lowest_reward", "reward"),
        ("Longest", "longest", "tokens"),
        ("Highest KL", "highest_k1", "k1_mean"),
    ):
        rows = report["worst"].get(key)
        if rows:
            print(f"  {title}:")
            for t in rows:
                value = t[field]
                shown = f"{value:.4g}" if isinstance(value, float) else value
                print(f"    {field}={shown}  {t['trajectory_id']} ({t['finish']}, {t['tokens']} tokens)")
                if t.get("text"):
                    print(f"      {t['text']!r}")
    print()
    if report["alarms"]:
        for alarm in report["alarms"]:
            print(f"  ⚠ {alarm}")
    else:
        print("  ✓ No alarms.")
    print()


def main():
    import argparse

    from palingenesis.rollouts import read_records

    parser = argparse.ArgumentParser(description="Read RL / on-policy distillation rollout dumps")
    parser.add_argument("--dir", required=True, help="The run's output_dir, or its rollouts/ directory")
    parser.add_argument("--since", type=int, default=0, help="Only dumped steps >= this one")
    parser.add_argument("--step", type=int, action="append", help="Only this dumped step (repeatable)")
    parser.add_argument("--top", type=int, default=5, help="Rows in each slowest / worst list")
    parser.add_argument(
        "--threshold", action="append", default=[], help=f"name=value, overrides an alarm ({', '.join(THRESHOLDS)})"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON (for agent consumption)")
    args = parser.parse_args()

    thresholds = {}
    for item in args.threshold:
        name, _, value = item.partition("=")
        if name not in THRESHOLDS:
            parser.error(f"unknown threshold {name!r} (one of {', '.join(THRESHOLDS)})")
        thresholds[name] = float(value)
    directory = Path(args.dir)
    if (directory / "rollouts").is_dir():
        directory = directory / "rollouts"
    records = (r for r in read_records(directory, args.step) if r.get("step", 0) >= args.since)
    report = analyze(records, args.top, thresholds)
    report["directory"] = str(directory)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    sys.exit(0 if report["status"] == "HEALTHY" else 1)


if __name__ == "__main__":
    main()
