#!/usr/bin/env python3
"""Monitor an active or completed training run.

Tails a training log file and reports real-time statistics:
    - Current loss and trend
    - Throughput (tokens/sec)
    - ETA to completion
    - Whether training appears stuck or healthy

Usage:
    # Monitor from log file (can be running or completed)
    python -m agent_tooling.monitor_run --log_file outputs/train.log

    # Monitor last N steps only
    python -m agent_tooling.monitor_run --log_file outputs/train.log --last 50

    # Get quick status (for agent use)
    python -m agent_tooling.monitor_run --log_file outputs/train.log --brief
"""

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StepInfo:
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    grad_norm: float = 0.0
    step_time: float = 0.0


def parse_training_log(text: str) -> list[StepInfo]:
    """Parse structured training log lines into StepInfo objects.

    Expected format:
        step=123 loss=2.3456 lr=1.00e-05 tok/s=5432 grad_norm=0.456 dt=1.23s
    """
    steps = []

    # Pattern for our training loop output
    pattern = re.compile(
        r"step=(\d+)\s+loss=([0-9.eE+\-naif]+)\s+lr=([0-9.eE+\-]+)\s+"
        r"tok/s=([0-9.]+)\s+grad_norm=([0-9.eE+\-naif]+)(?:\s+dt=([0-9.]+)s)?"
    )

    for line in text.split("\n"):
        m = pattern.search(line)
        if m:
            info = StepInfo(
                step=int(m.group(1)),
                loss=float(m.group(2)),
                lr=float(m.group(3)),
                tokens_per_sec=float(m.group(4)),
                grad_norm=float(m.group(5)),
                step_time=float(m.group(6)) if m.group(6) else 0.0,
            )
            steps.append(info)

    return steps


def analyze_run(steps: list[StepInfo], max_steps: int | None = None) -> dict:
    """Analyze training run health from step data."""
    if not steps:
        return {"status": "NO_DATA", "message": "No training steps found in log."}

    latest = steps[-1]
    first = steps[0]
    n = len(steps)

    # Trends (last 10 steps vs previous 10)
    recent = steps[-10:] if n >= 10 else steps
    earlier = steps[-20:-10] if n >= 20 else steps[: max(n // 2, 1)]

    avg_recent_loss = sum(s.loss for s in recent) / len(recent)
    avg_earlier_loss = sum(s.loss for s in earlier) / len(earlier) if earlier else avg_recent_loss
    loss_trend = (avg_recent_loss - avg_earlier_loss) / max(avg_earlier_loss, 1e-8)

    avg_tok_s = sum(s.tokens_per_sec for s in recent) / len(recent)
    avg_step_time = sum(s.step_time for s in recent) / len(recent) if recent[0].step_time > 0 else 0

    # ETA
    eta_steps = (max_steps - latest.step) if max_steps and max_steps > latest.step else None
    eta_seconds = eta_steps * avg_step_time if eta_steps and avg_step_time > 0 else None

    # Health checks
    issues = []
    import math

    if math.isnan(latest.loss) or math.isinf(latest.loss):
        issues.append("CRITICAL: Latest loss is NaN/Inf — training has crashed.")
    elif loss_trend > 0.1 and n > 20:
        issues.append(f"WARNING: Loss trending UP ({loss_trend:+.1%} over last 20 steps). May be diverging.")
    elif abs(loss_trend) < 0.001 and n > 50:
        issues.append("INFO: Loss has plateaued — may need LR adjustment or more data diversity.")

    if latest.grad_norm > 10:
        issues.append(f"WARNING: Gradient norm is high ({latest.grad_norm:.1f}). Clipping may be too loose.")
    if latest.grad_norm < 1e-6:
        issues.append("WARNING: Gradient norm near zero — possible vanishing gradient or dead training.")

    # Throughput stability
    if n > 5:
        tok_rates = [s.tokens_per_sec for s in recent]
        if min(tok_rates) < max(tok_rates) * 0.5:
            issues.append("WARNING: Throughput is unstable (>2x variation). Possible data loading bottleneck.")

    status = "HEALTHY"
    if any("CRITICAL" in i for i in issues):
        status = "CRASHED"
    elif any("WARNING" in i for i in issues):
        status = "WARNING"

    if not issues:
        issues.append("Training is progressing normally.")

    return {
        "status": status,
        "current_step": latest.step,
        "current_loss": round(latest.loss, 4),
        "current_lr": latest.lr,
        "current_grad_norm": round(latest.grad_norm, 4),
        "tokens_per_sec": round(avg_tok_s, 0),
        "step_time_s": round(avg_step_time, 2),
        "loss_trend_pct": round(loss_trend * 100, 2),
        "total_steps_logged": n,
        "initial_loss": round(first.loss, 4),
        "improvement_pct": round((first.loss - latest.loss) / max(first.loss, 1e-8) * 100, 1),
        "eta_steps": eta_steps,
        "eta_seconds": round(eta_seconds, 0) if eta_seconds else None,
        "eta_human": _format_time(eta_seconds) if eta_seconds else "unknown",
        "issues": issues,
    }


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_brief(result: dict):
    """One-line status for agent consumption."""
    print(
        f"[{result['status']}] step={result['current_step']} "
        f"loss={result['current_loss']} "
        f"trend={result['loss_trend_pct']:+.1f}% "
        f"tok/s={result['tokens_per_sec']:.0f} "
        f"eta={result['eta_human']}"
    )


def print_full(result: dict):
    print("=" * 70)
    print("TRAINING RUN MONITOR")
    print("=" * 70)
    print(f"  Status: {result['status']}")
    print(f"  Step: {result['current_step']} (of {result.get('eta_steps', '?')} remaining)")
    print(
        f"  Loss: {result['current_loss']} (initial: {result['initial_loss']}, improvement: {result['improvement_pct']:.1f}%)"
    )
    print(f"  Loss trend: {result['loss_trend_pct']:+.2f}% (last 20 steps)")
    print(f"  Learning rate: {result['current_lr']:.2e}")
    print(f"  Gradient norm: {result['current_grad_norm']}")
    print(f"  Throughput: {result['tokens_per_sec']:.0f} tok/s ({result['step_time_s']:.2f}s/step)")
    print(f"  ETA: {result['eta_human']}")
    print()
    for issue in result["issues"]:
        prefix = "  ✓" if "normally" in issue else "  ⚠"
        print(f"{prefix} {issue}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--last", type=int, help="Only analyze last N steps")
    parser.add_argument("--brief", action="store_true", help="One-line output")
    parser.add_argument("--max_steps", type=int, help="Total expected training steps")
    args = parser.parse_args()

    with open(args.log_file) as f:
        text = f.read()

    steps = parse_training_log(text)
    if args.last and len(steps) > args.last:
        steps = steps[-args.last :]

    result = analyze_run(steps, max_steps=args.max_steps)

    if args.brief:
        print_brief(result)
    else:
        print_full(result)

    sys.exit(0 if result["status"] == "HEALTHY" else 1)


if __name__ == "__main__":
    main()
