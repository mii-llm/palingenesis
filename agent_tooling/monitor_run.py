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

import sys
from dataclasses import dataclass


@dataclass
class StepInfo:
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    grad_norm: float = 0.0
    step_time: float = 0.0


def parse_training_log(text: str) -> list[StepInfo]:
    """The trainer's step lines (SFT, DEFT, DPO formats alike), one StepInfo per step."""
    from agent_tooling._logparse import parse_steps

    return [
        StepInfo(
            step=f["step"],
            loss=f["loss"],
            lr=f.get("lr", 0.0),
            tokens_per_sec=f.get("tok/s", 0.0),
            grad_norm=f.get("grad_norm", 0.0),
            step_time=f.get("dt", 0.0),
        )
        for f in parse_steps(text)
    ]


def analyze_run(steps: list[StepInfo], max_steps: int | None = None) -> dict:
    """Analyze training run health from step data."""
    if not steps:
        return {"status": "NO_DATA", "issues": ["No training step lines (step=N loss=X ...) found in the log."]}

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
    elif n >= 100:
        # Plateau: judged on 50-step means (single steps are too noisy)
        last50 = sum(s.loss for s in steps[-50:]) / 50
        prev50 = sum(s.loss for s in steps[-100:-50]) / 50
        if abs(last50 - prev50) / max(prev50, 1e-8) < 0.005:
            issues.append("INFO: Loss has plateaued (mean of the last 50 steps within 0.5% of the 50 before).")

    # Gradient norms are logged before clipping and their scale depends on the model,
    # vocabulary and objective, so judge the latest one against the run's own history.
    norms = sorted(s.grad_norm for s in steps[-50:-1] if s.grad_norm > 0)
    if len(norms) >= 10 and latest.grad_norm > 5 * norms[len(norms) // 2]:
        issues.append(
            f"WARNING: Gradient norm spiked to {latest.grad_norm:.1f} "
            f"(5x the recent median {norms[len(norms) // 2]:.1f})."
        )
    if 0 < latest.grad_norm < 1e-6:
        issues.append("WARNING: Gradient norm near zero — possible vanishing gradient or dead training.")

    # Throughput: per-step tok/s varies by design with variable-length batches, so
    # look for a sustained slowdown (recent median well below the earlier median).
    if n >= 20:

        def median(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2]

        now, before = median([s.tokens_per_sec for s in recent]), median([s.tokens_per_sec for s in earlier])
        if before > 0 and now < 0.5 * before:
            issues.append(f"WARNING: Throughput dropped from ~{before:.0f} to ~{now:.0f} tok/s (median of 10 steps).")

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
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_brief(result: dict):
    """One-line status for agent consumption."""
    if result["status"] == "NO_DATA":
        print(f"[NO_DATA] {result['issues'][0]}")
        return
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
    if result["status"] == "NO_DATA":
        print(f"  {result['issues'][0]}\n")
        return
    remaining = f", {result['eta_steps']} remaining" if result.get("eta_steps") is not None else ""
    print(f"  Step: {result['current_step']}{remaining}")
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
    parser.add_argument("--config", help="Training config: takes max_steps from train.max_steps")
    args = parser.parse_args()
    if args.max_steps is None and args.config:
        import agent_tooling._path_setup  # noqa: F401
        from palingenesis.config import Config

        max_steps = Config.from_yaml(args.config).train.max_steps
        args.max_steps = max_steps if max_steps > 0 else None

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
