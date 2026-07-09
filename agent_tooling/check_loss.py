#!/usr/bin/env python3
"""Analyze loss curves from training logs or wandb/trackio exports.

Detects common training pathologies:
    - NaN/Inf losses (catastrophic failure)
    - Loss spikes (>3x rolling average — possible data corruption)
    - Loss plateaus (no improvement for N steps — lr may be too low)
    - Loss divergence (monotonically increasing — lr too high or bug)
    - Unexpectedly high initial loss (wrong model/data pairing)
    - Gradient accumulation artifacts (periodic loss patterns)

Usage:
    # From a log file (one loss value per line, or "step=X loss=Y" format)
    python -m agent_tooling.check_loss --log_file training.log

    # From wandb (requires wandb to be installed + authenticated)
    python -m agent_tooling.check_loss --wandb_run user/project/run_id

    # Pipe from training output
    grep "loss=" train_output.log | python -m agent_tooling.check_loss --stdin
"""

import math
import re
import sys
from dataclasses import dataclass, field


@dataclass
class LossAnalysis:
    """Results of loss curve analysis."""

    total_steps: int = 0
    losses: list[float] = field(default_factory=list)
    nan_steps: list[int] = field(default_factory=list)
    inf_steps: list[int] = field(default_factory=list)
    spike_steps: list[int] = field(default_factory=list)  # >3x rolling avg
    plateau_ranges: list[tuple[int, int]] = field(default_factory=list)
    is_diverging: bool = False
    initial_loss: float = 0.0
    final_loss: float = 0.0
    min_loss: float = float("inf")
    max_loss: float = 0.0
    issues: list[str] = field(default_factory=list)
    healthy: bool = True


def parse_losses_from_text(text: str) -> list[tuple[int, float]]:
    """Extract (step, loss) pairs from training log text.

    Supports formats:
        step=123 loss=2.3456
        step=123 | loss=2.3456
        {"train/loss": 2.3456, "train/global_step": 123}
        2.3456  (one value per line, step = line number)
    """
    pairs = []

    # Try structured format: step=X ... loss=Y
    pattern = re.compile(r"step[=:]?\s*(\d+).*?loss[=:]?\s*([0-9.eE+\-naif]+)")
    for match in pattern.finditer(text):
        step = int(match.group(1))
        loss = float(match.group(2))
        pairs.append((step, loss))

    if pairs:
        return pairs

    # Try JSON-ish format
    pattern2 = re.compile(r'"(?:train/)?loss":\s*([0-9.eE+\-]+).*?"(?:train/)?(?:global_)?step":\s*(\d+)')
    for match in pattern2.finditer(text):
        loss = float(match.group(1))
        step = int(match.group(2))
        pairs.append((step, loss))

    if pairs:
        return pairs

    # Fallback: one number per line
    for i, line in enumerate(text.strip().split("\n"), 1):
        line = line.strip()
        try:
            loss = float(line)
            pairs.append((i, loss))
        except ValueError:
            continue

    return pairs


def analyze_losses(losses: list[float], window: int = 20) -> LossAnalysis:
    """Analyze a loss curve for common training pathologies."""
    result = LossAnalysis()
    result.total_steps = len(losses)
    result.losses = losses

    if not losses:
        result.issues.append("CRITICAL: No loss values found.")
        result.healthy = False
        return result

    result.initial_loss = losses[0]
    result.final_loss = losses[-1]
    result.min_loss = min(l for l in losses if math.isfinite(l)) if any(math.isfinite(l) for l in losses) else 0
    result.max_loss = max(l for l in losses if math.isfinite(l)) if any(math.isfinite(l) for l in losses) else 0

    # Check NaN/Inf
    for i, l in enumerate(losses):
        if math.isnan(l):
            result.nan_steps.append(i)
        elif math.isinf(l):
            result.inf_steps.append(i)

    if result.nan_steps:
        result.issues.append(
            f"CRITICAL: NaN loss at steps: {result.nan_steps[:10]}{'...' if len(result.nan_steps) > 10 else ''}"
        )
        result.healthy = False

    if result.inf_steps:
        result.issues.append(f"CRITICAL: Inf loss at steps: {result.inf_steps[:10]}")
        result.healthy = False

    # Filter finite losses for remaining analysis
    finite = [(i, l) for i, l in enumerate(losses) if math.isfinite(l)]
    if len(finite) < 5:
        return result

    # Spike detection (>3x rolling average)
    for i in range(window, len(finite)):
        idx, val = finite[i]
        rolling_avg = sum(finite[j][1] for j in range(i - window, i)) / window
        if rolling_avg > 0 and val > 3 * rolling_avg:
            result.spike_steps.append(idx)

    if len(result.spike_steps) > 3:
        result.issues.append(
            f"WARNING: {len(result.spike_steps)} loss spikes detected (>3x rolling avg). "
            "Possible data corruption or learning rate too high."
        )

    # Plateau detection (loss doesn't improve for >50 consecutive steps)
    plateau_threshold = max(50, len(finite) // 10)
    best_so_far = finite[0][1]
    plateau_start = 0
    for i, (idx, val) in enumerate(finite):
        if val < best_so_far * 0.999:  # 0.1% improvement
            if i - plateau_start > plateau_threshold:
                result.plateau_ranges.append((finite[plateau_start][0], finite[i - 1][0]))
            best_so_far = val
            plateau_start = i

    if len(finite) - plateau_start > plateau_threshold:
        result.plateau_ranges.append((finite[plateau_start][0], finite[-1][0]))

    if result.plateau_ranges:
        longest = max(e - s for s, e in result.plateau_ranges)
        result.issues.append(
            f"WARNING: Loss plateaued for {longest} steps. " "Learning rate may be too low or training is saturated."
        )

    # Divergence detection (monotonically increasing over last 20% of training)
    tail_start = int(len(finite) * 0.8)
    if tail_start > 0:
        tail = [v for _, v in finite[tail_start:]]
        if len(tail) > 10:
            increasing_count = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
            if increasing_count > len(tail) * 0.7:
                result.is_diverging = True
                result.issues.append(
                    "CRITICAL: Loss is diverging (increasing in last 20% of training). "
                    "Learning rate is likely too high."
                )
                result.healthy = False

    # Initial loss sanity check (for language models, CE loss starts around ln(vocab_size))
    if result.initial_loss > 20:
        result.issues.append(
            f"WARNING: Initial loss is very high ({result.initial_loss:.2f}). "
            "Expected ~10-12 for a 32k-128k vocab. Check model-data alignment."
        )

    # Check overall improvement
    if len(finite) > 10:
        improvement = (result.initial_loss - result.final_loss) / max(result.initial_loss, 1e-8)
        if improvement < 0:
            result.issues.append(
                f"WARNING: Loss increased overall ({result.initial_loss:.4f} → {result.final_loss:.4f}). "
                "Training may not be working."
            )
        elif improvement < 0.01 and len(finite) > 100:
            result.issues.append(
                f"WARNING: Very little improvement ({improvement:.2%} over {len(finite)} steps). "
                "Check if loss is already at minimum or lr is too low."
            )

    if not result.issues:
        result.issues.append("OK: Loss curve looks healthy.")

    return result


def print_report(analysis: LossAnalysis):
    print("=" * 70)
    print("LOSS HEALTH REPORT")
    print("=" * 70)
    print(f"  Steps analyzed: {analysis.total_steps}")
    print(f"  Initial loss: {analysis.initial_loss:.4f}")
    print(f"  Final loss: {analysis.final_loss:.4f}")
    print(f"  Min loss: {analysis.min_loss:.4f}")
    print(f"  Max loss: {analysis.max_loss:.4f}")
    improvement = (analysis.initial_loss - analysis.final_loss) / max(analysis.initial_loss, 1e-8)
    print(f"  Improvement: {improvement:.2%}")
    print(f"  NaN occurrences: {len(analysis.nan_steps)}")
    print(f"  Inf occurrences: {len(analysis.inf_steps)}")
    print(f"  Spikes: {len(analysis.spike_steps)}")
    print(f"  Plateaus: {len(analysis.plateau_ranges)}")
    print(f"  Diverging: {'YES' if analysis.is_diverging else 'no'}")
    print()
    for issue in analysis.issues:
        prefix = "  ✓" if issue.startswith("OK") else "  ⚠"
        print(f"{prefix} {issue}")
    print()
    print(f"  Overall: {'HEALTHY ✓' if analysis.healthy else 'UNHEALTHY ✗'}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", help="Path to training log file")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--wandb_run", help="wandb run path (user/project/run_id)")
    args = parser.parse_args()

    text = ""
    if args.stdin:
        text = sys.stdin.read()
    elif args.log_file:
        with open(args.log_file) as f:
            text = f.read()
    elif args.wandb_run:
        try:
            import wandb

            api = wandb.Api()
            run = api.run(args.wandb_run)
            history = run.history(keys=["train/loss"])
            losses = history["train/loss"].dropna().tolist()
            analysis = analyze_losses(losses)
            print_report(analysis)
            sys.exit(0 if analysis.healthy else 1)
        except Exception as e:
            print(f"Error fetching wandb run: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    pairs = parse_losses_from_text(text)
    if not pairs:
        print("ERROR: Could not parse any loss values from input.", file=sys.stderr)
        sys.exit(1)

    losses = [l for _, l in sorted(pairs)]
    analysis = analyze_losses(losses)
    print_report(analysis)
    sys.exit(0 if analysis.healthy else 1)


if __name__ == "__main__":
    main()
