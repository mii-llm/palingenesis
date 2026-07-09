#!/usr/bin/env python3
"""All-in-one training health diagnostic.

Runs all available checks in sequence and produces a unified report.
Designed to be called by an AI agent to quickly determine if training
is healthy or what's wrong.

Usage:
    # Pre-training check (no GPU needed for masking + memory)
    python -m agent_tooling.diagnose --config configs/llama3_8b.yaml --mode pre

    # Post-training check (analyzes loss log)
    python -m agent_tooling.diagnose --config configs/llama3_8b.yaml --mode post --log_file train.log

    # Full check (requires GPU — runs gradient check too)
    python -m agent_tooling.diagnose --config configs/llama3_8b.yaml --mode full

Output: Structured JSON-compatible report suitable for agent consumption.
"""

import json
import sys
import time

import agent_tooling._path_setup  # noqa: F401

from palingenesis.config import Config


def diagnose_pre(config: Config, gpu_memory_gb: float = 80.0) -> dict:
    """Pre-training diagnostics — no GPU needed."""
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "pre", "checks": {}}

    # 1. Memory estimate
    print("Running memory estimate...")
    from agent_tooling.profile_memory import estimate_memory

    mem = estimate_memory(config, gpu_memory_gb)
    report["checks"]["memory"] = {
        "status": "pass" if mem["fits"] else "fail",
        "estimated_gb": round(mem["total_estimated_gb"], 1),
        "available_gb": mem["gpu_memory_gb"],
        "headroom_gb": round(mem["headroom_gb"], 1),
    }

    # 2. Masking validation
    print("Running masking validation (50 samples)...")
    from agent_tooling.validate_masking import validate

    masking = validate(config, num_samples=50)
    has_critical = any("CRITICAL" in i or "BUG" in i for i in masking["issues"])
    report["checks"]["masking"] = {
        "status": "fail" if has_critical else "pass",
        "train_ratio": round(masking["total_trained"] / max(masking["total_tokens"], 1), 3),
        "samples_checked": masking["total_samples"],
        "issues": masking["issues"],
    }

    # 3. Config sanity
    issues = _check_config_sanity(config)
    report["checks"]["config"] = {
        "status": "pass" if not issues else "warn",
        "issues": issues,
    }

    # Overall
    all_pass = all(c["status"] == "pass" for c in report["checks"].values())
    report["overall"] = "HEALTHY" if all_pass else "ISSUES_FOUND"
    return report


def diagnose_post(config: Config, log_file: str) -> dict:
    """Post-training diagnostics — analyzes loss logs."""
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "post", "checks": {}}

    # 1. Loss analysis
    print(f"Analyzing loss from: {log_file}")
    from agent_tooling.check_loss import analyze_losses, parse_losses_from_text

    with open(log_file) as f:
        text = f.read()
    pairs = parse_losses_from_text(text)
    if pairs:
        losses = [l for _, l in sorted(pairs)]
        analysis = analyze_losses(losses)
        report["checks"]["loss"] = {
            "status": "pass" if analysis.healthy else "fail",
            "steps": analysis.total_steps,
            "initial": round(analysis.initial_loss, 4),
            "final": round(analysis.final_loss, 4),
            "nan_count": len(analysis.nan_steps),
            "spike_count": len(analysis.spike_steps),
            "diverging": analysis.is_diverging,
            "issues": analysis.issues,
        }
    else:
        report["checks"]["loss"] = {"status": "fail", "issues": ["Could not parse loss values from log."]}

    report["overall"] = "HEALTHY" if all(c["status"] == "pass" for c in report["checks"].values()) else "ISSUES_FOUND"
    return report


def diagnose_full(config: Config, gpu_memory_gb: float = 80.0) -> dict:
    """Full diagnostics — requires GPU."""
    report = diagnose_pre(config, gpu_memory_gb)
    report["mode"] = "full"

    # Add gradient check
    import torch

    if torch.cuda.is_available():
        print("Running gradient health check (1 forward-backward pass)...")
        from agent_tooling.check_gradients import check_gradients

        grad_result = check_gradients(config)
        has_critical = any("CRITICAL" in i for i in grad_result["issues"])
        report["checks"]["gradients"] = {
            "status": "fail" if has_critical else "pass",
            "total_grad_norm": round(grad_result["total_grad_norm"], 4),
            "loss": round(grad_result["loss"], 4),
            "dead_layers": len(grad_result.get("dead_layers", [])),
            "issues": grad_result["issues"],
        }
    else:
        report["checks"]["gradients"] = {"status": "skip", "reason": "No GPU available"}

    report["overall"] = (
        "HEALTHY" if all(c["status"] in ("pass", "skip") for c in report["checks"].values()) else "ISSUES_FOUND"
    )
    return report


def _check_config_sanity(config: Config) -> list[str]:
    """Static config checks for common mistakes."""
    issues = []

    # LR sanity
    if config.train.learning_rate > 1e-3:
        issues.append(f"Learning rate {config.train.learning_rate} is very high for SFT. Consider 1e-5 to 5e-5.")
    if config.train.learning_rate < 1e-7:
        issues.append(f"Learning rate {config.train.learning_rate} is very low. Training may not converge.")

    # Batch size * grad accum
    effective_batch = config.train.per_device_batch_size * config.train.gradient_accumulation_steps
    if effective_batch < 4:
        issues.append(f"Effective batch size is {effective_batch} — very small. May be noisy.")
    if effective_batch > 512:
        issues.append(f"Effective batch size is {effective_batch} — very large. May need higher LR.")

    # Seq length + context parallel
    if config.data.max_seq_length > 32768 and not config.parallel.context_parallel:
        issues.append(
            f"Sequence length is {config.data.max_seq_length} but Context Parallel is off. "
            "Consider enabling for memory efficiency on multi-GPU."
        )

    # Chunked loss for long sequences
    if config.data.max_seq_length > 4096 and not config.memory.chunked_loss:
        issues.append(
            "Long sequences without chunked loss — CE will materialize huge logit tensor. "
            "Enable memory.chunked_loss=true."
        )

    # Gradient checkpointing
    if config.train.gradient_checkpointing == "none" and config.data.max_seq_length > 2048:
        issues.append("No activation checkpointing with long sequences — may OOM. Enable selective or full.")

    # Warmup
    if config.train.warmup_ratio == 0:
        issues.append("No warmup — first steps may have unstable gradients.")

    return issues


def print_report(report: dict):
    print("\n" + "=" * 70)
    print(f"TRAINING DIAGNOSTIC REPORT ({report['mode'].upper()} mode)")
    print(f"Time: {report['timestamp']}")
    print("=" * 70)

    for name, check in report["checks"].items():
        status_icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}[check["status"]]
        print(f"\n  [{status_icon}] {name.upper()}")
        for k, v in check.items():
            if k in ("status", "issues"):
                continue
            print(f"      {k}: {v}")
        if "issues" in check:
            for issue in check["issues"]:
                print(f"      → {issue}")

    print(f"\n{'=' * 70}")
    overall_icon = "✓" if report["overall"] == "HEALTHY" else "✗"
    print(f"  [{overall_icon}] OVERALL: {report['overall']}")
    print("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="All-in-one training diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pre", "post", "full"], default="pre")
    parser.add_argument("--log_file", help="Training log file (for post mode)")
    parser.add_argument("--gpu_memory_gb", type=float, default=80.0)
    parser.add_argument("--json", action="store_true", help="Output as JSON (for agent consumption)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.mode == "pre":
        report = diagnose_pre(config, args.gpu_memory_gb)
    elif args.mode == "post":
        if not args.log_file:
            print("ERROR: --log_file required for post mode", file=sys.stderr)
            sys.exit(1)
        report = diagnose_post(config, args.log_file)
    else:  # full
        report = diagnose_full(config, args.gpu_memory_gb)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    sys.exit(0 if report["overall"] == "HEALTHY" else 1)


if __name__ == "__main__":
    main()
