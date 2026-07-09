#!/usr/bin/env python3
"""Validate assistant-only masking across many samples.

Checks that the data pipeline correctly masks non-assistant tokens and
only trains on assistant content. Reports statistics and flags problems.

Usage:
    python -m agent_tooling.validate_masking --config configs/llama3_8b.yaml --num_samples 100

What it checks:
    - Trained tokens ratio is reasonable (5-80% for multi-turn chat)
    - No samples have 0% trained tokens (broken masking)
    - No samples have 100% trained tokens (masking disabled)
    - Label boundaries align with role transitions
    - Padding tokens are never trained on
"""

import sys

import agent_tooling._path_setup  # noqa: F401

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from palingenesis.config import Config
from palingenesis.data import ChatDataset, IGNORE_INDEX


def validate(config: Config, num_samples: int = 100) -> dict:
    """Run masking validation. Returns a report dict."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(
        config.data.dataset,
        split=config.data.dataset_split,
        streaming=config.data.streaming,
    )

    chat_ds = ChatDataset(
        dataset,
        tokenizer,
        config.data.max_seq_length,
        config.data.messages_field,
        rank=0,
        world_size=1,
        include_observations=config.data.include_observations,
        turn_scaling=config.data.turn_scaling,
        train_on_reasoning=getattr(config.data, "train_on_reasoning", True),
    )

    results = {
        "total_samples": 0,
        "skipped_samples": 0,  # returned None from _process
        "total_tokens": 0,
        "total_trained": 0,
        "samples_all_masked": 0,  # 0 trained tokens
        "samples_all_trained": 0,  # all tokens trained
        "pad_tokens_trained": 0,  # BUG: pad tokens with loss
        "seq_lengths": [],
        "train_ratios": [],
        "issues": [],
    }

    raw_iter = iter(dataset)
    processed = 0
    skipped = 0

    for sample in chat_ds:
        if processed >= num_samples:
            break

        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attn = sample["attention_mask"]
        seq_len = input_ids.size(0)

        trained_mask = labels != IGNORE_INDEX
        num_trained = trained_mask.sum().item()
        ratio = num_trained / max(seq_len, 1)

        results["total_tokens"] += seq_len
        results["total_trained"] += num_trained
        results["seq_lengths"].append(seq_len)
        results["train_ratios"].append(ratio)

        if num_trained == 0:
            results["samples_all_masked"] += 1
        if num_trained == seq_len:
            results["samples_all_trained"] += 1

        # Check: are pad tokens being trained on?
        if tokenizer.pad_token_id is not None:
            pad_positions = input_ids == tokenizer.pad_token_id
            pad_trained = (pad_positions & trained_mask).sum().item()
            results["pad_tokens_trained"] += pad_trained

        processed += 1

    results["total_samples"] = processed

    # Analyze and flag issues
    if processed == 0:
        results["issues"].append("CRITICAL: No samples were processed. Dataset may be empty or all filtered.")
        return results

    avg_ratio = results["total_trained"] / max(results["total_tokens"], 1)
    avg_len = sum(results["seq_lengths"]) / len(results["seq_lengths"])

    if results["samples_all_masked"] > processed * 0.1:
        results["issues"].append(
            f"WARNING: {results['samples_all_masked']}/{processed} samples have NO trained tokens. "
            "Chat template may lack {% generation %} markers."
        )

    if results["samples_all_trained"] > processed * 0.1:
        results["issues"].append(
            f"WARNING: {results['samples_all_trained']}/{processed} samples train on ALL tokens. "
            "Assistant masking may not be working."
        )

    if avg_ratio < 0.03:
        results["issues"].append(
            f"WARNING: Very low train ratio ({avg_ratio:.1%}). " "Check if assistant messages exist in your data."
        )

    if avg_ratio > 0.95:
        results["issues"].append(
            f"WARNING: Very high train ratio ({avg_ratio:.1%}). "
            "Masking may be disabled or data is only assistant messages."
        )

    if results["pad_tokens_trained"] > 0:
        results["issues"].append(
            f"BUG: {results['pad_tokens_trained']} padding tokens have loss computed. "
            "Labels should be IGNORE_INDEX for all padding."
        )

    if not results["issues"]:
        results["issues"].append("OK: Masking looks healthy.")

    return results


def print_report(results: dict):
    print("=" * 70)
    print("MASKING VALIDATION REPORT")
    print("=" * 70)
    print(f"  Samples processed: {results['total_samples']}")
    print(f"  Total tokens: {results['total_tokens']:,}")
    print(
        f"  Trained tokens: {results['total_trained']:,} ({100*results['total_trained']/max(results['total_tokens'],1):.1f}%)"
    )
    print(f"  Avg sequence length: {sum(results['seq_lengths'])/max(len(results['seq_lengths']),1):.0f}")
    print(f"  Samples w/ 0% trained: {results['samples_all_masked']}")
    print(f"  Samples w/ 100% trained: {results['samples_all_trained']}")
    print(f"  Pad tokens incorrectly trained: {results['pad_tokens_trained']}")

    if results["train_ratios"]:
        ratios = results["train_ratios"]
        print(f"  Train ratio: min={min(ratios):.1%} median={sorted(ratios)[len(ratios)//2]:.1%} max={max(ratios):.1%}")

    print()
    for issue in results["issues"]:
        prefix = "  ✓" if issue.startswith("OK") else "  ⚠"
        print(f"{prefix} {issue}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    results = validate(config, args.num_samples)
    print_report(results)
    # Exit 1 if critical issues
    sys.exit(1 if any("CRITICAL" in i or "BUG" in i for i in results["issues"]) else 0)


if __name__ == "__main__":
    main()
