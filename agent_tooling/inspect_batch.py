#!/usr/bin/env python3
"""Inspect a processed batch — shows tokenization, masking, and label alignment.

Run this to verify that:
  1. Chat template is applied correctly
  2. Only assistant tokens have labels (non-IGNORE_INDEX)
  3. Sequence lengths are reasonable
  4. Padding is correct

Usage:
    python -m agent_tooling.inspect_batch --config configs/llama3_8b.yaml --num_samples 3

Output: Colored text showing which tokens are trained on (green = loss computed,
        gray = masked/ignored), plus statistics.
"""

import sys

import agent_tooling._path_setup  # noqa: F401 — adds src/ to sys.path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from palingenesis.config import Config
from palingenesis.data import ChatDataset, IGNORE_INDEX


# ANSI colors
GREEN = "\033[92m"
GRAY = "\033[90m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def inspect(config: Config, num_samples: int = 3):
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

    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Batch Inspector — {config.model.name_or_path}{RESET}")
    print(f"Dataset: {config.data.dataset} | Max seq: {config.data.max_seq_length}")
    print(f"{'=' * 80}\n")

    count = 0
    total_tokens = 0
    total_trained = 0

    for sample in chat_ds:
        if count >= num_samples:
            break

        input_ids = sample["input_ids"]
        labels = sample["labels"]
        seq_len = input_ids.size(0)
        trained_mask = labels != IGNORE_INDEX
        num_trained = trained_mask.sum().item()

        total_tokens += seq_len
        total_trained += num_trained

        print(
            f"{BOLD}Sample {count + 1}{RESET} — {seq_len} tokens, {num_trained} trained ({100*num_trained/seq_len:.1f}%)"
        )
        print(f"{'-' * 60}")

        # Decode token by token, color by whether it's trained
        tokens = input_ids.tolist()
        label_list = labels.tolist()

        # Show in chunks to avoid massive output
        chunk_size = 50
        for i in range(0, min(seq_len, 500), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_labels = label_list[i : i + chunk_size]

            line = ""
            for tok, lab in zip(chunk_tokens, chunk_labels):
                decoded = tokenizer.decode([tok], skip_special_tokens=False)
                decoded = decoded.replace("\n", "\\n").replace("\t", "\\t")
                if lab == IGNORE_INDEX:
                    line += f"{GRAY}{decoded}{RESET}"
                else:
                    line += f"{GREEN}{decoded}{RESET}"

            print(f"  [{i:4d}] {line}")

        if seq_len > 500:
            print(f"  ... ({seq_len - 500} more tokens)")

        # Role boundary analysis
        print(f"\n  {YELLOW}Label transitions:{RESET}")
        in_trained = False
        transitions = []
        for i, lab in enumerate(label_list):
            is_trained = lab != IGNORE_INDEX
            if is_trained != in_trained:
                transitions.append((i, "→TRAIN" if is_trained else "→MASK"))
                in_trained = is_trained

        for pos, kind in transitions[:20]:
            context = tokenizer.decode(tokens[max(0, pos - 2) : pos + 3], skip_special_tokens=False)
            context = context.replace("\n", "\\n")[:60]
            print(f"    pos {pos:5d}: {kind}  context: ...{context}...")

        print()
        count += 1

    # Summary
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Summary across {count} samples:{RESET}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Trained tokens: {total_trained} ({100*total_trained/max(total_tokens,1):.1f}%)")
    print(
        f"  Masked tokens: {total_tokens - total_trained} ({100*(total_tokens-total_trained)/max(total_tokens,1):.1f}%)"
    )
    print(f"  Avg sequence length: {total_tokens/max(count,1):.0f}")

    # Sanity checks
    issues = []
    if total_trained == 0:
        issues.append(f"{RED}CRITICAL: No tokens are being trained on! Check chat template masking.{RESET}")
    if total_trained == total_tokens:
        issues.append(f"{YELLOW}WARNING: ALL tokens are trained on — assistant masking may not be working.{RESET}")
    if total_trained / max(total_tokens, 1) < 0.05:
        issues.append(
            f"{YELLOW}WARNING: Only {100*total_trained/max(total_tokens,1):.1f}% of tokens trained — very low ratio.{RESET}"
        )

    if issues:
        print(f"\n{BOLD}Issues:{RESET}")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n  {GREEN}✓ Masking looks healthy{RESET}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect processed batches")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to inspect")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    inspect(config, args.num_samples)


if __name__ == "__main__":
    main()
