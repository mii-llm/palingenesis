"""Prompt sources — the task-specific data layer behind the task-agnostic OPD engine.

The trainer knows how to sample on-policy, score with the teacher, and take
reverse-KL steps; everything task-shaped lives behind the PromptSource
protocol:

  - what conversation to roll out next (templates, shot mixing, length budget)
  - how to evaluate the student on held-out prompts
  - which per-batch quality stats to log

Built-ins (selected by ``data.format`` in the config):

  - "mcqa":     multiple-choice pools — pool-row JSONL, shot regimes, letter
                accuracy as the dev metric (the original OPD data layer)
  - "messages": generic chat — JSONL of {"messages": [...]}, held-out mean
                reverse KL as the dev metric (you can't auto-grade free-form
                answers, but distance-to-teacher on unseen prompts is exactly
                the trained quantity, measured out of sample)

A custom source is any object with the same three methods; pass it as
``OPDTrainer(config, source=...)``.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Protocol

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.formatting import PromptRenderer, build_messages, extract_letter, load_reference_shots
from palingenesis.opd.pool import load_pool, question_hash, split_pool

logger = logging.getLogger(__name__)


class Engine(Protocol):
    """The trainer services a source may use during evaluate()."""

    def greedy_generate(self, messages_list: list[list[dict[str, str]]], max_new_tokens: int) -> list[str]:
        """Greedy-decode one completion per conversation, cleaned and decoded."""

    def dev_kl(self, messages_list: list[list[dict[str, str]]], max_new_tokens: int) -> dict[str, float]:
        """Sample on-policy and teacher-score without grad; mean kl/token and length."""


class PromptSource(Protocol):
    def sample(self) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
        """One training rollout: (messages, max_new_tokens, meta)."""

    def evaluate(self, engine: Engine) -> dict[str, float]:
        """Held-out metrics, e.g. {"dev_acc": 0.41} or {"dev_kl": 0.83}."""

    def batch_stats(self, rollouts: list[tuple[dict[str, Any], str]]) -> dict[str, float]:
        """Per-batch stats from (meta, decoded_completion) pairs. May be {}."""


def build_source(config: OPDConfig, rng: random.Random) -> "McqaPoolSource | ChatMessagesSource":
    if config.data.format == "mcqa":
        return McqaPoolSource(config, rng)
    if config.data.format == "messages":
        return ChatMessagesSource(config, rng)
    raise ValueError(f"unknown data.format: {config.data.format!r} (expected 'mcqa' or 'messages')")


class McqaPoolSource:
    """Multiple-choice pools: shot regimes for training, letter accuracy for dev."""

    def __init__(self, config: OPDConfig, rng: random.Random):
        self.config = config
        logger.info("Loading MCQA pool from %s", config.data.prompts_path)
        pool = load_pool(config.data.prompts_path)
        self.train_rows, self.dev_rows = split_pool(pool, config.data.dev_size, config.train.seed)
        logger.info("Pool: %d train / %d dev", len(self.train_rows), len(self.dev_rows))
        self.reference_shots = load_reference_shots(config.data.shots_path) if config.data.shots_path else []
        self.renderer = PromptRenderer(
            self.train_rows,
            self.reference_shots,
            p_reference_shots=config.data.p_reference_shots,
            p_pool_shots=config.data.p_pool_shots,
            pool_shots_max_k=config.data.pool_shots_max_k,
            cot_fraction=config.sampling.cot_fraction,
            system_message=config.data.system_message or None,
            rng=rng,
        )

    def sample(self):
        messages, row, fast = self.renderer.sample()
        mnt = self.config.sampling.max_new_tokens if fast else self.config.sampling.cot_max_new_tokens
        return messages, mnt, {"row": row, "fast": fast}

    def evaluate(self, engine: Engine) -> dict[str, float]:
        """Greedy few-shot fast-mode accuracy on the held-out dev slice."""
        rows = self.dev_rows[: self.config.train.eval_dev_samples]
        prompts = [
            build_messages(r, few_shots=self.reference_shots, fast=True,
                           system_message=self.config.data.system_message or None)
            for r in rows
        ]
        texts = engine.greedy_generate(prompts, max_new_tokens=8)
        correct = sum(1 for r, text in zip(rows, texts) if extract_letter(text) == r["answer"])
        return {"dev_acc": correct / max(1, len(rows))}

    def batch_stats(self, rollouts):
        if not rollouts:
            return {}
        ok = sum(
            1 for meta, text in rollouts
            if (letter := extract_letter(text)) and letter in {le for le, _ in meta["row"]["options"]}
        )
        return {"format_ok": ok / len(rollouts)}


class ChatMessagesSource:
    """Generic chat prompts: JSONL of {"messages": [...]}, ending with a user turn.

    Rows whose last message is not a user turn are skipped (the student must
    have something to complete). The dev metric is the on-policy reverse KL to
    the teacher on held-out prompts — lower = closer to the teacher where the
    student actually goes.
    """

    def __init__(self, config: OPDConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        logger.info("Loading chat prompts from %s", config.data.prompts_path)
        rows, skipped = [], 0
        with open(config.data.prompts_path) as f:
            for line in f:
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                if messages and messages[-1]["role"] == "user":
                    rows.append(messages)
                else:
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d rows whose last message is not a user turn", skipped)
        if not rows:
            raise ValueError(f"No usable rows in {config.data.prompts_path}")

        # deterministic dev split by content hash (same idea as split_pool)
        by_hash = {question_hash(json.dumps(m, ensure_ascii=False)): m for m in rows}
        dev_hashes = sorted(by_hash)[: config.data.dev_size]
        dev_set = set(dev_hashes)
        self.dev_rows = [by_hash[h] for h in dev_hashes]
        self.train_rows = [m for m in rows
                           if question_hash(json.dumps(m, ensure_ascii=False)) not in dev_set]
        logger.info("Chat prompts: %d train / %d dev", len(self.train_rows), len(self.dev_rows))

    def sample(self):
        return self.rng.choice(self.train_rows), self.config.sampling.max_new_tokens, {}

    def evaluate(self, engine: Engine) -> dict[str, float]:
        rows = self.dev_rows[: self.config.train.eval_dev_samples]
        return engine.dev_kl(rows, self.config.sampling.max_new_tokens)

    def batch_stats(self, rollouts):
        return {}
