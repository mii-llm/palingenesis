"""Model-free data cleaning for `pgs prepare`.

Everything here needs the tokenizer at most, never a forward pass, and is deterministic
for a seed: rows the trainer would drop or cut, exact duplicates, a cap on rows that share
a prompt, caps per group (source), a token budget, and a held-out split that keeps a prompt
on one side. Every dropped row is recorded with its reason (CleanReport).
"""

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

ROW_INDEX = "_row"  # position of the row in the loaded data, kept in the output
LENGTH = "_score_length"  # tokens of the row as training sees it (after truncation)
TRAINED_TOKENS = "_score_response_token_count"  # tokens that get loss

DROPPED_BASENAME = "dropped.jsonl"


@dataclass
class CleanReport:
    """What cleaning removed and why. `dropped` lists every removed row: its index in the
    loaded data, the reason, and for duplicates the row that was kept."""

    loaded: int = 0
    truncated: int = 0  # kept, but cut to fit max_seq_length
    reasons: Counter = field(default_factory=Counter)
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def drop(self, sample: dict, reason: str, **detail) -> None:
        self.reasons[reason] += 1
        self.dropped.append({"row": sample.get(ROW_INDEX), "reason": reason, **detail})

    def summary(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "dropped": dict(sorted(self.reasons.items())),
            "truncated_to_max_seq_length": self.truncated,
        }

    def write(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / DROPPED_BASENAME
        with path.open("w") as f:
            for row in self.dropped:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path


def _messages(sample: dict, messages_field: str) -> list:
    from palingenesis.validate_data import normalize_messages

    normalized = normalize_messages(sample, messages_field)
    if normalized:
        return normalized
    raw = sample.get(messages_field)
    return raw if isinstance(raw, list) else []


def _digest(obj: Any) -> str:
    # whitespace runs are collapsed: re-exports of one conversation differ in them
    text = " ".join(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).split())
    return hashlib.sha1(text.encode()).hexdigest()


def conversation_key(sample: dict, messages_field: str = "messages", tools_field: str = "tools") -> str:
    """Identity of a training example: its messages and tool definitions (the tools are
    rendered into the prompt, so the same messages under other tools are another example)."""
    return _digest([_messages(sample, messages_field), sample.get(tools_field) or None])


def prompt_key(sample: dict, messages_field: str = "messages") -> str:
    """Identity of the task: everything up to and including the first user turn. A row
    without a user turn is its own task."""
    messages = _messages(sample, messages_field)
    head = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        head.append((m.get("role"), m.get("content")))
        if m.get("role") == "user":
            return _digest(head)
    return conversation_key(sample, messages_field)


def tokenize_rows(samples: list[dict], chat, report: CleanReport, keep_tokens: bool = False):
    """Run every row through the trainer's own ChatDataset (template, tools, truncation at
    turn boundaries, masking) and drop the ones training would drop.

    Kept rows get LENGTH and TRAINED_TOKENS. Returns (kept rows, entries): entries are
    (position in kept, input_ids int32, trained mask) per kept row when keep_tokens, for
    scoring; else empty.
    """
    import torch

    from palingenesis.data import IGNORE_INDEX

    kept, entries = [], []
    for i, sample in enumerate(samples):
        before = chat.stats.copy()
        try:
            processed = chat._process(sample)
        except Exception as exc:
            logger.debug(f"row {i} cannot be rendered: {exc!r}")
            report.drop(sample, "unrenderable", error=repr(exc)[:200])
            continue
        if processed is None or processed["input_ids"].numel() < 2:
            too_long = chat.stats["dropped_too_long"] > before["dropped_too_long"]
            report.drop(sample, "no_trained_turn_fits" if too_long else "nothing_to_train_on")
            continue
        if chat.stats["truncated"] > before["truncated"]:
            report.truncated += 1
        trained = processed["labels"] != IGNORE_INDEX
        sample[LENGTH] = int(processed["input_ids"].numel())
        sample[TRAINED_TOKENS] = int(trained.sum())
        if keep_tokens:
            entries.append((len(kept), processed["input_ids"].to(torch.int32), trained))
        kept.append(sample)
        if i and i % 5000 == 0:
            logger.info(f"  Tokenized {i}/{len(samples)}")
    return kept, entries


def drop_exact_duplicates(
    samples: list[dict], report: CleanReport, messages_field: str = "messages", tools_field: str = "tools"
) -> list[dict]:
    """Keep the first row of every set with the same conversation_key."""
    first: dict[str, Any] = {}
    kept = []
    for s in samples:
        key = conversation_key(s, messages_field, tools_field)
        if key in first:
            report.drop(s, "duplicate", duplicate_of=first[key])
        else:
            first[key] = s.get(ROW_INDEX)
            kept.append(s)
    return kept


def cap_groups(
    samples: list[dict],
    key: Callable[[dict], str],
    report: CleanReport,
    reason: str,
    max_rows: int = 0,
    max_tokens: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Keep at most max_rows rows and max_tokens tokens (LENGTH) per group; 0 = no cap.

    The rows kept in a group are a seeded random draw, never the first in file order
    (files are often sorted by time or outcome); the output keeps the input order. A group
    always keeps at least one row.
    """
    if max_rows <= 0 and max_tokens <= 0:
        return samples
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(key(s), []).append(i)
    rng = random.Random(seed)
    keep: set[int] = set()
    for name in sorted(groups):
        order = groups[name]
        rng.shuffle(order)
        tokens = 0
        for n, i in enumerate(order):
            length = samples[i].get(LENGTH, 0)
            if n and ((max_rows > 0 and n >= max_rows) or (max_tokens > 0 and tokens + length > max_tokens)):
                break
            keep.add(i)
            tokens += length
    kept = []
    for i, s in enumerate(samples):
        if i in keep:
            kept.append(s)
        else:
            report.drop(s, reason, group=key(s))
    return kept


def take_token_budget(samples: list[dict], token_budget: int, seed: int = 42) -> list[dict]:
    """A seeded random subset whose tokens (LENGTH) add up to at most token_budget."""
    if token_budget <= 0:
        return samples
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    keep, tokens = set(), 0
    for i in order:
        length = samples[i].get(LENGTH, 0)
        if tokens + length > token_budget:
            continue  # a shorter row further on may still fit
        keep.add(i)
        tokens += length
    return [s for i, s in enumerate(samples) if i in keep]


def split_holdout(
    samples: list[dict],
    n_holdout: int,
    seed: int = 42,
    messages_field: str = "messages",
    key: Callable[[dict], str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Reserve about n_holdout rows as a held-out set: (train pool, holdout).

    Rows that share a prompt (prompt_key) go to the same side, so the held-out set never
    holds a task the model trained on under another answer or trajectory. Whole prompt
    groups are drawn at random until n_holdout rows are reached; a group that would
    overshoot is skipped, so the holdout can be smaller than asked when groups are large.
    key: the prompt identity of a row (default prompt_key), for callers that cached it.
    """
    if n_holdout <= 0 or n_holdout >= len(samples):
        return samples, []
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(key(s) if key else prompt_key(s, messages_field), []).append(i)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    held: set[int] = set()
    for name in names:
        if len(held) + len(groups[name]) <= n_holdout:
            held.update(groups[name])
        if len(held) == n_holdout:
            break
    train = [s for i, s in enumerate(samples) if i not in held]
    holdout = [s for i, s in enumerate(samples) if i in held]
    return train, holdout


def token_stats(samples: list[dict]) -> dict[str, Any]:
    """Token totals and length percentiles of rows annotated by tokenize_rows."""
    lengths = sorted(s[LENGTH] for s in samples if LENGTH in s)
    if not lengths:
        return {}
    trained = sum(s.get(TRAINED_TOKENS, 0) for s in samples)

    def pct(p: float) -> int:
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    return {
        "total": sum(lengths),
        "trained": trained,
        "trained_share": round(trained / max(sum(lengths), 1), 4),
        "length_p50": pct(0.5),
        "length_p95": pct(0.95),
        "length_max": lengths[-1],
    }
