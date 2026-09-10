"""Prompt-pool schema, dedup, and loading for OPD.

Pool row schema (prompts.jsonl):
  {"question": str, "options": [["A", "text"], ...], "answer": "A",
   "category": str, "source": str}

Converting a raw dataset into this schema is an adapter, and adapters are
experiment policy — they live next to the experiment, not here (each is a
small function: map fields, letter the options, run `valid_row`, dedup with
`load_benchmark_hashes`). This module ships the mechanism only: row
validation, benchmark dedup, and the deterministic train/dev split.

Dedup matters more than usual here: training pools are often drawn from the
same corpora a target benchmark was curated from, so every row must be checked
against the benchmark's questions by normalized hash before it enters the pool
— otherwise you are training on the test set.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable


def norm_text(s: str) -> str:
    """Aggressive normalization for near-duplicate detection."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def question_hash(question: str) -> str:
    return hashlib.md5(norm_text(question).encode()).hexdigest()


def load_benchmark_hashes(path: str, field: str = "question") -> set[str]:
    """Hashes of a benchmark's questions (JSONL), for pool dedup."""
    hashes = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                hashes.add(question_hash(json.loads(line)[field]))
    return hashes


# ---------------------------------------------------------------------------
# Row validation (for adapter authors)
# ---------------------------------------------------------------------------

def valid_row(question: str, options: list[tuple[str, str]], answer: str) -> bool:
    if not question or not (2 <= len(options) <= 10):
        return False
    if answer not in {letter for letter, _ in options}:
        return False
    if any(not text.strip() for _, text in options):
        return False
    if len(question) > 1500 or sum(len(t) for _, t in options) > 2000:
        return False
    return True


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def write_pool(rows: Iterable[dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_pool(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["options"] = [tuple(o) for o in row["options"]]
                rows.append(row)
    return rows


def split_pool(rows: list[dict[str, Any]], dev_size: int = 500, seed: int = 0):
    """Deterministic train/dev split by question hash (stable across runs and input order).

    Dev rows are unique by question hash, so a pool with duplicated rows
    (upweighting by repetition is a legitimate reweighting strategy) can
    neither fill dev with copies of one question nor leak a dev question
    into train through its duplicates.
    """
    by_hash: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_hash.setdefault(question_hash(r["question"]), r)
    dev_hashes = sorted(by_hash)[:dev_size]
    dev = [by_hash[h] for h in dev_hashes]
    train = [r for r in rows if question_hash(r["question"]) not in set(dev_hashes)]
    return train, dev
