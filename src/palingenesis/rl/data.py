"""Prompts for RL: loading, the held-out split, and the sampler that draws training rows.

Rows are plain dicts; every column travels with the rollout to the reward functions and
the environment. The prompt is a chat (a list of messages) or a string (one user turn),
in `data.prompt_field` or the first of messages, prompt, question, problem.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

PROMPT_FIELDS = ("messages", "prompt", "question", "problem")


def load_rows(source: str, split: str = "train") -> list[dict[str, Any]]:
    """A JSONL / JSON / parquet file, or a Hugging Face dataset (id, split)."""
    path = Path(source)
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data[split]
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    if path.exists():
        raise ValueError(f"{source}: unsupported file type (use .jsonl, .json or .parquet)")
    from datasets import load_dataset

    return [dict(r) for r in load_dataset(source, split=split)]


def prompt_messages(row: dict[str, Any], field: str = "", system_prompt: str = "") -> list[dict[str, Any]]:
    """The conversation a row's rollout starts from."""
    name = field or next((f for f in PROMPT_FIELDS if row.get(f) is not None), None)
    if name is None or row.get(name) is None:
        raise KeyError(f"row has no prompt: set data.prompt_field (columns: {sorted(row)})")
    value = row[name]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):  # a chat stored as a JSON string
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    messages = [{"role": "user", "content": value}] if isinstance(value, str) else [dict(m) for m in value]
    if not messages:
        raise ValueError("empty prompt")
    if system_prompt and messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def _key(row: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()


def split_rows(rows: list[dict], eval_size: int) -> tuple[list[dict], list[dict]]:
    """(train, eval): `eval_size` rows chosen by content hash, so the split is the same on
    every run and survives reordering of the file."""
    if eval_size <= 0:
        return rows, []
    ranked = sorted(range(len(rows)), key=lambda i: _key(rows[i]))
    held = set(ranked[:eval_size])
    return [r for i, r in enumerate(rows) if i not in held], [rows[i] for i in sorted(held)]


class PromptSampler:
    """Training rows in shuffled epochs. With `retire_above`, a row whose group's mean reward
    reaches it is not drawn again (it no longer teaches anything)."""

    def __init__(self, rows: list[dict[str, Any]], seed: int = 0, retire_above: float = 0.0):
        if not rows:
            raise ValueError("no training rows")
        self.rows = rows
        self.rng = random.Random(seed)
        self.retire_above = retire_above
        self.retired: set[int] = set()
        self.order: list[int] = []
        self.epoch = 0

    def draw(self) -> tuple[int, dict[str, Any]]:
        while True:
            if not self.order:
                live = [i for i in range(len(self.rows)) if i not in self.retired]
                if not live:
                    raise RuntimeError(
                        "every training prompt has been retired (data.retire_above): "
                        "the policy solves the whole dataset"
                    )
                self.rng.shuffle(live)
                self.order, self.epoch = live, self.epoch + 1
            index = self.order.pop()
            if index not in self.retired:
                return index, self.rows[index]

    def observe(self, index: int, mean_reward: float) -> None:
        if self.retire_above and mean_reward >= self.retire_above:
            self.retired.add(index)

    def state(self) -> dict:
        return {
            "rng": self.rng.getstate(),
            "retired": sorted(self.retired),
            "order": self.order,
            "epoch": self.epoch,
        }

    def load_state(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        self.retired, self.order, self.epoch = (
            set(state["retired"]),
            list(state["order"]),
            state["epoch"],
        )
