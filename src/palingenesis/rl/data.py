"""Prompts for RL: loading, the held-out split, and the sampler that draws training rows.

Rows are plain dicts; every column travels with the rollout to the reward functions and
the environment. The prompt is a chat (a list of messages) or a string (one user turn),
in `data.prompt_field` or the first of messages, prompt, question, problem.
"""

import hashlib
import json
import heapq
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
    reaches it is not drawn again (it no longer teaches anything).

    With `revisit_after` > 0 (a retry queue, off by default), a row is brought back on a schedule instead of waiting for
    the next epoch: a group where no rollout succeeded (solved share 0) returns after `revisit_after` draws, up to
    `revisit_max` times, so a task the policy cannot solve yet is tried again once the policy has moved; a partly solved
    group (0 < share < 1: the rows that give gradient) returns after `revisit_partial_after` draws (default half the
    delay), also up to `revisit_max` times. Due revisits are served before the epoch order, capped at `revisit_share` of all
    draws. Per row the sampler keeps
    attempts, the draw of its first success and whether it was solved only after a group that failed completely, so a
    run can report how many tasks went from "tried, not solved" to "solved" (`metrics()`, `state()`)."""

    def __init__(self, rows: list[dict[str, Any]], seed: int = 0, retire_above: float = 0.0, revisit_after: int = 0,
                 revisit_max: int = 3, revisit_partial_after: int | None = None, revisit_share: float = 1.0):
        if not rows:
            raise ValueError("no training rows")
        self.rows = rows
        self.rng = random.Random(seed)
        self.retire_above = retire_above
        self.retired: set[int] = set()
        self.order: list[int] = []
        self.epoch = 0
        self.revisit_after = int(revisit_after)
        self.revisit_max = int(revisit_max)
        self.revisit_partial_after = (int(revisit_partial_after) if revisit_partial_after is not None
                                      else max(1, self.revisit_after // 2))
        self.revisit_share = float(revisit_share)
        self.served_revisits = 0
        self.draws = 0
        self.pending: list[tuple[int, int]] = []  # (due draw, row index), a heap
        self.attempts: dict[int, int] = {}
        self.revisits: dict[int, int] = {}
        self.first_solved: dict[int, int] = {}  # row index -> draw count when a group first had a success
        self.failed: set[int] = set()  # rows with at least one group where nothing succeeded
        self.solved_after_fail: set[int] = set()

    def draw(self) -> tuple[int, dict[str, Any]]:
        self.draws += 1
        # due revisits first, but at most `revisit_share` of all draws, so rows never tried keep flowing
        while (self.pending and self.pending[0][0] <= self.draws
               and self.served_revisits < self.revisit_share * self.draws):
            _, index = heapq.heappop(self.pending)
            if index not in self.retired:
                self.served_revisits += 1
                return index, self.rows[index]
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

    def observe(self, index: int, mean_reward: float, solved: float | None = None) -> None:
        """A group's outcome: mean reward (for retire_above) and, for the retry queue, the share of its rollouts that
        succeeded (reward >= data.success_threshold)."""
        if self.retire_above and mean_reward >= self.retire_above:
            self.retired.add(index)
        if solved is None:
            return
        self.attempts[index] = self.attempts.get(index, 0) + 1
        if solved > 0:
            self.first_solved.setdefault(index, self.draws)
            if index in self.failed:
                self.solved_after_fail.add(index)
        else:
            self.failed.add(index)
        if self.revisit_after <= 0 or index in self.retired or solved >= 1:
            return
        if self.revisits.get(index, 0) >= self.revisit_max:
            return
        self.revisits[index] = self.revisits.get(index, 0) + 1
        delay = self.revisit_after if solved <= 0 else self.revisit_partial_after
        heapq.heappush(self.pending, (self.draws + delay, index))

    def metrics(self) -> dict[str, float]:
        """Counters for the training log (cumulative)."""
        if not self.attempts:
            return {}
        return {"sampler/rows_tried": float(len(self.attempts)),
                "sampler/rows_ever_solved": float(len(self.first_solved)),
                "sampler/rows_failed_once": float(len(self.failed)),
                "sampler/rows_solved_after_fail": float(len(self.solved_after_fail)),
                "sampler/revisits_pending": float(len(self.pending))}

    def state(self) -> dict:
        return {
            "rng": self.rng.getstate(),
            "retired": sorted(self.retired),
            "order": self.order,
            "epoch": self.epoch,
            "draws": self.draws,
            "served_revisits": self.served_revisits,
            "pending": sorted(self.pending),
            "attempts": {str(k): v for k, v in self.attempts.items()},
            "revisits": {str(k): v for k, v in self.revisits.items()},
            "first_solved": {str(k): v for k, v in self.first_solved.items()},
            "failed": sorted(self.failed),
            "solved_after_fail": sorted(self.solved_after_fail),
        }

    def load_state(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        self.retired, self.order, self.epoch = (
            set(state["retired"]),
            list(state["order"]),
            state["epoch"],
        )
        self.draws = int(state.get("draws", 0))
        self.served_revisits = int(state.get("served_revisits", 0))
        self.pending = [tuple(x) for x in state.get("pending", [])]
        heapq.heapify(self.pending)
        self.attempts = {int(k): v for k, v in state.get("attempts", {}).items()}
        self.revisits = {int(k): v for k, v in state.get("revisits", {}).items()}
        self.first_solved = {int(k): v for k, v in state.get("first_solved", {}).items()}
        self.failed = set(state.get("failed", []))
        self.solved_after_fail = set(state.get("solved_after_fail", []))
