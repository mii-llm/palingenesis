"""Prompts for RL: loading, the held-out split, and the sampler that draws training rows.

Rows are plain dicts; every column travels with the rollout to the reward functions and
the environment. The prompt is a chat (a list of messages) or a string (one user turn),
in `data.prompt_field` or the first of messages, prompt, question, problem.
"""

import hashlib
import heapq
import json
import logging
import math
import numbers
import random
import threading
from pathlib import Path
from typing import Any

PROMPT_FIELDS = ("messages", "prompt", "question", "problem")
ROW_LIMITS = ("max_turns", "max_completion_tokens", "max_new_tokens")  # a row's own limits, over the config's
TURN_FLOOR = 256  # generated tokens a turn should be able to get within its episode budget (check_row_limits)

logger = logging.getLogger(__name__)


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


def row_limit(row: dict[str, Any], key: str) -> int | None:
    """A row's own limit (max_turns, max_completion_tokens, max_new_tokens): a positive integer, or None when absent.
    A NaN is absent (the missing cell of a pandas-loaded column); 0, negatives, booleans, fractions and strings raise,
    instead of meaning the config's value (0 did) or being clamped or truncated."""
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    integral = isinstance(value, numbers.Integral) or (isinstance(value, numbers.Real) and float(value).is_integer())
    if isinstance(value, bool) or not integral or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return int(value)


def check_row_limits(rows: list[dict[str, Any]], source: str, budget: int = 0) -> None:
    """Validate the rows' own limits when they are loaded: a bad value fails fast, naming the source and the row,
    instead of turning every rollout of the row into an error. budget: the episode budget (generated tokens over all
    turns) of rows without their own max_completion_tokens; one warning covers the rows whose max_turns cannot each
    get TURN_FLOOR tokens of it (they would end on "length" long before their turn limit: set both together)."""
    short = []
    for i, row in enumerate(rows):
        try:
            limits = {key: row_limit(row, key) for key in ROW_LIMITS}
        except ValueError as e:
            raise ValueError(f"{source}, row {i} ({json.dumps(row, default=str)[:100]}): {e}") from None
        turns, episode = limits["max_turns"], limits["max_completion_tokens"] or budget
        if turns and episode and turns * TURN_FLOOR > episode:
            short.append((i, turns, episode))
    if short:
        i, turns, episode = short[0]
        logger.warning(
            "%s: %d row(s) set max_turns beyond their episode budget at %d tokens a turn (row %d: max_turns %d with "
            "%d tokens over the whole episode); give them max_completion_tokens too (or raise "
            "rollout.max_completion_tokens)",
            source,
            len(short),
            TURN_FLOOR,
            i,
            turns,
            episode,
        )


class PromptSampler:
    """Training rows in shuffled epochs. With `retire_above`, a row whose group's mean reward
    reaches it is not drawn again (it no longer teaches anything).

    The retry queue (off by default: `revisit_after_steps` 0) brings a row back on a schedule instead of waiting for
    the next epoch. Delays are counted in policy versions (training steps: the pipeline passes the weights' version to
    draw() and observe()), not in draws, whose number per step moves with refill, the zero-variance rate and the world
    size. A group where no rollout succeeded (solved share 0) brings its row back `revisit_after_steps` versions after
    it is observed, up to `revisit_max_failed` times over the run; a partly solved group (the rows that give gradient)
    after `revisit_partial_after_steps` (default half the delay, at least 1), up to `revisit_max_partial` times. A
    near-solved group (share >= `revisit_partial_max_share`, 7/8 by default: close to the retire line) and a solved
    one are not queued. Due revisits are served before the epoch order, capped at `revisit_share` of all draws so rows
    never tried keep flowing; ties among due entries break by a seeded random key, not by row index.

    With the queue on, a cooldown applies to both paths: a row served at version v is not served again (from the queue
    or the epoch order) until the version is past v, nor while a group of it is still generating, so one batch never
    holds two groups of the same prompt (two baselines for one prompt, from the same weights). Drawing a row from the
    epoch order cancels its queued revisit. Only when every live row is cooling down (fewer live rows than draws per
    step) is a row served anyway (`sampler/cooldown_overrides`). With the queue off, the draw order is the plain
    shuffled epochs.

    Per row the sampler keeps attempts, the history of (draw version, solved share) of every group, the version of
    its first success and whether it was solved only after a group that failed completely, so a run can report how
    many tasks went from "tried, not solved" to "solved" (`metrics()`, `state()`) and compare that offline with a
    no-learning null from the same history. draw() records in `last_revisit` whether the row came from the queue.

    draw / observe run on the pipeline's event-loop thread while the trainer thread may call state() for a
    checkpoint: a lock guards every mutation and the snapshot."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        seed: int = 0,
        retire_above: float = 0.0,
        revisit_after_steps: int = 0,
        revisit_max_failed: int = 1,
        revisit_partial_after_steps: int | None = None,
        revisit_share: float = 0.5,
        revisit_max_partial: int = 3,
        revisit_partial_max_share: float = 0.875,
    ):
        if not rows:
            raise ValueError("no training rows")
        self.revisit_after_steps = int(revisit_after_steps)
        self.revisit_partial_after_steps = (
            int(revisit_partial_after_steps)
            if revisit_partial_after_steps is not None
            else max(1, self.revisit_after_steps // 2)
        )
        self.revisit_max_failed, self.revisit_max_partial = int(revisit_max_failed), int(revisit_max_partial)
        self.revisit_share, self.revisit_partial_max_share = float(revisit_share), float(revisit_partial_max_share)
        if self.revisit_after_steps < 0:
            raise ValueError("revisit_after_steps must be >= 0 (0 = no retry queue)")
        if self.revisit_after_steps and self.revisit_partial_after_steps < 1:
            raise ValueError("revisit_partial_after_steps must be >= 1 (delays are counted in policy versions)")
        if self.revisit_max_failed < 0 or self.revisit_max_partial < 0:
            raise ValueError("revisit_max_failed and revisit_max_partial must be >= 0")
        if not 0.0 < self.revisit_share <= 1.0 or not 0.0 < self.revisit_partial_max_share <= 1.0:
            raise ValueError("revisit_share and revisit_partial_max_share must be in (0, 1]")
        self.rows = rows
        self.rng = random.Random(seed)
        # heap ties: a stream of its own, so the epoch shuffles stay as they were
        self.tie_rng = random.Random(f"revisit-{seed}")
        self.lock = threading.RLock()
        self.retire_above = retire_above
        self.retired: set[int] = set()
        self.order: list[int] = []
        self.epoch = 0
        self.version = 0  # the clock: the newest policy version passed to draw() / observe()
        self.draws = 0
        self.served_revisits = 0
        self.cooldown_overrides = 0
        self.last_revisit = False  # whether the last draw came from the queue
        self.pending: list[tuple[int, float, int]] = []  # (due version, tie key, row index), a heap
        self.queued: dict[int, tuple[int, float, int]] = {}  # row -> its live heap entry (others are skipped at pop)
        self.last_served: dict[int, int] = {}  # row -> version of its last draw
        self.in_flight: dict[int, int] = {}  # row -> groups drawn and not yet observed / released (queue on)
        self.attempts: dict[int, int] = {}
        self.history: dict[int, list[tuple[int, float]]] = {}  # row -> (draw version, solved share) per group
        self.revisits_failed: dict[int, int] = {}
        self.revisits_partial: dict[int, int] = {}
        self.first_solved: dict[int, int] = {}  # row index -> draw version of its first group with a success
        self.failed: set[int] = set()  # rows with at least one group where nothing succeeded
        self.solved_after_fail: set[int] = set()

    @property
    def queue_on(self) -> bool:
        return self.revisit_after_steps > 0

    def _clock(self, version: int | None) -> None:
        if version is not None:
            self.version = max(self.version, int(version))

    def _cooling(self, index: int) -> bool:
        return self.last_served.get(index, -1) >= self.version or self.in_flight.get(index, 0) > 0

    def draw(self, version: int | None = None) -> tuple[int, dict[str, Any]]:
        """The next row (index, row). version: the current policy version (the clock of the queue's delays and of
        the cooldown); None keeps the last one."""
        with self.lock:
            self._clock(version)
            self.draws += 1
            index = self._pop_revisit()
            self.last_revisit = index is not None
            if index is None:
                index = self._pop_epoch()
            else:
                self.served_revisits += 1
            self.last_served[index] = self.version
            if self.queue_on:
                self.in_flight[index] = self.in_flight.get(index, 0) + 1
                self.queued.pop(index, None)  # drawn from the epoch order: its queued revisit is served by this draw
            return index, self.rows[index]

    def _pop_revisit(self) -> int | None:
        """A due revisit, at most `revisit_share` of all draws, so rows never tried keep flowing."""
        found, deferred = None, []
        while (
            self.pending
            and self.pending[0][0] <= self.version
            and self.served_revisits < self.revisit_share * self.draws
        ):
            entry = heapq.heappop(self.pending)
            index = entry[2]
            if self.queued.get(index) != entry:  # replaced by a newer entry, or cancelled by an epoch draw
                continue
            if index in self.retired:
                del self.queued[index]
                continue
            if self._cooling(index):  # served at this version or still generating: due again once it is not
                deferred.append(entry)
                continue
            del self.queued[index]
            found = index
            break
        for entry in deferred:
            heapq.heappush(self.pending, entry)
        return found

    def _new_epoch(self) -> None:
        live = [i for i in range(len(self.rows)) if i not in self.retired]
        if not live:
            raise RuntimeError(
                "every training prompt has been retired (data.retire_above): the policy solves the whole dataset"
            )
        self.rng.shuffle(live)
        self.order, self.epoch = live, self.epoch + 1

    def _pop_epoch(self) -> int:
        renewed = False
        while True:
            if not self.order:
                self._new_epoch()
                renewed = True
            if not self.queue_on:  # the plain epochs (queue off: unchanged order)
                index = self.order.pop()
                if index not in self.retired:
                    return index
                continue
            # the cooldown: the next row of the epoch that is not cooling down (the others keep their place)
            for j in range(len(self.order) - 1, -1, -1):
                index = self.order[j]
                if index not in self.retired and not self._cooling(index):
                    del self.order[j]
                    return index
            if renewed:  # a fresh epoch and every live row is cooling down (fewer live rows than draws per step)
                self.order = [i for i in self.order if i not in self.retired]
                self.cooldown_overrides += 1
                return self.order.pop()
            # every row left in this epoch was served at this version (from the queue) or is generating: done
            self.order = []

    def release(self, index: int) -> None:
        """A drawn row's group finished without an observation (every rollout errored): it is no longer in flight."""
        with self.lock:
            self._release(index)

    def _release(self, index: int) -> None:
        n = self.in_flight.get(index, 0)
        if n > 1:
            self.in_flight[index] = n - 1
        else:
            self.in_flight.pop(index, None)

    def observe(self, index: int, mean_reward: float, solved: float | None = None, version: int | None = None) -> None:
        """A group's outcome: mean reward (for retire_above) and, for the retry queue and the counters, the share of
        its rollouts that succeeded (raw task reward >= data.success_threshold; None: not judged, e.g. a broken
        verifier). version: the current policy version (a revisit is due `delay` versions after it)."""
        with self.lock:
            self._clock(version)
            self._release(index)
            if self.retire_above and mean_reward >= self.retire_above:
                self.retired.add(index)
            if solved is None:
                return
            drawn = self.last_served.get(index, self.version)
            self.attempts[index] = self.attempts.get(index, 0) + 1
            self.history.setdefault(index, []).append((drawn, round(float(solved), 4)))
            if solved > 0:
                self.first_solved.setdefault(index, drawn)
                if index in self.failed:
                    self.solved_after_fail.add(index)
            else:
                self.failed.add(index)
            if not self.queue_on or index in self.retired:
                return
            failed = solved <= 0
            if not failed and solved >= self.revisit_partial_max_share - 1e-9:
                return  # solved, or near-solved (7/8): little left to learn, close to the retire line
            counts, cap = (
                (self.revisits_failed, self.revisit_max_failed)
                if failed
                else (self.revisits_partial, self.revisit_max_partial)
            )
            if counts.get(index, 0) >= cap:
                return
            counts[index] = counts.get(index, 0) + 1
            delay = self.revisit_after_steps if failed else self.revisit_partial_after_steps
            entry = (self.version + delay, self.tie_rng.random(), index)
            self.queued[index] = entry  # replaces an older entry of the row (that one is skipped at pop)
            heapq.heappush(self.pending, entry)

    def metrics(self) -> dict[str, float]:
        """Counters for the training log (cumulative, per rank: the trainer sums them over ranks)."""
        with self.lock:
            if not self.attempts:
                return {}
            out = {
                "sampler/rows_tried": float(len(self.attempts)),
                "sampler/rows_ever_solved": float(len(self.first_solved)),
                "sampler/rows_failed_once": float(len(self.failed)),
                "sampler/rows_solved_after_fail": float(len(self.solved_after_fail)),
            }
            if self.queue_on:
                out.update(
                    {
                        "sampler/revisits_pending": float(sum(i not in self.retired for i in self.queued)),
                        "sampler/revisits_served": float(self.served_revisits),
                        "sampler/cooldown_overrides": float(self.cooldown_overrides),
                    }
                )
            return out

    def state(self) -> dict:
        with self.lock:  # a consistent snapshot while the event loop may observe a finishing group
            return {
                "rng": self.rng.getstate(),
                "tie_rng": self.tie_rng.getstate(),
                "retired": sorted(self.retired),
                "order": list(self.order),
                "epoch": self.epoch,
                "version": self.version,
                "draws": self.draws,
                "served_revisits": self.served_revisits,
                "cooldown_overrides": self.cooldown_overrides,
                "pending": sorted(self.queued.values()),  # live entries only
                "last_served": {str(k): v for k, v in self.last_served.items()},
                "attempts": {str(k): v for k, v in self.attempts.items()},
                "history": {str(k): [list(h) for h in v] for k, v in self.history.items()},
                "revisits_failed": {str(k): v for k, v in self.revisits_failed.items()},
                "revisits_partial": {str(k): v for k, v in self.revisits_partial.items()},
                "first_solved": {str(k): v for k, v in self.first_solved.items()},
                "failed": sorted(self.failed),
                "solved_after_fail": sorted(self.solved_after_fail),
            }

    def load_state(self, state: dict) -> None:
        """A state from state(); older states (without the queue's keys, or with draw-counted delays) load too."""
        with self.lock:
            self.rng.setstate(state["rng"])
            if "tie_rng" in state:
                self.tie_rng.setstate(state["tie_rng"])
            self.retired, self.order, self.epoch = set(state["retired"]), list(state["order"]), state["epoch"]
            self.version = int(state.get("version", 0))
            self.draws = int(state.get("draws", 0))
            self.served_revisits = int(state.get("served_revisits", 0))
            self.cooldown_overrides = int(state.get("cooldown_overrides", 0))
            self.queued = {}
            for x in state.get("pending", []):
                # (due draw, row) entries of the draw-counted queue cannot be converted: they are due now
                entry = (
                    (int(x[0]), float(x[1]), int(x[2]))
                    if len(x) == 3
                    else (self.version, self.tie_rng.random(), int(x[1]))
                )
                self.queued[entry[2]] = entry
            self.pending = list(self.queued.values())
            heapq.heapify(self.pending)
            self.last_served = {int(k): v for k, v in state.get("last_served", {}).items()}
            self.in_flight = {}  # groups in flight at the save are not resumed
            self.attempts = {int(k): v for k, v in state.get("attempts", {}).items()}
            self.history = {int(k): [tuple(h) for h in v] for k, v in state.get("history", {}).items()}
            old = {int(k): v for k, v in state.get("revisits", {}).items()}  # one counter for both kinds
            self.revisits_failed = {int(k): v for k, v in state.get("revisits_failed", old).items()}
            self.revisits_partial = {int(k): v for k, v in state.get("revisits_partial", old).items()}
            self.first_solved = {int(k): v for k, v in state.get("first_solved", {}).items()}
            self.failed = set(state.get("failed", []))
            self.solved_after_fail = set(state.get("solved_after_fail", []))
