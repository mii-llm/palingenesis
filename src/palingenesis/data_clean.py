"""Model-free data cleaning for `pgs prepare`.

Everything here needs the tokenizer at most, never a forward pass, and is deterministic
for a seed: rows the trainer would drop or cut, exact duplicates, a cap on rows that share
a prompt, caps per group (source), a token budget, and a held-out split that keeps a prompt
on one side. Every dropped row is recorded with its reason (CleanReport).
"""

import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # optional: dedupe keys are the same without it, only slower
    import orjson
except ImportError:
    orjson = None

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


# Whitespace that str.split() splits on and JSON writes as is: json.dumps and orjson escape every
# character below U+0020 (tab and newline included), which leaves the space and these.
_UNICODE_SPACES = (
    "\x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_UNICODE_SPACE_BYTES = tuple(c.encode() for c in _UNICODE_SPACES)
_ORJSON_MAX_DEPTH = 254  # orjson refuses deeper nesting


def _orjson_writes_as_json(obj: Any, depth: int = 0) -> bool:
    """Whether orjson writes `obj` byte for byte as _canonical_json's json.dumps call does: plain
    JSON types (exact dict with string keys, list or tuple, str, bool, None), integers within 64 bits, and
    floats whose shortest repr both libraries write the same (not inf or nan). Anything else
    (subclasses, dates, numpy values, ...) goes through json.dumps and its default=str."""
    kind = type(obj)
    if kind is str or kind is bool or obj is None:
        return True
    if kind is dict:
        if depth >= _ORJSON_MAX_DEPTH:
            return False
        for key, value in obj.items():
            if type(key) is not str or (type(value) is not str and not _orjson_writes_as_json(value, depth + 1)):
                return False
        return True
    if kind is list or kind is tuple:
        if depth >= _ORJSON_MAX_DEPTH:
            return False
        for value in obj:
            if type(value) is not str and not _orjson_writes_as_json(value, depth + 1):
                return False
        return True
    if kind is int:
        return -(2**63) <= obj < 2**64
    if kind is float:
        return orjson.dumps(obj) == float.__repr__(obj).encode()
    return False


def _canonical_json(obj: Any) -> bytes:
    """Compact JSON with sorted keys, as UTF-8. orjson (if installed) writes the same bytes as the
    standard library for the values it is used on, about ten times faster."""
    if orjson is not None and _orjson_writes_as_json(obj):
        try:
            return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
        except orjson.JSONEncodeError:  # e.g. a lone surrogate: json.dumps decides, as without orjson
            pass
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def _collapse_whitespace(data: bytes) -> bytes:
    """`" ".join(text.split())` of UTF-8 JSON text, without splitting when there is nothing to collapse."""
    if data.isascii() or not any(space in data for space in _UNICODE_SPACE_BYTES):
        # The only whitespace is the space (JSON text never starts or ends with one), and bytes.split()
        # splits on it alone: multi-byte UTF-8 sequences hold no ASCII byte.
        return b" ".join(data.split()) if b"  " in data else data
    return " ".join(data.decode().split()).encode()


def _digest(obj: Any) -> str:
    # whitespace runs are collapsed: re-exports of one conversation differ in them
    return hashlib.sha1(_collapse_whitespace(_canonical_json(obj))).hexdigest()


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


# Rows per worker below which tokenize_rows stays in one process (starting workers costs more).
MIN_ROWS_PER_WORKER = 256
# Rows a worker takes at a time: small enough to balance rows of very different lengths.
MAX_SHARD_ROWS = 256


def default_workers() -> int:
    """Processes tokenize_rows uses when none is asked for: the CPUs this process may run on, at most 8."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, 8))


def _tokenize_one(chat, sample: dict, keep_tokens: bool) -> tuple:
    """One row through the trainer's ChatDataset: ("drop", reason, detail) or ("keep", length,
    trained tokens, truncated), plus the input_ids (int32) and trained-mask arrays when keep_tokens."""
    import torch

    from palingenesis.data import IGNORE_INDEX

    before = chat.stats.copy()
    try:
        processed = chat._process(sample)
    except Exception as exc:
        return ("drop", "unrenderable", {"error": repr(exc)[:200]})
    if processed is None or processed["input_ids"].numel() < 2:
        too_long = chat.stats["dropped_too_long"] > before["dropped_too_long"]
        return ("drop", "no_trained_turn_fits" if too_long else "nothing_to_train_on", {})
    input_ids = processed["input_ids"]
    trained = processed["labels"] != IGNORE_INDEX
    outcome = ("keep", int(input_ids.numel()), int(trained.sum()), chat.stats["truncated"] > before["truncated"])
    if keep_tokens:  # arrays, not tensors: a worker sends them back pickled, without shared-memory handles
        outcome += (input_ids.to(torch.int32).numpy(), trained.numpy())
    return outcome


_worker_chat = None  # the ChatDataset of a tokenize_rows worker process


def _init_worker(chat, log_level: int) -> None:
    global _worker_chat
    import torch

    torch.set_num_threads(1)  # one core per worker: the workers already use them all
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    _worker_chat = chat


def _tokenize_shard(shard: list[dict], keep_tokens: bool) -> tuple[list[tuple], Counter]:
    """A worker's rows, in order, and what they added to its ChatDataset's counters."""
    chat = _worker_chat
    before = chat.stats.copy()
    outcomes = [_tokenize_one(chat, sample, keep_tokens) for sample in shard]
    delta = chat.stats.copy()
    delta.subtract(before)
    return outcomes, +delta


def _start_method() -> str:
    """fork where it is safe (Linux, CUDA not initialized): workers inherit the ChatDataset and its
    tokenizer without a copy or a reload. spawn elsewhere: the ChatDataset is pickled once per worker."""
    import multiprocessing as mp

    import torch

    if sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods() and not torch.cuda.is_initialized():
        return "fork"
    return "spawn"


def _tokenize_parallel(samples: list[dict], chat, keep_tokens: bool, workers: int) -> list[tuple]:
    """Outcomes of every row in input order, from `workers` processes that each hold their own
    ChatDataset. Contiguous shards keep the order; what the workers add to their ChatDataset's
    counters is added to `chat.stats`."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    shard_rows = max(1, min(MAX_SHARD_ROWS, math.ceil(len(samples) / (4 * workers))))
    method = _start_method()
    logger.info(f"  Tokenizing {len(samples)} rows in {workers} processes ({method})")
    # Rust tokenizers warn in a forked child once their thread pool has run in the parent; a worker is
    # meant to use one thread. The variable is set while the workers start, then restored.
    parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def restore() -> None:
        if parallelism is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = parallelism

    executor = ProcessPoolExecutor(
        workers,
        mp_context=mp.get_context(method),
        initializer=_init_worker,
        initargs=(chat, logging.getLogger().getEffectiveLevel()),
    )
    outcomes: list[tuple] = []
    try:
        futures = [
            executor.submit(_tokenize_shard, samples[i : i + shard_rows], keep_tokens)
            for i in range(0, len(samples), shard_rows)
        ]
        restore()  # every worker has started
        for future in futures:
            shard_outcomes, counts = future.result()
            chat.stats.update(counts)
            previous = len(outcomes)
            outcomes += shard_outcomes
            if len(outcomes) // 5000 > previous // 5000:
                logger.info(f"  Tokenized {len(outcomes)}/{len(samples)}")
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "a tokenization worker died (out of memory?); run again with fewer processes "
            "(preprocess.num_workers / --num_workers; 1 tokenizes in this process)"
        ) from exc
    finally:
        restore()
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def tokenize_rows(
    samples: list[dict], chat, report: CleanReport, keep_tokens: bool = False, num_workers: int = 1
):
    """Run every row through the trainer's own ChatDataset (template, tools, truncation at
    turn boundaries, masking) and drop the ones training would drop.

    Kept rows get LENGTH and TRAINED_TOKENS. Returns (kept rows, entries): entries are
    (position in kept, input_ids int32, trained mask) per kept row when keep_tokens, for
    scoring; else empty.

    num_workers: processes to tokenize in (0 = default_workers()). Fewer are used when there
    are fewer than MIN_ROWS_PER_WORKER rows per process; 1 runs in this process. The result,
    the report and chat.stats are the same whatever the number of processes.
    """
    import torch

    workers = num_workers if num_workers > 0 else default_workers()
    workers = min(workers, len(samples) // MIN_ROWS_PER_WORKER)
    if workers > 1:
        outcomes = _tokenize_parallel(samples, chat, keep_tokens, workers)
    else:
        outcomes = (_tokenize_one(chat, sample, keep_tokens) for sample in samples)

    kept, entries = [], []
    for i, (sample, outcome) in enumerate(zip(samples, outcomes)):
        if outcome[0] == "drop":
            _, reason, detail = outcome
            if reason == "unrenderable":
                logger.debug(f"row {i} cannot be rendered: {detail['error']}")
            report.drop(sample, reason, **detail)
            continue
        if outcome[3]:
            report.truncated += 1
        sample[LENGTH] = outcome[1]
        sample[TRAINED_TOKENS] = outcome[2]
        if keep_tokens:
            entries.append((len(kept), torch.from_numpy(outcome[4]), torch.from_numpy(outcome[5])))
        kept.append(sample)
        if workers <= 1 and i and i % 5000 == 0:
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
