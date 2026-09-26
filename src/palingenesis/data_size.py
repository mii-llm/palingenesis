"""How much data a run has, without a pass over it; parallel streaming of local files.

The LR schedule needs the number of optimizer steps. Counting them exactly means running
the whole pipeline (render, tokenize, filter, mix, pack) once before training: an extra
epoch of CPU time on a large dataset. A schedule does not need that precision, so the
count is estimated instead:

  rows       from metadata: a map-style dataset's length, a parquet footer, a JSONL's
             newlines (a byte scan, ~GB/s), a Hub dataset's published split sizes
  per row    a uniform random sample of rows (default 2,000) goes through the real
             per-row pipeline: how many training examples a row yields (filters, long
             documents split) and how long they are
  mixture    an epoch of weighted sources ends when the first source runs out (the
             MixedDataset rule), which fixes how many examples each source contributes
  packing    the real packer runs over a synthetic stream of the sampled lengths drawn
             in mixture proportions: packed sequences per example

The per-rank micro-batches per epoch follow; the error is that of a sample mean (about
1-2% at 2,000 rows). train.exact_steps runs the full scan instead.

Streaming local files: `datasets` shards a stream by file, so one JSONL or parquet file
streams through one DataLoader worker (the others stop) and every rank reads the whole
file. local_stream() splits a JSONL file into byte ranges (a line belongs to the range its
first byte is in) and a parquet file into row groups, so ranks and workers read disjoint
parts in parallel.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_ROWS = 2000
_CHUNK = 1 << 24  # 16 MiB


# ------------------------------------------------------------------ files


def local_file(dataset_id: str) -> Path | None:
    """The local data file behind a dataset id (a prepared-output directory resolves to its
    data file), or None for a Hub dataset."""
    path = Path(dataset_id)
    if path.is_dir():
        from palingenesis.prepare import find_prepared_dataset

        prepared = find_prepared_dataset(path)
        if prepared is not None:
            path = prepared
    if path.is_file() and path.suffix in (".jsonl", ".json", ".parquet"):
        return path
    return None


def _is_jsonl(path: Path) -> bool:
    """JSON Lines (one object per line), including .json files written that way."""
    if path.suffix == ".jsonl":
        return True
    with open(path, "rb") as f:
        head = f.read(4096).lstrip()
    return head[:1] == b"{"


def count_lines(path: Path) -> int:
    """Lines of a text file (a final line without a newline included): a byte scan in large chunks."""
    lines, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    return lines + (last != b"\n")


def count_rows(dataset_id: str, split: str = "train", dataset: Any = None) -> int | None:
    """The number of rows, from metadata (None when unknown without reading the data)."""
    if dataset is not None:
        try:
            return len(dataset)  # map-style: Arrow knows it
        except TypeError:
            pass
    path = local_file(dataset_id)
    if path is not None:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return pq.ParquetFile(path).metadata.num_rows
        if _is_jsonl(path):
            return count_lines(path)
        return None
    try:
        from datasets import load_dataset_builder

        splits = load_dataset_builder(dataset_id).info.splits or {}
        info = splits.get(split)
        return int(info.num_examples) if info is not None and info.num_examples else None
    except Exception:  # noqa: BLE001 — no metadata: the caller decides
        return None


def _is_streaming_obj(dataset) -> bool:
    try:
        from datasets import IterableDataset

        return isinstance(dataset, IterableDataset)
    except ImportError:
        return False


# ------------------------------------------------------------------ sampling


def _jsonl_sample(path: Path, k: int, rng: random.Random) -> list[dict]:
    """Uniform rows of a JSONL file without reading it: random byte offsets pick lines with
    probability proportional to their length, so each picked line is kept with weights
    1/length (a weighted resample back to uniform)."""
    size = path.stat().st_size
    picked: list[tuple[bytes, int]] = []
    with open(path, "rb") as f:
        for _ in range(4 * k):
            offset = rng.randrange(size)
            back = max(0, offset - (1 << 20))
            f.seek(back)
            before = f.read(offset - back)
            start = back + before.rfind(b"\n") + 1 if b"\n" in before else back
            f.seek(start)
            line = f.readline()
            if line.strip():
                picked.append((line, len(line)))
    if not picked:
        return []
    rows = rng.choices([line for line, _ in picked], weights=[1.0 / n for _, n in picked], k=min(k, len(picked)))
    return [json.loads(line) for line in rows]


def _parquet_sample(path: Path, k: int, rng: random.Random) -> list[dict]:
    """Rows from up to 16 random row groups (reading whole groups: a footer has no row index)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    groups = rng.sample(range(pf.num_row_groups), min(16, pf.num_row_groups))
    rows: list[dict] = []
    for g in groups:
        table = pf.read_row_group(g)
        take = rng.sample(range(table.num_rows), min(table.num_rows, math.ceil(k / len(groups))))
        rows.extend(table.take(sorted(take)).to_pylist())
    return rows[:k]


def sample_rows(dataset_id: str, split: str, k: int, seed: int, dataset: Any = None) -> tuple[list[dict], bool]:
    """(up to k rows drawn uniformly, whether the draw is uniform). A Hub stream can only be
    sampled from its start (through a shuffle buffer): not uniform when its order is not."""
    rng = random.Random(seed)
    if dataset is not None:
        try:
            n = len(dataset)
            return [dataset[i] for i in sorted(rng.sample(range(n), min(k, n)))], True
        except TypeError:
            return list(dataset.shuffle(seed=seed, buffer_size=10_000).take(k)), False
    path = local_file(dataset_id)
    if path is not None and path.suffix == ".parquet":
        return _parquet_sample(path, k, rng), True
    if path is not None and _is_jsonl(path):
        return _jsonl_sample(path, k, rng), True
    from datasets import load_dataset

    stream = load_dataset(dataset_id, split=split, streaming=True)
    return list(stream.shuffle(seed=seed, buffer_size=10_000).take(k)), False


# ------------------------------------------------------------------ parallel local streams


def _jsonl_shard(path: list[str], start: list[int], end: list[int]):
    """The lines whose first byte is in [start, end), for each of this shard's ranges
    (`datasets` hands a shard a slice of every list-valued argument)."""
    for file, lo, hi in zip(path, start, end):
        with open(file, "rb") as f:
            if lo > 0:
                f.seek(lo - 1)
                f.readline()  # to the first line starting at or after `lo`
            while f.tell() < hi:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    yield json.loads(line)


def _parquet_shard(path: list[str], groups: list[list[int]]):
    import pyarrow.parquet as pq

    for file, ids in zip(path, groups):
        pf = pq.ParquetFile(file)
        for g in ids:
            yield from pf.read_row_group(g).to_pylist()


def local_stream(path: Path, shards: int = 0):
    """A streaming dataset over one local JSONL or parquet file, split into shards that
    ranks and DataLoader workers read in parallel (byte ranges / row groups). None when
    the file cannot be split (a JSON array)."""
    from datasets import IterableDataset

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        n = pq.ParquetFile(path).num_row_groups
        if n < 2:
            logger.warning(
                "%s has a single row group: it streams through one reader. Write it with smaller row "
                "groups (pyarrow row_group_size) for parallel streaming.",
                path,
            )
        groups = [list(range(n))[i :: max(1, min(n, shards or 64))] for i in range(max(1, min(n, shards or 64)))]
        return IterableDataset.from_generator(
            _parquet_shard, gen_kwargs={"path": [str(path)] * len(groups), "groups": groups}
        )
    if not _is_jsonl(path):
        return None
    size = path.stat().st_size
    n = shards or max(1, min(256, size // (32 << 20) or 1)) * 8  # ~4 MiB+ per shard, many per worker
    bounds = [size * i // n for i in range(n + 1)]
    return IterableDataset.from_generator(
        _jsonl_shard,
        gen_kwargs={"path": [str(path)] * n, "start": bounds[:-1], "end": bounds[1:]},
    )


# ------------------------------------------------------------------ estimate


@dataclass
class SourceEstimate:
    name: str
    rows: int | None  # None: unknown (a stream without metadata): treated as never exhausted
    weight: float
    examples_per_row: float  # training examples a row yields (after filters / splitting)
    lengths: list[int] = field(default_factory=list)  # tokens per sampled example
    uniform: bool = True


@dataclass
class StepEstimate:
    micro_batches_per_epoch: int  # per rank
    examples_per_epoch: float  # per rank, before packing
    sequences_per_epoch: float  # per rank, after packing
    relative_error: float  # sampling error of the mean length (1 sigma)
    sources: list[SourceEstimate]

    def describe(self) -> str:
        parts = []
        for s in self.sources:
            rows = f"{s.rows:,}" if s.rows is not None else "unbounded"
            parts.append(
                f"{s.name}: {rows} rows, {s.examples_per_row:.2f} examples/row, "
                f"mean {sum(s.lengths) / max(1, len(s.lengths)):.0f} tokens"
                + ("" if s.uniform else " (sampled from the stream's start: not uniform)")
            )
        return "; ".join(parts)


def mixture_epoch(
    sources: list[SourceEstimate], world_size: int, epoch_examples: float | None = None
) -> tuple[float, list[float]]:
    """(examples per rank per epoch, mixture probabilities). `epoch_examples` given (the
    MixedDataset draws of data.mix_epoch "total"): that. Otherwise the epoch ends when the
    first source with a known size runs out: source i lasts rows_i * examples_per_row_i / p_i
    draws."""
    total = sum(s.weight for s in sources if s.examples_per_row > 0)
    probs = [(s.weight / total if s.examples_per_row > 0 else 0.0) for s in sources]
    if epoch_examples is not None:
        return epoch_examples, probs
    lasts = [
        s.rows * s.examples_per_row / world_size / p for s, p in zip(sources, probs) if p > 0 and s.rows is not None
    ]
    if not lasts:
        raise ValueError("no data source has a known size: set train.max_steps")
    return min(lasts), probs


def packed_per_example(sources: list[SourceEstimate], probs: list[float], max_len: int, seed: int) -> float:
    """Packed sequences per example: the real packer over a synthetic stream of the sampled
    lengths in mixture proportions."""
    import torch

    from palingenesis.data import PackedDataset

    rng = random.Random(seed)
    pools = [(s.lengths, p) for s, p in zip(sources, probs) if p > 0 and s.lengths]
    n = 8192
    stream = []
    for _ in range(n):
        lengths = rng.choices([pl for pl, _ in pools], weights=[p for _, p in pools])[0]
        length = min(max_len, rng.choice(lengths))
        ids = torch.zeros(length, dtype=torch.long)
        stream.append({"input_ids": ids, "labels": ids})
    blocks = sum(1 for _ in PackedDataset(stream, max_len, sort_buffer=256)._bin_packing())
    return blocks / n


def estimate_steps(
    sources: list[SourceEstimate],
    world_size: int,
    batch_size: int,
    packing: bool,
    max_len: int,
    seed: int = 0,
    epoch_examples: float | None = None,
) -> StepEstimate:
    examples, probs = mixture_epoch(sources, world_size, epoch_examples)
    sequences = examples * (packed_per_example(sources, probs, max_len, seed) if packing else 1.0)
    # sampling error of the mixture's mean example length
    variances = []
    for s, p in zip(sources, probs):
        if p > 0 and len(s.lengths) > 1:
            mean = sum(s.lengths) / len(s.lengths)
            var = sum((x - mean) ** 2 for x in s.lengths) / (len(s.lengths) - 1)
            variances.append((p, mean, var, len(s.lengths)))
    mean_all = sum(p * m for p, m, _, _ in variances) or 1.0
    length_error = math.sqrt(sum(p * p * v / n for p, _, v, n in variances)) / mean_all if packing else 0.0
    # the share of rows the filters keep is a sampled proportion too (binomial error)
    rate_error = max(
        (
            math.sqrt(min(s.examples_per_row, 1.0) * max(0.0, 1.0 - s.examples_per_row) / max(1, len(s.lengths)))
            / max(s.examples_per_row, 1e-9)
            for s, p in zip(sources, probs)
            if p > 0
        ),
        default=0.0,
    )
    return StepEstimate(
        micro_batches_per_epoch=int(sequences // batch_size),
        examples_per_epoch=examples,
        sequences_per_epoch=sequences,
        relative_error=math.sqrt(length_error**2 + rate_error**2),
        sources=sources,
    )


def measure_source(
    name: str, rows: int | None, weight: float, sample: list[dict], uniform: bool, make_stage
) -> SourceEstimate:
    """Run the sampled rows through the source's per-row pipeline (make_stage(dataset) -> the
    ChatDataset/PretrainDataset for it) and measure what they yield."""
    from datasets import Dataset

    if not sample:
        return SourceEstimate(name, rows, weight, 0.0, [], uniform)
    stage = make_stage(Dataset.from_list(sample))
    lengths = [int(ex["input_ids"].numel()) for ex in stage]
    return SourceEstimate(name, rows, weight, len(lengths) / len(sample), lengths, uniform)


def sample_size() -> int:
    return int(os.environ.get("PALINGENESIS_SIZE_SAMPLE", SAMPLE_ROWS))
