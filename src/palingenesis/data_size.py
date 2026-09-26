"""How much data a run has, without a pass over it; parallel streaming of local files.

The LR schedule needs the number of optimizer steps. Counting them exactly means running
the whole pipeline (render, tokenize, filter, mix, pack) once before training: an extra
epoch of CPU time on a large dataset. A schedule does not need that precision, so the
count is estimated instead:

  rows       from metadata: a map-style dataset's length, a parquet footer, a JSONL's
             newlines (a byte scan, ~GB/s), a Hub dataset's published split sizes
  per row    uniform random rows go through the real per-row pipeline: the examples a row
             yields (filters, long documents split), their tokens, kept preference pairs
  by size    the row count's scan also records every row's size (Arrow: string sizes), and
             the sample is used per size bin weighted by the bin's share of all rows: what a
             row's length decides (the long tail filters drop) is known for every row
  rounds     rows are drawn in rounds until the reported error reaches target_error()
  mixture    data.mix_epoch: the sources' rows together (or until the first runs out)
  packing    tokens / (max_len * the real packer's fill rate on a synthetic stream)

The per-rank micro-batches per epoch follow, with a calibrated standard error (about 0.1% for
packed SFT, 0.5% for preference pairs in the tests). train.exact_steps runs the full scan instead.

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


# Row sizes (bytes of a JSONL line, or of a row's strings in Arrow data) predict most of
# what a row becomes (its token count, whether it fits max_seq_length), and the whole
# population's sizes cost no tokenization: the newline scan that counts a JSONL's rows finds
# every line's length, and Arrow computes string sizes column-wise. Estimates are
# post-stratified on them: per size bin from the sample, weighted by the bin's share of the
# population, so the length-driven part of a row's fate (the long tail that filters drop) is
# known for every row instead of sampled.

_BIN_RATIO = 1.15  # geometric size bins: rows in a bin are within 15% of each other's size


def size_bin(size: float) -> int:
    return int(math.log(max(size, 1.0)) / math.log(_BIN_RATIO))


@dataclass
class Census:
    """How many rows the population has in each size bin."""

    counts: dict[int, int]

    @property
    def rows(self) -> int:
        return sum(self.counts.values())


_CENSUS_CACHE: dict[tuple, Census] = {}


def jsonl_census(path: Path) -> Census:
    """Row count and line-size histogram of a JSONL file in one byte scan (cached per file)."""
    import numpy as np

    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _CENSUS_CACHE:
        counts: dict[int, int] = {}
        carry = 0  # bytes of the line in progress from the previous chunk
        log_ratio = math.log(_BIN_RATIO)
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK):
                ends = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
                if len(ends):
                    lengths = np.diff(np.concatenate(([-1], ends))).astype(np.int64)
                    lengths[0] += carry
                    lengths = lengths[lengths > 1]  # blank lines are not rows
                    bins = (np.log(np.maximum(lengths, 1)) / log_ratio).astype(np.int64)
                    for value, n in zip(*np.unique(bins, return_counts=True)):
                        counts[int(value)] = counts.get(int(value), 0) + int(n)
                    carry = len(chunk) - 1 - int(ends[-1])
                else:
                    carry += len(chunk)
        if carry > 0:  # a last line without a newline
            counts[size_bin(carry + 1)] = counts.get(size_bin(carry + 1), 0) + 1
        _CENSUS_CACHE[key] = Census(counts)
    return _CENSUS_CACHE[key]


def count_lines(path: Path) -> int:
    """Rows of a JSONL file (non-blank lines, a last line without a newline included)."""
    return jsonl_census(path).rows


def arrow_row_sizes(table) -> "list[int]":
    """Bytes of every string in each row (nested lists and structs included): the Arrow
    counterpart of a JSONL line's length, computed column-wise."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    n = table.num_rows
    total = np.zeros(n, dtype=np.int64)

    def add(array, owner):
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        t = array.type
        if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t):
            lengths = pc.fill_null(pc.binary_length(array), 0).to_numpy(zero_copy_only=False)
            np.add.at(total, owner, lengths)
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            parents = pc.list_parent_indices(array).to_numpy(zero_copy_only=False)
            add(pc.list_flatten(array), owner[parents])
        elif pa.types.is_struct(t):
            for i in range(t.num_fields):
                add(array.field(i), owner)

    for column in table.columns:
        add(column, np.arange(n))
    return total.tolist()


def _census_of_sizes(sizes) -> Census:
    counts: dict[int, int] = {}
    for size in sizes:
        b = size_bin(size)
        counts[b] = counts.get(b, 0) + 1
    return Census(counts)


def census(dataset_id: str, split: str = "train", dataset: Any = None) -> Census | None:
    """The population's size histogram (None when it would take reading a remote stream)."""
    if dataset is not None and not _is_streaming_obj(dataset):
        table = dataset.data.table if hasattr(dataset, "data") else None
        if table is None:
            return None
        if getattr(dataset, "_indices", None) is not None:
            table = table.take(dataset._indices.column(0))
        return _census_of_sizes(arrow_row_sizes(table))
    path = local_file(dataset_id)
    if path is None:
        return None
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in _CENSUS_CACHE:
            pf = pq.ParquetFile(path)
            sizes: list[int] = []
            for g in range(pf.num_row_groups):
                sizes.extend(arrow_row_sizes(pf.read_row_group(g)))
            _CENSUS_CACHE[key] = _census_of_sizes(sizes)
        return _CENSUS_CACHE[key]
    return jsonl_census(path) if _is_jsonl(path) else None


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


@dataclass
class Sample:
    """Sampled rows with their sizes and sampling weights (1/length for byte-offset picks:
    weighted, they are uniform over rows)."""

    rows: list[dict]
    sizes: list[int]
    weights: list[float]
    uniform: bool = True
    exhaustive: bool = False  # every row of the population, once: the estimate is exact

    def extend(self, other: "Sample") -> None:
        """Pool another round of draws (independent draws of the same design)."""
        self.rows += other.rows
        self.sizes += other.sizes
        self.weights += other.weights
        self.uniform = self.uniform and other.uniform


def _jsonl_sample(path: Path, k: int, rng: random.Random) -> Sample:
    """Rows of a JSONL file without reading it: random byte offsets pick the line they fall
    in, with probability proportional to its length; weights 1/length undo that."""
    size = path.stat().st_size
    rows, sizes, weights = [], [], []
    with open(path, "rb") as f:
        for _ in range(k):
            offset = rng.randrange(size)
            back = max(0, offset - (1 << 20))
            f.seek(back)
            before = f.read(offset - back)
            start = back + before.rfind(b"\n") + 1 if b"\n" in before else back
            f.seek(start)
            line = f.readline()
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:  # a line longer than the look-back window: skip it
                continue
            n = len(line.rstrip(b"\n")) + 1
            sizes.append(n)
            weights.append(1.0 / n)
    return Sample(rows, sizes, weights)


def _arrow_sample(table, k: int, rng: random.Random) -> Sample:
    take = sorted(rng.sample(range(table.num_rows), min(k, table.num_rows)))
    part = table.take(take)
    return Sample(part.to_pylist(), arrow_row_sizes(part), [1.0] * len(take))


def _parquet_sample(path: Path, k: int, rng: random.Random) -> Sample:
    """Rows from up to 16 random row groups (reading whole groups: a footer has no row index)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    groups = rng.sample(range(pf.num_row_groups), min(16, pf.num_row_groups))
    parts = []
    for g in groups:
        table = pf.read_row_group(g)
        take = rng.sample(range(table.num_rows), min(table.num_rows, math.ceil(k / len(groups))))
        parts.append(table.take(sorted(take)))
    table = pa.concat_tables(parts)
    return Sample(table.to_pylist(), arrow_row_sizes(table), [1.0] * table.num_rows)


def sample_rows(dataset_id: str, split: str, k: int, seed: int, dataset: Any = None) -> Sample:
    """About k rows drawn uniformly (weighted), with their sizes. A Hub stream can only be
    sampled from its start (through a shuffle buffer): not uniform when its order is not."""
    rng = random.Random(seed)
    if dataset is not None and not _is_streaming_obj(dataset):
        table = dataset.data.table
        if getattr(dataset, "_indices", None) is not None:
            table = table.take(dataset._indices.column(0))
        sample = _arrow_sample(table, k, rng)
        sample.exhaustive = table.num_rows <= k
        return sample
    if dataset is None:
        path = local_file(dataset_id)
        if path is not None and path.suffix == ".parquet":
            import pyarrow.parquet as pq

            if pq.ParquetFile(path).metadata.num_rows <= k:  # small: all of it, exactly
                table = pq.read_table(path)
                return Sample(table.to_pylist(), arrow_row_sizes(table), [1.0] * table.num_rows, exhaustive=True)
            return _parquet_sample(path, k, rng)
        if path is not None and _is_jsonl(path):
            if count_lines(path) <= k:  # small: all of it, exactly
                lines = [line for line in open(path, "rb") if line.strip()]
                return Sample(
                    [json.loads(line) for line in lines],
                    [len(line.rstrip(b"\n")) + 1 for line in lines],
                    [1.0] * len(lines),
                    exhaustive=True,
                )
            return _jsonl_sample(path, 2 * k, rng)  # weighted picks: twice as many for the same precision
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split, streaming=True)
    rows = list(dataset.shuffle(seed=seed, buffer_size=10_000).take(k))
    return Sample(rows, [len(json.dumps(r, default=str)) for r in rows], [1.0] * len(rows), uniform=False)


def stratified_mean(sample: Sample, values: list[float], population: Census | None, min_per_bin: int = 12):
    """(mean of `values` over the population, its standard error). With a census: per size
    bin (adjacent bins merged until each holds min_per_bin sampled rows), weighted by the
    bin's share of the population. Without: the weighted sample mean."""
    w = sample.weights
    if population is None or not values:
        total = sum(w) or 1.0
        mean = sum(wi * v for wi, v in zip(w, values)) / total
        n_eff = total**2 / max(sum(wi * wi for wi in w), 1e-30)
        var = sum(wi * (v - mean) ** 2 for wi, v in zip(w, values)) / total
        return mean, math.sqrt(var / max(n_eff, 1.0))
    by_bin: dict[int, list[int]] = {}
    for i, size in enumerate(sample.sizes):
        by_bin.setdefault(size_bin(size), []).append(i)
    # strata: runs of consecutive bins (over sample and population) with enough sampled rows
    keys = sorted(set(by_bin) | set(population.counts))
    strata, current, filled = [], [], 0
    for key in keys:
        current.append(key)
        filled += len(by_bin.get(key, ()))
        if filled >= min_per_bin:
            strata.append(current)
            current, filled = [], 0
    if current:
        if strata:
            strata[-1].extend(current)
        else:
            strata.append(current)
    rows = population.rows
    mean = var = 0.0
    for stratum in strata:
        share = sum(population.counts.get(k, 0) for k in stratum) / rows
        idx = [i for k in stratum for i in by_bin.get(k, ())]
        if not idx or share == 0:
            continue
        ws = sum(w[i] for i in idx)
        m = sum(w[i] * values[i] for i in idx) / ws
        v = sum(w[i] * (values[i] - m) ** 2 for i in idx) / ws
        n_eff = ws**2 / sum(w[i] ** 2 for i in idx)
        mean += share * m
        var += share * share * v / max(n_eff, 1.0)
    return mean, math.sqrt(var)


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
    lengths: list[int] = field(default_factory=list)  # tokens per example, over the sample
    uniform: bool = True
    per_row: list[list[int]] = field(default_factory=list)  # each sampled row's example lengths
    sample: Sample | None = None
    population: Census | None = None
    examples_error: float = 0.0  # standard error of examples_per_row
    tokens_per_row: float = 0.0  # tokens a row yields (all its examples)
    tokens_error: float = 0.0

    def draw_rows(self, rng: random.Random, n: int) -> list[list[int]]:
        """n rows' example lengths, drawn as the population's rows would be: a size bin by its
        population share, then a sampled row of that bin by weight."""
        if self.sample is None or not self.per_row:
            return [[length] for length in rng.choices(self.lengths or [1], k=n)]
        by_bin: dict[int, list[int]] = {}
        for i, size in enumerate(self.sample.sizes):
            by_bin.setdefault(size_bin(size), []).append(i)
        if self.population is not None:
            bins = [b for b in self.population.counts if b in by_bin]
            shares = [self.population.counts[b] for b in bins]
        else:
            bins = list(by_bin)
            shares = [sum(self.sample.weights[i] for i in by_bin[b]) for b in bins]
        out = []
        for b in rng.choices(bins, weights=shares, k=n):
            idx = by_bin[b]
            i = rng.choices(idx, weights=[self.sample.weights[j] for j in idx])[0]
            out.append(self.per_row[i])
        return out


@dataclass
class StepEstimate:
    micro_batches_per_epoch: int  # per rank
    examples_per_epoch: float  # per rank, before packing
    sequences_per_epoch: float  # per rank, after packing
    relative_error: float  # standard error of the estimate, relative (1 sigma)
    sources: list[SourceEstimate]

    def describe(self) -> str:
        parts = []
        for s in self.sources:
            rows = f"{s.rows:,}" if s.rows is not None else "unbounded"
            parts.append(
                f"{s.name}: {rows} rows, {s.examples_per_row:.3f} examples/row, "
                f"mean {sum(s.lengths) / max(1, len(s.lengths)):.0f} tokens"
                + (", size-stratified" if s.population is not None else "")
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


def packing_fill(sources: list[SourceEstimate], probs: list[float], max_len: int, seed: int) -> float:
    """How full the packer's blocks are (tokens / (blocks * max_len)): the real packer over a
    synthetic stream of examples, drawn row by row as the population's rows (by size) in
    mixture proportions. A ratio, so the stream's own token total cancels out: the number of
    tokens comes from the stratified estimate, only the packer's efficiency from here."""
    import torch

    from palingenesis.data import PackedDataset

    rng = random.Random(seed)
    live = [(s, p) for s, p in zip(sources, probs) if p > 0 and (s.lengths or s.per_row)]
    stream: list[dict] = []
    tokens = 0
    while len(stream) < 16384:
        source = rng.choices([s for s, _ in live], weights=[p for _, p in live])[0]
        for lengths in source.draw_rows(rng, 64):
            for length in lengths:
                n = min(max_len, length)
                ids = torch.zeros(n, dtype=torch.long)
                stream.append({"input_ids": ids, "labels": ids})
                tokens += n
    blocks = sum(1 for _ in PackedDataset(stream, max_len, sort_buffer=256)._bin_packing())
    return tokens / (blocks * max_len)


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
    live = [(s, p) for s, p in zip(sources, probs) if p > 0]
    if packing:
        # sequences = tokens / (max_len * fill): tokens per example from the stratified
        # per-row estimates (capped at max_len, as the packer cuts), fill from the packer
        tokens_per_example = sum(p * s.tokens_per_row / max(s.examples_per_row, 1e-9) for s, p in live)
        sequences = examples * tokens_per_example / (max_len * packing_fill(sources, probs, max_len, seed))
        error = math.sqrt(sum((p * s.tokens_error / max(s.tokens_per_row, 1e-9)) ** 2 for s, p in live))
    else:
        sequences = examples
        error = math.sqrt(sum((p * s.examples_error / max(s.examples_per_row, 1e-9)) ** 2 for s, p in live))
    return StepEstimate(
        micro_batches_per_epoch=int(sequences // batch_size),
        examples_per_epoch=examples,
        sequences_per_epoch=sequences,
        relative_error=error,
        sources=sources,
    )


def measure_source(
    name: str,
    rows: int | None,
    weight: float,
    sample: Sample,
    population: Census | None,
    make_stage,
    per_row: list[list[int]] | None = None,
) -> SourceEstimate:
    """Run the sampled rows through the source's per-row pipeline (make_stage(dataset) -> the
    ChatDataset/PretrainDataset for it) and estimate what a row of the population yields
    (post-stratified on row size). `per_row`: the measured rows of earlier rounds, extended
    in place."""
    if not sample.rows:
        return SourceEstimate(name, rows, weight, 0.0, [], sample.uniform)
    stage = make_stage([])  # its per-row step (_process) is what its iteration applies to each row
    per_row = per_row if per_row is not None else []
    for row in sample.rows[len(per_row) :]:  # rows of earlier rounds are measured already
        out = stage._process(row)
        per_row.append([] if out is None else [int(out["input_ids"].numel())])
    examples, examples_error = stratified_mean(sample, [len(r) for r in per_row], population)
    tokens, tokens_error = stratified_mean(sample, [sum(r) for r in per_row], population)
    lengths = [n for r in per_row for n in r]
    return SourceEstimate(
        name,
        rows,
        weight,
        examples,
        lengths,
        sample.uniform,
        per_row,
        sample,
        population,
        examples_error,
        tokens,
        tokens_error,
    )


def sample_size() -> int:
    return int(os.environ.get("PALINGENESIS_SIZE_SAMPLE", SAMPLE_ROWS))


def target_error() -> float:
    """Relative standard error sequential sampling stops at (PALINGENESIS_SIZE_PRECISION)."""
    return float(os.environ.get("PALINGENESIS_SIZE_PRECISION", 0.005))


def sequential(draw, evaluate, rounds: int = 8, population: int | None = None):
    """Sample until the estimate is precise enough: draw(round) -> a Sample, evaluate(sample)
    -> (estimate, relative standard error). Stops at target_error(), after `rounds` rounds,
    or once the pooled sample reaches a quarter of the `population` (two rounds at least):
    past that, sampling costs about what the exact count would. Returns the pooled sample
    and its evaluation."""
    sample = draw(0)
    result = evaluate(sample)
    limit = max(2 * len(sample.rows), population // 4) if population else None
    for r in range(1, rounds):
        if not sample.rows or sample.exhaustive or not sample.uniform or result[1] <= target_error():
            break
        if limit is not None and len(sample.rows) >= limit:
            break
        sample.extend(draw(r))
        result = evaluate(sample)
    return sample, result
