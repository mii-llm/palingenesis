"""Sizing a run without a pass over the data, parallel local streams, per-epoch order."""

import json
import random
from collections import Counter

import pytest

from palingenesis import data_size
from tests.test_last_turn_integration import TOK, needs_tok


def _rows(n: int, seed: int = 0, tag: str = "") -> list[dict]:
    """Chat rows with a heavy-tailed answer length (some too long for the context)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        words = int(rng.lognormvariate(3.5, 0.9))
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"{tag} question {i}"},
                    {"role": "assistant", "content": " ".join(f"w{j % 97}" for j in range(words))},
                ]
            }
        )
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_count_rows_and_uniform_jsonl_sampling(tmp_path):
    rng = random.Random(1)
    # rows of very different byte lengths: offset sampling picks long lines more often,
    # the 1/length reweighting must undo it
    rows = [{"id": i, "pad": "x" * (5 if i % 2 else 500)} for i in range(4000)]
    path = _write_jsonl(tmp_path / "d.jsonl", rows)
    assert data_size.count_rows(str(path)) == 4000
    sample = data_size._jsonl_sample(path, 4000, rng)
    long_share = sum(len(r["pad"]) == 500 for r in sample) / len(sample)
    assert abs(long_share - 0.5) < 0.05, long_share  # uniform over rows, not over bytes


def test_local_jsonl_stream_shards_cover_every_line_once(tmp_path):
    rows = [{"id": i, "text": "y" * (i % 50)} for i in range(3001)]
    path = _write_jsonl(tmp_path / "d.jsonl", rows)
    stream = data_size.local_stream(path, shards=7)
    assert stream.num_shards == 7
    seen = Counter(r["id"] for r in stream)
    assert sorted(seen) == list(range(3001)) and set(seen.values()) == {1}
    # ranks read disjoint parts
    from datasets.distributed import split_dataset_by_node

    parts = [set(r["id"] for r in split_dataset_by_node(stream, rank=k, world_size=3)) for k in range(3)]
    assert sum(len(p) for p in parts) == 3001 and set().union(*parts) == set(range(3001))


def test_local_parquet_stream_by_row_group(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "d.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": i} for i in range(1000)]), path, row_group_size=100)
    assert data_size.count_rows(str(path)) == 1000
    stream = data_size.local_stream(path)
    assert stream.num_shards == 10 and sorted(r["id"] for r in stream) == list(range(1000))


def _exact_micro_batches(config, batch_size, dataset=None):
    from palingenesis.data import build_dataloader

    return sum(1 for _ in build_dataloader(dataset if dataset is not None else config, TOK, config, 0, 1, batch_size))


@needs_tok
@pytest.mark.parametrize("packing", [True, False])
def test_estimate_matches_the_exact_count(tmp_path, packing, monkeypatch):
    from palingenesis.config import DataConfig
    from palingenesis.data import estimate_run

    monkeypatch.setenv("PALINGENESIS_SIZE_SAMPLE", "1500")
    path = _write_jsonl(tmp_path / "chat.jsonl", _rows(6000))
    config = DataConfig(dataset=str(path), max_seq_length=256, packing=packing, num_workers=0, length_group_buffer=0)
    est = estimate_run(config, TOK, world_size=1, batch_size=4)
    exact = _exact_micro_batches(config, 4)
    assert abs(est.micro_batches_per_epoch - exact) / exact < 0.06, (est.micro_batches_per_epoch, exact)


@needs_tok
def test_estimate_of_a_weighted_mixture(tmp_path, monkeypatch):
    from palingenesis.config import DataConfig
    from palingenesis.data import estimate_run

    monkeypatch.setenv("PALINGENESIS_SIZE_SAMPLE", "1500")
    small = _write_jsonl(tmp_path / "small.jsonl", _rows(1500, seed=1, tag="s"))
    large = _write_jsonl(tmp_path / "large.jsonl", _rows(6000, seed=2, tag="l"))
    config = DataConfig(
        sources=[{"dataset": str(small), "weight": 0.5}, {"dataset": str(large), "weight": 0.5}],
        max_seq_length=256,
        packing=True,
        num_workers=0,
    )
    est = estimate_run(config, TOK, world_size=1, batch_size=4)
    exact = _exact_micro_batches(config, 4)
    # the epoch ends when the small source runs out: ~2 x its examples
    assert abs(est.micro_batches_per_epoch - exact) / exact < 0.08, (est.micro_batches_per_epoch, exact)


@needs_tok
def test_each_epoch_has_a_new_order_and_mixtures_draw_afresh(tmp_path):
    from palingenesis.config import DataConfig
    from palingenesis.data import build_dataset

    small = _write_jsonl(tmp_path / "small.jsonl", _rows(200, seed=1, tag="small"))
    large = _write_jsonl(tmp_path / "large.jsonl", _rows(2000, seed=2, tag="large"))
    config = DataConfig(
        sources=[{"dataset": str(small), "weight": 0.5}, {"dataset": str(large), "weight": 0.5}],
        max_seq_length=256,
        packing=False,
        length_group_buffer=0,
    )
    ds = build_dataset(config, TOK, config, 0, 1, 1)

    def epoch(e):
        ds.set_epoch(e)
        return [TOK.decode(x["input_ids"]) for x in ds]

    first, again, second = epoch(0), epoch(0), epoch(1)
    assert first == again  # an epoch's order is reproducible
    assert first != second  # and changes between epochs
    large_first = {t for t in first if "large" in t}
    large_second = {t for t in second if "large" in t}
    # the large source contributes ~200 rows per epoch: a fresh random draw each epoch, not
    # its same first rows
    assert len(large_first & large_second) < 0.5 * len(large_first)


def test_lockstep_single_rank_passes_everything():
    from palingenesis.train import _lockstep

    assert list(_lockstep(iter(range(5)), None)) == [0, 1, 2, 3, 4]


@needs_tok
def test_streaming_a_local_file_is_sized_and_yields_the_same_examples(tmp_path, monkeypatch):
    from palingenesis.config import DataConfig
    from palingenesis.data import build_dataset, estimate_run

    monkeypatch.setenv("PALINGENESIS_SIZE_SAMPLE", "800")
    path = _write_jsonl(tmp_path / "chat.jsonl", _rows(1200, seed=3))
    mapped = DataConfig(dataset=str(path), max_seq_length=256, packing=False, length_group_buffer=0)
    streamed = DataConfig(dataset=str(path), max_seq_length=256, packing=False, length_group_buffer=0, streaming=True)
    assert estimate_run(streamed, TOK, 1, 1).micro_batches_per_epoch > 0  # no max_steps needed
    a = sorted(TOK.decode(x["input_ids"]) for x in build_dataset(mapped, TOK, mapped, 0, 1, 1))
    b = sorted(TOK.decode(x["input_ids"]) for x in build_dataset(streamed, TOK, streamed, 0, 1, 1))
    assert a == b


def _lockstep_rank(rank: int, world: int, port: int, out):
    import torch.distributed as dist

    from palingenesis.train import _lockstep

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    group = dist.new_group(backend="gloo")
    out.put((rank, list(_lockstep(iter(range(5 if rank == 0 else 3)), group))))
    dist.destroy_process_group()


def test_lockstep_ends_the_epoch_on_every_rank_together():
    """Rank 0 has 5 batches, rank 1 has 3: both stop after 3 (no rank steps alone)."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_lockstep_rank, args=(r, 2, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(out.get(timeout=60) for _ in procs)
    for p in procs:
        p.join(timeout=30)
    assert results == {0: [0, 1, 2], 1: [0, 1, 2]}
