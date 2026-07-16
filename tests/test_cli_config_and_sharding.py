"""Tests for two footgun fixes:

1. Config.from_cli accepts a positional ``*.yaml`` path (not just ``--config``)
   and warns loudly when no config is found (avoids silently training the
   built-in Llama/ultrachat demo defaults).
2. _shard_streaming_dataset never over-shards a single-shard streaming dataset
   across dataloader workers (which used to raise IndexError), while keeping
   correct contiguous sharding for map-style datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from palingenesis.config import Config

# ══════════════════════════════════════════════════════════════════════════════
# from_cli config resolution
# ══════════════════════════════════════════════════════════════════════════════


def test_from_cli_positional_yaml(tmp_path):
    p = tmp_path / "cpt.yaml"
    p.write_text("model:\n  name_or_path: foo/bar\n")
    cfg = Config.from_cli([str(p)])
    assert cfg.model.name_or_path == "foo/bar"


def test_from_cli_positional_yaml_with_overrides(tmp_path):
    p = tmp_path / "cpt.yml"
    p.write_text("train:\n  learning_rate: 0.001\n")
    cfg = Config.from_cli([str(p), "--train.epochs", "3"])
    # override applied
    assert cfg.train.epochs == 3
    # positional yaml consumed as config, not misparsed as an override value
    assert abs(cfg.train.learning_rate - 0.001) < 1e-12


def test_from_cli_flag_still_works(tmp_path):
    p = tmp_path / "cpt.yaml"
    p.write_text("model:\n  name_or_path: baz/qux\n")
    cfg = Config.from_cli(["--config", str(p), "--train.epochs", "2"])
    assert cfg.model.name_or_path == "baz/qux"
    assert cfg.train.epochs == 2


def test_from_cli_no_config_warns_and_uses_defaults(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        cfg = Config.from_cli(["--train.epochs", "2"])
    # still applies overrides on top of defaults
    assert cfg.train.epochs == 2
    # default demo model retained
    assert cfg.model.name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
    # loud warning emitted
    assert any("No config file passed" in r.getMessage() for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# _shard_streaming_dataset
# ══════════════════════════════════════════════════════════════════════════════


class _FakeWorkerInfo:
    def __init__(self, num_workers, worker_id):
        self.num_workers = num_workers
        self.id = worker_id


def test_shard_streaming_single_shard_many_workers_no_crash(monkeypatch):
    """Single-shard streaming dataset + 4 workers must NOT raise (was IndexError).

    HF assigns the lone shard to worker 0 and empties the surplus workers, so
    across all 4 workers we get the full dataset exactly once (no crash, no dup).
    """
    from datasets import Dataset

    from palingenesis import data as data_mod

    collected = []
    for worker_id in range(4):
        streaming = Dataset.from_dict({"text": list("abcdefgh")}).to_iterable_dataset(num_shards=1)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda wid=worker_id: _FakeWorkerInfo(4, wid))
        out = data_mod._shard_streaming_dataset(streaming, rank=0, world_size=1)
        # Iterating must not raise (previously IndexError on empty over-shard).
        collected.extend(ex["text"] for ex in out)

    # Union across workers covers the dataset exactly once (no duplication).
    assert sorted(collected) == list("abcdefgh")


def test_shard_mapstyle_contiguous(monkeypatch):
    from datasets import Dataset

    from palingenesis import data as data_mod

    ds = Dataset.from_dict({"text": list("abcdefgh")})
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    shard0 = data_mod._shard_streaming_dataset(ds, rank=0, world_size=2)
    shard1 = data_mod._shard_streaming_dataset(ds, rank=1, world_size=2)

    got0 = [ex["text"] for ex in shard0]
    got1 = [ex["text"] for ex in shard1]

    # Disjoint and complete coverage across the two rank shards.
    assert set(got0).isdisjoint(got1)
    assert sorted(got0 + got1) == list("abcdefgh")
    assert len(got0) == 4 and len(got1) == 4


def test_shard_streaming_multinode_splits(monkeypatch):
    """world_size>1 streaming shards via split_dataset_by_node without crashing."""
    from datasets import Dataset

    from palingenesis import data as data_mod

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    def _rank_items(rank):
        streaming = Dataset.from_dict({"text": [str(i) for i in range(8)]}).to_iterable_dataset(num_shards=1)
        out = data_mod._shard_streaming_dataset(streaming, rank=rank, world_size=2)
        return [ex["text"] for ex in out]

    r0, r1 = _rank_items(0), _rank_items(1)
    # No crash, disjoint, and together they cover the whole dataset.
    assert set(r0).isdisjoint(r1)
    assert sorted(r0 + r1, key=int) == [str(i) for i in range(8)]


# ══════════════════════════════════════════════════════════════════════════════
# build_dataloader: multi-source path must honor config.streaming
# ══════════════════════════════════════════════════════════════════════════════


def _write_jsonl(path, n):
    import json

    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"documento {i} " * 10}) + "\n")
    return str(path)


def _build_and_count(tmp_path, streaming):
    from transformers import AutoTokenizer

    from palingenesis.config import DataConfig
    from palingenesis.data import build_dataloader

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    f1 = _write_jsonl(tmp_path / "a.jsonl", 120)
    f2 = _write_jsonl(tmp_path / "b.jsonl", 120)

    cfg = DataConfig()
    cfg.max_seq_length = 64
    cfg.packing = True
    cfg.num_workers = 0  # keep single-process: exercises shuffle/streaming branch without spawn
    cfg.streaming = streaming
    cfg.sources = [
        {"dataset": f1, "weight": 0.5, "mode": "pretrain", "text_field": "text"},
        {"dataset": f2, "weight": 0.5, "mode": "pretrain", "text_field": "text"},
    ]

    dl = build_dataloader(cfg, tok, cfg, rank=0, world_size=1, batch_size=2)
    it = iter(dl)
    seqs = sum(next(it)["input_ids"].shape[0] for _ in range(3))
    return seqs


def test_build_dataloader_sources_streaming_true(tmp_path):
    assert _build_and_count(tmp_path, streaming=True) == 6


def test_build_dataloader_sources_streaming_false(tmp_path):
    # Previously this path ignored streaming=False (hardcoded streaming=True) and,
    # combined with the shard bug, could crash; the map-style shuffle also needed
    # a different call signature. Must now build and yield cleanly.
    assert _build_and_count(tmp_path, streaming=False) == 6
