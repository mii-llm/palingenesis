"""Tests for the pre-tokenized dataset cache.

Covers the full loop: fingerprint → materialize → cache-validity → reload, and the
config guards. The reload MUST be byte-for-byte identical to the on-the-fly stream
(same tokens, same masks, same packing) — that's the whole point: skip tokenization
without changing what the model trains on.

A GPT-2 base + the project's real `{% generation %}` ChatML template hosts the
masking so the tests are network-free and deterministic.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.config import DataConfig  # noqa: E402
from palingenesis.data import (  # noqa: E402
    PRETOK_DATA,
    PRETOK_META,
    PretokenizedDataset,
    build_dataset,
    build_pretokenized_dataloader,
    materialize_pretokenized,
    pretokenize_fingerprint,
    pretokenized_cache_valid,
)

# Reuse the real template + tokenizer helper from the integration suite.
from test_last_turn_integration import CHAT_TEMPLATE  # noqa: E402


def _make_tokenizer():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None
    tok.add_special_tokens(
        {"additional_special_tokens": [x for x in ("<|im_start|>", "<|im_end|>", "<think>", "</think>") if x not in tok.get_vocab()]}
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.eos_token = "<|im_end|>"
    tok.pad_token = tok.eos_token
    return tok


TOK = _make_tokenizer()
needs_tok = pytest.mark.skipif(TOK is None, reason="gpt2 tokenizer not cached (offline)")

ROWS = [
    {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
    for i in range(12)
]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _cfg(path, **kw):
    base = dict(
        sources=[{"dataset": str(path), "split": "train", "mode": "sft", "messages_field": "messages"}],
        max_seq_length=256,
        packing=False,
        num_workers=0,
        length_group_buffer=0,
        streaming=True,
        seed=42,
    )
    base.update(kw)
    return DataConfig(**base)


class _Wrap:
    """pretokenize_fingerprint / materialize expect an object with a `.data` attr."""

    def __init__(self, data):
        self.data = data


# ── fingerprint ────────────────────────────────────────────────────────────────
@needs_tok
def test_fingerprint_stable_and_sensitive(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    fp = lambda c: pretokenize_fingerprint(_Wrap(c), TOK)  # noqa: E731

    base = _cfg(p)
    assert fp(base) == fp(_cfg(p)), "same config → same fingerprint"
    assert fp(base) != fp(_cfg(p, max_seq_length=128)), "max_seq_length must change fp"
    assert fp(base) != fp(_cfg(p, packing=True)), "packing must change fp"
    assert fp(base) != fp(_cfg(p, train_on_reasoning=False)), "train_on_reasoning must change fp"
    assert fp(base) != fp(_cfg(p, seed=7)), "seed must change fp"

    # A different data file (different path) must change the fingerprint.
    p2 = tmp_path / "d2.jsonl"
    _write(p2, ROWS)
    assert fp(base) != fp(_cfg(p2)), "source path must change fp"


@needs_tok
def test_fingerprint_tracks_file_mtime_size(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    before = pretokenize_fingerprint(_Wrap(_cfg(p)), TOK)
    _write(p, ROWS + [ROWS[0]])  # change size/mtime
    after = pretokenize_fingerprint(_Wrap(_cfg(p)), TOK)
    assert before != after, "editing the underlying file must invalidate the cache"


# ── materialize → validity → reload roundtrip ────────────────────────────────────
def _materialize(cfg, cache_dir):
    fingerprint = pretokenize_fingerprint(_Wrap(cfg), TOK)

    def factory():
        return build_dataset(cfg, TOK, cfg, 0, 1, 1)

    n = materialize_pretokenized(factory, cache_dir, fingerprint, _Wrap(cfg), TOK)
    return fingerprint, n


@needs_tok
def test_cache_validity_lifecycle(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    cache = tmp_path / "cache"
    cfg = _cfg(p)

    ok, reason = pretokenized_cache_valid(cache, "anything")
    assert not ok and reason == "no cache found"

    fingerprint, n = _materialize(cfg, cache)
    assert n == len(ROWS)
    assert (cache / PRETOK_DATA).exists() and (cache / PRETOK_META).exists()

    ok, _ = pretokenized_cache_valid(cache, fingerprint)
    assert ok, "fresh cache with matching fingerprint must be valid"

    ok, reason = pretokenized_cache_valid(cache, "STALE")
    assert not ok and "changed" in reason


def _seqs_from_dataset(cfg):
    return [
        (ex["input_ids"].tolist(), ex["labels"].tolist(), ex["attention_mask"].tolist(),
         ex.get("position_ids").tolist() if "position_ids" in ex else None)
        for ex in build_dataset(cfg, TOK, cfg, 0, 1, 1)
    ]


def _seqs_from_cache(cache, cfg):
    """Iterate the stored rows directly (pre-collate) — the DataLoader adds
    pad-to-multiple padding, which is a separate, already-tested concern."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(cache / PRETOK_DATA), split="train", streaming=False)
    return [
        (ex["input_ids"].tolist(), ex["labels"].tolist(), ex["attention_mask"].tolist(),
         ex.get("position_ids").tolist() if "position_ids" in ex else None)
        for ex in PretokenizedDataset(ds, 0, 1)
    ]


@needs_tok
def test_reload_identical_unpacked(tmp_path):
    """Non-packed: cached tensors equal the on-the-fly build (batch_size=1 → no padding)."""
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    cache = tmp_path / "cache"
    cfg = _cfg(p, packing=False)

    orig = _seqs_from_dataset(cfg)
    _materialize(cfg, cache)
    got = _seqs_from_cache(cache, cfg)

    assert len(got) == len(orig) == len(ROWS)
    assert got == orig, "reloaded stream must be byte-identical to the on-the-fly stream"
    assert all(s[3] is None for s in got), "no position_ids when unpacked"


@needs_tok
def test_reload_identical_packed_has_position_ids(tmp_path):
    """Packed: cached rows carry position_ids and match the on-the-fly packed stream."""
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    cache = tmp_path / "cache"
    cfg = _cfg(p, packing=True, max_seq_length=64)

    orig = _seqs_from_dataset(cfg)
    _, n = _materialize(cfg, cache)

    import json as _json

    meta = _json.loads((cache / PRETOK_META).read_text())
    assert meta["has_position_ids"] is True
    assert meta["num_sequences"] == n

    got = _seqs_from_cache(cache, cfg)
    assert len(got) == len(orig) >= 1
    assert got == orig
    assert all(s[3] is not None for s in got), "packed rows must carry position_ids"


@needs_tok
def test_dataloader_batches_and_sharding(tmp_path):
    """The load-time DataLoader yields well-formed batches, and rank sharding
    partitions the cached rows disjointly across ranks (no dupes, no loss)."""
    p = tmp_path / "d.jsonl"
    _write(p, ROWS)
    cache = tmp_path / "cache"
    cfg = _cfg(p, packing=False)
    _materialize(cfg, cache)

    dl = build_pretokenized_dataloader(cache, TOK, cfg, 0, 1, 4)
    batch = next(iter(dl))
    assert set(batch.keys()) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["labels"].shape

    from datasets import load_dataset

    def rank_rows(rank, ws):
        ds = load_dataset("parquet", data_files=str(cache / PRETOK_DATA), split="train", streaming=False)
        return [tuple(ex["input_ids"].tolist()) for ex in PretokenizedDataset(ds, rank, ws)]

    r0, r1 = rank_rows(0, 2), rank_rows(1, 2)
    assert set(r0).isdisjoint(set(r1)), "ranks must not share sequences"
    assert len(r0) + len(r1) == len(ROWS), "rank shards must cover every sequence"


@needs_tok
def test_cached_length_grouping_when_unpacked(tmp_path):
    """Non-packed cache re-applies length grouping: worst-case (alternating short/long)
    file order pads far more without it, and grouping loses no sequences."""
    p = tmp_path / "d.jsonl"
    rows = []
    for i in range(8):
        content = "A" if i % 2 == 0 else " ".join(["word"] * 80)  # alternate short / long
        rows.append({"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": content}]})
    _write(p, rows)
    cache = tmp_path / "cache"
    _materialize(_cfg(p, packing=False), cache)

    def padded_and_real(buf):
        cfg = _cfg(p, packing=False, length_group_buffer=buf)
        dl = build_pretokenized_dataloader(cache, TOK, cfg, 0, 1, 2)
        padded = real = nseq = 0
        for batch in dl:
            padded += batch["input_ids"].numel()
            real += int(batch["attention_mask"].sum())
            nseq += batch["input_ids"].size(0)
        return padded, real, nseq

    grouped_pad, grouped_real, grouped_n = padded_and_real(8)
    ungrouped_pad, ungrouped_real, ungrouped_n = padded_and_real(0)

    assert grouped_n == ungrouped_n == len(rows), "no sequences dropped"
    assert grouped_real == ungrouped_real, "grouping changes ORDER only, never token content"
    assert grouped_pad < ungrouped_pad, "length grouping must cut pad-token FLOPs"


@needs_tok
def test_empty_stream_raises(tmp_path):
    cache = tmp_path / "cache"

    def factory():
        return iter(())

    with pytest.raises(ValueError, match="0 sequences"):
        materialize_pretokenized(factory, cache, "fp", _Wrap(_cfg(tmp_path / "x.jsonl")), TOK)


# ── config guards ────────────────────────────────────────────────────────────────
def test_validate_rejects_pretokenize_plus_msft():
    from palingenesis.config import Config, ConfigError

    cfg = Config()
    cfg.data.pretokenize = True
    cfg.data.sources = [{"dataset": "a"}, {"dataset": "b"}]
    cfg.data.msft_tracking = True
    with pytest.raises(ConfigError, match="msft_tracking"):
        cfg.validate()


def test_validate_rejects_pretokenize_plus_seqlen_curriculum():
    from palingenesis.config import Config, ConfigError

    cfg = Config()
    cfg.data.pretokenize = True
    cfg.data.seq_len_curriculum = True
    with pytest.raises(ConfigError, match="seq_len_curriculum"):
        cfg.validate()


def test_validate_allows_plain_pretokenize():
    from palingenesis.config import Config

    cfg = Config()
    cfg.data.pretokenize = True
    cfg.validate()  # must not raise
