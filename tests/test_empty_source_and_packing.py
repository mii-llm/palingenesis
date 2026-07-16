"""Regression tests for the silent 'training does 0 steps' bug.

Two independent failure modes combined to zero out a run:
  1. MixedDataset broke the whole epoch the first time ANY source raised
     StopIteration — so a single empty source (wrong field/format/split, or a
     replay file with no `text`) ended the epoch after a handful of items.
  2. PackedDataset discarded the sub-max_len remainder, so a short stream produced
     0 packed blocks → empty DataLoader → 0 optimizer steps.

These tests operate directly on the dataset classes (no tokenizer/model needed).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from palingenesis.data import MixedDataset, PackedDataset  # noqa: E402


def _seq(toks):
    t = torch.tensor(toks, dtype=torch.long)
    return {"input_ids": t, "labels": t.clone(), "attention_mask": torch.ones(len(toks), dtype=torch.long)}


# ── MixedDataset: an empty source is dropped, not fatal ──────────────────────────
def test_empty_source_is_dropped_not_fatal(caplog):
    non_empty = [_seq([i, i]) for i in range(5)]
    empty: list = []
    mixed = MixedDataset([non_empty, empty], [0.5, 0.5], seed=0, names=["good", "BAD_EMPTY"])

    import logging

    with caplog.at_level(logging.WARNING):
        out = list(mixed)

    got = sorted(int(x["input_ids"][0]) for x in out)
    assert got == [0, 1, 2, 3, 4], "all items from the non-empty source must survive the empty one"
    assert any("BAD_EMPTY" in r.message and "0 usable" in r.message for r in caplog.records), \
        "the empty source must be named in a loud warning"


def test_all_empty_sources_yields_nothing_without_hanging():
    mixed = MixedDataset([[], []], [0.5, 0.5], seed=0, names=["a", "b"])
    assert list(mixed) == []


def test_healthy_mix_still_terminates_on_real_exhaustion():
    a = [_seq([1]) for _ in range(3)]
    b = [_seq([2]) for _ in range(3)]
    out = list(MixedDataset([a, b], [0.5, 0.5], seed=1))
    # Stops when the first NON-empty source exhausts → bounded, non-empty.
    assert 1 <= len(out) <= 6


# ── PackedDataset: trailing remainder is emitted, never dropped ──────────────────
def test_packing_emits_trailing_partial_sorted():
    base = [_seq([1, 2]), _seq([3, 4]), _seq([5, 6])]  # 6 tokens, max_len 4
    blocks = list(PackedDataset(base, max_len=4, eos_id=0, sort_buffer=256))
    all_ids = [t for blk in blocks for t in blk["input_ids"].tolist()]
    assert sorted(all_ids) == [1, 2, 3, 4, 5, 6], "no tokens may be dropped"
    assert blocks[0]["input_ids"].numel() == 4, "first block is a full max_len block"
    assert blocks[-1]["input_ids"].numel() == 2, "trailing remainder is emitted as a partial block"


def test_packing_short_stream_still_yields_one_block():
    """The exact 0-steps trigger: a stream shorter than one packed block used to
    yield NOTHING. It must now yield a single partial block."""
    base = [_seq([1, 2, 3]), _seq([4, 5])]  # 5 tokens < max_len 64
    blocks = list(PackedDataset(base, max_len=64, eos_id=0, sort_buffer=256))
    assert len(blocks) == 1
    assert sorted(blocks[0]["input_ids"].tolist()) == [1, 2, 3, 4, 5]
    # position_ids reset per document (sorted: [4,5] then [1,2,3]).
    assert blocks[0]["position_ids"].tolist() == [0, 1, 0, 1, 2]


def test_packing_sequential_short_stream_yields_partial():
    base = [_seq([1, 2, 3]), _seq([4, 5])]
    blocks = list(PackedDataset(base, max_len=64, eos_id=0, sort_buffer=0))
    assert len(blocks) == 1
    assert blocks[0]["input_ids"].tolist() == [1, 2, 3, 4, 5], "sequential preserves arrival order"


def test_packing_empty_base_yields_nothing():
    assert list(PackedDataset([], max_len=64, eos_id=0, sort_buffer=256)) == []


def test_packing_carries_remainder_across_buffer_flushes():
    """With a small sort_buffer the remainder must carry across flushes (no token loss
    and no spurious partial block per flush)."""
    base = [_seq([i, i + 100]) for i in range(10)]  # 20 tokens, buffer flushes every 3
    blocks = list(PackedDataset(base, max_len=4, eos_id=0, sort_buffer=3))
    all_ids = [t for blk in blocks for t in blk["input_ids"].tolist()]
    assert len(all_ids) == 20, "every token is preserved across flushes"
    # 20 tokens / 4 = exactly 5 full blocks, no partial.
    assert [b["input_ids"].numel() for b in blocks] == [4, 4, 4, 4, 4]
