"""Tests for the padding/throughput fixes.

Covers:
- LengthGroupedDataset: batch-aligned groups of similar lengths, no sample lost
- _collate_fn pad_to_multiple: shapes rounded up, padding correctly masked
- _dynamic_num_chunks: chunk count follows the ACTUAL batch size, not max_seq_length
- _grad_norm: vectorized version matches the naive per-param computation
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _fake_sample(n: int) -> dict:
    return {
        "input_ids": torch.arange(n),
        "attention_mask": torch.ones(n, dtype=torch.long),
        "labels": torch.arange(n),
    }


def test_length_grouped_dataset_groups_similar_lengths():
    from palingenesis.data import LengthGroupedDataset

    # Interleaved short/long stream; buffer 8, batch 4 → after sorting each
    # buffer must split into a short group {1,2,3,4} and a long group {97..100}
    lengths = [1, 100, 2, 99, 3, 98, 4, 97]
    ds = LengthGroupedDataset(iter([_fake_sample(n) for n in lengths]), batch_size=4, buffer_size=8, seed=0)
    out = [s["input_ids"].size(0) for s in ds]

    assert sorted(out) == sorted(lengths), "Every sample must be yielded exactly once"
    groups = [set(out[i : i + 4]) for i in range(0, 8, 4)]
    assert {1, 2, 3, 4} in groups and {97, 98, 99, 100} in groups, (
        f"Expected length-homogeneous groups, got {groups}"
    )
    print("✓ test_length_grouped_dataset_groups_similar_lengths PASSED")


def test_length_grouped_dataset_partial_tail_stays_last():
    from palingenesis.data import LengthGroupedDataset

    # 10 samples, batch 4 → 2 full groups + tail of 2. The tail must be
    # yielded LAST so DataLoader batch boundaries stay aligned (drop_last
    # then discards it), and full groups must be contiguous slices of the
    # sorted buffer.
    lengths = [5, 50, 6, 49, 7, 48, 8, 47, 9, 46]
    ds = LengthGroupedDataset(iter([_fake_sample(n) for n in lengths]), batch_size=4, buffer_size=16, seed=1)
    out = [s["input_ids"].size(0) for s in ds]

    assert sorted(out) == sorted(lengths)
    ordered = sorted(lengths)  # [5,6,7,8,9,46,47,48,49,50]
    full_groups = {frozenset(out[0:4]), frozenset(out[4:8])}
    expected = {frozenset(ordered[0:4]), frozenset(ordered[4:8])}
    assert full_groups == expected, f"Full groups {full_groups} aren't sorted-buffer slices {expected}"
    assert set(out[8:]) == set(ordered[8:]), f"Partial tail {out[8:]} must be the last sorted slice"
    print("✓ test_length_grouped_dataset_partial_tail_stays_last PASSED")


def test_collate_pad_to_multiple():
    from palingenesis.data import IGNORE_INDEX, _collate_fn

    batch = [_fake_sample(5), _fake_sample(7)]
    out = _collate_fn(batch, pad_id=0, pad_to_multiple=64)

    assert out["input_ids"].shape == (2, 64)
    assert out["attention_mask"].shape == (2, 64)
    # Real tokens preserved, padding masked out of both attention and loss
    assert out["attention_mask"][0].sum() == 5 and out["attention_mask"][1].sum() == 7
    assert (out["labels"][0, 5:] == IGNORE_INDEX).all()
    assert (out["input_ids"][1, 7:] == 0).all()

    # Default (pad_to_multiple=1) keeps the old pad-to-longest behavior
    out_plain = _collate_fn(batch, pad_id=0)
    assert out_plain["input_ids"].shape == (2, 7)
    print("✓ test_collate_pad_to_multiple PASSED")


def test_dynamic_num_chunks_follows_batch():
    from palingenesis.train import _dynamic_num_chunks

    vocab = 152_064
    # Full 16×4096 batch: ~40GB fp32 logits → 20 chunks at 2GB target (was 64)
    assert _dynamic_num_chunks(16 * 4096, vocab) == 20
    # Length-grouped short batch 16×512: ~5GB → 3 chunks
    assert _dynamic_num_chunks(16 * 512, vocab) == 3
    # Tiny batch: no chunking at all
    assert _dynamic_num_chunks(2 * 128, vocab) == 1
    # Never explodes past 64
    assert _dynamic_num_chunks(64 * 32768, vocab) == 64
    print("✓ test_dynamic_num_chunks_follows_batch PASSED")


def test_grad_norm_vectorized_matches_naive():
    from palingenesis.train import _grad_norm

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
    model(torch.randn(3, 8)).sum().backward()

    naive = sum(p.grad.float().norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    fast = _grad_norm(model)
    assert abs(naive - fast) < 1e-5, f"naive={naive}, vectorized={fast}"

    # No grads → 0.0, not a crash
    model.zero_grad(set_to_none=True)
    assert _grad_norm(model) == 0.0
    print("✓ test_grad_norm_vectorized_matches_naive PASSED")


if __name__ == "__main__":
    test_length_grouped_dataset_groups_similar_lengths()
    test_length_grouped_dataset_partial_tail_stays_last()
    test_collate_pad_to_multiple()
    test_dynamic_num_chunks_follows_batch()
    test_grad_norm_vectorized_matches_naive()
    print("\nALL LENGTH-GROUPING / THROUGHPUT TESTS PASSED ✓")
