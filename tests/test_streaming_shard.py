"""Streaming shard/shuffle order.

`shuffle().shard()` on a streaming dataset leaves every DataLoader worker
except the first with an empty shard list (datasets 5.x) — the whole run dies
on the first batch with num_workers > 1. The pipeline must shard per worker
FIRST and buffer-shuffle after (data._shard_then_shuffle).
"""

import sys

sys.path.insert(0, "src")


class FakeStream:
    """Records the order of shard/shuffle calls."""

    def __init__(self):
        self.log = []

    def shard(self, num_shards, index):
        self.log.append(("shard", num_shards, index))
        return self

    def shuffle(self, seed, buffer_size):
        self.log.append(("shuffle", seed, buffer_size))
        return self

    def __iter__(self):
        return iter(())


def test_shard_happens_before_shuffle():
    from palingenesis.data import _shard_then_shuffle

    ds = FakeStream()
    _shard_then_shuffle(ds, rank=1, world_size=2, shuffle_buffer=100, shuffle_seed=7)
    assert ds.log == [("shard", 2, 1), ("shuffle", 7, 100)]


def test_no_shuffle_when_buffer_zero():
    """Curriculum-ordered (preserve_order) data must pass through unshuffled."""
    from palingenesis.data import _shard_then_shuffle

    ds = FakeStream()
    _shard_then_shuffle(ds, rank=0, world_size=2, shuffle_buffer=0, shuffle_seed=7)
    assert ds.log == [("shard", 2, 0)]


def test_single_process_still_shuffles():
    from palingenesis.data import _shard_then_shuffle

    ds = FakeStream()
    _shard_then_shuffle(ds, rank=0, world_size=1, shuffle_buffer=100, shuffle_seed=7)
    assert ds.log == [("shuffle", 7, 100)]  # no shard needed, shuffle still applies


def test_chat_dataset_wires_shuffle_after_shard():
    from palingenesis.data import ChatDataset

    ds = ChatDataset(FakeStream(), tokenizer=None, max_seq_length=64,
                     rank=0, world_size=2, shuffle_buffer=50, shuffle_seed=3)
    iterator = iter(ds)
    next(iterator, None)  # drives __iter__ far enough to apply shard+shuffle
    assert ds.dataset.log == [("shard", 2, 0), ("shuffle", 3, 50)]
