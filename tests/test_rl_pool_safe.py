# ruff: noqa: F811
"""Fail-safe weight pushes and engine pools that survive a lost worker (palingenesis.rl.pool): checksummed buckets,
pushes rejected whole and sent again, the weights an engine had restored when applying a push fails half-way, the
two-rank link per engine (over torch.distributed's Gloo backend here; NCCL is the same protocol on GPU tensors) and
its fall back to TCP, a pool that goes on without a lost worker and takes it back, and a run that uses all of it
with the decoupled objective."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy  # noqa: E402
from test_rl_pool import (  # noqa: E402, F401
    KWARGS,
    SCRIPTED,
    WireEngine,
    _train,
    gate,
    generate,
    wait_for,
    workers,
)

from palingenesis.rl.pool import buckets as bucketed  # noqa: E402
from palingenesis.rl.pool.client import RemotePool  # noqa: E402


class PartsEngine(WireEngine):
    """Loads a push one tensor at a time, and fails after the first when the file named by PGS_TEST_BREAK holds the
    push's version: a completion's second token is the sum of the weights the engine holds, whichever push they
    came from."""

    def __init__(self):
        super().__init__()
        self.weights: dict[str, float] = {}

    def update_weights(self, named, version):
        broken = os.environ.get("PGS_TEST_BREAK", "")
        for i, (name, tensor) in enumerate(named):
            if i == 1 and broken and os.path.exists(broken) and Path(broken).read_text() == str(version):
                self.total = sum(self.weights.values())
                raise OSError("the device ran out of memory")
            self.weights[name] = float(tensor.double().sum())
        self.total = sum(self.weights.values())
        self.version = version


def parts_engine():
    return PartsEngine()


PARTS = ("--engine", f"{__file__}:parts_engine")


def weights(value: float) -> dict[str, torch.Tensor]:
    return {"a": torch.full((3,), value), "b": torch.full((5, 2), value), "c": torch.tensor([7], dtype=torch.int64)}


def total(value: float) -> int:
    return int(round(13 * value + 7))


# ------------------------------------------------------------------- buckets


def test_buckets_hold_every_tensor_at_an_aligned_offset_and_give_it_back_bit_for_bit():
    torch.manual_seed(0)
    named = {
        "w": torch.randn(7, 5),
        "odd": torch.randn(3).to(torch.bfloat16),
        "ids": torch.arange(5, dtype=torch.int32),
        "big": torch.randn(64, 9),
        "none": torch.empty(0, 4),
        "flag": torch.tensor([True, False, True]),
    }
    made = []

    def allocate(index, nbytes):
        made.append(nbytes)
        return torch.full((nbytes,), 0xAB, dtype=torch.uint8)  # (stale bytes of an earlier snapshot in the gaps)

    snapshot = bucketed.pack(iter(named.items()), 256, allocate)
    spec = snapshot.spec("tcp")
    assert len(snapshot.buffers) == 3 and made == spec["sizes"]  # w + odd + ids | big (alone: larger) | none + flag
    assert all(offset % bucketed.ALIGN == 0 for *_, offset in spec["manifest"])
    assert bucketed.verify(spec, snapshot.buffers) == ""
    back = bucketed.unpack(spec, snapshot.buffers)
    assert list(back) == list(named)
    for name, tensor in named.items():
        assert back[name].dtype == tensor.dtype and back[name].shape == tensor.shape and torch.equal(back[name], tensor)
    # the wire dtype: float tensors are cast as they are copied, the others travel as they are
    cast = bucketed.pack(named.items(), 1 << 20, lambda i, n: torch.empty(n, dtype=torch.uint8), torch.bfloat16)
    back = bucketed.unpack(cast.spec("tcp"), cast.buffers)
    assert back["w"].dtype == torch.bfloat16 and torch.equal(back["w"], named["w"].to(torch.bfloat16))
    assert back["ids"].dtype == torch.int32 and cast.nbytes < snapshot.nbytes


def test_a_checksum_sees_one_flipped_bit_and_two_swapped_words():
    torch.manual_seed(1)
    buffer = torch.randint(0, 256, (8 * 1000 + 5,), dtype=torch.uint8)
    reference = bucketed.checksum(buffer).tolist()
    assert bucketed.checksum(buffer.clone()).tolist() == reference
    for position in (0, 4003, 8 * 1000 + 4):  # the first word, the middle, the tail that is no whole word
        flipped = buffer.clone()
        flipped[position] ^= 1
        assert bucketed.checksum(flipped).tolist() != reference
    swapped = buffer.clone()
    swapped[0:8], swapped[8:16] = buffer[8:16], buffer[0:8]
    assert bucketed.checksum(swapped).tolist()[0] == reference[0]  # the plain sum cannot tell
    assert bucketed.checksum(swapped).tolist()[1] != reference[1]
    spec = {"sizes": [buffer.numel()], "sums": [reference], "manifest": []}
    assert "another checksum" in bucketed.verify(spec, [flipped])
    assert "bytes" in bucketed.verify(spec, [buffer[:-1]]) and "buckets arrived" in bucketed.verify(spec, [])


# ---------------------------------------------------------------------- links


def test_a_link_carries_buffers_between_its_two_ends_and_says_when_it_cannot():
    """Two ends of a Gloo link in one process (the worker's end on a thread): the handshake, buckets of several
    sizes, and a transfer whose other end never comes."""
    store = bucketed.Link.host_store(20)
    prefix = bucketed.Link.new_prefix()
    sent = [torch.arange(n, dtype=torch.int64).view(torch.uint8).clone() for n in (1, 1000, 70_000)]
    got, errors = [], []

    def worker_end():
        try:
            link = bucketed.Link("gloo", 1, ("127.0.0.1", store.port), prefix, 20)
            link.handshake()
            for buffer in sent:
                into = torch.zeros_like(buffer)
                link.receive(into)
                got.append(into)
            link.close()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=worker_end, daemon=True)
    thread.start()
    link = bucketed.Link("gloo", 0, store, prefix, 20)
    link.handshake()
    for buffer in sent:
        link.send(buffer)
    thread.join(30)
    assert not errors and all(torch.equal(a, b) for a, b in zip(got, sent)) and len(got) == 3
    link.close()
    with pytest.raises(bucketed.LinkError, match="transfer failed"):
        link.send(sent[0])  # a closed link
    start = time.monotonic()
    with pytest.raises(bucketed.LinkError, match="could not be built"):
        bucketed.Link("gloo", 0, store, bucketed.Link.new_prefix(), 1.0)  # nobody joins
    assert time.monotonic() - start < 30


# ----------------------------------------------------------- verified pushes


def test_a_push_that_arrives_corrupt_is_rejected_whole_and_the_next_one_is_applied(workers, tmp_path):
    corrupt = tmp_path / "corrupt"
    pool = RemotePool([workers(*SCRIPTED, env={"PGS_TEST_CORRUPT": str(corrupt)})], pin=False, verify=True)
    try:
        pool.update_weights(weights(1.0).items(), 1)
        assert pool.applied() == [1] and generate(pool) == ([1, total(1.0), 0], 1)
        corrupt.write_text("")
        pool.update_weights(weights(2.0).items(), 2)  # sent twice, rejected twice: the engine keeps version 1
        assert wait_for(lambda: pool.stats["rejected"] == 2 and pool.sending == [0, 0])
        assert not pool.failure and pool.applied() == [1] and generate(pool) == ([1, total(1.0), 0], 1)
        state = pool.state()[0]
        assert (state["version"], state["rejected"], state["confirmed"]) == (1, 2, True)
        assert pool.metrics()["pool/lag"] == 1 and pool.metrics()["pool/rejected"] == 2
        corrupt.unlink()
        pool.update_weights(weights(3.0).items(), 3)
        assert wait_for(lambda: pool.applied() == [3]) and generate(pool) == ([3, total(3.0), 0], 3)
        assert pool.targets[0].rejections == 0
        # an engine that rejects push after push ends the run with the worker's reason
        corrupt.write_text("")
        for version in (4, 5, 6):
            pool.update_weights(weights(4.0).items(), version)
            assert wait_for(lambda: pool.sending == [0, 0])
        with pytest.raises(RuntimeError, match="rejected 3 time.*another checksum"):
            pool.update_weights(weights(4.0).items(), 7)
        assert generate(pool) == ([3, total(3.0), 0], 3)  # still whole, still on the last push it applied
    finally:
        pool.close(shutdown=True)


def test_a_push_older_than_the_engine_s_version_is_rejected(workers):
    pool = RemotePool([workers(*SCRIPTED)], pin=False, verify=True)
    try:
        pool.update_weights(weights(1.0).items(), 5)
        pool.update_weights(weights(2.0).items(), 3)
        assert wait_for(lambda: pool.stats["rejected"] == 2 and pool.sending == [0, 0])
        assert "older than the engine's" in pool.state()[0].get("error", "older than the engine's")
        assert generate(pool) == ([5, total(1.0), 0], 5)
    finally:
        pool.close(shutdown=True)


def test_an_engine_that_fails_half_way_through_a_push_gets_the_weights_it_had_back(workers, tmp_path):
    broken = tmp_path / "break"
    pool = RemotePool([workers(*PARTS, env={"PGS_TEST_BREAK": str(broken)})], pin=False, verify=True)
    try:
        pool.update_weights(weights(1.0).items(), 1)
        assert generate(pool) == ([1, total(1.0), 0], 1)
        broken.write_text("2")
        pool.update_weights(weights(2.0).items(), 2)  # "a" is loaded, then the engine fails
        assert wait_for(lambda: pool.stats["rejected"] == 2 and pool.sending == [0, 0])
        state = pool.state()[0]
        assert (state["version"], state["restored"], state["confirmed"]) == (1, 2, True)
        assert generate(pool) == ([1, total(1.0), 0], 1)  # not 3 x 2 + 10 x 1 + 7: every weight is version 1's
        assert not pool.failure
        broken.unlink()
        pool.update_weights(weights(3.0).items(), 3)
        assert wait_for(lambda: pool.applied() == [3]) and generate(pool) == ([3, total(3.0), 0], 3)
    finally:
        pool.close(shutdown=True)
    # a first push that fails half-way: there is nothing to restore, so nothing the engine samples is trained
    broken.write_text("1")
    pool = RemotePool([workers(*PARTS, env={"PGS_TEST_BREAK": str(broken)})], pin=False, verify=True)
    try:
        with pytest.raises(RuntimeError, match="holds no whole weights"):
            pool.update_weights(weights(1.0).items(), 1)
        assert pool.state()[0]["confirmed"] is False and generate(pool)[1] == -1
    finally:
        pool.close(shutdown=True)


def test_weights_travel_over_a_link_per_engine_and_over_tcp_when_a_link_fails(workers):
    """rollout.push_link gloo: buckets cast to bf16 on the trainer's side go to each engine over its own group. One
    engine's link breaking moves that engine to TCP with the push that failed; the other keeps its link."""
    engines = [workers(*SCRIPTED), workers(*SCRIPTED)]
    pool = RemotePool(engines, pin=False, link="gloo", bucket_mb=1, wire_dtype="bfloat16", link_timeout=4.0)
    try:
        assert all(e.link is not None for e in pool.targets) and all(s["linked"] for s in pool.state())
        pool.update_weights(weights(1.5).items(), 1)
        assert pool.applied() == [1, 1] and pool.stats["link_pushes"] == 2
        assert generate(pool) == ([1, total(1.5), 0], 1)
        pool.targets[0].link.close()  # this end of the first engine's link is gone: its worker waits, then says so
        pool.update_weights(weights(2.5).items(), 2)
        assert wait_for(lambda: pool.applied() == [2, 2], 60), (pool.applied(), pool.failure)
        assert pool.stats["link_failures"] == 1 and pool.targets[0].link is None and pool.targets[1].link is not None
        assert not pool.failure and pool.stats["rejected"] == 0
        assert [s["linked"] for s in pool.state()] == [False, True]
        pool.update_weights(weights(3.5).items(), 3)
        assert wait_for(lambda: pool.applied() == [3, 3]) and pool.stats["link_pushes"] == 4
        assert {tuple(generate(pool)[0]) for _ in range(4)} == {(3, total(3.5), 0)}
        metrics = pool.metrics()
        assert metrics["pool/link_failures"] == 1 and metrics["pool/push_mb_s"] > 0
    finally:
        pool.close(shutdown=True)


def test_a_worker_without_the_link_backend_is_served_over_tcp(workers, monkeypatch):
    address = workers(*SCRIPTED)
    real = bucketed.Link

    class NoLink(real):
        def __init__(self, *args, **kwargs):
            raise bucketed.LinkError("nccl link (rank 0) could not be built: no GPU")

    monkeypatch.setattr(bucketed, "Link", NoLink)
    pool = RemotePool([address], pin=False, link="gloo", link_timeout=2.0)
    try:
        assert pool.targets[0].link is None and pool.stats["link_failures"] == 1
        pool.update_weights(weights(1.0).items(), 1)
        assert pool.applied() == [1] and generate(pool) == ([1, total(1.0), 0], 1)
    finally:
        pool.close(shutdown=True)


# ------------------------------------------------------- losing an engine


def test_a_pool_goes_on_without_a_lost_worker_and_takes_a_restarted_one_back(workers):
    """rollout.min_engines 1 of 2: the run survives a worker's death (its requests go to the other engine, updates
    go on), and a worker restarted on the same address rejoins: it generates again once a push has reached it."""
    doomed, other = workers(*SCRIPTED), workers(*SCRIPTED)
    pool = RemotePool([doomed, other], pin=False, verify=True, min_engines=1, reconnect_s=0.2)
    try:
        pool.update_weights(weights(1.0).items(), 1)
        workers.processes[doomed].kill()
        assert wait_for(lambda: bool(pool.engines[0].dead))
        assert {generate(pool)[1] for _ in range(4)} == {1} and not pool.failure
        pool.update_weights(weights(2.0).items(), 2)  # no error: one engine is enough
        assert wait_for(lambda: pool.targets[1].applied == 2)
        assert pool.metrics()["pool/engines"] == 1 and pool.metrics()["pool/lag"] == 0
        workers("--port", doomed.rpartition(":")[2], *SCRIPTED)
        assert wait_for(lambda: not pool.engines[0].dead, 60) and pool.stats["rejoined"] == 1
        # it holds none of this run's weights: no request goes to it (what it sampled would be version -1)
        assert {generate(pool)[1] for _ in range(6)} == {2}
        assert [s["confirmed"] for s in pool.state()] == [False, True]
        pool.update_weights(weights(3.0).items(), 3)
        assert wait_for(lambda: pool.applied() == [3, 3])
        assert {tuple(generate(pool)[0]) for _ in range(6)} == {(3, total(3.0), 0)}
        # below min_engines the run ends as before
        workers.processes[other].kill()
        workers.processes[doomed].kill()
        assert wait_for(lambda: bool(pool.failure))
        with pytest.raises(RuntimeError, match="went away"):
            pool.update_weights(weights(4.0).items(), 4)
    finally:
        pool.close()


# ----------------------------------------------------------------- end to end


def test_a_run_with_the_decoupled_objective_verified_link_pushes_and_weights_synced_at_publish(tmp_path, workers):
    """Everything together on two hf engine workers: loss.type decoupled clipped against the reference policy,
    staleness 2 with stale tokens masked, weights pushed when the trainer publishes them, over a Gloo link per
    engine with checksums. The engines sample the trained policy (the mismatch of current-version tokens is
    numerical) and the trainer never waits for a push."""
    policy = tiny_policy(tmp_path)
    engines = [workers("--model", policy, "--backend", "hf", *KWARGS) for _ in range(2)]
    options = {
        "loss.type": "decoupled",
        "loss.proximal": "reference",
        "loss.kl_ref": "start",
        "loss.kl_ref_reset": 2,
        "rollout.max_staleness": 2,
        "rollout.stale": "mask",
        "rollout.sync": "publish",
        "rollout.push_link": "gloo",
        "rollout.push_bucket_mb": 1,
        "rollout.min_engines": 1,
        "train.steps": 5,
    }
    trainer, steps = _train(tmp_path, engines, **options)
    assert len(steps) == 5 and any(m.get("trained_sequences", 0) > 0 for m in steps)
    last = steps[-1]
    assert last["pool/link_pushes"] >= 2 and last["pool/link_failures"] == 0 and last["pool/rejected"] == 0
    assert all("time/sync_publish" in m for m in steps) and all(m["staleness"] <= 2 for m in steps)
    trained = [m for m in steps if m.get("trained_sequences", 0) > 0]
    assert all("policy/behaviour_weight" in m and "policy/proximal_dev" in m for m in trained)
    current = [m["policy/mismatch_k3/current"] for m in steps if "policy/mismatch_k3/current" in m]
    assert current and max(current) < 1e-3
