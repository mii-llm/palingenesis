"""OPD orchestrator: staleness gating, bounded queue, stale-batch drops, error propagation."""

import sys
import threading
import time

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")

from palingenesis.opd.orchestrator import Batch, Orchestrator, PublishedWeights  # noqa: E402


class FakePipeline:
    """Records the policy version each batch is generated with."""

    def __init__(self):
        self.weights = PublishedWeights(torch.nn.Linear(2, 2))
        self.generated: list[int] = []
        self.fail = False

    def run(self, requests, temperature):
        if self.fail:
            raise ValueError("teacher server gone")
        version = self.weights.version
        self.generated.append(version)
        return Batch(samples=list(requests), version=version)


def wait_until(condition, timeout=5.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def orchestrator(max_staleness):
    pipeline = FakePipeline()
    orch = Orchestrator(pipeline, lambda: ["request"], temperature=1.0, max_staleness=max_staleness)
    orch.start()
    return pipeline, orch


def test_on_policy_generates_each_batch_with_the_newest_weights():
    pipeline, orch = orchestrator(max_staleness=0)
    try:
        for step in range(4):
            batch = orch.next(pipeline.weights.version)
            assert batch.version == pipeline.weights.version == step
            time.sleep(0.05)  # "training": the producer must not run ahead
            assert pipeline.generated == list(range(step + 1))
            pipeline.weights.publish()
    finally:
        orch.stop()


def test_staleness_one_overlaps_one_batch():
    pipeline, orch = orchestrator(max_staleness=1)
    try:
        wait_until(lambda: len(pipeline.generated) == 2)
        time.sleep(0.05)
        assert pipeline.generated == [0, 0]  # batch 1 generated during step 0; batch 2 waits for version 1
        versions = []
        for _ in range(5):
            batch = orch.next(pipeline.weights.version)
            versions.append(pipeline.weights.version - batch.version)
            pipeline.weights.publish()
        assert max(versions) <= 1 and orch.dropped == 0
    finally:
        orch.stop()


def test_stale_batches_are_dropped_and_counted():
    pipeline = FakePipeline()
    orch = Orchestrator(pipeline, lambda: ["r"], temperature=1.0, max_staleness=1)
    orch.queue.put(Batch(samples=["a", "b", "c"], version=0))
    orch.queue.put(Batch(samples=["d"], version=4))
    batch = orch.next(version=5)
    assert batch.samples == ["d"] and orch.dropped == 3


def test_producer_errors_reach_the_trainer():
    pipeline = FakePipeline()
    pipeline.fail = True
    orch = Orchestrator(pipeline, lambda: ["r"], temperature=1.0, max_staleness=0)
    orch.start()
    with pytest.raises(RuntimeError, match="rollout producer failed") as e:
        orch.next(0)
    assert isinstance(e.value.__cause__, ValueError)
    orch.stop()


def test_weight_sync_waits_for_the_optimizer():
    """An engine copies the weights under the lock the trainer holds while stepping."""
    weights = PublishedWeights(torch.nn.Linear(2, 2))

    class Engine:
        version = 0

        def update_weights(self, named, version):
            self.seen = [(n, t.clone()) for n, t in named]
            self.version = version

    engine = Engine()
    weights.lock.acquire()  # the trainer is mid-step
    weights.publish()
    done = threading.Event()
    threading.Thread(target=lambda: (weights.sync(engine), done.set())).start()
    time.sleep(0.05)
    assert not done.is_set()
    weights.lock.release()
    assert done.wait(60) and engine.version == 1  # (the first sync imports transformers)
    assert [n for n, _ in engine.seen] == ["weight", "bias"]
    assert weights.sync(engine) == 0.0  # up to date: nothing to do
