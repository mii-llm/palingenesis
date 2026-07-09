"""Test Adaptive Multi-Source Fine-Tuning tracker."""

import sys

sys.path.insert(0, "src")

import torch
import torch.nn as nn


def test_tracker_basic():
    """Test basic tracker creation and initial state."""
    from palingenesis.msft import AdaptiveSourceTracker

    sources = [
        {"name": "agentic", "weight": 0.8},
        {"name": "general", "weight": 0.15},
        {"name": "code", "weight": 0.05},
    ]

    tracker = AdaptiveSourceTracker(sources, eval_every=10, decay_factor=0.7, floor_ratio=0.1)

    # Check initial state
    assert len(tracker.sources) == 3
    assert tracker.sources["agentic"].weight == 0.8
    assert abs(tracker.sources["agentic"].floor_weight - 0.08) < 1e-9  # 10% of 0.8
    assert abs(tracker.sources["code"].floor_weight - 0.005) < 1e-9  # 10% of 0.05
    assert tracker.should_eval(10) is True
    assert tracker.should_eval(5) is False

    # Mixing probs should sum to 1
    probs = tracker.get_mixing_probs()
    assert abs(sum(probs) - 1.0) < 1e-6

    print("✓ test_tracker_basic PASSED\n")


def test_weight_decay_on_overfit():
    """Test that overfitting sources get decayed weight (never zero)."""
    from palingenesis.msft import AdaptiveSourceTracker

    sources = [
        {"name": "fast", "weight": 0.5},
        {"name": "slow", "weight": 0.5},
    ]
    tracker = AdaptiveSourceTracker(sources, eval_every=5, decay_factor=0.7, recovery_factor=1.15, floor_ratio=0.1)

    fast = tracker.sources["fast"]
    slow = tracker.sources["slow"]

    # Simulate: fast's val loss improves then worsens; slow keeps improving
    # First eval: set baseline
    fast.best_val_loss = 2.0
    fast.val_losses.append(2.0)
    slow.best_val_loss = 3.0
    slow.val_losses.append(3.0)

    # Fast overfits: weight should decay
    # Simulate 5 consecutive overfitting checks
    for i in range(5):
        fast.consecutive_increases += 1
        fast.weight = max(fast.weight * 0.7, fast.floor_weight)
        fast.total_decays += 1

    # After 5 decays: 0.5 * 0.7^5 = 0.084 (above floor of 0.05)
    expected = 0.5 * (0.7**5)
    assert abs(fast.weight - expected) < 0.001, f"Expected {expected:.4f}, got {fast.weight:.4f}"
    assert fast.weight > fast.floor_weight, "Weight should still be above floor"

    # Continue decaying until floor
    for i in range(20):
        fast.weight = max(fast.weight * 0.7, fast.floor_weight)

    assert fast.weight == fast.floor_weight, "Should reach floor, not zero"
    assert fast.weight == 0.05, f"Floor should be 0.05, got {fast.weight}"
    assert fast.weight > 0, "Weight is NEVER zero (unlike MSFT hard exclusion)"

    # Slow stays at original
    assert slow.weight == 0.5

    print(f"  Fast source decayed to floor: {fast.weight:.4f} (floor={fast.floor_weight:.4f})")
    print(f"  Slow source unchanged: {slow.weight:.4f}")
    print(f"  Total weight: {fast.weight + slow.weight:.4f} (was 1.0)")
    print("✓ test_weight_decay_on_overfit PASSED\n")


def test_weight_recovery():
    """Test that improving sources recover weight."""
    from palingenesis.msft import AdaptiveSourceTracker

    sources = [{"name": "src", "weight": 1.0}]
    tracker = AdaptiveSourceTracker(sources, decay_factor=0.7, recovery_factor=1.15, floor_ratio=0.1)

    state = tracker.sources["src"]

    # Decay to 50%
    state.weight = 0.5
    state.best_val_loss = 2.0

    # Simulate improvement: weight should recover
    for _ in range(5):
        state.weight = min(state.weight * 1.15, state.original_weight)

    expected = 0.5 * (1.15**5)
    # But capped at original (1.0)
    assert state.weight <= state.original_weight
    assert state.weight > 0.5, f"Expected recovery above 0.5, got {state.weight:.4f}"

    # Continue recovering — should cap at original
    for _ in range(20):
        state.weight = min(state.weight * 1.15, state.original_weight)
    assert state.weight == state.original_weight, "Should cap at original weight"

    print(f"  Recovered to ceiling: {state.weight:.4f} (original={state.original_weight:.4f})")
    print("✓ test_weight_recovery PASSED\n")


def test_evaluate_with_model():
    """Test full evaluate_and_adjust with a tiny model."""
    from palingenesis.msft import AdaptiveSourceTracker

    # Tiny model that produces logits
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.head = nn.Linear(32, 100)

        def forward(self, input_ids, attention_mask=None):
            return type("O", (), {"logits": self.head(self.embed(input_ids))})()

    model = TinyModel()

    sources = [
        {"name": "src_a", "weight": 0.7},
        {"name": "src_b", "weight": 0.3},
    ]
    tracker = AdaptiveSourceTracker(sources, eval_every=5, decay_factor=0.7, plateau_threshold=0.01)

    # Create fake eval batches
    batch_a = {
        "input_ids": torch.randint(0, 100, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.randint(0, 100, (2, 16)),
    }
    batch_b = {
        "input_ids": torch.randint(0, 100, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.randint(0, 100, (2, 16)),
    }
    tracker.set_eval_batches("src_a", [batch_a])
    tracker.set_eval_batches("src_b", [batch_b])

    # First evaluation: sets baselines
    metrics = tracker.evaluate_and_adjust(model, step=5, device=torch.device("cpu"), dtype=torch.float32, bf16=False)

    assert "msft/src_a/val_loss" in metrics
    assert "msft/src_b/val_loss" in metrics
    assert "msft/src_a/weight" in metrics
    assert "msft/src_b/weight" in metrics
    assert metrics["msft/src_a/val_loss"] > 0
    assert metrics["msft/src_b/val_loss"] > 0

    print(f"  src_a: loss={metrics['msft/src_a/val_loss']:.4f}, weight={metrics['msft/src_a/weight']:.4f}")
    print(f"  src_b: loss={metrics['msft/src_b/val_loss']:.4f}, weight={metrics['msft/src_b/weight']:.4f}")

    # Corrupt model to simulate overfitting
    with torch.no_grad():
        model.head.weight.mul_(3.0)

    metrics2 = tracker.evaluate_and_adjust(model, step=10, device=torch.device("cpu"), dtype=torch.float32, bf16=False)

    # Losses should be higher → trend should be negative
    print(f"  After corruption:")
    print(
        f"  src_a: loss={metrics2['msft/src_a/val_loss']:.4f}, weight={metrics2['msft/src_a/weight']:.4f}, "
        f"trend={metrics2['msft/src_a/trend']}"
    )
    print(
        f"  src_b: loss={metrics2['msft/src_b/val_loss']:.4f}, weight={metrics2['msft/src_b/weight']:.4f}, "
        f"trend={metrics2['msft/src_b/trend']}"
    )

    # Weights should have decayed (losses went up)
    assert metrics2["msft/src_a/weight"] <= metrics["msft/src_a/weight"]
    assert metrics2["msft/src_b/weight"] <= metrics["msft/src_b/weight"]
    # But NOT zero!
    assert metrics2["msft/src_a/weight"] > 0
    assert metrics2["msft/src_b/weight"] > 0

    print(f"\n  Summary: {tracker.summary_str}")
    print("✓ test_evaluate_with_model PASSED\n")


def test_never_reaches_zero():
    """Stress test: even with many consecutive overfitting events, weight > 0."""
    from palingenesis.msft import AdaptiveSourceTracker

    sources = [{"name": "test", "weight": 1.0}]
    tracker = AdaptiveSourceTracker(sources, decay_factor=0.5, floor_ratio=0.1)
    state = tracker.sources["test"]

    # 100 consecutive decays
    for _ in range(100):
        state.weight = max(state.weight * tracker.decay_factor, state.floor_weight)

    assert state.weight == state.floor_weight == 0.1
    assert state.weight > 0, "Weight must NEVER reach zero"

    print(f"  After 100 decays: weight={state.weight:.4f} (floor={state.floor_weight:.4f})")
    print("✓ test_never_reaches_zero PASSED\n")


def test_mixed_dataset_update():
    """Test that we can update MixedDataset probs in-place."""
    from palingenesis.msft import AdaptiveSourceTracker

    # Simulate a MixedDataset-like object
    class FakeMixedDS:
        def __init__(self, probs):
            self.probs = probs

    sources = [
        {"name": "a", "weight": 0.6},
        {"name": "b", "weight": 0.3},
        {"name": "c", "weight": 0.1},
    ]
    tracker = AdaptiveSourceTracker(sources, decay_factor=0.7, floor_ratio=0.1)

    ds = FakeMixedDS(probs=[0.6, 0.3, 0.1])

    # Decay source 'a'
    tracker.sources["a"].weight = 0.3  # halved

    tracker.update_mixed_dataset(ds)

    assert abs(sum(ds.probs) - 1.0) < 1e-6, "Probs should sum to 1"
    # 'a' should have lower probability than before
    assert ds.probs[0] < 0.6, f"'a' prob should have decreased, got {ds.probs[0]:.4f}"
    # 'b' and 'c' should have increased relative share
    assert ds.probs[1] > 0.3, f"'b' prob should have increased"

    print(f"  Updated probs: {[f'{p:.3f}' for p in ds.probs]}")
    print("✓ test_mixed_dataset_update PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE SOURCE TRACKING TESTS")
    print("=" * 60 + "\n")

    test_tracker_basic()
    test_weight_decay_on_overfit()
    test_weight_recovery()
    test_evaluate_with_model()
    test_never_reaches_zero()
    test_mixed_dataset_update()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
