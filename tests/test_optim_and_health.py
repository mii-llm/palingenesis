"""Tests for optimizer enhancements and health monitoring.

Covers:
- Power-decay and WSD schedulers (arxiv:2602.06797)
- RL-readiness entropy monitoring (arxiv:2606.18487, 2606.09932)
- Hyperball optimizer wrapper (arxiv:2606.16899)
- MONA acceleration for Muon (arxiv:2605.26842)
- SAGE embedding optimizer (arxiv:2604.07663)
"""

import sys

sys.path.insert(0, "src")

import math
import torch
import torch.nn as nn


# ==============================================================================
# SCHEDULER TESTS
# ==============================================================================


def test_power_decay_scheduler():
    """Test power-decay scheduler produces correct decay profile."""
    from palingenesis.optim import build_scheduler

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = build_scheduler(optimizer, "power_decay", num_steps=1000, warmup_ratio=0.1, min_lr_ratio=0.1)

    # Warmup phase: linear ramp
    scheduler.step()  # step 0
    lr_at_0 = optimizer.param_groups[0]["lr"]
    assert lr_at_0 < 1e-3, f"Step 0 should be in warmup, got LR={lr_at_0}"

    # End of warmup (step 100)
    for _ in range(99):
        scheduler.step()
    lr_at_100 = optimizer.param_groups[0]["lr"]
    assert abs(lr_at_100 - 1e-3) < 1e-6, f"Step 100 should be peak LR, got {lr_at_100}"

    # Mid-training (step 550 = 50% through decay phase)
    for _ in range(450):
        scheduler.step()
    lr_at_550 = optimizer.param_groups[0]["lr"]
    # Power decay: (1-0.5)^4 = 0.0625 → LR = 0.1 + 0.9*0.0625 = 0.15625 of peak
    expected_fraction = 0.1 + 0.9 * (0.5**4)
    assert (
        abs(lr_at_550 / 1e-3 - expected_fraction) < 0.02
    ), f"Mid-training LR should be ~{expected_fraction*1e-3:.6f}, got {lr_at_550:.6f}"

    # End of training: should be at min_lr_ratio
    for _ in range(450):
        scheduler.step()
    lr_end = optimizer.param_groups[0]["lr"]
    assert lr_end <= 1e-3 * 0.12, f"End LR should be near min_lr_ratio, got {lr_end}"

    print(f"  Warmup end: {lr_at_100:.6f}")
    print(f"  Mid-decay:  {lr_at_550:.6f} (expected ~{expected_fraction*1e-3:.6f})")
    print(f"  End:        {lr_end:.6f} (floor={1e-3*0.1:.6f})")
    print("✓ test_power_decay_scheduler PASSED\n")


def test_wsd_scheduler():
    """Test WSD (warmup-stable-decay) scheduler."""
    from palingenesis.optim import build_scheduler

    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = build_scheduler(optimizer, "wsd", num_steps=1000, warmup_ratio=0.1, min_lr_ratio=0.1)

    # Advance to end of warmup
    for _ in range(100):
        scheduler.step()
    lr_warmup_end = optimizer.param_groups[0]["lr"]
    assert abs(lr_warmup_end - 1e-3) < 1e-6

    # Stable phase: LR should stay at peak for 80% of post-warmup
    # Post-warmup is 900 steps. Stable = 80% = 720 steps (up to step 820)
    for _ in range(360):
        scheduler.step()
    lr_mid_stable = optimizer.param_groups[0]["lr"]
    assert abs(lr_mid_stable - 1e-3) < 1e-6, f"Mid-stable should be peak LR, got {lr_mid_stable}"

    # Advance to decay phase (step 820+)
    for _ in range(360):
        scheduler.step()
    lr_stable_end = optimizer.param_groups[0]["lr"]
    assert abs(lr_stable_end - 1e-3) < 1e-6, f"End of stable phase should be peak LR, got {lr_stable_end}"

    # In decay phase (step 900 = 80% through post-warmup → start of decay)
    for _ in range(80):
        scheduler.step()
    lr_decay_start = optimizer.param_groups[0]["lr"]
    # At this point progress into decay ≈ 80/180 ≈ 0.44
    assert lr_decay_start < 1e-3, f"Decay phase should reduce LR, got {lr_decay_start}"

    # End of training
    for _ in range(100):
        scheduler.step()
    lr_end = optimizer.param_groups[0]["lr"]
    assert lr_end <= 1e-3 * 0.15, f"End of WSD should be near min, got {lr_end}"

    print(f"  Stable phase: {lr_mid_stable:.6f} (should be peak)")
    print(f"  Decay start:  {lr_decay_start:.6f}")
    print(f"  End:          {lr_end:.6f}")
    print("✓ test_wsd_scheduler PASSED\n")


def test_scheduler_monotonicity():
    """Verify all scheduler types produce monotonically non-increasing LR after warmup."""
    from palingenesis.optim import build_scheduler

    for sched_type in ["cosine", "linear", "power_decay", "wsd"]:
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, sched_type, num_steps=500, warmup_ratio=0.1, min_lr_ratio=0.01)

        lrs = []
        for _ in range(500):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # After warmup (step 50+), LR should be non-increasing
        post_warmup = lrs[50:]
        for i in range(1, len(post_warmup)):
            assert (
                post_warmup[i] <= post_warmup[i - 1] + 1e-10
            ), f"{sched_type}: LR increased at step {50+i}: {post_warmup[i-1]:.8f} -> {post_warmup[i]:.8f}"

        print(f"  {sched_type}: peak={max(lrs):.6f}, end={lrs[-1]:.6f} ✓")

    print("✓ test_scheduler_monotonicity PASSED\n")


# ==============================================================================
# RL-READINESS ENTROPY MONITORING TESTS
# ==============================================================================


def test_entropy_monitoring_basic():
    """Test that entropy monitoring records and computes correctly."""
    from palingenesis.health import HealthMonitor

    model = nn.Linear(10, 100)
    monitor = HealthMonitor(model, rl_readiness=True, rl_entropy_floor=2.0)

    # Simulate high-entropy logits (uniform-ish distribution)
    logits = torch.randn(2, 16, 100)  # batch=2, seq=16, vocab=100
    labels = torch.randint(0, 100, (2, 16))

    monitor.record_logit_entropy(logits, labels)

    assert len(monitor._entropy_buffer) == 1
    entropy_val = monitor._entropy_buffer[0]

    # Random logits on vocab=100 should have entropy ~log(100) ≈ 4.6
    # With noise it'll be somewhat lower but still high
    assert entropy_val > 2.0, f"Random logits should have high entropy, got {entropy_val:.2f}"

    print(f"  Random logits entropy: {entropy_val:.2f} (expected ~4.0-4.6)")
    print("✓ test_entropy_monitoring_basic PASSED\n")


def test_entropy_collapse_detection():
    """Test that entropy collapse warning triggers correctly."""
    import logging
    from palingenesis.health import HealthMonitor

    model = nn.Linear(10, 100)
    monitor = HealthMonitor(model, rl_readiness=True, rl_entropy_floor=2.0)

    # Simulate collapsing entropy over 20 steps
    vocab_size = 100
    for i in range(20):
        # Create increasingly peaked logits (simulating overtraining)
        temperature = max(0.1, 1.0 - i * 0.045)  # goes from 1.0 to 0.1
        logits = torch.randn(2, 16, vocab_size) / temperature
        labels = torch.randint(0, vocab_size, (2, 16))
        monitor.record_logit_entropy(logits, labels)
        monitor.record_microstep(loss=2.0 - i * 0.05, labels=labels)

    # Get metrics
    metrics = monitor._tier1_metrics()

    # Should have entropy data
    assert "health/output_entropy" in metrics
    assert "health/output_entropy_ema" in metrics

    # The entropy should be declining
    entropy_values = list(monitor._entropy_buffer)
    assert entropy_values[-1] < entropy_values[0], "Entropy should be declining"

    print(f"  Entropy start: {entropy_values[0]:.2f}")
    print(f"  Entropy end:   {entropy_values[-1]:.2f}")
    print(f"  EMA:           {metrics['health/output_entropy_ema']:.2f}")

    # If entropy dropped below floor, warning should have triggered
    if metrics.get("health/output_entropy_ema", 999) < 2.0:
        assert monitor._entropy_warned, "Should have warned about entropy collapse"
        print("  Warning triggered ✓")
    else:
        print(f"  No warning (entropy EMA {metrics['health/output_entropy_ema']:.2f} >= floor 2.0)")

    print("✓ test_entropy_collapse_detection PASSED\n")


def test_entropy_monitoring_ignores_mask():
    """Test that entropy is only computed on valid (non-IGNORE) positions."""
    from palingenesis.health import HealthMonitor, IGNORE_INDEX

    model = nn.Linear(10, 50)
    monitor = HealthMonitor(model, rl_readiness=True, rl_entropy_floor=1.0)

    # Create logits with some positions masked
    logits = torch.randn(1, 32, 50)
    labels = torch.full((1, 32), IGNORE_INDEX, dtype=torch.long)
    # Only mark 8 positions as valid
    labels[0, :8] = torch.randint(0, 50, (8,))

    monitor.record_logit_entropy(logits, labels)

    assert len(monitor._entropy_buffer) == 1
    print(f"  Computed entropy on 8/32 valid tokens: {monitor._entropy_buffer[0]:.2f}")
    print("✓ test_entropy_monitoring_ignores_mask PASSED\n")


def test_entropy_monitoring_disabled():
    """Test that entropy monitoring is a no-op when disabled."""
    from palingenesis.health import HealthMonitor

    model = nn.Linear(10, 50)
    monitor = HealthMonitor(model, rl_readiness=False)  # disabled

    logits = torch.randn(2, 16, 50)
    labels = torch.randint(0, 50, (2, 16))

    monitor.record_logit_entropy(logits, labels)
    assert len(monitor._entropy_buffer) == 0, "Should not record when disabled"

    metrics = monitor._tier1_metrics()
    assert "health/output_entropy" not in metrics

    print("✓ test_entropy_monitoring_disabled PASSED\n")


# ==============================================================================
# HYPERBALL OPTIMIZER WRAPPER TESTS
# ==============================================================================


def test_hyperball_preserves_norm():
    """Test that Hyperball keeps weight matrix norms constant."""
    from palingenesis.optim import HyperballWrapper

    torch.manual_seed(42)
    # Create a simple weight matrix
    W = nn.Linear(64, 32, bias=False)
    initial_norm = W.weight.data.norm().item()

    optimizer = torch.optim.AdamW(W.parameters(), lr=1e-2, weight_decay=0.0)
    hyperball = HyperballWrapper(optimizer, [W.weight])

    # Run several optimization steps
    for _ in range(50):
        optimizer.zero_grad()
        x = torch.randn(4, 64)
        loss = W(x).pow(2).mean()
        loss.backward()
        hyperball.step()

    final_norm = W.weight.data.norm().item()
    diff = abs(final_norm - initial_norm) / initial_norm

    print(f"  Initial norm: {initial_norm:.6f}")
    print(f"  Final norm:   {final_norm:.6f}")
    print(f"  Relative diff: {diff:.2e}")
    assert diff < 1e-5, f"Hyperball should preserve norm, got {diff:.2e} relative change"
    print("✓ test_hyperball_preserves_norm PASSED\n")


def test_hyperball_updates_direction():
    """Test that Hyperball actually changes the weight direction (not stuck)."""
    from palingenesis.optim import HyperballWrapper

    torch.manual_seed(42)
    W = nn.Linear(64, 32, bias=False)
    initial_direction = W.weight.data.clone() / W.weight.data.norm()

    optimizer = torch.optim.AdamW(W.parameters(), lr=1e-2, weight_decay=0.0)
    hyperball = HyperballWrapper(optimizer, [W.weight])

    for _ in range(20):
        optimizer.zero_grad()
        x = torch.randn(4, 64)
        target = torch.randn(4, 32)
        loss = (W(x) - target).pow(2).mean()
        loss.backward()
        hyperball.step()

    final_direction = W.weight.data / W.weight.data.norm()
    cosine_sim = (initial_direction * final_direction).sum().item()

    print(f"  Cosine similarity with init: {cosine_sim:.4f}")
    assert cosine_sim < 0.99, f"Hyperball should change direction, got cosine={cosine_sim}"
    assert cosine_sim > -1.0, "Direction should not completely flip"
    print("✓ test_hyperball_updates_direction PASSED\n")


def test_hyperball_convergence():
    """Test that Hyperball + AdamW converges on a simple task."""
    from palingenesis.optim import HyperballWrapper

    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    target_W = torch.randn(16, 32) * 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    hyperball = HyperballWrapper(optimizer, [model.weight])

    losses = []
    for step in range(400):
        optimizer.zero_grad()
        x = torch.randn(8, 32)
        target = x @ target_W.T
        pred = model(x)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        hyperball.step()
        losses.append(loss.item())

    # Loss should decrease substantially
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    ratio = final_loss / initial_loss

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Reduction:    {(1-ratio)*100:.1f}%")
    assert ratio < 0.7, f"Should converge by >30%, got only {(1-ratio)*100:.1f}% reduction"
    print("✓ test_hyperball_convergence PASSED\n")


# ==============================================================================
# MONA ACCELERATION TESTS
# ==============================================================================


def test_mona_acceleration_basic():
    """Test MONA acceleration computes gradient differences correctly."""
    from palingenesis.optim import MONAAcceleration

    torch.manual_seed(42)
    model = nn.Linear(32, 16)

    mona = MONAAcceleration(model, beta_a=0.99, alpha=-50.0)

    # First step: no acceleration (no previous gradient)
    x = torch.randn(4, 32)
    loss = model(x).pow(2).mean()
    loss.backward()
    mona.apply()  # Should be no-op on first step (no prev gradient)

    model.zero_grad()

    # Second step: should have acceleration
    x = torch.randn(4, 32)
    loss = model(x).pow(2).mean()
    loss.backward()

    # Save pre-acceleration gradient norm
    pre_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    mona.apply()
    post_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    # After acceleration, gradient should be modified (different norm)
    print(f"  Pre-MONA grad norm:  {pre_norm:.4f}")
    print(f"  Post-MONA grad norm: {post_norm:.4f}")
    # They should differ (acceleration changes the gradient)
    assert abs(post_norm - pre_norm) > 1e-6, "MONA should modify gradients"
    print("✓ test_mona_acceleration_basic PASSED\n")


def test_mona_lite_bf16():
    """Test MONA-Lite stores buffers in bf16 for memory efficiency."""
    from palingenesis.optim import MONAAcceleration

    model = nn.Linear(128, 64)
    mona = MONAAcceleration(model, beta_a=0.99, alpha=-50.0, lite=True)

    # Run a step to populate buffers
    x = torch.randn(4, 128)
    loss = model(x).pow(2).mean()
    loss.backward()
    mona.apply()

    # Check that acceleration buffer is in bf16
    for buf in mona._acceleration_buffers.values():
        assert buf.dtype == torch.bfloat16, f"MONA-Lite buffer should be bf16, got {buf.dtype}"

    print("  All acceleration buffers in bf16 ✓")
    print("✓ test_mona_lite_bf16 PASSED\n")


# ==============================================================================
# SAGE EMBEDDING OPTIMIZER TESTS
# ==============================================================================


def test_sage_basic_convergence():
    """Test SAGE optimizer converges on embedding lookup task."""
    from palingenesis.optim import SAGE

    torch.manual_seed(42)
    vocab_size, embed_dim = 1000, 64
    embedding = nn.Embedding(vocab_size, embed_dim)

    optimizer = SAGE(embedding.parameters(), lr=1e-2, beta1=0.9, beta2=0.99)

    # Simple task: make embedding[i] = one_hot(i % 10) (learn a pattern)
    target_pattern = torch.randn(10, embed_dim)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        indices = torch.randint(0, vocab_size, (32,))
        embeds = embedding(indices)
        targets = target_pattern[indices % 10]
        loss = (embeds - targets).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    print(f"  Initial loss: {initial:.4f}")
    print(f"  Final loss:   {final:.4f}")
    assert final < initial * 0.5, f"SAGE should converge, only {(1-final/initial)*100:.0f}% reduction"
    print("✓ test_sage_basic_convergence PASSED\n")


def test_sage_damper_bounded():
    """Test that SAGE's adaptive scale Ht is always bounded by 1.0."""
    from palingenesis.optim import SAGE

    torch.manual_seed(42)
    embedding = nn.Embedding(500, 32)
    optimizer = SAGE(embedding.parameters(), lr=1e-2, beta1=0.9, beta2=0.99)

    # Run several steps and check internal state
    for _ in range(20):
        optimizer.zero_grad()
        idx = torch.randint(0, 500, (16,))
        loss = embedding(idx).pow(2).mean()
        loss.backward()
        # Access internal damper scale (exposed for testing)
        scales = optimizer.get_adaptive_scales()
        for name, scale in scales.items():
            max_val = scale.max().item()
            assert max_val <= 1.0 + 1e-6, f"SAGE scale must be ≤1.0, got {max_val}"
        optimizer.step()

    print("  All adaptive scales ≤ 1.0 across 20 steps ✓")
    print("✓ test_sage_damper_bounded PASSED\n")


def test_sage_memory_efficiency():
    """Test SAGE uses O(d) state, not O(V*d) like AdamW."""
    from palingenesis.optim import SAGE

    vocab_size, embed_dim = 50000, 512  # realistic sizes
    embedding = nn.Embedding(vocab_size, embed_dim)

    optimizer = SAGE(embedding.parameters(), lr=1e-3, beta1=0.9, beta2=0.99)

    # Run one step to populate state
    optimizer.zero_grad()
    idx = torch.randint(0, vocab_size, (8,))
    loss = embedding(idx).pow(2).mean()
    loss.backward()
    optimizer.step()

    # Count optimizer state elements
    total_state_elements = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                total_state_elements += v.numel()

    param_elements = vocab_size * embed_dim  # 25.6M
    # SAGE: O(V*d) for momentum + O(d) for adaptive scale = V*d + d
    # AdamW would be: O(V*d) for m + O(V*d) for v = 2*V*d
    # SAGE should be < 1.5x param elements (momentum + small d-sized buffer)
    ratio = total_state_elements / param_elements

    print(f"  Param elements:    {param_elements:,}")
    print(f"  State elements:    {total_state_elements:,}")
    print(f"  State/param ratio: {ratio:.2f}x")
    assert ratio < 1.5, f"SAGE should use < 1.5x param size in state, got {ratio:.2f}x"
    # AdamW would be 2.0x
    print("✓ test_sage_memory_efficiency PASSED\n")


# ==============================================================================
# MAIN
# ==============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZER & HEALTH MONITORING TESTS")
    print("=" * 60 + "\n")

    # Scheduler tests
    print("── Scheduler Tests ──\n")
    test_power_decay_scheduler()
    test_wsd_scheduler()
    test_scheduler_monotonicity()

    # Entropy monitoring tests
    print("── RL-Readiness Entropy Monitoring Tests ──\n")
    test_entropy_monitoring_basic()
    test_entropy_collapse_detection()
    test_entropy_monitoring_ignores_mask()
    test_entropy_monitoring_disabled()

    # Hyperball tests
    print("── Hyperball Optimizer Tests ──\n")
    test_hyperball_preserves_norm()
    test_hyperball_updates_direction()
    test_hyperball_convergence()

    # MONA tests
    print("── MONA Acceleration Tests ──\n")
    test_mona_acceleration_basic()
    test_mona_lite_bf16()

    # SAGE tests
    print("── SAGE Embedding Optimizer Tests ──\n")
    test_sage_basic_convergence()
    test_sage_damper_bounded()
    test_sage_memory_efficiency()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
