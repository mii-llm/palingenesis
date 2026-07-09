"""Deep tests for performance techniques: AdaGC, SpikeDetector, ModelEMA, BaseModelMerge.

Each technique is tested for:
- Correctness (does it produce the right output?)
- Edge cases (empty tensors, zero gradients, extreme values)
- Long-running stability (doesn't drift or explode over 1000+ steps)
- Composition (works when stacked with other techniques)
"""

import sys

sys.path.insert(0, "src")

import math
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# ADAGC: Per-Tensor Adaptive Gradient Clipping
# ══════════════════════════════════════════════════════════════════════════════


def test_adagc_warmup_uses_global_clip():
    """During warmup, AdaGC should use global clipping (standard behavior)."""
    from palingenesis.perf import AdaGC

    model = nn.Linear(32, 16)
    adagc = AdaGC(model, lambda_rel=1.5, beta=0.95, warmup_steps=10, global_max_norm=1.0)

    # Create artificially large gradients
    model.weight.grad = torch.ones_like(model.weight) * 100.0
    model.bias.grad = torch.ones_like(model.bias) * 100.0

    # During warmup (step < 10): should clip globally to max_norm=1.0
    norm = adagc.clip(step=5)

    # After global clip, gradient norm should be <= 1.0
    post_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
    assert post_norm <= 1.01, f"Warmup should global-clip to 1.0, got {post_norm:.3f}"
    print(f"  Warmup clip: {norm:.2f} → {post_norm:.4f}")
    print("✓ test_adagc_warmup_uses_global_clip PASSED\n")


def test_adagc_post_warmup_clips_per_tensor():
    """After warmup, AdaGC clips each tensor independently based on its EMA."""
    from palingenesis.perf import AdaGC

    model = nn.Linear(32, 16)
    adagc = AdaGC(model, lambda_rel=1.5, beta=0.95, warmup_steps=5, global_max_norm=100.0)

    # Warmup: feed normal gradients to build EMAs
    for step in range(5):
        model.weight.grad = torch.randn_like(model.weight) * 0.1
        model.bias.grad = torch.randn_like(model.bias) * 0.1
        adagc.clip(step=step)

    # Post-warmup: create a spike on weight only
    model.weight.grad = torch.randn_like(model.weight) * 10.0  # 100× normal
    model.bias.grad = torch.randn_like(model.bias) * 0.1  # normal

    bias_norm_before = model.bias.grad.norm().item()
    adagc.clip(step=6)
    bias_norm_after = model.bias.grad.norm().item()

    # Bias should be unchanged (within its own EMA threshold)
    assert abs(bias_norm_after - bias_norm_before) < 0.01, "Non-spiking tensor shouldn't be clipped"
    # Weight should be clipped (was 100× normal)
    assert adagc.total_clips > 0, "Should have clipped the spiking tensor"
    print(f"  Clips applied: {adagc.total_clips}")
    print("✓ test_adagc_post_warmup_clips_per_tensor PASSED\n")


def test_adagc_ema_adapts_over_time():
    """AdaGC's EMA should track increasing gradient magnitudes without false clips."""
    from palingenesis.perf import AdaGC

    model = nn.Linear(64, 32)
    adagc = AdaGC(model, lambda_rel=2.0, beta=0.95, warmup_steps=20, global_max_norm=100.0)

    # Warmup with magnitude 0.1
    for step in range(20):
        model.weight.grad = torch.randn_like(model.weight) * 0.1
        model.bias.grad = torch.randn_like(model.bias) * 0.1
        adagc.clip(step=step)

    # Gradually increase gradient magnitude — EMA should adapt
    clips_during_ramp = 0
    for step in range(20, 120):
        magnitude = 0.1 + (step - 20) * 0.005  # slowly increases from 0.1 to 0.6
        model.weight.grad = torch.randn_like(model.weight) * magnitude
        model.bias.grad = torch.randn_like(model.bias) * magnitude
        prev_clips = adagc.total_clips
        adagc.clip(step=step)
        if adagc.total_clips > prev_clips:
            clips_during_ramp += 1

    # With gradual increase and lambda=2.0, should have very few clips
    # (EMA adapts to the slow ramp)
    assert clips_during_ramp < 30, f"EMA should adapt to gradual changes, got {clips_during_ramp} clips"
    print(f"  Clips during gradual ramp: {clips_during_ramp}/100 steps")
    print("✓ test_adagc_ema_adapts_over_time PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# SPIKE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


def test_spike_detector_warmup_never_flags():
    """During warmup, no spikes should be detected regardless of values."""
    from palingenesis.perf import SpikeDetector

    detector = SpikeDetector(z_threshold=3.0, warmup=50)

    # Feed extreme values during warmup
    for i in range(50):
        result = detector.check(1000.0 * (i + 1))  # increasing, extreme
        assert result is False, f"Warmup step {i} should never flag"

    print("  50 warmup steps with extreme values: no flags")
    print("✓ test_spike_detector_warmup_never_flags PASSED\n")


def test_spike_detector_detects_true_spikes():
    """After warmup, genuine spikes (>z_threshold σ) are detected."""
    from palingenesis.perf import SpikeDetector

    detector = SpikeDetector(z_threshold=3.0, warmup=50)

    # Warmup with stable values around 1.0
    for _ in range(50):
        detector.check(1.0 + torch.randn(1).item() * 0.1)

    # Now inject a massive spike
    is_spike = detector.check(100.0)
    assert is_spike is True, "100.0 after training at ~1.0 should be a spike"

    # Normal value should not be a spike
    is_spike = detector.check(1.1)
    assert is_spike is False, "1.1 after training at ~1.0 should not be a spike"

    print(f"  Detected spike at 100.0 (mean≈{detector.mean:.2f})")
    print("✓ test_spike_detector_detects_true_spikes PASSED\n")


def test_spike_detector_doesnt_drift_from_spikes():
    """Spike values should NOT update the running statistics (prevents drift)."""
    from palingenesis.perf import SpikeDetector

    detector = SpikeDetector(z_threshold=3.0, warmup=50, ema_decay=0.99)

    # Warmup at ~1.0
    for _ in range(50):
        detector.check(1.0)

    mean_before = detector.mean

    # Inject 10 spikes — mean should NOT drift toward them
    for _ in range(10):
        detector.check(1000.0)  # massive spike

    mean_after = detector.mean
    drift = abs(mean_after - mean_before)

    assert drift < 0.1, f"Mean should not drift from spikes, drifted {drift:.4f}"
    print(f"  Mean before spikes: {mean_before:.4f}")
    print(f"  Mean after 10 spikes: {mean_after:.4f} (drift={drift:.6f})")
    print("✓ test_spike_detector_doesnt_drift_from_spikes PASSED\n")


def test_spike_detector_long_running_stability():
    """Over 10000 steps with occasional spikes, detector remains stable."""
    from palingenesis.perf import SpikeDetector
    import random

    random.seed(42)
    detector = SpikeDetector(z_threshold=5.0, warmup=100, ema_decay=0.99)

    spikes_detected = 0
    for step in range(10000):
        # Normal gradient norm with occasional spike (1% chance)
        if random.random() < 0.01:
            value = random.uniform(50, 200)  # spike
        else:
            value = 1.0 + random.gauss(0, 0.2)  # normal

        if detector.check(value):
            spikes_detected += 1

    # Should detect roughly 1% of 9900 post-warmup steps (about 50-150 spikes)
    assert 20 < spikes_detected < 200, f"Expected ~100 spikes, got {spikes_detected}"
    # Mean should still be near 1.0 (not drifted by spikes)
    assert abs(detector.mean - 1.0) < 0.5, f"Mean drifted to {detector.mean:.2f}"
    print(f"  10000 steps, {spikes_detected} spikes detected, mean={detector.mean:.3f}")
    print("✓ test_spike_detector_long_running_stability PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EMA
# ══════════════════════════════════════════════════════════════════════════════


def test_ema_shadow_tracks_weights():
    """EMA shadow weights should converge toward the current model weights."""
    from palingenesis.perf import ModelEMA

    torch.manual_seed(42)
    model = nn.Linear(32, 16)
    ema = ModelEMA(model, decay=0.9)

    # Modify model weights several times, update EMA
    for _ in range(100):
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight) * 0.01)
        ema.update()

    # Shadow should be close to current weights (with decay=0.9, fast tracking)
    shadow_weight = ema._shadow["weight"]
    current_weight = model.weight.data.float().cpu()
    diff = (shadow_weight - current_weight).abs().mean().item()
    assert diff < 0.1, f"EMA should track weights, got diff={diff:.4f}"
    print(f"  EMA-to-current weight diff: {diff:.4f}")
    print("✓ test_ema_shadow_tracks_weights PASSED\n")


def test_ema_apply_copies_to_model():
    """apply_to_model should copy shadow weights INTO the model."""
    from palingenesis.perf import ModelEMA

    torch.manual_seed(42)
    model = nn.Linear(32, 16)
    original_weight = model.weight.data.clone()
    ema = ModelEMA(model, decay=0.999)

    # Update EMA several times (shadow converges to current model ≈ original)
    for _ in range(50):
        ema.update()

    # Verify shadow is close to original
    shadow = ema._shadow["weight"]
    assert (shadow - original_weight.float().cpu()).abs().max().item() < 0.5, "Shadow should be near original"

    # Now drastically change the model
    with torch.no_grad():
        model.weight.fill_(999.0)

    assert model.weight.data.mean().item() == 999.0, "Model should be 999"

    # Apply EMA — should restore to shadow values (near original, NOT 999)
    ema.apply_to_model()
    max_val = model.weight.data.abs().max().item()
    assert max_val < 5.0, f"After EMA apply, weights should be near original, got max={max_val}"
    print(f"  After apply: max weight = {max_val:.2f} (restored from shadow, not 999)")
    print("✓ test_ema_apply_copies_to_model PASSED\n")


def test_ema_on_cpu_no_gpu_memory():
    """EMA shadows should be on CPU, not GPU memory."""
    from palingenesis.perf import ModelEMA

    model = nn.Linear(64, 32)
    ema = ModelEMA(model, decay=0.999)

    for name, tensor in ema._shadow.items():
        assert tensor.device.type == "cpu", f"Shadow {name} should be on CPU, got {tensor.device}"

    print("  All shadow tensors on CPU ✓")
    print("✓ test_ema_on_cpu_no_gpu_memory PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# BASE MODEL MERGE (SLERP)
# ══════════════════════════════════════════════════════════════════════════════


def test_base_merge_lerp_correctness():
    """LERP merge should interpolate weights toward base by exactly merge_ratio."""
    from palingenesis.perf import BaseModelMerge

    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    initial_weight = model.weight.data.clone()

    merger = BaseModelMerge(model, merge_ratio=0.3, method="lerp")

    # Move model weights far from initial
    with torch.no_grad():
        model.weight.fill_(10.0)

    # Merge: should move 30% toward base
    merger.merge_step()

    # Expected: 0.7 * 10.0 + 0.3 * initial = 7.0 + 0.3*initial
    expected = 0.7 * 10.0 + 0.3 * initial_weight.to(model.weight.dtype)
    diff = (model.weight.data - expected).abs().max().item()
    assert diff < 1e-4, f"LERP merge should be exact, got diff={diff}"
    print(f"  LERP accuracy: max diff = {diff:.2e}")
    print("✓ test_base_merge_lerp_correctness PASSED\n")


def test_base_merge_slerp_preserves_norm():
    """SLERP should preserve the Frobenius norm (unlike LERP which shrinks it)."""
    from palingenesis.perf import BaseModelMerge

    torch.manual_seed(42)
    model = nn.Linear(64, 32, bias=False)
    initial_norm = model.weight.data.float().norm().item()

    merger = BaseModelMerge(model, merge_ratio=0.2, method="slerp")

    # Modify weights (different direction, similar magnitude)
    with torch.no_grad():
        model.weight.data = torch.randn_like(model.weight) * (initial_norm / model.weight.data.norm().item())

    norm_before = model.weight.data.float().norm().item()
    merger.merge_step()
    norm_after = model.weight.data.float().norm().item()

    # SLERP should approximately preserve norm
    norm_ratio = norm_after / norm_before
    assert 0.85 < norm_ratio < 1.15, f"SLERP should preserve norm, ratio={norm_ratio:.3f}"

    print(f"  Norm before: {norm_before:.3f}, after: {norm_after:.3f}, ratio: {norm_ratio:.3f}")
    print("✓ test_base_merge_slerp_preserves_norm PASSED\n")


def test_base_merge_repeated_converges_to_base():
    """Repeated merges should converge the model back toward base weights."""
    from palingenesis.perf import BaseModelMerge

    torch.manual_seed(42)
    model = nn.Linear(32, 16, bias=False)
    base_weight = model.weight.data.clone()

    merger = BaseModelMerge(model, merge_ratio=0.1, method="lerp")

    # Move weights far away
    with torch.no_grad():
        model.weight.fill_(100.0)

    # Apply merge many times — should converge toward base
    for _ in range(50):
        merger.merge_step()

    final = model.weight.data
    diff_from_base = (final.float() - base_weight.float()).abs().mean().item()
    diff_from_100 = abs(final.mean().item() - 100.0)

    assert diff_from_100 > 50, "Should have moved significantly from 100.0"
    assert diff_from_base < 10, f"Should be closer to base after 50 merges, diff={diff_from_base:.2f}"
    print(f"  After 50 merges: mean={final.mean().item():.2f} (started at 100, base≈0)")
    print("✓ test_base_merge_repeated_converges_to_base PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITION: Multiple techniques together
# ══════════════════════════════════════════════════════════════════════════════


def test_full_perf_stack_composition():
    """All perf techniques compose without interference over a training simulation."""
    from palingenesis.perf import AdaGC, SpikeDetector, ModelEMA, BaseModelMerge

    torch.manual_seed(42)
    model = nn.Linear(64, 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    adagc = AdaGC(model, lambda_rel=1.5, beta=0.95, warmup_steps=10, global_max_norm=1.0)
    spike_det = SpikeDetector(z_threshold=5.0, warmup=10)
    ema = ModelEMA(model, decay=0.99)
    merger = BaseModelMerge(model, merge_ratio=0.05, method="lerp")

    target = torch.randn(4, 32)
    losses = []

    for step in range(100):
        optimizer.zero_grad()
        x = torch.randn(4, 64)
        loss = (model(x) - target).pow(2).mean()
        loss.backward()

        # AdaGC clips
        grad_norm = adagc.clip(step=step)

        # Spike detection
        gn_val = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
        is_spike = spike_det.check(gn_val)

        if not is_spike:
            optimizer.step()

        # EMA every 5 steps
        if step % 5 == 0:
            ema.update()

        # Base merge every 20 steps
        if step % 20 == 0 and step > 0:
            merger.merge_step()

        losses.append(loss.item())

    # Verify training worked
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, f"Loss should decrease: {initial:.3f} → {final:.3f}"

    # Verify no NaN/Inf
    assert all(math.isfinite(l) for l in losses), "No NaN/Inf losses"

    # Verify EMA is valid
    for t in ema._shadow.values():
        assert t.isfinite().all(), "EMA shadow should be finite"

    print(f"  Loss: {initial:.3f} → {final:.3f} ({(1-final/initial)*100:.0f}% reduction)")
    print(f"  Spikes: {spike_det.spikes_detected}, AdaGC clips: {adagc.total_clips}")
    print("✓ test_full_perf_stack_composition PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE TECHNIQUES — DEEP TESTS")
    print("=" * 60 + "\n")

    print("── AdaGC ──\n")
    test_adagc_warmup_uses_global_clip()
    test_adagc_post_warmup_clips_per_tensor()
    test_adagc_ema_adapts_over_time()

    print("── SpikeDetector ──\n")
    test_spike_detector_warmup_never_flags()
    test_spike_detector_detects_true_spikes()
    test_spike_detector_doesnt_drift_from_spikes()
    test_spike_detector_long_running_stability()

    print("── ModelEMA ──\n")
    test_ema_shadow_tracks_weights()
    test_ema_apply_copies_to_model()
    test_ema_on_cpu_no_gpu_memory()

    print("── BaseModelMerge ──\n")
    test_base_merge_lerp_correctness()
    test_base_merge_slerp_preserves_norm()
    test_base_merge_repeated_converges_to_base()

    print("── Composition ──\n")
    test_full_perf_stack_composition()

    print("=" * 60)
    print("ALL PERFORMANCE TESTS PASSED ✓")
    print("=" * 60)
