"""Tests for the adaptive LR sweep module.

Verifies:
- Candidate generation covers appropriate ranges
- Refinement narrows correctly around the best trial
- Horizon correction is adaptive and bounded
- Early stopping triggers on divergent/stuck trials
- Full orchestrator flow works end-to-end with mock trials
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from palingenesis.autopilot.adaptive_sweep import (
    TrialResult,
    adaptive_lr_sweep,
    correct_lr_adaptive,
    estimate_horizon_exponent,
    generate_coarse_candidates,
    generate_refinement_candidates,
    should_early_stop_trial,
    _surge_aware_batch_factor,
)


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def test_coarse_candidates_cover_reasonable_range():
    """Coarse candidates span ~2.4 decades centered on scaling-law estimate."""
    candidates = generate_coarse_candidates(
        model_params_b=4.0, effective_batch_tokens=524288,
    )

    assert len(candidates) == 5
    # All should be in a sane SFT range
    for c in candidates:
        assert 1e-7 <= c <= 5e-3, f"Candidate {c:.2e} outside sane range"

    # Should span at least 2 orders of magnitude
    span = math.log10(candidates[-1]) - math.log10(candidates[0])
    assert span >= 2.0, f"Candidates only span {span:.1f} decades (need >= 2)"

    print(f"  Candidates: {[f'{c:.2e}' for c in candidates]}")
    print(f"  Span: {span:.1f} decades")
    print("✓ test_coarse_candidates_cover_reasonable_range PASSED\n")


def test_coarse_candidates_scale_with_model_size():
    """Larger models get lower LR candidates (scaling law)."""
    small = generate_coarse_candidates(model_params_b=0.5, effective_batch_tokens=100000)
    large = generate_coarse_candidates(model_params_b=8.0, effective_batch_tokens=100000)

    # Center of large-model candidates should be lower
    small_center = small[len(small) // 2]
    large_center = large[len(large) // 2]
    assert large_center < small_center, (
        f"Large model center ({large_center:.2e}) should be < "
        f"small model center ({small_center:.2e})"
    )

    print(f"  0.5B center: {small_center:.2e}")
    print(f"  8.0B center: {large_center:.2e}")
    print("✓ test_coarse_candidates_scale_with_model_size PASSED\n")


def test_coarse_candidates_scale_with_batch_size():
    """Batch size scaling follows the surge phenomenon — not monotone."""
    # At B_crit (~100K), candidates should be highest
    at_bcrit = generate_coarse_candidates(model_params_b=4.0, effective_batch_tokens=100000)
    # Well below B_crit (small batch: 1 GPU, batch=1, seq=4096, GA=1 = 4K tokens)
    small_batch = generate_coarse_candidates(model_params_b=4.0, effective_batch_tokens=4096)
    # Well above B_crit (large batch: 8 GPU, batch=4, seq=8192, GA=16 = 4M tokens)
    large_batch = generate_coarse_candidates(model_params_b=4.0, effective_batch_tokens=4_000_000)

    bcrit_center = at_bcrit[len(at_bcrit) // 2]
    small_center = small_batch[len(small_batch) // 2]
    large_center = large_batch[len(large_batch) // 2]

    # Surge: at B_crit center should be HIGHER than both small and large
    assert bcrit_center > small_center, (
        f"B_crit center ({bcrit_center:.2e}) should be > small ({small_center:.2e})"
    )
    assert bcrit_center > large_center, (
        f"B_crit center ({bcrit_center:.2e}) should be > large ({large_center:.2e})"
    )

    print(f"  Small batch (4K tok): {small_center:.2e}")
    print(f"  At B_crit (100K tok): {bcrit_center:.2e}")
    print(f"  Large batch (4M tok): {large_center:.2e}")
    print("✓ test_coarse_candidates_scale_with_batch_size PASSED\n")


def test_surge_aware_batch_factor():
    """The surge function peaks at B_crit and falls on both sides."""
    from palingenesis.autopilot.adaptive_sweep import _surge_aware_batch_factor

    # At B_crit (100K): factor should be 1.0 (peak, by normalization)
    f_crit = _surge_aware_batch_factor(100_000)
    assert abs(f_crit - 1.0) < 0.01, f"At B_crit: factor={f_crit}, expected 1.0"

    # Below B_crit: factor < 1 (rising phase but not at peak)
    f_small = _surge_aware_batch_factor(4_096)  # single GPU, no GA
    assert f_small < 1.0, f"Small batch: factor={f_small}, expected < 1.0"

    # Well below: very small factor
    f_tiny = _surge_aware_batch_factor(1_024)  # toy batch
    assert f_tiny < f_small, f"Tinier batch should have lower factor"

    # Above B_crit: factor < 1 (falling phase)
    f_large = _surge_aware_batch_factor(4_000_000)  # 8GPU × batch4 × seq8K × GA16
    assert f_large < 1.0, f"Large batch: factor={f_large}, expected < 1.0"

    # The peak is at B_crit
    assert f_crit > f_small, f"Peak should be higher than small"
    assert f_crit > f_large, f"Peak should be higher than large"

    print(f"  1K tokens: factor={f_tiny:.3f}")
    print(f"  4K tokens: factor={f_small:.3f}")
    print(f"  100K tokens (B_crit): factor={f_crit:.3f}")
    print(f"  4M tokens: factor={f_large:.3f}")
    print("✓ test_surge_aware_batch_factor PASSED\n")


def test_coarse_candidates_hardware_params():
    """Test candidate generation with explicit hardware parameters."""
    # Scenario: single RTX 4090, batch=1, seq=4096, GA=8 → 32K tokens
    candidates_4090 = generate_coarse_candidates(
        model_params_b=4.0,
        effective_batch_tokens=1 * 4096 * 8 * 1,  # 32K
        per_device_batch_size=1,
        max_seq_length=4096,
        gradient_accumulation_steps=8,
        num_gpus=1,
    )

    # Scenario: 8x A100, batch=4, seq=4096, GA=4 → 524K tokens
    candidates_8a100 = generate_coarse_candidates(
        model_params_b=4.0,
        effective_batch_tokens=4 * 4096 * 4 * 8,  # 524K
        per_device_batch_size=4,
        max_seq_length=4096,
        gradient_accumulation_steps=4,
        num_gpus=8,
    )

    c_4090 = candidates_4090[len(candidates_4090) // 2]
    c_8a100 = candidates_8a100[len(candidates_8a100) // 2]

    # 32K tokens is below B_crit, 524K is above.
    # Both should produce lower center than at B_crit, but 524K is closer.
    # The key point: both produce valid candidates in the SFT range.
    assert 1e-6 <= c_4090 <= 1e-3
    assert 1e-6 <= c_8a100 <= 1e-3

    print(f"  RTX 4090 (32K tok/step): center={c_4090:.2e}")
    print(f"  8×A100 (524K tok/step): center={c_8a100:.2e}")
    print("✓ test_coarse_candidates_hardware_params PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════


def test_refinement_narrows_around_best():
    """Refinement candidates are in the neighborhood of the best trials."""
    coarse_results = [
        TrialResult(lr=1e-5, final_loss=2.5, initial_loss=3.0, loss_curve=[3.0, 2.7, 2.5]),
        TrialResult(lr=2e-5, final_loss=2.1, initial_loss=3.0, loss_curve=[3.0, 2.5, 2.1]),  # best
        TrialResult(lr=5e-5, final_loss=2.3, initial_loss=3.0, loss_curve=[3.0, 2.4, 2.3]),  # 2nd
        TrialResult(lr=1e-4, final_loss=3.5, initial_loss=3.0, loss_curve=[3.0, 3.2, 3.5], diverged=True),
        TrialResult(lr=5e-6, final_loss=2.8, initial_loss=3.0, loss_curve=[3.0, 2.9, 2.8]),
    ]

    refinement = generate_refinement_candidates(coarse_results, n_refine=3)

    assert len(refinement) == 3
    # Refinement candidates should be within ~1 decade of the best trial (2e-5)
    for lr in refinement:
        assert 5e-6 <= lr <= 1e-4, f"Refinement {lr:.2e} outside reasonable range"

    # Should be sorted
    assert refinement == sorted(refinement)

    print(f"  Refinement LRs: {[f'{c:.2e}' for c in refinement]}")
    print("✓ test_refinement_narrows_around_best PASSED\n")


def test_refinement_handles_edge_best():
    """When best is at the edge of coarse range, refinement extends beyond."""
    # Best is the lowest LR tried
    coarse_results = [
        TrialResult(lr=5e-6, final_loss=2.0, initial_loss=3.0, loss_curve=[3.0, 2.5, 2.0]),  # best (lowest)
        TrialResult(lr=1e-5, final_loss=2.3, initial_loss=3.0, loss_curve=[3.0, 2.6, 2.3]),
        TrialResult(lr=2e-5, final_loss=2.8, initial_loss=3.0, loss_curve=[3.0, 2.9, 2.8]),
    ]

    refinement = generate_refinement_candidates(coarse_results, n_refine=3)

    # Should explore below 5e-6 since best was at the edge
    assert any(lr < 5e-6 for lr in refinement), (
        f"Expected some candidates below 5e-6, got {[f'{c:.2e}' for c in refinement]}"
    )

    print(f"  Edge refinement: {[f'{c:.2e}' for c in refinement]}")
    print("✓ test_refinement_handles_edge_best PASSED\n")


def test_refinement_returns_empty_for_all_diverged():
    """When all trials diverged, refinement returns empty list."""
    all_bad = [
        TrialResult(lr=1e-5, final_loss=float("inf"), initial_loss=3.0, diverged=True),
        TrialResult(lr=2e-5, final_loss=float("nan"), initial_loss=3.0, diverged=True),
    ]

    refinement = generate_refinement_candidates(all_bad)
    assert refinement == []
    print("✓ test_refinement_returns_empty_for_all_diverged PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# HORIZON CORRECTION
# ══════════════════════════════════════════════════════════════════════════════


def test_horizon_exponent_from_flat_curve():
    """Flat loss curve (high curvature) → higher correction exponent."""
    # Loss flattens quickly: 3.0 → 2.5 → 2.4 (decelerating)
    flat_results = [
        TrialResult(lr=2e-5, final_loss=2.4, initial_loss=3.0, loss_curve=[3.0, 2.5, 2.45, 2.42, 2.4]),
    ]
    alpha_flat = estimate_horizon_exponent(flat_results)

    # Loss still dropping: 3.0 → 2.5 → 2.0 (linear/accelerating)
    linear_results = [
        TrialResult(lr=2e-5, final_loss=2.0, initial_loss=3.0, loss_curve=[3.0, 2.5, 2.0, 1.5, 1.0]),
    ]
    alpha_linear = estimate_horizon_exponent(linear_results)

    # Flat curve should get higher alpha (stronger correction)
    assert alpha_flat > alpha_linear, (
        f"Flat curve α={alpha_flat:.3f} should be > linear α={alpha_linear:.3f}"
    )
    # Both should be in valid range
    assert 0.05 <= alpha_flat <= 0.25
    assert 0.05 <= alpha_linear <= 0.25

    print(f"  Flat curve α: {alpha_flat:.3f}")
    print(f"  Linear curve α: {alpha_linear:.3f}")
    print("✓ test_horizon_exponent_from_flat_curve PASSED\n")


def test_horizon_exponent_fallback():
    """Insufficient data falls back to literature default 0.088."""
    # Only 1 result with too-short loss_curve (< 3 points)
    results = [TrialResult(lr=2e-5, final_loss=2.0, initial_loss=3.0, loss_curve=[3.0, 2.0])]
    alpha = estimate_horizon_exponent(results)
    assert alpha == 0.088, f"Expected fallback 0.088, got {alpha}"
    print("✓ test_horizon_exponent_fallback PASSED\n")


def test_adaptive_correction_bounded():
    """Adaptive correction never goes beyond 5x in either direction."""
    results = [
        TrialResult(lr=2e-5, final_loss=2.0, initial_loss=3.0,
                    loss_curve=[3.0, 2.8, 2.6, 2.4, 2.2, 2.0]),
    ]

    # Very long horizon (should correct down significantly)
    corrected = correct_lr_adaptive(2e-5, sweep_steps=100, full_steps=1_000_000, results=results)
    assert corrected >= 2e-5 / 5, f"Correction too aggressive: {corrected:.2e}"

    # Very short horizon (no correction needed)
    corrected_short = correct_lr_adaptive(2e-5, sweep_steps=100, full_steps=100, results=results)
    assert corrected_short == 2e-5, "Should return same LR when full <= sweep"

    print(f"  100 → 1M steps: {2e-5:.2e} → {corrected:.2e}")
    print(f"  100 → 100 steps: {2e-5:.2e} → {corrected_short:.2e}")
    print("✓ test_adaptive_correction_bounded PASSED\n")


def test_correction_is_monotonic_with_horizon():
    """Longer horizons always produce lower corrected LR."""
    results = [
        TrialResult(lr=3e-5, final_loss=2.0, initial_loss=3.0,
                    loss_curve=[3.0, 2.7, 2.4, 2.2, 2.0]),
    ]

    corrected_1k = correct_lr_adaptive(3e-5, 100, 1000, results)
    corrected_5k = correct_lr_adaptive(3e-5, 100, 5000, results)
    corrected_20k = correct_lr_adaptive(3e-5, 100, 20000, results)

    assert corrected_1k >= corrected_5k >= corrected_20k, (
        f"Not monotonic: 1k={corrected_1k:.2e}, 5k={corrected_5k:.2e}, 20k={corrected_20k:.2e}"
    )

    print(f"  100→1K: {corrected_1k:.2e}")
    print(f"  100→5K: {corrected_5k:.2e}")
    print(f"  100→20K: {corrected_20k:.2e}")
    print("✓ test_correction_is_monotonic_with_horizon PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════


def test_early_stop_on_divergence():
    """Diverging loss curve triggers early stop."""
    # NaN at end
    assert should_early_stop_trial([2.0, 1.9, 1.8, float("nan")], step=4, total_steps=10)

    # Increasing loss for > 30% of steps
    curve = [2.0, 1.9, 1.8, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5]
    assert should_early_stop_trial(curve, step=10, total_steps=20)

    print("✓ test_early_stop_on_divergence PASSED\n")


def test_early_stop_worse_than_best():
    """Trial already worse than best known → stop early."""
    curve = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5]  # 6 steps out of 10
    # Best known final is 2.0 — this trial at midpoint is already 2.5
    assert should_early_stop_trial(curve, step=6, total_steps=10, best_known_final=2.0)

    # But not if best known is worse
    assert not should_early_stop_trial(curve, step=6, total_steps=10, best_known_final=3.0)

    print("✓ test_early_stop_worse_than_best PASSED\n")


def test_early_stop_no_progress():
    """No loss decrease after 25% of steps → stop."""
    # Stuck curve: loss barely changes
    curve = [3.0, 3.0, 2.99, 3.01, 3.0]
    assert should_early_stop_trial(curve, step=5, total_steps=10)

    print("✓ test_early_stop_no_progress PASSED\n")


def test_no_early_stop_healthy_trial():
    """Healthy decreasing loss should NOT trigger early stop."""
    curve = [3.0, 2.8, 2.6, 2.4, 2.2, 2.0]
    assert not should_early_stop_trial(curve, step=6, total_steps=20)
    assert not should_early_stop_trial(curve, step=6, total_steps=20, best_known_final=2.5)

    print("✓ test_no_early_stop_healthy_trial PASSED\n")


def test_no_early_stop_too_few_steps():
    """Don't stop if we haven't reached minimum steps yet."""
    # Even diverging, too early to tell
    curve = [3.0, 3.5]
    assert not should_early_stop_trial(curve, step=2, total_steps=100)

    print("✓ test_no_early_stop_too_few_steps PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# FULL ORCHESTRATOR (with mock trial runner)
# ══════════════════════════════════════════════════════════════════════════════


def test_adaptive_sweep_end_to_end():
    """Full adaptive sweep with synthetic trial runner finds reasonable LR."""

    # Simulate a loss landscape where LR=3e-5 is optimal
    def mock_trial(lr: float, steps: int) -> TrialResult:
        """Synthetic trial: optimal at 3e-5, diverges above 1e-4."""
        # Loss = base + distance_from_optimal * penalty
        log_lr = math.log10(lr)
        log_optimal = math.log10(3e-5)
        distance = abs(log_lr - log_optimal)

        initial_loss = 3.0
        if lr > 1e-4:
            # Divergence
            return TrialResult(
                lr=lr, final_loss=float("inf"), initial_loss=initial_loss,
                loss_curve=[initial_loss, initial_loss * 1.5, float("inf")],
                steps_completed=steps // 3, diverged=True,
            )

        # Closer to optimal → lower final loss
        final_loss = initial_loss * (0.6 + 0.3 * distance)
        # Build synthetic curve
        curve = [initial_loss - (initial_loss - final_loss) * (i / steps) for i in range(steps)]

        return TrialResult(
            lr=lr, final_loss=final_loss, initial_loss=initial_loss,
            loss_curve=curve, steps_completed=steps, diverged=False,
        )

    best_lr, all_results = adaptive_lr_sweep(
        run_trial_fn=mock_trial,
        model_params_b=4.0,
        effective_batch_tokens=524288,
        steps_per_trial=50,
        full_training_steps=5000,
        n_coarse=5,
        n_refine=3,
    )

    # Best LR should be in the neighborhood of 3e-5 (within 1 decade)
    assert 3e-6 <= best_lr <= 3e-4, f"Best LR {best_lr:.2e} too far from optimal 3e-5"

    # Should have run coarse + refinement trials
    assert len(all_results) >= 5, f"Expected >= 5 trials, got {len(all_results)}"

    # Divergent trial should be marked
    diverged = [r for r in all_results if r.diverged]
    valid = [r for r in all_results if not r.diverged]
    assert len(valid) >= 4, "Most trials should succeed"

    print(f"  Best LR found: {best_lr:.2e}")
    print(f"  Total trials: {len(all_results)} ({len(valid)} valid, {len(diverged)} diverged)")
    print("✓ test_adaptive_sweep_end_to_end PASSED\n")


def test_adaptive_sweep_all_diverged_fallback():
    """When all trials diverge, returns conservative fallback LR."""

    def always_diverge(lr: float, steps: int) -> TrialResult:
        return TrialResult(
            lr=lr, final_loss=float("inf"), initial_loss=3.0,
            loss_curve=[3.0, float("inf")], diverged=True,
        )

    best_lr, results = adaptive_lr_sweep(
        run_trial_fn=always_diverge,
        model_params_b=4.0,
        effective_batch_tokens=524288,
        steps_per_trial=50,
        full_training_steps=5000,
    )

    # Should return safe fallback
    assert best_lr == 2e-5, f"Expected 2e-5 fallback, got {best_lr:.2e}"
    print(f"  All-diverged fallback: {best_lr:.2e}")
    print("✓ test_adaptive_sweep_all_diverged_fallback PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRIAL RESULT PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════


def test_trial_result_curvature():
    """Curvature property correctly identifies loss curve shape."""
    # Decelerating (flattening): positive curvature
    decelerating = TrialResult(
        lr=2e-5, final_loss=2.4, initial_loss=3.0,
        loss_curve=[3.0, 2.6, 2.5, 2.45, 2.4],
    )
    assert decelerating.curvature > 0, f"Expected positive curvature, got {decelerating.curvature}"

    # Accelerating (still dropping fast at end): negative curvature
    accelerating = TrialResult(
        lr=2e-5, final_loss=1.0, initial_loss=3.0,
        loss_curve=[3.0, 2.9, 2.7, 2.2, 1.0],
    )
    assert accelerating.curvature < 0, f"Expected negative curvature, got {accelerating.curvature}"

    # Linear: near-zero curvature (equal spacing)
    linear = TrialResult(
        lr=2e-5, final_loss=1.0, initial_loss=3.0,
        loss_curve=[3.0, 2.5, 2.0, 1.5, 1.0],
    )
    assert abs(linear.curvature) < 0.05, f"Expected ~0 curvature, got {linear.curvature}"

    print(f"  Decelerating curvature: {decelerating.curvature:.4f}")
    print(f"  Accelerating curvature: {accelerating.curvature:.4f}")
    print(f"  Linear curvature: {linear.curvature:.4f}")
    print("✓ test_trial_result_curvature PASSED\n")


def test_trial_result_score_penalizes_diverged():
    """Diverged trials get infinite score."""
    good = TrialResult(lr=2e-5, final_loss=2.0, initial_loss=3.0, loss_curve=[3.0, 2.0])
    bad = TrialResult(lr=2e-5, final_loss=float("inf"), initial_loss=3.0, diverged=True)

    assert good.score < float("inf")
    assert bad.score == float("inf")
    assert good.score < bad.score

    print("✓ test_trial_result_score_penalizes_diverged PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: profile.correct_lr_for_horizon
# ══════════════════════════════════════════════════════════════════════════════


def test_profile_correction_backward_compat():
    """The updated correct_lr_for_horizon still works with old call signature."""
    from palingenesis.autopilot.profile import correct_lr_for_horizon

    # Default alpha (no explicit arg)
    corrected = correct_lr_for_horizon(2e-5, sweep_steps=100, full_steps=5000)
    assert 5e-6 < corrected < 2e-5, f"Default correction out of range: {corrected:.2e}"

    # Explicit alpha
    corrected_strong = correct_lr_for_horizon(2e-5, sweep_steps=100, full_steps=5000, alpha=0.25)
    corrected_weak = correct_lr_for_horizon(2e-5, sweep_steps=100, full_steps=5000, alpha=0.05)
    assert corrected_strong < corrected_weak, "Stronger alpha should produce lower LR"

    # No correction when full <= sweep
    same = correct_lr_for_horizon(2e-5, sweep_steps=100, full_steps=50)
    assert same == 2e-5

    print(f"  Default α: {corrected:.2e}")
    print(f"  Strong α=0.25: {corrected_strong:.2e}")
    print(f"  Weak α=0.05: {corrected_weak:.2e}")
    print("✓ test_profile_correction_backward_compat PASSED\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("=" * 70)
    print("PALINGENESIS — ADAPTIVE SWEEP TESTS")
    print("=" * 70 + "\n")

    print("── Candidate Generation ──\n")
    test_coarse_candidates_cover_reasonable_range()
    test_coarse_candidates_scale_with_model_size()
    test_coarse_candidates_scale_with_batch_size()
    test_surge_aware_batch_factor()
    test_coarse_candidates_hardware_params()

    print("── Refinement ──\n")
    test_refinement_narrows_around_best()
    test_refinement_handles_edge_best()
    test_refinement_returns_empty_for_all_diverged()

    print("── Horizon Correction ──\n")
    test_horizon_exponent_from_flat_curve()
    test_horizon_exponent_fallback()
    test_adaptive_correction_bounded()
    test_correction_is_monotonic_with_horizon()

    print("── Early Stopping ──\n")
    test_early_stop_on_divergence()
    test_early_stop_worse_than_best()
    test_early_stop_no_progress()
    test_no_early_stop_healthy_trial()
    test_no_early_stop_too_few_steps()

    print("── Full Orchestrator ──\n")
    test_adaptive_sweep_end_to_end()
    test_adaptive_sweep_all_diverged_fallback()

    print("── Trial Properties ──\n")
    test_trial_result_curvature()
    test_trial_result_score_penalizes_diverged()

    print("── Backward Compatibility ──\n")
    test_profile_correction_backward_compat()

    print("=" * 70)
    print("ALL ADAPTIVE SWEEP TESTS PASSED ✓")
    print("=" * 70)
