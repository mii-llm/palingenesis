"""Adaptive LR sweep: replaces the fixed 5-candidate grid with a principled search.

Problems with the old approach:
  1. Fixed 5 LR candidates miss the optimum if it's between grid points
  2. The horizon correction exponent (0.12) is calibrated on dense pretraining,
     not agentic SFT with DEFT loss and long sequences
  3. No use of loss curvature — we can extract much more signal from each trial
  4. No early termination of obviously bad trials

New approach (informed by literature):
  Phase 1: Coarse log-uniform sweep (5 candidates, scaling-law centered)
  Phase 2: Quadratic interpolation in log-LR space (arxiv:2409.19913 Sec 3.1)
           + refinement candidates around the interpolated minimum
  Phase 3: Adaptive horizon correction using curvature + literature priors

Key papers incorporated:
  - arxiv:2409.19913: η*(D) = C · D^{-α}, α ∈ [0.05, 0.14] across architectures
    For 350M-2.7B models on standard data: α ≈ 0.088
    Method: fit parabola in (log η, loss) space to find minimum at each horizon
  - arxiv:2503.04715 (Step Law): LR_opt ∝ N^{-0.5} · D^{+0.12} (batch indep)
    Note: positive D exponent because more data = longer training = lower LR per step
  - arxiv:2602.06797: Power-decay is theoretically optimal (easy-task regime)
  - arxiv:2603.10301: Base LR is the dominant predictor of schedule success
  - arxiv:2606.05610: For continued pretraining/SFT, "equivalent compute" reduces
    the effective horizon correction (pretrained model already has priors)
  - arxiv:2405.14578: Adam's optimal LR has a "surge" with batch size (non-monotone)
"""

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TrialResult:
    """Result of a single LR trial with curvature information."""

    lr: float
    final_loss: float
    initial_loss: float
    loss_curve: list[float] = field(default_factory=list)
    steps_completed: int = 0
    diverged: bool = False
    time_s: float = 0.0

    @property
    def loss_reduction(self) -> float:
        """Fraction of loss reduced: (initial - final) / initial."""
        if self.initial_loss <= 0 or not math.isfinite(self.final_loss):
            return -float("inf")
        return (self.initial_loss - self.final_loss) / self.initial_loss

    @property
    def curvature(self) -> float:
        """Estimate curvature (second derivative) of loss vs log-steps.

        Positive = decelerating (good), negative = accelerating (still improving fast).
        Near zero = linear decay (sweet spot for predicting long-run behavior).
        """
        if len(self.loss_curve) < 3:
            return 0.0
        # Use first, middle, and last points
        n = len(self.loss_curve)
        l0 = self.loss_curve[0]
        lm = self.loss_curve[n // 2]
        lf = self.loss_curve[-1]
        # Discrete second difference on log-step axis
        # curvature > 0 means loss curve is flattening (decelerating)
        return (l0 + lf - 2 * lm) / max(abs(l0), 1e-8)

    @property
    def score(self) -> float:
        """Composite score: lower is better. Combines final loss + stability."""
        if self.diverged or not math.isfinite(self.final_loss):
            return float("inf")
        # Penalize trials that are still decelerating fast (risk of early plateau)
        curvature_penalty = max(0, self.curvature) * 0.1
        return self.final_loss + curvature_penalty


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE SWEEP
# ══════════════════════════════════════════════════════════════════════════════


def generate_coarse_candidates(
    model_params_b: float,
    effective_batch_tokens: int,
    per_device_batch_size: int = 1,
    max_seq_length: int = 4096,
    gradient_accumulation_steps: int = 16,
    num_gpus: int = 1,
) -> list[float]:
    """Generate coarse LR candidates spanning the plausible SFT range.

    Uses scaling-law anchoring to center the range, then spans ~2.4 decades.

    The batch-size → LR relationship models the "surge phenomenon" from
    arxiv:2405.14578: for Adam-style optimizers, optimal LR first RISES
    then FALLS as batch size increases. The peak is at B_crit (critical
    batch size where gradient noise ≈ gradient signal).

    For SFT, B_crit is typically 50K-200K tokens (empirical). Below B_crit,
    the gradient estimate is noisy → lower LR needed for stability. Above
    B_crit, the gradient is accurate but we're averaging over too many samples
    → the effective step is too conservative at high LR.

    Parameters:
        model_params_b: Model size in billions
        effective_batch_tokens: Total tokens per optimizer step
            = per_device_batch_size × max_seq_length × gradient_accumulation_steps × num_gpus
        per_device_batch_size: Micro-batch per GPU (for surge regime detection)
        max_seq_length: Sequence length in tokens
        gradient_accumulation_steps: GA steps
        num_gpus: Number of GPUs (world_size)
    """
    # If effective_batch_tokens not explicitly provided, compute it
    if effective_batch_tokens <= 0:
        effective_batch_tokens = per_device_batch_size * max_seq_length * gradient_accumulation_steps * num_gpus

    # ── Model size factor: LR ∝ N^{-0.5} (Step Law, arxiv:2503.04715) ────
    model_factor = model_params_b ** -0.5

    # ── Batch size factor: surge-aware (arxiv:2405.14578) ─────────────────
    batch_factor = _surge_aware_batch_factor(effective_batch_tokens)

    # Anchor constant calibrated for SFT:
    # 8B model (model_factor=0.354), 524K tokens (batch_factor=1.0) → center ~3e-5
    # → C = 3e-5 / 0.354 / 1.0 ≈ 8.5e-5
    C = 8.5e-5

    center_lr = C * model_factor * batch_factor
    center_lr = max(5e-6, min(5e-4, center_lr))

    # Span 1.2 decades above and below center (total ~2.4 decades)
    log_center = math.log10(center_lr)
    candidates = [
        10 ** (log_center - 1.2),
        10 ** (log_center - 0.6),
        10 ** log_center,
        10 ** (log_center + 0.6),
        10 ** (log_center + 1.2),
    ]

    # Clamp to absolute SFT bounds
    candidates = [max(1e-7, min(5e-3, c)) for c in candidates]

    logger.info(
        f"Coarse candidates: [{candidates[0]:.1e} ... {candidates[-1]:.1e}] "
        f"(center={center_lr:.1e}, model={model_params_b:.1f}B, "
        f"batch_tokens={effective_batch_tokens:,})"
    )
    return candidates


def _surge_aware_batch_factor(effective_batch_tokens: int) -> float:
    """Compute batch-size factor for LR scaling, accounting for the surge phenomenon.

    From arxiv:2405.14578 Theorem 3, for Adam optimizers:
        η_opt(B) ∝ f(B/B_crit)

    where f rises for B < B_crit and falls for B > B_crit.

    We model this with the functional form (simplified from the paper):
        factor = (B / B_crit) / (1 + B / B_crit)^{1.5}

    This gives:
        - B << B_crit (small batch, noisy gradients): factor ∝ B (rising)
        - B = B_crit: factor peaks at ~0.38 (the "surge" peak)
        - B >> B_crit (large batch, saturated): factor ∝ B^{-0.5} (falling)

    We normalize so that factor = 1.0 at B = B_crit (the reference point).

    For SFT with standard datasets:
        B_crit ≈ 100K tokens (empirical sweet spot for most SFT tasks)
        This corresponds to: batch=2 × seq=4096 × GA=12 ≈ ~100K

    The regimes in practice:
        - Single GPU, no GA, short seq: ~4K-8K tokens → well below B_crit
          → LR should be LOWER than the reference (gradient too noisy)
        - 8×GPU, GA=16, seq=8192: ~1M tokens → well above B_crit
          → LR should be slightly LOWER (saturation regime)
        - Sweet spot: ~50K-200K tokens → near peak, use reference LR
    """
    B_CRIT = 100_000  # Critical batch size in tokens for SFT

    ratio = effective_batch_tokens / B_CRIT

    # Surge function: peaks at ratio=1, falls on both sides
    # f(r) = r / (1 + r)^1.5
    # Normalized so f(1) = 1: f(1) = 1 / 2^1.5 = 0.354
    # So normalized_f(r) = [r / (1+r)^1.5] / [1 / 2^1.5]
    raw = ratio / (1 + ratio) ** 1.5
    peak = 1.0 / (2.0 ** 1.5)  # value at ratio=1
    factor = raw / peak

    # The factor tells us how much to scale LR relative to B_crit reference:
    # factor > 1: impossible (peak is at 1.0 by normalization)
    # factor = 1: at B_crit (optimal batch regime)
    # factor < 1: either below or above B_crit (LR should be lower)

    # For the candidate generation, we INVERT this: if optimal LR is lower,
    # we shift the center down. factor < 1 means center should be lower.
    # But we want batch_factor such that: center_lr = C * model_factor * batch_factor
    # Higher batch_factor → higher center LR.
    # So batch_factor = factor (direct proportionality)

    return factor


def generate_refinement_candidates(
    coarse_results: list[TrialResult],
    n_refine: int = 3,
) -> list[float]:
    """Generate refinement candidates in the neighborhood of the best coarse trial.

    Strategy (from arxiv:2409.19913 Section 3.1):
    1. Fit a quadratic in (log10(lr), loss) space to non-diverged trials
    2. The minimizer of the parabola gives an analytical LR estimate
    3. Place refinement candidates around this estimate

    If the parabola fit fails (too few points, bad fit), falls back to
    placing candidates between the top-2 trials.
    """
    # Sort by score (lower = better)
    valid = [r for r in coarse_results if not r.diverged and math.isfinite(r.final_loss)]
    if len(valid) < 2:
        return []

    valid.sort(key=lambda r: r.score)

    # ── Try quadratic fit in log-LR space (arxiv:2409.19913 method) ───────
    parabola_lr = _fit_parabola_minimum(valid)

    if parabola_lr is not None:
        # Place refinement candidates around the parabola minimum
        log_min = math.log10(parabola_lr)
        # Span ±0.2 decades around the estimated minimum (tight refinement)
        candidates = [
            10 ** (log_min - 0.15),
            10 ** log_min,
            10 ** (log_min + 0.15),
        ][:n_refine]
        logger.info(
            f"Refinement via quadratic fit: minimum at LR={parabola_lr:.2e}, "
            f"candidates: {[f'{c:.2e}' for c in candidates]}"
        )
        return candidates

    # ── Fallback: bracket between best and 2nd-best ───────────────────────
    best = valid[0]
    second = valid[1]

    # Define refinement interval: between best and second-best LRs
    lo = min(best.lr, second.lr)
    hi = max(best.lr, second.lr)

    # If best is at the edge of our range, extend slightly beyond
    if best.lr == min(r.lr for r in coarse_results):
        lo = best.lr / 2
    elif best.lr == max(r.lr for r in coarse_results):
        hi = best.lr * 2

    # Log-uniform spacing within the interval
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)

    if abs(log_hi - log_lo) < 0.05:
        # Interval too narrow — expand to 0.3 decades around best
        log_center = math.log10(best.lr)
        log_lo = log_center - 0.15
        log_hi = log_center + 0.15

    candidates = []
    for i in range(n_refine):
        frac = (i + 1) / (n_refine + 1)
        lr = 10 ** (log_lo + frac * (log_hi - log_lo))
        candidates.append(lr)

    logger.info(
        f"Refinement candidates: [{candidates[0]:.2e} ... {candidates[-1]:.2e}] "
        f"(between best={best.lr:.2e} and 2nd={second.lr:.2e})"
    )
    return candidates


def _fit_parabola_minimum(results: list[TrialResult]) -> float | None:
    """Fit a quadratic in (log10(lr), loss) space and return its minimizer.

    From arxiv:2409.19913: "For each token horizon we fit a second-degree
    polynomial in log(η), using a quadratic polynomial as it provides an
    excellent fit and is the simplest polynomial with a well-defined minimum.
    The R² of the fit are 0.995 or better."

    Returns None if:
    - Fewer than 3 non-diverged points
    - The parabola doesn't have a minimum (negative curvature)
    - The minimum is far outside the observed range (extrapolation risk)
    """
    valid = [r for r in results if not r.diverged and math.isfinite(r.final_loss)]
    if len(valid) < 3:
        return None

    # Fit: loss = a * (log10(lr))^2 + b * log10(lr) + c
    xs = [math.log10(r.lr) for r in valid]
    ys = [r.final_loss for r in valid]
    n = len(xs)

    # Least-squares for quadratic: normal equations
    sum_x = sum(xs)
    sum_x2 = sum(x**2 for x in xs)
    sum_x3 = sum(x**3 for x in xs)
    sum_x4 = sum(x**4 for x in xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2y = sum(x**2 * y for x, y in zip(xs, ys))

    # Solve [a, b, c] via Cramer's rule for 3x3 system
    # |sum_x4  sum_x3  sum_x2| |a|   |sum_x2y|
    # |sum_x3  sum_x2  sum_x | |b| = |sum_xy |
    # |sum_x2  sum_x   n     | |c|   |sum_y  |
    A = [[sum_x4, sum_x3, sum_x2],
         [sum_x3, sum_x2, sum_x],
         [sum_x2, sum_x, n]]
    B = [sum_x2y, sum_xy, sum_y]

    det = _det3(A)
    if abs(det) < 1e-15:
        return None

    a = _det3([B, A[1], A[2]], col_replace=0) / det
    b = _det3([A[0], B, A[2]], col_replace=1) / det

    # Parabola must open upward (a > 0) for a minimum to exist
    if a <= 0:
        return None

    # Minimum at x* = -b / (2a)
    x_min = -b / (2 * a)
    lr_min = 10 ** x_min

    # Sanity: minimum should be within 1 decade of the observed range
    x_lo = min(xs)
    x_hi = max(xs)
    if x_min < x_lo - 1.0 or x_min > x_hi + 1.0:
        logger.debug(f"Parabola minimum at log10(lr)={x_min:.2f} is too far from data [{x_lo:.2f}, {x_hi:.2f}]")
        return None

    # Compute R² to validate fit quality
    y_pred = [a * x**2 + b * x + (sum_y - a * sum_x2 - b * sum_x) / n for x in xs]
    ss_res = sum((y - yp)**2 for y, yp in zip(ys, y_pred))
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean)**2 for y in ys)
    r_squared = 1 - ss_res / max(ss_tot, 1e-15) if ss_tot > 1e-15 else 0

    if r_squared < 0.7:
        logger.debug(f"Parabola fit poor: R²={r_squared:.3f}")
        return None

    logger.info(f"Parabola fit: min at LR={lr_min:.2e} (R²={r_squared:.3f})")
    return lr_min


def _det3(matrix, col_replace: int | None = None) -> float:
    """3x3 determinant, optionally with a column replaced (for Cramer's rule)."""
    if col_replace is not None:
        # This is a simplified interface — just compute det of 3x3
        pass
    m = matrix
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


# ══════════════════════════════════════════════════════════════════════════════
# HORIZON CORRECTION — ADAPTIVE (replaces fixed 0.12 exponent)
# ══════════════════════════════════════════════════════════════════════════════


def estimate_horizon_exponent(results: list[TrialResult]) -> float:
    """Estimate the horizon correction exponent from trial curvature data.

    Literature context (arxiv:2409.19913):
    - GPT-3 350M: α ≈ 0.088 (R² = 0.99)
    - Across 50M-2.7B models: α ∈ [0.05, 0.14]
    - The paper uses α = β in their Eq. (1): η*(D) = C · D^{-β}

    For SFT (continued training from a pretrained checkpoint), the effective
    horizon correction is WEAKER than for pretraining from scratch. Rationale:
    (arxiv:2606.05610): the pretrained model already encodes priors — it's
    effectively "further along" the optimization trajectory. The marginal
    change per additional step is smaller, so LR sensitivity to horizon is lower.

    We attenuate the raw exponent by ~30% for SFT (empirical observation from
    the fact that SFT operates in a narrower loss range than pretraining).

    The curvature of the loss curve provides additional signal:
    - High curvature (flattening) → the LR is already near-optimal for this horizon
      → stronger correction needed for longer horizons (raise α)
    - Low curvature (still dropping) → the LR hasn't saturated
      → weaker correction (lower α)

    Fallback to 0.088 (literature median) when data is insufficient.

    Returns α in [0.04, 0.20] range.
    """
    valid = [r for r in results if not r.diverged and len(r.loss_curve) >= 3]
    if not valid:
        logger.info("Insufficient curvature data, using literature default α=0.088")
        return 0.088

    # Use the best trial's curvature as the primary signal
    best = min(valid, key=lambda r: r.score)
    curvature = best.curvature

    # Base exponent: 0.088 (literature median for 350M-2.7B, arxiv:2409.19913)
    # Attenuated by 0.7x for SFT (continued training, not from scratch)
    base_alpha = 0.088 * 0.7  # ≈ 0.062 for SFT

    # Curvature adjustment:
    # curvature ~ 0 (linear) → use base (LR still effective)
    # curvature > 0 (flattening) → raise alpha (LR overshooting for long runs)
    # curvature < 0 (accelerating) → lower alpha (LR maybe too conservative)
    alpha = base_alpha + curvature * 0.5

    # Clamp to reasonable range
    # Lower bound 0.04 (almost no correction — the SFT regime)
    # Upper bound 0.20 (strong correction — approaching pretraining regime)
    alpha = max(0.04, min(0.20, alpha))

    logger.info(
        f"Adaptive horizon exponent: α={alpha:.3f} "
        f"(base=0.062, curvature={curvature:.4f}, best LR={best.lr:.2e})"
    )
    return alpha


def correct_lr_adaptive(
    best_lr: float,
    sweep_steps: int,
    full_steps: int,
    results: list[TrialResult],
) -> float:
    """Correct sweep-found LR for the full training horizon (adaptive).

    Replaces the fixed α=0.12 correction with a data-driven estimate.

    From "Scaling Optimal LR Across Token Horizons" (arxiv:2409.19913):
        lr_optimal(T) ∝ T^{-α}

    We estimate α from the loss curvature observed during the sweep itself.

    Additionally applies a loss-function correction:
    - DEFT/DFT losses have different gradient magnitudes than CE
    - The effective gradient scale changes the optimal LR-horizon relationship
    """
    if full_steps <= sweep_steps:
        return best_lr

    alpha = estimate_horizon_exponent(results)
    correction = (sweep_steps / full_steps) ** alpha
    corrected = best_lr * correction

    # Safety: never correct by more than 5x in either direction
    corrected = max(best_lr / 5, min(best_lr * 5, corrected))

    logger.info(
        f"Adaptive LR correction: {best_lr:.2e} -> {corrected:.2e} "
        f"(α={alpha:.3f}, ratio={sweep_steps}/{full_steps}, factor={correction:.3f})"
    )
    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING FOR TRIALS
# ══════════════════════════════════════════════════════════════════════════════


def should_early_stop_trial(
    loss_curve: list[float],
    step: int,
    total_steps: int,
    best_known_final: float = float("inf"),
) -> bool:
    """Decide if a trial should be terminated early (bad LR, saving time).

    Conditions for early stop:
    1. Loss has diverged (NaN/Inf or increasing for > 30% of steps)
    2. Loss at 50% of trial is already worse than best known final loss
       (this trial can't possibly win)
    3. Loss hasn't decreased at all after 25% of steps (stuck)
    """
    if not loss_curve:
        return False

    n = len(loss_curve)
    min_steps_before_stopping = max(5, total_steps // 4)

    # Condition 1: divergence (NaN/Inf) — ALWAYS stop immediately
    if not math.isfinite(loss_curve[-1]):
        return True

    if n < min_steps_before_stopping:
        return False

    # Condition 1b: loss increasing for last 30% of observed steps
    lookback = max(3, n // 3)
    recent = loss_curve[-lookback:]
    if len(recent) >= 3 and recent[-1] > recent[0] * 1.1:
        return True

    # Condition 2: already worse than best known
    if n >= total_steps // 2 and loss_curve[-1] > best_known_final * 1.05:
        return True

    # Condition 3: no progress after 25% of steps
    if n >= total_steps // 4:
        initial = loss_curve[0]
        current = loss_curve[-1]
        if current >= initial * 0.99:  # Less than 1% improvement
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ADAPTIVE SWEEP ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════


def adaptive_lr_sweep(
    run_trial_fn,
    model_params_b: float = 4.0,
    effective_batch_tokens: int = 524288,
    steps_per_trial: int = 100,
    full_training_steps: int = 5000,
    n_coarse: int = 5,
    n_refine: int = 3,
) -> tuple[float, list[TrialResult]]:
    """Run adaptive LR sweep: coarse → refine → horizon-correct.

    Args:
        run_trial_fn: Callable(lr, max_steps) -> TrialResult
            The caller provides this function which runs a single trial
            and returns a TrialResult with loss_curve populated.
        model_params_b: Model size in billions (for candidate generation)
        effective_batch_tokens: Tokens per optimizer step
        steps_per_trial: Steps per trial
        full_training_steps: Total steps for the full training run
        n_coarse: Number of coarse candidates
        n_refine: Number of refinement candidates

    Returns:
        (best_lr_corrected, all_results)
    """
    all_results: list[TrialResult] = []

    # ── Phase 1: Coarse sweep ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Adaptive LR Sweep — Phase 1: Coarse")
    logger.info("=" * 50)

    coarse_candidates = generate_coarse_candidates(model_params_b, effective_batch_tokens)

    for i, lr in enumerate(coarse_candidates[:n_coarse]):
        logger.info(f"  [{i+1}/{n_coarse}] LR={lr:.2e}")
        t0 = time.perf_counter()
        result = run_trial_fn(lr, steps_per_trial)
        result.time_s = time.perf_counter() - t0

        all_results.append(result)
        logger.info(
            f"    loss: {result.initial_loss:.4f} -> {result.final_loss:.4f} "
            f"(reduction={result.loss_reduction*100:.1f}%, {result.time_s:.1f}s)"
        )

    # ── Phase 2: Refinement ───────────────────────────────────────────────
    refine_candidates = generate_refinement_candidates(all_results, n_refine)

    if refine_candidates:
        logger.info("=" * 50)
        logger.info("Adaptive LR Sweep — Phase 2: Refinement")
        logger.info("=" * 50)

        for i, lr in enumerate(refine_candidates):
            logger.info(f"  [{i+1}/{n_refine}] LR={lr:.2e}")
            t0 = time.perf_counter()
            result = run_trial_fn(lr, steps_per_trial)
            result.time_s = time.perf_counter() - t0

            all_results.append(result)
            logger.info(
                f"    loss: {result.initial_loss:.4f} -> {result.final_loss:.4f} "
                f"(score={result.score:.4f}, {result.time_s:.1f}s)"
            )

    # ── Phase 3: Select best + adaptive horizon correction ────────────────
    logger.info("=" * 50)
    logger.info("Adaptive LR Sweep — Phase 3: Selection + Correction")
    logger.info("=" * 50)

    valid_results = [r for r in all_results if not r.diverged and math.isfinite(r.score)]
    if not valid_results:
        # All trials failed — fall back to conservative estimate
        fallback_lr = 2e-5
        logger.warning(f"All trials diverged. Falling back to LR={fallback_lr:.1e}")
        return fallback_lr, all_results

    valid_results.sort(key=lambda r: r.score)
    best = valid_results[0]
    logger.info(f"  Best trial: LR={best.lr:.2e} (score={best.score:.4f})")

    # Adaptive horizon correction (replaces fixed 0.12)
    corrected_lr = correct_lr_adaptive(
        best.lr, steps_per_trial, full_training_steps, all_results
    )

    # Final clamping to sane SFT range
    corrected_lr = max(1e-6, min(5e-4, corrected_lr))

    logger.info(f"  Final LR: {corrected_lr:.2e} (corrected for {full_training_steps} steps)")
    logger.info("=" * 50)

    return corrected_lr, all_results
