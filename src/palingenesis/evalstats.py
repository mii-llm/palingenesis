"""Evaluation statistics in the standard library only (no torch, numpy or scipy): unbiased pass@k, its per-sample
reward transform for training, confidence intervals and paired tests for comparing two checkpoints on the same items.

    pass_at_k(n, c, k)                 unbiased pass@k of one item from n samples with c successes (Chen et al. 2021,
                                       arXiv:2107.03374): 1 - C(n - c, k) / C(n, k)
    max_at_k(scores, k)                the unbiased estimate of E[max of k samples] for any scores (pass@k when they are
                                       0 / 1): the mean of max over every k-subset of the n samples
    pass_at_k_rewards(scores, k)       per-sample rewards whose mean is max_at_k and whose score-function gradient is
                                       the unbiased gradient of E[max of k] (PKPO, Walder & Karkhanis 2025,
                                       arXiv:2505.15201; for 0 / 1 rewards the closed form of Pass@k Training, Chen et
                                       al. 2025, arXiv:2508.10751)
    wilson_interval(c, n)              a binomial proportion's interval (good at small n and near 0 / 1)
    bootstrap_ci(values)               percentile bootstrap of the mean, over items or clusters of items
    paired_bootstrap(a, b)             the mean difference b - a on the same items, its interval and a two-sided p-value
    paired_permutation(a, b)           sign-flip test of the mean difference (exact up to 2^20 sign patterns)
    mcnemar_exact(a, b)                exact McNemar test for paired 0 / 1 outcomes (one sample per item)

Items are the unit of resampling: several samples of one item are not independent, so intervals resample items (or
clusters of items: paraphrases, translations, the rows of one task family) and summarise each item first.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Sequence

__all__ = [
    "bootstrap_ci",
    "max_at_k",
    "mcnemar_exact",
    "paired_bootstrap",
    "paired_permutation",
    "pass_at_k",
    "pass_at_k_rewards",
    "wilson_interval",
]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k of one item: the probability that at least one of k samples drawn without replacement from the
    n generated ones succeeds, 1 - C(n - c, k) / C(n, k), computed as a product (no huge binomials). Needs 1 <= k <= n
    and 0 <= c <= n. The mean over items is the unbiased pass@k of the model at the sampling used."""
    if not (isinstance(n, int) and isinstance(c, int) and isinstance(k, int)):
        raise TypeError(f"pass_at_k takes integers, got n={n!r}, c={c!r}, k={k!r}")
    if not 1 <= k <= n:
        raise ValueError(f"pass_at_k needs 1 <= k <= n, got k={k}, n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"pass_at_k needs 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    fail = 1.0
    for i in range(n - c + 1, n + 1):  # C(n - c, k) / C(n, k) = prod_{i = n-c+1}^{n} (1 - k / i)
        fail *= 1.0 - k / i
    return 1.0 - fail


def _subset_max_sums(scores: Sequence[float], k: int) -> list[float]:
    """For every sample i, the sum over the k-subsets S of the n samples that contain i of max_{j in S} score_j,
    divided by C(n, k). Sorted ascending (ties in any order: they give equal values by Pascal's rule), the subsets
    containing the sample of rank m (1-based) have their maximum at m (C(m - 1, k - 1) of them) or at a higher rank j
    (C(j - 2, k - 2) of them)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranked = [float(scores[i]) for i in order]
    total = math.comb(n, k)
    out = [0.0] * n
    # suffix[m] = sum_{j > m} score_(j) C(j - 2, k - 2), ranks 1-based
    suffix = 0.0
    for m in range(n, 0, -1):
        value = ranked[m - 1] * math.comb(m - 1, k - 1) + suffix
        out[order[m - 1]] = value / total
        if k >= 2 and m >= 2:
            suffix += ranked[m - 1] * math.comb(m - 2, k - 2)
    return out


def max_at_k(scores: Sequence[float], k: int) -> float:
    """The unbiased estimate, from n >= k samples of one item, of the expected best score among k samples: the mean
    over all k-subsets of their maximum. With 0 / 1 scores it equals pass_at_k(n, sum(scores), k)."""
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError(f"max_at_k needs 1 <= k <= len(scores), got k={k}, {n} scores")
    return sum(_subset_max_sums(scores, k)) / k


def pass_at_k_rewards(scores: Sequence[float], k: int) -> list[float]:
    """Per-sample training rewards for the pass@k objective of a group of n samples of one prompt (PKPO,
    arXiv:2505.15201): r'_i = (n / k) x (1 / C(n, k)) x sum over the k-subsets containing i of their best score.

    Their mean is max_at_k(scores, k), and sum_i r'_i grad log p(y_i) is (n / k) x the unbiased estimate of the
    gradient of E[max of k samples]: a sample is credited with the share of the k-subsets it belongs to that succeed.
    For 0 / 1 scores with c successes, a success gets 1 and a failure 1 - C(n - c - 1, k - 1) / C(n - 1, k - 1), the
    closed form of Pass@k Training (arXiv:2508.10751). k = 1 returns the scores; k = n gives every sample the group's
    best score (no signal once a baseline is subtracted), so useful values are 1 < k < n. Needs 1 <= k <= n."""
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError(f"pass_at_k_rewards needs 1 <= k <= len(scores), got k={k}, {n} scores")
    if k == 1:
        return [float(s) for s in scores]
    return [n / k * v for v in _subset_max_sums(scores, k)]


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion (successes out of n): (low, high). (0, 1) for n = 0."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"wilson_interval needs 0 <= successes <= n, got {successes} of {n}")
    if n == 0:
        return 0.0, 1.0
    z = _normal_quantile(0.5 + confidence / 2)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _normal_quantile(p: float) -> float:
    """The standard normal quantile (Acklam's rational approximation, relative error < 1.2e-9)."""
    if not 0 < p < 1:
        raise ValueError(f"quantile of {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)  # fmt: skip
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)  # fmt: skip
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)  # fmt: skip
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    low = 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - low:
        return -_normal_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _clusters(n: int, clusters: Sequence[Hashable] | None) -> list[list[int]]:
    if clusters is None:
        return [[i] for i in range(n)]
    if len(clusters) != n:
        raise ValueError(f"{len(clusters)} cluster labels for {n} items")
    by: dict[Hashable, list[int]] = {}
    for i, label in enumerate(clusters):
        by.setdefault(label, []).append(i)
    return list(by.values())


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile q in [0, 1] of sorted values."""
    if not sorted_values:
        return float("nan")
    position = q * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    clusters: Sequence[Hashable] | None = None,
    statistic: Callable[[list[float]], float] | None = None,
) -> dict[str, float]:
    """Percentile bootstrap of a statistic (default the mean) of per-item values: {"estimate", "lo", "hi", "n"}.
    `clusters`: a label per item; whole clusters are resampled (items of one cluster are not independent), so the
    interval is wider when they are correlated. Deterministic for a seed."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("bootstrap_ci of no values")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap_ci needs resamples >= 1 and 0 < confidence < 1")
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    groups = _clusters(len(values), clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [values[i] for _ in groups for i in groups[rng.randrange(len(groups))]]
        draws.append(stat(sample))
    draws.sort()
    tail = (1 - confidence) / 2
    return {
        "estimate": stat(values),
        "lo": _percentile(draws, tail),
        "hi": _percentile(draws, 1 - tail),
        "n": float(len(values)),
    }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    clusters: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """The mean difference b - a over the same items (a[i] and b[i] score item i), with a percentile bootstrap
    interval over items (or clusters) and a two-sided p-value: twice the share of resampled differences on the other
    side of 0 from the estimate (capped at 1; at least 1 / resamples when none is). {"diff", "lo", "hi", "p", "n"}."""
    if len(a) != len(b):
        raise ValueError(f"paired_bootstrap needs the same items: {len(a)} and {len(b)} values")
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    out = bootstrap_ci(diffs, resamples, confidence, seed, clusters)
    estimate = out["estimate"]
    rng = random.Random(seed)
    groups = _clusters(len(diffs), clusters)
    other_side = 0
    for _ in range(resamples):
        sample = [diffs[i] for _ in groups for i in groups[rng.randrange(len(groups))]]
        mean = sum(sample) / len(sample)
        if (estimate >= 0 and mean <= 0) or (estimate <= 0 and mean >= 0):
            other_side += 1
    p = min(1.0, 2 * max(other_side, 1) / resamples)
    return {"diff": estimate, "lo": out["lo"], "hi": out["hi"], "p": p, "n": out["n"]}


def paired_permutation(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10000,
    seed: int = 0,
    clusters: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """Two-sided sign-flip permutation test of the mean difference b - a (under the null the two systems are
    exchangeable on every item, or cluster: its differences flip sign together). Exact when there are at most 20
    non-zero clusters, Monte Carlo with `resamples` draws otherwise. {"diff", "p", "n", "exact"}."""
    if len(a) != len(b):
        raise ValueError(f"paired_permutation needs the same items: {len(a)} and {len(b)} values")
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    if not diffs:
        raise ValueError("paired_permutation of no items")
    sums = [sum(diffs[i] for i in g) for g in _clusters(len(diffs), clusters)]
    sums = [s for s in sums if s != 0.0]
    observed = abs(sum(sums))
    tol = 1e-12 * max(1.0, observed)
    n = len(diffs)
    if not sums:
        return {"diff": 0.0, "p": 1.0, "n": float(n), "exact": 1.0}
    if len(sums) <= 20:
        hits = 0
        for mask in range(1 << len(sums)):
            total = sum(-s if mask >> j & 1 else s for j, s in enumerate(sums))
            hits += abs(total) >= observed - tol
        return {"diff": sum(diffs) / n, "p": hits / (1 << len(sums)), "n": float(n), "exact": 1.0}
    rng = random.Random(seed)
    hits = 0
    for _ in range(resamples):
        total = sum(s if rng.random() < 0.5 else -s for s in sums)
        hits += abs(total) >= observed - tol
    return {"diff": sum(diffs) / n, "p": (hits + 1) / (resamples + 1), "n": float(n), "exact": 0.0}


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact (binomial) McNemar test for paired 0 / 1 outcomes, one per item: only discordant items count, b-only
    successes against a-only ones, two-sided p = min(1, 2 x P[Binomial(d, 1/2) <= min]). {"b_only", "a_only", "p"}."""
    if len(a) != len(b):
        raise ValueError(f"mcnemar_exact needs the same items: {len(a)} and {len(b)} outcomes")
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    d = a_only + b_only
    if d == 0:
        return {"b_only": 0.0, "a_only": 0.0, "p": 1.0}
    tail = sum(math.comb(d, i) for i in range(min(a_only, b_only) + 1)) / 2**d
    return {"b_only": float(b_only), "a_only": float(a_only), "p": min(1.0, 2 * tail)}
