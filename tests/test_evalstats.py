"""palingenesis.evalstats: unbiased pass@k and its training rewards against brute force over subsets, intervals and
paired tests against hand-computed values."""

import itertools
import math
import random

import pytest

from palingenesis.evalstats import (
    bootstrap_ci,
    max_at_k,
    mcnemar_exact,
    paired_bootstrap,
    paired_permutation,
    pass_at_k,
    pass_at_k_rewards,
    wilson_interval,
)


def test_pass_at_k_hand_values_and_edges():
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)  # 1 - C(3,2)/C(4,2) = 1 - 3/6
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)  # k = 1: the success rate
    assert pass_at_k(10, 0, 5) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(5, 3, 3) == 1.0  # n - c < k: every 3-subset holds a success
    assert pass_at_k(8, 2, 8) == 1.0
    for n, c, k in [(20, 7, 4), (200, 13, 50), (16, 1, 15)]:
        assert pass_at_k(n, c, k) == pytest.approx(1 - math.comb(n - c, k) / math.comb(n, k), rel=1e-12)
    with pytest.raises(ValueError, match="1 <= k <= n"):
        pass_at_k(4, 1, 5)
    with pytest.raises(ValueError, match="0 <= c <= n"):
        pass_at_k(4, 5, 2)
    with pytest.raises(TypeError):
        pass_at_k(4.0, 1, 2)


def _brute(scores, k):
    n = len(scores)
    per = [0.0] * n
    total = math.comb(n, k)
    for subset in itertools.combinations(range(n), k):
        best = max(scores[j] for j in subset)
        for i in subset:
            per[i] += best / total
    return [n / k * v for v in per], sum(max(scores[j] for j in s) for s in itertools.combinations(range(n), k)) / total


@pytest.mark.parametrize("seed", range(6))
def test_pass_at_k_rewards_match_brute_force_over_subsets(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    scores = [rng.choice([0.0, 0.25, 0.5, 1.0, rng.random()]) for _ in range(n)]  # ties included
    for k in range(1, n + 1):
        expected, mean = _brute(scores, k)
        assert pass_at_k_rewards(scores, k) == pytest.approx(expected, abs=1e-12)
        assert max_at_k(scores, k) == pytest.approx(mean, abs=1e-12)
        assert sum(pass_at_k_rewards(scores, k)) / n == pytest.approx(mean, abs=1e-12)


def test_pass_at_k_rewards_binary_closed_form():
    """0 / 1 rewards: a success gets 1, a failure 1 - C(n-c-1, k-1) / C(n-1, k-1) (Pass@k Training), and the mean is
    the unbiased pass@k."""
    n, c, k = 8, 3, 4
    scores = [1.0] * c + [0.0] * (n - c)
    out = pass_at_k_rewards(scores, k)
    assert out[:c] == pytest.approx([1.0] * c)
    assert out[c:] == pytest.approx([1 - math.comb(n - c - 1, k - 1) / math.comb(n - 1, k - 1)] * (n - c))
    assert out[c] == pytest.approx(1 - 4 / 35)  # C(4, 3) / C(7, 3)
    assert sum(out) / n == pytest.approx(pass_at_k(n, c, k))
    assert pass_at_k_rewards(scores, 1) == scores
    assert pass_at_k_rewards(scores, n) == [1.0] * n  # k = n: everyone gets the best
    with pytest.raises(ValueError):
        pass_at_k_rewards(scores, 9)


def test_wilson_interval_reference_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.23659, 0.76341), abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(81, 100, confidence=0.99)
    assert lo < 0.81 < hi and hi - lo > wilson_interval(81, 100)[1] - wilson_interval(81, 100)[0]
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_bootstrap_ci_is_deterministic_and_clusters_widen_it():
    values = [1.0] * 30 + [0.0] * 30
    a = bootstrap_ci(values, resamples=2000, seed=1)
    assert a == bootstrap_ci(values, resamples=2000, seed=1)
    assert a["estimate"] == 0.5 and a["lo"] < 0.5 < a["hi"] and a["n"] == 60
    # half-width close to the normal one, 1.96 * sqrt(0.25 / 60) = 0.1265
    assert (a["hi"] - a["lo"]) / 2 == pytest.approx(0.1265, abs=0.02)
    # the same values as 6 perfectly correlated clusters of 10: far fewer independent units
    clusters = [i // 10 for i in range(60)]
    b = bootstrap_ci(values, resamples=2000, seed=1, clusters=clusters)
    assert b["hi"] - b["lo"] > 1.8 * (a["hi"] - a["lo"])
    assert bootstrap_ci([3.0], resamples=10)["lo"] == 3.0
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="cluster labels"):
        bootstrap_ci([1.0, 2.0], clusters=[0])


def test_paired_tests():
    a = [0.0] * 40 + [1.0] * 60
    b = [1.0] * 40 + [1.0] * 60  # b fixes 40 items, breaks none
    out = paired_bootstrap(a, b, resamples=2000)
    assert out["diff"] == pytest.approx(0.4) and out["lo"] > 0.25 and out["p"] == pytest.approx(2 / 2000)
    same = paired_bootstrap(a, a, resamples=500)
    assert same["diff"] == 0.0 and same["p"] == 1.0
    # sign flips: 3 items all +1 -> exact p = 2 / 8
    assert paired_permutation([0, 0, 0], [1, 1, 1]) == {"diff": 1.0, "p": 0.25, "n": 3.0, "exact": 1.0}
    assert paired_permutation([1, 0], [1, 0])["p"] == 1.0
    # mixed signs, exact enumeration by hand: diffs +2, -1 -> |sum| over {+-2 +-1}: 1, 3, 3, 1 >= 1: p = 1
    assert paired_permutation([0, 1], [2, 0])["p"] == 1.0
    big = paired_permutation([0.0] * 30, [1.0] * 30, resamples=999)
    assert big["exact"] == 0.0 and big["p"] == pytest.approx(1 / 1000)
    # clusters flip together: two clusters of identical +1 differences -> 4 patterns, 2 extreme
    assert paired_permutation([0] * 4, [1] * 4, clusters=[0, 0, 1, 1])["p"] == 0.5
    with pytest.raises(ValueError):
        paired_permutation([1], [1, 2])


def test_mcnemar_exact():
    # 1 a-only, 2 b-only: d = 3, P[Bin(3, .5) <= 1] = 4 / 8, two-sided 1.0
    assert mcnemar_exact([1, 1, 0, 0, 1], [1, 0, 1, 1, 1]) == {"b_only": 2.0, "a_only": 1.0, "p": 1.0}
    # 0 a-only, 6 b-only: 2 * 1 / 64
    out = mcnemar_exact([0] * 6 + [1] * 4, [1] * 6 + [1] * 4)
    assert out["p"] == pytest.approx(2 / 64)
    assert mcnemar_exact([1, 0], [1, 0])["p"] == 1.0
