"""Selection strategies of `pgs prepare` (prepare.select_by_budget)."""

import math

import pytest

from palingenesis.prepare import STRATEGIES, classify_familiarity, select_by_budget


def _pool(n=400):
    # File order grouped by source (as in concatenated public datasets): source A first.
    samples = []
    for i in range(n):
        nll = 0.2 + 2.0 * ((i * 37) % n) / n
        samples.append(
            {
                "id": i,
                "source": "A" if i < n // 2 else "B",
                "_score_response_nll": nll,
                "_score_response_ppl": math.exp(nll),
            }
        )
    return classify_familiarity(samples)


def test_unknown_strategy_is_an_error():
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        select_by_budget(_pool(), 10, "optimall")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_returns_budget_distinct_samples(strategy):
    pool = _pool()
    sel = select_by_budget(pool, 100, strategy)
    assert len(sel) == 100
    assert len({s["id"] for s in sel}) == 100


@pytest.mark.parametrize("strategy", ["optimal", "balanced", "random", "curriculum"])
def test_bucket_draws_do_not_follow_file_order(strategy):
    # A first-N pick inside a bucket would take (almost) only source A.
    sel = select_by_budget(_pool(), 100, strategy)
    frac_b = sum(s["source"] == "B" for s in sel) / len(sel)
    assert 0.3 < frac_b < 0.7


def test_curriculum_keeps_unfamiliar_samples():
    pool = _pool()
    sel = select_by_budget(pool, 100, "curriculum")
    assert any(s["_score_familiarity"] == "unfamiliar" for s in sel)


def test_selection_is_deterministic_for_a_seed():
    a = [s["id"] for s in select_by_budget(_pool(), 50, "optimal", seed=3)]
    b = [s["id"] for s in select_by_budget(_pool(), 50, "optimal", seed=3)]
    assert a == b


def test_flow_weight_is_exp_of_minus_loss_over_median_loss():
    pool = _pool()
    select_by_budget(pool, 50, "flow")
    losses = sorted(s["_score_response_nll"] for s in pool)
    tau = losses[len(losses) // 2]
    s = pool[7]
    assert s["_score_flow_weight"] == pytest.approx(math.exp(-s["_score_response_nll"] / tau), abs=1e-4)


def test_profile_groups_reports_per_source_familiarity():
    from palingenesis.prepare import profile_groups

    pool = _pool()
    for s in pool:
        s["_score_response_token_count"] = 10 if s["source"] == "A" else 30
    prof = profile_groups(pool, "source")
    assert set(prof) == {"A", "B"}
    assert prof["A"]["count"] == prof["B"]["count"] == 200
    assert prof["A"]["mean_response_tokens"] == 10.0 and prof["B"]["mean_response_tokens"] == 30.0
    assert sum(prof["A"]["familiarity"].values()) == 200
    assert profile_groups([{"_score_response_nll": 1.0}], "source")["<none>"]["count"] == 1


def test_default_strategy_is_random():
    from palingenesis.config import PreprocessConfig

    assert PreprocessConfig().strategy == "random"
