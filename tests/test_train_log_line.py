"""The training log line carries the per-source eval losses (eval_sources) next to the overall eval loss."""

from palingenesis.train import _per_source_eval


def test_per_source_eval_lists_each_source_loss_sorted():
    m = {
        "eval/loss": 1.0,
        "eval/omnia/loss": 0.9,
        "eval/alpha/loss": 1.23456,
        "eval/alpha/ppl": 3.4,
        "eval/regressions": 1,
    }
    assert _per_source_eval(m) == " eval_sources=alpha:1.2346,omnia:0.9000"


def test_per_source_eval_is_empty_without_eval_sources():
    assert _per_source_eval({"eval/loss": 1.0, "eval/ppl": 2.7}) == ""
