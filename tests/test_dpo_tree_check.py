"""The judging rule of scripts/gpu_checks/dpo_tree_check.py (float32: relative to the float32 floor)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_checks" / "dpo_tree_check.py"


@pytest.fixture()
def check(monkeypatch):
    spec = importlib.util.spec_from_file_location("dpo_tree_check_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PRECISION", "float32")
    monkeypatch.setattr(module, "DEVICE", "cpu")
    return module


# measured on an A100, Qwen3.5-0.8B, float32, TF32 off (p99 and signed mean: of the order of the mean)
FLOOR = {
    "token_logprob_mean_abs_diff": 0.00059,
    "token_logprob_p99_abs_diff": 0.0025,
    "token_logprob_max_abs_diff": 0.0042,
    "token_logprob_signed_mean_diff": 0.00005,
    "loss_diff": 0.00036,
    "grad_rel_l2": 0.0051,
    "grad_cosine": 0.99999,
}
VARLEN = {
    **FLOOR,
    "token_logprob_mean_abs_diff": 0.00061,
    "token_logprob_max_abs_diff": 0.0058,
    "loss_diff": 0.0002,
    "grad_rel_l2": 0.0027,
    "grad_cosine": 1.0,
}
TREE = {
    **FLOOR,
    "token_logprob_mean_abs_diff": 0.00129,
    "token_logprob_p99_abs_diff": 0.006,
    "token_logprob_max_abs_diff": 0.0179,
    "loss_diff": 0.00107,
    "grad_rel_l2": 0.0033,
    "grad_cosine": 1.0,
    "tree_levels": 2,
}


def test_measured_varlen_and_tree_pass(check):
    assert check.judge(VARLEN, FLOOR, tree=False) == []
    assert check.judge(TREE, FLOOR, tree=True) == []


@pytest.mark.parametrize(
    "key, value",
    [
        ("token_logprob_mean_abs_diff", 0.01),
        ("token_logprob_max_abs_diff", 0.5),
        ("loss_diff", 0.01),
        ("grad_cosine", 0.999),
        ("grad_rel_l2", 0.05),
    ],
)
def test_gross_error_fails(check, key, value):
    failed = check.judge({**TREE, key: value}, FLOOR, tree=False)
    assert failed and any(key in line for line in failed)
