"""full_fkl / full_jsd (opd.losses) against dense references, and the system placement of with_privileged."""

import sys

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
F = torch.nn.functional

from palingenesis.opd import losses  # noqa: E402
from palingenesis.opd.formatting import with_privileged  # noqa: E402


def setup(seed=0, n=7, h=6, v=11):
    torch.manual_seed(seed)
    head = torch.nn.Linear(h, v, bias=False).double()
    hidden = torch.randn(n, h, dtype=torch.float64, requires_grad=True)
    teacher = torch.log_softmax(torch.randn(n, v, dtype=torch.float64), -1)
    targets = torch.randint(0, v, (n,))
    weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
    return head, hidden, teacher, targets, weights


def dense_fkl(lq, lp):
    return (lp.exp() * (lp - lq)).sum(-1)


def dense_jsd(lq, lp, beta):
    lm = torch.log(beta * lp.exp() + (1 - beta) * lq.exp())
    return beta * (lp.exp() * (lp - lm)).sum(-1) + (1 - beta) * (lq.exp() * (lq - lm)).sum(-1)


@pytest.mark.parametrize("name", ["fkl", "jsd"])
def test_matches_dense_reference_with_gradients(name):
    head, hidden, teacher, targets, weights = setup()
    vocab = losses.SharedVocab(teacher.shape[1])
    if name == "fkl":
        value, stats = losses.full_fkl(hidden, head, targets, weights, vocab, lambda a, b: teacher[a:b])
    else:
        value, stats = losses.full_jsd(hidden, head, targets, weights, vocab, lambda a, b: teacher[a:b], beta=0.3)
    grad_h, grad_w = torch.autograd.grad(value, [hidden, head.weight])

    hidden2 = hidden.detach().clone().requires_grad_(True)
    lq = F.log_softmax(head(hidden2), -1)
    per_token = dense_fkl(lq, teacher) if name == "fkl" else dense_jsd(lq, teacher, 0.3)
    ref = (weights * per_token).sum()
    ref_h, ref_w = torch.autograd.grad(ref, [hidden2, head.weight])
    assert torch.allclose(value, ref, atol=1e-12)
    assert torch.allclose(grad_h, ref_h, atol=1e-10) and torch.allclose(grad_w, ref_w, atol=1e-10)
    assert stats["kl"] == pytest.approx(float(per_token.detach().sum()), abs=1e-9)


@pytest.mark.parametrize("name", ["fkl", "jsd", "rkl"])
def test_zero_when_student_equals_teacher(name):
    head, hidden, _, targets, weights = setup(1)
    teacher = F.log_softmax(head(hidden), -1).detach()
    fn = {"fkl": losses.full_fkl, "jsd": losses.full_jsd, "rkl": losses.full_rkl}[name]
    value, stats = fn(hidden, head, targets, weights, losses.SharedVocab(teacher.shape[1]), lambda a, b: teacher[a:b])
    assert abs(float(value.detach())) < 1e-12 and abs(stats["k1"]) < 1e-9


def test_jsd_needs_beta_inside_the_interval():
    head, hidden, teacher, targets, weights = setup()
    with pytest.raises(ValueError):
        losses.full_jsd(hidden, head, targets, weights, losses.SharedVocab(11), lambda a, b: teacher[a:b], beta=1.0)


def test_sliced_equals_unsliced():
    head, hidden, teacher, targets, weights = setup(2, n=13)
    vocab = losses.SharedVocab(teacher.shape[1])
    full, _ = losses.full_fkl(hidden, head, targets, weights, vocab, lambda a, b: teacher[a:b])
    old = losses.SLICE_ELEMENTS
    losses.SLICE_ELEMENTS = 3 * teacher.shape[1]  # 3 rows per slice
    try:
        sliced, _ = losses.full_fkl(hidden, head, targets, weights, vocab, lambda a, b: teacher[a:b])
    finally:
        losses.SLICE_ELEMENTS = old
    assert torch.allclose(full, sliced, atol=1e-12)


def test_with_privileged_system_placement():
    notes = "NOTES"
    bare = [{"role": "user", "content": "q"}]
    assert with_privileged(bare, notes, "{privileged}\n\n{content}", placement="system") == [
        {"role": "system", "content": "NOTES"},
        {"role": "user", "content": "q"},
    ]
    with_system = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"}]
    out = with_privileged(with_system, notes, "{privileged}\n\n{content}", placement="system")
    assert out[0]["content"] == "NOTES\n\nBe brief." and out[1:] == with_system[1:]
    assert with_system[0]["content"] == "Be brief."  # the input is not modified
    # The default placement is unchanged: the last user turn.
    assert with_privileged(bare, notes, "{content}\n\n{privileged}")[0]["content"] == "q\n\nNOTES"
    with pytest.raises(ValueError):
        with_privileged(bare, notes, "{content}", placement="assistant")
