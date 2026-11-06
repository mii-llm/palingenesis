"""Off-policy distillation losses: divergences against reference formulas, the top-k supports, the
row-chunked head loss (chunked == unchunked, gradients included) and the distiller's per-token mix."""

import math

import pytest
import torch
import torch.nn.functional as F

from palingenesis.distill import losses as L
from palingenesis.distill.chunked import chunked_rows_loss


def _dists(rows=5, vocab=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    zp, zq = torch.randn(rows, vocab, generator=g) * 2, torch.randn(rows, vocab, generator=g) * 2
    return F.log_softmax(zp, -1), F.log_softmax(zq, -1)


def _ref(kind, lp, lq, beta):
    p, q = lp.exp(), lq.exp()
    if kind == "forward_kl":
        return (p * (lp - lq)).sum(-1)
    if kind == "reverse_kl":
        return (q * (lq - lp)).sum(-1)
    if kind == "jsd":
        m = beta * p + (1 - beta) * q
        return beta * (p * (lp - m.log())).sum(-1) + (1 - beta) * (q * (lq - m.log())).sum(-1)
    if kind == "skew_forward_kl":
        return (p * (lp - (beta * p + (1 - beta) * q).log())).sum(-1)
    if kind == "skew_reverse_kl":
        return (q * (lq - (beta * q + (1 - beta) * p).log())).sum(-1)
    if kind == "tvd":
        return 0.5 * (p - q).abs().sum(-1)


@pytest.mark.parametrize("kind", ["forward_kl", "reverse_kl", "jsd", "skew_forward_kl", "skew_reverse_kl", "tvd"])
@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_divergences_match_reference(kind, beta):
    lp, lq = _dists()
    torch.testing.assert_close(L.divergence(lp, lq, None, kind, beta), _ref(kind, lp, lq, beta), rtol=1e-5, atol=1e-6)


def test_forward_kl_matches_torch_kl_div():
    lp, lq = _dists()
    expected = F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(L.divergence(lp, lq, None, "forward_kl"), expected)


@pytest.mark.parametrize("kind", ["forward_kl", "reverse_kl", "jsd", "tvd"])
def test_divergences_are_zero_on_equal_and_positive_otherwise(kind):
    lp, lq = _dists()
    assert torch.allclose(L.divergence(lp, lp, None, kind), torch.zeros(5), atol=1e-6)
    assert (L.divergence(lp, lq, None, kind) > 0).all()


def test_mask_excludes_entries_and_gradients_stay_finite():
    lp, lq = _dists(vocab=6)
    mask = torch.tensor([True, True, True, False, False, False]).expand(5, 6)
    # renormalised over the first three ids, the masked divergence equals the 3-way one
    lp3, lq3 = lp[:, :3].log_softmax(-1), lq[:, :3].log_softmax(-1)
    lp_m = torch.cat([lp3, torch.full((5, 3), -float("inf"))], -1)
    lq_m = torch.cat([lq3, torch.full((5, 3), -float("inf"))], -1).requires_grad_(True)
    for kind in ("forward_kl", "reverse_kl", "jsd", "skew_forward_kl", "skew_reverse_kl", "tvd"):
        d = L.divergence(lp_m, lq_m, mask, kind, 0.3)
        torch.testing.assert_close(d, L.divergence(lp3, lq3, None, kind, 0.3), rtol=1e-5, atol=1e-6)
        d.sum().backward()
        assert torch.isfinite(lq_m.grad).all()
        lq_m.grad = None


def test_sparse_tail_with_full_support_equals_dense():
    """k = vocabulary: the tail bucket is empty and the coarse divergence is the dense one."""
    z_s = torch.randn(4, 9)
    teacher = F.log_softmax(torch.randn(4, 9), -1)
    top = teacher.topk(9, -1)
    for kind in ("forward_kl", "reverse_kl", "jsd"):
        lp, lq, mask = L.sparse_support(z_s, top.indices, top.values, None, None, 1.0, "tail")
        sparse = L.divergence(lp, lq, mask, kind, 0.5)
        dense = L.divergence(teacher, F.log_softmax(z_s, -1), None, kind, 0.5)
        torch.testing.assert_close(sparse, dense, rtol=1e-4, atol=1e-5)


def test_sparse_tail_bucket_matches_hand_computation():
    z_s = torch.randn(3, 8)
    teacher = F.log_softmax(torch.randn(3, 8), -1)
    top = teacher.topk(3, -1)
    lp, lq, mask = L.sparse_support(z_s, top.indices, top.values, None, None, 1.0, "tail")
    q = F.softmax(z_s, -1)
    p_top, q_top = top.values.exp(), q.gather(1, top.indices)
    p_c = torch.cat([p_top, 1 - p_top.sum(-1, keepdim=True)], -1)
    q_c = torch.cat([q_top, 1 - q_top.sum(-1, keepdim=True)], -1)
    expected = (p_c * (p_c.log() - q_c.log())).sum(-1)
    torch.testing.assert_close(L.divergence(lp, lq, mask, "forward_kl"), expected, rtol=1e-5, atol=1e-6)


def test_sparse_gold_token_joins_the_support_once():
    z_s = torch.randn(2, 8)
    teacher = F.log_softmax(torch.randn(2, 8), -1)
    top = teacher.topk(2, -1)
    gold = torch.tensor([int(top.indices[0, 0]), int(teacher[1].argmin())])  # row 0: in the top-k; row 1: not
    lp, lq, mask = L.sparse_support(z_s, top.indices, top.values, gold, teacher.gather(1, gold[:, None])[:, 0], 1.0)
    assert mask.shape == (2, 4)  # k + gold + tail
    assert not mask[0, 2] and mask[1, 2]  # the duplicate gold of row 0 is masked out
    torch.testing.assert_close(lp[1, 2], teacher[1, gold[1]])


def test_sparse_renorm_and_temperature():
    z_s = torch.randn(3, 10)
    z_t = torch.randn(3, 10)
    top = F.log_softmax(z_t, -1).topk(4, -1)
    lp, lq, mask = L.sparse_support(z_s, top.indices, top.values, None, None, 2.0, "renorm", cache_temperature=1.0)
    # renorm at temperature 2: both sides are softmax(z / 2) restricted to the support
    torch.testing.assert_close(lp, (z_t.gather(1, top.indices) / 2).log_softmax(-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lq, (z_s.gather(1, top.indices) / 2).log_softmax(-1), rtol=1e-5, atol=1e-6)
    assert mask.all()


def test_sparse_ignores_missing_and_out_of_vocab_ids():
    z_s = torch.randn(2, 6)
    ids = torch.tensor([[0, -1, 9], [1, 2, 3]])
    lp_t = torch.log(torch.tensor([[0.5, 0.2, 0.2], [0.3, 0.3, 0.3]]))
    _, _, mask = L.sparse_support(z_s, ids, lp_t, None, None, 1.0, "tail")
    assert mask.tolist() == [[True, False, False, True], [True, True, True, True]]


def test_feature_losses():
    s, t = torch.randn(4, 6), torch.randn(4, 6)
    torch.testing.assert_close(L.feature_loss(s, t, "mse"), ((s - t) ** 2).mean(-1))
    torch.testing.assert_close(L.feature_loss(s, s * 3, "cosine"), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(L.feature_loss(s, s * 3, "nmse"), torch.zeros(4), atol=1e-6, rtol=0)


# ── the chunked head loss ───────────────────────────────────────────────────


def _setup(seed=0):
    torch.manual_seed(seed)
    head = torch.nn.Linear(8, 13, bias=False)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    rows = torch.tensor([1, 2, 3, 5, 8, 9, 12, 13])
    teacher = F.log_softmax(torch.randn(rows.numel(), 13), -1)
    return head, hidden, rows, teacher


def _fn(teacher, weights=None):
    def fn(logits, a, b):
        d = L.divergence(teacher[a:b], F.log_softmax(logits, -1), None, "forward_kl")
        w = weights[a:b] if weights is not None else 1.0
        return (w * d).sum(), {"n": b - a}

    return fn


@pytest.mark.parametrize("slice_rows", [1, 3, 8, 100])
def test_chunked_rows_loss_equals_unchunked(slice_rows):
    head, hidden, rows, teacher = _setup()
    # reference: plain autograd on the full logits
    ref_head = torch.nn.Linear(8, 13, bias=False)
    ref_head.weight.data.copy_(head.weight.data)
    h_ref = hidden.detach().clone().requires_grad_(True)
    logits = ref_head(h_ref.reshape(-1, 8)[rows])
    ref = L.divergence(teacher, F.log_softmax(logits, -1), None, "forward_kl").sum() / 4.0
    (ref * 2.0).backward()

    stats = {}
    loss = chunked_rows_loss(hidden, rows, head, _fn(teacher), 4.0, slice_rows, stats)
    (loss * 2.0).backward()  # downstream scaling reaches the hidden states (the bridge)
    torch.testing.assert_close(loss, ref.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, h_ref.grad, rtol=1e-5, atol=1e-6)
    assert stats["n"] == rows.numel()
    # head gradients are accumulated inside the call, at the denominator's scale (like chunked CE)
    torch.testing.assert_close(head.weight.grad * 2.0, ref_head.weight.grad, rtol=1e-5, atol=1e-6)


def test_chunked_rows_loss_without_grad_and_without_rows():
    head, hidden, rows, teacher = _setup()
    with torch.no_grad():
        v = chunked_rows_loss(hidden, rows, head, _fn(teacher), 1.0, 3)
    assert v.ndim == 0 and v > 0
    empty = chunked_rows_loss(hidden, rows[:0], head, _fn(teacher), 1.0, 3)
    empty.backward()
    assert float(empty.detach()) == 0.0


def test_temperature_scale_applies_t_squared():
    from palingenesis.distill.config import DistillConfig
    from palingenesis.distill.distiller import Distiller, TeacherRows, TeacherTargets

    torch.manual_seed(0)
    head = torch.nn.Linear(4, 9, bias=False)
    hidden = torch.randn(1, 3, 4)
    rows = torch.tensor([0, 1, 2])
    ids = torch.arange(9).expand(3, 9).clone()
    t_lp = F.log_softmax(torch.randn(3, 9), -1)
    targets = TeacherTargets(
        rows=rows,
        gold=torch.tensor([1, 2, 3]),
        token_weights=torch.ones(3),
        teachers={
            "c": TeacherRows(
                "c", 1.0, torch.ones(3, dtype=torch.bool), ids=ids, lp=t_lp, gold_lp=torch.full((3,), float("nan"))
            )
        },
    )

    class _Meta:
        temperature = 2.0

    class _Cache:
        meta = _Meta()
        k = 9

    values = {}
    for scale in (False, True):
        cfg = DistillConfig(enabled=True, temperature=2.0, temperature_scale=scale, topk_mode="renorm")
        d = Distiller(cfg, [], {}, {"c": _Cache()})
        loss, _ = d.loss(hidden, targets, head, 1.0)
        values[scale] = float(loss)
    assert math.isclose(values[True], 4.0 * values[False], rel_tol=1e-5)


def test_mixing_teachers_is_a_weighted_mean_per_token():
    from palingenesis.distill.config import DistillConfig
    from palingenesis.distill.distiller import Distiller, TeacherRows, TeacherTargets

    torch.manual_seed(1)
    head = torch.nn.Linear(4, 9, bias=False)
    hidden = torch.randn(1, 4, 4)
    rows = torch.arange(4)
    ids = torch.arange(9).expand(4, 9).clone()
    lp_a, lp_b = F.log_softmax(torch.randn(4, 9), -1), F.log_softmax(torch.randn(4, 9), -1)
    nan = torch.full((4,), float("nan"))
    cover_b = torch.tensor([True, True, False, False])

    class _Meta:
        temperature = 1.0

    class _Cache:
        meta = _Meta()
        k = 9

    def run(teachers):
        targets = TeacherTargets(
            rows=rows, gold=torch.zeros(4, dtype=torch.long), token_weights=torch.ones(4), teachers=teachers
        )
        d = Distiller(DistillConfig(enabled=True), [], {}, {n: _Cache() for n in teachers})
        return float(d.loss(hidden, targets, head, 1.0)[0])

    both = run(
        {
            "a": TeacherRows("a", 1.0, torch.ones(4, dtype=torch.bool), ids=ids, lp=lp_a, gold_lp=nan),
            "b": TeacherRows("b", 3.0, cover_b, ids=ids, lp=lp_b, gold_lp=nan),
        }
    )
    logits = head(hidden[0]).detach()

    def kd(lp):
        p, q, mask = L.sparse_support(logits, ids, lp, None, None, 1.0)
        return L.divergence(p, q, mask, "forward_kl")

    da, db = kd(lp_a), kd(lp_b)
    expected = float(((da + 3 * db) / 4)[:2].sum() + da[2:].sum())  # rows 2-3: teacher a only
    assert math.isclose(both, expected, rel_tol=1e-5)
