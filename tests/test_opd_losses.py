"""OPD losses against naive references that materialize the full [N, V] logits.

Every loss is checked for its value and its gradients (hidden states and head
weights) with slices of 3 rows, so the slice-by-slice differentiation of
losses._ChunkedHead is exercised across slice boundaries.
"""

import math
import sys

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from palingenesis.logits import PostProcessedHead  # noqa: E402
from palingenesis.opd import losses  # noqa: E402

N, H, V = 11, 8, 13  # tokens, hidden size, student vocab
SHARED = 10  # teacher vocab = shared prefix
SWAP = {12: 4}  # student-only terminator 12 -> teacher terminator 4


@pytest.fixture(autouse=True)
def three_row_slices(monkeypatch):
    monkeypatch.setattr(losses, "SLICE_ELEMENTS", 3 * V)


def make(seed=0, softcap=False):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, H, generator=g, dtype=torch.float64).requires_grad_(True)
    head = nn.Linear(H, V, bias=False).double()
    with torch.no_grad():
        head.weight.copy_(torch.randn(V, H, generator=g, dtype=torch.float64))
    if softcap:
        head = PostProcessedHead(head, multiplier=0.7, softcap=3.0)
    weights = torch.rand(N, generator=g, dtype=torch.float64) / N
    return hidden, head, weights, g


def grads(loss, hidden, head):
    return torch.autograd.grad(loss, [hidden, head.weight])


def assert_same(ours, ref, hidden, head):
    (loss, _), ref_loss = ours, ref
    torch.testing.assert_close(loss.double(), ref_loss.double(), rtol=1e-6, atol=1e-9)
    for got, want in zip(grads(loss, hidden, head), grads(ref_loss, hidden, head)):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-9)


def project(logp):
    """Reference SharedVocab projection with explicit probabilities."""
    p = logp.exp()
    shared = p[:, :SHARED].clone()
    for s, t in SWAP.items():
        shared[:, t] = shared[:, t] + p[:, s]
    return shared.log()


def teacher_logp(g):
    return F.log_softmax(torch.randn(N, SHARED + 3, generator=g, dtype=torch.float64), -1)[:, :SHARED]


# --------------------------------------------------------------------- full_rkl


@pytest.mark.parametrize("softcap", [False, True])
def test_full_rkl_matches_naive(softcap):
    hidden, head, weights, g = make(softcap=softcap)
    t_logp = teacher_logp(g)
    targets = torch.randint(0, SHARED, (N,), generator=g)
    vocab = losses.SharedVocab(SHARED, SWAP)
    ours = losses.full_rkl(hidden, head, targets, weights, vocab, lambda a, b: t_logp[a:b])

    lq = project(F.log_softmax(head(hidden), -1))
    kl = (lq.exp() * (lq - t_logp)).sum(-1)
    assert_same(ours, (weights * kl).sum(), hidden, head)
    stats = ours[1]
    assert stats["kl"] == pytest.approx(kl.sum().item())
    k1 = lq.gather(1, targets[:, None]).squeeze(1) - t_logp.gather(1, targets[:, None]).squeeze(1)
    assert stats["k1"] == pytest.approx(k1.sum().item())
    p = F.softmax(head(hidden), -1)
    residual = p[:, SHARED:].sum(-1) - p[:, 12]  # student-only mass that is not swapped
    assert stats["residual"] == pytest.approx(residual.sum().item(), abs=1e-9)


def test_full_rkl_is_zero_when_student_equals_teacher():
    hidden, head, weights, _ = make()
    vocab = losses.SharedVocab(V)
    t_logp = F.log_softmax(head(hidden), -1).detach()
    loss, stats = losses.full_rkl(
        hidden, head, torch.zeros(N, dtype=torch.long), weights, vocab, lambda a, b: t_logp[a:b]
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert all(g.abs().max() < 1e-12 for g in grads(loss, hidden, head))


def test_chunked_gradient_respects_downstream_scaling():
    """Gradients are computed in the forward; backward must still scale them."""
    hidden, head, weights, g = make()
    t_logp = teacher_logp(g)
    vocab = losses.SharedVocab(SHARED, SWAP)
    loss, _ = losses.full_rkl(hidden, head, torch.zeros(N, dtype=torch.long), weights, vocab, lambda a, b: t_logp[a:b])
    once = grads(loss, hidden, head)
    loss, _ = losses.full_rkl(hidden, head, torch.zeros(N, dtype=torch.long), weights, vocab, lambda a, b: t_logp[a:b])
    thrice = grads(3.0 * loss, hidden, head)
    for a, b in zip(once, thrice):
        torch.testing.assert_close(3.0 * a, b)


def test_no_grad_path_gives_the_same_value():
    hidden, head, weights, g = make()
    t_logp = teacher_logp(g)
    vocab = losses.SharedVocab(SHARED, SWAP)
    with_grad, stats = losses.full_rkl(
        hidden, head, torch.zeros(N, dtype=torch.long), weights, vocab, lambda a, b: t_logp[a:b]
    )
    with torch.no_grad():
        without, stats_ng = losses.full_rkl(
            hidden, head, torch.zeros(N, dtype=torch.long), weights, vocab, lambda a, b: t_logp[a:b]
        )
    assert without.item() == pytest.approx(with_grad.item())
    assert stats_ng == pytest.approx(stats)
    assert head.weight.grad is None


# ---------------------------------------------------------------------- topk_kl


def naive_coarse_kl(lq, lp, valid, beta):
    """Coarse KL with explicit probabilities and tails (floored like losses._log1mexp)."""
    q = torch.where(valid, lq.exp(), torch.zeros_like(lq))
    p = torch.where(valid, lp.exp(), torch.zeros_like(lp))
    cap = math.exp(-losses.TAIL_FLOOR)
    q_tail = 1 - torch.minimum(q.sum(-1), torch.tensor(cap, dtype=q.dtype))
    p_tail = 1 - torch.minimum(p.sum(-1), torch.tensor(cap, dtype=p.dtype))
    safe = lambda x: torch.where(x > 0, x, torch.ones_like(x))  # noqa: E731
    reverse = (q * (safe(q).log() - safe(p).log())).sum(-1) + q_tail * (q_tail.log() - p_tail.log())
    forward = (p * (safe(p).log() - safe(q).log())).sum(-1) + p_tail * (p_tail.log() - q_tail.log())
    return beta * reverse + (1 - beta) * forward


def topk_inputs(g, k=4):
    support = torch.randint(0, SHARED, (N, k), generator=g)
    for i in range(N):  # distinct ids per row
        support[i] = torch.randperm(SHARED, generator=g)[:k]
    t_lp = torch.log(torch.rand(N, k, generator=g, dtype=torch.float64) * 0.2)
    valid = torch.ones(N, k, dtype=torch.bool)
    valid[0, -1] = False  # padding
    valid[3, 1:] = False
    return support, t_lp, valid


@pytest.mark.parametrize("beta", [1.0, 0.0, 0.3])
def test_topk_kl_matches_naive(beta):
    hidden, head, weights, g = make()
    support, t_lp, valid = topk_inputs(g)
    targets = support[:, 0]
    token_lp = t_lp[:, 0]
    vocab = losses.SharedVocab(SHARED, SWAP)
    ours = losses.topk_kl(hidden, head, targets, weights, vocab, support, t_lp, valid, token_lp, beta=beta)

    lq_all = project(F.log_softmax(head(hidden), -1))
    kl = naive_coarse_kl(lq_all.gather(1, support), t_lp, valid, beta)
    assert_same(ours, (weights * kl).sum(), hidden, head)
    assert ours[1]["k1"] == pytest.approx((lq_all.gather(1, targets[:, None]).squeeze(1) - token_lp).sum().item())


def test_coarse_kl_tail_edge_cases():
    # the teacher's support holds all its mass: tail floored, finite, penalizes student tail mass
    lq = torch.log(torch.tensor([[0.3, 0.2]], dtype=torch.float64))
    lp = torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float64))
    valid = torch.ones(1, 2, dtype=torch.bool)
    kl = losses.coarse_kl(lq, lp, valid, beta=1.0)
    assert torch.isfinite(kl).all()
    torch.testing.assert_close(kl, naive_coarse_kl(lq, lp, valid, 1.0))
    assert kl.item() > 6.0  # dominated by the student's 0.5 tail against a ~1e-6 teacher tail
    # identical coarse distributions: zero in both directions
    same = losses.coarse_kl(lp, lp, valid, beta=0.5)
    assert same.item() == pytest.approx(0.0, abs=1e-9)
    # no valid entry at all: both sides are one tail bucket, KL 0 and finite gradient
    lq = lq.clone().requires_grad_(True)
    kl = losses.coarse_kl(lq, lp, torch.zeros(1, 2, dtype=torch.bool), beta=1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-9)
    kl.backward()
    assert torch.isfinite(lq.grad).all()


# ------------------------------------------------------------------ sampled_rkl


def naive_pg(lp, advantage, behaviour, lo, hi):
    if behaviour is None:
        return -(torch.exp(lp - lp.detach()) * advantage)
    ratio = torch.exp(lp - behaviour)
    kept = ((ratio.detach() >= lo) & (ratio.detach() <= hi)).double()
    return -(ratio * advantage) * kept


@pytest.mark.parametrize("with_behaviour", [False, True])
def test_sampled_rkl_matches_naive(with_behaviour):
    hidden, head, weights, g = make()
    targets = torch.randint(0, V, (N,), generator=g)
    teacher_lp = torch.log(torch.rand(N, generator=g, dtype=torch.float64))
    lp_ref = F.log_softmax(head(hidden), -1).gather(1, targets[:, None]).squeeze(1)
    behaviour = None
    if with_behaviour:  # some ratios inside [0.5, 2], some outside (ICE-POP zeroes those)
        shift = torch.tensor([0.0, 0.3, -0.3, 1.0, -1.0, 0.1, 2.0, -0.6, 0.0, 0.69, -0.69], dtype=torch.float64)
        behaviour = (lp_ref + shift).detach()
    ours = losses.sampled_rkl(hidden, head, targets, weights, teacher_lp, behaviour)

    advantage = (teacher_lp - lp_ref).detach()
    ref = (weights * naive_pg(lp_ref, advantage, behaviour, 0.5, 2.0)).sum()
    assert_same(ours, ref, hidden, head)
    stats = ours[1]
    assert stats["k1"] == pytest.approx((lp_ref - teacher_lp).sum().item())
    assert stats["is_dropped"] == (3 if with_behaviour else 0)  # |shift| > ln 2


def test_sampled_rkl_gradient_is_the_reverse_kl_gradient():
    """On-policy, E_y[grad] equals grad KL(pi||p_T): check exactly by enumerating y.

    With a single position the expectation over the sampled token is a finite sum;
    weighting each sample by its probability must give the analytic gradient."""
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(1, H, generator=g, dtype=torch.float64).requires_grad_(True)
    head = nn.Linear(H, V, bias=False).double()
    t_logp = F.log_softmax(torch.randn(1, V, generator=g, dtype=torch.float64), -1)
    logp = F.log_softmax(head(hidden), -1)
    kl = (logp.exp() * (logp - t_logp)).sum()
    want = torch.autograd.grad(kl, hidden)[0]
    got = torch.zeros_like(hidden)
    for y in range(V):
        loss, _ = losses.sampled_rkl(
            hidden, head, torch.tensor([y]), torch.ones(1, dtype=torch.float64), t_logp[0, y : y + 1]
        )
        got += logp[0, y].exp().detach() * torch.autograd.grad(loss, hidden)[0]
    torch.testing.assert_close(got, want)


# ------------------------------------------------------------------------- xtok


def xtok_inputs(g):
    # 11 student tokens: chunks 0,0 | 1 | -1 (special) | 2,2,2 | 3 (masked: whitespace) | 4 | 5,5 | 6
    chunk = torch.tensor([0, 0, 1, -1, 2, 2, 2, 3, 4, 5, 5])
    keep = torch.tensor([True, True, True, False, True, True, True])
    teacher_chunk_lp = torch.log(torch.rand(7, generator=g, dtype=torch.float64))
    ends = [2, 3, 4, 7, 8, 9, 11]
    return losses.ChunkTargets(chunk, teacher_chunk_lp, keep, ends)


@pytest.mark.parametrize("spread", ["chunk", "proportional"])
@pytest.mark.parametrize("with_behaviour", [False, True])
def test_xtok_matches_naive(spread, with_behaviour):
    hidden, head, weights, g = make()
    targets = torch.randint(0, V, (N,), generator=g)
    chunks = xtok_inputs(g)
    lp = F.log_softmax(head(hidden), -1).gather(1, targets[:, None]).squeeze(1)
    behaviour = (lp + 0.2 * torch.randn(N, generator=g, dtype=torch.float64)).detach() if with_behaviour else None
    ours = losses.xtok(hidden, head, targets, weights, chunks, behaviour, spread=spread)

    on = (chunks.chunk >= 0) & chunks.keep[chunks.chunk.clamp(min=0)]
    advantage = torch.zeros(N, dtype=torch.float64)
    k1 = 0.0
    for c in range(7):
        members = (chunks.chunk == c) & on
        if not members.any():
            continue
        l_s = lp[members].detach().sum()
        a_c = chunks.teacher_lp[c] - l_s
        k1 += -a_c.item()
        if spread == "chunk":
            advantage[members] = a_c
        else:
            advantage[members] = a_c * lp[members].detach() / l_s
    ref = (weights * naive_pg(lp, advantage, behaviour, 0.5, 2.0) * on).sum()
    assert_same(ours, ref, hidden, head)
    assert ours[1]["k1"] == pytest.approx(k1)
    assert ours[1]["supervised"] == on.sum().item()


def test_xtok_dense_term_matches_naive():
    hidden, head, weights, g = make()
    targets = torch.randint(0, V, (N,), generator=g)
    chunks = xtok_inputs(g)
    rows = torch.tensor([2, 7, 8])  # one-to-one chunks
    support = torch.stack([torch.randperm(V, generator=g)[:5] for _ in rows])
    t_lp = torch.log(torch.rand(3, 5, generator=g, dtype=torch.float64) * 0.2)
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[1, 2] = False
    dense = losses.DenseTargets(rows, support, t_lp, valid)
    ours = losses.xtok(hidden, head, targets, weights, chunks, dense=dense, dense_weight=0.7, beta=0.5)
    base, _ = losses.xtok(hidden, head, targets, weights, chunks)

    logp = F.log_softmax(head(hidden), -1)
    kl = naive_coarse_kl(logp[rows].gather(1, support), t_lp, valid, 0.5)
    ref = base + 0.7 * (weights[rows] * kl).sum()
    assert_same(ours, ref, hidden, head)
    assert ours[1]["dense_tokens"] == 3


def test_xtok_slices_never_cut_a_chunk():
    bounds = losses.slice_bounds(11, 3, [2, 3, 4, 7, 8, 9, 11])
    assert bounds == [(0, 3), (3, 4), (4, 7), (7, 9), (9, 11)]
    # a chunk longer than a slice becomes one oversized slice
    assert losses.slice_bounds(10, 2, [1, 7, 10]) == [(0, 1), (1, 7), (7, 10)]
    assert losses.slice_bounds(7, 3) == [(0, 3), (3, 6), (6, 7)]


# ------------------------------------------------------------------------ rs_kd


def test_teacher_samples_estimate_the_distribution_without_bias():
    """Temperature 1: weighted draws average to p. Another proposal temperature: the
    self-normalized importance weights converge to p as draws grow."""
    from palingenesis.opd.teachers import sample_teacher

    g = torch.Generator().manual_seed(0)
    logp = F.log_softmax(torch.randn(4, 30, generator=g, dtype=torch.float64) * 2, -1)

    def estimate(rounds, temperature, repeats):
        total = torch.zeros_like(logp)
        for _ in range(repeats):
            ids, w, lp = sample_teacher(logp, rounds, temperature, g)
            torch.testing.assert_close(lp, logp.gather(1, ids))
            torch.testing.assert_close(w.sum(1), torch.ones(4, dtype=torch.float64))
            total += torch.zeros_like(logp).scatter_add_(1, ids, w)
        return total / repeats

    assert (estimate(12, 1.0, 4000) - logp.exp()).abs().max() < 0.01  # small samples, unbiased on average
    assert (estimate(20000, 0.7, 1) - logp.exp()).abs().max() < 0.01  # flatter proposal, reweighted


def test_rs_kd_gradient_is_the_forward_kl_gradient_in_expectation():
    from palingenesis.opd.teachers import sample_teacher

    hidden, head, weights, g = make()
    teacher_full = F.log_softmax(torch.randn(N, SHARED + 3, generator=g, dtype=torch.float64), -1)
    t_logp = teacher_full[:, :SHARED]  # the teacher's ids beyond SHARED have no student match
    vocab = losses.SharedVocab(SHARED, SWAP)
    targets = torch.randint(0, SHARED, (N,), generator=g)
    lq = project(F.log_softmax(head(hidden), -1))
    full = (weights * (t_logp.exp() * (t_logp - lq)).sum(-1)).sum()  # forward KL over the shared vocabulary
    want = grads(full, hidden, head)

    repeats, got, values = 400, [torch.zeros_like(x) for x in want], []
    for _ in range(repeats):
        ids, w, lp = sample_teacher(teacher_full, 16, 1.0, g)
        loss, stats = losses.rs_kd(
            hidden, head, targets, weights, vocab, ids, w, lp, t_logp.gather(1, targets[:, None]).squeeze(1)
        )
        values.append(loss.item())
        for acc, x in zip(got, grads(loss, hidden, head)):
            acc += x / repeats
        assert 1 <= stats["rs_unique"] / N <= 16
    for a, b in zip(got, want):
        assert ((a - b).norm() / b.norm()).item() < 0.05
    assert sum(values) / repeats == pytest.approx(full.item(), rel=0.05)


def test_rs_kd_drops_draws_outside_the_shared_vocabulary():
    hidden, head, weights, g = make()
    vocab = losses.SharedVocab(SHARED)
    ids = torch.tensor([[1, SHARED + 1]] * N)  # the second draw has no student counterpart
    w = torch.full((N, 2), 0.5, dtype=torch.float64)
    lp = torch.full((N, 2), -1.0, dtype=torch.float64)
    targets = torch.ones(N, dtype=torch.long)
    loss, _ = losses.rs_kd(hidden, head, targets, weights, vocab, ids, w, lp, torch.full((N,), -1.0))
    lq = F.log_softmax(head(hidden), -1)[:, 1]
    assert loss.item() == pytest.approx((weights * 0.5 * (-1.0 - lq)).sum().item())


def test_token_entropy():
    hidden, head, _, _ = make(softcap=True)
    lp = F.log_softmax(head(hidden), -1)
    torch.testing.assert_close(losses.token_entropy(hidden, head), -(lp.exp() * lp).sum(-1).detach())
