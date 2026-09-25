"""The fused full_rkl path against autograd through the definition.

CPU: the torch implementation of the fused math, in fp64. GPU (when present):
the Triton kernels against the same math in fp64, at a real vocabulary size.
"""

import sys

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
F = torch.nn.functional

from palingenesis.opd import fused_rkl, losses  # noqa: E402

# (multiplier, softcap) of the student's and the teacher's logits: none, Granite-like scales, Gemma 2-like softcaps
TRANSFORMS = [((1.0, None), (1.0, None)), ((0.125, None), (0.0625, None)), ((1.0, 3.0), (1.3, 2.5))]


def setup(n, h, v_s, v_t, device="cpu", dtype=torch.float64, seed=0, transforms=TRANSFORMS[0]):
    from palingenesis.logits import PostProcessedHead

    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (torch.randn(n, h, generator=g, dtype=dtype) * 2).to(device).requires_grad_()
    head = torch.nn.Linear(h, v_s, bias=False, dtype=dtype, device=device)
    teacher_head = torch.nn.Linear(h + 3, v_t, bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        head.weight.copy_(torch.randn(v_s, h, generator=g, dtype=dtype) / h**0.5 * 3)
        teacher_head.weight.copy_(torch.randn(v_t, h + 3, generator=g, dtype=dtype) / h**0.5 * 3)
    teacher_hidden = torch.randn(n, h + 3, generator=g, dtype=dtype).to(device)
    weights = torch.rand(n, generator=g, dtype=dtype).to(device)
    (m_s, c_s), (m_t, c_t) = transforms
    if (m_s, c_s) != (1.0, None):
        head = PostProcessedHead(head, m_s, c_s)
    if (m_t, c_t) != (1.0, None):
        teacher_head = PostProcessedHead(teacher_head, m_t, c_t)
    return hidden, head, teacher_hidden, teacher_head, weights


def reference(hidden, head, teacher_hidden, teacher_head, targets, weights, n_shared):
    lq = F.log_softmax(head(hidden), -1)[:, :n_shared]
    lp = F.log_softmax(teacher_head(teacher_hidden), -1)[:, :n_shared]
    kl = (lq.exp() * (lq - lp)).sum(-1)
    k1 = lq.gather(1, targets[:, None]).squeeze(1) - lp.gather(1, targets[:, None]).squeeze(1)
    residual = 1 - lq.exp().sum(-1)
    return (weights * kl).sum(), {"kl": kl.sum().item(), "k1": k1.sum().item(), "residual": residual.sum().item()}


def grads(loss, hidden, head):
    return torch.autograd.grad(loss, [hidden, head.weight])


@pytest.mark.parametrize("v_s, v_t, n_shared", [(40, 40, 40), (43, 40, 37), (40, 45, 40)])
@pytest.mark.parametrize("rows", [None, 3])
@pytest.mark.parametrize("transforms", TRANSFORMS)
def test_fused_matches_the_definition(v_s, v_t, n_shared, rows, transforms):
    """Same vocabularies, a student-only tail (residual mass), a teacher-only tail; one slice or many;
    plain, scaled and softcapped logits."""
    hidden, head, t_hidden, t_head, weights = setup(11, 8, v_s, v_t, transforms=transforms)
    targets = torch.randint(0, n_shared, (11,))
    want, want_stats = reference(hidden, head, t_hidden, t_head, targets, weights, n_shared)
    got, stats = fused_rkl.fused_full_rkl(hidden, head, t_hidden, t_head, targets, weights, n_shared, rows=rows)
    torch.testing.assert_close(got, want)
    assert stats == pytest.approx(want_stats, abs=1e-10)
    for a, b in zip(grads(2.5 * got, hidden, head), grads(2.5 * want, hidden, head)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("transforms", TRANSFORMS)
def test_fused_agrees_with_the_generic_loss(transforms):
    hidden, head, t_hidden, t_head, weights = setup(9, 8, 30, 30, transforms=transforms)
    targets = torch.randint(0, 30, (9,))
    t_logp = F.log_softmax(t_head(t_hidden), -1)
    generic, g_stats = losses.full_rkl(hidden, head, targets, weights, losses.SharedVocab(30), lambda a, b: t_logp[a:b])
    fused, f_stats = fused_rkl.fused_full_rkl(hidden, head, t_hidden, t_head, targets, weights, 30)
    torch.testing.assert_close(fused, generic)
    assert f_stats == pytest.approx(g_stats, abs=1e-10)
    for a, b in zip(grads(fused, hidden, head), grads(generic, hidden, head)):
        torch.testing.assert_close(a, b)


def test_no_grad_and_frozen_head():
    hidden, head, t_hidden, t_head, weights = setup(7, 8, 30, 30)
    targets = torch.randint(0, 30, (7,))
    with torch.no_grad():
        value, _ = fused_rkl.fused_full_rkl(hidden, head, t_hidden, t_head, targets, weights, 30)
    want, _ = reference(hidden, head, t_hidden, t_head, targets, weights, 30)
    torch.testing.assert_close(value, want.detach())
    head.weight.requires_grad_(False)
    got, _ = fused_rkl.fused_full_rkl(hidden, head, t_hidden, t_head, targets, weights, 30)
    torch.testing.assert_close(torch.autograd.grad(got, hidden)[0], torch.autograd.grad(want, hidden)[0])


def test_supported_for_bias_free_heads_with_any_transform_without_swap():
    from palingenesis.logits import PostProcessedHead

    plain = torch.nn.Linear(4, 10, bias=False)
    capped = PostProcessedHead(plain, 0.5, 30.0)
    assert fused_rkl.supported(plain, plain, {})
    assert fused_rkl.supported(capped, plain, {})
    assert fused_rkl.unwrap(capped) == fused_rkl.Head(plain.weight, 0.5, 30.0)
    assert not fused_rkl.supported(plain, plain, {9: 8})
    assert not fused_rkl.supported(torch.nn.Linear(4, 10), plain, {})
    assert not fused_rkl.supported(PostProcessedHead(torch.nn.Linear(4, 10), 0.5), plain, {})


@pytest.mark.skipif(not torch.cuda.is_available() or fused_rkl.triton is None, reason="needs CUDA and Triton")
@pytest.mark.parametrize(
    "v_s, v_t, n_shared, transforms",
    [
        (248320, 248320, 248320, TRANSFORMS[0]),  # Qwen3.5
        (151936, 151669, 151643, TRANSFORMS[0]),  # a student-only tail
        (256000, 256000, 256000, ((1.0, 30.0), (1.0, 30.0))),  # Gemma 2: softcap 30
        (49159, 49159, 49159, ((0.125, None), (0.0625, None))),  # Granite 3.3 2B / 8B: logits_scaling 8 / 16
    ],
)
def test_triton_kernels_match_fp64(v_s, v_t, n_shared, transforms):
    """The kernels on fp32 logits against the fused math in fp64, at real vocabulary sizes."""
    (m_s, c_s), (m_t, c_t) = transforms
    student = fused_rkl.Head(None, m_s, c_s or 0.0)
    teacher = fused_rkl.Head(None, m_t, c_t or 0.0)
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = 64
    zs = torch.randn(rows, v_s, device="cuda", generator=g) * 4
    zt = (
        zs[:, :v_t] + torch.randn(rows, v_t, device="cuda", generator=g)
        if v_t <= v_s
        else torch.randn(rows, v_t, device="cuda", generator=g) * 4
    )
    targets = torch.randint(0, n_shared, (rows,), device="cuda", generator=g)
    weights = torch.rand(rows, device="cuda", generator=g)
    zs, zt = zs / m_s, zt / m_t  # logits of the usual scale after the transform
    stats, grad = fused_rkl._slice_triton(zs, zt, targets, weights, n_shared, torch.float32, student, teacher)
    ref_stats, ref_grad = fused_rkl._slice_torch(
        zs.double(), zt.double(), targets, weights.double(), n_shared, torch.float64, student, teacher
    )
    fp32_stats, fp32_grad = fused_rkl._slice_torch(zs, zt, targets, weights, n_shared, torch.float32, student, teacher)
    torch.testing.assert_close(stats.double(), ref_stats, rtol=1e-5, atol=2e-5)

    def norm_error(grad):
        return ((grad.double() - ref_grad).norm() / ref_grad.norm()).item()

    # as exact as fp32 allows: within a small factor of the same math in fp32 torch (~1e-6)
    assert norm_error(grad) < 2 * norm_error(fp32_grad) + 1e-7


@pytest.mark.skipif(not torch.cuda.is_available() or fused_rkl.triton is None, reason="needs CUDA and Triton")
def test_fused_on_gpu_is_closer_to_fp64_than_autocast():
    """bf16 GEMM inputs as under autocast, but fp32 logits: closer to the fp64 definition than losses.full_rkl."""
    hidden, head, t_hidden, t_head, weights = setup(300, 256, 50000, 50000, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 50000, (300,), device="cuda")
    weights = weights / 300
    h, w, th, tw = (
        t.detach().double().requires_grad_(i < 2) for i, t in enumerate((hidden, head.weight, t_hidden, t_head.weight))
    )
    lq, lp = F.log_softmax(h @ w.T, -1), F.log_softmax(th @ tw.T, -1)
    want = (weights.double() * (lq.exp() * (lq - lp)).sum(-1)).sum()
    want_grads = torch.autograd.grad(want, [h, w])

    def errors(loss):
        return [abs(loss.item() / want.item() - 1)] + [
            ((a.double() - b).norm() / b.norm()).item() for a, b in zip(grads(loss, hidden, head), want_grads)
        ]

    fused, _ = fused_rkl.fused_full_rkl(hidden, head, t_hidden, t_head, targets, weights, 50000, rows=128)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generic, _ = losses.full_rkl(
            hidden,
            head,
            targets,
            weights,
            losses.SharedVocab(50000),
            lambda a, b: torch.log_softmax(t_head(t_hidden[a:b]).float(), -1),
        )
    ours, autocast = errors(fused), errors(generic)
    assert ours[0] < 1e-5 and max(ours[1:]) < 0.02
    assert all(o < a for o, a in zip(ours, autocast))
