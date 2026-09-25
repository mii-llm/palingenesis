"""full_rkl's fast path: exact full-vocabulary reverse KL with a fused, analytic gradient.

For a row with student logits y_s (vocabulary V_s) and teacher logits y_t, over
the shared ids v < n:

    q = softmax(y_s), p = softmax(y_t), d_v = log q_v - log p_v
    KL = sum_{v<n} q_v d_v,    S = sum_{v<n} q_v   (1 - S: the student's residual mass)
    dKL/dy_s[j] = q_j ((d_j + 1) [j < n] - (KL + S))

Each model's logits are its output projection z = h W^T followed by the model's
own transform (logits.PostProcessedHead): y = m z, then y = c tanh(y / c) with a
final-logit softcap c (Gemma 2), where m is a logit scale (Cohere, Granite,
Falcon-H1). The gradient reaches z through dy/dz = m (1 - (y / c)^2).

losses.full_rkl gets the same value and gradient through autograd, a slice of
rows at a time, with a dozen elementwise passes over the [rows, V] logits and
the head's weight cast to bf16 (and its gradient back) for every slice. Here,
per slice:

  1. two GEMMs give both models' logits in fp32 straight from the bf16 inputs
     (autocast rounds logits to bf16 first; these are exact to fp32 accumulation);
  2. one streaming pass per row (Triton) applies the transforms and computes both
     log-sum-exps, KL, S and the sampled-token estimate k1;
  3. a second pass writes the gradient above, in bf16, for the two backward GEMMs:
     into the slice's hidden states, and accumulated in fp32 into the head's weight.

The head weight is cast once per call, and statistics stay on the GPU until
the end (no host sync per slice). Without CUDA or Triton the same math runs in
plain torch (`_slice_torch`), which the tests compare against autograd.

Applies when both output heads are bias-free linear layers, optionally with
the model's logit transform, and the student's terminator needs no remapping
(SharedVocab.swap empty). Anything else takes losses.full_rkl.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from palingenesis.logits import PostProcessedHead

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover — CPU-only installs
    triton = None

# fp32 elements of one slice's logits, per model: 2^28 = 1 GiB (about 1080 rows at a 248k vocabulary).
SLICE_ELEMENTS = 2**28


@dataclass(frozen=True)
class Head:
    """An output head as the kernels see it: logits = transform(hidden @ weight^T)."""

    weight: Tensor
    multiplier: float = 1.0
    softcap: float = 0.0  # 0: none


def unwrap(head: nn.Module) -> Head | None:
    """The head's weight and logit transform, or None when the fused path cannot compute it."""
    multiplier, softcap = 1.0, 0.0
    if isinstance(head, PostProcessedHead):
        multiplier, softcap = head.multiplier, head.softcap or 0.0
        head = head.head
    if type(head) is not nn.Linear or head.bias is not None:
        return None
    return Head(head.weight, float(multiplier), float(softcap))


def supported(head: nn.Module, teacher_head: nn.Module, swap: dict) -> bool:
    """Whether fused_full_rkl computes exactly what losses.full_rkl would for these heads."""
    return not swap and unwrap(head) is not None and unwrap(teacher_head) is not None


if triton is not None:

    @triton.jit
    def _transform(z, valid, multiplier, softcap, SOFTCAP: tl.constexpr):
        """The model's logits from the projection's (-inf where not `valid`)."""
        y = z * multiplier
        if SOFTCAP:
            y = softcap * libdevice.tanh(y / softcap)
        return tl.where(valid, y, -float("inf"))

    @triton.jit
    def _stats_kernel(
        zs_ptr,
        zt_ptr,
        target_ptr,
        out_ptr,
        v_s,
        v_t,
        n_shared,
        stride_s,
        stride_t,
        mult_s,
        cap_s,
        mult_t,
        cap_t,
        SOFTCAP_S: tl.constexpr,
        SOFTCAP_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per row: lse_s, lse_t, KL, S, k1 in one streaming pass over both rows of logits."""
        row = tl.program_id(0).to(tl.int64)
        zs_row = zs_ptr + row * stride_s
        zt_row = zt_ptr + row * stride_t
        m_s = -float("inf")
        l_s = 0.0
        m_t = -float("inf")
        l_t = 0.0
        a = 0.0  # sum_{v<n} exp(y_s - m_s) (y_s - y_t)
        b = 0.0  # sum_{v<n} exp(y_s - m_s)
        v_max = tl.maximum(v_s, v_t)
        for start in range(0, v_max, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            ys = _transform(tl.load(zs_row + cols, mask=cols < v_s, other=0.0), cols < v_s, mult_s, cap_s, SOFTCAP_S)
            yt = _transform(tl.load(zt_row + cols, mask=cols < v_t, other=0.0), cols < v_t, mult_t, cap_t, SOFTCAP_T)
            new_m_s = tl.maximum(m_s, tl.max(ys, 0))
            scale = tl.exp(m_s - new_m_s)
            e = tl.exp(ys - new_m_s)
            shared = cols < n_shared
            l_s = l_s * scale + tl.sum(e, 0)
            a = a * scale + tl.sum(tl.where(shared, e * (ys - yt), 0.0), 0)
            b = b * scale + tl.sum(tl.where(shared, e, 0.0), 0)
            m_s = new_m_s
            new_m_t = tl.maximum(m_t, tl.max(yt, 0))
            l_t = l_t * tl.exp(m_t - new_m_t) + tl.sum(tl.exp(yt - new_m_t), 0)
            m_t = new_m_t
        lse_s = m_s + tl.log(l_s)
        lse_t = m_t + tl.log(l_t)
        mass = b / l_s
        kl = a / l_s + mass * (lse_t - lse_s)
        target = tl.load(target_ptr + row)
        ys_target = _transform(tl.load(zs_row + target), True, mult_s, cap_s, SOFTCAP_S)
        yt_target = _transform(tl.load(zt_row + target), True, mult_t, cap_t, SOFTCAP_T)
        k1 = (ys_target - lse_s) - (yt_target - lse_t)
        out = out_ptr + row * 5
        tl.store(out, lse_s)
        tl.store(out + 1, lse_t)
        tl.store(out + 2, kl)
        tl.store(out + 3, mass)
        tl.store(out + 4, k1)

    @triton.jit
    def _grad_kernel(
        zs_ptr,
        zt_ptr,
        stats_ptr,
        weight_ptr,
        grad_ptr,
        v_s,
        n_shared,
        stride_s,
        stride_t,
        stride_g,
        mult_s,
        cap_s,
        mult_t,
        cap_t,
        SOFTCAP_S: tl.constexpr,
        SOFTCAP_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """grad[j] = w q_j ((d_j + 1) [j < n] - (KL + S)) dy_s/dz_s, in the gradient buffer's dtype."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = cols < v_s
        stats = stats_ptr + row * 5
        lse_s = tl.load(stats)
        lse_t = tl.load(stats + 1)
        c = tl.load(stats + 2) + tl.load(stats + 3)
        w = tl.load(weight_ptr + row)
        shared = cols < n_shared
        ys = _transform(tl.load(zs_ptr + row * stride_s + cols, mask=valid, other=0.0), valid, mult_s, cap_s, SOFTCAP_S)
        yt = _transform(
            tl.load(zt_ptr + row * stride_t + cols, mask=shared, other=0.0), shared, mult_t, cap_t, SOFTCAP_T
        )
        lq = ys - lse_s
        g = w * tl.exp(lq) * (tl.where(shared, lq - (yt - lse_t) + 1.0, 0.0) - c)
        if SOFTCAP_S:  # dtanh(x) = 1 - tanh(x)^2 = 4 sigmoid(2x) sigmoid(-2x): no cancellation near saturation
            x2 = 2.0 * mult_s * tl.load(zs_ptr + row * stride_s + cols, mask=valid, other=0.0) / cap_s
            g = g * mult_s * 4.0 * tl.sigmoid(x2) * tl.sigmoid(-x2)
        else:
            g = g * mult_s
        tl.store(grad_ptr + row * stride_g + cols, g.to(grad_ptr.dtype.element_ty), mask=valid)


def _slice_triton(
    zs: Tensor, zt: Tensor, targets: Tensor, weights: Tensor, n_shared: int, grad_dtype, student: Head, teacher: Head
):
    rows, v_s = zs.shape
    transforms = (student.multiplier, student.softcap or 1.0, teacher.multiplier, teacher.softcap or 1.0)
    flags = {"SOFTCAP_S": student.softcap > 0, "SOFTCAP_T": teacher.softcap > 0}
    stats = torch.empty(rows, 5, device=zs.device, dtype=torch.float32)
    _stats_kernel[(rows,)](
        zs,
        zt,
        targets,
        stats,
        v_s,
        zt.shape[1],
        n_shared,
        zs.stride(0),
        zt.stride(0),
        *transforms,
        **flags,
        BLOCK=4096,
        num_warps=16,
    )
    grad = None
    if grad_dtype is not None:
        grad = torch.empty(rows, v_s, device=zs.device, dtype=grad_dtype)
        block = 4096
        _grad_kernel[(rows, triton.cdiv(v_s, block))](
            zs,
            zt,
            stats,
            weights,
            grad,
            v_s,
            n_shared,
            zs.stride(0),
            zt.stride(0),
            grad.stride(0),
            *transforms,
            **flags,
            BLOCK=block,
            num_warps=8,
        )
    return stats, grad


def _apply(z: Tensor, head: Head) -> Tensor:
    y = z * head.multiplier
    return torch.tanh(y / head.softcap) * head.softcap if head.softcap else y


def _slice_torch(
    zs: Tensor, zt: Tensor, targets: Tensor, weights: Tensor, n_shared: int, grad_dtype, student: Head, teacher: Head
):
    """The same statistics and gradient as the kernels, in plain torch."""
    ys, yt = _apply(zs, student), _apply(zt, teacher)
    lq = torch.log_softmax(ys, -1)
    lp = torch.log_softmax(yt, -1)
    q = lq.exp()
    d = lq[:, :n_shared] - lp[:, :n_shared]
    kl = (q[:, :n_shared] * d).sum(-1)
    mass = q[:, :n_shared].sum(-1)
    rows = torch.arange(zs.shape[0], device=zs.device)
    k1 = lq[rows, targets] - lp[rows, targets]
    stats = torch.stack([ys.logsumexp(-1), yt.logsumexp(-1), kl, mass, k1], 1)
    grad = None
    if grad_dtype is not None:
        inner = -(kl + mass)[:, None].expand_as(q).clone()
        inner[:, :n_shared] += d + 1
        grad = weights[:, None] * q * inner * student.multiplier
        if student.softcap:
            grad = grad * (1 - (ys / student.softcap) ** 2)
        grad = grad.to(grad_dtype)
    return stats, grad


def _mm_fp32(a: Tensor, b: Tensor) -> Tensor:
    """a @ b in at least fp32: fp32 accumulation and output for bf16 inputs on CUDA."""
    if a.dtype == torch.bfloat16:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a, b)


class _FusedRKL(torch.autograd.Function):
    """sum_t weights[t] * KL_t, with gradients into the student's hidden states and head weight.

    As losses._ChunkedHead: the forward computes the gradients slice by slice,
    the backward scales them by the incoming gradient."""

    @staticmethod
    def forward(ctx, hidden, weight, teacher_hidden, student, teacher, targets, weights, n_shared, rows, stats_out):
        cuda = hidden.is_cuda
        # GEMM inputs: bf16 on the GPU (as autocast), the inputs' own precision (>= fp32) on the CPU
        dtype = torch.bfloat16 if cuda else torch.promote_types(hidden.dtype, torch.float32)
        acc = torch.float32 if cuda else dtype
        slice_fn = _slice_triton if cuda and triton is not None else _slice_torch
        h = hidden.detach().to(dtype)
        w = weight.detach().to(dtype)
        th = teacher_hidden.to(dtype)
        tw = teacher.weight.detach().to(dtype)
        weights = weights.to(acc)
        want_hidden, want_weight = hidden.requires_grad, weight.requires_grad
        grad_hidden = torch.empty_like(hidden) if want_hidden else None
        grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=acc) if want_weight else None
        totals = torch.zeros(4, device=hidden.device, dtype=acc)  # loss, kl, k1, residual
        for a in range(0, h.shape[0], rows):
            b = min(a + rows, h.shape[0])
            zs = _mm_fp32(h[a:b], w.T)
            zt = _mm_fp32(th[a:b], tw.T)
            stats, grad = slice_fn(
                zs,
                zt,
                targets[a:b],
                weights[a:b],
                n_shared,
                dtype if want_hidden or want_weight else None,
                student,
                teacher,
            )
            del zs, zt
            kl = stats[:, 2]
            totals += torch.stack([(weights[a:b] * kl).sum(), kl.sum(), stats[:, 4].sum(), (1 - stats[:, 3]).sum()])
            if want_hidden:
                grad_hidden[a:b] = torch.mm(grad, w).to(hidden.dtype)
            if want_weight:
                if grad.dtype == torch.bfloat16:
                    torch.addmm(grad_weight, grad.T, h[a:b], out_dtype=torch.float32, out=grad_weight)
                else:
                    grad_weight.addmm_(grad.T, h[a:b])
        stats_out.append(totals)
        ctx.has = (want_hidden, want_weight)
        ctx.save_for_backward(
            grad_hidden if want_hidden else torch.empty(0),
            grad_weight.to(weight.dtype) if want_weight else torch.empty(0),
        )
        return totals[0].clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            grad_hidden * grad_output if ctx.has[0] else None,
            grad_weight * grad_output if ctx.has[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_full_rkl(
    hidden: Tensor,
    head: nn.Module,
    teacher_hidden: Tensor,
    teacher_head: nn.Module,
    targets: Tensor,
    weights: Tensor,
    n_shared: int,
    rows: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """losses.full_rkl for heads `supported` accepts, fused: (sum_t weights[t] KL_t, stats sums).

    `targets` are the completion ids (k1); ids below `n_shared` are the same token
    in both vocabularies. Stats: kl, k1, residual, summed over tokens."""
    student, teacher = unwrap(head), unwrap(teacher_head)
    if student is None or teacher is None:
        raise ValueError("fused_full_rkl needs bias-free linear heads (see fused_rkl.supported)")
    rows = rows or max(1, SLICE_ELEMENTS // max(student.weight.shape[0], teacher.weight.shape[0]))
    stats: list[Tensor] = []
    with torch.autocast(hidden.device.type, enabled=False):  # the dtypes are chosen explicitly
        loss = _FusedRKL.apply(
            hidden, student.weight, teacher_hidden, student, teacher, targets, weights, n_shared, rows, stats
        )
    kl, k1, residual = stats[0][1:].tolist()  # the call's one host sync
    return loss, {"kl": kl, "k1": k1, "residual": residual}
