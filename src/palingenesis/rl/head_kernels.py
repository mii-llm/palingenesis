"""Fused vocabulary passes of the sampled-token log-probs (Triton, CUDA).

The head GEMM writes a slice of fp32 logits z [rows, V]. Everything else the loss needs from
that slice takes one read of it:

  forward   y = softcap(multiplier · z); an online log-sum-exp gives lse, log π(target) and,
            optionally, the entropy lse - Σ softmax(y) · y (its numerator accumulates with the
            same running-max rescaling as the sum)
  backward  dL/dz = g · (onehot(target) - softmax(y)) · dy/dz, written in bf16: the input of
            the backward GEMMs

instead of the separate max / subtract / exp / sum / gather / multiply / scatter / cast
kernels of the PyTorch expression, each a full pass over a 248k-wide vocabulary.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the PyTorch path is used
    triton = None

BLOCK = 4096


def available(t: Tensor) -> bool:
    return triton is not None and t.is_cuda


if triton is not None:

    @triton.jit
    def _transform(z, multiplier, softcap, HAS_SOFTCAP: tl.constexpr):
        y = z * multiplier
        if HAS_SOFTCAP:
            # softcap · tanh(y / softcap), with tanh(x) = 2 σ(2x) - 1
            y = softcap * (2.0 / (1.0 + tl.exp(-2.0 * y / softcap)) - 1.0)
        return y

    @triton.jit
    def _forward_kernel(
        Z,
        stride_z,
        TARGETS,
        LSE,
        LOGPROBS,
        ENTROPY,
        V,
        multiplier,
        softcap,
        HAS_SOFTCAP: tl.constexpr,
        WITH_ENTROPY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        z_row = Z + row.to(tl.int64) * stride_z
        offsets = tl.arange(0, BLOCK)
        # per-lane running max, sum of exp and sum of exp · y (reduced once at the end)
        m = tl.full([BLOCK], float("-inf"), tl.float32)
        s = tl.zeros([BLOCK], tl.float32)
        t = tl.zeros([BLOCK], tl.float32)
        for start in range(0, V, BLOCK):
            cols = start + offsets
            valid = cols < V
            y = _transform(tl.load(z_row + cols, mask=valid, other=0.0), multiplier, softcap, HAS_SOFTCAP)
            y = tl.where(valid, y, float("-inf"))
            m_new = tl.maximum(m, y)
            scale = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
            e = tl.where(valid, tl.exp(y - m_new), 0.0)
            s = s * scale + e
            if WITH_ENTROPY:
                t = t * scale + tl.where(valid, e * y, 0.0)
            m = m_new
        row_max = tl.max(m, 0)
        rescale = tl.where(m == float("-inf"), 0.0, tl.exp(m - row_max))
        total = tl.sum(s * rescale, 0)
        lse = row_max + tl.log(total)
        target = tl.load(TARGETS + row)
        y_target = _transform(tl.load(z_row + target), multiplier, softcap, HAS_SOFTCAP)
        tl.store(LSE + row, lse)
        tl.store(LOGPROBS + row, y_target - lse)
        if WITH_ENTROPY:
            tl.store(ENTROPY + row, lse - tl.sum(t * rescale, 0) / total)

    @triton.jit
    def _backward_kernel(
        Z,
        stride_z,
        TARGETS,
        LSE,
        GRAD,
        OUT,
        stride_out,
        V,
        multiplier,
        softcap,
        HAS_SOFTCAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        z_row = Z + row.to(tl.int64) * stride_z
        out_row = OUT + row.to(tl.int64) * stride_out
        g = tl.load(GRAD + row)
        lse = tl.load(LSE + row)
        target = tl.load(TARGETS + row)
        offsets = tl.arange(0, BLOCK)
        for start in range(0, V, BLOCK):
            cols = start + offsets
            valid = cols < V
            y = _transform(tl.load(z_row + cols, mask=valid, other=0.0), multiplier, softcap, HAS_SOFTCAP)
            grad = tl.where(cols == target, g, 0.0) - g * tl.exp(y - lse)
            if HAS_SOFTCAP:
                c = y / softcap
                grad = grad * (1.0 - c * c)
            grad = grad * multiplier
            tl.store(out_row + cols, grad.to(OUT.dtype.element_ty), mask=valid)


def forward(z: Tensor, targets: Tensor, multiplier: float, softcap: float | None, entropy: bool):
    """(log π(targets), lse, entropy or empty) of an fp32 logits slice z [rows, V]."""
    rows, vocab = z.shape
    lse = torch.empty(rows, dtype=torch.float32, device=z.device)
    lp = torch.empty(rows, dtype=torch.float32, device=z.device)
    ent = torch.empty(rows if entropy else 1, dtype=torch.float32, device=z.device)
    _forward_kernel[(rows,)](
        z,
        z.stride(0),
        targets,
        lse,
        lp,
        ent,
        vocab,
        float(multiplier),
        float(softcap or 1.0),
        HAS_SOFTCAP=bool(softcap),
        WITH_ENTROPY=entropy,
        BLOCK=BLOCK,
        num_warps=16,
    )
    return lp, lse, ent[:rows] if entropy else ent[:0]


def backward(
    z: Tensor, targets: Tensor, lse: Tensor, grad: Tensor, multiplier: float, softcap: float | None, out: Tensor
) -> Tensor:
    """dL/dz for dL/dlp = grad, written into `out` [rows, V] (bf16: the backward GEMMs' input)."""
    rows, vocab = z.shape
    _backward_kernel[(rows,)](
        z,
        z.stride(0),
        targets,
        lse,
        grad,
        out,
        out.stride(0),
        vocab,
        float(multiplier),
        float(softcap or 1.0),
        HAS_SOFTCAP=bool(softcap),
        BLOCK=BLOCK,
        num_warps=16,
    )
    return out
