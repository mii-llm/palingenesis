"""Multi-tensor AdamW (Triton): one kernel launch steps every parameter of a dtype bucket.

Why: the per-parameter paths launch one kernel (or a compiled function) per parameter, hundreds per optimizer step,
each with its own launch latency and, for a torch.compile'd update, its own guard evaluation on the host. For a
0.8B-2B model the step is then host-bound rather than bound by the ~14 bytes per parameter it has to move.

How (after NVIDIA Apex's multi_tensor_apply and torch's fused / _foreach optimizers, both BSD-3; re-implemented, no
code copied): per step the host builds one small table per bucket, the element offset of every tensor (weight,
gradient, moments, compensation) from the bucket's base tensors, the tensor's noise seed and its six AdamW scalars,
and copies it to the GPU with one non-blocking transfer from pinned memory. A block map (cached by the tensors'
sizes) gives every program its (tensor, first element); each program reads its tensor's row of the table and steps
`BLOCK * ITERS` elements. A bucket is (parameter dtype, gradient dtype, update kind), so a bf16 model with bf16
gradients takes one launch for all its parameters. Pointers are never built from integers: every tensor is
addressed as base pointer + int64 element offset, which keeps Triton's alignment analysis (16-byte vector loads).

Semantics: the bf16 kernels compute exactly the operations of `palingenesis.bf16_adamw`'s reference update, in the
same order, with IEEE square root and division and without FMA contraction (`enable_fp_fusion=False`), so a step
gives the same bits as the eager reference (`compile=False`) on the same GPU: the counter-based stochastic rounding
(hash of the element's offset and a per-(parameter, step) seed) is unchanged, so resumed runs stay bit-exact. The
fp32 kernel is torch.optim.AdamW's update (decoupled weight decay, bias corrections), equal to it up to rounding.

Gradient clipping can be folded into the step: `sum_of_squares` reduces every gradient in one launch per bucket,
and the clip factor is multiplied into the fp32 gradient inside the step kernel, instead of a separate pass that
reads and writes every gradient.

Off CUDA, or without Triton, every function here runs the same math with torch ops, per tensor.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

BLOCK = 1024  # elements per inner iteration of a program (8 x 16-byte vectors per thread of 4 warps)
ITERS = 8  # inner iterations: a program steps BLOCK * ITERS elements of one tensor
CHUNK = BLOCK * ITERS
NUM_WARPS = 4
_ALIGN = 16  # bytes: Triton's vector width


def triton_available() -> tuple[bool, str]:
    """Whether the Triton kernels can run in this process (and why not)."""
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "triton is not installed"
    return True, ""


def local(t: torch.Tensor) -> torch.Tensor:
    """The rank's local tensor of an FSDP2 DTensor (attribute access: to_local() costs a dispatch), else `t`."""
    inner = getattr(t, "_local_tensor", None)
    return inner if inner is not None else t


# ----------------------------------------------------------------------------------------------------------------
# Block map (host side, cached): which tensor and which elements each program steps
# ----------------------------------------------------------------------------------------------------------------


def block_map(numels: Sequence[int], chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """(tensor index int32 [B], first element int64 [B]) of the B programs that cover tensors of `numels` elements,
    `chunk` elements per program, in tensor order. Tensors of 0 elements get no program."""
    counts = torch.tensor([(n + chunk - 1) // chunk for n in numels], dtype=torch.int64)
    if counts.numel() == 0 or int(counts.sum()) == 0:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int64)
    tensor = torch.repeat_interleave(torch.arange(len(numels), dtype=torch.int64), counts)
    first_block = torch.cumsum(counts, 0) - counts
    start = (torch.arange(int(counts.sum()), dtype=torch.int64) - first_block[tensor]) * chunk
    return tensor.to(torch.int32), start


@dataclass
class _Map:
    numels: torch.Tensor  # int64 [T] on the device
    tensor: torch.Tensor  # int32 [B]
    start: torch.Tensor  # int64 [B]
    blocks: int


class _MapCache:
    """Block maps on the device, keyed by (device, sizes): built once, reused every step."""

    def __init__(self, limit: int = 32):
        self._maps: dict[tuple, _Map] = {}
        self._limit = limit

    def get(self, device: torch.device, numels: tuple[int, ...]) -> _Map:
        key = (device, numels)
        m = self._maps.get(key)
        if m is None:
            if len(self._maps) >= self._limit:  # a changing set of parameters: drop the oldest
                self._maps.pop(next(iter(self._maps)))
            tensor, start = block_map(numels)
            m = _Map(
                torch.tensor(numels, dtype=torch.int64).to(device),
                tensor.to(device),
                start.to(device),
                int(tensor.numel()),
            )
            self._maps[key] = m
        return m


_MAPS = _MapCache()


# ----------------------------------------------------------------------------------------------------------------
# Kernels (built on first use: importing this module does not import Triton)
# ----------------------------------------------------------------------------------------------------------------

_KERNELS: dict | None = None


def _kernels() -> dict:
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _hash32(x):
        # palingenesis.bf16_adamw.hash32 on uint32 (products wrap mod 2^32, shifts are logical): the same bits
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        x = x * 0x5BD1E995
        return x ^ (x >> 16)

    @triton.jit
    def _round_stochastic(x, noise):
        # palingenesis.bf16_adamw.round_bf16: add 16 noise bits below the bf16 mantissa, truncate; a carry into the
        # infinity exponent truncates instead (saturates); inf and nan take the plain cast
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7F800000) != 0x7F800000
        hi = (bits + noise) >> 16
        hi = tl.where((hi & 0x7FFF) == 0x7F80, bits >> 16, hi)
        out = hi.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        return tl.where(finite, out, x.to(tl.bfloat16))

    @triton.jit
    def _round_nearest(x):
        # palingenesis.bf16_adamw.round_bf16_nearest: round-to-nearest-even on the bits, and the result as fp32
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7F800000) != 0x7F800000
        hi = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        out = tl.where(finite, hi.to(tl.uint16).to(tl.bfloat16, bitcast=True), x.to(tl.bfloat16))
        as_f32 = tl.where(finite, (hi << 16).to(tl.float32, bitcast=True), x)
        return out, as_f32

    @triton.jit
    def _adamw_bf16_kernel(
        p_base,
        g_base,
        m_base,
        v_base,
        c_base,
        table,  # int64 [T, 6]: element offsets of p, g, m, v, c from their bases; the noise seed
        hp,  # fp32 [T, 6]: decay, beta1, beta2, step_size, 1 / sqrt(bias_correction2), eps
        numels,  # int64 [T]
        block_tensor,  # int32 [B]
        block_start,  # int64 [B]
        scale_ptr,  # fp32 [1]: gradient scale (clip factor), read when HAS_SCALE
        KAHAN: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        ALIGNED: tl.constexpr,
        G_VEC: tl.constexpr,  # gradient elements per 16 bytes
        BLOCK: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = tl.load(block_tensor + pid).to(tl.int64)
        start = tl.load(block_start + pid)
        n = tl.load(numels + t)
        row = table + t * 6
        op = tl.load(row)
        og = tl.load(row + 1)
        om = tl.load(row + 2)
        ov = tl.load(row + 3)
        oc = tl.load(row + 4)
        seed = tl.load(row + 5)
        if ALIGNED:  # every program starts on a multiple of BLOCK * ITERS elements; offsets keep 16-byte alignment
            start = tl.multiple_of(start, BLOCK)
            op = tl.multiple_of(op, 8)
            og = tl.multiple_of(og, G_VEC)
            om = tl.multiple_of(om, 8)
            ov = tl.multiple_of(ov, 8)
            oc = tl.multiple_of(oc, 8)
        h = hp + t * 6
        decay = tl.load(h)
        beta1 = tl.load(h + 1)
        beta2 = tl.load(h + 2)
        step_size = tl.load(h + 3)
        inv_bc2_sqrt = tl.load(h + 4)
        eps = tl.load(h + 5)
        if HAS_SCALE:
            scale = tl.load(scale_ptr)
        for i in range(ITERS):
            idx = start + i * BLOCK + tl.arange(0, BLOCK)
            mask = idx < n
            gf = tl.load(g_base + og + idx, mask=mask, other=0.0).to(tl.float32)
            if HAS_SCALE:
                gf = gf * scale
            mf = tl.load(m_base + om + idx, mask=mask, other=0.0).to(tl.float32) * beta1 + gf * (1.0 - beta1)
            vf = tl.load(v_base + ov + idx, mask=mask, other=0.0).to(tl.float32) * beta2 + gf * gf * (1.0 - beta2)
            pf = tl.load(p_base + op + idx, mask=mask, other=0.0).to(tl.float32)
            if KAHAN:
                pf = pf + tl.load(c_base + oc + idx, mask=mask, other=0.0).to(tl.float32)
            pf = pf * decay - tl.div_rn(step_size * mf, tl.sqrt_rn(vf) * inv_bc2_sqrt + eps)
            r = _hash32((idx + seed).to(tl.uint32))
            if KAHAN:
                stored, stored_f32 = _round_nearest(pf)
                tl.store(p_base + op + idx, stored, mask=mask)
                tl.store(c_base + oc + idx, (pf - stored_f32).to(tl.bfloat16), mask=mask)
            else:
                tl.store(p_base + op + idx, _round_stochastic(pf, r & 0xFFFF), mask=mask)
            tl.store(v_base + ov + idx, _round_stochastic(vf, r >> 16), mask=mask)
            tl.store(m_base + om + idx, mf.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _adamw_kernel(
        p_base,
        g_base,
        m_base,
        v_base,
        table,  # int64 [T, 4]: element offsets of p, g, m, v from their bases
        hp,  # fp32 [T, 6]
        numels,
        block_tensor,
        block_start,
        scale_ptr,
        HAS_SCALE: tl.constexpr,
        ALIGNED: tl.constexpr,
        P_VEC: tl.constexpr,
        G_VEC: tl.constexpr,
        BLOCK: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        # torch.optim.AdamW (and BF16AdamW._step_fp32): p *= decay; m = lerp(m, g, 1 - beta1);
        # v = v * beta2 + (1 - beta2) g^2; p -= step_size * m / (sqrt(v) / sqrt(bias_correction2) + eps)
        pid = tl.program_id(0)
        t = tl.load(block_tensor + pid).to(tl.int64)
        start = tl.load(block_start + pid)
        n = tl.load(numels + t)
        row = table + t * 4
        op = tl.load(row)
        og = tl.load(row + 1)
        om = tl.load(row + 2)
        ov = tl.load(row + 3)
        if ALIGNED:
            start = tl.multiple_of(start, BLOCK)
            op = tl.multiple_of(op, P_VEC)
            og = tl.multiple_of(og, G_VEC)
            om = tl.multiple_of(om, P_VEC)
            ov = tl.multiple_of(ov, P_VEC)
        h = hp + t * 6
        decay = tl.load(h)
        beta1 = tl.load(h + 1)
        beta2 = tl.load(h + 2)
        step_size = tl.load(h + 3)
        inv_bc2_sqrt = tl.load(h + 4)
        eps = tl.load(h + 5)
        if HAS_SCALE:
            scale = tl.load(scale_ptr)
        for i in range(ITERS):
            idx = start + i * BLOCK + tl.arange(0, BLOCK)
            mask = idx < n
            gf = tl.load(g_base + og + idx, mask=mask, other=0.0).to(tl.float32)
            if HAS_SCALE:
                gf = gf * scale
            m = tl.load(m_base + om + idx, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_base + ov + idx, mask=mask, other=0.0).to(tl.float32)
            p = tl.load(p_base + op + idx, mask=mask, other=0.0).to(tl.float32) * decay
            m = m + (1.0 - beta1) * (gf - m)
            v = v * beta2 + (1.0 - beta2) * gf * gf
            p = p - step_size * tl.div_rn(m, tl.sqrt_rn(v) * inv_bc2_sqrt + eps)
            tl.store(m_base + om + idx, m.to(m_base.dtype.element_ty), mask=mask)
            tl.store(v_base + ov + idx, v.to(v_base.dtype.element_ty), mask=mask)
            tl.store(p_base + op + idx, p.to(p_base.dtype.element_ty), mask=mask)

    @triton.jit
    def _sum_of_squares_kernel(
        g_base,
        table,  # int64 [T]: element offsets from the base
        numels,
        block_tensor,
        block_start,
        out,  # fp32 [B]: each program's partial sum
        ALIGNED: tl.constexpr,
        G_VEC: tl.constexpr,
        BLOCK: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = tl.load(block_tensor + pid).to(tl.int64)
        start = tl.load(block_start + pid)
        n = tl.load(numels + t)
        og = tl.load(table + t)
        if ALIGNED:
            start = tl.multiple_of(start, BLOCK)
            og = tl.multiple_of(og, G_VEC)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(ITERS):
            idx = start + i * BLOCK + tl.arange(0, BLOCK)
            g = tl.load(g_base + og + idx, mask=idx < n, other=0.0).to(tl.float32)
            acc += g * g
        tl.store(out + pid, tl.sum(acc, axis=0))

    _KERNELS = {"bf16": _adamw_bf16_kernel, "fp32": _adamw_kernel, "sumsq": _sum_of_squares_kernel}
    return _KERNELS


# ----------------------------------------------------------------------------------------------------------------
# Launch helpers
# ----------------------------------------------------------------------------------------------------------------


def _to_device(rows: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    host = torch.tensor(rows, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True)


def _on(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The clip factor as the 1-element fp32 tensor on `device` the kernels read."""
    return scale.to(device=device, dtype=torch.float32).reshape(1)


def _offsets(tensors: Sequence[torch.Tensor]) -> tuple[list[int], bool]:
    """Element offsets of `tensors` (one dtype) from the first, and whether every address is 16-byte aligned."""
    size = tensors[0].element_size()
    base = tensors[0].data_ptr()
    out, aligned = [], True
    for t in tensors:
        addr = t.data_ptr()
        aligned = aligned and addr % _ALIGN == 0
        offset, rest = divmod(addr - base, size)
        if rest:  # memory from outside torch's allocator, at an odd address: element offsets cannot reach it
            raise RuntimeError(
                "the multi-tensor optimizer kernels address tensors by element offsets from a base tensor, and a "
                f"{t.dtype} tensor here is not a whole number of elements away from it; use the per-parameter "
                "optimizer (train.fused_optimizer: false)"
            )
        out.append(offset)
    return out, aligned


class _Launcher:
    """Kernel launches with a fallback when the aligned (vectorized) variant does not compile on this Triton."""

    unaligned_only = False

    @classmethod
    def run(cls, kernel, grid: int, args: tuple, aligned: bool, device: torch.device, **meta):
        if cls.unaligned_only:
            aligned = False
        # Triton launches on torch's current CUDA device: make it the tensors' (a model spread over several GPUs)
        with torch.cuda.device(device):
            try:
                kernel[(grid,)](*args, ALIGNED=aligned, num_warps=NUM_WARPS, enable_fp_fusion=False, **meta)
            except Exception as e:  # noqa: BLE001 - only the alignment hints are retried; a second failure raises
                if not aligned:
                    raise
                # a failure here is raised before anything is launched (compilation, argument binding)
                kernel[(grid,)](*args, ALIGNED=False, num_warps=NUM_WARPS, enable_fp_fusion=False, **meta)
                # the unhinted variant ran: the hints were the problem (not, say, an out-of-memory error)
                cls.unaligned_only = True
                logger.warning(
                    f"multi-tensor optimizer kernel with alignment hints failed ({type(e).__name__}: {e}); "
                    "using unhinted loads (correct, possibly slower)"
                )


@dataclass
class Bf16Row:
    """One bf16 parameter's flat local tensors and per-step values for `adamw_bf16`."""

    p: torch.Tensor
    g: torch.Tensor
    m: torch.Tensor
    v: torch.Tensor
    c: torch.Tensor | None  # Kahan compensation (None: stochastic rounding)
    hp: list[float]  # [decay, beta1, beta2, step_size, 1 / sqrt(bias_correction2), eps]
    seed: int


@dataclass
class Fp32Row:
    """One parameter's flat local tensors and AdamW scalars for `adamw`."""

    p: torch.Tensor
    g: torch.Tensor
    m: torch.Tensor
    v: torch.Tensor
    hp: list[float]


def _buckets(rows: list, key) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def adamw_bf16(rows: list[Bf16Row], scale: torch.Tensor | None = None) -> int:
    """Step bf16 parameters (palingenesis.bf16_adamw's update; Kahan rows have `c`) on CUDA with one launch per
    (device, gradient dtype, rounding) bucket. `scale`: an fp32 device scalar multiplied into every gradient (the
    folded clip factor). Returns the number of kernel launches."""
    launches = 0
    kernel = _kernels()["bf16"]
    for (device, g_dtype, kahan), bucket in _buckets(
        [r for r in rows if r.p.numel()], lambda r: (r.p.device, r.g.dtype, r.c is not None)
    ).items():
        offs_p, al_p = _offsets([r.p for r in bucket])
        offs_g, al_g = _offsets([r.g for r in bucket])
        offs_m, al_m = _offsets([r.m for r in bucket])
        offs_v, al_v = _offsets([r.v for r in bucket])
        cs = [r.c for r in bucket] if kahan else [r.m for r in bucket]
        offs_c, al_c = _offsets(cs)
        table = _to_device(
            [[a, b, c, d, e, r.seed] for a, b, c, d, e, r in zip(offs_p, offs_g, offs_m, offs_v, offs_c, bucket)],
            torch.int64,
            device,
        )
        hp = _to_device([r.hp for r in bucket], torch.float32, device)
        m = _MAPS.get(device, tuple(r.p.numel() for r in bucket))
        first = bucket[0]
        _Launcher.run(
            kernel,
            m.blocks,
            (first.p, first.g, first.m, first.v, cs[0], table, hp, m.numels, m.tensor, m.start,
             _on(scale, device) if scale is not None else hp),
            al_p and al_g and al_m and al_v and al_c,
            device,
            KAHAN=kahan,
            HAS_SCALE=scale is not None,
            G_VEC=_ALIGN // first.g.element_size(),
            BLOCK=BLOCK,
            ITERS=ITERS,
        )  # fmt: skip
        launches += 1
    return launches


def adamw(rows: list[Fp32Row], scale: torch.Tensor | None = None) -> int:
    """Step parameters with torch.optim.AdamW's update on CUDA, one launch per (device, parameter dtype, gradient
    dtype) bucket. Returns the number of kernel launches."""
    launches = 0
    kernel = _kernels()["fp32"]
    for (device, p_dtype, g_dtype), bucket in _buckets(
        [r for r in rows if r.p.numel()], lambda r: (r.p.device, r.p.dtype, r.g.dtype)
    ).items():
        offs = [_offsets([getattr(r, a) for r in bucket]) for a in ("p", "g", "m", "v")]
        table = _to_device([list(cols) for cols in zip(*(o for o, _ in offs))], torch.int64, device)
        hp = _to_device([r.hp for r in bucket], torch.float32, device)
        m = _MAPS.get(device, tuple(r.p.numel() for r in bucket))
        first = bucket[0]
        _Launcher.run(
            kernel,
            m.blocks,
            (first.p, first.g, first.m, first.v, table, hp, m.numels, m.tensor, m.start,
             _on(scale, device) if scale is not None else hp),
            all(a for _, a in offs),
            device,
            HAS_SCALE=scale is not None,
            P_VEC=_ALIGN // first.p.element_size(),
            G_VEC=_ALIGN // first.g.element_size(),
            BLOCK=BLOCK,
            ITERS=ITERS,
        )  # fmt: skip
        launches += 1
    return launches


def sum_of_squares(grads: list[torch.Tensor], device: torch.device, triton: bool) -> torch.Tensor:
    """Sum of the squares of every element of `grads` (flat local tensors), accumulated in fp32, as a 0-d tensor on
    `device`. With Triton: one launch per gradient dtype and a sum of the per-program partials (deterministic: a
    fixed reduction order for given tensor sizes)."""
    grads = [g for g in grads if g.numel()]
    total = torch.zeros((), dtype=torch.float32, device=device)
    if not grads:
        return total
    if not triton:
        for g in grads:
            total += g.float().pow(2).sum().to(device)
        return total
    kernel = _kernels()["sumsq"]
    for (where, _dtype), bucket in _buckets(grads, lambda g: (g.device, g.dtype)).items():
        offs, aligned = _offsets(bucket)
        table = _to_device(offs, torch.int64, where)
        m = _MAPS.get(where, tuple(g.numel() for g in bucket))
        partial = torch.empty(m.blocks, dtype=torch.float32, device=where)
        _Launcher.run(
            kernel,
            m.blocks,
            (bucket[0], table, m.numels, m.tensor, m.start, partial),
            aligned,
            where,
            G_VEC=_ALIGN // bucket[0].element_size(),
            BLOCK=BLOCK,
            ITERS=ITERS,
        )
        total += partial.sum().to(device)
    return total


def clip_factor(norm: torch.Tensor, max_norm: float) -> torch.Tensor:
    """torch.nn.utils.clip_grad_norm_'s factor: min(1, max_norm / (norm + 1e-6)), as a 1-element fp32 tensor."""
    return torch.clamp(max_norm / (norm.float() + 1e-6), max=1.0).reshape(1)


# ----------------------------------------------------------------------------------------------------------------
# Reference math (torch ops): the CPU / no-Triton path, and what the kernels are checked against
# ----------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def adamw_reference(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, hp: Sequence[float]) -> None:
    """torch.optim.AdamW's update of one parameter in place (flat or not); g may be of another dtype."""
    decay, beta1, beta2, step_size, inv_bc2_sqrt, eps = hp
    g = g.to(m.dtype)
    p.mul_(decay)
    m.lerp_(g, 1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    p.addcdiv_(m, (v.sqrt() * inv_bc2_sqrt).add_(eps), value=-step_size)


# ----------------------------------------------------------------------------------------------------------------
# AdamW (fp32 weights) on the multi-tensor kernel
# ----------------------------------------------------------------------------------------------------------------


class FusedAdamW(torch.optim.Optimizer):
    """torch.optim.AdamW (decoupled weight decay, no amsgrad) stepped by one Triton launch per dtype bucket on CUDA,
    with gradient clipping that can be folded into the step (`clip_grad_norm_`). Parameters off CUDA, or every
    parameter when Triton is missing, take the same update with torch ops. State layout and state_dict are
    torch.optim.AdamW's (exp_avg, exp_avg_sq in the parameter's dtype, a float32 tensor `step`), so checkpoints load
    into either optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        self.triton = triton_available()[0]
        self.grad_scale: torch.Tensor | None = None  # the folded clip factor of the next step
        self.last_launches = 0

    @staticmethod
    def _scalars(group: dict, step: int) -> list[float]:
        lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        return [1.0 - lr * wd, beta1, beta2, lr / (1.0 - beta1**step), (1.0 - beta2**step) ** -0.5, eps]

    def _params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p, group

    def fused(self, p: torch.Tensor) -> bool:
        """Whether `p` is stepped by the kernel."""
        lp = local(p)
        return self.triton and lp.is_cuda and lp.dtype in (torch.float32, torch.bfloat16) and lp.is_contiguous()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """The global L2 norm of the gradients (fp32), and the clip factor min(1, max_norm / (norm + 1e-6)) kept for
        the next step(), which multiplies it into the gradient in the kernel (no pass over the gradients). One
        process only: the norm is not reduced across ranks. max_norm <= 0: the norm only."""
        return _clip(self, max_norm)

    def zero_grad(self, set_to_none: bool = True):
        self.grad_scale = None
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        scale, self.grad_scale = self.grad_scale, None
        rows, hp_cache = [], {}
        for p, group in self._params():
            if p.grad.is_sparse:
                raise RuntimeError("FusedAdamW does not support sparse gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            key = (id(group), state["step"])
            hp = hp_cache.get(key)
            if hp is None:
                hp = hp_cache[key] = self._scalars(group, state["step"])
            lp, lg, lm, lv = local(p), local(p.grad), local(state["exp_avg"]), local(state["exp_avg_sq"])
            if self.fused(p) and lg.is_contiguous() and lm.is_contiguous() and lv.is_contiguous():
                rows.append(Fp32Row(lp.view(-1), lg.view(-1), lm.view(-1), lv.view(-1), hp))
            else:
                g = lg if scale is None else lg.float() * scale.to(lg.device)
                adamw_reference(lp, g, lm, lv, hp)
        self.last_launches = adamw(rows, scale) if rows else 0
        return loss

    def state_dict(self):
        out = super().state_dict()
        # torch.optim.AdamW keeps `step` as a float32 tensor: emit that (new dicts: the live state keeps plain ints)
        out["state"] = {
            k: {**s, "step": torch.tensor(float(s["step"]), dtype=torch.float32)} if "step" in s else s
            for k, s in out["state"].items()
        }
        return out

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():  # plain int steps: the scalars are computed on the host
            if torch.is_tensor(state.get("step")):
                state["step"] = int(state["step"].item())


@torch.no_grad()
def _clip(optimizer, max_norm: float) -> torch.Tensor:
    """Global gradient norm of an optimizer's parameters, and its clip factor kept in `optimizer.grad_scale`."""
    grads = [local(p.grad).reshape(-1) for p, _ in optimizer._params()]
    if not grads:
        optimizer.grad_scale = None
        return torch.zeros(())
    device = grads[0].device
    triton = optimizer.triton and device.type == "cuda" and all(g.is_cuda and g.is_contiguous() for g in grads)
    norm = sum_of_squares(grads, device, triton).sqrt()
    optimizer.grad_scale = clip_factor(norm, max_norm) if max_norm > 0 else None
    return norm
