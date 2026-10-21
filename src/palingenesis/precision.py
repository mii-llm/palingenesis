"""Precision of small weight edits: stochastic rounding to bf16 / fp16, state-dict casts, edit-survival measurement and
simulated serving formats (fp8).

Why: merges and margin surgery apply edits with ||delta|| around 1e-3 to 1e-5 of ||W||. Public checkpoints are stored in
bf16, so their weights already sit on the bf16 grid (8 significant bits: one ulp is 2^-7 to 2^-8 of |w|), and
round-to-nearest of W + delta returns W wherever |delta_i| < ulp(w_i) / 2, i.e. almost everywhere: the edit vanishes
(1-6% survives in simulation). Stochastic rounding (SR) rounds up with probability equal to the remaining fraction of an
ulp, so E[SR(w + d)] = w + d exactly: each element moves by a whole ulp or not at all, and the edit survives in
expectation (and, summed over millions of elements, in every direction that matters: projection ~ 1). An edit made from
fp32 weights that are off the grid survives nearest rounding only on average (the weight's own position on the grid
acts as the dither), never when the source is bf16.

    stochastic_round(x, dtype, seed=)        unbiased SR from fp32 / fp64 to bf16 / fp16
    cast_state_dict(state, dtype, rounding)  a whole state dict, per-tensor seeds from (seed, name): order-independent
    edit_survival(before, intended, cast)    how much of the intended edit the saved weights still carry
    simulate_cast(x, fmt)                    nearest rounding to bf16 / fp16 / fp8 (per-tensor amax scale), as float32

Determinism: with `seed`, noise is drawn on the CPU from a seeded torch.Generator and moved to the tensor's device, so
the result is the same on CPU and CUDA (a `generator` passed explicitly draws on its own device instead). Large tensors
are processed in fixed-size chunks of the flattened tensor (int32 / float64 temporaries per chunk only), so the output
depends on the seed and the tensor, never on memory.
"""

from __future__ import annotations

import math
import zlib

import torch

CHUNK = 1 << 24  # elements per chunk: 64 MB of int32 noise, 128 MB of float64 temporaries
SR_DTYPES = (torch.bfloat16, torch.float16)
ROUNDINGS = ("nearest", "stochastic")
FP8 = {"fp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}


def tensor_seed(seed: int, name: str) -> int:
    """A per-tensor seed from (run seed, tensor name): a stable hash (crc32 of the name, chained on the crc32 of the
    seed), so casts do not depend on dict order or on which other tensors are present. 32 bits on purpose: torch's CPU
    generator (mt19937) uses only the low 32 bits of a seed, so a seed packed in the high bits would be ignored."""
    return zlib.crc32(name.encode(), zlib.crc32(str(int(seed)).encode()))


def _generator(generator: torch.Generator | None, seed: int | None) -> torch.Generator | None:
    if generator is not None:
        return generator
    if seed is None:
        return None  # torch's global CPU generator
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _gen_device(g: torch.Generator | None) -> torch.device:
    return g.device if g is not None else torch.device("cpu")


def _sr_bf16_from_fp32(x: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
    """bf16 is the top 16 bits of fp32. Adding uniform 16-bit noise to the raw bit pattern and truncating the low 16
    bits rounds the magnitude up with probability low16 / 2^16: unbiased. IEEE floats are sign-magnitude, so the same
    integer addition acts on the magnitude for negatives too (the sign bit is never reached from a finite value).
    A carry from the largest finite bf16 into the inf exponent is replaced by truncation (saturate at the top);
    inf / nan take torch's own cast (truncating a nan's low payload bits could turn it into inf)."""
    bits = x.view(torch.int32)
    finite = torch.isfinite(x)
    noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, generator=g, device=_gen_device(g)).to(x.device)
    noise = torch.where(finite, noise, torch.zeros_like(noise))  # no int32 overflow from nan bit patterns
    rounded = (bits + noise) & -65536
    overflow = finite & ((rounded & 0x7FFFFFFF) == 0x7F800000)
    rounded = torch.where(overflow, bits & -65536, rounded)
    out = (rounded >> 16).to(torch.int16).view(torch.bfloat16)
    return torch.where(finite, out, x.to(torch.bfloat16))


def _sr_neighbours(x: torch.Tensor, dtype: torch.dtype, g: torch.Generator | None) -> torch.Tensor:
    """SR between the two representable neighbours (any float input, bf16 / fp16 target): lo = the neighbour toward
    zero, hi = the next one away, P(hi) = (|x| - |lo|) / (|hi| - |lo|), all in float64. The nearest cast r is one of the
    two neighbours (also under double rounding, which is monotone); the other is one step of the 16-bit sign-magnitude
    pattern away (+1 = away from zero, -1 = toward zero; -0 + 1 is the smallest negative subnormal). Finite values beyond
    the largest finite are saturated to it (an edited weight never becomes inf); inf / nan are kept."""
    finite = torch.isfinite(x)
    big = torch.finfo(dtype).max
    xd = torch.where(finite, x.double().clamp(-big, big), torch.zeros((), dtype=torch.float64, device=x.device))
    r = xd.to(dtype)
    rd = r.double()
    below = rd.abs() <= xd.abs()  # r is the neighbour toward zero (or x is on the grid)
    rb = r.view(torch.int16)
    other = torch.where(below, rb + 1, rb - 1).view(dtype)
    lo = torch.where(below, r, other)
    hi = torch.where(below, other, r)
    lod, hid = lo.double().abs(), hi.double().abs()
    p = ((xd.abs() - lod) / (hid - lod)).nan_to_num(0.0)  # on the grid at the top: 0 / inf = 0
    u = torch.rand(x.shape, dtype=torch.float64, generator=g, device=_gen_device(g)).to(x.device)
    out = torch.where(u < p, hi, lo)
    return torch.where(finite, out, x.to(dtype))


def stochastic_round(
    x: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Unbiased stochastic rounding of a float tensor to bfloat16 or float16: each element becomes one of its two
    representable neighbours, the upper one with probability proportional to its distance from the lower, so
    E[result] = x and |result - x| < one ulp. Values on the target grid are returned exactly; inf and nan are kept;
    finite values never round to inf (saturated at the largest finite). An input already in `dtype` is returned
    unchanged.

    fp32 -> bf16 uses the bit trick (uniform noise added to the low 16 bits, then truncation); every other pair
    (fp32 / fp64 -> fp16, fp64 -> bf16) picks between the two neighbours in float64.

    seed: noise from a fresh CPU generator seeded with it (device-independent: same result on CPU and CUDA).
    generator: draw from this generator instead (on its device). Neither: torch's global CPU generator."""
    if dtype not in SR_DTYPES:
        raise ValueError(f"stochastic rounding targets bfloat16 or float16, not {dtype}")
    if not x.is_floating_point():
        raise ValueError(f"stochastic rounding needs a float tensor, not {x.dtype}")
    if x.dtype == dtype:
        return x
    g = _generator(generator, seed)
    flat = x.detach().reshape(-1)
    out = torch.empty(flat.shape, dtype=dtype, device=x.device)
    for i in range(0, flat.numel(), CHUNK):
        part = flat[i : i + CHUNK]
        if x.dtype == torch.float32 and dtype == torch.bfloat16:
            out[i : i + CHUNK] = _sr_bf16_from_fp32(part.contiguous(), g)
        else:
            out[i : i + CHUNK] = _sr_neighbours(part, dtype, g)
    return out.reshape(x.shape)


def cast_tensor(
    x: torch.Tensor, dtype: torch.dtype, rounding: str = "nearest", seed: int = 0, name: str = ""
) -> torch.Tensor:
    """One float tensor to `dtype`: stochastic rounding (seeded by tensor_seed(seed, name)) when rounding="stochastic"
    and the target is bf16 / fp16, else torch's round-to-nearest (exact when widening, e.g. bf16 -> fp32)."""
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}, not {rounding!r}")
    if x.dtype == dtype:
        return x
    if rounding == "stochastic" and dtype in SR_DTYPES:
        return stochastic_round(x, dtype, seed=tensor_seed(seed, name))
    return x.to(dtype)


def cast_state_dict(
    state: dict[str, torch.Tensor],
    dtype: torch.dtype,
    rounding: str = "nearest",
    seed: int = 0,
    *,
    only_float: bool = True,
) -> dict[str, torch.Tensor]:
    """A copy of `state` with its float tensors cast to `dtype` (rounding "nearest" or "stochastic"; see cast_tensor).
    Each tensor's noise is seeded from (seed, name), so the result does not depend on dict order. Non-float tensors
    pass through untouched (only_float=False casts them too). Tensors that alias each other (tied embeddings: same
    storage, shape and strides) are cast once, seeded by the smallest of their names, and the output keeps them as one
    tensor, so tied weights stay tied and save_pretrained still writes them once."""
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}, not {rounding!r}")

    def key(t: torch.Tensor):
        if t.device.type == "meta":
            return id(t)
        return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.stride(), t.dtype, t.device)

    groups: dict = {}
    for name, t in state.items():
        groups.setdefault(key(t), []).append(name)
    canonical = {name: min(names) for names in groups.values() for name in names}
    done: dict[str, torch.Tensor] = {}
    out = {}
    for name, t in state.items():
        if not t.is_floating_point() and only_float:
            out[name] = t
            continue
        c = canonical[name]
        if c not in done:
            t = t.detach()
            done[c] = cast_tensor(t, dtype, rounding, seed, c) if t.is_floating_point() else t.to(dtype)
        out[name] = done[c]
    return out


class SurvivalMeter:
    """Streaming edit_survival: update() one tensor at a time (float64 accumulation, chunked), result() at the end."""

    def __init__(self):
        self.dd = 0.0  # ||delta||^2
        self.cc = 0.0  # ||cast - before||^2
        self.cd = 0.0  # <cast - before, delta>
        self.changed = 0
        self.n = 0

    def update(self, before: torch.Tensor, after_intended: torch.Tensor, after_cast: torch.Tensor) -> None:
        if not (before.shape == after_intended.shape == after_cast.shape):
            raise ValueError(f"shapes differ: {before.shape}, {after_intended.shape}, {after_cast.shape}")
        b, a, c = (t.detach().reshape(-1) for t in (before, after_intended, after_cast))
        for i in range(0, b.numel(), CHUNK):
            bd = b[i : i + CHUNK].double()
            d = a[i : i + CHUNK].double() - bd
            e = c[i : i + CHUNK].double() - bd
            self.dd += float((d * d).sum())
            self.cc += float((e * e).sum())
            self.cd += float((e * d).sum())
            self.changed += int((e != 0).sum())
        self.n += b.numel()

    def result(self) -> dict[str, float]:
        nd = math.sqrt(self.dd)
        return {
            "norm_ratio": math.sqrt(self.cc) / nd if nd > 0 else float("nan"),
            "projection": self.cd / self.dd if self.dd > 0 else float("nan"),
            "changed_frac": self.changed / self.n if self.n else float("nan"),
        }


def edit_survival(
    before: torch.Tensor | dict,
    after_intended: torch.Tensor | dict,
    after_cast: torch.Tensor | dict,
) -> dict[str, float]:
    """How much of an intended edit delta = after_intended - before the saved weights still carry.

    before: the stored source weights (in their stored dtype); after_intended: source + delta (fp32); after_cast: what
    will be saved. All math in float64. Returns
        norm_ratio    ||cast - before|| / ||delta||        (1 = as large as intended; SR is >= 1: whole-ulp jumps)
        projection    <cast - before, delta> / ||delta||^2  (1 = the edit survives in direction and size; ~0 = lost)
        changed_frac  share of elements that differ from before
    Single tensors or dicts with the same keys (summed over keys, streamed one tensor at a time); nan when delta = 0."""
    meter = SurvivalMeter()
    if isinstance(after_cast, dict):
        if not (isinstance(before, dict) and isinstance(after_intended, dict)):
            raise TypeError("pass three tensors or three dicts")
        if not (set(before) == set(after_intended) == set(after_cast)):
            raise ValueError("before, after_intended and after_cast must have the same keys")
        for k in after_cast:
            if after_cast[k].is_floating_point():
                meter.update(before[k], after_intended[k], after_cast[k])
    else:
        meter.update(before, after_intended, after_cast)
    return meter.result()


def simulate_cast(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """x rounded to a serving format (round-to-nearest) and returned as float32, to check whether an edit survives
    serving: compare the served change with the edit,
        q = simulate_cast(W, fmt); edit_survival(q, q + delta, simulate_cast(W + delta, fmt))
    (not edit_survival(q, W + delta, ...), which would count W's own quantization error as part of the edit).

    fmt: "bf16", "fp16" (plain casts; fp16 overflows to inf like a real cast), "fp8_e4m3", "fp8_e5m2" (torch's float8
    dtypes with per-tensor amax scaling as in common fp8 serving: scale = amax / max_representable, x / scale cast to
    fp8 and multiplied back). The scale is recomputed from x, so an edit that moves the tensor's amax shifts the whole
    grid, as it would when the edited checkpoint is quantized. Not simulated: GPTQ / AWQ int4 (and other calibrated or
    group-wise schemes), whose rounding depends on calibration data and error feedback."""
    if fmt in ("bf16", "fp16"):
        return x.detach().to(torch.bfloat16 if fmt == "bf16" else torch.float16).float()
    if fmt not in FP8:
        raise ValueError(f"unknown format {fmt!r}; one of bf16, fp16, {', '.join(FP8)}")
    f8 = getattr(torch, FP8[fmt], None)
    if f8 is None:
        raise RuntimeError(f"this torch ({torch.__version__}) has no {FP8[fmt]}")
    fmax = torch.finfo(f8).max
    xf = x.detach().float()
    amax = float(xf.abs().max()) if xf.numel() else 0.0
    scale = amax / fmax if amax > 0 and math.isfinite(amax) else 1.0
    return (xf / scale).clamp(-fmax, fmax).to(f8).float() * scale
