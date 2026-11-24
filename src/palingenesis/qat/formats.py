"""Low-precision floating-point formats with block scales: NVFP4, MXFP4, FP8 (per row and 128 x 128 blocks).

Pure PyTorch, no kernels: the same functions compute the fake-quantised weights of quantisation-aware training
(palingenesis.qat.fp), the PTQ baseline and the packed checkpoints vLLM loads (palingenesis.qat.compressed), so the
three agree by construction.

Element formats
  E2M1 (FP4)   values +-{0, 0.5, 1, 1.5, 2, 3, 4, 6}, no inf / NaN (OCP MX v1.0, section 5.3.3). Rounding is
               round-to-nearest, ties to even; magnitudes above 6 saturate to 6.
  E4M3 (FP8)   the "fn" variant: max 448, no inf, NaN = S.1111.111; subnormals down to 2^-9. Conversion clamps
               to +-448 first (saturating, as every FP8 quantiser does), then rounds to nearest even (torch's cast).

Scale formats
  E8M0         a power of two 2^e, e in [-127, 127], stored as e + 127 in a uint8 (255 = NaN). MX formats.
  E4M3         NVFP4's block scale, with an fp32 per-tensor scale on top.
  fp32         FP8's per-row or per-block scale.

Formats (FORMATS)
  nvfp4      E2M1 values, one E4M3 scale per 16 consecutive inputs, one fp32 scale per tensor (NVIDIA NVFP4;
             compressed-tensors "nvfp4-pack-quantized"). Stored convention of compressed-tensors / vLLM:
               global_scale = 448 * 6 / amax(tensor)                       (a divisor, fp32)
               block_scale  = e4m3(clamp(amax(block) / 6 * global_scale, 2^-6, 448))
               W ~= q * (block_scale / global_scale),  q = e2m1(W / (block_scale / global_scale))
             The lower clamp 2^-6 is torchao's (E4M3's smallest normal): vLLM's Marlin kernel (the weight-only
             path on GPUs without FP4 tensor cores) zeroes smaller block scales, and it keeps a block of
             near-zero weights from dividing by zero.
  mxfp4      E2M1 values, one E8M0 scale per 32 inputs (OCP MX v1.0; compressed-tensors "mxfp4-pack-quantized").
             The exponent from the block's amax (`mx_rounding`):
               floor  e = floor(log2(amax)) - 2              the OCP reference (the block max may saturate)
               even   amax's significand rounded up from 1.75 first (torchao EVEN; compressed-tensors' rule)
               ceil   e = ceil(log2(amax / 6))               no element ever saturates (torchao RCEIL)
  fp8        E4M3 values, one fp32 scale per output row: amax(row) / 448 (compressed-tensors FP8 "channel").
  fp8_block  E4M3 values, one fp32 scale per 128 x 128 tile (DeepSeek-V3 style; compressed-tensors "block").

Activations use the same element formats with dynamic scales per token (per row of the flattened input):
fp8 per token, fp8_group per token and 128 inputs, nvfp4 / mxfp4 per token and block.

Non-finite inputs propagate: a NaN or inf weight gives NaN outputs (MXFP4: its whole block; NVFP4: the whole
tensor, through the per-tensor scale; FP8: the element), so a corrupted weight shows up as a NaN loss instead of
being quantised away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_SCALE_MIN = 2.0**-6  # NVFP4 block scales: E4M3's smallest normal (see the module docstring)
E8M0_BIAS = 127
E8M0_NAN = 255
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # code i (bits 0bEEM) holds E2M1_VALUES[i]; bit 3 = sign
MX_ROUNDING = ("floor", "even", "ceil")


@dataclass(frozen=True)
class FormatSpec:
    """One quantisation format: element type, scale type and the block a scale covers.

    block_rows x block_cols elements share a scale; block_cols 0 means the whole row (per-row / per-token).
    """

    name: str
    element: str  # "e2m1" | "e4m3"
    scale: str  # "e4m3" (NVFP4, with a per-tensor fp32 scale) | "e8m0" | "fp32"
    block_cols: int
    block_rows: int = 1

    @property
    def element_max(self) -> float:
        return E2M1_MAX if self.element == "e2m1" else E4M3_MAX

    @property
    def saturation_bound(self) -> float:
        """Scaled magnitudes beyond this are clipped: past the midpoint between the max and the next value the
        format would have (8 for E2M1, 480 for E4M3); up to it, rounding to the max is ordinary rounding."""
        return 7.0 if self.element == "e2m1" else 464.0

    @property
    def has_global_scale(self) -> bool:
        return self.scale == "e4m3"

    def block_shape(self, rows: int, cols: int) -> tuple[int, int]:
        return self.block_rows, (self.block_cols or cols)

    def check_shape(self, rows: int, cols: int) -> str | None:
        """Why a [rows, cols] matrix cannot use this format, or None."""
        br, bc = self.block_shape(rows, cols)
        if cols % bc:
            return f"{cols} columns are not a multiple of the {bc}-element block"
        if rows % br:
            return f"{rows} rows are not a multiple of the {br}-row block"
        return None


FORMATS: dict[str, FormatSpec] = {
    "nvfp4": FormatSpec("nvfp4", "e2m1", "e4m3", 16),
    "mxfp4": FormatSpec("mxfp4", "e2m1", "e8m0", 32),
    "fp8": FormatSpec("fp8", "e4m3", "fp32", 0),
    "fp8_block": FormatSpec("fp8_block", "e4m3", "fp32", 128, 128),
}
ACTIVATION_FORMATS: dict[str, FormatSpec] = {
    "fp8": FormatSpec("fp8", "e4m3", "fp32", 0),  # per token
    "fp8_group": FormatSpec("fp8_group", "e4m3", "fp32", 128),  # per token and 128 inputs (with fp8_block weights)
    "nvfp4": FormatSpec("nvfp4", "e2m1", "e4m3", 16),
    "mxfp4": FormatSpec("mxfp4", "e2m1", "e8m0", 32),
}


# ─────────────────────────────────────────────────────────────── element and scale rounding


def round_e2m1(x: Tensor) -> Tensor:
    """x (float32) rounded to the nearest E2M1 value, ties to even, saturating at +-6; NaN stays NaN.

    The grid is uniform within each binade: step 0.5 below 2, 1 in [2, 4), 2 in [4, 6]. Rounding x / step
    to the nearest even integer is round-to-nearest-even on the grid (an even multiple has mantissa bit 0)."""
    a = x.abs().clamp(max=E2M1_MAX)
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.copysign(torch.round(a / step) * step, x)


def round_e4m3(x: Tensor) -> Tensor:
    """x (float32) rounded to the nearest E4M3 (fn) value, ties to even, saturating at +-448, as float32."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def round_element(x: Tensor, element: str) -> Tensor:
    return round_e2m1(x) if element == "e2m1" else round_e4m3(x)


def e8m0_exponent(amax: Tensor, rounding: str = "floor", element_emax: int = 2) -> Tensor:
    """The unbiased E8M0 exponent (int32) of blocks whose largest magnitude is `amax` (float32, >= 0).

    In [-127, 127]; 128 marks a non-finite amax (the stored scale is NaN). An all-zero block gets -127.
    element_emax: the element format's largest exponent (2 for E2M1, whose max is 1.5 * 2^2)."""
    if rounding not in MX_ROUNDING:
        raise ValueError(f"mx_rounding must be one of {MX_ROUNDING}, got {rounding!r}")
    amax = amax.to(torch.float32)
    finite = torch.isfinite(amax)
    if rounding == "ceil":  # the smallest power of two >= amax / element max: nothing saturates
        bits = (amax / E2M1_MAX).view(torch.int32)
        e = ((bits >> 23) & 0xFF) - E8M0_BIAS + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    else:
        bits = amax.view(torch.int32)
        if rounding == "even":  # carry into the exponent from significand 1.75 (E2M1's mantissa: 1 bit)
            bits = bits + (1 << 21)
        e = ((bits >> 23) & 0xFF) - E8M0_BIAS - element_emax
    e = e.clamp(-E8M0_BIAS, E8M0_BIAS)
    return torch.where(finite, e, torch.full_like(e, E8M0_NAN - E8M0_BIAS))


def e8m0_to_float(exponent: Tensor) -> Tensor:
    """2^exponent as float32, exactly (2^-127 is an fp32 subnormal); exponent 128 gives NaN."""
    e = exponent.to(torch.int32)
    bits = torch.where(e >= -126, (e + 127) << 23, torch.full_like(e, 1 << 22))
    bits = torch.where(e > 127, torch.full_like(e, 0x7FC00000), bits)
    return bits.view(torch.float32)


# ─────────────────────────────────────────────────────────────── block quantisation


@dataclass
class Quantized:
    """A matrix quantised in one format.

    values:        the element values (float32 on the element grid), shape [rows, cols]
    scale:         the stored block scale, decoded to float32 (E4M3 / E8M0 values exactly), [rows / br, cols / bc]
    global_scale:  NVFP4's per-tensor divisor 448 * 6 / amax (float32, shape [] or [rows, 1] per token), else None
    saturated:     per block, whether some element was clipped (beyond FormatSpec.saturation_bound), [rows/br, cols/bc]
    """

    spec: FormatSpec
    values: Tensor
    scale: Tensor
    global_scale: Tensor | None
    saturated: Tensor

    def effective_scale(self) -> Tensor:
        """What a block's values are multiplied by: scale / global_scale (NVFP4), else scale."""
        if self.global_scale is None:
            return self.scale
        g = self.global_scale
        return self.scale / (g if g.dim() == 0 else g.reshape(-1, 1))

    def dequantize(self) -> Tensor:
        rows, cols = self.values.shape
        br, bc = self.spec.block_shape(rows, cols)
        v = self.values.reshape(rows // br, br, cols // bc, bc)
        s = self.effective_scale().reshape(rows // br, 1, cols // bc, 1)
        return (v * s).reshape(rows, cols)


def _blocks(x: Tensor, spec: FormatSpec) -> Tensor:
    rows, cols = x.shape
    problem = spec.check_shape(rows, cols)
    if problem:
        raise ValueError(f"{spec.name}: a [{rows}, {cols}] matrix: {problem}")
    br, bc = spec.block_shape(rows, cols)
    return x.reshape(rows // br, br, cols // bc, bc)


def global_scale_from_amax(amax: Tensor) -> Tensor:
    """NVFP4's per-tensor divisor 448 * 6 / amax (float32). An all-zero tensor gets 1 (every value is 0 whatever
    the scale), a tiny amax the largest finite float32 (no overflow to inf); a non-finite amax gives NaN."""
    amax = amax.to(torch.float32)
    out = ((E4M3_MAX * E2M1_MAX) / amax).clamp(max=torch.finfo(torch.float32).max)
    out = torch.where(amax > 0, out, torch.ones_like(out))
    return torch.where(torch.isfinite(amax), out, torch.full_like(out, float("nan")))


def compute_scale(
    amax: Tensor, spec: FormatSpec, global_scale: Tensor | None = None, mx_rounding: str = "even"
) -> Tensor:
    """The stored block scale (decoded to float32) for blocks of largest magnitude `amax` (float32).

    nvfp4: e4m3(clamp(amax / 6 * global_scale, 2^-6, 448)); mxfp4: 2^e8m0_exponent(amax); fp8: amax / 448
    (1 for an all-zero row or block, whose values are 0 whatever the scale)."""
    if spec.scale == "e4m3":
        if global_scale is None:
            raise ValueError("nvfp4 needs the per-tensor global scale")
        raw = amax / E2M1_MAX * global_scale
        return round_e4m3(raw.clamp(min=E4M3_SCALE_MIN)).where(torch.isfinite(raw), raw)
    if spec.scale == "e8m0":
        return e8m0_to_float(e8m0_exponent(amax, mx_rounding))
    s = amax / E4M3_MAX
    return torch.where((s > 0) | torch.isnan(s), s, torch.ones_like(s))


def quantize(
    x: Tensor,
    spec: FormatSpec,
    *,
    clip: Tensor | float | None = None,
    global_amax: Tensor | None = None,
    scale: Tensor | None = None,
    global_scale: Tensor | None = None,
    mx_rounding: str = "even",
    track_saturation: bool = True,
) -> Quantized:
    """Quantise the matrix x [rows, cols] (computed in float32).

    clip:         a factor on each block's amax before the scale is computed (per row block: [rows / br, 1],
                  or a number); < 1 clips the largest values for a finer grid (MSE-optimal / learned clipping)
    global_amax:  NVFP4: the amax the per-tensor scale is computed from (default: x's own; a fused layer's
                  shards share the largest; a [rows, 1] tensor gives per-row (per-token) global scales)
    scale, global_scale:  frozen stored scales (as Quantized holds them): used as given, nothing is recomputed
    track_saturation:     False skips the `saturated` statistic (an empty tensor; the training forward's path)
    """
    xf = x.to(torch.float32)
    blocks = _blocks(xf, spec)
    rows, cols = xf.shape
    gs = None
    if spec.has_global_scale:
        if global_scale is not None:
            gs = global_scale.to(torch.float32)
        else:
            if global_amax is None:
                global_amax = xf.abs().amax()
            gs = global_scale_from_amax(global_amax)
    if scale is None:
        amax = blocks.abs().amax(dim=(1, 3))
        if clip is not None:
            amax = amax * (clip if isinstance(clip, (int, float)) else clip.to(torch.float32).reshape(-1, 1))
        g = None if gs is None else (gs if gs.dim() == 0 else gs.reshape(-1, 1))
        scale = compute_scale(amax, spec, g, mx_rounding)
    else:
        scale = scale.to(torch.float32)
    q = Quantized(spec, xf, scale, gs, scale)  # values filled below
    eff = q.effective_scale().reshape(blocks.shape[0], 1, blocks.shape[2], 1)
    scaled = blocks / eff
    if track_saturation:
        q.saturated = (scaled.abs() > spec.saturation_bound).any(dim=3).any(dim=1)
    else:
        q.saturated = torch.zeros(0, dtype=torch.bool, device=x.device)
    q.values = round_element(scaled, spec.element).reshape(rows, cols)
    return q


def fake_quantize(x: Tensor, spec: FormatSpec, **kwargs) -> Tensor:
    """x quantised and dequantised (float32), no gradient path (see straight_through)."""
    with torch.no_grad():
        return quantize(x, spec, **kwargs).dequantize()


def lsq_fake_quantize(
    x: Tensor, spec: FormatSpec, clip: Tensor, global_amax: Tensor | None = None, mx_rounding: str = "even"
) -> Tensor:
    """Fake quantisation with learned clipping (LSQ-style gradients, Esser et al. 2020, arXiv:1902.08153).

    The forward equals quantize(x, clip=clip).dequantize(). Backward: the rounding of values and of the stored
    scale pass gradients straight through; elements clipped at the element max get no gradient (their output
    is +-max * scale, whose gradient goes to the scale); `clip` [rows / br, 1] receives the scale's gradient."""
    xf = x.to(torch.float32)
    blocks = _blocks(xf, spec)
    rows, cols = xf.shape
    amax = blocks.detach().abs().amax(dim=(1, 3))
    gs = None
    if spec.has_global_scale:
        gs = global_scale_from_amax(xf.detach().abs().amax() if global_amax is None else global_amax)
    c = clip.to(torch.float32).reshape(-1, 1)
    stored = compute_scale(amax * c.detach(), spec, gs, mx_rounding)
    # the stored scale in the forward; in the backward d scale / d clip = scale / clip (the scale is
    # proportional to the clipped amax up to its rounding, which the estimator passes straight through)
    scale = stored * (c / c.detach())
    eff = scale / gs if gs is not None else scale
    eff = eff.reshape(blocks.shape[0], 1, blocks.shape[2], 1)
    scaled = (blocks / eff).clamp(-spec.element_max, spec.element_max)
    q = scaled + (round_element(scaled.detach(), spec.element) - scaled).detach()
    return (q * eff).reshape(rows, cols)


def straight_through(x: Tensor, x_q: Tensor, strength: Tensor | float = 1.0) -> Tensor:
    """strength * x_q + (1 - strength) * x with the gradient of x (straight-through estimator), in x's dtype.

    Written so that strength 1 gives exactly x_q (x + (x_q - x) would round twice in bf16, and the training
    forward would then differ from the exported weights by an ulp) and strength 0 exactly x."""
    xd = x.detach()
    return (strength * x_q.to(x.dtype) + (1 - strength) * xd) + (x - xd)


def quantize_activation(
    x: Tensor,
    spec: FormatSpec,
    global_amax: Tensor | None = None,
    mx_rounding: str = "even",
) -> Tensor:
    """Dynamic fake quantisation of activations x [..., k] (per token = per row of the flattened input), as
    float32, no gradient path. NVFP4: per-token global scales unless `global_amax` (a static, calibrated
    per-tensor amax) is given."""
    with torch.no_grad():
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        if spec.has_global_scale and global_amax is None:
            global_amax = flat.abs().amax(dim=1, keepdim=True)
        q = quantize(flat, spec, global_amax=global_amax, mx_rounding=mx_rounding, track_saturation=False)
        return q.dequantize().reshape(x.shape)


# ─────────────────────────────────────────────────────────────── packing (compressed-tensors layout)

_E2M1_CODE = torch.full((13,), -1, dtype=torch.uint8)  # index: 2 * |value| -> the 3-bit magnitude code
for _code, _value in enumerate(E2M1_VALUES):
    _E2M1_CODE[int(2 * _value)] = _code


def pack_e2m1(values: Tensor) -> Tensor:
    """E2M1 values [rows, cols] (float32 on the grid, cols even) as uint8 [rows, cols / 2]: code bits 0bSEEM,
    the even column in the low nibble (compressed-tensors' and ModelOpt's layout). -0.0 packs as 0x8."""
    if values.shape[-1] % 2:
        raise ValueError("E2M1 packing needs an even number of columns")
    mag = (values.abs() * 2).to(torch.int64)
    if not torch.equal(mag.to(values.dtype), values.abs() * 2) or bool((mag > 12).any()):
        raise ValueError("pack_e2m1: values are not on the E2M1 grid")
    codes = _E2M1_CODE.to(values.device)[mag]
    if bool((codes == 255).any()):
        raise ValueError("pack_e2m1: values are not on the E2M1 grid")
    codes = codes | (torch.signbit(values).to(torch.uint8) << 3)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(packed: Tensor) -> Tensor:
    """pack_e2m1's inverse: float32 values [rows, 2 * cols]."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low, high = packed & 0x0F, packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int64)
    mag = table[codes & 0x7]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def e8m0_encode(scale: Tensor) -> Tensor:
    """A power-of-two float32 scale (or NaN) as its E8M0 byte (uint8)."""
    bits = scale.to(torch.float32).view(torch.int32)
    field = (bits >> 23) & 0xFF  # 2^-127 is an fp32 subnormal: field 0, byte 0, as its biased exponent
    biased = torch.where(torch.isnan(scale), torch.full_like(field, E8M0_NAN), field)
    return biased.to(torch.uint8)


def e8m0_decode(byte: Tensor) -> Tensor:
    return e8m0_to_float(byte.to(torch.int32) - E8M0_BIAS)
