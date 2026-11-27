"""Components of ternary quantisation: scale schemes, quantisers and straight-through estimators.

A ternary weight is `alpha * code` with `code` in {-1, 0, 1} and one `alpha` per group of weights (a whole tensor,
an output row, or `group_size` consecutive input weights of a row). Three choices define it, each a registry here
and an argument of `palingenesis.qat:TernaryQAT`:

    scale       how a group's magnitude `alpha` and dead-zone half-width `delta` follow from its weights
                  absmean   alpha = mean |w|, delta = alpha / 2                 (BitNet b1.58, arXiv:2402.17764)
                  absmax    alpha = ratio * max |w| (ratio 2/3), delta = alpha / 2   (three equal bins of [-max, max]:
                            the starting point of ParetoQ's stretched elastic quantiser, arXiv:2502.02631)
                  twn       delta = 0.7 * mean |w|, alpha = mean of the |w| above delta   (TWN, arXiv:1605.04711)
                  mse       the alpha, delta that minimise the group's squared error (exact, by sorting)
    quantizer   how codes follow from w, alpha and delta
                  round       sign(w) where |w| > delta, else 0
                  stochastic  training only: |code| = 1 with probability clip(|w| / alpha, 0, 1) (unbiased inside
                              [-alpha, alpha]); evaluation and export use `round`
    ste         the gradient the latent weight receives through the rounding (the forward is always alpha * code)
                  identity    d W_q / d W = 1                                    (Bengio et al., arXiv:1308.3432)
                  clipped     1 where |w| <= clip * alpha, else 0                (hard-tanh window)
                  tanh        the slope of a smooth staircase through the same levels, sharpened over training
                              (large near the thresholds, small in the flat parts)

`scale_mode` (in TernaryQAT) says when the scheme is evaluated: every forward from the current latent weights
("dynamic"), once before the first step ("frozen"), or once and then trained with LSQ gradients ("learned",
arXiv:1902.08153).

All functions take weights in group layout `[rows, n_groups, group]` and return alpha, delta as `[rows, n_groups, 1]`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

EPS = 1e-8


def group_view(w: Tensor, group_size: int) -> Tensor:
    """w [rows, cols] as [rows, n_groups, g] (group_size > 0), [rows, 1, cols] (-1: per row),
    or [1, 1, rows * cols] (0: per tensor)."""
    rows, cols = w.shape
    if group_size == 0:
        return w.reshape(1, 1, -1)
    if group_size < 0:
        return w.reshape(rows, 1, cols)
    return w.reshape(rows, cols // group_size, group_size)


# ─────────────────────────────────────────────────────────────── scale schemes


def scale_absmean(g: Tensor) -> tuple[Tensor, Tensor]:
    alpha = g.abs().mean(-1, keepdim=True).clamp(min=EPS)
    return alpha, 0.5 * alpha


def scale_absmax(g: Tensor, ratio: float = 2.0 / 3.0) -> tuple[Tensor, Tensor]:
    alpha = (ratio * g.abs().amax(-1, keepdim=True)).clamp(min=EPS)
    return alpha, 0.5 * alpha


def scale_twn(g: Tensor, factor: float = 0.7) -> tuple[Tensor, Tensor]:
    mag = g.abs()
    delta = factor * mag.mean(-1, keepdim=True)
    kept = mag > delta
    alpha = (mag * kept).sum(-1, keepdim=True) / kept.sum(-1, keepdim=True).clamp(min=1)
    return alpha.clamp(min=EPS), delta


def scale_mse(g: Tensor) -> tuple[Tensor, Tensor]:
    """The exact minimiser of sum (w - alpha * code)^2 over alpha > 0 and ternary codes: with the |w| sorted in
    decreasing order and c_k their running sum, keep the k largest for the k that maximises c_k^2 / k, and
    alpha = c_k / k. The kept set is then exactly {|w| > alpha / 2}."""
    mag = g.abs().float().sort(dim=-1, descending=True).values
    csum = mag.cumsum(-1)
    k = torch.arange(1, mag.shape[-1] + 1, device=g.device, dtype=csum.dtype)
    best = (csum.pow(2) / k).argmax(-1, keepdim=True)
    alpha = (csum.gather(-1, best) / (best + 1)).clamp(min=EPS).to(g.dtype)
    return alpha, 0.5 * alpha


SCALES: dict[str, Callable[..., tuple[Tensor, Tensor]]] = {
    "absmean": scale_absmean,
    "absmax": scale_absmax,
    "twn": scale_twn,
    "mse": scale_mse,
}


# ─────────────────────────────────────────────────────────────── quantisers


def codes_round(g: Tensor, alpha: Tensor, delta: Tensor) -> Tensor:
    return torch.sign(g) * (g.abs() > delta)


def codes_stochastic(g: Tensor, alpha: Tensor, delta: Tensor) -> Tensor:
    p = (g.abs() / alpha).clamp(0, 1)
    return torch.sign(g) * (torch.rand_like(p) < p)


QUANTIZERS: dict[str, Callable[[Tensor, Tensor, Tensor], Tensor]] = {
    "round": codes_round,
    "stochastic": codes_stochastic,
}


# ─────────────────────────────────────────────────────────────── straight-through estimators


def ste_identity(g: Tensor, alpha: Tensor, delta: Tensor, **_) -> Tensor:
    return g


def ste_clipped(g: Tensor, alpha: Tensor, delta: Tensor, clip: float = 1.0, **_) -> Tensor:
    bound = clip * alpha
    return torch.maximum(torch.minimum(g, bound), -bound)


def ste_tanh(g: Tensor, alpha: Tensor, delta: Tensor, sharpness: float = 4.0, **_) -> Tensor:
    """alpha * [tanh(k (u - d)) + tanh(k (u + d))] / 2 with u = w / alpha, d = delta / alpha: a smooth staircase
    with steps at +-delta that tends to alpha * code as k grows."""
    u, d = g / alpha, delta / alpha
    return 0.5 * alpha * (torch.tanh(sharpness * (u - d)) + torch.tanh(sharpness * (u + d)))


STES: dict[str, Callable[..., Tensor]] = {
    "identity": ste_identity,
    "clipped": ste_clipped,
    "tanh": ste_tanh,
}


# ─────────────────────────────────────────────────────────────── the quantiser of one tensor


class TernarySpec:
    """One tensor's ternary quantiser: granularity, scale scheme, code rule and gradient estimator."""

    def __init__(
        self,
        group_size: int = 128,
        scale: str = "absmean",
        quantizer: str = "round",
        ste: str = "identity",
        scale_args: dict | None = None,
        ste_args: dict | None = None,
    ):
        for kind, name, table in (("scale", scale, SCALES), ("quantizer", quantizer, QUANTIZERS), ("ste", ste, STES)):
            if name not in table:
                raise ValueError(f"{kind} must be one of {sorted(table)} (got {name!r})")
        self.group_size = group_size
        self.scale, self.quantizer, self.ste = scale, quantizer, ste
        self.scale_args = dict(scale_args or {})
        self.ste_args = dict(ste_args or {})

    def describe(self) -> dict:
        return {
            "group_size": self.group_size,
            "scale": self.scale,
            "quantizer": self.quantizer,
            "ste": self.ste,
            "scale_args": self.scale_args,
            "ste_args": self.ste_args,
        }

    def scales(self, g: Tensor) -> tuple[Tensor, Tensor]:
        """(alpha, delta) of weights in group layout, without gradient."""
        with torch.no_grad():
            return SCALES[self.scale](g.detach(), **self.scale_args)

    def codes(self, g: Tensor, alpha: Tensor, delta: Tensor, sample: bool = False) -> Tensor:
        rule = self.quantizer if sample else "round"
        with torch.no_grad():
            return QUANTIZERS[rule](g.detach(), alpha, delta)

    def quantize(self, w: Tensor, fixed: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """alpha * code in w's shape (deterministic, no gradient). `fixed`: (alpha, delta) to use instead of the
        scheme's."""
        g = group_view(w.detach(), self.group_size)
        alpha, delta = fixed if fixed is not None else self.scales(g)
        return (alpha * self.codes(g, alpha, delta)).reshape(w.shape).to(w.dtype)

    def effective(
        self,
        w: Tensor,
        strength: Tensor | float = 1.0,
        *,
        fixed: tuple[Tensor, Tensor] | None = None,
        log_scale: Tensor | None = None,
        sample: bool = False,
        sharpness: float | None = None,
    ) -> Tensor:
        """The weight the forward computes with: w + strength * (W_q - w), W_q = alpha * code in value, with the
        gradient of the chosen estimator. `log_scale` (learned scales): alpha and delta are multiplied by
        exp(log_scale), which receives the LSQ gradient code - w / alpha."""
        g = group_view(w, self.group_size)
        alpha, delta = fixed if fixed is not None else self.scales(g)
        if log_scale is not None:
            factor = log_scale.exp().to(g.dtype)
            alpha, delta = alpha * factor, delta * factor.detach()
        a = alpha.detach()
        codes = self.codes(g, a, delta, sample)
        args = dict(self.ste_args)
        if sharpness is not None:
            args["sharpness"] = sharpness
        surrogate = STES[self.ste](g, a, delta, **args)
        if log_scale is None:
            w_q = surrogate + (a * codes - surrogate).detach()
        else:
            # alpha * (u + (code - u).detach()), u = surrogate / alpha: d / d alpha = code - u (LSQ)
            u = surrogate / alpha
            w_q = alpha * (u + (codes - u).detach())
        w_q = w_q.reshape(w.shape)
        if isinstance(strength, Tensor):
            strength = strength.to(w.dtype)
        return w + strength * (w_q - w) if not _is_one(strength) else w_q


def _is_one(strength) -> bool:
    return not isinstance(strength, Tensor) and strength == 1.0


def lsq_grad_scale(n_weights: int) -> float:
    """LSQ's gradient scale for a step size shared by n weights with ternary codes (Qp = 1): 1 / sqrt(n)."""
    return 1.0 / math.sqrt(max(n_weights, 1))


# ─────────────────────────────────────────────────────────────── reading ternary tensors back


def decode_ternary(w: Tensor, group_size: int, rtol: float = 1e-3) -> tuple[Tensor, Tensor] | None:
    """(codes int8, alpha) in group layout if every group of `w` holds only 0 and one magnitude, else None.
    alpha of an all-zero group is 0."""
    g = group_view(w.detach().float(), group_size)
    mag = g.abs()
    alpha = mag.amax(-1, keepdim=True)
    nonzero = mag > 0
    if not bool(((mag - alpha).abs() <= rtol * alpha)[nonzero].all()):
        return None
    return torch.sign(g).to(torch.int8), alpha


def is_ternary(w: Tensor, group_size: int, rtol: float = 1e-3) -> bool:
    """Whether every group of `w` holds at most the three values -alpha, 0, alpha."""
    return decode_ternary(w, group_size, rtol) is not None
