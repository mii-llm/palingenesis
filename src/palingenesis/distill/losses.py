"""Token-level divergences between a teacher p and a student q, in log space.

Every function takes log-probabilities over a common support [R, S] (the full vocabulary for an
online teacher, the teacher's top-k plus the gold token and a tail bucket for a cached one) and a
mask of the support entries that exist, and returns the per-row divergence [R]. Entries outside the
mask hold no mass on either side. All arithmetic is fp32; zero-probability entries contribute zero
(never 0 * inf).

The support constructions:

  dense_support      full vocabulary: teacher and student log-softmax over their shared ids
  sparse_support     a cached top-k: the teacher's top-k log-probs (+ the gold token's), the student's
                     at the same ids, and (topk_mode "tail") one more entry per side holding the rest
                     of its mass, so both remain distributions over the same partition of the
                     vocabulary; ("renorm") both renormalised over the support instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# Smallest probability a tail bucket may hold (log(0) would make reverse KL infinite).
TAIL_FLOOR = 1e-8
_NEG = -1e30  # "minus infinity" that stays finite in products


def _masked(lp: Tensor, mask: Tensor | None) -> Tensor:
    return lp if mask is None else lp.masked_fill(~mask, _NEG)


def _xlogy_terms(lp_a: Tensor, lp_b: Tensor) -> Tensor:
    """exp(a) * (a - b) per entry, exactly 0 where exp(a) underflows."""
    p = lp_a.exp()
    return torch.where(p > 0, p * (lp_a - lp_b), torch.zeros_like(p))


def kl(lp: Tensor, lq: Tensor, mask: Tensor | None = None) -> Tensor:
    """KL(p || q) per row."""
    return _xlogy_terms(_masked(lp, mask), _masked(lq, mask)).sum(-1)


def _mixture(lp: Tensor, lq: Tensor, weight_p: float) -> Tensor:
    """log(weight_p * p + (1 - weight_p) * q)."""
    return torch.logaddexp(lp + math.log(weight_p), lq + math.log1p(-weight_p))


def divergence(lp: Tensor, lq: Tensor, mask: Tensor | None, kind: str, beta: float = 0.5) -> Tensor:
    """The divergence `kind` (distill.DIVERGENCES) between teacher lp and student lq, per row."""
    lp, lq = _masked(lp.float(), mask), _masked(lq.float(), mask)
    if kind == "forward_kl":
        return _xlogy_terms(lp, lq).sum(-1)
    if kind == "reverse_kl":
        return _xlogy_terms(lq, lp).sum(-1)
    if kind == "jsd":
        m = _mixture(lp, lq, beta)
        return beta * _xlogy_terms(lp, m).sum(-1) + (1.0 - beta) * _xlogy_terms(lq, m).sum(-1)
    if kind == "skew_forward_kl":
        return _xlogy_terms(lp, _mixture(lp, lq, beta)).sum(-1)
    if kind == "skew_reverse_kl":
        return _xlogy_terms(lq, _mixture(lq, lp, beta)).sum(-1)
    if kind == "tvd":
        diff = (lp.exp() - lq.exp()).abs()
        return 0.5 * (diff if mask is None else diff * mask).sum(-1)
    raise ValueError(f"unknown divergence {kind!r}")


def log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x <= 0, floored at log(TAIL_FLOOR)."""
    x = x.clamp(max=math.log1p(-TAIL_FLOOR))
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def dense_support(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float, shared: int | None = None
) -> tuple[Tensor, Tensor]:
    """(teacher lp, student lq) over the shared vocabulary [:shared] (both log-softmax over their own
    full vocabulary first, so mass a model puts on ids the other lacks is not redistributed)."""
    lq = F.log_softmax(student_logits.float() / temperature, -1)
    lp = F.log_softmax(teacher_logits.float() / temperature, -1)
    size = shared or min(lq.shape[-1], lp.shape[-1])
    return lp[..., :size], lq[..., :size]


def sparse_support(
    student_logits: Tensor,
    ids: Tensor,
    teacher_lp: Tensor,
    gold: Tensor | None,
    gold_lp: Tensor | None,
    temperature: float,
    mode: str = "tail",
    cache_temperature: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """(teacher lp, student lq, mask) over a cached top-k support [R, K (+1 gold) (+1 tail)].

    ids [R, K]: the teacher's top-k ids in the student's vocabulary (-1: none); teacher_lp [R, K]
    their log-probs at the cache's temperature. gold [R] / gold_lp [R] (optional): the trained token
    and its teacher log-prob, added to the support when it is not among the top-k (NaN gold_lp: not
    known, left out). mode "tail": teacher and student each get a bucket with the rest of their
    mass (requires temperature == cache_temperature: the teacher's tail at another temperature is
    unknown); "renorm": both renormalised over the support, the teacher's log-probs rescaled to
    `temperature`.
    """
    vocab = student_logits.shape[-1]
    ids = ids.long()
    tlp = teacher_lp.float()
    valid = (ids >= 0) & (ids < vocab) & torch.isfinite(tlp)
    if gold is not None and gold_lp is not None:
        g = gold.long()
        glp = gold_lp.float()
        in_topk = ((ids == g[:, None]) & valid).any(-1)
        extra_ok = (~in_topk) & (g >= 0) & (g < vocab) & torch.isfinite(glp)
        ids = torch.cat([ids, g.clamp(min=0)[:, None]], -1)
        tlp = torch.cat([tlp, glp.nan_to_num(_NEG)[:, None]], -1)
        valid = torch.cat([valid, extra_ok[:, None]], -1)
    safe = ids.clamp(0, vocab - 1)
    if mode == "renorm":
        z = student_logits.float().gather(-1, safe) / temperature
        lq = torch.log_softmax(z.masked_fill(~valid, _NEG), -1)
        tl = (tlp * cache_temperature / temperature).masked_fill(~valid, _NEG)
        lp = torch.log_softmax(tl, -1)
        return lp, lq, valid
    if mode != "tail":
        raise ValueError(f"unknown topk mode {mode!r}")
    lq_full = F.log_softmax(student_logits.float() / temperature, -1)
    lq = lq_full.gather(-1, safe).masked_fill(~valid, _NEG)
    lp = tlp.masked_fill(~valid, _NEG)
    lp_tail = log1mexp(torch.logsumexp(lp, -1, keepdim=True))
    lq_tail = log1mexp(torch.logsumexp(lq, -1, keepdim=True))
    mask = torch.cat([valid, torch.ones_like(valid[:, :1])], -1)
    return torch.cat([lp, lp_tail], -1), torch.cat([lq, lq_tail], -1), mask


def gold_cross_entropy(student_logits: Tensor, gold: Tensor) -> Tensor:
    """-log q(gold) per row at temperature 1."""
    return F.cross_entropy(student_logits.float(), gold.long(), reduction="none")


def feature_loss(student: Tensor, teacher: Tensor, kind: str) -> Tensor:
    """Per-row feature distance [R]: mse (mean over the dimension), nmse (of the L2-normalised
    vectors), cosine (1 - cos)."""
    s, t = student.float(), teacher.float()
    if kind == "mse":
        return (s - t).pow(2).mean(-1)
    if kind == "nmse":
        return (F.normalize(s, dim=-1) - F.normalize(t, dim=-1)).pow(2).sum(-1)
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(s, t, dim=-1)
    raise ValueError(f"unknown feature loss {kind!r}")
