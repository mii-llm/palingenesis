"""Distillation losses on the student's completion tokens.

Every loss takes the student's final hidden states at the completion positions
[N, H] and its output head, and projects them to logits a slice of rows at a
time: [N, V] logits are never materialized (V is 150-250k). Each slice is
differentiated as soon as it is computed and only the gradients are kept, so
peak memory is one slice of logits whatever N is (_ChunkedHead).

The losses are sums over tokens of ``weights[t] * loss[t]``; the caller puts the
normalization in `weights` (1 / global token count, or 1 / (length * sequences)),
so micro-batches accumulate to exactly the full-batch gradient.

  full_rkl     exact reverse KL over the shared vocabulary, from the teacher's
               full distribution (HF teacher). The default.
  topk_kl      KL between coarse-grained distributions: the teacher's top-k tokens
               plus the realized token, and a tail bucket with the remaining mass
               on both sides. `beta` weights reverse vs forward KL.
  sampled_rkl  the sampled-token estimate of reverse KL as a reward: advantage
               sg[log p_T(y) - log p_S(y)], REINFORCE with an importance ratio to
               the rollout policy, zeroed outside [is_low, is_high] (ICE-POP).
  xtok         cross-tokenizer: the same REINFORCE with one advantage per text
               chunk, sg[l_T(c) - l_S(c)] (chunk log-probabilities, align.py),
               plus an optional top-k KL at chunks of one token on each side.

Each returns (loss, stats): `stats` holds sums over tokens (the caller divides),
including ``k1``, the sampled estimate sum(log p_S - log p_T) that every loss can
report, so runs with different losses are compared on one scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Rows of [rows, V] fp32 logits per slice: 2^27 elements = 512 MB.
SLICE_ELEMENTS = 2**27
# Floor on a tail bucket's probability: a teacher top-k that holds all the mass
# would otherwise give log(0) and an infinite reverse KL.
TAIL_FLOOR = 1e-6


@dataclass
class SharedVocab:
    """Student log-probs seen in the teacher's vocabulary (see token_bridge).

    Ids below `size` are the same token in both; `swap` adds the student-only
    terminator's mass to the teacher's terminator slot.
    """

    size: int
    swap: dict[int, int] = field(default_factory=dict)

    def project(self, logp: Tensor) -> Tensor:
        out = logp[:, : self.size]
        if not self.swap:
            return out
        source = torch.tensor(list(self.swap), device=logp.device)
        target = torch.tensor(list(self.swap.values()), device=logp.device)
        return out.index_copy(1, target, torch.logaddexp(out[:, target], logp[:, source]))


class _ChunkedHead(torch.autograd.Function):
    """sum over slices of fn(head(hidden[a:b]), a, b), differentiated slice by slice.

    Forward computes each slice's loss and immediately its gradients with respect
    to the slice's hidden states and the head's parameters; backward only scales
    them by the incoming gradient. Correct under any downstream scaling.
    """

    @staticmethod
    def forward(ctx, hidden, head, fn, bounds, stats, *params):
        grad_hidden = torch.zeros_like(hidden) if hidden.requires_grad else None
        grad_params = [torch.zeros_like(p) if p.requires_grad else None for p in params]
        inputs_of = [p for p in params if p.requires_grad]
        total = hidden.new_zeros((), dtype=torch.promote_types(hidden.dtype, torch.float32))
        for a, b in bounds:
            with torch.enable_grad():
                h = hidden[a:b].detach().requires_grad_(hidden.requires_grad)
                loss, slice_stats = fn(_upcast(head(h)), a, b)
            if loss.requires_grad:
                grads = list(torch.autograd.grad(loss, ([h] if hidden.requires_grad else []) + inputs_of,
                                                 allow_unused=True))
                if hidden.requires_grad and (g := grads.pop(0)) is not None:
                    grad_hidden[a:b] = g
                for i, p in enumerate(params):
                    if p.requires_grad and (g := grads.pop(0)) is not None:
                        grad_params[i] += g
            total += loss.detach()
            _accumulate(stats, slice_stats)
        ctx.has_grad = [grad_hidden is not None] + [g is not None for g in grad_params]
        ctx.save_for_backward(*(g if g is not None else torch.empty(0) for g in [grad_hidden, *grad_params]))
        return total

    @staticmethod
    def backward(ctx, grad_output):
        saved = [g * grad_output if has else None for g, has in zip(ctx.saved_tensors, ctx.has_grad)]
        return saved[0], None, None, None, None, *saved[1:]


def slice_bounds(n: int, rows: int, ends: list[int] | None = None) -> list[tuple[int, int]]:
    """Consecutive [a, b) slices of at most `rows` rows, cut only at `ends` when given.

    `ends` (sorted, last == n) lists where a slice may end; a stretch longer than
    `rows` between two allowed ends becomes one oversized slice.
    """
    if ends is None:
        return [(a, min(a + rows, n)) for a in range(0, n, rows)]
    bounds, a, k = [], 0, 0
    while a < n:
        b = None
        while k < len(ends) and ends[k] <= a + rows:
            b = ends[k]
            k += 1
        if b is None:
            b = ends[k]
            k += 1
        bounds.append((a, b))
        a = b
    return bounds


def chunked_head(hidden: Tensor, head: nn.Module, fn: Callable[[Tensor, int, int], tuple[Tensor, dict]],
                 ends: list[int] | None = None, rows: int | None = None) -> tuple[Tensor, dict[str, float]]:
    """Run `fn(logits[a:b] (fp32), a, b) -> (loss, stats)` over row slices of the head's output."""
    rows = rows or max(1, SLICE_ELEMENTS // head.weight.shape[0])
    bounds = slice_bounds(hidden.shape[0], rows, ends)
    stats: dict[str, float] = {}
    if not torch.is_grad_enabled():
        total = hidden.new_zeros((), dtype=torch.promote_types(hidden.dtype, torch.float32))
        for a, b in bounds:
            loss, slice_stats = fn(_upcast(head(hidden[a:b])), a, b)
            total += loss
            _accumulate(stats, slice_stats)
        return total, stats
    return _ChunkedHead.apply(hidden, head, fn, bounds, stats, *head.parameters()), stats


def _upcast(logits: Tensor) -> Tensor:
    """Logits in at least fp32 (bf16 log-softmax over 150k+ ids loses the tail)."""
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def _accumulate(stats: dict[str, float], new: dict) -> None:
    for k, v in new.items():
        stats[k] = stats.get(k, 0.0) + float(v)


def _log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x <= 0, at least ~log(TAIL_FLOOR)."""
    x = x.clamp(max=-TAIL_FLOOR)
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def coarse_kl(student_lp: Tensor, teacher_lp: Tensor, valid: Tensor, beta: float) -> Tensor:
    """beta * KL(q||p) + (1 - beta) * KL(p||q) over [K support tokens + tail bucket].

    `student_lp`/`teacher_lp` [R, K] are log-probs of the support tokens (entries
    where `valid` is False are ignored); each side's tail bucket holds the mass
    outside its valid entries.
    """
    neg_inf = torch.finfo(student_lp.dtype).min
    lq = student_lp.masked_fill(~valid, neg_inf)
    lp = teacher_lp.masked_fill(~valid, neg_inf)
    lq = torch.cat([lq, _log1mexp(torch.logsumexp(lq, -1, keepdim=True))], -1)
    lp = torch.cat([lp, _log1mexp(torch.logsumexp(lp, -1, keepdim=True))], -1)
    mask = torch.cat([valid, torch.ones_like(valid[:, :1])], -1)
    diff = torch.where(mask, lq - lp, torch.zeros_like(lq))
    reverse = (lq.exp() * diff).sum(-1)
    forward = (lp.exp() * -diff).sum(-1)
    return beta * reverse + (1.0 - beta) * forward


def _policy_gradient(lp: Tensor, advantage: Tensor, behaviour: Tensor | None, is_low: float,
                     is_high: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token -ratio * advantage, ratio = pi/mu zeroed outside [is_low, is_high].

    Without behaviour log-probs the ratio is exp(lp - sg(lp)) = 1 with the
    score-function gradient: plain REINFORCE. Returns (loss, kept, |log ratio|).
    """
    if behaviour is None:
        return -torch.exp(lp - lp.detach()) * advantage, torch.ones_like(lp, dtype=torch.bool), torch.zeros_like(lp)
    log_ratio = lp - behaviour
    ratio = torch.exp(log_ratio)
    kept = (ratio.detach() >= is_low) & (ratio.detach() <= is_high)
    return -(ratio * advantage) * kept, kept, log_ratio.detach().abs()


# ----------------------------------------------------------------------- losses


def full_rkl(hidden: Tensor, head: nn.Module, targets: Tensor, weights: Tensor, vocab: SharedVocab,
             teacher_logprobs: Callable[[int, int], Tensor]) -> tuple[Tensor, dict[str, float]]:
    """Exact reverse KL sum_v q(v) (log q(v) - log p(v)) per token, over the shared vocabulary.

    `targets` are the completion ids in the teacher's vocabulary (for k1);
    `teacher_logprobs(a, b)` returns the teacher's log-probs [b - a, vocab.size]
    for rows a..b, computed on demand so the teacher's logits are never stored.
    Student mass on tokens outside the shared vocabulary is excluded and reported
    as `residual`.
    """

    def fn(logits: Tensor, a: int, b: int):
        lq = vocab.project(F.log_softmax(logits, -1))
        lp = teacher_logprobs(a, b).to(lq.device, lq.dtype)
        q = lq.exp()
        kl = (q * (lq - lp)).sum(-1)
        tgt = targets[a:b, None]
        k1 = (lq.gather(1, tgt) - lp.gather(1, tgt)).squeeze(1)
        stats = {"kl": kl.detach().sum(), "k1": k1.detach().sum(), "residual": (1 - q.detach().sum(-1)).sum()}
        return (weights[a:b] * kl).sum(), stats

    return chunked_head(hidden, head, fn)


def topk_kl(hidden: Tensor, head: nn.Module, targets: Tensor, weights: Tensor, vocab: SharedVocab,
            support: Tensor, teacher_support_lp: Tensor, valid: Tensor, teacher_token_lp: Tensor,
            beta: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    """coarse_kl over the teacher's support tokens [N, K] (ids in the shared vocabulary).

    The support is the teacher's top-k plus the realized token (so reverse KL sees
    the token the student actually chose); `valid` masks padding and duplicates.
    """

    def fn(logits: Tensor, a: int, b: int):
        lq_all = vocab.project(F.log_softmax(logits, -1))
        ok = valid[a:b]
        lq = lq_all.gather(1, support[a:b].clamp(min=0))
        kl = coarse_kl(lq, teacher_support_lp[a:b].to(lq.dtype), ok, beta)
        k1 = lq_all.gather(1, targets[a:b, None]).squeeze(1) - teacher_token_lp[a:b]
        return (weights[a:b] * kl).sum(), {"kl": kl.detach().sum(), "k1": k1.detach().sum()}

    return chunked_head(hidden, head, fn)


def sampled_rkl(hidden: Tensor, head: nn.Module, targets: Tensor, weights: Tensor, teacher_token_lp: Tensor,
                behaviour_lp: Tensor | None = None, is_low: float = 0.5,
                is_high: float = 2.0) -> tuple[Tensor, dict[str, float]]:
    """REINFORCE on the per-token reward log p_T(y) - log p_S(y) (sampled reverse KL).

    `targets` are the sampled ids in the student's vocabulary; `teacher_token_lp`
    the teacher's log-prob of the same token (its mapped id).
    """

    def fn(logits: Tensor, a: int, b: int):
        lp = F.log_softmax(logits, -1).gather(1, targets[a:b, None]).squeeze(1)
        advantage = (teacher_token_lp[a:b] - lp).detach()
        mu = behaviour_lp[a:b] if behaviour_lp is not None else None
        loss, kept, abs_log_ratio = _policy_gradient(lp, advantage, mu, is_low, is_high)
        stats = {"kl": -advantage.sum(), "k1": -advantage.sum(), "is_dropped": (~kept).sum(),
                 "abs_log_ratio": abs_log_ratio.sum()}
        return (weights[a:b] * loss).sum(), stats

    return chunked_head(hidden, head, fn)


@dataclass
class ChunkTargets:
    """Per-token chunk ids and per-chunk teacher log-probs for xtok, over a micro-batch.

    `chunk` [N]: global chunk id of each student token (-1: none); `teacher_lp` [C]:
    the teacher's log-probability of each chunk's text (sum over its tokens); `keep`
    [C]: chunks that carry loss; `ends`: row offsets where chunks (and sequences)
    end, so no slice cuts a chunk.
    """

    chunk: Tensor
    teacher_lp: Tensor
    keep: Tensor
    ends: list[int]


@dataclass
class DenseTargets:
    """Top-k teacher distributions at one-to-one chunks, mapped to student ids.

    `rows` [D]: student rows; `support`, `teacher_lp`, `valid` [D, K].
    """

    rows: Tensor
    support: Tensor
    teacher_lp: Tensor
    valid: Tensor


def xtok(hidden: Tensor, head: nn.Module, targets: Tensor, weights: Tensor, chunks: ChunkTargets,
         behaviour_lp: Tensor | None = None, spread: str = "chunk", is_low: float = 0.5, is_high: float = 2.0,
         dense: DenseTargets | None = None, dense_weight: float = 0.0,
         beta: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    """Cross-tokenizer REINFORCE with chunk advantages A_c = sg[l_T(c) - l_S(c)].

    spread="chunk" gives every token of chunk c the advantage A_c: the chunk's
    reward credited to all the actions that produced its text. spread="proportional"
    gives token t the share A_c * log p_S(t) / l_S(c) (summing to A_c over the
    chunk), the per-token target of rescaling the chunk's log-probability to the
    teacher's. The optional dense term adds `dense_weight` * coarse_kl at
    one-to-one chunks.
    """
    n_chunks = chunks.teacher_lp.shape[0]
    dense_rows = dense.rows.tolist() if dense is not None and dense_weight > 0 else []

    def fn(logits: Tensor, a: int, b: int):
        logp = F.log_softmax(logits, -1)
        lp = logp.gather(1, targets[a:b, None]).squeeze(1)
        cid = chunks.chunk[a:b]
        scratch = torch.full_like(cid, n_chunks)                           # off tokens go to a scratch slot
        keep = torch.cat([chunks.keep, chunks.keep.new_zeros(1)])
        on = keep[torch.where(cid >= 0, cid, scratch)]
        safe = torch.where(on, cid, scratch)
        student_chunk = lp.new_zeros(n_chunks + 1).index_add(0, safe, lp.detach())
        count = lp.new_zeros(n_chunks + 1).index_add(0, safe, torch.ones_like(lp))
        teacher_chunk = torch.cat([chunks.teacher_lp.to(lp), lp.new_zeros(1)])
        advantage = (teacher_chunk - student_chunk)[safe]
        if spread == "proportional":
            share = torch.where(student_chunk[safe] < 0, lp.detach() / student_chunk[safe], 1.0 / count[safe])
            advantage = advantage * share
        advantage = advantage * on
        mu = behaviour_lp[a:b] if behaviour_lp is not None else None
        loss, kept, abs_log_ratio = _policy_gradient(lp, advantage, mu, is_low, is_high)
        loss = (weights[a:b] * loss * on).sum()
        present = torch.zeros(n_chunks + 1, device=lp.device, dtype=torch.bool)
        present[safe[on]] = True
        k1 = (student_chunk - teacher_chunk)[:n_chunks][present[:n_chunks]].sum()
        stats = {"kl": k1.detach(), "k1": k1.detach(), "is_dropped": (~kept & on).sum(),
                 "abs_log_ratio": (abs_log_ratio * on).sum(), "supervised": on.sum()}
        rows = [i for i, r in enumerate(dense_rows) if a <= r < b]
        if rows:
            sel = torch.tensor(rows, device=lp.device)
            local = dense.rows[sel] - a
            lq = logp[local].gather(1, dense.support[sel].clamp(min=0))
            kl = coarse_kl(lq, dense.teacher_lp[sel].to(lq.dtype), dense.valid[sel], beta)
            loss = loss + dense_weight * (weights[a:b][local] * kl).sum()
            stats["dense_kl"] = kl.detach().sum()
            stats["dense_tokens"] = len(rows)
        return loss, stats

    return chunked_head(hidden, head, fn, ends=chunks.ends)
