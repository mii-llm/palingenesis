"""Policy-gradient objectives, advantages and loss aggregation.

target_logprobs(hidden, head, targets) is log π_θ(y_t) of every sampled token, differentiable,
with logits in fp32 one slice of rows at a time: [N, V] logits are never materialized, and
the backward pass recomputes each slice. Every objective is then a few lines over the
per-token log-probs lp, the sampler's log-probs μ (vLLM's processed log-probs: the
distribution actually sampled from), advantages A and aggregation weights w:

  ratio ρ = exp(lp - μ)  (policy vs sampler: staleness and engine mismatch in one ratio)

  masked_is  -Σ w · M · S · min(sg ρ, C) · A · lp,
             M drops tokens PPO's clip would (A>0, ρ>1+ε_high or A<0, ρ<1-ε_low). While ρ ≤ C
             the gradient is DAPO's clipped surrogate against μ; no recompute of π_old.
  cispo      -Σ w · S · min(sg ρ, C) · A · lp                              (MiniMax-M1, ScaleRL)
  icepop     -Σ w · S · 1[ρ ∈ [lo, hi]] · sg ρ · A · lp                    (INTELLECT-3)
  gspo       per sequence s = exp(mean_t log ρ), -Σ_seq w · min(s A, clip(s, 1±ε) A)
             (gradient s · A · mean_t ∇lp where unclipped)                  (Qwen GSPO)
  ipo        -Σ w · S · 1[|e^lp - e^μ| ≤ ε] · min(sg ρ, C) · A · lp       (probability-difference mask)
  reinforce  -Σ w · S · A · lp                                             (score function, on-policy only)

reinforce leaves ρ out: its gradient is the policy gradient only when the tokens were sampled from π_θ, which the
configuration guarantees (rollout.max_staleness 0, no forced tokens); ρ is still measured and logged, and S still
drops a trajectory the sampler and the trainer disagree on.

S is the sequence mask (Trust Region Masking): a trajectory whose mean |ρ - 1| exceeds
loss.seq_mask is dropped whole, since token-level clipping cannot bound a long sequence's
divergence. All coefficients are detached: the loss's value is not an objective, its
gradient is.

The observation loss (loss.observation_weight, ECHO) is added to any of them: λ x the mean over trajectories of
the summed NLL of the environment's output tokens divided by the trajectory's observation length
(observation_weights gives each target its weight).

Truncated sampling (rollout.top_p / top_k). The sampler draws y_t from the policy renormalised over its support S_t
(the tokens top-k / top-p keep), and μ is that renormalised distribution. The ratio then needs the trained policy
renormalised over the same S_t (sampled_logprobs, from the sampler's candidates):

  log π_S(y_t) = z_y - logsumexp_{v ∈ S_t} z_v        (z = logits / temperature)

which only needs the |S_t| logits of the support (one gathered row of the head per candidate, not the vocabulary).
The full-vocabulary log π(y_t) is smaller by log π(S_t) and biases every ratio low (DeepSeek-V3.2, arXiv:2512.02556,
section 3.1; TRL AsyncGRPOTrainer, "Top-k and top-p sampling").

Score centering (loss.score_centering topk; arXiv:2609.20807, Appendix A.2, as in TRL AsyncGRPOTrainer). The token
objectives above are -A · f(ρ_t) ∇log π(y_t) with f(ρ) = ρ g(ρ) (g: the clip mask and cap of the loss type). Its
expectation under the sampler at a position is not 0 when π ≠ μ: E_μ[f(ρ_v) ∇log π(v)] = Σ_v π(v) g(ρ_v) ∇log π(v),
so a constant advantage still pulls π toward μ. Centering subtracts that expectation, estimated from the sampler's
top-k candidates V_t (q_v = μ(v), p_v = π(v)) and one ratio for the tail outside them,
ρ_tail = (1 - Σ_V p) / (1 - Σ_V q), on which Σ_tail π(v) ∇log π(v) = -Σ_V p_v ∇log π(v):

  correction_t = Σ_{v ∈ V_t} sg(p_v (g(ρ_v) - g(ρ_tail))) · log π(v)
  loss         -Σ w · S · (sg(coef_t) · lp_t - A · correction_t)
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from palingenesis.evalstats import pass_at_k_rewards
from palingenesis.opd.fused_rkl import Head, unwrap
from palingenesis.opd.losses import SLICE_ELEMENTS
from palingenesis.rl import head_kernels

logger = logging.getLogger(__name__)
# A step with more of its trained tokens left out for an incomplete support than this warns: ties at the k-th logit
# are rare, so a high share means the engine returned raw (not processed) log-probs or too few of them.
SUPPORT_INCOMPLETE_WARN = 0.05

# ------------------------------------------------------------ token log-probs


class LowPrecisionWeight:
    """A bf16 copy of the output head's fp32 master weight, refreshed only when the optimizer
    has changed it (the weight's version counter), so micro-batches share one cast."""

    def __init__(self):
        self.key = None
        self.value = None

    def get(self, weight: Tensor) -> Tensor:
        dtype = torch.bfloat16 if weight.is_cuda else weight.dtype
        key = (weight.data_ptr(), weight._version, dtype)
        if key != self.key:
            self.value, self.key = weight.detach().to(dtype), key
        return self.value


def _logits(h: Tensor, w: Tensor, head: Head) -> tuple[Tensor, Tensor]:
    """(fp32 projection z, the model's logits y) for a slice: bf16 GEMM, fp32 accumulate and output."""
    if h.is_cuda and h.dtype == torch.bfloat16:
        z = torch.mm(h, w.T, out_dtype=torch.float32)
    else:  # other devices: no mixed-dtype GEMM, compute in fp32
        z = torch.mm(h.float(), w.float().T)
    y = z * head.multiplier if head.multiplier != 1.0 else z
    if head.softcap:
        y = head.softcap * torch.tanh(y / head.softcap)
    return z, y


class _LinearTargetLogProbs(torch.autograd.Function):
    """log π(target) for a bias-free linear head (with its multiplier / softcap): two GEMM
    passes per slice and an analytic gradient; the forward keeps each row's log-sum-exp so
    the backward only recomputes the logits."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, head, w_low, rows, entropy):
        h = hidden.detach().to(w_low.dtype)
        n = h.shape[0]
        lp = torch.empty(n, dtype=torch.float32, device=h.device)
        lse = torch.empty(n, dtype=torch.float32, device=h.device)
        ent = torch.empty(n if entropy else 0, dtype=torch.float32, device=h.device)
        fused = head_kernels.available(h) and h.dtype == torch.bfloat16
        for a in range(0, n, rows):
            if fused:  # one pass over the slice's logits (head_kernels)
                z = torch.mm(h[a : a + rows], w_low.T, out_dtype=torch.float32)
                lp[a : a + rows], lse[a : a + rows], e = head_kernels.forward(
                    z, targets[a : a + rows], head.multiplier, head.softcap, entropy
                )
                if entropy:
                    ent[a : a + rows] = e
                continue
            _, y = _logits(h[a : a + rows], w_low, head)
            lse[a : a + rows] = y.logsumexp(-1)
            lp[a : a + rows] = y.gather(1, targets[a : a + rows, None]).squeeze(1) - lse[a : a + rows]
            if entropy:
                p = (y - lse[a : a + rows, None]).exp_()
                ent[a : a + rows] = lse[a : a + rows] - (p * y).sum(-1)
        ctx.save_for_backward(h, targets, lse)
        ctx.head, ctx.w_low, ctx.rows, ctx.hidden_dtype = head, w_low, rows, hidden.dtype
        ctx.mark_non_differentiable(ent)
        return lp, ent

    @staticmethod
    def backward(ctx, grad_lp, grad_ent):
        h, targets, lse = ctx.saved_tensors
        head, w_low, rows = ctx.head, ctx.w_low, ctx.rows
        want_hidden, want_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.zeros(h.shape, dtype=ctx.hidden_dtype, device=h.device) if want_hidden else None
        grad_weight = torch.zeros(w_low.shape, dtype=torch.float32, device=h.device) if want_weight else None
        index = torch.arange(rows, device=h.device)
        fused = head_kernels.available(h) and h.dtype == torch.bfloat16
        buffer = torch.empty(min(rows, h.shape[0]), w_low.shape[0], dtype=h.dtype, device=h.device) if fused else None
        for a in range(0, h.shape[0], rows):
            g = grad_lp[a : a + rows].float()
            if not g.any():
                continue
            b = a + g.shape[0]
            if fused:  # one pass: softmax, one-hot, softcap / multiplier chain rule, bf16 out
                z = torch.mm(h[a:b], w_low.T, out_dtype=torch.float32)
                grad_z = head_kernels.backward(
                    z, targets[a:b], lse[a:b], g.contiguous(), head.multiplier, head.softcap, buffer[: b - a]
                )
                del z
                if want_hidden:
                    grad_hidden[a:b] = torch.mm(grad_z, w_low).to(ctx.hidden_dtype)
                if want_weight:
                    torch.addmm(grad_weight, grad_z.T, h[a:b], out_dtype=torch.float32, out=grad_weight)
                continue
            _, y = _logits(h[a:b], w_low, head)
            # d lp / d y = onehot(target) - softmax(y)
            grad_y = (y - lse[a:b, None]).exp_().mul_(-g[:, None])
            grad_y[index[: b - a], targets[a:b]] += g
            if head.softcap:
                grad_y.mul_(1 - (y / head.softcap) ** 2)
            if head.multiplier != 1.0:
                grad_y.mul_(head.multiplier)
            grad_z = grad_y.to(h.dtype)
            del y, grad_y
            if want_hidden:
                grad_hidden[a:b] = torch.mm(grad_z, w_low).to(ctx.hidden_dtype)
            if want_weight:
                if grad_z.is_cuda and grad_z.dtype == torch.bfloat16:
                    torch.addmm(grad_weight, grad_z.T, h[a:b], out_dtype=torch.float32, out=grad_weight)
                else:
                    grad_weight.addmm_(grad_z.T.float(), h[a:b].float())
        return grad_hidden, grad_weight, None, None, None, None, None


class _TargetLogProbs(torch.autograd.Function):
    """log π(target) through any head module (autograd per slice in the backward)."""

    @staticmethod
    def forward(ctx, hidden, targets, head, rows, entropy, scale, *params):
        n = hidden.shape[0]
        lp = torch.empty(n, dtype=torch.float32, device=hidden.device)
        ent = torch.empty(n if entropy else 0, dtype=torch.float32, device=hidden.device)
        with torch.no_grad():
            for a in range(0, n, rows):
                logits = head(hidden[a : a + rows]).float() * scale
                lse = torch.logsumexp(logits, -1)
                lp[a : a + rows] = logits.gather(1, targets[a : a + rows, None]).squeeze(1) - lse
                if entropy:
                    ent[a : a + rows] = lse - (torch.softmax(logits, -1) * logits).sum(-1)
        ctx.save_for_backward(hidden, targets)
        ctx.head, ctx.rows, ctx.scale, ctx.params = head, rows, scale, params
        ctx.mark_non_differentiable(ent)
        return lp, ent

    @staticmethod
    def backward(ctx, grad_lp, grad_ent):
        hidden, targets = ctx.saved_tensors
        params = [p for p in ctx.params if p.requires_grad]
        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_params = [torch.zeros_like(p) for p in params]
        for a in range(0, hidden.shape[0], ctx.rows):
            g = grad_lp[a : a + ctx.rows]
            if not g.any():
                continue
            with torch.enable_grad():
                h = hidden[a : a + ctx.rows].detach().requires_grad_(grad_hidden is not None)
                logits = ctx.head(h).float() * ctx.scale
                lp = logits.gather(1, targets[a : a + ctx.rows, None]).squeeze(1) - torch.logsumexp(logits, -1)
                inputs = ([h] if grad_hidden is not None else []) + params
                grads = torch.autograd.grad(lp, inputs, g, allow_unused=True)
            if grad_hidden is not None:
                grad_hidden[a : a + ctx.rows] = grads[0]
                grads = grads[1:]
            for acc, grad in zip(grad_params, grads):
                if grad is not None:
                    acc += grad
        it = iter(grad_params)
        out_params = [next(it) if p.requires_grad else None for p in ctx.params]
        return grad_hidden, None, None, None, None, None, *out_params


def target_logprobs(
    hidden: Tensor,
    head: nn.Module,
    targets: Tensor,
    entropy: bool = False,
    low: LowPrecisionWeight | None = None,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """(log π(targets), entropies) for hidden-state rows [N, H]: differentiable log-probs with
    fp32 logits, a slice of rows at a time (and, with `entropy`, the policy's entropy at each
    row, for logging). Linear heads take the fused path: bf16 GEMMs on the GPU with fp32
    accumulation, one cast of the weight per optimizer step (`low`). `scale` multiplies the fp32 logits (1 / the
    sampling temperature): scaling the bf16 hidden state instead would round it once more."""
    rows = max(1, SLICE_ELEMENTS // head.weight.shape[0])
    linear = unwrap(head)
    if linear is not None and (scale == 1.0 or not linear.softcap):  # logits * scale = z * (multiplier * scale)
        if scale != 1.0:
            linear = Head(linear.weight, linear.multiplier * scale, linear.softcap)
        w_low = (low or LowPrecisionWeight()).get(linear.weight)
        with torch.autocast(hidden.device.type, enabled=False):
            return _LinearTargetLogProbs.apply(hidden, linear.weight, targets, linear, w_low, rows, entropy)
    return _TargetLogProbs.apply(hidden, targets, head, rows, entropy, float(scale), *head.parameters())


# ------------------------------------------------- the sampler's candidates


@dataclass
class SamplerCandidates:
    """The sampler's candidates at a micro-batch's trained tokens, padded to [N, K] (most likely first).

    ids, logprobs: [N, K] token ids (0 at padding) and the sampler's processed log-probs μ(v) (-inf at padding).
    support: the candidates are the sampler's support (rollout.top_p / top_k): the policy is renormalised over them.
    complete: [N] bool, the token's support was seen whole: fewer candidates than requested (with exactly as many,
        tokens tied at the k-th logit may be missing; such tokens are left out of the loss). All True without support.
    present: [N] bool, the token has candidates (a forced token has none: it is scored over the whole vocabulary).
    """

    ids: Tensor
    logprobs: Tensor
    support: bool
    complete: Tensor
    present: Tensor

    @classmethod
    def build(cls, parts, requested: int, support: bool, device) -> "SamplerCandidates":
        """From opd.rollout.Candidates of the micro-batch's trained tokens, in token order (one part per row)."""
        import numpy as np

        counts = np.concatenate([p.counts for p in parts]).astype(np.int64) if parts else np.zeros(0, np.int64)
        flat_ids = np.concatenate([p.ids for p in parts]) if parts else np.zeros(0, np.int32)
        flat_lp = np.concatenate([p.logprobs for p in parts]) if parts else np.zeros(0, np.float32)
        n, width = len(counts), max(1, int(counts.max()) if len(counts) else 1)
        rows = np.repeat(np.arange(n), counts)
        cols = np.arange(len(flat_ids)) - np.repeat(np.cumsum(counts) - counts, counts)
        ids = np.zeros((n, width), np.int64)
        logprobs = np.full((n, width), -np.inf, np.float32)
        ids[rows, cols] = flat_ids
        logprobs[rows, cols] = flat_lp
        present = counts > 0
        complete = counts < requested if support else np.ones(n, bool)
        return cls(
            torch.from_numpy(ids).to(device, non_blocking=True),
            torch.from_numpy(logprobs).to(device, non_blocking=True),
            support,
            torch.from_numpy(complete).to(device, non_blocking=True),
            torch.from_numpy(present).to(device, non_blocking=True),
        )


@dataclass
class SampledLogProbs:
    """sampled_logprobs' result. lp: [N] log-prob of each trained token under the distribution the sampler drew it
    from (renormalised over the support with top-p / top-k), differentiable; entropy: [N] that distribution's
    entropy (or [0]); candidate_lp: [N, K] log-probs of the candidates under the same distribution, differentiable
    (-inf at padding); valid: [N] bool, False for tokens to leave out of the loss (support possibly incomplete, or the
    sampled token missing from it); sampler: the candidates."""

    lp: Tensor
    entropy: Tensor
    candidate_lp: Tensor
    valid: Tensor
    sampler: SamplerCandidates


class _GatheredLogits(torch.autograd.Function):
    """z[n, j] = hidden[n] · weight[ids[n, j]]: the raw projection of a few chosen tokens per row (computed in fp32,
    or fp64 for fp64 inputs, from the head's low-precision copy, as the full-vocabulary path accumulates bf16 GEMMs
    in fp32), a slice of rows at a time; the backward recomputes the gathered rows instead of keeping them."""

    @staticmethod
    def forward(ctx, hidden, weight, ids, w_low, rows):
        dtype = torch.promote_types(torch.promote_types(w_low.dtype, hidden.dtype), torch.float32)
        z = torch.empty(ids.shape, dtype=dtype, device=hidden.device)
        for a in range(0, ids.shape[0], rows):
            gathered = w_low[ids[a : a + rows]].to(dtype)  # [r, J, H]
            z[a : a + rows] = torch.bmm(gathered, hidden[a : a + rows].detach().to(dtype).unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(hidden, ids)
        ctx.w_low, ctx.rows, ctx.weight_shape, ctx.dtype = w_low, rows, weight.shape, dtype
        return z

    @staticmethod
    def backward(ctx, grad_z):
        hidden, ids = ctx.saved_tensors
        w_low, rows = ctx.w_low, ctx.rows
        want_hidden, want_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.zeros(hidden.shape, dtype=hidden.dtype, device=hidden.device) if want_hidden else None
        grad_weight = torch.zeros(ctx.weight_shape, dtype=ctx.dtype, device=hidden.device) if want_weight else None
        for a in range(0, ids.shape[0], rows):
            g = grad_z[a : a + rows].to(ctx.dtype)
            if want_hidden:
                gathered = w_low[ids[a : a + rows]].to(ctx.dtype)
                grad_hidden[a : a + rows] = torch.bmm(g.unsqueeze(1), gathered).squeeze(1).to(hidden.dtype)
            if want_weight:
                h = hidden[a : a + rows].to(ctx.dtype)
                grad_weight.index_add_(
                    0, ids[a : a + rows].reshape(-1), (g.unsqueeze(-1) * h.unsqueeze(1)).flatten(0, 1)
                )
        return grad_hidden, grad_weight, None, None, None


GATHER_ELEMENTS = 2**25  # fp32 elements of gathered head rows per slice (128 MiB)


def candidate_logits(hidden: Tensor, head: nn.Module, ids: Tensor, low=None, scale: float = 1.0) -> Tensor:
    """[N, J] logits (the model's: multiplier, soft-capping) of the tokens `ids` [N, J] at each row of `hidden`
    [N, H], times `scale`, in fp32 (fp64 for fp64 inputs) and differentiable. Needs a bias-free linear head (opd.fused_rkl.unwrap)."""
    linear = unwrap(head)
    if linear is None:
        raise ValueError(
            "renormalising over the sampler's candidates (rollout.top_p / top_k, loss.score_centering) needs a plain "
            "linear output head (no bias)"
        )
    w_low = (low or LowPrecisionWeight()).get(linear.weight)
    rows = max(1, GATHER_ELEMENTS // max(1, ids.shape[1] * w_low.shape[1]))
    with torch.autocast(hidden.device.type, enabled=False):
        y = _GatheredLogits.apply(hidden, linear.weight, ids, w_low, rows)
        if linear.multiplier != 1.0:
            y = y * linear.multiplier
        if linear.softcap:
            y = linear.softcap * torch.tanh(y / linear.softcap)
        return y * scale if scale != 1.0 else y


def sampled_logprobs(
    hidden: Tensor,
    head: nn.Module,
    targets: Tensor,
    sampler: SamplerCandidates,
    entropy: bool = False,
    low: LowPrecisionWeight | None = None,
    scale: float = 1.0,
) -> SampledLogProbs:
    """target_logprobs for tokens sampled with candidates: log π_S(y_t) = z_y - logsumexp_{v ∈ S_t} z_v over the
    sampler's support S_t when `sampler.support` (z = logits x scale; its gradient flows through the
    renormalisation; the entropy is that of π_S), the full-vocabulary log π(y_t) otherwise and for tokens without
    candidates. Also the log-probs of every candidate under the same distribution (score centering): over the support
    z_v - logsumexp_S z, over the vocabulary log π(y) + z_v - z_y (exact, and the gradient of the normaliser flows
    through log π(y))."""
    n = targets.shape[0]
    device = hidden.device
    pad = torch.isneginf(sampler.logprobs)  # [N, K]
    z = candidate_logits(hidden, head, torch.cat([targets[:, None], sampler.ids], 1), low, scale)
    z_y, z_c = z[:, 0], z[:, 1:]
    renormalised = sampler.present if sampler.support else torch.zeros(n, dtype=torch.bool, device=device)
    member = ((sampler.ids == targets[:, None]) & ~pad).any(1)
    valid = ~renormalised | (sampler.complete & member)
    lp = torch.zeros(n, dtype=z.dtype, device=device)
    ent = torch.zeros(n if entropy else 0, dtype=z.dtype, device=device)
    candidate_lp = torch.full(z_c.shape, float("-inf"), dtype=z.dtype, device=device)
    over = renormalised.nonzero().squeeze(1)
    full = (~renormalised).nonzero().squeeze(1)
    if over.numel():
        zs = z_c.index_select(0, over).masked_fill(pad.index_select(0, over), float("-inf"))
        lse = torch.logsumexp(zs, -1)
        lp = lp.index_copy(0, over, z_y.index_select(0, over) - lse)
        candidate_lp = candidate_lp.index_copy(0, over, zs - lse[:, None])
        if entropy:
            with torch.no_grad():
                probs = (zs - lse[:, None]).exp()
                ent[over] = lse - (probs * zs.masked_fill(pad.index_select(0, over), 0.0)).sum(-1)
    if full.numel():
        lp_full, ent_full = target_logprobs(
            hidden.index_select(0, full), head, targets.index_select(0, full), entropy, low, scale
        )
        lp_full = lp_full.to(z.dtype)
        lp = lp.index_copy(0, full, lp_full)
        shift = z_c.index_select(0, full) - z_y.index_select(0, full)[:, None]
        candidate_lp = candidate_lp.index_copy(
            0, full, (lp_full[:, None] + shift).masked_fill(pad.index_select(0, full), float("-inf"))
        )
        if entropy:
            ent[full] = ent_full.to(z.dtype)
    return SampledLogProbs(lp, ent, candidate_lp, valid, sampler)


def _keep_factor(config, ratio: Tensor, advantage: Tensor, p: Tensor | None = None, q: Tensor | None = None) -> Tensor:
    """g(ρ) = f(ρ) / ρ of the loss type's token coefficient f (its clip mask and cap; ρ = 0 is finite): the factor
    score centering weighs each candidate's π(v) ∇log π(v) by. `advantage` broadcasts against `ratio`; `p`, `q`: the
    policy's and the sampler's probabilities (ipo's mask)."""
    capped = torch.where(ratio > config.is_cap, config.is_cap / ratio.clamp(min=1e-30), torch.ones_like(ratio))
    if config.type == "masked_is":
        clipped = ((advantage > 0) & (ratio > 1 + config.eps_high)) | ((advantage < 0) & (ratio < 1 - config.eps_low))
        return (~clipped).float() * capped
    if config.type == "cispo":
        return capped
    if config.type == "icepop":
        return ((ratio >= config.icepop_low) & (ratio <= config.icepop_high)).float()
    if config.type == "ipo":
        return ((p - q).abs() <= config.ipo_eps).float() * capped
    raise ValueError(f"loss.score_centering is not defined for loss.type {config.type!r}")


def centering_correction(sampled: SampledLogProbs, advantage: Tensor, config) -> tuple[Tensor, dict[str, Tensor]]:
    """Score centering's per-token term Σ_v sg(p_v (g(ρ_v) - g(ρ_tail))) log π(v) over the sampler's top
    loss.score_centering_k candidates (see the module docstring), differentiable through log π(v); and statistics
    (sampler_mass: Σ_V q summed over tokens, the share of the sampler's mass the candidates cover)."""
    k = config.score_centering_k
    lp_v = sampled.candidate_lp[:, :k]
    mu_v = sampled.sampler.logprobs[:, :k]
    with torch.no_grad():
        q = mu_v.exp()  # 0 at padding
        p = lp_v.detach().exp()
        ratio = p / q.clamp(min=torch.finfo(q.dtype).tiny)
        a = advantage[:, None]
        g_v = _keep_factor(config, ratio, a, p, q)
        rest_q, rest_p = (1 - q.sum(-1)).clamp(min=0.0), (1 - p.sum(-1)).clamp(min=0.0)
        tail_ratio = rest_p / rest_q.clamp(min=torch.finfo(q.dtype).tiny)
        g_tail = _keep_factor(config, tail_ratio[:, None], a, rest_p[:, None], rest_q[:, None])
        g_tail = g_tail.masked_fill((rest_q <= 1e-6)[:, None], 0.0)  # no tail: the candidates hold the whole support
        residual = torch.where(q > 0, p * (g_v - g_tail), torch.zeros_like(p))
        stats = {"sampler_mass": q.sum(), "centered_tokens": (q > 0).any(1).sum().float()}
    correction = (residual * lp_v.masked_fill(q <= 0, 0.0)).sum(-1)
    return correction, stats


def sampling_metrics(values: dict[str, float]) -> dict[str, float]:
    """The step's candidate statistics (sampled_logprobs, score centering) as per-token means."""
    metrics = {}
    if values.get("support_tokens"):
        incomplete = values.get("support_incomplete", 0.0) / values["support_tokens"]
        metrics["sampling/support_incomplete"] = incomplete
        metrics["sampling/support_size"] = values.get("support_size", 0.0) / values["support_tokens"]
        if incomplete > SUPPORT_INCOMPLETE_WARN:
            logger.warning(
                "%.0f%% of the trained tokens are out of the loss: their support filled every candidate the engine "
                "returned (or lacked the sampled token). The engine's top log-probs must be processed ones "
                "(vLLM --logprobs-mode processed_logprobs) and cover rollout.top_k + 8 tokens.",
                100 * incomplete,
            )
    if values.get("centered_tokens"):
        metrics["sampling/sampler_mass"] = values.get("sampler_mass", 0.0) / values["centered_tokens"]
    return metrics


# ------------------------------------------------------------------ objective


def policy_loss(
    lp: Tensor,
    behaviour: Tensor,
    advantage: Tensor,
    weight: Tensor,
    seq: Tensor,
    n_seq: int,
    seq_len: Tensor,
    config,
    forced: Tensor | None = None,
    sampled: SampledLogProbs | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """The loss over a micro-batch's trained tokens.

    lp, behaviour, advantage, weight, seq: [N] per token (seq: the token's sequence index in
    0..n_seq-1); seq_len: [n_seq] trained tokens per sequence; config: the RLLossConfig; forced: [N] bool or None,
    tokens whose behaviour log-prob is a forcing probability (think-skip): their ratio is an intended importance
    weight, so it stays out of the sampler-mismatch statistics and the sequence mask. sampled: sampled_logprobs'
    result when the tokens came with the sampler's candidates (lp is then sampled.lp): tokens it marks invalid are
    left out of the loss, and loss.score_centering uses its candidates.
    Returns the loss (to backward) and summed statistics as 0-d tensors (no host sync).
    """
    centering = getattr(config, "score_centering", "off") != "off"
    if centering and sampled is None:
        raise ValueError("loss.score_centering needs the sampler's candidates (sampled_logprobs)")
    candidate_stats = {}
    if sampled is not None:
        weight = weight * sampled.valid
        if sampled.sampler.support:
            with torch.no_grad():
                counted = sampled.sampler.present
                candidate_stats = {
                    "support_tokens": counted.sum().float(),
                    "support_incomplete": (counted & ~sampled.valid).sum().float(),
                    "support_size": (~torch.isneginf(sampled.sampler.logprobs) & counted[:, None]).sum().float(),
                }
    with torch.no_grad():
        log_ratio = lp.detach() - behaviour
        ratio = log_ratio.exp()
        deviation = (ratio - 1).abs()
        seen = ratio  # the sampler-mismatch view of the ratios
        if forced is not None:
            deviation = deviation.masked_fill(forced, 0.0)
            seen = ratio.masked_fill(forced, 1.0)
        seq_deviation = torch.zeros(n_seq, device=lp.device).index_add_(0, seq, deviation) / seq_len
        keep_seq = (
            seq_deviation <= config.seq_mask
            if config.seq_mask > 0
            else torch.ones_like(seq_deviation, dtype=torch.bool)
        )
        s_mask = keep_seq[seq].float()
        stats = {  # tensors: the caller syncs once per step
            "tokens": torch.tensor(float(lp.numel()), device=lp.device),
            "ratio": seen.sum(),
            "ratio_max": seen.max(),
            "abs_ratio_dev": deviation.sum(),
            "mismatch_k3": (
                ratio - 1 - log_ratio if forced is None else (ratio - 1 - log_ratio).masked_fill(forced, 0.0)
            ).sum(),
            "seq_masked": (~keep_seq).sum().float(),
            "sequences": torch.tensor(float(n_seq), device=lp.device),
            **candidate_stats,
        }

    if config.type == "gspo":
        with torch.no_grad():
            seq_log_ratio = torch.zeros(n_seq, device=lp.device).index_add_(0, seq, log_ratio) / seq_len
            s = seq_log_ratio.exp()
            seq_adv = torch.zeros(n_seq, device=lp.device).index_add_(0, seq, advantage) / seq_len
            clipped = ((seq_adv > 0) & (s > 1 + config.gspo_eps)) | ((seq_adv < 0) & (s < 1 - config.gspo_eps))
            coef = ((~clipped).float() * s)[seq] * advantage / seq_len[seq] * s_mask
            stats["clipped"] = clipped[seq].sum().float()
        return -(weight * coef * lp).sum(), stats

    with torch.no_grad():
        capped = ratio.clamp(max=config.is_cap)
        if config.type == "masked_is":
            clipped = ((advantage > 0) & (ratio > 1 + config.eps_high)) | (
                (advantage < 0) & (ratio < 1 - config.eps_low)
            )
            coef = (~clipped).float() * capped * advantage
        elif config.type == "cispo":
            clipped = ratio > config.is_cap
            coef = capped * advantage
        elif config.type == "icepop":
            clipped = (ratio < config.icepop_low) | (ratio > config.icepop_high)
            coef = (~clipped).float() * ratio * advantage
        elif config.type == "ipo":
            clipped = (lp.detach().exp() - behaviour.exp()).abs() > config.ipo_eps
            coef = (~clipped).float() * capped * advantage
        elif config.type == "reinforce":
            if forced is not None and bool(forced.any()):
                raise ValueError("loss.type reinforce cannot train forced tokens (they need an importance weight)")
            clipped = torch.zeros_like(ratio, dtype=torch.bool)
            coef = advantage
        else:
            raise ValueError(f"unknown loss.type {config.type!r}")
        coef = coef * s_mask
        stats["clipped"] = clipped.sum().float()
    if centering:
        correction, centering_stats = centering_correction(sampled, advantage, config)
        stats.update(centering_stats)
        return -(weight * (coef * lp - s_mask * advantage * correction)).sum(), stats
    return -(weight * coef * lp).sum(), stats


# Buckets of the sampler-mismatch breakdown: the token's position in the completion (tokens after the prompt, tool
# results included: [0, 1024), [1024, 4096), 4096 and beyond), then the policy version that sampled it.
POSITION_BOUNDS = (1024, 4096)
MISMATCH_BUCKETS = ("pos_0_1k", "pos_1k_4k", "pos_4k_plus", "current", "stale")


def mismatch_by_bucket(
    lp: Tensor, behaviour: Tensor, position: Tensor, sampled_by: Tensor, forced: Tensor | None = None
) -> dict[str, Tensor]:
    """policy_loss's mismatch statistics (k3 = ρ - 1 - log ρ and |ρ - 1|) summed per bucket, with the bucket's token
    count, as 0-d tensors (no host sync): mismatch_k3/<bucket>, abs_ratio_dev/<bucket>, bucket_tokens/<bucket>.

    position: [N] long, the token's index in the completion; sampled_by: [N] long, 0 = sampled by the policy being
    trained (the sampler's error alone), 1 = by an older version (staleness on top), -1 = unknown (counted in the
    position buckets only). Tokens in `forced` (think-skip) carry an intended importance weight and are left out.
    """
    with torch.no_grad():
        log_ratio = lp.detach() - behaviour
        ratio = log_ratio.exp()
        keep = torch.ones_like(ratio) if forced is None else (~forced).float()
        values = torch.stack([(ratio - 1 - log_ratio) * keep, (ratio - 1).abs() * keep, keep])  # [3, N]
        bounds = torch.tensor(POSITION_BOUNDS, device=position.device, dtype=position.dtype)
        by_position = torch.bucketize(position, bounds, right=True)  # 0, 1, 2
        # columns 3 / 4: current / stale; column 5 collects the unknown ones and is dropped
        by_version = torch.where(sampled_by < 0, 5, 3 + sampled_by.clamp(max=1))
        sums = values.new_zeros(3, 6).index_add_(1, by_position, values).index_add_(1, by_version, values)
    stats = {}
    for i, name in enumerate(MISMATCH_BUCKETS):
        stats[f"mismatch_k3/{name}"] = sums[0, i]
        stats[f"abs_ratio_dev/{name}"] = sums[1, i]
        stats[f"bucket_tokens/{name}"] = sums[2, i]
    return stats


def mismatch_metrics(values: dict[str, float]) -> dict[str, float]:
    """Per-token means of a step's summed mismatch_by_bucket statistics, for the buckets that had tokens, and the
    share of tokens sampled by an older policy version."""
    metrics = {}
    for name in MISMATCH_BUCKETS:
        n = values.get(f"bucket_tokens/{name}", 0.0)
        if n > 0:
            metrics[f"policy/mismatch_k3/{name}"] = values[f"mismatch_k3/{name}"] / n
            metrics[f"policy/abs_ratio_dev/{name}"] = values[f"abs_ratio_dev/{name}"] / n
    versioned = values.get("bucket_tokens/current", 0.0) + values.get("bucket_tokens/stale", 0.0)
    if versioned > 0:
        metrics["policy/stale_tokens"] = values.get("bucket_tokens/stale", 0.0) / versioned
    return metrics


# ------------------------------------------------------- advantages & weights


def group_rewards(scored, pass_k: int = 1) -> list[float]:
    """The rewards a group's scored rollouts are compared on: their own (pass_k 1), or the pass@k rewards of
    evalstats.pass_at_k_rewards (loss.pass_k), with k capped at the number of scored rollouts."""
    rewards = [t.reward for t in scored]
    if pass_k <= 1 or not rewards:
        return rewards
    return pass_at_k_rewards(rewards, min(pass_k, len(rewards)))


def assign_advantages(
    groups, std: str = "batch", eps: float = 1e-6, reduce=None, baseline: str = "mean", pass_k: int = 1
) -> dict[str, float]:
    """Group-relative advantages in place: reward minus the group's baseline, divided by the batch's std of those
    values ("batch"), the group's ("group"), or not. The baseline is the mean reward of the group's scored rollouts
    ("mean") or of the other scored rollouts ("leave_one_out": n / (n - 1) x the former; a rollout alone in its
    group has no baseline and gets 0). `reduce` sums a list of floats over data-parallel ranks (the batch spans them).
    pass_k > 1 (loss.pass_k): the rewards are first replaced by their pass@k rewards (group_rewards).
    """
    centered = []
    for group in groups:
        scored = [t for t in group if t.scored]
        if not scored:
            continue
        rewards = dict(zip(map(id, scored), group_rewards(scored, pass_k)))
        mean = sum(rewards.values()) / len(scored)
        scale = len(scored) / (len(scored) - 1) if baseline == "leave_one_out" and len(scored) > 1 else 1.0
        for t in group:
            t.advantage = (rewards[id(t)] - mean) * scale if t.scored else 0.0
        if std == "group":
            sd = math.sqrt(sum(t.advantage**2 for t in scored) / len(scored))
            for t in scored:
                t.advantage /= sd + eps
        centered += [t.advantage for t in scored]
    squares, count, absolute = sum(a * a for a in centered), len(centered), sum(abs(a) for a in centered)
    if reduce is not None:
        squares, count, absolute = reduce([squares, count, absolute])
    if std == "batch" and count:
        sd = math.sqrt(squares / count)
        for group in groups:
            for t in group:
                t.advantage /= sd + eps
    return {"advantage_abs": absolute / max(1, count)}


def token_weights(groups, aggregation: str, budget: int, sequence_level: bool = False, reduce=None) -> dict[int, float]:
    """Per-trajectory weight of each of its trained tokens, keyed by id(trajectory), so that
    micro-batches (and data-parallel ranks, through `reduce`) accumulate exactly the full
    batch's loss:
      prompt    1 / (prompts x the group's trained tokens)     token-mean per prompt, mean over prompts
      token     1 / (all trained tokens)                        DAPO
      constant  1 / (trained sequences x completion budget)     Dr. GRPO
    `sequence_level` (GSPO): 1 / (prompts x the group's trained sequences) per sequence.
    """
    trained = [[t for t in g if t.trained and t.sampled_tokens] for g in groups]
    trained = [g for g in trained if g]
    totals = [len(trained), sum(t.sampled_tokens for g in trained for t in g), sum(len(g) for g in trained)]
    prompts, total_tokens, sequences = reduce(totals) if reduce is not None else totals
    weights: dict[int, float] = {}
    for g in trained:
        group_tokens = sum(t.sampled_tokens for t in g)
        for t in g:
            if sequence_level:
                weights[id(t)] = 1.0 / (prompts * len(g))
            elif aggregation == "prompt":
                weights[id(t)] = 1.0 / (prompts * group_tokens)
            elif aggregation == "token":
                weights[id(t)] = 1.0 / total_tokens
            else:
                weights[id(t)] = 1.0 / (sequences * budget)
    return weights


def observation_weights(trajectories, weight: float, reduce=None) -> dict[int, float]:
    """The observation loss (ECHO, arXiv:2605.24517; loss.observation_weight): per trajectory with observations,
    keyed by id(trajectory), the weight of each of its targets' NLL,
        λ / (trajectories with observations, over the ranks x the trajectory's observation length |O|),
    so that the step adds λ x the mean over those trajectories of (summed target NLL / |O|): each trajectory's
    observation term is normalised by its observation length (targets or not), as in the paper, and averaged over
    trajectories like the policy loss over prompts. Empty when the loss is off."""
    if weight <= 0:
        return {}
    observed = [t for t in trajectories if t.observation_tokens]
    count = len(observed)
    if reduce is not None:
        count = reduce([count])[0]
    return {id(t): weight / (count * t.observation_tokens) for t in observed}
