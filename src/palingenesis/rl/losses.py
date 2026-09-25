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

S is the sequence mask (Trust Region Masking): a trajectory whose mean |ρ - 1| exceeds
loss.seq_mask is dropped whole, since token-level clipping cannot bound a long sequence's
divergence. All coefficients are detached: the loss's value is not an objective, its
gradient is.
"""

import math

import torch
from torch import Tensor, nn

from palingenesis.opd.fused_rkl import Head, unwrap
from palingenesis.opd.losses import SLICE_ELEMENTS

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
        for a in range(0, n, rows):
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
        for a in range(0, h.shape[0], rows):
            g = grad_lp[a : a + rows].float()
            if not g.any():
                continue
            b = a + g.shape[0]
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
    def forward(ctx, hidden, targets, head, rows, entropy, *params):
        n = hidden.shape[0]
        lp = torch.empty(n, dtype=torch.float32, device=hidden.device)
        ent = torch.empty(n if entropy else 0, dtype=torch.float32, device=hidden.device)
        with torch.no_grad():
            for a in range(0, n, rows):
                logits = head(hidden[a : a + rows]).float()
                lse = torch.logsumexp(logits, -1)
                lp[a : a + rows] = logits.gather(1, targets[a : a + rows, None]).squeeze(1) - lse
                if entropy:
                    ent[a : a + rows] = lse - (torch.softmax(logits, -1) * logits).sum(-1)
        ctx.save_for_backward(hidden, targets)
        ctx.head, ctx.rows, ctx.params = head, rows, params
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
                logits = ctx.head(h).float()
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
        return grad_hidden, None, None, None, None, *out_params


def target_logprobs(
    hidden: Tensor, head: nn.Module, targets: Tensor, entropy: bool = False, low: LowPrecisionWeight | None = None
) -> tuple[Tensor, Tensor]:
    """(log π(targets), entropies) for hidden-state rows [N, H]: differentiable log-probs with
    fp32 logits, a slice of rows at a time (and, with `entropy`, the policy's entropy at each
    row, for logging). Linear heads take the fused path: bf16 GEMMs on the GPU with fp32
    accumulation, one cast of the weight per optimizer step (`low`)."""
    rows = max(1, SLICE_ELEMENTS // head.weight.shape[0])
    linear = unwrap(head)
    if linear is not None:
        w_low = (low or LowPrecisionWeight()).get(linear.weight)
        with torch.autocast(hidden.device.type, enabled=False):
            return _LinearTargetLogProbs.apply(hidden, linear.weight, targets, linear, w_low, rows, entropy)
    return _TargetLogProbs.apply(hidden, targets, head, rows, entropy, *head.parameters())


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
) -> tuple[Tensor, dict[str, float]]:
    """The loss over a micro-batch's trained tokens.

    lp, behaviour, advantage, weight, seq: [N] per token (seq: the token's sequence index in
    0..n_seq-1); seq_len: [n_seq] trained tokens per sequence; config: the RLLossConfig.
    Returns the loss (to backward) and summed statistics as 0-d tensors (no host sync).
    """
    with torch.no_grad():
        log_ratio = lp.detach() - behaviour
        ratio = log_ratio.exp()
        deviation = (ratio - 1).abs()
        seq_deviation = torch.zeros(n_seq, device=lp.device).index_add_(0, seq, deviation) / seq_len
        keep_seq = (
            seq_deviation <= config.seq_mask
            if config.seq_mask > 0
            else torch.ones_like(seq_deviation, dtype=torch.bool)
        )
        s_mask = keep_seq[seq].float()
        stats = {  # tensors: the caller syncs once per step
            "tokens": torch.tensor(float(lp.numel()), device=lp.device),
            "ratio": ratio.sum(),
            "ratio_max": ratio.max(),
            "abs_ratio_dev": deviation.sum(),
            "mismatch_k3": (ratio - 1 - log_ratio).sum(),
            "seq_masked": (~keep_seq).sum().float(),
            "sequences": torch.tensor(float(n_seq), device=lp.device),
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
        else:
            raise ValueError(f"unknown loss.type {config.type!r}")
        coef = coef * s_mask
        stats["clipped"] = clipped.sum().float()
    return -(weight * coef * lp).sum(), stats


# ------------------------------------------------------- advantages & weights


def assign_advantages(groups, std: str = "batch", eps: float = 1e-6, reduce=None) -> dict[str, float]:
    """Group-relative advantages in place: reward minus the mean of the group's scored rollouts,
    divided by the batch's std of those values ("batch"), the group's ("group"), or not.
    `reduce` sums a list of floats over data-parallel ranks (the batch spans them).
    """
    centered = []
    for group in groups:
        scored = [t for t in group if t.scored]
        if not scored:
            continue
        mean = sum(t.reward for t in scored) / len(scored)
        for t in group:
            t.advantage = t.reward - mean if t.scored else 0.0
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
