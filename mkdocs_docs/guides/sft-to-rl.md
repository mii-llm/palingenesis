# SFT - RL Transition

*The checkpoint you hand to GRPO determines whether RL succeeds or fails. Most people hand it the wrong one.*

---

## The mechanism of failure

Here's what happens inside the model during SFT:

In the beginning, the output distribution is broad. Given a prompt, the model assigns non-trivial probability to many possible continuations. Some are correct, some are wrong, but the distribution has *entropy* — there's diversity in what it might say.

As SFT progresses, the model becomes confident on training data. It assigns 90%+ probability to the "correct" continuation. Loss decreases. Benchmarks improve. Everything looks great.

But something is being destroyed: the *width* of the distribution. The model is collapsing from "considers many options" to "always produces one answer." The entropy is dropping.

Now you hand this checkpoint to GRPO. GRPO works by sampling K rollouts, scoring them, and reinforcing the better ones relative to the worse ones. But if the model is already deterministic — if all K rollouts are essentially the same — there's no variance in rewards. The "advantage" of one rollout over another is zero. The gradient vanishes. RL learns nothing.

This isn't a GRPO bug. It's a *property of the SFT checkpoint*. You've burned the exploration capacity that RL needs.

---

## The empirical evidence

From Stanford (June 2026, arxiv:2606.18487):

> On Qwen2.5-Coder-3B, pre-RL pass@1 rises monotonically with SFT depth, but peak GRPO pass@10 *falls* from 0.806 to 0.481. The highest pass@1 checkpoint loses to shallower counterparts in every seed.

From HKUST (June 2026, arxiv:2606.09932):

> Models from excessive SFT produce over-confident token distributions and exhibit sharp parameter landscapes. Large gradient norms during RL do not translate into effective parameter movement.

The conclusion: **the best SFT checkpoint for RL is NOT the one with lowest SFT loss.**

---

## Detection: entropy monitoring

Palingenesis tracks output entropy during training when you enable it:

```yaml
logging:
  rl_readiness: true
  rl_entropy_floor: 1.5
```

Every logging step, it samples random valid tokens from the current batch, computes the entropy of the model's output distribution, and records it as `health/output_entropy`. This works with every loss path — including chunked and Cut Cross-Entropy, where the full logit tensor is never materialized (sampled positions are projected through the LM head instead), so enabling it costs essentially nothing.

When the moving average drops below `rl_entropy_floor`, you get a warning:

```
⚠️  RL-READINESS WARNING: Output entropy collapsed to 1.23 (floor=1.50).
If you plan RL after SFT, consider stopping NOW.
```

This is your signal. The model is becoming too confident. If you continue, the checkpoint will be useless for RL.

---

## The rules

### 1. Stop early

2-3 epochs is usually enough. The "best" epoch for RL is typically epoch 1.5-2.5 — after the model has learned format and basic skills, but before it has memorized specific solutions.

### 2. Keep SFT and RL data disjoint

From April 2026 (arxiv:2604.13515): when the same problems appear in both SFT and GRPO training sets, interference patterns emerge. The model learns conflicting signals — "imitate this solution" (SFT) vs "explore beyond this solution" (RL). Use separate data splits.

### 3. Preserve diversity actively

Don't just stop early — actively maintain entropy during SFT:

```yaml
train:
  ema: true              # EMA averages over history, smoothing out confidence
  ema_decay: 0.999
  base_merge: true       # Periodically pull back toward the diverse base model
  base_merge_ratio: 0.1
  base_merge_every: 200

plugins:
  deft: false            # REQUIRED off — DEFT takes precedence and silently
  pre_rl: true           #   disables pre_rl (one objective per run)
  pre_rl_entropy_coeff: 0.1
  pre_rl_kl_coeff: 0.5

memory:
  chunked_loss: false    # REQUIRED off — pre_rl needs the full logits; the
                         #   chunked-CE path also takes precedence over it
```

The `pre_rl` plugin splits tokens by entropy and KL from a stale reference snapshot. "Safe" tokens (low entropy, low KL) get normal cross-entropy. "Unsafe" tokens — where the distribution is still wide or already drifting — are excluded from imitation and instead get:

- **Entropy bonus**: `+H(p_θ)` — directly rewards output diversity
- **KL anchor**: `-KL(p_θ || p_ref)` — penalizes drift from the base model's distribution

Together, these slow the entropy collapse while still allowing the model to learn the task format.

!!! warning "pre_rl is one objective among alternatives, not an add-on"
    The training loop selects **exactly one** loss objective per run, in priority order: chunked DEFT → chunked CE → CADFT → DEFT → DFT → InfoSFT → pre_rl → plain CE. Setting `pre_rl: true` while `deft: true` or `memory.chunked_loss: true` does **nothing** — the earlier branch wins silently. Choosing pre_rl means trading away DEFT's token weighting and the chunked-loss memory savings: full logits are materialized (batch × seq × vocab, plus float32 copies inside the loss and a cached reference snapshot — roughly 15–20 GB extra at batch 4 × seq 4096 × 150k vocab).

### 4. Use the EMA checkpoint for RL

If you have `ema: true`, the EMA checkpoint is a time-average over all training steps. It's less confident than the final model (which overfit to the last batches) and retains more of the early diversity. Use `output/final/` (which has EMA applied) rather than periodic checkpoints.

---

## Entropy thresholds

These thresholds are calibrated from the Stanford and HKUST papers on 3B-7B models:

| Output entropy | Interpretation | What to do |
|:---:|---|---|
| > 3.0 | Healthy distribution. Model considers many options. | Continue training normally. |
| 2.0 – 3.0 | Starting to narrow. Still viable for RL. | Monitor closely. Consider stopping within 100 steps. |
| 1.0 – 2.0 | Concerning. GRPO will struggle. | Stop SFT. Use the EMA checkpoint. Enable `pre_rl` if continuing. |
| < 1.0 | Collapsed. Model is near-deterministic. | Do NOT start RL. Recovery needed. |

---

## Recovery (if you've already over-trained)

If entropy has collapsed but you need an RL-viable checkpoint:

**Option 1: Model fusion.** Interpolate the overtrained model with the base:

```python
# θ_new = 0.7 × θ_sft + 0.3 × θ_base
```

This partially restores the base model's diverse distribution. SLERP (spherical interpolation) is better than LERP here — it preserves weight norms. Palingenesis's `base_merge` feature does exactly this.

**Option 2: Use an earlier checkpoint.** The periodic checkpoints saved during training each have different entropy levels. Pick the one closest to your target entropy (2.0-3.0 range) rather than the "best loss" one.

**Option 3: Re-train with `pre_rl: true`.** Shorter. More principled. But requires another training run.

---

## The full config

```yaml title="For RL-bound SFT"
train:
  epochs: 2
  ema: true
  ema_decay: 0.999
  base_merge: true
  base_merge_ratio: 0.1
  base_merge_every: 200
  base_merge_method: slerp

plugins:
  deft: false            # pre_rl replaces DEFT — they cannot combine
  pre_rl: true
  pre_rl_entropy_coeff: 0.1
  pre_rl_kl_coeff: 0.5

memory:
  chunked_loss: false    # pre_rl requires full logits

logging:
  rl_readiness: true
  rl_entropy_floor: 1.5
```

This produces a checkpoint that: (1) has learned the task format on the tokens that are safe to imitate, (2) retains enough output diversity for GRPO to explore, (3) is anchored near the base model's distribution so RL doesn't drift too far.
