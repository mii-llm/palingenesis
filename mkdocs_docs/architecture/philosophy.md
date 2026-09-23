# Philosophy

*The decisions behind the defaults, and why we disagree with the mainstream.*

---

## Full fine-tune over LoRA

The prevailing wisdom: use LoRA (or QLoRA) to fine-tune cheaply. Train only a low-rank delta. Save memory. Deploy fast.

We disagree — not because LoRA is bad, but because its premise is outdated.

LoRA was essential when fine-tuning a 7B model required 80+ GB of GPU memory. The alternative was "don't fine-tune at all." In that world, LoRA was a breakthrough.

But the memory problem is largely solvable without a low-rank adapter. Gradient release eliminates the gradient buffer. Lion 8-bit reduces optimizer state 8×. Selective checkpointing trades a little compute for activation memory. Freezing the linear-attention layers of a hybrid model (`freeze_non_attention`) keeps the trainable set to its attention pathway. Together they fit a Qwen3.5-4B fine-tune (36% of the weights trainable) in about 16 GiB (`pgs profile` on `configs/qwen35_4b/a100_40gb.yaml`).

What you lose with LoRA:
- Rank-limited representations (the delta can only express rank-16 or rank-64 perturbations)
- Interference between adapter and base weights (the "intruder dimension" phenomenon)
- Inability to fully restructure attention patterns for new tasks

What you gain with full fine-tune:
- The complete parameter space is available for learning
- The model can reorganize its internals to suit your task
- No merge step, no adapter management, no serving complexity

Our position: if the hardware supports full fine-tune (and with our optimizations, it almost always does), there's no reason to accept the LoRA compromise.

---

## Opinionated defaults over flexibility

Most training frameworks are "batteries not included." They give you AdamW, cosine schedule, standard cross-entropy, and wish you luck. The message: these are reasonable defaults, but you should probably tune them.

We've spent months reading the literature to answer the question: *what should the defaults actually be?* Not for a generic ML task — specifically for fine-tuning language models in 2025-2026.

The library defaults stay conservative (cross-entropy, cosine schedule, AdamW, global-norm clipping) because they are the baseline everything else must beat. The research-backed options are one line each to turn on, and several shipped configs do:

| Decision | Library default | Option | Status |
|----------|-----------------|--------|--------|
| Loss | Cross-entropy | **DEFT**, DFT, InfoSFT, CADFT | Paper claims (math reasoning, from base models); not reproduced here |
| Scheduler | Cosine | **Power-decay**, WSD | Derived for pretraining; not benchmarked here for fine-tuning |
| Gradient handling | Store all | **Gradient release** | Exact; saves the gradient buffer; needs gradient_accumulation_steps 1 |
| Optimizer | AdamW (16 B/param) | **Lion8bit** (4 B/param), **Muon** | Lion needs a 3-10× lower LR than AdamW |
| Weight update | Standard step | **Hyperball projection** | 20-30% speedup reported in pretraining; experimental for fine-tuning |
| Checkpointing | Save periodically | **Best + final + purge old** | On by default when an eval set is configured |
| Gradient clipping | Global norm | **Per-tensor AdaGC** | Experimental |

You can override any of these. The implementations are checked against their papers' definitions; their benefit on your task is something to measure.

---

## Why not RL (yet)?

Palingenesis is an SFT tool with offline preference optimization ([DPO and variants](../guides/dpo.md)) and [on-policy distillation](../guides/distillation.md). It doesn't do GRPO, PPO or other reward-driven RL. This is a deliberate scope decision.

The research finding that motivates this (CacheRL, June 2026): *"RL provides stability but yields limited gains beyond strong SFT. Data quality and reward design are more important than complex optimization."*

In other words: if your SFT is strong enough, RL adds marginal value. And getting SFT right — proper token weighting, correct masking, optimal scheduling, good data curation — is where 90% of the quality comes from.

That said, palingenesis is *RL-aware*: it monitors output entropy, warns before collapse, and produces checkpoints that preserve the diversity RL needs. It's the ideal SFT stage for an SFT→RL pipeline.

---

## The data thesis

> Train on the samples the model finds *informative*. Not the ones that are easy. Not the ones that are impressive. The ones where the gradient points somewhere useful.

Most practitioners dump their entire dataset into training and hope for the best. This is inefficient at best, harmful at worst.

The research is clear:
- Samples the model already knows (PPL < 1.5) contribute zero gradient signal
- Samples the model can't follow at all (PPL > 500) produce random gradients
- The sweet spot is medium difficulty: informative enough to learn from, tractable enough to generalize

Palingenesis's `prepare` command scores every sample with the model you will train (on exactly the tokens training will use) and selects by difficulty. Compare against a random subset of the same size to see what it buys on your task.

---

## On simplicity

The codebase has one training loop. One config format. One checkpoint format. One CLI.

There's no plugin system that requires writing adapters. No callback hooks that create implicit control flow. No "trainer" class hierarchy that forces you to understand inheritance before training a model.

The architecture is: you write a YAML, you run a command, you get a model. Everything else is internal — optimized, tested, but internal.

If you need to understand the internals (and eventually you will, if you're pushing boundaries), every module has a docstring citing its paper. The code is the documentation for the implementation. This page is the documentation for the *decisions*.
