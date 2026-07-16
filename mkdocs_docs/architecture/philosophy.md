# Philosophy

*The decisions behind the defaults, and why we disagree with the mainstream.*

---

## Full fine-tune over LoRA

The prevailing wisdom: use LoRA (or QLoRA) to fine-tune cheaply. Train only a low-rank delta. Save memory. Deploy fast.

We disagree — not because LoRA is bad, but because its premise is outdated.

LoRA was essential when fine-tuning a 7B model required 80+ GB of GPU memory. The alternative was "don't fine-tune at all." In that world, LoRA was a breakthrough.

But the memory problem is solvable without compromising representation capacity. Gradient release eliminates the gradient buffer. Lion 8-bit reduces optimizer state 8×. Selective checkpointing halves activation memory. The result: a 4B full fine-tune in 15 GB. A 7B in 24 GB. On hardware that costs $0.50/hour on cloud.

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

The answers, backed by papers:

| Decision | Standard default | Our default | Why |
|----------|-----------------|-------------|-----|
| Loss | Cross-entropy | **DEFT** | math-reasoning gains per original paper (not reproduced); parameter-free; subsumes CE |
| Scheduler | Cosine | **Power-decay** | Provably optimal when β > 3 (always true for LLMs) |
| Gradient handling | Store all | **Release immediately** | Saves 2× param memory, zero accuracy cost |
| Optimizer | AdamW (16 B/param) | **Lion8bit** (4 B/param) or **Muon** (fastest) | 4× less memory or 2× faster convergence |
| Weight update | Standard step | **Hyperball projection** | 20-30% speedup by making scale-invariance explicit |
| Checkpointing | Save last | **Save best + purge old** | The last checkpoint is often overtrained |
| Gradient clipping | Global norm | **Per-tensor AdaGC** | Handles heterogeneous gradient scales correctly |

You can override any of these. But the defaults are the result of reading 352 papers and running ablations. They're not arbitrary.

---

## Why not RL (yet)?

Palingenesis is an SFT tool. It doesn't do GRPO, DPO, PPO, or any reinforcement learning. This is a deliberate scope decision.

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

Palingenesis's `prepare` command finds this sweet spot automatically. 10 minutes of scoring saves hours of wasted training.

---

## On simplicity

The codebase has one training loop. One config format. One checkpoint format. One CLI.

There's no plugin system that requires writing adapters. No callback hooks that create implicit control flow. No "trainer" class hierarchy that forces you to understand inheritance before training a model.

The architecture is: you write a YAML, you run a command, you get a model. Everything else is internal — optimized, tested, but internal.

If you need to understand the internals (and eventually you will, if you're pushing boundaries), every module has a docstring citing its paper. The code is the documentation for the implementation. This page is the documentation for the *decisions*.
