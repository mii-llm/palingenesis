# Optimizers

*The right optimizer depends on your memory budget and GPU count.*

---

## Decision tree

```
Memory tight (single GPU, ≤40 GB)?
  └── Yes → lion8bit + gradient_release
        (4 bytes/param, 0 grad memory)

Memory abundant (multi-GPU FSDP)?
  └── Yes → muon + mona + hyperball
        (fastest convergence, ~2-3× AdamW speed)

Maximum simplicity?
  └── adamw (reliable baseline, 16 bytes/param)
```

---

## Lion 8-bit

The memory king. Sign-based updates (like Muon but simpler), one momentum buffer, 8-bit quantized. Total optimizer memory: ~1 GB for a 4B model.

```yaml
train:
  optimizer: lion8bit
  learning_rate: 1.5e-5   # applied as-is (no hidden scaling)
```

Lion's update is `sign(β₁·m + (1-β₁)·g)` — uniform magnitude across all dimensions. Works because the loss landscape of Transformers is approximately sign-symmetric.

!!! warning "Lion wants a *lower* LR than AdamW"
    Because every update element has magnitude 1, the effective step is larger than AdamW's at the same LR. The [Lion paper](https://arxiv.org/abs/2302.06675) recommends a LR **3–10× smaller** than AdamW's, paired with a 3–10× *larger* weight decay. The configured `learning_rate` is exactly what the optimizer receives — there is no internal adjustment.

Composes with: gradient_release ✓, Hyperball ✓, AdaGC ✓, EMA ✓

---

## Muon

Matrix orthogonalization via Newton-Schulz iteration. Treats entire weight matrices as geometric units instead of independent scalars. 1.5-2× faster convergence than AdamW.

```yaml
train:
  optimizer: muon
  learning_rate: 1.5e-5   # Internal 10× scaling applied automatically
```

Does NOT compose with: gradient_release ✗ (needs full gradient matrix for polar decomposition)

---

## Hyperball

Not an optimizer — a wrapper. Applies to any base optimizer. Projects weight matrices back to their initial Frobenius norm after each step. Zero memory, zero compute.

The theory: in prenorm Transformers, weight matrices between normalization layers are scale-invariant. The loss is `L(cW) = L(W)` for any scalar c. Weight decay's real purpose is controlling the *angular* learning rate. Hyperball makes this explicit.

```yaml
train:
  hyperball: true   # Works with any optimizer
```

---

## MONA

Curvature-aware acceleration. Augments gradients with an EMA of gradient *differences* before the optimizer processes them:

```
D_k = G_k - G_{k-1}              (gradient difference ≈ H·Δθ)
A_k = β_a·A_{k-1} + (1-β_a)·D_k  (acceleration buffer)
G̃_k = G_k + α·A_k                (augmented gradient)
```

Near sharp minima, `‖D_k‖` is large → acceleration pushes toward flatter regions. Near flat regions, acceleration is small → stable convergence.

```yaml
train:
  mona: true
  mona_beta_a: 0.975   # Higher for larger models (0.99 for 68B)
  mona_lite: true       # bf16 buffers + streaming (75% less overhead)
```

---

## SAGE

Specialized for embedding layers. Regular sign-based optimizers (Lion, Muon) fail on embeddings because embedding gradients are sparse and high-variance (Zipfian token frequency).

SAGE adds an O(d) adaptive damper that scales each embedding dimension by its relative "loudness" — loud dimensions get damped, quiet ones pass through at full magnitude. Provably bounded ≤ 1.0.

Available as a standalone optimizer for embedding-specific use, but palingenesis handles this automatically in hybrid mode.

---

## Schedulers

### power_decay (recommended)

`η(t) = η_peak · (1 - progress)^γ` where γ = 4.

Provably optimal when model capacity exceeds β > 3 (always true for LLMs). Cosine saturates — power-decay doesn't.

### wsd (warmup-stable-decay)

Maintains peak LR for 80% of post-warmup training, then power-decays. Best for long runs where you want anytime stopping during the stable phase.

### cosine

The legacy default. Still works. Slightly suboptimal. Use power_decay instead.


---

*For the full explanation of how these optimizers compose and why each exists, see the [Single GPU guide](../guides/single-gpu.md) (memory-constrained stack) and [Multi-GPU guide](../guides/multi-gpu.md) (convergence-optimal stack).*
