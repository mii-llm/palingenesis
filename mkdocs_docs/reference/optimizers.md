# Optimizers

*The right optimizer depends on your memory budget and GPU count.*

All of these are set under `train:` (`optimizer`, `hyperball`, `mona`, `lr_scheduler`); every option is in the [configuration reference](configuration.md#train).

---

## Which one

| Situation | Choose | Why |
|---|---|---|
| Memory tight (single GPU, ≤40 GB) | `lion8bit` + `memory.gradient_release` | 4 bytes/param, 0 gradient memory |
| Memory abundant (multi-GPU FSDP) | `adamw`, or `muon` (+ `hyperball`) to experiment | Muon/Hyperball speedups are reported for pretraining, not measured here for fine-tuning |
| Maximum simplicity | `adamw` | reliable baseline, 16 bytes/param |

---

## Lion 8-bit

The memory king. Sign-based updates (like Muon but simpler), one momentum buffer, 8-bit quantized. Total optimizer memory: ~1 GB for a 4B model.

```yaml
train:
  optimizer: lion8bit
  learning_rate: 3.0e-6   # applied as-is; ~3-10x below an AdamW fine-tuning LR
```

Lion's update is $\operatorname{sign}\big(\beta_1 m + (1-\beta_1)\,g\big)$, with $m$ the momentum and $g$ the gradient — uniform magnitude across all dimensions. Works because the loss landscape of Transformers is approximately sign-symmetric.

!!! warning "Lion wants a *lower* LR than AdamW"
    Because every update element has magnitude 1, the effective step is larger than AdamW's at the same LR. The [Lion paper](https://arxiv.org/abs/2302.06675) recommends a LR **3–10× smaller** than AdamW's, paired with a 3–10× *larger* weight decay. The configured `learning_rate` is exactly what the optimizer receives — there is no internal adjustment.

Composes with: gradient_release ✓, Hyperball ✓, AdaGC ✓, EMA ✓

---

## Muon

Matrix orthogonalization via Newton-Schulz iteration. Treats entire weight matrices as geometric units instead of independent scalars. Reported faster than AdamW in pretraining; not benchmarked here for fine-tuning. Hidden 2-D matrices use Muon; embeddings, the output head, norms, biases and stacked 3-D expert weights use AdamW.

```yaml
train:
  optimizer: muon
  learning_rate: 1.5e-5   # Internal 10× scaling applied automatically
```

Does NOT compose with: gradient_release ✗ (needs full gradient matrix for polar decomposition)

---

## Hyperball

Not an optimizer, but a wrapper around any base optimizer (AdamW, Lion, Muon, 8-bit variants). It implements Algorithm 1 of [arXiv:2606.16899](https://arxiv.org/abs/2606.16899).

The theory: in prenorm Transformers, weight matrices between normalization layers are scale-invariant, so $L(cW) = L(W)$ for any scalar $c > 0$. Weight decay's real role is to control the *angular* learning rate. Hyperball makes that explicit. Each attention/MLP matrix keeps its initial Frobenius norm $R$, and every update moves it by a fixed angle:

$$
W \leftarrow R \cdot \operatorname{normalize}\!\big(W - \eta R \operatorname{normalize}(u)\big), \qquad \operatorname{normalize}(X) = \frac{X}{\lVert X \rVert_F}
$$

where $u$ is the base optimizer's update direction (Adam: $\hat m / (\sqrt{\hat v} + \epsilon)$, Lion: the sign, Muon: the orthogonalised momentum), $R$ the matrix's norm at the first step and $\eta$ the angular step (`hyperball_lr`).
{: .pg-where }

- **Angular step $\eta$.** The gradient scale and the base optimizer's own LR cancel out: the step is always $\eta R$ before renormalisation. $\eta$ follows the LR schedule.
- **What is constrained.** Only attention and MLP matrices. Embeddings, norms, biases and the output head train on the base optimizer as usual, including weight decay. The constraint replaces weight decay on the constrained matrices.
- **How the direction is obtained.** Exactly, for any base optimizer: with weight decay switched off, the base step moves $W$ by $-\text{lr}\cdot u$, so $\operatorname{normalize}(u) = -\Delta W / \lVert \Delta W \rVert$.
- **Memory.** Matrices are stepped in buckets of at most 1 GiB, so the extra memory is one bucket snapshot, not a copy of the model.

```yaml
train:
  hyperball: true
  hyperball_lr: 0.0   # angular step η; 0 = per matrix, the base optimizer's first relative step
```

**Choosing `hyperball_lr`:**

- `0` (default) calibrates each matrix to the relative step its base optimizer's first update with a non-zero learning rate made (the first warmup step has lr 0 and is skipped). For Adam and Lion that is $\texttt{learning\_rate} / \operatorname{rms}(W)$. The calibration is not stored in checkpoints: a resumed run recalibrates from its first step.
- A positive value is the paper's single angular step for all matrices.
- For scale: AdamW at 2e-5 on matrices with RMS 0.02 moves them by about 1e-3 per step.

!!! warning "Use fp32 weights"
    Angular steps of 1e-3 or less partly round away in bf16 weights (`model.torch_dtype: float32`).

---

## MONA

Curvature-aware acceleration. Augments gradients with an EMA of gradient *differences* before the optimizer processes them:

$$
\begin{aligned}
D_k &= G_k - G_{k-1} && \text{gradient difference} \approx H\,\Delta\theta \\
A_k &= \beta_a A_{k-1} + (1-\beta_a)\, D_k && \text{acceleration buffer} \\
\tilde G_k &= G_k + \alpha A_k && \text{augmented gradient}
\end{aligned}
$$

where $G_k$ is the gradient at step $k$, $H$ the Hessian, $\Delta\theta$ the last parameter change, $\beta_a$ is `mona_beta_a` and $\alpha$ the acceleration coefficient, $\alpha = -1 / \big(2(1-\beta_a)\big)$ (the paper's rule; not a config option).
{: .pg-where }

Near sharp minima, $\lVert D_k \rVert$ is large → acceleration pushes toward flatter regions. Near flat regions, acceleration is small → stable convergence.

```yaml
train:
  mona: true
  mona_beta_a: 0.975   # Higher for larger models (0.99 for 68B)
  mona_lite: true       # bf16 buffers + streaming (75% less overhead)
```

---

## SAGE

Specialized for embedding layers. Regular sign-based optimizers (Lion, Muon) fail on embeddings because embedding gradients are sparse and high-variance (Zipfian token frequency).

SAGE adds an $O(d)$ adaptive damper that scales each embedding dimension by its relative "loudness" — loud dimensions get damped, quiet ones pass through at full magnitude. Provably bounded $\le 1$.

Available as a standalone optimizer class (`palingenesis.optim.SAGE`); no training config selects it.

---

## Schedulers

### power_decay

$$
\eta(p) = \eta_{\min} + (\eta_{\text{peak}} - \eta_{\min}) \cdot (1 - p)^{\gamma}, \qquad \gamma = 4
$$

where $p \in [0, 1]$ is the progress through the steps after warmup, $\eta_{\text{peak}}$ is `train.learning_rate` and $\eta_{\min}$ is `train.min_learning_rate`.
{: .pg-where }

Motivated by a functional-scaling-law analysis (arXiv:2602.06797) that finds power decay better than cosine in its model of pretraining; not benchmarked here for fine-tuning.

### wsd (warmup-stable-decay)

Maintains peak LR for 80% of post-warmup training, then power-decays. Best for long runs where you want anytime stopping during the stable phase.

### cosine

The `train.lr_scheduler` default: warmup, then cosine decay to `min_learning_rate`. The baseline every other choice should be compared against.


---

*For the full explanation of how these optimizers compose and why each exists, see the [Single GPU guide](../guides/single-gpu.md) (memory-constrained stack) and [Multi-GPU guide](../guides/multi-gpu.md) (convergence-optimal stack).*
