# Optimizers

*The right optimizer depends on your memory budget and GPU count.*

All of these are set under `train:` (`optimizer`, `hyperball`, `mona`, `lr_scheduler`); every option is in the [configuration reference](configuration.md#train).

---

## Which one

| Situation | Choose | Why |
|---|---|---|
| bf16 weights (`model.torch_dtype: bfloat16`, the default) | `adamw_bf16` | Plain `adamw` on bf16 weights rounds most fine-tuning updates away; `adamw_bf16` keeps them at the same memory |
| Memory tight (single GPU, ≤40 GB) | `lion8bit` + `memory.gradient_release` | 4 bytes/param, 0 gradient memory |
| Memory abundant (multi-GPU FSDP) | `adamw`, or `muon` (+ `hyperball`) to experiment | Muon/Hyperball speedups are reported for pretraining, not measured here for fine-tuning |
| Maximum simplicity | `adamw` | reliable baseline, 16 bytes/param |

---

## AdamW bf16

AdamW for bf16 weights, with bf16 moments and **stochastic rounding** (`palingenesis.bf16_adamw.BF16AdamW`).

```yaml
model:
  torch_dtype: bfloat16
train:
  optimizer: adamw_bf16
```

**The problem it fixes.** A bf16 weight has 8 significant bits: one ulp is $2^{-8}$ to $2^{-7}$ of $|w|$. AdamW at a fine-tuning learning rate of 2e-5 moves a weight of magnitude 0.02 by about $10^{-3}$ of its size per step, below half an ulp, so round-to-nearest returns the old weight. `optimizer: adamw` on bf16 weights (torch's AdamW keeps its moments in the weights' dtype) therefore leaves most weights unchanged at most steps. The usual fix, `model.torch_dtype: float32`, keeps fp32 master weights at twice the memory and with an fp32 residual stream under autocast.

**What it does.** Every element's update is computed in fp32 from the stored bf16 values; only the stores round:

| Stored | Rounding | Why |
|---|---|---|
| weight | stochastic | $\mathbb{E}[\text{stored}] = $ the fp32 result: an update of 1/16 of an ulp moves the weight by one ulp with probability 1/16, so small updates accumulate over steps |
| second moment $v$ | stochastic | a non-negative running average near its own rounding floor, where nearest rounding stalls it (and biases $\sqrt{v}$, the step size) |
| first moment $m$ | nearest | signed and roughly zero-mean: nearest rounding is already unbiased |

The update reads $m$ and $v$ before they are rounded. Parameters that are not bf16 (fp32 adapters, an fp32 head) take the exact fp32 AdamW update.

**Memory per trained parameter** (weights + gradient + optimizer state):

| Regime | Weight | Gradient | Moments | Total |
|---|:-:|:-:|:-:|:-:|
| `torch_dtype: float32` + `adamw` (fp32 masters) | 4 | 4 | 8 | 16 B |
| `torch_dtype: bfloat16` + `adamw` (updates round away) | 2 | 2 | 4 | 8 B |
| `torch_dtype: bfloat16` + `adamw_bf16` | 2 | 2 | 4 | **8 B** |
| `torch_dtype: bfloat16` + `adamw_bf16`, `bf16_rounding: kahan` | 2 | 2 | 4 + 2 | **10 B** |

**Stochastic or compensated (`train.bf16_rounding`).** Stochastic rounding (`stochastic`, the default) is unbiased, but each weight's rounding error is a random walk: after $N$ steps of an update $u$ much smaller than an ulp its spread is about $\text{ulp}\sqrt{N u / \text{ulp}}$, while the update itself only adds up to $N u$ when its sign is steady. With `kahan` a bf16 compensation buffer (+2 B/param) keeps what nearest rounding dropped from the weight, so weight + compensation carry about 16 significant bits and no noise, and the stored weight is the exact trajectory rounded to nearest, as a bf16 forward of fp32 masters sees it. Measured on CPU at an RL-sized learning rate (1e-6, gradients of random sign, 300 steps): stochastic rounding ends 4.2x the distance the fp32 weights moved away from them, `kahan` within 2% (`tests/test_bf16_adamw.py`). Use `kahan` for small learning rates (RL, DPO, late low-LR phases); `stochastic` when every byte counts and steps are a sizeable fraction of an ulp.

```yaml
train:
  optimizer: adamw_bf16
  bf16_rounding: kahan   # or stochastic (default)
```

**Reproducibility.** The rounding noise is a counter-based integer hash of (the element's position, a seed from `train.seed`, the parameter's name and its step count), not a random generator: runs are bit-for-bit reproducible, a resumed run rounds exactly as the uninterrupted one (the step count is in the optimizer state), data-parallel ranks holding the same parameter round it identically, and CPU and CUDA give the same bits.

**Speed.** On CUDA each parameter's update is one fused kernel compiled by `torch.compile` (one read of weight, gradient and both moments, one write of weight and moments); per-step scalars and seeds reach the GPU in one non-blocking copy. Off CUDA it runs eagerly. `train.fused_optimizer: true` steps all parameters in one Triton launch per dtype bucket instead, with the same bits as the eager update ([below](#multi-tensor-kernels-trainfused_optimizer)). Measure it on your GPU with `python scripts/bench_optimizers.py step --model <model>` (step time and state bytes per parameter against fused torch AdamW and bitsandbytes 8-bit) and `python scripts/bench_optimizers.py train --model <model>` (tokens/s, peak memory and the loss after a fixed number of steps against fp32 masters).

**What is measured on CPU** (`tests/test_bf16_adamw.py`): 200 steps of updates of 1/16 ulp move weights by 100% ($\pm 2\%$) of the fp32-master drift, against under 50% for `adamw` on bf16; a small regression at lr 3e-4 ends within 5% of the fp32-master loss (plain bf16 AdamW: 37% higher); the second moment stays within 0.3% of its fp32 value where nearest rounding stalls 4% low; the fused (compiled) update gives the same bits as the eager one, for both roundings.

Composes with: LLRD, gradient clipping, `memory.gradient_release` (per-parameter step), Hyperball, FSDP2 (each rank steps its shard; the noise is keyed by the element's offset in the local shard, so a resume must keep the sharding layout to be bit-exact), checkpoints and resume. Not with: Schedule-Free (it replaces the optimizer). `pgs rl` and `pgs distill` keep their fp32-master optimizers.

After: stochastic rounding for low-precision training (Gupta et al., [arXiv:1502.02551](https://arxiv.org/abs/1502.02551); Zamirai et al., [arXiv:2010.06192](https://arxiv.org/abs/2010.06192), which also evaluates Kahan summation); Kahan-compensated bf16 optimizers as in [optimi](https://github.com/warner-benjamin/optimi) (MIT); rounding only the second moment stochastically follows the design notes of the [Halo](https://github.com/whitecircle/halo) framework's AdamWBF16 (re-implemented, no code shared).

---

## Multi-tensor kernels (`train.fused_optimizer`)

```yaml
train:
  optimizer: adamw_bf16   # or adamw
  fused_optimizer: true
```

**The problem.** The per-parameter paths launch at least one kernel per parameter: a 2B model has hundreds of parameter tensors, so `adamw_bf16`'s compiled update pays a `torch.compile` guard check and a launch hundreds of times per step, and the step becomes bound by the host rather than by the ~14 bytes per parameter it has to move.

**What it does** (`palingenesis.fused_optim`). Per step the host builds one small table per bucket (the element offset of each tensor's weight, gradient, moments and compensation from the bucket's base tensors, its noise seed and its six AdamW scalars) and copies it to the GPU in one non-blocking transfer from pinned memory. A block map, cached by the tensors' sizes, gives every Triton program a (tensor, first element) pair; each program steps 8192 elements of one tensor. A bucket is (parameter dtype, gradient dtype, update kind), so a bf16 model with bf16 gradients takes **one launch for all its parameters** (plus one for fp32 parameters, if any). Tensors are addressed as base pointer + int64 element offset, so Triton keeps its 16-byte vector loads; tensors that are not 16-byte aligned (views into a flat buffer) take an unhinted variant.

| Optimizer | Kernel | Semantics |
|---|---|---|
| `adamw_bf16` (both `bf16_rounding`s) | bf16 weights, bf16 moments, Kahan buffer | exactly the operations of the eager update, in the same order, IEEE `sqrt` and division, no FMA contraction: **the same bits** as `BF16AdamW(compile=False)`. The counter-based rounding noise is unchanged, so resumes stay bit-exact. (The default compiled per-parameter update may differ from it by one ulp on a few elements, where `torch.compile` contracts a multiply-add.) |
| `adamw` | weights, moments in the weight's dtype (fp32 masters) | torch.optim.AdamW's update, equal to it up to fp32 rounding; the state and `state_dict` are torch AdamW's, so a checkpoint resumes with either |

**Gradient clipping folded into the step.** On one process the trainer calls the optimizer's `clip_grad_norm_` instead of torch's: the global norm is one launch per gradient dtype (each program's partial sum of squares, then one deterministic sum), and the clip factor is multiplied into the fp32 gradient inside the step kernel, instead of a separate pass that reads and writes every gradient. The norm is accumulated in fp32 (torch rounds the per-tensor norms of bf16 gradients to bf16). The `.grad` tensors keep their unclipped values, which is what the health monitor's gradient metrics then see. Several processes (FSDP2): torch's clipping, then the multi-tensor step on each rank's shard.

**Composes with** LLRD and the other parameter groups (per-tensor scalars), weight decay, spike detection (a skipped step drops its pending clip factor with `zero_grad`), and FSDP2 by design (each rank steps its local shards; not yet run on several GPUs). AdamC, Hyperball and MONA change the groups' scalars, wrap or precede `step()` as with the per-parameter optimizer. Not with `memory.gradient_release` (it steps each parameter inside the backward: nothing to batch; the config refuses the pair), `plugins.schedule_free`, `muon` or the 8-bit optimizers. **Falls back** to the per-parameter optimizer, with a warning, without a CUDA device or without Triton, and for parameters in host memory (`parallel.cpu_offload`). Importing palingenesis or validating a config does not import Triton.

**What is measured where.** On CPU (`tests/test_fused_optim.py`): an emulation of the kernels' integer formulation (the uint32 hash, rounding on the top 16 bits) gives the reference functions' bits on special values (zeros, subnormals, the largest finite values, infinities, NaN), and run over the block map with per-tensor scalars and seeds it gives the per-parameter optimizer's bits for both roundings, with and without a clip factor; FusedAdamW matches torch AdamW and their state dicts interchange; folded clipping matches clip-then-step. On a GPU, `scripts/gpu_checks/offload_fused_optim.sh` checks the Triton kernels bit for bit against the eager update (odd sizes, misaligned views, extreme magnitudes, fp32 parameters, a clip factor), bit-exact resume, step time, host time and GPU launches per step against the per-parameter path and torch's fused AdamW on Qwen3.5-0.8B and -2B shapes, and a 30-step SFT loss curve against fp32 master weights. One known difference: Triton may flush denormal intermediates (gradients below ~1e-19) to zero where the eager update keeps them; the script reports how often that changes a stored value.

After NVIDIA Apex's `multi_tensor_apply` and torch's fused / `_foreach` optimizers (both BSD-3); re-implemented in Triton, no code copied.

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
