# Training Plugins — Research-Backed Loss Functions & Techniques

This document covers the research-backed plugins available in palingenesis. Each plugin is opt-in, torch.compile compatible, and backed by peer-reviewed papers.

## Loss Function Hierarchy

All token-level SFT losses follow the universal gradient structure (from arxiv:2602.11424):

```
∂L/∂z_target = -p^α × (1 - p)
```

Where `p` = model's probability for the correct token, and `α` = focus index:

| Loss | α | Behavior | Best For |
|------|---|----------|----------|
| NLL (standard CE) | 0 | Uniform: learns everything including noise | General |
| InfoSFT | variable (information-weighted) | Focus on medium-confidence tokens | Anti-forgetting |
| **DEFT** | per-token Rényi-2 entropy | Automatic: NLL when uncertain, DFT when confident | **Best overall** |
| DFT | 1 (fixed) | Sharpening: suppresses hard tokens | Math/code reasoning |
| CADFT | 1 + sample reweighting | DFT + variance control across samples | Heterogeneous data |

## DEFT — Dynamic Entropy Fine-Tuning

**Paper**: "Gradients Must Earn Their Influence" (arxiv:2602.11424, Feb 2026)  
**Config**: `plugins.deft: true`

The single best loss function for SFT. Parameter-free, automatically adapts per-token.

### How it works

DEFT computes a per-token focus index using the Rényi-2 collision probability:

```
α(context) = Σ P_θ(v|c)²    for all v in vocabulary
```

- When the model's prediction is **diffuse** (uncertain, many tokens plausible): α ≈ 0 → NLL-like. Full gradient coverage for learning new knowledge.
- When the model's prediction is **concentrated** (confident, one token dominates): α ≈ 1 → DFT-like. Efficient sharpening without destabilizing rare tokens.

The loss is then: `L = -p^α × log(p)` (equivalent to weighting CE by the trust gate `p^α`).

### Evidence

- +70-80% over NLL on math reasoning (Math500, OlympiadBench, AIME) — *as reported in the original paper*
- Tested on LLaMA-3.1-8B, DeepSeekMath-7B, Qwen2.5-Math-1.5B/7B
- Matches or exceeds both DFT and InfoSFT across regimes
- **Caveat**: These results are from arxiv:2602.11424 and have not been independently reproduced outside the original authors' experiments. Gains may vary on non-math tasks and different model/data combinations.

### Compute cost

Near zero. One softmax (already computed) → square → sum per position. Fused by torch.compile.

---

## DFT — Dynamic Fine-Tuning

**Paper**: Wu et al. 2025, validated in CADFT (arxiv:2606.11206)  
**Config**: `plugins.dft: true`

Fixes the fundamental SFT pathology: standard CE gradient scales as `1/p_t` for the target token, causing rare tokens to dominate training with explosive gradients.

### How it works

Multiply each token's CE loss by `p_t` (the model's own probability):

```
L_DFT = p_t × (-log p_t)
```

Gradient becomes `-(1 + log p_t)` — bounded as p_t → 0.

### When to use

- Pure math/code reasoning tasks where model already has strong priors
- When you observe gradient norm instability from rare tokens

### WARNING: Distributional Drift

DFT alone causes progressive drift from the base model on knowledge-intensive tasks (arxiv:2509.23753, ICLR 2026). On medical benchmarks: DFT scores 4 points BELOW standard SFT.

**Fix**: Pair DFT with KL anchoring via `pre_rl` mode:
```yaml
plugins:
  dft: true
  pre_rl: true
  pre_rl_entropy_coeff: 0.0   # no entropy bonus
  pre_rl_kl_coeff: 0.05       # lightweight KL anchor
```

DEFT has the same drift risk. For knowledge tasks, prefer DEFT + pre_rl or just standard CE.

---

## CADFT — Compatibility-Aware Dynamic Fine-Tuning

**Paper**: arxiv:2606.11206 (Apr 2026)  
**Config**: `plugins.cadft: true`, `plugins.cadft_beta: 1.0`

Extends DFT with sample-level variance control.

### How it works

1. **Token level**: DFT (p_t weighting)
2. **Sample level**: Z-score each sample's NLL within the batch, then exponentially down-weight outliers:
   ```
   ĉ = (sample_nll - μ_batch) / σ_batch
   weight = exp(-β × max(0, ĉ))
   ```

Samples that are far above the batch-mean NLL (incompatible with current model state) get suppressed. This prevents high-variance gradient updates from samples the model can't currently learn from.

### Evidence

- +5pp over DFT on math benchmarks across model scales
- Consistent improvement on code generation (HumanEval, MultiPL-E)
- Better initialization for downstream GRPO

---

## InfoSFT — Information-Aware Token Weighting

**Paper**: arxiv:2605.14967 (May 2025)  
**Config**: `plugins.info_sft: true`, `plugins.info_sft_pbar: 0.93`

Concentrates learning on "medium-confidence" tokens — those that are informative (not too easy, not too hard).

### Formula

```
w(q) = q × [logit(p̄) - logit(q)]₊
```

Where `q` = model's probability for the correct token, `p̄` = calibration constant (0.93).

Tokens where the model is already very confident (q > p̄) get zero weight. Tokens where the model has zero confidence also get low weight (the `q` factor). Maximum weight at the "information frontier."

---

## SymNoise — Symmetric Noisy Embeddings

**Paper**: arxiv:2312.01523 (ICLR 2024 + NeurIPS 2025)  
**Config**: `plugins.sym_noise: true`, `plugins.sym_noise_alpha: 5.0`

Injects Bernoulli {-1, +1} noise into token embeddings during training. Acts as a strong regularizer that prevents overfitting on surface patterns.

### Key difference from NEFTune

- NEFTune uses uniform noise; SymNoise uses Bernoulli
- Paper ablation: Bernoulli > Uniform > Gaussian (+6.7% on AlpacaEval)
- `alpha` controls noise magnitude: `noise = alpha / sqrt(seq_len × dim)`

### Stacking

SymNoise stacks with ALL loss functions. It operates on embeddings (before the forward pass), while loss functions operate on logits (after).

---

## Pre-RL Mode — Diversity Preservation for GRPO/DPO

**Paper**: arxiv:2605.29303 (May 2026)  
**Config**: `plugins.pre_rl: true`

When your pipeline is SFT → GRPO/DPO, standard SFT sharpens the policy so aggressively that RL can't explore. Pre-RL mode preserves diversity:

1. Entropy bonus on uncertain tokens (keeps them exploratory)
2. KL penalty from base model (prevents drift)
3. Selective masking (only train on "safe" tokens)

Also functions as **KL anchoring** for DFT/DEFT (set `entropy_coeff: 0` for pure anchoring).

---

## Recommended Configurations

### Best overall (math/code reasoning)
```yaml
plugins:
  deft: true
  sym_noise: true
  sym_noise_alpha: 5.0
```

### Knowledge-intensive (medical, factual QA)
```yaml
plugins:
  deft: true
  sym_noise: true
  pre_rl: true
  pre_rl_entropy_coeff: 0.0
  pre_rl_kl_coeff: 0.05
```

### Preparing for GRPO/DPO
```yaml
plugins:
  pre_rl: true
  pre_rl_entropy_coeff: 0.1
  pre_rl_kl_coeff: 0.5
  sym_noise: true
```

### Heterogeneous data (mixed quality, varying difficulty)
```yaml
plugins:
  cadft: true
  cadft_beta: 1.0
  sym_noise: true
```

---

## Trainer plugins

Your own code inside `pgs train` (SFT and [off-policy distillation](offpolicy_distillation.md)): substitute layers, add loss terms and regularisers, change teacher targets, set optimizer groups, ramp anything with training progress, log metrics, fold weights on export. A plugin is a subclass of `palingenesis.hooks.TrainerPlugin` overriding the hooks it needs; every hook is optional.

```yaml
plugins:
  custom:
    - my_package.plugins:MyPlugin                       # module:Class
    - {class: ./plugins/qat.py:MyQAT, args: {bits: 2}}  # a file, with constructor arguments
    - my-registered-plugin                              # an entry point of group "palingenesis.plugins"
```

```python
from palingenesis.train import train
train(config, plugins=[MyPlugin(), "my_package.plugins:Other"])   # after the config's
```

A package registers plugins for the short form with an entry point:

```toml
[project.entry-points."palingenesis.plugins"]
my-registered-plugin = "my_package.plugins:MyPlugin"
```

### Hooks, in the order they run

| Hook | When | Use it to |
|---|---|---|
| `setup(ctx)` | once, after the config is validated, before anything loads | check or adjust `ctx.config` |
| `transform_model(model, ctx)` | once, after the student loads, **before** float8, activation checkpointing, FSDP and compile | substitute layers, wrap the embedding or the head, freeze, add modules or parameters (they are then sharded, checkpointed and resumed like the rest) |
| `feature_requests(ctx)` | once (needs `distill.enabled`) | return `FeatureSpec`s: student / teacher module outputs to capture and match at the trained positions |
| `param_groups(model, ctx)` | once, before the scheduler is built | return overrides `{"params": [...] or "names": regex, "lr", "eps", "weight_decay", "betas"}`; the parameters move to groups of their own (the schedule still applies relative to the group's lr) |
| `on_train_start(model, ctx)` | once, before the first step, after a resume | calibration, statistics, initial snapshots |
| `on_forward_start(model, batch, ctx)` | every micro-batch | set step-dependent module state (quantisation strength, temperatures) |
| `transform_targets(targets, batch, ctx)` | every micro-batch with a teacher | reweight tokens (`targets.token_weights`), change teacher targets, drop teachers |
| `loss(inputs, ctx)` | every micro-batch, after the forward | return a scalar or `{name: scalar}`: extra loss terms (logged as `plugins/<name>/<key>`) |
| `on_before_optimizer_step(model, optimizer, ctx)` | each step, gradients complete, before clipping | mask gradients, add regulariser gradients |
| `on_after_optimizer_step(model, optimizer, ctx)` | each step, after `optimizer.step()` | clamp or project weights |
| `metrics(model, ctx)` | logging steps | return scalars (logged as `plugins/<name>/<key>`); `ctx.log(key, value)` from any hook |
| `export_state_dict(state, model, ctx)` | final and best-model exports (never resume checkpoints) | fold quantised weights, rename, drop training-only tensors; `state` is the full state dict under the architecture's names |
| `state_dict()` / `load_state_dict(state)` | every checkpoint / resume | plugin state, saved as `plugins.pt` next to the checkpoint |
| `on_train_end(model, ctx)` | after the last step, before the final save | final statistics |

`ctx` (`PluginContext`) carries the config, `step` (optimizer steps done), `total_steps`, `progress` (0 to 1), `rank`, `world_size`, `dp_world_size`, `device`, `is_fsdp`, `output_dir` and, when distilling, the `distiller` (its teachers and caches). `palingenesis.hooks.Ramp(start, end, start_value, end_value, shape)` is a schedule over `progress` (`linear`, `cosine`, `step`); `Ramp.parse` accepts a number or a mapping, so ramps can come straight from plugin arguments.

### `loss` inputs and scaling

`LossInputs` holds the model, the batch, the student's final hidden states `[B, T, H]` (with gradient), the shifted labels, `rows` (flat indices of the trained positions), the captured `features` (`{name: (student [N, Ds], teacher [N, Dt])}`), the teacher targets, the output head, the global token denominator and the accumulation steps.

The trainer adds a plugin's term multiplied by `1 / (gradient accumulation steps × data-parallel ranks)`: gradients are summed over the accumulation window and over ranks, so return a **per-rank mean** (for example a mean over this rank's tokens). For a sum over tokens normalised like the main loss, multiply by `inputs.token_sum_scale`.

Loss hooks run where the trainer has the student's forward in hand (SFT with chunked / cut / plain cross-entropy, the gated objectives, off-policy distillation). Paths that run their own backward refuse them at startup: SeCO, DPO, `plugins.pre_rl`, `memory.gradient_release`. Put gradient-level terms in `on_before_optimizer_step` instead: they work on every path.

### FSDP

- Modules added in `transform_model` exist before FSDP wraps the model: they are sharded, their gradients reduced and their weights checkpointed like the rest.
- In the optimizer-step hooks, parameters and gradients are sharded DTensors. FSDP2 shards dimension 0, so `p.to_local()` holds whole rows: per-row and per-group statistics are local; a per-tensor statistic needs an all-reduce over `p.device_mesh.get_group()` (see `palingenesis.qat`).
- `export_state_dict` receives the gathered full state on rank 0.

### Example: a regulariser and a metric

```python
import torch
from palingenesis.hooks import Ramp, TrainerPlugin

class L2ToStart(TrainerPlugin):
    """Pull the weights towards their initial values with a ramped strength (gradient-level)."""

    name = "l2_to_start"

    def __init__(self, lam=1e-4):
        self.lam = Ramp(0.0, 0.2, 0.0, lam)

    def on_train_start(self, model, ctx):
        self.start = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def on_before_optimizer_step(self, model, optimizer, ctx):
        lam = self.lam(ctx.progress)
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.start:
                p.grad.add_(2 * lam * (p - self.start[n]))

    def metrics(self, model, ctx):
        return {"lambda": self.lam(ctx.progress)}
```

`palingenesis.qat.TernaryQAT` is a complete example (layer substitution, a straight-through quantiser, a strength ramp, a grid regulariser, latent clamping, optimizer groups, hidden-state targets, folded export): see [Quantisation-aware training while distilling](offpolicy_distillation.md#quantisation-aware-training-while-distilling).
