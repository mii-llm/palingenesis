# Ternary QAT (1.58-bit weights)

*Train or distill a model whose weights are `alpha * {-1, 0, 1}`, choose each part of the quantiser in the config, and merge ternary models into a ternary model.*

!!! tip "When to use this"
    - You want a model for CPU kernels that add and subtract instead of multiplying (bitnet.cpp, llama.cpp `TQ1_0` / `TQ2_0`), at about 1.6 to 1.8 bits per weight.
    - You can afford training: rounding a trained model to ternary weights destroys it (see [Measured](#measured)), so every ternary model here is trained, with the full-precision model as the teacher.
    - For 4-bit and 8-bit floating-point formats served by vLLM, use [FP4 / FP8 QAT](qat.md).

## Run the baseline

```bash
pgs train --config configs/ternary_qat_qwen35_08b.yaml     # writes runs/ternary/final (ternary values, any loader reads it)
pgs merge ternary --check runs/ternary/final               # exit 0: every ternary tensor holds only -alpha, 0, alpha per group
```

The config is the whole run: model, data, schedule, the distillation loss and the quantiser. Its plugin section:

```yaml
distill:
  enabled: true
  divergence: forward_kl
  teachers: {fp: {model: Qwen/Qwen3.5-0.8B, dtype: bfloat16}}   # the full-precision model, run live
plugins:
  custom:
    - class: palingenesis.qat:TernaryQAT
      args:
        group_size: 128
        scale: absmean
        scale_mode: dynamic
        quantizer: round
        ste: identity
        quantize_embedding: true
        quantize_head: true
        latent_lr: 1.0e-4
        latent_eps: 1.0e-12
```

Every substituted layer keeps a full-precision **latent** weight (the same parameter under the same name). The forward computes with `alpha * code`; the gradient reaches the latent weight through a straight-through estimator. The export (`final/`) holds the ternary values in the model's own layout, plus `ternary_config.json` (which tensors are ternary, their group size, the quantiser).

## The components

Each is one argument of the plugin, and one function in a registry of `palingenesis.qat.ternary` (`SCALES`, `QUANTIZERS`, `STES`): adding a scheme is adding a function.

| Component | Argument | Choices |
|---|---|---|
| Granularity | `group_size` | `128` (one scale per 128 input weights), `-1` (per output row), `0` (per tensor) |
| Scale scheme | `scale` | `absmean` (alpha = mean \|w\|, BitNet b1.58), `absmax` (alpha = 2/3 max \|w\|: three equal bins, ParetoQ's starting point), `twn` (dead zone 0.7 mean \|w\|, alpha = mean of the rest), `mse` (the exact least-squares alpha and dead zone, by sorting) |
| When scales are computed | `scale_mode` | `dynamic` (from the latent weights, every forward), `frozen` (once, from the loaded weights), `learned` (frozen, times a trained log-factor with LSQ gradients; `scale_lr`) |
| Code rule | `quantizer` | `round` (sign where \|w\| exceeds the dead zone), `stochastic` (training forwards sample \|code\| = 1 with probability \|w\| / alpha; evaluation and export round) |
| Gradient estimator | `ste` | `identity`, `clipped` (`ste_args: {clip: 1.0}`: no gradient beyond clip x alpha), `tanh` (slope of a smooth staircase; `ste_sharpness` is a number or a ramp) |
| Latent handling | `latent_lr`, `latent_eps`, `latent_weight_decay`, `clamp`, `grid_weight`, `strength` | their own optimizer group; \|w\| kept below `clamp` x alpha; a pull towards the grid; a ramp from full precision (0) to ternary (1) |
| What is ternary | `include`, `exclude`, `quantize_embedding`, `quantize_head`, `activation_bits` | regular expressions over `nn.Linear` names; the tied embedding / head; int8 / int4 inputs |
| Distillation loss | `distill.divergence`, `distill.temperature`, `distill.ce_weight`; plugin `hidden_layers`, `hidden_loss`, `hidden_weight` | forward / reverse KL, JSD, skewed KLs, TVD on the logits; MSE, normalised MSE, cosine or lens KL on hidden states |
| Teacher | `distill.teachers` | online (`model:`), cached top-k targets (`cache:`, built by `pgs distill-cache`), another model with the same vocabulary |
| Export | `export`, `save_latent` | `folded` (ternary values), `latent`; `save_latent: true` also writes `final-latent/` |
| Merge | `pgs merge ternary --method` | `requantize`, `vote`, `ties`, `consensus` |

`frozen` and `learned` scales are stored next to each weight (`scale_alpha`, `scale_delta`, `log_scale`) and are single-process (not FSDP); the folded export drops them. The default (`absmean`, `dynamic`) runs under FSDP.

### Each variant, as a change to the baseline

```yaml
# scale scheme and granularity
args: {scale: mse, group_size: -1}
# scales fixed at the start, or trained
args: {scale: mse, scale_mode: frozen}
args: {scale: mse, scale_mode: learned, scale_lr: 1.0e-3}
# estimators and code rule
args: {ste: clipped, ste_args: {clip: 1.0}}
args: {ste: tanh, ste_sharpness: {start: 0.0, end: 0.8, start_value: 2.0, end_value: 10.0}}
args: {quantizer: stochastic}
# gradual quantisation, a late pull to the grid, no clamp
args: {strength: {start: 0.0, end: 0.2}}
args: {grid_weight: {start: 0.8, end: 1.0, start_value: 0.0, end_value: 1.0}}
args: {clamp: 0.0}
# body only (embedding and head in full precision), int8 activations
args: {quantize_embedding: false, quantize_head: false}
args: {activation_bits: 8}
# hidden-state matching on top of the logits
args: {hidden_layers: [-1], hidden_loss: nmse, hidden_weight: 1.0}
```

```yaml
# distillation loss and teacher (the distill section)
distill: {divergence: reverse_kl}
distill: {ce_weight: 0.5}                       # KL + cross-entropy on the corpus
distill: {enabled: false}                       # no teacher: cross-entropy only
distill: {teachers: {fp: {model: Qwen/Qwen3.5-0.8B, cache: cache/q35_k64, top_k: 64}}}   # offline targets
```

Offline targets are built once with `pgs distill-cache --config <the training config> --teacher fp`.

## Merging ternary models

The mean of ternary tensors is not ternary: with codes in {-1, 0, 1} and one scale per model, the mean of k models takes up to 3^k values per group. `pgs merge soup` on ternary exports therefore gives a full-precision model. `pgs merge ternary` projects back, and its methods differ in what they project:

| `--method` | What it does | Needs |
|---|---|---|
| `requantize` | mean of the sources, then the quantiser again. On `final-latent/` directories this merges the latent weights and re-ternarises | – |
| `vote` | per weight, the sign backed by more model weight wins if it also outweighs the zeros; +1 against -1 with equal weight gives 0. `--tie sign` keeps a sign that only ties with zero (two models) | – |
| `ties` | sign election by total mass, mean of the entries that agree with the elected sign, then re-ternarisation (TIES-Merging); zero codes act as trimmed entries | optional `--base` (then the election runs on the changes) |
| `consensus` | a code keeps the shared start's value unless models changed it; a change is taken when the models that made it agree (a change made by one expert survives whole) | `--base`: the ternary model the experts started from |

```bash
pgs merge ternary runs/a/final runs/b/final runs/c/final --method vote -o merged/
pgs merge ternary runs/a/final-latent runs/b/final-latent --method requantize -o merged/
pgs merge ternary runs/a/final runs/b/final --method consensus --base runs/start/final -o merged/
pgs merge ternary runs/a/final:0.7 runs/b/final:0.3 --method ties -o merged/
pgs merge ternary --check merged/        # also run at the end of every merge; a failure is an error
```

Each group's scale is refit by least squares to the weighted mean of the sources given the merged codes (`--no-refit`: the mean of the sources' scales). Tensors that are not ternary (norms, biases) are averaged. `merge_config.json` records the share of weights where the sources had opposite signs (`conflict_share`), where some were zero and some not (`support_split_share`), and `ternary_verified`.

In Python: `palingenesis.qat.merge.merge_ternary(models, out, method=...)`, `merge_tensor(...)` for one tensor, `check_ternary(model_dir)`.

## Measured

MEASURED_PLACEHOLDER

## References

- BitNet b1.58 (absmean ternary weights): [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- Ternary Weight Networks (threshold rule): [arXiv:1605.04711](https://arxiv.org/abs/1605.04711)
- ParetoQ (ternary QAT from a pretrained model, token budgets): [arXiv:2502.02631](https://arxiv.org/abs/2502.02631)
- LSQ (learned step size): [arXiv:1902.08153](https://arxiv.org/abs/1902.08153)
- Straight-through estimator: [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- TIES-Merging (sign election): [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)
