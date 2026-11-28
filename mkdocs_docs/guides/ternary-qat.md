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
        scale: mse                  # the plugin's own default is absmean (BitNet b1.58); see Measured
        scale_mode: dynamic
        quantizer: round
        ste: identity
        quantize_embedding: true
        quantize_head: true
        latent_lr: 0.016
        latent_lr_relative: true    # the step of each latent tensor: 0.016 x its mean |W|
        latent_eps: 1.0e-12
```

The config differs from the plugin's defaults in two places, both from the measurements below: the scale scheme (`mse` instead of `absmean`) and a latent learning rate in units of each tensor's own scale.

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
| Latent handling | `latent_lr`, `latent_lr_relative`, `latent_eps`, `latent_weight_decay`, `clamp`, `grid_weight`, `strength` | their own optimizer group (with `latent_lr_relative` one per tensor, at `latent_lr` x its mean \|W\|); \|w\| kept below `clamp` x alpha; a pull towards the grid; a ramp from full precision (0) to ternary (1) |
| What is ternary | `include`, `exclude`, `quantize_embedding`, `quantize_head`, `activation_bits` | regular expressions over `nn.Linear` names; the tied embedding / head; int8 / int4 inputs |
| Distillation loss | `distill.divergence`, `distill.temperature`, `distill.ce_weight`; plugin `hidden_layers`, `hidden_loss`, `hidden_weight` | forward / reverse KL, JSD, skewed KLs, TVD on the logits; MSE, normalised MSE, cosine or lens KL on hidden states |
| Teacher | `distill.teachers` | online (`model:`), cached top-k targets (`cache:`, built by `pgs distill-cache`), another model with the same vocabulary |
| Export | `export`, `save_latent` | `folded` (ternary values), `latent`; `save_latent: true` also writes `final-latent/` |
| Merge | `pgs merge ternary --method` | `requantize`, `vote`, `ties`, `consensus` |

`frozen` and `learned` scales are stored next to each weight (`scale_alpha`, `scale_delta`, `log_scale`) and are single-process (not FSDP); the folded export drops them. The default (`absmean`, `dynamic`) runs under FSDP.

### Each variant, as a change to the baseline

```yaml
# scale scheme and granularity
args: {scale: absmean}            # BitNet b1.58
args: {scale: absmax, group_size: -1}
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

One A100, online distillation from the model's own full-precision weights (forward KL, temperature 1), raw web text in rows of about 890 tokens, 32 rows per step, cosine schedule, fp32 latent weights with bf16 autocast. Readout: KL(teacher ‖ student) per token on 64 × 512 held-out tokens of web text and of WikiText-2 test (mean of the two), perplexity, and zero-shot ARC-Easy (2,376 items; one standard error ≈ 0.01). Screening model: SmolLM2-135M, decoder linears ternary, 8M tokens (284 steps) per run, one seed unless stated. The numbers are small-budget screens: they say which choices matter early in training, not what a long run converges to.

**Rounding without training does not work.** SmolLM2-135M rounded to ternary: KL 21.7 (`absmean`) and 13.2 (`mse`) nats per token, perplexity above 10⁶. Every model below starts from that point.

**The scale scheme and the learning rate matter most.**

| Change from the default (`absmean`, per 64, `dynamic`, `round`, `identity`, relative latent lr 0.016) | KL | Perplexity (web) | ARC-Easy |
|---|---|---|---|
| default, three seeds | 2.73 to 2.78 | 183 to 190 | 0.30 to 0.31 |
| `scale: mse`, three seeds | 1.14, 1.16, 1.89 | 46 to 82 | 0.34 to 0.39 |
| `scale: absmax` | 1.15 | 48 | 0.39 |
| `scale: twn` | 1.51 | 60 | 0.36 |
| `scale: mse`, `scale_mode: learned` | 1.10 | 46 | 0.38 |
| `scale: mse`, `scale_mode: frozen` | 1.16 | 49 | 0.39 |
| relative latent lr 0.004 / 0.064 (`absmean`) | 3.26 / 3.16 | 325 / 298 | 0.30 / 0.29 |
| relative latent lr 0.004 / 0.064 (`mse`) | 1.58 / 3.15 | | 0.36 / 0.29 |
| absolute latent lr 1e-4 | 3.94 | 736 | 0.28 |
| `group_size: -1` / `0` (`absmean`) | 2.98 / 3.02 | 236 / 247 | 0.30 / 0.29 |
| `clamp: 0` (`absmean` / `mse`) | 2.22 / 1.04 | 107 / | 0.33 / 0.40 |
| `quantizer: stochastic` | 3.74 | 542 | 0.28 |
| `ste: clipped` / `tanh` (`absmean`) | 2.67 / 2.83 | 172 / 200 | 0.31 / 0.30 |
| `strength` ramp over 20% / late `grid_weight` / `latent_weight_decay: 0.1` | 2.76 / 2.92 / 2.76 | | |
| `activation_bits: 8` (`absmean` / `mse`) | 2.74 / 1.20 | | 0.30 / 0.38 |
| embedding and head ternary too (`absmean` / `mse`) | 2.90 / 2.70 | | 0.29 / 0.31 |
| (full-precision model) | 0 | 17.8 | 0.58 |

- `mse` (or `absmax`) instead of `absmean` more than halves the KL, and still does at 32M tokens (0.68 against 1.21). With `absmean` and `dynamic` scales the default clamp also works against the run: the clamp cuts the tail of each group at 1.5 × mean |w|, the mean falls, and the next clamp is tighter.
- The latent learning rate is best given in units of each tensor's mean |W| (`latent_lr_relative: true`): one value then suits every tensor. 0.016 was best here; four times lower or higher costs as much as a wrong scale scheme.
- One of three `mse` seeds ended far from the other two, so differences of 0.1 to 0.2 between `mse` runs (learned scales, no clamp, hidden states) are not resolved by these runs.
- Gradient-estimator variants, a strength ramp, weight decay, Adam eps and int8 activations changed little. Stochastic rounding and per-row or per-tensor scales were worse.
- A ternary embedding and head cost little with `absmean` and a lot with `mse` on this model (576-dimensional embeddings, 8M tokens).

**Distillation loss.**

| Change | KL (`absmean`) | KL (`mse`) |
|---|---|---|
| forward KL, online teacher | 2.78 | 1.14 |
| no teacher (cross-entropy only) | 2.88 | 1.26 |
| KL + 0.5 cross-entropy | 2.80 | 1.23 |
| `reverse_kl` / `jsd` / `tvd` | diverged (KL above 40) | 3.30 / – / – |
| temperature 2 | 3.32 | |
| hidden states of 5 layers (`nmse`) + KL | 2.27 | 1.00 |
| last hidden state / lens KL of a middle layer + KL | 2.74 / 2.83 | |
| teacher 2.7× larger (same vocabulary) | 2.79 | |
| cached top-64 targets (offline) / + 0.1 cross-entropy | 2.80 / 2.81 | |

Forward KL is the safe choice: mode-seeking divergences diverged or lost badly from a start this far from the teacher. The teacher run costs a third of the step time (232 s against 155 s without it); cached targets bring the step to 178 s but building the cache takes the same teacher pass (91 s), so a cache pays only when it is reused. A larger teacher did not help.

**Tokens.** SmolLM2-135M, `mse`: KL 4.51 at 2M tokens, 1.14 at 8M, 0.68 at 32M (perplexity 31 against 17.8 for the teacher; ARC-Easy 0.27, 0.38, 0.45 against 0.58).

**A larger model, fully ternary.** Qwen3.5-0.8B with every decoder linear and the tied embedding / head ternary (`group_size: 128`, `scale: mse`, relative latent lr 0.016), 25M tokens of English and Italian web text (888 steps, 1.9 h at 3.8k tokens/s): WikiText-2 perplexity 42.0 (full precision 23.0), Italian web 40.9 (32.2), ARC-Easy 0.451 (0.615), ARC-Challenge 0.259 (0.374). The gap to the full-precision model is large at this budget; published ternary models of this size use billions of tokens.

**Merging.** SmolLM2-135M, `mse`, 8M tokens per expert; every merged model verified ternary.

| | KL |
|---|---|
| four experts trained from the full-precision weights on four shards | 1.15 to 1.27 |
| their merge: `requantize` of the latent weights / of the ternary weights / `vote` / `ties` | 1.29 / 1.31 / 1.35 / 1.47 |
| one model with the same 32M tokens | 0.68 |
| a shared ternary start (8M tokens), then three experts continued for 8M tokens each | 0.85 to 0.89 |
| their merge: `requantize` (latent) / `vote` / `ties` / `ties --base` / `consensus` | 0.84 / 0.85 / 0.96 / 0.93 / 0.99 |
| the start continued alone for the same 24M tokens | 0.69 |

Experts that share a start almost never disagree on a sign (0.006% to 0.011% of the weights); they disagree on which weights are zero (11% to 13%). A vote or a latent average therefore gives a working ternary model, about as good as one expert, and never as good as one model trained on all the data for the same total compute.

## References

- BitNet b1.58 (absmean ternary weights): [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- Ternary Weight Networks (threshold rule): [arXiv:1605.04711](https://arxiv.org/abs/1605.04711)
- ParetoQ (ternary QAT from a pretrained model, token budgets): [arXiv:2502.02631](https://arxiv.org/abs/2502.02631)
- LSQ (learned step size): [arXiv:1902.08153](https://arxiv.org/abs/1902.08153)
- Straight-through estimator: [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- TIES-Merging (sign election): [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)
