# Plugins

*Opt-in training enhancements. Each backed by a specific paper. All compose with torch.compile.*

---

## DEFT

Dynamic Entropy Fine-Tuning (off by default). See [Loss Functions](loss.md) for details, including a measurement where it hurt.

```yaml
plugins:
  deft: true
```

---

## SymNoise (NEFTune++)

Adds symmetric noise to embedding vectors during forward. Acts as regularization — prevents overfitting to surface-level token patterns, forces the model to learn robust representations.

```yaml
plugins:
  sym_noise: true
  sym_noise_alpha: 5.0   # Higher = stronger regularization (try 7.0 for small models)
```

"Symmetric" refers to the noise distribution: each embedding coordinate gets ±α/√(L·d) with equal probability (Bernoulli, where NEFTune uses uniform noise). Noise is added only in training mode; evaluation and inference see clean embeddings.

---

## Pre-RL mode

If your SFT model will undergo GRPO/DPO/PPO afterward, this plugin preserves the output diversity that RL needs:

- Entropy bonus: prevents the model from becoming too confident
- KL anchor: penalizes drift from the base model distribution

```yaml
plugins:
  pre_rl: true
  pre_rl_entropy_coeff: 0.1   # Strength of entropy preservation
  pre_rl_kl_coeff: 0.5        # Strength of KL anchor to base
```

!!! warning "Loss objectives don't stack"
    One objective per run: enabling more than one of `deft`, `dft`, `cadft`, `info_sft`, `pre_rl` is a configuration error. DEFT, DFT, CADFT and InfoSFT run chunked under `memory.chunked_loss` (never the full logits); `pre_rl` compares full logits with a reference snapshot, so it materializes them whatever `chunked_loss` says.

See [SFT → RL Transition](../guides/sft-to-rl.md) for the full guide.

---

## DFT / CADFT / InfoSFT

Other token-weighting schemes. DEFT contains DFT as its confident-token limit; InfoSFT and CADFT weight differently. All run chunked under `memory.chunked_loss`, and exactly one objective can be enabled.

| Plugin | Mechanism | When to use |
|--------|-----------|-------------|
| `dft` | Weight = p_θ(y_t) (model confidence) | Comparing against DEFT |
| `cadft` | DFT + sample-level compatibility score | Multi-domain data |
| `info_sft` | Weight = information content relative to calibration | Research comparison |

---

## Schedule-Free

Replaces the LR scheduler entirely with iterate averaging (Defazio et al., NeurIPS 2024). No schedule to configure, anytime stopping works.

```yaml
plugins:
  schedule_free: true
```

Requires `pip install schedulefree`. Mutually exclusive with `lr_scheduler` config.

!!! note
    Not yet validated with Muon optimizer. Use with AdamW.

---

## Trainer plugins

Your own code inside `pgs train` (SFT and [off-policy distillation](../guides/offpolicy-distillation.md)): substitute layers, add loss terms and regularisers, change teacher targets, set optimizer groups, ramp anything with training progress, log metrics, fold weights on export. A plugin is a subclass of `palingenesis.hooks.TrainerPlugin` overriding the hooks it needs; every hook is optional.

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

`palingenesis.qat.TernaryQAT` is a complete example (layer substitution, a straight-through quantiser, a strength ramp, a grid regulariser, latent clamping, optimizer groups, hidden-state targets, folded export): see [Quantisation-aware training while distilling](../guides/offpolicy-distillation.md#quantisation-aware-training-while-distilling).
