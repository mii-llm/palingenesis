# Plugins

*Opt-in training enhancements. Each backed by a specific paper. All compose with torch.compile.*

---

## DEFT

Dynamic Entropy Fine-Tuning. The default loss plugin. See [Loss Functions](loss.md) for details.

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

"Symmetric" means the noise is zero-mean and applied identically to train+eval embeddings during training, preventing a train/eval distribution mismatch.

---

## Pre-RL mode

If your SFT model will undergo GRPO/DPO/PPO afterward, this plugin preserves the output diversity that RL needs:

- Entropy bonus: prevents the model from becoming too confident
- KL anchor: penalizes drift from the base model distribution

```yaml
plugins:
  deft: false                 # required: DEFT takes precedence over pre_rl
  pre_rl: true
  pre_rl_entropy_coeff: 0.1   # Strength of entropy preservation
  pre_rl_kl_coeff: 0.5        # Strength of KL anchor to base

memory:
  chunked_loss: false         # required: pre_rl needs full logits
```

!!! warning "Loss objectives don't stack"
    The trainer picks a single loss objective per run (priority: chunked DEFT → chunked CE → CADFT → DEFT → DFT → InfoSFT → pre_rl → CE). With `deft: true` or `memory.chunked_loss: true` set, `pre_rl: true` is silently ignored.

See [SFT → RL Transition](../guides/sft-to-rl.md) for the full guide.

---

## DFT / CADFT / InfoSFT

Earlier token-weighting schemes. DEFT subsumes all of these — you probably don't need them individually. They exist for ablation studies.

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
