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
