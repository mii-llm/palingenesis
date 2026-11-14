# LoRA

*Low-rank adapters on a frozen base, as a trainer plugin (`palingenesis.lora`).*

Full fine-tuning stays the library default. LoRA trains a low-rank delta `scaling · B A` next to each targeted weight `W` and leaves `W` frozen. The plugin works with `pgs train` (SFT, DPO, off-policy distillation). The export is a plain model with the adapters folded in, plus the adapter alone in the PEFT layout.

```yaml
plugins:
  custom:
    - class: palingenesis.lora:LoRA     # entry point name: lora
      args: {rank: 16, alpha: 32}
train:
  learning_rate: 2e-4                   # ~10x a full fine-tune's rate
  weight_decay: 0.0
model:
  torch_dtype: bfloat16                 # the frozen base needs no fp32 copy; adapters are fp32
```

`pgs init lora` writes a complete example.

---

## When LoRA wins, and when full fine-tuning is better

LoRA is the better tool when:

- **You need many adapters on one base.** You train one adapter per task or customer, then store and serve them side by side (an adapter is megabytes, the base is shared).
- **The model does not fit with gradients and optimizer state for every weight.** Examples are a large model on one GPU, or a fine-tune next to other jobs.
- **You want less forgetting.** In "LoRA Learns Less and Forgets Less" ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)), LoRA kept more of the base model's abilities than full fine-tuning did, and learned less of the target domain. This is not automatic. "An Illusion of Equivalence" ([arXiv:2410.21228](https://arxiv.org/abs/2410.21228)) finds LoRA-specific "intruder dimensions" that can make LoRA forget more when rank, alpha or learning rate are poorly chosen.
- **You are doing RL.** Policy-gradient RL carries little information per episode. "LoRA Without Regret" (Thinking Machines, 2025) reports that rank 1 matched full fine-tuning there. The RL trainer does not load this plugin yet (see below).

Full fine-tuning is better when:

- **The dataset carries more information than the adapter can hold.** Examples are large SFT sets, continued pretraining, and new languages or domains. "LoRA Without Regret" finds LoRA matches full fine-tuning only while the adapter's capacity exceeds what the data has to teach. Beyond that, LoRA falls behind.
- **You train with very large batches.** The same study found LoRA tolerates large batches less well than full fine-tuning.
- **Memory is not the constraint.** The library's memory tools already fit most full fine-tunes on one GPU: gradient release, 8-bit optimizers, selective checkpointing and `freeze_non_attention`. See [Philosophy](../architecture/philosophy.md).

## Options

| Argument | Default | Meaning |
|---|---|---|
| `rank` | 16 | rank `r` of `B [out, r] · A [r, in]` |
| `alpha` | 32 | scaling `alpha / r` |
| `rslora` | false | scaling `alpha / sqrt(r)` (rsLoRA, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) |
| `dropout` | 0.0 | dropout on the adapter's input only |
| `dora` | false | weight-decomposed LoRA (DoRA, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)): a trainable magnitude per output row, `y = m / ‖W + scaling·BA‖_row · (xWᵀ + scaling·x Aᵀ Bᵀ) + b`, with the row norm detached as in the paper (section 4.3) |
| `lr_ratio` | 1.0 | LoRA+ ([arXiv:2402.12354](https://arxiv.org/abs/2402.12354)): `B` learns at `lr_ratio × train.learning_rate`, `A` at the base rate |
| `targets` | attention, MLP and Gated DeltaNet projections | regular expressions, each matched against a whole trailing part of the module name: `q_proj`, `self_attn\.q_proj`, `layers\.(0\|1)\.mlp\..*` |
| `exclude` | none | regular expressions (searched anywhere in the name) that override `targets` |
| `save_adapter` | true | also write `<export>/adapter/` in the PEFT layout |
| `metrics_every` | 50 | steps between adapter-norm metrics |

The default targets are `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (Llama, Qwen, Mistral-style models) plus the Gated DeltaNet projections: `in_proj_qkv in_proj_z out_proj` for Qwen3.5 and `in_proj_qkvz` for Qwen3-Next. The per-head gate projections (`in_proj_a`, `in_proj_b`, `in_proj_ba`) have only a few rows each, so they are not targeted. The output head and the input embedding are never adapted, because the chunked and cut cross-entropy losses read `lm_head.weight` directly. Fused MoE expert weights (3-D parameters, not `nn.Linear`) are not adapted either.

`A` is Kaiming-uniform (seeded from `train.seed` and the module name, so every rank and world size gets the same init). `B` is zero, so the adapted model equals the base at step 0.

## Learning rate, rank and alpha

The learning rate matters more than the choice of variant.

- **Set it about 10x higher than a full fine-tune's.** "LoRA Without Regret" found the optimal LoRA learning rate is about 10x the full fine-tuning optimum (about 15x for runs under ~100 steps).
- **Keep `alpha` fixed and change `rank` freely.** With scaling `alpha / r` and a fixed `alpha` (they use 32), the optimal learning rate barely moved between ranks 4 and 512. That is why the default is `alpha: 32` rather than `alpha = r`. With `rslora: true` the effective scale grows as `sqrt(r)`, so retune the learning rate when you change rank.
- **Adapt every layer, the MLP especially.** Attention-only LoRA underperformed in the same study.
- **Tune the learning rate per variant before comparing variants.** "Learning Rate Matters: Vanilla LoRA May Suffice" ([arXiv:2602.04998](https://arxiv.org/abs/2602.04998)) did this for nine variants on Qwen3-0.6B, Gemma-3-1B and Llama-2-7B/13B. Their best results fell within about 0.4-1.8% of vanilla LoRA. DoRA was the one consistent small gain (~1% at rank 8, same learning-rate range). That study used `alpha = r` and did not test rsLoRA or LoRA+.
- **On weight decay:** decay on the factors also shrinks `A`, which starts random, not only the update. Most LoRA recipes use 0. The example config does.

## Memory: what LoRA saves and what it does not

LoRA removes the base weights' **gradients and optimizer state**. With AdamW that is up to 3 extra copies of every weight. Only the adapters (fp32) carry a gradient and two moments. The frozen base stays in its storage dtype (bf16 is fine: it is never updated). At startup the plugin logs this picture: frozen base size, adapter, gradient and optimizer bytes, and what full fine-tuning would add.

LoRA does **not** shrink activations. Each adapted layer still keeps its input for the backward, because `A` needs it. A long sequence costs about what it costs with full fine-tuning. For that, use `train.gradient_checkpointing`, `memory.seco` ([Long Sequences](long-sequences.md)) or `pgs fit`.

Measured on an H100 with Qwen3.5-0.8B on agentic traces of up to 16,384 tokens (one row per micro-batch, rank 16 on every projection, bf16), steady-state throughput and peak allocated memory:

| | Checkpointing | Tokens/s | Peak |
|---|---|:---:|:---:|
| Full fine-tuning | `selective` | 23,500 | 16.8 GiB |
| LoRA | `selective` | 18,700 | 20.0 GiB |
| LoRA | `full` | 19,100 | 8.0 GiB |
| LoRA | `none` | 23,400 | 27.3 GiB |
| DoRA | `selective` | 13,900 | 27.0 GiB |

On a small model with long sequences LoRA is not the cheaper option by default: the adapters add operations and saved activations to every layer, and the weights it stops updating were a small share of the memory. With `gradient_checkpointing: full` it halves the peak of full fine-tuning at about 80% of its speed, and that is the setting to use for LoRA on long sequences (`pgs fit` measures it for your config). DoRA recomputes every adapted weight's row norms at each forward: budget about 25% less throughput and more memory.

## Export

`final/` (and `best/model/` with an eval set) holds a plain model of the original architecture, loadable anywhere. Each adapter is folded into its weight in float32 (`W + scaling·BA`; DoRA: rows rescaled to `m`) and rounded to nearest in the weight's dtype:

- **float32 base:** the fold is exact. The exported model reproduces the adapted model's logits (tested to 1e-5).
- **bf16 base:** during training the delta was added in full. In the saved bf16 weight, any element of the delta smaller than half a bf16 ulp of its weight is lost (one ulp is 2⁻⁷ to 2⁻⁸ of |w|). The export logs how much of the delta the saved weights keep (the projection of `precision.edit_survival`) and warns below 0.9.
  - On a tiny random model, round-to-nearest kept 91% of a delta at ‖ΔW‖/‖W‖ ≈ 3·10⁻³, 41% at 7·10⁻⁴ and 9% at 1.4·10⁻⁴.
  - Stochastic rounding (`palingenesis.precision`) keeps the delta in expectation, so it is the right tool for merges that are averaged further. For reproducing one model it was worse: its noise exceeds a sub-ulp delta in a single forward. Measured as error on the adapter's effect on the logits, round-to-nearest gave 8% vs 11% with stochastic rounding at 3·10⁻³, and 85% vs 240% at 7·10⁻⁴.
  - The `plugins/lora/delta_rel` metric tracks ‖ΔW‖/‖W‖ during training.

The adapter alone goes to `<export>/adapter/` (`adapter_model.safetensors`, `adapter_config.json`, fp32). It loads with `peft.PeftModel.from_pretrained(base, "<output>/final/adapter")` on the causal-LM model the run trained (module names as the trainer loads it). Base plus adapter serves exactly what was trained. For Qwen3.5, whose checkpoints are multimodal, the merged export is written in the source checkpoint's layout like any export, but the adapter uses the causal LM's module names (`model.layers.*`).

Resume checkpoints carry the adapters as ordinary model weights: `train.resume_from` restores them, and their optimizer state, unchanged.

## Metrics

- `plugins/lora/trainable_params`, `plugins/lora/trainable_share`: what the optimizer trains.
- `plugins/lora/delta_rel`: ‖scaling·BA‖_F / ‖W‖_F over all adapted weights. It is computed as `tr((AAᵀ)(BᵀB))`, never by forming the delta.
- `plugins/lora/magnitude_change` (DoRA): mean relative change of the magnitudes from the base's row norms.

## Interactions

- **`train.freeze_non_attention`:** freezing only freezes. The base stays frozen and the adapters inside linear-attention layers freeze too, so only the attention layers' adapters train.
- **Refused at startup:** `train.optimizer: muon`, `train.hyperball`, `train.mona`, `train.base_merge` and `memory.float8_training`. They act on full weight matrices and are not defined for low-rank factors on a frozen base.
- **`lr_ratio` with `train.llrd_decay`:** also refused, since one rate for every `B` would override the per-layer rates. Plain LoRA with LLRD works.
- **DPO:** the reference model is loaded separately and is the base. With LoRA the policy is the same base plus adapters, so a run with adapters disabled could serve as the reference and skip the second copy. The trainer does not do this yet.
- **The config warning for DPO / SeCO with a bf16 model** does not apply to the adapters, which are fp32.
- **FSDP:** adapters are sharded with their layer. Only trainable parameters must share a dtype under FSDP2, and every adapter is fp32. Under `train.bf16` FSDP computes them in bf16 and reduces gradients in fp32.
- **DoRA cost:** each forward builds a weight-sized fp32 temporary per adapted layer for the row norm (no gradient), so it costs more compute and transient memory than plain LoRA.

## Variants shipped and left out

Shipped:

- LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- rsLoRA ([arXiv:2312.03732](https://arxiv.org/abs/2312.03732))
- DoRA ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353))
- LoRA+ ([arXiv:2402.12354](https://arxiv.org/abs/2402.12354))
- adapter dropout

Left out:

- **Initialisations from the base's SVD or from data:**
  - PiSSA ([2404.02948](https://arxiv.org/abs/2404.02948)), OLoRA ([2406.01775](https://arxiv.org/abs/2406.01775)) and MiLoRA ([2406.09044](https://arxiv.org/abs/2406.09044)) modify the base weights. With the learning rate tuned per method they did not beat vanilla LoRA ([2602.04998](https://arxiv.org/abs/2602.04998)). PiSSA needed ~10x and OLoRA ~18x lower learning rates.
  - LoRA-GA ([2407.05000](https://arxiv.org/abs/2407.05000)), EVA ([2410.07170](https://arxiv.org/abs/2410.07170)) and LoRA-One ([2502.01235](https://arxiv.org/abs/2502.01235)) need a pass over the data before training. LoRA-GA also showed no gain once the learning rate was tuned.
- **Rank allocation and shared or tiny parameterisations:** AdaLoRA ([2303.10512](https://arxiv.org/abs/2303.10512)), VeRA ([2310.11454](https://arxiv.org/abs/2310.11454)) and LoRA-XS ([2405.17604](https://arxiv.org/abs/2405.17604)). They save adapter bytes, which are rarely the constraint.
- **Optimizer-side corrections:** LoRA-Pro ([2407.18242](https://arxiv.org/abs/2407.18242)) and LoFT. LoFT was in the re-evaluation above with no gain.
- **Full-rank or high-rank reparameterisations:** RandLoRA ([2502.00987](https://arxiv.org/abs/2502.00987)), GraLoRA and HiRA. RandLoRA and GraLoRA were re-evaluated with no gain. HiRA has no independent replication we could find. Several 2026 preprints were left out for the same reason.
- **QLoRA** ([2305.14314](https://arxiv.org/abs/2305.14314)): a 4-bit base needs bitsandbytes kernels in the forward. It is not supported.

## What was verified, and what was not

Verified on CPU (`tests/test_lora.py`, tiny Llama and Qwen3.5 hybrid models):

- the adapted model equals the base at init (LoRA, rsLoRA, DoRA)
- only adapters get gradients and optimizer state
- the fold reproduces the adapted logits (fp32), and bf16 folding behaves as described above
- the exported `final/` from a trainer run matches the in-memory adapted model
- the training loss falls on fixed data
- resume restores the adapters
- LoRA+ groups are built as specified
- full and selective activation checkpointing give the same adapter gradients as the plain forward
- `freeze_non_attention` and the Gated DeltaNet layers work as described above
- a DPO run completes
- on one GPU under bf16 autocast (Qwen3.5-0.8B, 16k-token rows, LoRA and DoRA, every checkpointing mode): the loss falls as with full fine-tuning; the numbers above
- PEFT loads the exported adapter and reproduces the merged model (when peft is installed)

Not verified:

- runs with `model.compile`, `memory.seco`, context parallel or FSDP across several ranks
- quality on real models and datasets: the learning-rate guidance above comes from the cited studies, not from our own runs
- the RL and on-policy distillation trainers do not load trainer plugins, so LoRA is not available there yet
