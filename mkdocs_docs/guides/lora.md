# LoRA

*Low-rank adapters on a frozen base, as a trainer plugin (`palingenesis.lora`).*

Full fine-tuning stays the library default. LoRA trains a low-rank delta $sBA$ next to each targeted weight $W$ and leaves $W$ frozen:

$$
y = xW^\top + b + s\,xA^\top B^\top, \qquad s = \frac{\alpha}{r}
$$

where $W$ is the frozen weight `[out, in]`, $A$ `[r, in]` and $B$ `[out, r]` are the trained factors, $r$ is `rank`, $\alpha$ is `alpha`, and the scaling $s$ is $\alpha/\sqrt{r}$ instead with `rslora: true`. With `dropout`, the adapter's input $x$ is $\operatorname{dropout}(x)$; the base path never sees dropout.
{: .pg-where }

The plugin works with `pgs train` (SFT, DPO, off-policy distillation). The export is a plain model with the adapters folded in, plus the adapter alone in the PEFT layout. `pgs rl` trains the same adapters through its own `lora:` section: see [LoRA in RL](#lora-in-rl).

!!! abstract "Use it when"
    You want many cheap adapters on one shared base, or the model does not fit with gradients and optimizer state for every weight. Otherwise full fine-tuning (the default) is usually the better tool: see [when LoRA wins](#when-lora-wins-and-when-full-fine-tuning-is-better) below.

## Minimal setup

=== "Config"

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

    `pgs init lora` writes a complete example (`lora.yaml`), then:

    ```bash
    pgs train --config lora.yaml
    ```

=== "Python"

    `palingenesis.train.train` takes plugin objects next to the ones the config lists (see [Training from a Notebook](../getting-started/notebook.md)):

    ```python
    from palingenesis.config import Config
    from palingenesis.lora import LoRA
    from palingenesis.train import train

    config = Config.from_yaml("sft.yaml")    # pgs init sft: a config without the plugin
    config.train.learning_rate = 2e-4
    config.train.weight_decay = 0.0
    result = train(config, plugins=[LoRA(rank=16, alpha=32)])
    ```

---

## When LoRA wins, and when full fine-tuning is better

LoRA is the better tool when:

- **You need many adapters on one base.** You train one adapter per task or customer, then store and serve them side by side (an adapter is megabytes, the base is shared).
- **The model does not fit with gradients and optimizer state for every weight.** Examples are a large model on one GPU, or a fine-tune next to other jobs.
- **You want less forgetting.** In "LoRA Learns Less and Forgets Less" ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)), LoRA kept more of the base model's abilities than full fine-tuning did, and learned less of the target domain. This is not automatic. "An Illusion of Equivalence" ([arXiv:2410.21228](https://arxiv.org/abs/2410.21228)) finds LoRA-specific "intruder dimensions" that can make LoRA forget more when rank, alpha or learning rate are poorly chosen.
- **You are doing RL.** Policy-gradient RL carries little information per episode. "LoRA Without Regret" (Thinking Machines, 2025) reports that rank 1 matched full fine-tuning there. See [LoRA in RL](#lora-in-rl); nothing has been measured with it here yet.

Full fine-tuning is better when:

- **The dataset carries more information than the adapter can hold.** Examples are large SFT sets, continued pretraining, and new languages or domains. "LoRA Without Regret" finds LoRA matches full fine-tuning only while the adapter's capacity exceeds what the data has to teach. Beyond that, LoRA falls behind.
- **You train with very large batches.** The same study found LoRA tolerates large batches less well than full fine-tuning.
- **Memory is not the constraint.** The library's memory tools already fit most full fine-tunes on one GPU: gradient release, 8-bit optimizers, selective checkpointing and `freeze_non_attention`. See [Philosophy](../architecture/philosophy.md).

## Options

| Argument | Default | Meaning |
|---|---|---|
| `rank` | 16 | rank $r$ of $B\,[\text{out}, r] \cdot A\,[r, \text{in}]$ |
| `alpha` | 32 | scaling $s = \alpha / r$ |
| `rslora` | false | scaling $s = \alpha / \sqrt{r}$ (rsLoRA, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) |
| `dropout` | 0.0 | dropout on the adapter's input only |
| `dora` | false | weight-decomposed LoRA (DoRA, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)): a trainable magnitude $m$ per output row (initialised to the base's row norms), $y = \dfrac{m}{\lVert W + sBA \rVert_{\text{row}}} \left(xW^\top + s\,xA^\top B^\top\right) + b$, with the row norm detached as in the paper (section 4.3) |
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
- **Keep `alpha` fixed and change `rank` freely.** With scaling $\alpha / r$ and a fixed $\alpha$ (they use 32), the optimal learning rate barely moved between ranks 4 and 512. That is why the default is `alpha: 32` rather than $\alpha = r$. With `rslora: true` the effective scale grows as $\sqrt{r}$, so retune the learning rate when you change rank.
- **Adapt every layer, the MLP especially.** Attention-only LoRA underperformed in the same study.
- **Tune the learning rate per variant before comparing variants.** "Learning Rate Matters: Vanilla LoRA May Suffice" ([arXiv:2602.04998](https://arxiv.org/abs/2602.04998)) did this for nine variants on Qwen3-0.6B, Gemma-3-1B and Llama-2-7B/13B. Their best results fell within about 0.4-1.8% of vanilla LoRA. DoRA was the one consistent small gain (~1% at rank 8, same learning-rate range). That study used $\alpha = r$ and did not test rsLoRA or LoRA+.
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

`final/` (and `best/model/` with an eval set) holds a plain model of the original architecture, loadable anywhere. Each adapter is folded into its weight in float32 ($W' = W + sBA$; DoRA: rows rescaled to $m$) and rounded to nearest in the weight's dtype:

- **float32 base:** the fold is exact. The exported model reproduces the adapted model's logits (tested to 1e-5).
- **bf16 base:** during training the delta was added in full. In the saved bf16 weight, any element of the delta smaller than half a bf16 ulp of its weight is lost (one ulp is $2^{-7}$ to $2^{-8}$ of $\lvert w \rvert$). The export logs how much of the delta the saved weights keep (the projection of `precision.edit_survival`) and warns below 0.9.
  - On a tiny random model, round-to-nearest kept 91% of a delta at $\lVert \Delta W \rVert / \lVert W \rVert \approx 3\cdot10^{-3}$, 41% at $7\cdot10^{-4}$ and 9% at $1.4\cdot10^{-4}$.
  - Stochastic rounding (`palingenesis.precision`) keeps the delta in expectation, so it is the right tool for merges that are averaged further. For reproducing one model it was worse: its noise exceeds a sub-ulp delta in a single forward. Measured as error on the adapter's effect on the logits, round-to-nearest gave 8% vs 11% with stochastic rounding at $3\cdot10^{-3}$, and 85% vs 240% at $7\cdot10^{-4}$.
  - The `plugins/lora/delta_rel` metric tracks $\lVert \Delta W \rVert / \lVert W \rVert$ during training.

The adapter alone goes to `<export>/adapter/` (`adapter_model.safetensors`, `adapter_config.json`, fp32). It loads with `peft.PeftModel.from_pretrained(base, "<output>/final/adapter")` on the causal-LM model the run trained (module names as the trainer loads it). Base plus adapter serves exactly what was trained. For Qwen3.5, whose checkpoints are multimodal, the merged export is written in the source checkpoint's layout like any export, but the adapter uses the causal LM's module names (`model.layers.*`).

Resume checkpoints carry the adapters as ordinary model weights: `train.resume_from` restores them, and their optimizer state, unchanged.

## Metrics

- `plugins/lora/trainable_params`, `plugins/lora/trainable_share`: what the optimizer trains.
- `plugins/lora/delta_rel`: the size of the adapters' update relative to the weights they adapt, over all adapted weights $\ell$:

    $$
    \texttt{delta\_rel} = \sqrt{\frac{\sum_\ell s^2 \lVert B_\ell A_\ell \rVert_F^2}{\sum_\ell \lVert W_\ell \rVert_F^2}}, \qquad \lVert BA \rVert_F^2 = \operatorname{tr}\!\big((AA^\top)(B^\top B)\big)
    $$

    The trace form uses only $r \times r$ products: the delta is never formed.

- `plugins/lora/magnitude_change` (DoRA): mean relative change of the magnitudes from the base's row norms, $\sum \lvert m - m_0 \rvert \,/\, \sum \lvert m_0 \rvert$ over every row of every adapted weight, with $m_0$ the base's row norms.

## How to check it worked

1. **At startup** the log names what was adapted and what it costs:
    - `lora: N linear modules adapted (q_proj, ...), rank 16, alpha 32, scaling 2`: the layer names it matched;
    - `lora: ... trainable parameters of ... (x.xx%)`;
    - `lora memory: frozen base ..., adapters ... + gradients ... + AdamW moments ...`, with what full fine-tuning would add.
2. **During training** `plugins/lora/delta_rel` starts at 0 (`B` is zero) and grows; the loss falls as in a full fine-tune at a well-chosen learning rate.
3. **At export** the log reports how much of the delta the saved weights keep: `lora: N adapters folded into the base weights: the saved weights keep 0.9xx of the delta`, then `lora: adapter saved (PEFT layout) -> <export>/adapter`. Below 0.9 it warns (see the table below).

## Common failures

| What you see | Cause | Fix |
|---|---|---|
| `lora: no nn.Linear matched targets / exclude; the model's linear layers are named [...]` | `targets` do not match this architecture's module names | Write `targets` from the names the error lists |
| `lora does not support train.optimizer: muon, ...` | Muon, Hyperball, MONA, base merge and float8 act on full weight matrices | Turn them off for the LoRA run (see [Interactions](#interactions)) |
| `lora: lr_ratio (LoRA+) sets one learning rate for every B matrix ...` | `lr_ratio` together with `train.llrd_decay` | Use one of them |
| Warning `rank R is not below the smaller dimension of N module(s)` | The rank is as large as a small projection | Lower `rank`, or `exclude` those modules |
| Warning at export: `the base weights' dtype cannot hold much of the adapters' delta` | A bf16 base rounds away most of a small delta | Serve the base with the adapter (`save_adapter`), or train with `model.torch_dtype: float32` ([Export](#export)) |
| Out of memory on long sequences | LoRA does not shrink activations | `train.gradient_checkpointing: full`, `memory.seco`, or measure with `pgs fit` ([Single GPU](single-gpu.md#trading-speed-for-memory-pgs-fit)) |
| Loss barely moves | Learning rate at a full fine-tune's value | About 10x a full fine-tune's rate ([above](#learning-rate-rank-and-alpha)) |

## Interactions

- **`train.freeze_non_attention`:** freezing only freezes. The base stays frozen and the adapters inside linear-attention layers freeze too, so only the attention layers' adapters train.
- **Refused at startup:** `train.optimizer: muon`, `train.hyperball`, `train.mona`, `train.base_merge` and `memory.float8_training`. They act on full weight matrices and are not defined for low-rank factors on a frozen base.
- **`lr_ratio` with `train.llrd_decay`:** also refused, since one rate for every `B` would override the per-layer rates. Plain LoRA with LLRD works.
- **DPO:** the reference model is loaded separately and is the base. With LoRA the policy is the same base plus adapters, so a run with adapters disabled could serve as the reference and skip the second copy. The trainer does not do this yet.
- **The config warning for DPO / SeCO with a bf16 model** does not apply to the adapters, which are fp32.
- **FSDP:** adapters are sharded with their layer. Only trainable parameters must share a dtype under FSDP2, and every adapter is fp32. Under `train.bf16` FSDP computes them in bf16 and reduces gradients in fp32.
- **DoRA cost:** each forward builds a weight-sized fp32 temporary per adapted layer for the row norm (no gradient), so it costs more compute and transient memory than plain LoRA.

## LoRA in RL

`pgs rl` trains adapters instead of the full weights when its config sets `lora.rank`. It uses this page's adapted layer, fold and export; the RL config has no `plugins:` section, so the options live in a section of their own.

```yaml
# rl.yaml
model:
  policy: Qwen/Qwen3.5-0.8B
lora:
  rank: 16
  alpha: 32
train:
  learning_rate: 2.0e-5        # a starting point, not measured here
```

| Option | Default | Meaning |
|---|---|---|
| `lora.rank` | 0 | rank of the adapters; 0 = off, every weight trains |
| `lora.alpha` | 32 | scaling $s = \alpha / r$ |
| `lora.rslora` | false | scaling $s = \alpha / \sqrt{r}$ |
| `lora.dora` | false | a trained magnitude per output row |
| `lora.lr_ratio` | 1.0 | LoRA+: `B` learns at this multiple of `train.learning_rate` (not with FSDP) |
| `lora.targets` | `[]` | module name patterns; empty = the defaults of [Options](#options) |
| `lora.exclude` | `[]` | patterns of modules to leave alone |

There is no `dropout` option: the trainer must score the policy the engine sampled from, and dropout on the adapters would make the two differ at every token.

What a run with adapters does differently:

- **Optimizer.** It steps the adapters only. The adapters stay float32 parameters (megabytes), so nothing trained is rounded in the module; on one GPU the frozen base is held in bf16 alone. With `train.weight_decay`, `A` and `B` are decayed as configured (the delta shrinks towards the base) and DoRA's magnitudes are not. The log line `lora: N trainable parameters of M` is printed before the first step.
- **Rollouts.** After every optimizer step the engine receives the folded weights $W + sBA$ through the same weight push as full-weight RL (in place for the in-process engine, over CUDA IPC for `vllm_server`). vLLM's own LoRA support is not used: the engine serves a plain model at its usual speed, and nothing depends on adapter hot-reloading. The fold builds new tensors from the parameters the trainer computes with, so the trainer's weights are never written. The in-process engine copies one folded weight at a time; a `vllm_server` push keeps every tensor until the server has loaded them, which is one extra bf16 copy of the adapted weights on the GPU for the time of the push.
- **Precision of the push.** The engine's weights are bf16. The part of the delta below half an ulp of its weight is rounded away in the engine, while the trainer computes with the adapters next to the unchanged base. The importance ratio treats this like any other sampler/trainer difference. `lora/push_survival` reports the share of the delta the pushed weights carry (1 = all of it), measured every 50 pushes; watch it together with `policy/mismatch_k3`. Early in a run, while the delta is tiny, a low value is expected: the engine then still samples close to the base while the trainer scores the adapted policy. A warning is logged once if the share is still below 0.5 after 50 pushes; the remedies are a higher `train.learning_rate` or `lora.alpha`, or `lora.rslora`.
- **KL reference.** With `loss.kl_ref: start` the reference is the policy with its adapters switched off, since the frozen base is the starting policy. No second copy of the model is loaded. A different checkpoint as reference, or `loss.kl_ref_reset`, keeps the frozen copy (a reset loads the folded policy into it).
- **Checkpoints.** Every checkpoint (`step_*`, `final`) holds the folded model in float32 (4 bytes per parameter, as a full-weight RL checkpoint; set `train.keep_checkpoints` accordingly), which is a plain checkpoint of the policy's architecture that loads anywhere the base does and works with the [merging tools](merging.md), and `adapter/` in the PEFT layout. A bf16 base plus the fp32 delta is exact in float32, so nothing is lost in the fold. `train.resume_from` loads `model.policy` as the base and the checkpoint's adapter on top; a checkpoint saved with another `lora.rank`, `alpha`, `rslora`, `dora`, targets or `model.policy`, or by a full-weight run, is refused.
- **Metrics.** `lora/trainable_params`, `lora/trainable_share`, `lora/delta_rel`, `lora/magnitude_change` (DoRA), `lora/push_survival`. Every other RL metric is unchanged.

Supported: one GPU and FSDP (`train.fsdp`, `train.cpu_offload`, torchrun), every `loss.type`, `model.gradient_checkpointing`, `train.prompt_sharing`, the Qwen3.5 hybrid (attention, MLP and Gated DeltaNet projections by default).

Refused at startup: `train.optimizer: muon`, `model.multimodal`, `cmr.enabled`, and `lora.lr_ratio` or `lora.dora` under FSDP.

Not verified: any run on a GPU, and so the vLLM weight push and the bf16 base next to float32 adapters under autocast (the tests run on CPU with tiny models and the `hf` backend; `tests/test_rl_lora.py`), FSDP on more than one rank, the 8-bit optimizers, and MoE policies, where the routers stay frozen and a warning is logged.

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
