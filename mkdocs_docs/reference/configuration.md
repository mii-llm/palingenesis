# Configuration Reference

*Every parameter, its type, default, and when to change it. Override any value via YAML or CLI (`--section.field value`).*

---

## model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path. Any `AutoModelForCausalLM`-compatible model. |
| `trust_remote_code` | bool | `true` | Allow executing model code from HuggingFace. Required for Qwen, Gemma. |
| `torch_dtype` | str | `bfloat16` | Weight precision. Options: `bfloat16`, `float16`, `float32`. bf16 recommended for A100+. |
| `attn_implementation` | str | `sdpa` | Attention backend. `flash_attention_2` for packing with document masking, `sdpa` for compatibility, `eager` for debugging. |
| `use_liger_kernel` | bool | `true` | Fused Triton kernels for RMSNorm, SwiGLU, RoPE. ~15% speedup, zero accuracy change. Disable for unsupported architectures. |
| `compile` | bool | `true` | `torch.compile` each transformer layer. First step is slow (compilation), then 20-40% faster. |
| `compile_backend` | str | `inductor` | Compiler backend. `inductor` (default, fastest), `aot_eager` (debugging). |
| `compile_mode` | str | `default` | Compile optimization level. `default`, `reduce-overhead` (small batches), `max-autotune` (5-15% faster, longer compile). |

---

## data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset name, local `.jsonl`/`.json`/`.parquet` file, or a prepared-output directory from `pgs prepare`. |
| `dataset_split` | str | `train_sft` | Dataset split to use. |
| `streaming` | bool | `true` | Stream data (infinite, low RAM) or load to memory (finite, faster random access). |
| `max_seq_length` | int | `8192` | Maximum sequence length. Longer = more memory. Packing fills to this length. |
| `messages_field` | str | `messages` | JSON field containing chat messages. Also tries: `conversations`, `chat`, `dialogue`, `turns`. |
| `num_workers` | int | `4` | DataLoader worker processes. Increase if data loading is the bottleneck. |
| `packing` | bool | `false` | Pack multiple conversations into one sequence. 2-3× throughput. Requires `flash_attention_2` for correct masking. |
| `length_group_buffer` | int | `512` | Without packing, batches pad to their longest sample — with skewed length distributions most FLOPs go to pad tokens. Length-grouped batching buffers N samples, sorts by length, and emits batch-aligned groups so padding collapses to the within-group spread. Often a multi-× throughput win. `0` disables. Auto-disabled for `strategy: curriculum`. |
| `seed` | int | `42` | Random seed for data shuffling. |
| `sources` | list | `[]` | Multi-dataset mode. List of `{dataset, split, weight, mode, messages_field}` dicts. |
| `include_observations` | bool | `false` | **ECHO mode**: include tool/observation role tokens in loss. Teaches the model to predict tool outputs (world model). |
| `train_on_reasoning` | bool | `true` | Include reasoning traces (`<think>` blocks / `reasoning_content`) in the loss. Required for distilling reasoning behavior. Set `false` to train only on the post-reasoning response. |
| `turn_scaling` | str | `uniform` | Per-turn loss weight. `uniform` (equal), `progressive` (later turns heavier, √(idx/total)), `last_heavy` (final turn 2×). |
| `last_turn_only` | bool | `false` | Mask every assistant turn except the final one — in training, loss only on the last answer; in `eval_sources`, score only the last answer. Phase-neutral name (no `train_`/`eval_` prefix) since the same mask serves both. Use when earlier assistant turns are a fixed context you must not fit/score (e.g. few-shot exemplar answers in eval-format SFT). No-op for single-turn data. Overridable per source in `sources`/`eval_sources`. |
| `eval_dataset` | str | `""` | Single validation dataset. Enables best-model tracking. Empty = no validation. Superseded by `eval_sources` when set. |
| `eval_sources` | list | `[]` | Per-capability eval: each source is scored independently (no cross-contamination) and combined into a weighted composite for best-model tracking. Per-source keys below. |
| `eval_split` | str | `test` | Validation split. |
| `eval_samples` | int | `200` | Number of validation samples (fixed subset). |
| `eval_every` | int | `100` | Evaluate every N optimizer steps. |
| `pretrain_replay_dataset` | str | `""` | Generic pretraining data mixed during SFT (anti-forgetting). Empty = disabled. |
| `pretrain_replay_weight` | float | `0.1` | Fraction of tokens from replay data (0.1 = 10%). |
| `msft_tracking` | bool | `false` | Adaptive per-source weight scheduling. Decays overfitting sources, recovers improving ones. |
| `msft_eval_every` | int | `50` | Check per-source validation loss every N steps. |
| `msft_decay_factor` | float | `0.7` | Weight multiplier when a source overfits. |
| `msft_recovery_factor` | float | `1.15` | Weight multiplier when a source improves. |
| `msft_floor_ratio` | float | `0.1` | Minimum weight (fraction of original). Never fully excludes a source. |
| `seq_len_curriculum` | bool | `false` | Ramp max sequence length from short to full over training. |
| `seq_len_curriculum_min` | int | `1024` | Starting max sequence length during curriculum. |
| `seq_len_curriculum_ramp_steps` | int | `1000` | Steps to ramp from min to full `max_seq_length`. |

!!! note "Reasoning / thinking modes"
    `train_on_reasoning` is the **only** training-time control for `<think>` content:
    `true` (default) puts loss on both the reasoning block *and* the final answer
    (needed to distil reasoning); `false` trains only the post-`</think>` answer.

    There is deliberately **no** `enable_thinking` option here. `enable_thinking` is a
    *chat-template inference toggle* — it makes reasoning models scaffold a `<think>`
    block **during generation** — and has zero effect on which tokens receive loss.
    Non-reasoning models ignore it entirely.

    **For evaluation** you don't need it either: the in-training eval (`eval_sources`)
    is teacher-forced cross-entropy over the same masked tokens as training
    (`last_turn_only` decides which), so it never generates. Only an external
    *generation* harness needs to suppress thinking — an MCQA harness typically sets
    `enable_thinking=False` (with a `strip_think` fallback) to force a bare-letter
    answer. Where to look: the model's `tokenizer_config.json` chat template (does it
    define a thinking branch?) and the `enable_thinking=` argument in your eval harness.

!!! note "`eval_sources` per-source keys — and pick the right `mode`"
    Each entry in `eval_sources` accepts: `name`, `dataset`, `split`, `weight`
    (composite importance), `samples` (fixed subset size), `regression_floor`
    (optional alarm), and — mirroring the training `sources` — a **`mode`**:

    - **`mode: pretrain`** (+ `text_field`): raw-text, all-token CE/ppl, **no chat
      template**. Use this for language-modeling eval (e.g. held-out Italian docs). It
      matches how CPT actually trains, so the number is a true next-token perplexity.
    - **`mode: sft`** (default, + `messages_field`, optional `last_turn_only`):
      chat-templated, assistant-only CE. Use for genuine chat/MCQA tasks (e.g. an
      n-shot MCQA proxy: system + user + gold-letter assistant turn).

    Do **not** wrap plain LM text as an `assistant` message just to eval it — that
    conditions perplexity on the chat-template scaffolding and no longer measures raw
    LM. Use `mode: pretrain` with `text_field` instead.

---

## train

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `./checkpoints` | Where to save checkpoints and final model. Must be shared filesystem for multi-node. |
| `resume_from` | str\|null | `null` | Checkpoint path to resume from. `"auto"` finds the latest valid checkpoint in `output_dir`. |
| `epochs` | int | `1` | Number of training epochs. |
| `max_steps` | int | `-1` | Maximum optimizer steps. Overrides epochs if positive. When unset, the LR schedule horizon (warmup + decay) is derived automatically from epochs × dataset size for map-style datasets; for streaming datasets with no length it falls back to 100k steps with a loud warning — set `max_steps` explicitly there, or short runs never leave warmup. |
| `per_device_batch_size` | int | `1` | Sequences per GPU per forward pass. Increase if memory allows. |
| `gradient_accumulation_steps` | int | `16` | Micro-batches before optimizer step. Effective batch = batch_size × GA × num_gpus. |
| `ga_ramp_start` | int | `0` | Batch size scheduling: start GA at this value, linearly ramp to full `gradient_accumulation_steps`. 0 = constant. |
| `learning_rate` | float | `2e-5` | Peak learning rate, applied as-is for adamw/lion8bit (Lion paper suggests 3–10× *lower* than AdamW). Muon applies a 10× internal scaling to its matrix params. |
| `min_learning_rate` | float | `2e-6` | Minimum LR at end of schedule (as fraction: `min_lr / lr` is the floor ratio). |
| `weight_decay` | float | `0.1` | Decoupled weight decay coefficient. |
| `warmup_ratio` | float | `0.05` | Fraction of total steps for linear LR warmup. |
| `max_grad_norm` | float | `1.0` | Global gradient clipping norm. Disabled when `adagc: true`. |
| `lr_scheduler` | str | `cosine` | LR schedule. `power_decay` (recommended), `wsd` (long runs), `cosine`, `linear`, `constant`. |
| `optimizer` | str | `adamw` | Optimizer. `adamw`, `muon`, `lion8bit`, `adamw8bit`, `paged_adamw8bit`. |
| `seed` | int | `42` | Training random seed. |
| `save_steps` | int | `500` | Save checkpoint every N steps. Auto-purges old ones (keeps last 5). |
| `logging_steps` | int | `1` | Log metrics every N steps. |
| `bf16` | bool | `true` | Enable bf16 mixed precision with fp32 gradient reduction. |
| `gradient_checkpointing` | str | `selective` | Activation checkpointing. `selective` (best trade-off), `full` (max memory savings), `none` (fastest). |
| `spike_detection` | bool | `true` | Skip optimizer step when gradient norm is anomalous (z-score based). |
| `spike_z_threshold` | float | `5.0` | Z-score threshold for spike detection. Higher = fewer skips. |
| `adagc` | bool | `false` | Per-tensor adaptive gradient clipping. Replaces global clipping. Better for stability. |
| `adagc_lambda` | float | `1.5` | Relative clip threshold: clip if tensor norm > λ × EMA. |
| `adagc_beta` | float | `0.95` | EMA decay for per-tensor norm tracking. |
| `ema` | bool | `false` | Exponential Moving Average of weights. Better generalization. Stored on CPU. |
| `ema_decay` | float | `0.999` | EMA decay factor. 0.999 ≈ 1000-step window. |
| `ema_every` | int | `10` | Update EMA every N steps. |
| `base_merge` | bool | `false` | Periodically merge toward pretrained weights (anti-forgetting). |
| `base_merge_ratio` | float | `0.1` | Mix ratio: θ = (1-r)×θ_current + r×θ_base. |
| `base_merge_every` | int | `500` | Steps between merges. |
| `base_merge_method` | str | `lerp` | Interpolation: `lerp` (linear) or `slerp` (spherical, preserves norms). |
| `adamc` | bool | `false` | Corrected weight decay for normalized layers. Prevents gradient explosion at end of training. |
| `llrd_decay` | float | `1.0` | Layer-wise LR decay. 1.0 = off. 0.9 = early layers get 0.9× LR per depth. |
| `freeze_non_attention` | bool | `false` | Freeze all non-attention layers. For hybrid models (Qwen3.5) where only attention should be adapted. |
| `hyperball` | bool | `false` | Norm-constrained optimization. 20-30% speedup, zero memory. Applied to 2D weight matrices (not embeddings/norms). |
| `mona` | bool | `false` | MONA curvature-aware acceleration. Augments gradients with EMA of gradient differences. |
| `mona_beta_a` | float | `0.975` | MONA acceleration EMA decay. Higher for larger models (0.99 for 68B). |
| `mona_lite` | bool | `true` | Store MONA buffers in bf16 + streaming computation. 75% overhead reduction. |

---

## parallel

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fsdp` | bool | `true` | Enable FSDP2 data parallelism (multi-GPU). Auto-disabled for single GPU. |
| `context_parallel` | bool | `false` | Ring Attention for sequence parallelism. For sequences > 16K on 4+ GPUs. |
| `cp_rotate_method` | str | `allgather` | KV rotation method: `allgather` (simpler) or `alltoall` (less memory). |
| `cpu_offload` | bool | `false` | Offload FSDP parameters to CPU. Extreme memory savings but very slow. |
| `reshard_after_forward` | bool | `true` | Reshard parameters after forward. `false` = keep unsharded (more memory, less communication). |

---

## memory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunked_loss` | bool | `true` | Split CE loss into chunks to avoid materializing full [B,S,V] logit tensor. Prevents OOM on large vocab. |
| `loss_num_chunks` | int | `8` | Number of chunks. Auto-tuned based on seq_len × vocab_size. |
| `float32_matmul_precision` | str | `high` | CUDA matmul precision. `highest` (exact), `high` (TF32, recommended), `medium` (faster, less precise). |
| `float8_training` | bool | `false` | FP8 training (H100+ SM89). 1.2-1.5× throughput. |
| `gradient_release` | bool | `false` | Fuse optimizer into backward. Eliminates gradient memory. Requires GA=1, incompatible with Muon. |
| `selective_diff` | bool | `true` | Skip activation saving for frozen layers. Auto-enabled with `freeze_non_attention`. |

---

## plugins

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deft` | bool | `false` | **DEFT**: Dynamic Entropy Fine-Tuning. Parameter-free adaptive token weighting. Reports math-reasoning gains per the original paper (not independently reproduced). Recommended for reasoning tasks. |
| `dft` | bool | `false` | DFT: Dynamic Fine-Tuning. Token weight = model confidence. Predecessor to DEFT. |
| `cadft` | bool | `false` | Compatibility-Aware DFT. DFT + sample-level compatibility scoring. |
| `cadft_beta` | float | `1.0` | CADFT compatibility sensitivity. |
| `info_sft` | bool | `false` | InfoSFT: information-aware token weighting. |
| `info_sft_pbar` | float | `0.93` | InfoSFT calibration constant. |
| `sym_noise` | bool | `false` | Symmetric noisy embeddings. Regularization that prevents overfitting to surface patterns. |
| `sym_noise_alpha` | float | `5.0` | Noise magnitude. Higher = stronger regularization. Try 7.0 for small models. |
| `schedule_free` | bool | `false` | Schedule-Free optimizer. Replaces LR scheduler with iterate averaging. |
| `pre_rl` | bool | `false` | Pre-RL mode: entropy bonus + KL anchor to preserve diversity for subsequent GRPO/DPO. Requires `deft: false` and `memory.chunked_loss: false` — earlier loss branches take precedence and silently disable it. |
| `pre_rl_entropy_coeff` | float | `0.1` | Entropy bonus weight. Higher = more output diversity preserved. |
| `pre_rl_kl_coeff` | float | `0.5` | KL penalty weight. Higher = less drift from base model. |

---

## preprocess

Offline data preparation, driven by the **same config** as training (see the [Data Preparation guide](../guides/data.md)). `pgs prepare --config <file>` reuses `model.name_or_path`, `data.dataset`, `data.dataset_split`, `data.messages_field`, and `data.max_seq_length` — this section only controls selection and output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Train on the prepared output in `output_dir` instead of the raw `data.dataset`. Fails loudly if nothing was prepared yet. Incompatible with `data.sources` (use `prepare-multi` for that). |
| `output_dir` | str | `./prepared` | Where `pgs prepare` writes `scored_data.parquet` + `prepared_meta.json`, and where training looks for them. |
| `format` | str | `parquet` | Output format. `parquet` (order-preserving, fast; falls back to jsonl if the sample schema can't be unified) or `jsonl`. |
| `max_samples` | int | `0` | Cap on samples read from the raw dataset before scoring. 0 = all. |
| `budget` | int | `0` | Samples to keep after scoring and filtering. 0 = keep all. |
| `strategy` | str | `optimal` | Selection strategy: `optimal` (J-shaped, budget-adaptive: easier mix below 2K samples, full 20/50/25/5 above 10K, backfills short buckets), `curriculum` (easy→hard, order preserved at train time), `balanced`, `medium_focus`, `hard_focus`, `flow`, `random`. |
| `eval_holdout` | int | `0` | Reserve N random samples as a held-out eval set (`eval_data.parquet`), excluded from the training selection. Training auto-uses it when `data.eval_dataset` is empty — a true same-distribution holdout, so `eval/loss` and `eval/gap` are trustworthy. |
| `batch_size` | int | `4` | Max samples per scoring forward pass. Scoring is length-sorted and padded-batch, so large values (64–256) are safe — `max_batch_tokens` bounds memory, not this. |
| `max_batch_tokens` | int | `16384` | Padded-token cap per scoring forward. Logits are batch×seq×vocab, so this is what bounds memory: 16K ≈ 5GB bf16 logits at 150K vocab. On an 80GB GPU with a ≤8B model, `32768` is safe and noticeably faster. |
| `hes` | bool | `false` | Also compute High-Entropy Sum reasoning-quality scores. Slower (second forward pass). |
| `hes_top_k_pct` | float | `0.5` | Top-k% highest-entropy tokens summed for HES. |

---

## logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | str | `palingenesis` | Project name for wandb/trackio. |
| `run_name` | str\|null | `null` | Run name. Auto-generated from model name if null. |
| `use_wandb` | bool | `true` | Log to Weights & Biases. |
| `use_trackio` | bool | `true` | Log to trackio. |
| `log_grad_norm` | bool | `true` | Include gradient norm in logs. |
| `health_tier2_every` | int | `10` | Steps between tier-2 health checks (grad cosine sim, GNS, CUDA memory). Should be a multiple of `train.logging_steps`. |
| `health_tier3_every` | int | `100` | Steps between tier-3 health checks (weight norms, stable rank, weight drift). Lower it for short test runs — otherwise a run under 100 steps never produces tier-3 metrics. |
| `rl_readiness` | bool | `false` | Monitor output entropy for SFT→RL readiness. Warns if entropy collapses. |
| `rl_entropy_floor` | float | `1.0` | Warning threshold. Alert when mean entropy drops below this. |

!!! info "Tracker behavior (automatic, no config)"
    - **Crash-resume continues the same run; fresh runs get a new one.** The wandb run id is persisted to `{output_dir}/tracker_run_id.json`. Resuming from a checkpoint (`train.resume_from`) appends to the existing wandb run; starting fresh in the same `output_dir` mints a new run id (reattaching to the old run would make wandb silently drop every row below the old history step).
    - **No step-monotonicity data loss.** Metrics are logged without an explicit wandb step; the x-axis is the `train/global_step` value inside each payload (via `define_metric`), so rows can never be dropped for being "out of order".
    - **A broken tracker never kills training.** Both backends are wrapped: init or log failures degrade to a warning and training continues.
    - **All metrics share one x-axis.** `train/*`, `eval/*`, and `health/*` are aligned on `train/global_step`, across restarts.

---

## CLI overrides

Any parameter can be overridden from the command line:

```bash
pgs train --config base.yaml \
    --train.learning_rate 1e-5 \
    --train.epochs 3 \
    --data.packing true \
    --plugins.deft true
```

Overrides are applied after the YAML is loaded.
