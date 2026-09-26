# Configuration Reference

*Every parameter, its type, default, and when to change it. Override any value via YAML or CLI (`--section.field value`).*

---

## model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path. Any `AutoModelForCausalLM`-compatible model. |
| `trust_remote_code` | bool | `true` | Allow executing model code from HuggingFace. Required for Qwen, Gemma. |
| `torch_dtype` | str | `bfloat16` | Weight precision. Options: `bfloat16`, `float16`, `float32`. bf16 recommended for A100+. |
| `attn_implementation` | str | `sdpa` | Attention backend. `sdpa` works everywhere (packing included); `flash_attention_2` (if installed) packs without an explicit mask; `eager` for debugging. |
| `use_liger_kernel` | bool | `true` | Liger's fused Triton kernels (RMSNorm, SwiGLU, RoPE, ...) for the model's `model_type`. Applied only when `compile` is false: torch.compile fuses the same ops (faster in our measurements) and cannot trace Liger's kernels. |
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
| `packing` | bool | `false` | Pack whole conversations into sequences of up to `max_seq_length` tokens, each attending only to itself (any `attn_implementation`; linear-attention hybrids need the `hybrid` extra). See [first training](../getting-started/first-training.md#packing-when-to-use-it) for when it pays off. |
| `length_group_buffer` | int | `512` | Without packing, batches pad to their longest sample — with skewed length distributions most FLOPs go to pad tokens. Length-grouped batching buffers N samples, sorts by length, and emits batch-aligned groups so padding collapses to the within-group spread. Often a multi-× throughput win. `0` disables. Auto-disabled for `strategy: curriculum`. |
| `seed` | int | `42` | Random seed for data shuffling. |
| `sources` | list | `[]` | Multi-dataset mode. List of `{dataset, split, weight, mode, messages_field}` dicts. |
| `include_observations` | bool | `false` | **ECHO mode**: include tool/observation role tokens in loss. Teaches the model to predict tool outputs (world model). |
| `train_on_reasoning` | bool | `true` | Include reasoning traces (`<think>` blocks / `reasoning`) in the loss. Required for distilling reasoning behavior. Set `false` to train only on the post-`</think>` response. Honored identically whether the chat template uses a `{% generation %}` span (fast path) or not (fallback path). |
| `turn_scaling` | str | `uniform` | Per-turn loss weight. `uniform` (equal), `progressive` (later turns heavier, √(idx/total)), `last_heavy` (final turn 2×). Weights have mean 1 per conversation. Applied by CE, chunked CE, Cut Cross-Entropy and chunked DEFT; rejected with other objectives, SeCO and DPO. |
| `tools_field` | str | `tools` | Row field with the tool definitions (list or JSON string), passed to the chat template as `tools=`. |
| `last_turn_only` | bool | `false` | Mask every assistant turn except the final one — in training, loss only on the last answer; in `eval_sources`, score only the last answer. Phase-neutral name (no `train_`/`eval_` prefix) since the same mask serves both. Use when earlier assistant turns are a fixed context you must not fit/score (e.g. few-shot exemplar answers in eval-format SFT). No-op for single-turn data. Overridable per source in `sources`/`eval_sources`. |
| `eval_dataset` | str | `""` | Single validation dataset. Enables best-model tracking. Empty = no validation. Superseded by `eval_sources` when set. |
| `eval_sources` | list | `[]` | Per-capability eval: each source is scored independently (no cross-contamination) and combined into a weighted composite for best-model tracking. Per-source keys below. |
| `eval_split` | str | `test` | Validation split. |
| `eval_samples` | int | `200` | Number of validation samples (fixed subset). |
| `eval_every` | int | `100` | Evaluate every N optimizer steps. |
| `pretrain_replay_dataset` | str | `""` | Raw text (split `train`, field `text`, loss on every token) mixed into SFT. Empty = disabled. Not measured by us; for a chat model, replaying the model's own answers to general prompts (a second `sources` entry) reduced forgetting in our [measurements](../guides/data.md#what-the-measurements-say). |
| `pretrain_replay_weight` | float | `0.1` | Per-example probability of drawing a replay document (not a token share: replay documents run up to `max_seq_length` tokens). |
| `msft_tracking` | bool | `false` | **Not wired into training yet** (a warning is logged; weights stay fixed). Intended: adaptive per-source weight scheduling. |
| `msft_eval_every` | int | `50` | Check per-source validation loss every N steps. |
| `msft_decay_factor` | float | `0.7` | Weight multiplier when a source overfits. |
| `msft_recovery_factor` | float | `1.15` | Weight multiplier when a source improves. |
| `msft_floor_ratio` | float | `0.1` | Minimum weight (fraction of original). Never fully excludes a source. |
| `pretokenize` | bool | `false` | Materialize the fully-assembled stream (tokenize → mask → mix → pack) to disk once, then load the tensors directly on later runs — skips all per-step tokenization **and** turns the exact step-count scan into a cheap read. A fingerprint over tokenizer/template/`max_seq_length`/sources/masking auto-rebuilds a stale cache. Incompatible with `msft_tracking` (it changes the stream during training → hard error). |
| `pretokenize_path` | str | `./pretokenized` | Directory for the pre-tokenized cache (`train.parquet` + `pretokenized_meta.json`). |

!!! note "Reasoning / thinking modes"
    `train_on_reasoning` is the **only** training-time control for `<think>` content:
    `true` (default) puts loss on both the reasoning block *and* the final answer
    (needed to distil reasoning); `false` trains only the post-`</think>` answer.

    This holds **regardless of the chat template**. When the template's `{% generation %}`
    span encloses the `<think>` block (so the fast masker would otherwise train it),
    `false` strips those tokens back out; templates without a generation span reach the
    same result through the fallback masker. The two paths are behaviorally identical.

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
| `max_steps` | int | `-1` | Maximum optimizer steps. Overrides epochs if positive. When unset, the LR-schedule horizon (warmup + decay) is computed **exactly** by scanning the assembled pipeline once (steps/epoch × epochs). If the count can't be known ahead of time — `streaming: true` or a `ga_ramp` — the run is **refused** with a clear error asking you to set `max_steps` (never a silent 100k guess, which would trap short runs inside warmup). Enable `pretokenize` to make the exact-count scan cheap. |
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
| `save_final` | bool | `true` | Save the final model to `output_dir/final` (Hugging Face format) at the end. |
| `logging_steps` | int | `1` | Log metrics every N steps. |
| `bf16` | bool | `true` | Enable bf16 mixed precision with fp32 gradient reduction. |
| `gradient_checkpointing` | str | `selective` | Activation checkpointing. `selective` keeps attention outputs and every other matmul (Qwen3-0.6B, 8×2048 tokens, compiled: 8.6 GiB of activations, +10% step time); `full` recomputes each layer (1.9 GiB, +25%); `none` keeps everything (19.8 GiB, fastest). |
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
| `freeze_non_attention` | bool | `false` | Hybrid models (Qwen3.5, Qwen3-Next, Mamba hybrids): freeze the linear-attention / recurrent layers (all their parameters except the layer norms); full-attention layers, embeddings, final norm and head train. No-op, with a warning, on models without such layers. |
| `hyperball` | bool | `false` | Hyperball (arXiv:2606.16899): attention/MLP matrices keep their initial norm and move by a fixed angular step each update; embeddings, norms, biases and the head stay on the base optimizer. Any base optimizer. See [Optimizers](optimizers.md#hyperball). |
| `hyperball_lr` | float | `0.0` | Hyperball's angular step η (fraction of each matrix norm moved per update, scaled by the LR schedule). `0` = per matrix, the base optimizer's first relative step (Adam/Lion: `learning_rate / rms(W)`). |
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
| `seco` | bool | `false` | SeCO chunk-wise training for long sequences: activation memory set by `seco_chunk_size`, not sequence length. Exact gradients. Single GPU. See [Long Sequences](../guides/long-sequences.md). |
| `seco_chunk_size` | int | `4096` | Tokens per SeCO chunk. |
| `seco_kv_offload` | bool | `false` | Keep full-attention K/V (and recurrent start states) in pinned CPU memory, streamed to the GPU block by block; GPU memory then grows only with the K/V gradient. Exact. Needs `attn_implementation: sdpa`. |
| `spaco_budget` | int | `0` | SpaCO: backpropagate only this many random chunks per sequence (a stochastic gradient estimate). `0` = exact SeCO. |

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
| `pre_rl` | bool | `false` | Pre-RL mode: entropy bonus + KL anchor to preserve diversity for subsequent GRPO/DPO. Exclusive with the other objectives (a configuration error otherwise); materializes the full logits whatever `memory.chunked_loss` says. |
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
| `strategy` | str | `random` | How `budget` samples are chosen: `random` (uniform, seeded), `optimal` (heuristic J-shaped familiarity mix; measured below `random`, see the [data guide](../guides/data.md#what-the-measurements-say)), `curriculum` (random subset ordered familiar→unfamiliar, order kept at train time), `balanced`, `medium_focus`, `hard_focus`, `flow` (lowest perplexity). Unknown names are an error. |
| `group_field` | str | `""` | Row column (e.g. `source`) to profile in `prepared_meta.json`: per group, count, median response NLL, mean response tokens and familiarity buckets, before and after selection. |
| `eval_holdout` | int | `0` | Reserve N random samples as a held-out eval set (`eval_data.parquet`), excluded from the training selection. Training auto-uses it when `data.eval_dataset` is empty — a true same-distribution holdout, so `eval/loss` and `eval/gap` are trustworthy. |
| `min_ppl` | float | `1.5` | Drop samples whose response perplexity is below this (already known). |
| `max_ppl` | float | `500.0` | Drop samples above this (noise, wrong language); `<= 0` disables. |
| `filter_score` | str | `response` | Perplexity the filters use: `response` (trained tokens only) or `full` (whole conversation). |
| `batch_size` | int | `4` | Max samples per scoring forward pass. Scoring is length-sorted and padded-batch, so large values (64–256) are safe — `max_batch_tokens` bounds memory, not this. |
| `max_batch_tokens` | int | `16384` | Padded-token cap per scoring forward. Logits are batch×seq×vocab, so this is what bounds memory: 16K ≈ 5GB bf16 logits at 150K vocab. On an 80GB GPU with a ≤8B model, `32768` is safe and noticeably faster. |
| `hes` | bool | `false` | Also compute High-Entropy Sum reasoning-quality scores. Slower (second forward pass). |
| `hes_top_k_pct` | float | `0.5` | Top-k% highest-entropy tokens summed for HES. |

---

## dpo

Preference optimization (DPO and variants), a mode of `pgs train`. When `enabled`, `data.dataset` and `data.eval_dataset` hold preference pairs; optimizer, schedule, FSDP, checkpoints and chat-template masking are shared with SFT. See the [DPO guide](../guides/dpo.md).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Train on preference pairs instead of SFT conversations. |
| `loss_type` | str | `sigmoid` | `sigmoid` (DPO), `hinge` (SLiC-HF), `ipo`, `robust` (rDPO), `sigmoid_norm` (length-normalised). |
| `beta` | float | `0.1` | Inverse temperature of the implicit reward `β·(log π − log π_ref)`. |
| `label_smoothing` | float | `0.0` | Assumed label-flip rate, `robust` only. Must be < 0.5. |
| `ld_alpha` | float | `1.0` | LD-DPO: weight of the longer answer's tail beyond the shared length. `1.0` = off. |
| `sft_weight` | float | `0.0` | Adds `sft_weight` × mean NLL of the chosen answer, anchoring it (RPO). |
| `reference_model` | str | `""` | Frozen reference policy. Empty = `model.name_or_path`. |
| `prompt_field` | str | `prompt` | Prompt field. Missing/empty = implicit prompt (chosen and rejected are full conversations). |
| `chosen_field` | str | `chosen` | Preferred completion: messages, a multi-turn continuation, or a string. |
| `rejected_field` | str | `rejected` | Dispreferred completion, same forms. |
| `truncate_rejected` | bool | `true` | Truncate rejected answers longer than `data.max_seq_length` (else drop the pair). Chosen answers are never truncated. |
| `disable_dropout` | bool | `true` | Zero dropout in the policy, so its log-probs are deterministic like the reference's. |

`train.per_device_batch_size` counts **pairs**. Use `model.torch_dtype: float32`: without an fp32 master copy, DPO-sized updates round to zero in bf16. Incompatible (rejected by validation): packing, `data.sources`, `data.eval_sources`, pretokenize, MSFT, sequence-length curriculum, pretrain replay, `preprocess.enabled`, context parallel, gradient release, and the token-weighting plugins.

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

---


## Distillation config (`pgs distill`)

`pgs distill` and `pgs distill-score` use their own config (`OPDConfig`) with the same strict YAML loading and command-line overrides. Teachers and prompt sources are named mappings, overridden as `--teachers.<name>.<option>` and `--sources.<name>.<option>` (names cannot contain dots). Examples: `configs/distill_math.yaml` (shared vocabulary), `configs/distill_xtok.yaml` (cross-tokenizer), `configs/distill_multi.yaml` (two teachers), `configs/distill_opd.yaml` (multiple choice), `configs/distill_chat.yaml` (generic chat).

A config in the first OPD format (`model.teacher`, `bridge:`, `data:`, `sampling:`, `train.loss_fn`) is rejected with a message saying where each option moved.

### model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | str | — | Student model (trained; fp32 master weights, bf16 autocast). |
| `gradient_checkpointing` | bool | `false` | Recompute the student's activations in the scoring backward. |
| `use_liger_kernel` | bool | `true` | Liger's fused kernels (RMSNorm, SwiGLU, RoPE) in the student and `hf` teachers, on CUDA: about 10% faster scoring and training for Qwen3.5-4B → 0.8B, and less memory. |
| `stop_tokens` | list | `[]` | Student tokens that end a completion, besides the chat template's end-of-turn token and the eos tokens of the tokenizer and generation config. |
| `chat_template_kwargs` | dict | `{}` | Extra `apply_chat_template` arguments for every model, e.g. `{enable_thinking: false}`. |

### teachers.&lt;name&gt;

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | — | Teacher model. |
| `tokenizer` | str | `""` | Tokenizer to render and score with (empty = the model's). |
| `backend` | str | `hf` | `hf`: in-process, frozen bf16, full distribution. `vllm`: a vLLM server scoring prefill-only top-k log-probs (launched on the student's GPU, or reached at `url`). |
| `loss` | str | `""` | `full_rkl`, `topk_kl`, `sampled_rkl`, `xtok` or `rs_kd`. Empty = `full_rkl` (hf) or `topk_kl` (vllm) for a teacher sharing the student's vocabulary, `xtok` otherwise. `full_rkl` and `rs_kd` need the hf backend; all but `xtok` need a shared vocabulary (checked when the tokenizers load). |
| `device` | str | `""` | hf: device (empty = the student's), e.g. `cuda:1`. |
| `offload` | bool | `false` | hf: keep the model on CPU between scoring calls (the output head stays on the device). |
| `url` | str | `""` | vllm: an already running server. |
| `gpu_memory_utilization` | float | `0.15` | vllm: GPU fraction of a launched server. |
| `eos_map` | dict | `{}` | Shared vocabulary: student end-of-turn token → teacher's, e.g. `{"<\|im_end\|>": "<\|eot_id\|>"}`. Empty = auto (student eos → teacher eos when the student's lies outside the shared vocabulary). |
| `probe_texts` | list | `[]` | Extra texts that must tokenize identically for a shared vocabulary. |

### sources.&lt;name&gt;

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | `messages` | `messages` (chat JSONL `{"messages": [...], "answer"?: ...}`: held-out reverse KL, plus greedy accuracy for rows with an answer), `mcqa` (pool-row JSONL: letter accuracy) or `agent_traces` (recorded agent conversations, JSONL or parquet: the student regenerates their assistant turns; see the distillation guide). |
| `path` | str | — | Prompt file. |
| `weight` | float | `1.0` | Sampling weight among the sources. |
| `teacher` | str | `""` | Teacher of this source's prompts (empty = the first teacher). |
| `topic_field` | str | `""` | Row field holding a topic (e.g. `domain`), for `topic_teachers`. |
| `topic_teachers` | dict | `{}` | `{teacher: [topics]}`: rows whose topic is listed go to that teacher, the others to `teacher`. A topic may be listed under one teacher only. |
| `messages_field` / `tools_field` | str | `messages` / `tools` | agent_traces: the columns holding the conversation and the tool schemas (lists or JSON strings). |
| `branches_per_trace` | int | `8` | agent_traces: assistant turns regenerated per sampled trace (0 = all). |
| `max_context` | int | `32768` | agent_traces: longest context (tokens) a regenerated turn may have; later turns are not sampled. |
| `max_new_tokens` | int | `512` | Completion budget (mcqa: fast-template prompts). |
| `dev_size` | int | `200` | Held-out rows split off `path` (deterministic, hash-ranked, unique). |
| `dev_path` | str | `""` | A separate held-out file instead (e.g. a benchmark's test split). |
| `system_message` | str | `""` | mcqa: system message (empty = library default). |
| `fast_template` / `cot_template` | str | `""` | mcqa: prompt templates — put the benchmark's *verbatim* template here. Placeholders `{question}`/`{options}` required, `{topic}`/`{merged_letters}` optional; validated at startup. |
| `shots_path` | str | `""` | mcqa: the benchmark's official few-shot file. |
| `p_reference_shots` / `p_pool_shots` | float | `0.5` / `0.25` | mcqa: probability of the official shots / of 1–k random pool shots; the remainder is zero-shot. |
| `pool_shots_max_k` | int | `5` | mcqa: max k for the pool-shot regime. |
| `cot_fraction` | float | `0.0` | mcqa: fraction of prompts rendered with the CoT template. |
| `cot_max_new_tokens` | int | `300` | mcqa: completion budget of CoT prompts. |

### rollout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `hf` | `hf` (the trainer's `generate`), `vllm` (in-process vLLM on the student's GPU, asleep while the trainer trains) or `vllm_server` (a vLLM server fed weights over CUDA IPC; experimental). |
| `batch_prompts` | int | `32` | Prompts per optimizer step. |
| `group_size` | int | `1` | Rollouts per prompt. |
| `temperature` | float | `1.0` | Sampling temperature (the losses compare the teacher with the temperature-scaled student). |
| `max_staleness` | int | `0` | Policy versions a batch may lag behind the weights it trains. 0 = on-policy; ≥ 1 generates the next batch while the trainer trains (vllm backends). |
| `micro_seqs` | int | `64` | hf: sequences per `generate()` call. |
| `gpu_memory_utilization` | float | `0.3` | vllm: GPU fraction for the engine's weights and KV cache. |
| `max_model_len` | int | `4096` | vllm: prompt + completion tokens. |
| `prefix_caching` | bool | `false` | vllm: automatic prefix caching. For agent traces: every turn of a trace reuses the trace's one prefill (Qwen3.5 hybrids included). |
| `enforce_eager` | bool | `false` | vllm: no CUDA graphs. |
| `sleep` | bool | `true` | vllm, `max_staleness: 0`: release the engine's weights and KV cache while the trainer trains. `false` keeps it resident (no wake-up each step) when its `gpu_memory_utilization` fits beside training. |
| `url` | str | `""` | vllm_server: a running server (started with `--weight-transfer-config '{"backend": "ipc"}'` on the trainer's GPU). |

### loss

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | `8` | Teacher top-k for `topk_kl` and the xtok dense term. |
| `beta` | float | `1.0` | `topk_kl` and dense term: weight of the reverse KL (1 − beta: forward KL). |
| `is_low` / `is_high` | float | `0.5` / `2.0` | `sampled_rkl`/`xtok`: tokens whose importance ratio to the rollout policy falls outside the range get no gradient (ICE-POP). |
| `length_norm` | bool | `false` | Mean per sequence, then over sequences, instead of per token over the batch. |
| `xtok_spread` | str | `chunk` | `chunk`: every token of a chunk gets the chunk's advantage. `proportional`: token t gets A_c · log p(t) / log p(chunk). |
| `xtok_dense_weight` | float | `0.0` | Weight of a top-k KL at chunks of exactly one token on each side. |
| `mask_whitespace` | bool | `true` | xtok: no loss on whitespace-only chunks. |
| `trace_kd_weight` | float | `0.0` | agent_traces: weight of distillation on the recorded assistant turns inside each trace (off-policy, from the teacher pass the regenerated turns need anyway). |
| `rs_rounds` | int | `50` | rs_kd: tokens drawn from the teacher's distribution per position. |
| `rs_temperature` | float | `1.0` | rs_kd: the draws' proposal is the teacher's distribution raised to this power; importance weights correct for it. |
| `token_weighting` | str | `none` | Which completion tokens the loss weighs, for every loss: `none`; `sure`: w = 1 + `sure_alpha` (1 − p), p the student's probability of the sampled token; `entropy`: only the `entropy_keep` fraction of tokens with the highest student entropy in each scoring micro-batch. |
| `sure_alpha` | float | `1.0` | `sure` weighting strength (0 = none). |
| `entropy_keep` | float | `0.2` | `entropy` weighting: fraction of tokens kept. |

### train

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `./runs/opd` | Checkpoints, `opd_config.json` provenance, vLLM server logs. |
| `steps` | int | `1000` | Optimizer steps (one per batch). |
| `learning_rate` | float | `1e-6` | AdamW, no weight decay. |
| `warmup_steps` | int | `20` | Linear warmup. |
| `lr_scheduler` | str | `cosine` | `cosine` or `constant`. |
| `max_grad_norm` | float | `1.0` | Gradient clipping. |
| `seed` | int | `0` | Prompt sampling and vLLM seed. |
| `score_micro_seqs` | int | `16` | Sequences per scoring forward (student and teacher); gradient accumulation keeps the math identical. |
| `eval_every` | int | `50` | Dev metrics before training, every N steps and at the end (0 = off). |
| `eval_samples` | int | `200` | Dev prompts per source and evaluation. |
| `save_steps` | int | `0` | Resumable checkpoint (`step_N`: the model in HF format plus optimizer, policy version and random states) every N steps; 0 = only the `final` model export. |
| `keep_checkpoints` | int | `3` | Newest `step_*` dirs kept (0 = keep all; `final` exempt). |
| `tree_chunk_size` | int | `8192` | agent_traces: tokens per chunk of a trace's shared context (activation memory). |
| `tree_branch_tokens` | int | `8192` | agent_traces: padded tokens per batched forward of regenerated turns (0 = one at a time). |
| `tree_min_gap` | int | `1024` | agent_traces: the context is cut at a turn only this far past the previous cut; turns in between re-read the gap in their batched forward (fewer, larger passes; exact). |
| `resume_from` | str | `""` | A `step_*` checkpoint dir, or `auto`: the newest complete one in `output_dir`, starting fresh if there is none. |

### logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_every` | int | `1` | Train-metric cadence. |
| `use_wandb` | bool | `false` | Mirror metrics to wandb; failures degrade to console, never kill the run. |
| `project` | str | `palingenesis-opd` | wandb project. |
| `run_name` | str | `""` | wandb run name (empty = auto). |
