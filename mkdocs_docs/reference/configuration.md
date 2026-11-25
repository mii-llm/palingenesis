# Configuration Reference

*Every parameter, its type, default, and when to change it. Override any value via YAML or CLI (`--section.field value`).*

---

## model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path. Any `AutoModelForCausalLM`-compatible model; with [`multimodal.enabled`](#multimodal), a Qwen-VL family checkpoint in its multimodal architecture. |
| `trust_remote_code` | bool | `true` | Allow executing model code from HuggingFace. Required for Qwen, Gemma. |
| `torch_dtype` | str | `bfloat16` | Weight precision. Options: `bfloat16`, `float16`, `float32`. bf16 recommended for A100+. |
| `attn_implementation` | str | `sdpa` | Attention backend. `sdpa` works everywhere (packing included); `flash_attention_2`, `flash_attention_3` (Hopper) or `flash_attention_4` (Hopper / Blackwell) through transformers' flash path (if installed; otherwise training stops with a config error before loading the model) pack without an explicit mask; a Hub kernel `org/repo[@revision]` loaded through the `kernels` package, e.g. `kernels-community/aiter-flash-attn@v2` on [AMD GPUs](../getting-started/amd.md); `eager` for debugging. |
| `flash_kernel` | str | `builtin` | The flash kernel behind `sdpa` (dense, packed variable-length rows and SeCO chunks alike): `builtin` (torch's FlashAttention-2), `fa3` (FlashAttention-3, Hopper; the flash-attn-3 package), `fa4` (FlashAttention-4, Hopper / Blackwell; flash-attn-4), or `auto` (FA3 on Hopper, FA4 on Blackwell, when installed; else builtin). Swapped in through torch's `activate_flash_attention_impl` and checked at startup against the built-in kernel on the model's head shape; an explicit `fa3` / `fa4` that cannot run stops training with a config error. Only with `attn_implementation: sdpa`. See [Single GPU](../guides/single-gpu.md#attention-kernels-on-hopper-and-blackwell). |
| `use_liger_kernel` | bool | `true` | Liger's fused Triton kernels (RMSNorm, SwiGLU, RoPE, ...) for the model's `model_type`. Applied only when `compile` is false: torch.compile fuses the same ops (faster in our measurements) and cannot trace Liger's kernels. |
| `compile` | bool | `true` | `torch.compile` each transformer layer. First step is slow (compilation), then 20-40% faster on fixed-shape batches. Measure it on long rows of varying length: on Qwen3.5-0.8B with agentic traces of up to 16k tokens (H100) it was 15% slower than `compile: false` with Liger's kernels (19.4k vs 22.9k tokens/s over 400 steps). |
| `compile_backend` | str | `inductor` | Compiler backend. `inductor` (default, fastest), `aot_eager` (debugging). |
| `compile_mode` | str | `default` | Compile optimization level. `default`, `reduce-overhead` (small batches), `max-autotune` (5-15% faster, longer compile). |
| `init` | str | `pretrained` | Where the weights come from. `pretrained`: the checkpoint at `name_or_path`. `random`: the architecture of `name_or_path`'s `config.json` with freshly initialised weights (training from scratch); the path then needs only `config.json` and the tokenizer files, and weights found there are ignored. Each rank builds the model on the CPU in `torch_dtype` (host memory: one copy per rank, as when loading a checkpoint). `pgs train` only. Rejected with `dpo.enabled`, `multimodal.enabled`, `train.base_merge`, `train.freeze_non_attention`, `plugins.pre_rl` and model scoring in `preprocess`. See [Training from random weights](../guides/pretraining.md#training-from-random-weights). |
| `init_seed` | int | `-1` | Seed of the random initialisation, the same on every rank. `-1` = `train.seed`. |

---

## data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `HuggingFaceH4/ultrachat_200k` | HuggingFace dataset name, local `.jsonl`/`.json`/`.parquet` file, or a prepared-output directory from `pgs prepare`. |
| `dataset_split` | str | `train_sft` | Dataset split to use. |
| `streaming` | bool | `true` | Stream the data instead of loading it (Arrow, memory-mapped). Local JSONL and parquet files stream as many shards read in parallel by ranks and workers; the run is still sized from the files' metadata. |
| `mix_epoch` | str | `"total"` | What an epoch of `sources` is: `total` = as many rows as the sources hold together, drawn by weight (a source without a weight gets its share of the rows; sources above their share repeat, those below contribute a fresh random subset each epoch). A row counts when its last training sequence is drawn, so a raw-text document that is split into several chunks counts once, with all its chunks: one raw-text source gives one pass over every chunk of every document. `first_exhausted` = until a source runs out. |
| `mix_weights` | str | `"sequences"` | What a source's `weight` is a share of. `sequences`: of the training sequences drawn, so a source of long sequences takes a larger share of the tokens than its weight. `tokens`: of the tokens trained on: each source is drawn in proportion to its weight divided by the mean length of its sequences, measured on the sequences read so far in the epoch, so `weight: 0.8` is 80% of the tokens whatever the documents' lengths (every token of a sequence counts, a conversation's prompt included). `tokens` needs a weight on every source. Sources without any weight are mixed by their rows either way. See [Mixing sources](../guides/pretraining.md#mixing-sources). |
| `max_seq_length` | int | `8192` | Maximum sequence length. Longer = more memory. Packing fills to this length. |
| `messages_field` | str | `messages` | JSON field containing chat messages. Also tries: `conversations`, `chat`, `dialogue`, `turns`. |
| `num_workers` | int | `4` | DataLoader worker processes. Increase if data loading is the bottleneck. |
| `packing` | bool | `false` | Pack whole conversations into sequences of up to `max_seq_length` tokens, each attending only to itself (any `attn_implementation`; linear-attention hybrids need the `hybrid` extra). With `memory.seco`, every packed document is trained on its own, with no kernel requirement (see [Long Sequences](../guides/long-sequences.md#rows-of-very-different-lengths-packing)). See [first training](../getting-started/first-training.md#packing-when-to-use-it) for when it pays off. |
| `length_group_buffer` | int | `512` | Without packing, batches pad to their longest sample — with skewed length distributions most FLOPs go to pad tokens. Length-grouped batching buffers N samples, sorts by length, and emits batch-aligned groups so padding collapses to the within-group spread. Often a multi-× throughput win. `0` disables. Auto-disabled for `strategy: curriculum`. |
| `seed` | int | `42` | Random seed for data shuffling. |
| `sources` | list | `[]` | Multi-dataset mode. List of `{dataset, split, weight, mode, messages_field}` dicts. |
| `include_observations` | bool | `false` | Observation loss (after ECHO, arXiv:2605.24517): the text of tool / observation messages also gets loss (never the template's markup), so the model is trained to predict what the environment returns. Untested in SFT; see the [agentic guide](../guides/agentic-training.md#observation-loss-include_observations). |
| `observation_weight` | float | `0.05` | λ of the observation loss. Before this option existed, `include_observations` trained the tool messages at the assistant's weight. |
| `observation_normalization` | str | `conversation` | `conversation`: each conversation adds λ × the mean NLL of its observation tokens to its assistant loss, and the loss's token count is the assistant's tokens only. `token`: observation tokens weigh λ each and count in the denominator (`token` with `observation_weight: 1.0` is the old behaviour). |
| `train_on_reasoning` | bool | `true` | Include reasoning traces (`<think>` blocks / `reasoning`) in the loss. Required for distilling reasoning behavior. Set `false` to train only on the post-`</think>` response. Honored identically whether the chat template uses a `{% generation %}` span (fast path) or not (fallback path). |
| `turn_scaling` | str | `uniform` | Per-turn loss weight. `uniform` (equal), `progressive` (later turns heavier, $\sqrt{\text{idx}/\text{total}}$), `last_heavy` (final turn 2×). Weights have mean 1 per conversation. Applied by CE, chunked CE, Cut Cross-Entropy, chunked DEFT and SeCO; rejected with other objectives and DPO. |
| `tools_field` | str | `tools` | Row field with the tool definitions (list or JSON string), passed to the chat template as `tools=`. |
| `loop_recovery` | float | `0.0` | Share of agent rows (0–1) that get a repeated tool call: about one site per 10 calls, the call and its observations pasted again 1–3 times without loss, the gold continuation trained. Teaches the escape from a repeated action. Load-time only (files untouched); a row whose copies would not fit `max_seq_length` stays unaugmented. See [Agentic training](../guides/agentic-training.md#repetition-loops-and-runaway-episodes). |
| `reasoning_loop_recovery` | float | `0.0` | Share of rows (0–1) where one reasoning sentence in about every third trained turn is written again 1–4 times without loss; the rest of the reasoning is trained. Works with `reasoning_content` and `reasoning` fields. |
| `repeat_unlikelihood` | float | `0.0` | Weight of $-\log(1 - p)$ on the token that would continue a repeat, at the positions where the gold stops repeating: after an exact repeat of a call or sentence, or where a near-repeat diverges inside an argument value or sentence. Each target counts like this many tokens. Requires `memory.seco: true`, no `pretokenize`. Logged as `train/repeat_p`, `train/repeat_targets`. |
| `stop_unlikelihood` | float | `0.0` | Weight of $-\log(1 - p)$ on starting another tool call in the final reply of a trace that used tools (the call opener at the reply start and after each separator, the separator at the end of the turn). The opener and separator are derived from the chat template (an added or special token is required, otherwise the term is off with a warning). Same requirements as `repeat_unlikelihood`; logged as `train/stop_p`, `train/stop_targets`. |
| `last_turn_only` | bool | `false` | Mask every assistant turn except the final one — in training, loss only on the last answer; in `eval_sources`, score only the last answer. Phase-neutral name (no `train_`/`eval_` prefix) since the same mask serves both. Use when earlier assistant turns are a fixed context you must not fit/score (e.g. few-shot exemplar answers in eval-format SFT). No-op for single-turn data. Overridable per source in `sources`/`eval_sources`. |
| `eval_dataset` | str | `""` | Single validation dataset. Enables best-model tracking. Empty = no validation. Superseded by `eval_sources` when set. |
| `eval_sources` | list | `[]` | Per-capability eval: each source is scored independently (no cross-contamination) and combined into a weighted composite for best-model tracking. Logged as `eval/<name>/loss` next to the composite `eval/loss`, and on the console log line as `eval=<composite> eval_sources=<name>:<loss>,...`. A source that cannot be read, has an unknown `mode` or gives no usable sample stops the run with a config error (it is never skipped). Per-source keys below. |
| `eval_split` | str | `test` | Validation split. |
| `eval_samples` | int | `200` | Number of validation samples (fixed subset). |
| `eval_every` | int | `100` | Evaluate every N optimizer steps. |
| `pretrain_replay_dataset` | str | `""` | Raw text (split `train`, field `text`, loss on every token; no end-of-document token is appended, unlike `mode: pretrain` sources) mixed into SFT. An epoch is one pass over the target data's rows, with the replay sequences drawn alongside. Empty = disabled. Not measured by us; for a chat model, replaying the model's own answers to general prompts (a second `sources` entry) reduced forgetting in our [measurements](../guides/data.md#what-the-measurements-say). |
| `pretrain_replay_weight` | float | `0.1` | Per-sequence probability of drawing a replay sequence (not a token share: a replay document longer than `max_seq_length` is split into chunks of that length, each a sequence). |
| `msft_tracking` | bool | `false` | **Not wired into training yet** (a warning is logged; weights stay fixed). Intended: adaptive per-source weight scheduling. |
| `msft_eval_every` | int | `50` | Check per-source validation loss every N steps. |
| `msft_decay_factor` | float | `0.7` | Weight multiplier when a source overfits. |
| `msft_recovery_factor` | float | `1.15` | Weight multiplier when a source improves. |
| `msft_floor_ratio` | float | `0.1` | Minimum weight (fraction of original). Never fully excludes a source. |
| `pretokenize` | bool | `false` | Materialize the fully-assembled stream (tokenize → mask → mix → pack) to disk once, then load the tensors directly on later runs — skips all per-step tokenization **and** gives the exact step count from the cache's row count. A fingerprint over tokenizer/template/`max_seq_length`/sources/masking auto-rebuilds a stale cache. Incompatible with `msft_tracking` (it changes the stream during training → hard error). |
| `pretokenize_path` | str | `./pretokenized` | Directory for the pre-tokenized cache (`train.parquet` + `pretokenized_meta.json`). Must be on a shared filesystem for multi-node: rank 0 builds it, every rank reads it. |

!!! note "`sources` per-source keys"
    Each entry in `sources` accepts: `dataset`, `split` (default `train`), `name`, `weight`
    (see `mix_weights`), `mode` (`sft`, the default, or `pretrain`), and per mode:

    - **`mode: sft`**: `messages_field`, `tools_field`, `last_turn_only`, `think_tags`,
      `chat_template_kwargs` (each overrides the `data` option of the same name).
    - **`mode: pretrain`**: `text_field` (default `text`) and `append_eos` (default `true`):
      the tokenizer's EOS token is appended to every document that does not already end
      with it, as the last token of the document's last chunk, and is trained like any
      other token (a warning is logged when that token is the chat template's end-of-turn
      token). `append_eos: "<|endoftext|>"` names another token of the vocabulary
      (for a chat model whose EOS is its end-of-turn token); `false` appends nothing.

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
    Each entry in `eval_sources` accepts: `name`, `dataset` (a local `.jsonl`, `.json` or
    `.parquet` file, a `pgs prepare` output directory, or a Hub dataset), `split`, `weight`
    (composite importance), `samples` (fixed subset size), `regression_floor`
    (optional alarm), and — mirroring the training `sources` — a **`mode`**:

    - **`mode: pretrain`** (+ `text_field`, `append_eos`): raw-text, all-token CE/ppl, **no chat
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
| `max_steps` | int | `-1` | Maximum optimizer steps. Overrides epochs if positive. When unset, the LR-schedule horizon (warmup + decay) is **estimated** from the data's row counts and a sample of rows through the pipeline (a few seconds, typically ±1–2%; streaming included), so training starts at once; see the data guide's "Run length and streaming". A Hub stream without published sizes, or a `ga_ramp`, needs `max_steps`. |
| `exact_steps` | bool | `false` | Scan the whole assembled pipeline once to count the steps exactly instead of estimating them (an extra pass over the data before training; not possible when streaming). |
| `per_device_batch_size` | int | `1` | Sequences per GPU per forward pass. Increase if memory allows. |
| `gradient_accumulation_steps` | int | `16` | Micro-batches before optimizer step. Effective batch = `per_device_batch_size` $\times$ GA $\times$ GPUs. |
| `ga_ramp_start` | int | `0` | Batch size scheduling: start GA at this value, linearly ramp to full `gradient_accumulation_steps`. 0 = constant. |
| `learning_rate` | float | `2e-5` | Peak learning rate, applied as-is for adamw/lion8bit (Lion paper suggests 3–10× *lower* than AdamW). Muon applies a 10× internal scaling to its matrix params. |
| `min_learning_rate` | float | `2e-6` | Minimum LR at end of schedule (as fraction: `min_lr / lr` is the floor ratio). |
| `weight_decay` | float | `0.1` | Decoupled weight decay coefficient. |
| `warmup_ratio` | float | `0.05` | Fraction of total steps for linear LR warmup. |
| `max_grad_norm` | float | `1.0` | Global gradient clipping norm. Disabled when `adagc: true`. |
| `lr_scheduler` | str | `cosine` | LR schedule. `power_decay` (recommended), `wsd` (long runs), `cosine`, `linear`, `constant`. |
| `optimizer` | str | `adamw` | Optimizer. `adamw`, `adamw_bf16`, `muon`, `lion8bit`, `adamw8bit`, `paged_adamw8bit`. `adamw_bf16`: AdamW on bf16 weights and bf16 moments with stochastic rounding, so updates smaller than a bf16 ulp still accumulate (needs `model.torch_dtype: bfloat16`; see [Optimizers](optimizers.md#adamw-bf16)). |
| `bf16_rounding` | str | `stochastic` | With `optimizer: adamw_bf16`: `stochastic` (unbiased rounding of the weight, 6 B/param of weight and state) or `kahan` (a bf16 compensation buffer, 8 B/param: no rounding noise; for small learning rates such as RL or DPO). |
| `fused_optimizer` | bool | `false` | With `optimizer: adamw` or `adamw_bf16`: step every parameter with one multi-tensor Triton launch per dtype bucket instead of one kernel per parameter, gradient clipping folded into the step on one process. `adamw_bf16` keeps its exact bits (those of its eager update) and bit-exact resumes; `adamw` is torch AdamW's update and state. Falls back to the usual optimizer (logged) without CUDA or Triton. Not with `memory.gradient_release`. See [Optimizers](optimizers.md#multi-tensor-kernels-trainfused_optimizer). |
| `seed` | int | `42` | Training random seed. |
| `save_steps` | int | `500` | Save a resumable checkpoint (`step-N/`) every N steps. Old ones are deleted after each save, keeping the newest `keep_checkpoints`. |
| `keep_checkpoints` | int | `5` | Step checkpoints kept on disk; the oldest are deleted after each save. `best/` and `final/` are never deleted. `0` keeps all. A full checkpoint holds the weights and the optimizer state (about 3× the fp32 weights for AdamW): size the disk for `keep_checkpoints` of them. |
| `save_final` | bool | `true` | Save the final model to `output_dir/final` (Hugging Face format) at the end. |
| `logging_steps` | int | `1` | Log metrics every N steps. |
| `bf16` | bool | `true` | Enable bf16 mixed precision with fp32 gradient reduction. |
| `stable_softmax` | bool | `false` | Cross-entropy, and the forward KL of off-policy distillation to an online teacher with the student's vocabulary, computed so the target token's gradient never rounds to zero (it does in fp32 once the target logit leads the others by about 17) and equal student and teacher logits give a zero gradient ([arXiv:2610.31642](https://arxiv.org/abs/2610.31642)). Covers plain, chunked and SeCO cross-entropy and the distillation gold cross-entropy; turns off Cut Cross-Entropy; refused with DPO, `plugins.pre_rl` and the gated objectives. A few extra element-wise passes over the logits. Not measured on training quality. |
| `gradient_checkpointing` | str | `selective` | Activation checkpointing. `selective` keeps attention outputs and every other matmul (Qwen3-0.6B, 8×2048 tokens, compiled: 8.6 GiB of activations, +10% step time); `full` recomputes each layer (1.9 GiB, +25%); `none` keeps everything (19.8 GiB, fastest). |
| `spike_detection` | bool | `true` | Skip optimizer step when gradient norm is anomalous (z-score based). |
| `spike_z_threshold` | float | `5.0` | Z-score threshold for spike detection. Higher = fewer skips. |
| `adagc` | bool | `false` | Per-tensor adaptive gradient clipping. Replaces global clipping. Better for stability. |
| `adagc_lambda` | float | `1.5` | Relative clip threshold: clip if the tensor's gradient norm $> \lambda \times$ its EMA, with $\lambda$ = `adagc_lambda`. |
| `adagc_beta` | float | `0.95` | EMA decay for per-tensor norm tracking. |
| `ema` | bool | `false` | Exponential Moving Average of weights. Better generalization. Stored on CPU. |
| `ema_decay` | float | `0.999` | EMA decay factor. 0.999 $\approx$ 1000-step window. |
| `ema_every` | int | `10` | Update EMA every N steps. |
| `base_merge` | bool | `false` | Periodically merge toward pretrained weights (anti-forgetting). |
| `base_merge_ratio` | float | `0.1` | Mix ratio $r$: $\theta \leftarrow (1-r)\,\theta + r\,\theta_{\text{base}}$ (with `lerp`). |
| `base_merge_every` | int | `500` | Steps between merges. |
| `base_merge_method` | str | `lerp` | Interpolation: `lerp` (linear) or `slerp` (spherical, preserves norms). |
| `adamc` | bool | `false` | Corrected weight decay for normalized layers. Prevents gradient explosion at end of training. |
| `llrd_decay` | float | `1.0` | Layer-wise LR decay. 1.0 = off. 0.9 = early layers get 0.9× LR per depth. |
| `freeze_non_attention` | bool | `false` | Hybrid models (Qwen3.5, Qwen3-Next, Mamba hybrids): freeze the linear-attention / recurrent layers (all their parameters except the layer norms); full-attention layers, embeddings, final norm and head train. No-op, with a warning, on models without such layers. |
| `hyperball` | bool | `false` | Hyperball (arXiv:2606.16899): attention/MLP matrices keep their initial norm and move by a fixed angular step each update; embeddings, norms, biases and the head stay on the base optimizer. Any base optimizer. See [Optimizers](optimizers.md#hyperball). |
| `hyperball_lr` | float | `0.0` | Hyperball's angular step $\eta$ (fraction of each matrix norm moved per update, scaled by the LR schedule). `0` = per matrix, the base optimizer's first relative step (Adam/Lion: $\texttt{learning\_rate} / \operatorname{rms}(W)$). |
| `mona` | bool | `false` | MONA curvature-aware acceleration. Augments gradients with EMA of gradient differences. |
| `mona_beta_a` | float | `0.975` | MONA acceleration EMA decay. Higher for larger models (0.99 for 68B). |
| `mona_lite` | bool | `true` | Store MONA buffers in bf16 + streaming computation. 75% overhead reduction. |

---

## parallel

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fsdp` | bool | `true` | Enable FSDP2 data parallelism (multi-GPU). Auto-disabled for single GPU. |
| `context_parallel` | bool | `false` | Split every sequence over a group of ranks (all-gather K/V, head-tail load balancing); exact. Needs `fsdp: true`. For sequences whose activations do not fit one GPU. Gated DeltaNet hybrids (Qwen3.5, Qwen3-Next) by state passing; refuses other recurrent / linear-attention layers (see the multi-GPU guide). |
| `cp_degree` | int | `0` | Ranks per context-parallel group; must divide the world size. `0` = min(world_size, 8) when it divides it, else all ranks. |
| `cp_rotate_method` | str | `allgather` | How K/V reach the other ranks: `allgather`, the only method implemented. |
| `cpu_offload` | bool | `false` | Keep parameters, gradients and optimizer state in pinned CPU memory (FSDP2's CPU offload; the optimizer steps on the CPU). Also on a **single GPU**, where FSDP2 runs over a one-rank process group: the GPU then holds one layer's weights at a time plus activations, for models whose weights and optimizer state do not fit it. Host RAM: about 16 B/param with `model.torch_dtype: float32` and `adamw`, 8 B/param with `bfloat16` and `train.optimizer: adamw_bf16`, plus pinned staging. Speed is bound by PCIe and the CPU's AdamW step. Implies FSDP2 (also with `fsdp: false`); not with `memory.seco` or `memory.gradient_release`. See [Single GPU](../guides/single-gpu.md#models-larger-than-the-gpu-cpu-offload). |
| `reshard_after_forward` | bool | `true` | Reshard parameters after forward. `false` = keep unsharded (more memory, less communication). |
| `tp_degree` | int | `1` | Tensor parallelism: every layer's attention heads, Gated DeltaNet heads and MLP / expert columns split over this many ranks (column / row parallel linears with sequence parallelism between them); the rest of the ranks are data parallel (FSDP2). Must divide the world size and the head counts. Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5, Gemma 1 / 2 / 3. Needs `fsdp: true`. See [Tensor and pipeline parallelism](../guides/multi-gpu.md#tensor-and-pipeline-parallelism). |
| `async_tp` | bool | `false` | Overlap the tensor-parallel all-gathers / reduce-scatters with their matmuls (torch's micro-pipelined TP over symmetric memory, in the compiled layers). Needs `tp_degree > 1` and `model.compile: true`; NVLink GPUs. |
| `pp_degree` | int | `1` | Pipeline parallelism: the layers split into stages over this many rank groups (outermost mesh dimension). Untied embeddings only. Needs `fsdp: true`. |
| `pp_schedule` | str | `1f1b` | `gpipe`, `1f1b`, `interleaved_1f1b`, `looped_bfs`, `interleaved_zero_bubble`, `zbv` (zero bubble, V-shaped), `dualpipev` (torch.distributed.pipelining). |
| `pp_microbatches` | int | `0` | Pipeline micro-batches per `train.per_device_batch_size` rows (must divide it); `0` = one row each. More micro-batches, smaller bubble. |
| `pp_stages_per_rank` | int | `0` | Stages per pipeline rank; `0` = the schedule's default (1 for gpipe / 1f1b, 2 for the others; zbv / dualpipev exactly 2). |
| `pp_first_stage_less_layers`, `pp_last_stage_less_layers` | int | `1`, `1` | Layers the first / last stage gives up for the embeddings / the final norm and output head, to balance the stages. |
| `expert_parallel` | bool | `false` | MoE models: every rank keeps `experts / ranks` experts of each layer and the tokens are sent to the rank that owns their experts (all-to-all), instead of FSDP gathering all expert weights on every rank. Needs `fsdp: true` and a number of ranks that divides the number of experts; not with context parallelism, SeCO or CPU offload. Experimental: gradients tested equal to one process on CPU ranks, not measured on several GPUs. See [Mixture-of-Experts Models](../guides/moe.md#expert-parallelism). |

---

## moe

Mixture-of-Experts models (ignored for dense models). The same section exists in RL (`pgs rl`) and distillation
(`pgs distill`) configs. See [Mixture-of-Experts Models](../guides/moe.md).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `balance` | bool | `true` | Router balancing and routing statistics (`moe/...` metrics). `false`: the routers train on the task loss alone. |
| `balance_loss_coef` | float | `-1` | Coefficient of the load-balancing loss (Switch Transformer), with the expert loads measured over the global batch: every rank and every micro-batch of the optimizer step. `-1`: the model config's `router_aux_loss_coef`, at most 0.01. `0`: off. |
| `z_loss_coef` | float | `0.0` | Coefficient of the router z-loss (ST-MoE used 0.001). `0`: off. |
| `bias_update_rate` | float | `-1` | Auxiliary-loss-free balancing: step of the routers' selection bias after every optimizer step, against the step's loads. Only for routers with a selection bias in their checkpoint (DeepSeek-V3 family). `-1`: 0.001 for those models, off for the others. `0`: off. |
| `log_layers` | bool | `false` | Also log the loads and router entropy of every MoE layer (`moe/layer_<i>/...`). |
| `experts_implementation` | str | `auto` | Expert compute: `auto` (transformers' default, a grouped GEMM where torch has one), `grouped` or `padded` (this library's forwards), or a transformers name (`eager`, `grouped_mm`, `batched_mm`). `pgs train`. Measure with `scripts/bench_moe_experts.py`. |

RL and distillation configs also have a `plugins:` section (`custom`: trainer plugins, as `plugins.custom` in `pgs train`, e.g. [`palingenesis.qat:FloatQAT`](../guides/qat.md)); the hooks they run are listed in [Plugins](plugins.md#in-pgs-distill-and-pgs-rl). Not with `lora.rank`.

RL configs also have a `lora:` section (`rank`, `alpha`, `rslora`, `dora`, `lr_ratio`, `targets`, `exclude`; `rank: 0` = off) that trains low-rank adapters instead of the full weights: see [LoRA in RL](../guides/lora.md#lora-in-rl).

### RL: `scenario`, `roles` and replicated bases

RL configs (`pgs rl`) run multi-role episodes when they have a `scenario:` section ([Multi-Agent RL](../guides/multi-agent.md)).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario.type` | str | `""` | A pattern over your own RL dataset (`solve_check`, `handoff`, `lead_workers`: scored by the `rewards:` section; `plan_execute`: Harbor rows, scored by the task's tests), a scenario with its own generator and verifier (`take_away`, `ledger_sum`, `checked_answer`: no `rewards:` section), or `module:Class` / `path/to/file.py:Class`. Empty: single-agent RL. |
| `scenario.args` | dict | `{}` | The scenario's own options, e.g. `{weights: {lead: {coverage: 0.2}}}`. Patterns: `prompt_field`, `gold_field`, `answer_tag`, `language_field`, `language`, `threshold`, and each pattern's own (`pgs scenario patterns` lists them; [Multi-Agent RL](../guides/multi-agent.md#b-your-rl-dataset-plus-a-pattern)). |
| `rewards` | dict | `{}` | With `solve_check`, `handoff` or `lead_workers`: required, and used as in single-agent RL (the final answer is the completion). With any other scenario, `plan_execute` included: refused. |
| `scenario.timeout` | float | `600` | Seconds an episode may take; a slower one is cancelled and left out of training. |
| `scenario.outcome` | str | `""` | The role whose reward is the task's outcome (retire / retry, `outcome/*`); empty: the scenario's. |
| `roles.<name>.adapter` | str | `""` | A LoRA adapter of its own on the policy's base (needs `lora.rank`). |
| `roles.<name>.partner` | str | `""` | A frozen partner: `base`, `scripted:<name>`, an endpoint URL or `module:callable`. |
| `roles.<name>.partner_args` | dict | `{}` | Arguments for an endpoint or class partner. |
| `roles.<name>.train` | bool | `true` | `false`: the policy plays the role, its tokens do not train. |
| `roles.<name>.system_prompt` | str | `""` | Put before the scenario's instructions. |
| `roles.<name>.max_turns` / `max_new_tokens` / `max_completion_tokens` | int | `0` | The role's limits (0: the scenario's or the rollout's). |
| `roles.<name>.chat_template_kwargs` | dict | `{}` | Template arguments for this role's sessions. |
| `roles.<name>.share` | float | `1.0` | The role's part of the loss, relative to the other trained roles. |
| `loss.advantage_std` | str | `batch` | `role` (scenarios): the standard deviation of the same role's centred rewards over the batch. |
| `loss.advantage_std_floor` | float | `0.0` | The least value `role` divides by. |
| `rollout.backend` | str | `vllm` | `remote`: the engines are `palingenesis.rl.pool.worker` processes on other GPUs or nodes (all backends: [RL engines](#rl-rollout-engines)). |
| `rollout.engines` | list | `[]` | remote: the workers' `host:port` addresses; rank *r* generates with `engines[r::ranks]`, rank 0 sends the weights to all. |
| `rollout.push` | str | `auto` | remote: what a weight update sends. `full`, `fold` (LoRA factors folded by each engine), `native` (LoRA factors as named adapters of the engine; not with `lora.dora`), `auto` (`native` when a role has its own adapter or `partner: base`, else `fold`). The first update also sends the frozen base. |
| `rollout.concurrency` | str | `fixed` | vllm_server, remote: `kv` caps the requests in flight to each engine from its KV-cache use (start at KV tokens / `max_model_len`, cut above `kv_usage_high`, raise below `kv_usage_low`); see [Rollout concurrency](../guides/rl.md#rollout-concurrency). |
| `rollout.concurrency_min` / `concurrency_max` | int | `4` / `0` | `kv`: the cap's bounds per engine (`0`: `rollout.max_num_seqs`, or what one step generates at once, at least 256). |
| `rollout.kv_usage_low` / `kv_usage_high` | float | `0.6` / `0.8` | `kv`: KV-cache usage below which the cap grows by 25%, above which it is cut by 30%. |
| `rollout.concurrency_poll_s` | float | `5.0` | `kv`: seconds between reads of the engines' metrics. |
| `rollout.token_allowance` | float | `0.0` | Speculative group launches start only while held and in-flight tokens stay within (1 + this) x a batch's tokens; `0` = off. Any backend. |
| `train.replicate_base` | bool | `false` | With `lora.rank` under torchrun: every rank holds the whole frozen base; only the adapters' gradients are summed over ranks, once per step, instead of FSDP. |

RL configs also have `rollout.routing_replay` (`auto` / `on` / `off`, default `off` until the vLLM path is verified on a GPU): train each token through the
experts the rollout engine routed it to. See [RL: routing replay](../guides/moe.md#rl-routing-replay).

RL configs lay out the policy step's micro-batches with `train.packing` (`padded` / `varlen` / `tree`, default `padded` until measured on a GPU):
`varlen` packs a micro-batch's trajectories end to end in one row without padding, `tree` packs their prefix trie so a group's prompt and any other
shared prefix is forwarded once; `train.micro_tokens` then bounds the packed (unique) tokens, and `train.packing_gather_tokens` (int, `0` = 8 x
`micro_tokens`) the key / value rows a `tree` micro-batch's attention gathers (every branch reads its whole path: about trajectories x shared prefix). Same log-probs and gradients as `padded`. Needs
`sdpa` attention and full-attention / Gated DeltaNet layers; not with `model.multimodal`, and `tree` not with `rollout.routing_replay`. See [RL: packing micro-batches](../guides/rl.md#packing-micro-batches).

### RL: truncated sampling and score centering

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rollout.top_p` | float | `1.0` | Nucleus sampling of training rollouts (1 = off). Needs `rollout.top_k`. The trainer renormalises the policy over each token's support, returned by the engine. See [RL: truncated sampling](../guides/rl.md#truncated-sampling-top-p-top-k). |
| `rollout.top_k` | int | `0` | Top-k sampling of training rollouts (0 = off), at most 1024; the engine returns `top_k + 8` log-probs per sampled token. |
| `loss.score_centering` | str | `off` | `topk`: subtract each token's expected policy gradient under the sampler, over its top `score_centering_k` tokens (arXiv:2609.20807). `masked_is`, `cispo`, `icepop`, `ipo` only. Unvalidated in palingenesis: A/B it before use. See [RL: score centering](../guides/rl.md#score-centering). |
| `loss.score_centering_k` | int | `8` | The sampler's most likely tokens per position the centering is estimated over. |

Neither works with `loss.kl_ref` or a `scenario:` yet. With `rollout.backend: remote`, start the workers with `--max-logprobs` at least the log-probs requested (`top_k + 8`, or `score_centering_k`).

---

### RL: rollout engines

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rollout.backend` | str | `vllm` | `vllm` (in-process vLLM, asleep while training), `vllm_server` (a vLLM server fed weights over CUDA IPC), `sglang` (in-process SGLang engine, asleep while training), `sglang_server` (an SGLang server on this GPU or at `rollout.url`), `remote` (pool workers: `--backend vllm`, `sglang` or `hf`), `hf` (the trainer's `generate`). Feature table: [Rollout engines](../guides/rl.md#rollout-engines-vllm-or-sglang). |
| `rollout.gpu_memory_utilization` | float | `0.3` | The engine's GPU share for weights and KV cache (vLLM's `gpu_memory_utilization`, SGLang's `mem_fraction_static`). |
| `rollout.max_model_len` | int | `8192` | Prompt + completion tokens of a trajectory (SGLang's `context_length`). |
| `rollout.max_num_seqs` | int | `0` | Concurrent sequences (vLLM: 0 = batch x group, at least 256; SGLang: 0 = sized by SGLang). |
| `rollout.enforce_eager` | bool | `false` | No CUDA graphs. |
| `rollout.prefix_caching` | bool | `true` | vLLM's prefix caching / SGLang's radix cache (flushed after every weight update). |
| `rollout.sleep` | bool | `true` | `vllm` / `sglang` with `max_staleness: 0`: release the engine's memory while the trainer trains. |
| `rollout.url` | str | `""` | `vllm_server` / `sglang_server`: a running server instead of launching one. |
| `rollout.vllm_args` | dict | `{}` | Extra `vllm.LLM` arguments (`vllm`). |
| `rollout.sglang_args` | dict | `{}` | Extra SGLang engine arguments (ServerArgs fields; `sglang`, `sglang_server`). Those set by the options here are refused. |
| `rollout.weight_sync` | str | `auto` | `sglang_server`: `tensor` (CUDA IPC, same GPU), `distributed` (NCCL from the trainer), `disk` (a checkpoint in `weight_sync_dir`); `auto`: `tensor` for a launched server, `distributed` with `rollout.url`. |
| `rollout.weight_sync_dir` | str | `""` | `weight_sync: disk`: a directory the server reads too (default `<output_dir>/rollout_weights`). |
| `rollout.weight_bucket_mb` | int | `512` | SGLang backends: bf16 weights per update call (one CUDA IPC handle per bucket). |
| `rollout.fp32_head` | bool | `false` | Logits in fp32 (vLLM `head_dtype`, SGLang `enable_fp32_lm_head`). |
| `rollout.ssm_state_dtype` | str | `float32` | Recurrent state of hybrid models (vLLM `mamba_ssm_cache_dtype`, SGLang `mamba_ssm_dtype`; `auto`: the engine's choice). |
| `rollout.kv_cache_dtype`, `rollout.quantization` | str | `""` | FP8 rollouts (`fp8`; SGLang `fp8_e4m3`). |

## memory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunked_loss` | bool | `true` | Split CE loss into chunks to avoid materializing full [B,S,V] logit tensor. Prevents OOM on large vocab. |
| `loss_num_chunks` | int | `8` | Number of chunks. Auto-tuned based on seq_len × vocab_size. |
| `float32_matmul_precision` | str | `high` | CUDA matmul precision. `highest` (exact), `high` (TF32, recommended), `medium` (faster, less precise). |
| `float8_training` | bool | `false` | FP8 training (H100+ SM89). 1.2-1.5× throughput. |
| `gradient_release` | bool | `false` | Fuse optimizer into backward. Eliminates gradient memory. Requires GA=1, incompatible with Muon. |
| `selective_diff` | bool | `true` | Skip activation saving for frozen layers. Auto-enabled with `freeze_non_attention`. |
| `seco` | bool | `false` | SeCO chunk-wise training for long sequences: activation memory set by `seco_chunk_size`, not sequence length. Exact gradients. Single GPU. With `data.packing`, every packed document is trained on its own: long ones chunk by chunk, short ones batched (not with `spaco_budget`, `distill.enabled` or MoE router balancing). See [Long Sequences](../guides/long-sequences.md). |
| `seco_chunk_size` | int | `4096` | Tokens per SeCO chunk; with packing, also the padded tokens of a batch of short documents. |
| `seco_kv_offload` | bool | `false` | Keep full-attention K/V (and recurrent start states) in pinned CPU memory, streamed to the GPU block by block; GPU memory then grows only with the K/V gradient. Exact. Needs `attn_implementation: sdpa`. |
| `spaco_budget` | int | `0` | SpaCO: backpropagate only this many random chunks per sequence (a stochastic gradient estimate). `0` = exact SeCO. |
| `activation_offload` | bool | `false` | Keep the activations the transformer layers save for the backward in pinned host memory: copied out in the forward and back in the backward on a side CUDA stream, overlapped with other layers' compute. Exact. What moves depends on `train.gradient_checkpointing`: every saved activation (`none`), each layer's input (`full`), or the input plus what the selective policy keeps (`selective`, on torch versions that route it through saved-tensor hooks). The last layer stays on the GPU. Host memory: about the moved activations, at most 40% of RAM per node. See [Single GPU](../guides/single-gpu.md#activation-offload). |
| `activation_offload_min_bytes` | int | `1048576` | Saved tensors smaller than this stay on the GPU. |
| `activation_offload_in_flight` | int | `1` | Layers whose device-to-host copies may still run while later layers compute (each keeps its activations on the GPU until done). >= 1. |
| `activation_offload_prefetch` | int | `1` | Layers fetched back ahead of the backward. `0`: a layer's tensors are fetched when its backward starts. |
| `activation_offload_boundaries_only` | bool | `false` | Offload only each checkpointed layer's inputs; what selective checkpointing keeps stays on the GPU. Needs `gradient_checkpointing: selective` or `full`. |

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
| `custom` | list | `[]` | Your own trainer plugins (`palingenesis.hooks.TrainerPlugin`): `module:Class`, `path/file.py:Class`, an entry point of group `palingenesis.plugins`, or `{class: ..., args: {...}}`, run in order. See [Trainer plugins](plugins.md#trainer-plugins). |

---

## preprocess

Offline data preparation, driven by the **same config** as training (see the [Data Preparation guide](../guides/data.md)). With `mode: pretrain` entries in `data.sources`, `pgs prepare --config <file>` cleans each of them (exact and near-duplicate documents, n-gram decontamination, `min_chars`, `eval_holdout`, `max_samples`, `format`) into `output_dir/<source name>/` without loading a model, and logs the `data.sources` entries to train on; the other options of this section apply to conversations. See [Preparing the data](../guides/pretraining.md#preparing-the-data). `pgs prepare --config <file>` reuses `model.name_or_path`, `data.dataset`, `data.dataset_split`, `data.messages_field`, `data.max_seq_length` and the masking options of `data` — this section controls cleaning, selection and output. Nothing here runs the model unless `score` is set or a perplexity option needs it.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Train on the prepared output in `output_dir` instead of the raw `data.dataset`. Fails loudly if nothing was prepared yet. Incompatible with `data.sources` (use `prepare-multi` for that). |
| `output_dir` | str | `./prepared` | Where `pgs prepare` writes `scored_data.parquet`, `prepared_meta.json` and `dropped.jsonl`, and where training looks for the data. |
| `format` | str | `parquet` | Output format. `parquet` (order-preserving, fast; falls back to jsonl if the sample schema can't be unified) or `jsonl`. |
| `max_samples` | int | `0` | Cap on samples read from the raw dataset. 0 = all. |
| `dedupe` | bool | `true` | Drop exact duplicate conversations (same messages and tool definitions; whitespace runs collapsed). The first row is kept. Raw-text sources: documents whose text is the same up to case and whitespace. |
| `near_dedupe` | float | `0.0` | Raw-text sources: also drop near-duplicate documents, those whose sets of word 5-grams have an estimated Jaccard similarity of at least this value (MinHash with 128 hashes; `0.8` is a common choice). The first document of a group is kept. 0 = off. Needs numpy. |
| `decontaminate` | list | `[]` | Raw-text sources: files of the evaluation sets you will report (JSONL, JSON, parquet; any other file is read as plain text). A document that shares a run of `decontaminate_ngram` words with any string in them (every column, nested values included, or `decontaminate_fields`; case and punctuation ignored) is dropped. Shared scaffolding is left out: see `decontaminate_shared_share`. |
| `decontaminate_fields` | list | `[]` | Columns of the evaluation files to read, e.g. `[question, answer]`. Empty = every string column. A file without any of them is an error. Naming the columns turns the shared-n-gram rule off by default. |
| `decontaminate_shared_share` | float | `-1` | N-grams found in more than this share of an evaluation file's rows (files of 10 rows or more) are not matched. `-1` = 0.2 when every column is read, off with `decontaminate_fields`. `0` = off: every n-gram is matched. |
| `decontaminate_ngram` | int | `13` | Words per decontamination n-gram. Evaluation strings shorter than this give no n-gram and protect nothing (the report counts them): lower it for short items. |
| `min_chars` | int | `0` | Raw-text sources: drop documents with fewer characters. 0 = keep all. |
| `max_per_prompt` | int | `0` | Keep at most N rows that share a system prompt and first user turn (a seeded random draw). 0 = no cap. |
| `group_field` | str | `""` | Row column that names a row's group (e.g. `source`). The group caps apply per value, and `prepared_meta.json` profiles each group (rows, tokens, mean response tokens; when scored, median response NLL and familiarity buckets) before and after caps and selection. |
| `max_per_group` | int | `0` | Keep at most N rows per `group_field` value. 0 = no cap. |
| `max_tokens_per_group` | int | `0` | Keep at most N tokens per `group_field` value. 0 = no cap. |
| `eval_holdout` | int | `0` | Reserve about N rows as a held-out eval set (`eval_data.parquet`), excluded from training and sharing no prompt with it. Training auto-uses it when `data.eval_dataset` is empty. |
| `budget` | int | `0` | Rows to keep after cleaning and caps. 0 = keep all. |
| `token_budget` | int | `0` | Tokens to keep: a random subset whose rows add up to at most this many tokens. 0 = no limit. Only with `strategy: random`. |
| `score` | bool | `false` | Score every row with the model (response perplexity, familiarity bucket). One forward pass over the data. [Experimental](../guides/data.md#scoring-with-the-model-experimental). |
| `strategy` | str | `random` | How `budget` rows are chosen: `random` (uniform, seeded). Experimental, each turns scoring on and logs a warning: `optimal` (heuristic J-shaped familiarity mix; measured below `random`), `curriculum` (random subset ordered familiar→unfamiliar, order kept at train time), `balanced`, `medium_focus`, `hard_focus`, `flow` (lowest perplexity). Unknown names are an error. |
| `min_ppl` | float | `0.0` | Drop samples whose response perplexity is below this. 0 = off; above 0 turns scoring on. |
| `max_ppl` | float | `0.0` | Drop samples above this. 0 = off; above 0 turns scoring on. |
| `filter_score` | str | `response` | Perplexity the filters use: `response` (trained tokens only) or `full` (whole conversation). |
| `batch_size` | int | `4` | Max samples per scoring forward pass. Scoring is length-sorted and padded-batch, so large values (64–256) are safe — `max_batch_tokens` bounds memory, not this. |
| `max_batch_tokens` | int | `16384` | Padded-token cap per scoring forward. Logits are batch×seq×vocab, so this is what bounds memory: 16K ≈ 5GB bf16 logits at 150K vocab. On an 80GB GPU with a ≤8B model, `32768` is safe and noticeably faster. |
| `hes` | bool | `false` | Also write High-Entropy Sum scores of the response (`_score_hes`, `_score_hes_mean`) from the scoring forward. Turns scoring on. No selection strategy reads them. |
| `hes_top_k_pct` | float | `0.5` | Top-k% highest-entropy response tokens summed for HES. |
| `num_workers` | int | `0` | Processes that render and tokenize the rows: 0 = the CPUs available, at most 8; 1 = in the calling process. Small datasets use fewer (at least 256 rows per process). The output is identical whatever the number. CLI: `--num_workers` (`pgs prepare`, `pgs prepare-multi`). |

---

## dpo

Preference optimization (DPO and variants), a mode of `pgs train`. When `enabled`, `data.dataset` and `data.eval_dataset` hold preference pairs; optimizer, schedule, FSDP, checkpoints and chat-template masking are shared with SFT. See the [DPO guide](../guides/dpo.md).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Train on preference pairs instead of SFT conversations. |
| `loss_type` | str | `sigmoid` | `sigmoid` (DPO), `hinge` (SLiC-HF), `ipo`, `robust` (rDPO), `sigmoid_norm` (length-normalised). |
| `beta` | float | `0.1` | Inverse temperature $\beta$ of the implicit reward $\beta\,(\log \pi - \log \pi_{\text{ref}})$. |
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

## distill

Off-policy knowledge distillation, a mode of `pgs train`: the student trains on the data's sequences against one or more teachers' distributions at every trained token. See the [Off-Policy Distillation guide](../guides/offpolicy-distillation.md) (every teacher key is listed there).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Off-policy distillation in `pgs train`: the student trains on the data's sequences against its teachers' distributions |
| `teachers` | dict | `{}` | name → `{model: ...}` (online, frozen) or `{cache: ...}` (a `pgs distill-cache` top-k cache); per teacher: `weight`, `sources`, `dtype`, `placement` (`auto`/`replicate`/`shard`), `device`, `attn_implementation`, `tokenizer`, `eos_map`, `probe_texts`, `top_k`, and the teacher view keys `system_prompt`, `prefix_messages`, `privileged_field` / `privileged_template` / `privileged_role`, `tools` (`keep` / `drop`) / `tools_field`, `messages_field`, `chat_template_kwargs`, `view_fn`, `max_seq_length` |
| `divergence` | str | `forward_kl` | `forward_kl`, `reverse_kl`, `jsd` (GKD), `skew_forward_kl`, `skew_reverse_kl` (DistiLLM), `tvd` |
| `beta` | float | `0.5` | `jsd` mixture weight of the teacher / `skew_*` skew coefficient, in (0, 1) |
| `temperature` | float | `1.0` | Softmax temperature of both distributions |
| `temperature_scale` | bool | `true` | Multiply the divergence by $T^2$, with $T$ = `temperature` |
| `kd_weight` | float | `1.0` | Weight of the divergence per trained token |
| `ce_weight` | float | `0.0` | Weight of cross-entropy on the gold token |
| `topk_mode` | str | `tail` | Cached teachers: `tail` (top-k + gold + a tail bucket per side) or `renorm` (both renormalised over the support) |
| `missing` | str | `error` | A trained token no teacher covers: `error` or `skip` (CE only) |
| `features` | list | `[]` | Hidden-state / module-output matching: `{student, teacher, weight, loss: mse/nmse/cosine/lens_kl, projection: auto/linear/none, via}` |
| `feature_lr` | float | `0.0` | Learning rate of the feature projections (0 = `train.learning_rate`) |
| `rows_per_slice` | int | `0` | Rows per slice of the chunked loss (0 = automatic, ~512 MB of fp32 logits per side) |

---

## multimodal

Images in the training data, for the Qwen-VL family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5). See the [Multimodal guide](../guides/multimodal.md) for the data format and for what is supported, refused and verified.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Load the checkpoint in its multimodal architecture and feed the images in the data's content parts to its vision tower. `false`: the language model alone is trained, image parts are ignored (with a warning) and exports copy the vision tower unchanged. Not supported with `memory.seco`, `parallel.context_parallel`, `data.pretokenize`, cached teachers or teacher views |
| `train_vision_tower` | bool | `false` | Train the vision encoder (frozen otherwise) |
| `train_projector` | bool | `false` | Train the projector that maps vision features into the language model (Qwen: the patch merger) |
| `min_pixels` | int | `0` | Lower bound of the image processor's resize, in pixels (0: the checkpoint's) |
| `max_pixels` | int | `0` | Upper bound of the resize. An image takes at most `max_pixels / (patch · merge)²` tokens. Serve the model with the same bounds; exports record them in `preprocessor_config.json` |

---

## logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | str | `palingenesis` | Project name for wandb/trackio. |
| `run_name` | str\|null | `null` | Run name. Auto-generated from model name if null. |
| `use_wandb` | bool | `true` | Log to Weights & Biases. |
| `use_trackio` | bool | `true` | Log to trackio. |
| `trackers` | list | `[]` | Further metric backends, `module:Class` or `path/to/file.py:Class` ([Observability](../guides/observability.md#your-own-backend)). |
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
| `loss` | str | `""` | `full_rkl`, `topk_kl`, `sampled_rkl`, `xtok` or `rs_kd`. Empty = `full_rkl` (hf) or `topk_kl` (vllm) for a teacher sharing the student's vocabulary, `xtok` otherwise. `full_rkl` and `rs_kd` need the hf backend; all but `xtok` need a shared vocabulary (checked when the tokenizers load). agent_traces runs: `full_rkl` (hf) or `topk_kl` (vllm). |
| `device` | str | `""` | hf: device (empty = the student's), e.g. `cuda:1`. |
| `offload` | bool | `false` | hf: keep the model on CPU between scoring calls (the output head stays on the device). |
| `url` | str | `""` | vllm: an already running server. |
| `gpu_memory_utilization` | float | `0.15` | vllm: GPU fraction of a launched server. |
| `eos_map` | dict | `{}` | Shared vocabulary: student end-of-turn token → teacher's, e.g. `{"<\|im_end\|>": "<\|eot_id\|>"}`. Empty = auto (student eos → teacher eos when the student's lies outside the shared vocabulary). |
| `probe_texts` | list | `[]` | Extra texts that must tokenize identically for a shared vocabulary. |

### sources.&lt;name&gt;

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | `messages` | `messages` (chat JSONL `{"messages": [...], "answer"?: ...}`: held-out reverse KL, plus greedy accuracy for rows with an answer), `mcqa` (pool-row JSONL: letter accuracy) or `agent_traces` (recorded agent conversations, JSONL or parquet: the student regenerates their assistant turns; see the distillation guide). An `agent_traces` run also takes `messages` sources, as one-turn traces. |
| `path` | str | — | Prompt file. |
| `weight` | float | `1.0` | Sampling weight among the sources. |
| `teacher` | str | `""` | Teacher of this source's prompts (empty = the first teacher). |
| `topic_field` | str | `""` | Row field holding a topic (e.g. `domain`), for `topic_teachers`. |
| `topic_teachers` | dict | `{}` | `{teacher: [topics]}`: rows whose topic is listed go to that teacher, the others to `teacher`. A topic may be listed under one teacher only. |
| `privileged_field` | str | `""` | messages: a row field only the teacher sees (OPSD), placed by `privileged_template`; rows without it are ordinary rows. |
| `privileged_template` | str | `{content}\n\n{privileged}` | Where the privileged text goes. Placeholders: `{content}` (the rewritten turn), `{privileged}` (the field, required), `{question}` (the last user turn), `{system}` (the system turn). |
| `privileged_role` | str | `user` | `user` (rewrite the last user turn) or `system` (rewrite or create the system turn). |
| `privileged_views` | list | `[]` | Several kinds of privileged context instead of the three options above: `[{name, template, role, field, weight, tools, tools_field}]`. `tools: drop` / `tools_field: <row field>` change the tool schemas the teacher's prompt lists (no template needed); a view must change the text or the tools. A view with a `field` applies to rows that fill it; one without builds the teacher's turn from the conversation (RE2: `{content}\n\nRead the question again: {question}`). One applicable view per rollout, drawn by weight; logged as `privileged/<name>`. See [Privileged context](../guides/distillation.md#privileged-context-on-policy-self-distillation). |
| `privileged_template_field` | str | `""` | A row field with the row's own template, used instead of the view's where non-empty; checked at load. |
| `chat_template_kwargs` | dict | `{}` | Chat-template arguments for this source over `model.chat_template_kwargs` (e.g. `{enable_thinking: true}`). `messages` rows: a row's own kwargs win. `agent_traces` runs: every row of the source renders with them; a `messages` row that disagrees is an error. |
| `messages_field` / `tools_field` | str | `messages` / `tools` | agent_traces: the columns holding the conversation and the tool schemas (lists or JSON strings). |
| `branches_per_trace` | int | `8` | agent_traces: assistant turns regenerated per sampled trace (0 = all). |
| `onset_branches` | int | `0` | agent_traces: extra regenerated turns per trace starting mid-turn where a recorded assistant turn begins to repeat earlier assistant text (`opd.traces.repetition_runs`; for the student's own episodes). 0 = off. |
| `onset_kd_tokens` | int | `-1` | agent_traces, with `trace_kd_weight`: recorded-turn distillation keeps this many tokens of each repetitive run (-1 = all). |
| `onset_ngram` / `onset_window` / `onset_threshold` / `onset_common` | int / int / float / float | `6` / `32` / `0.75` / `0.5` | The repetition scan: n-gram length, window, share of repeated tokens that starts a run, share of earlier turns above which an n-gram is scaffolding. |
| `error_patterns` | list | `[]` | agent_traces: regular expressions (`re.search`) on the observations answering an assistant turn (messages up to the next assistant turn with a role in `error_roles`); a match marks the turn's tool call as failed (`opd.traces.FailedTurns`). |
| `error_roles` | list | `[tool]` | agent_traces: roles of the observation messages `error_patterns` and `error_field` read. |
| `error_field` | str | `""` | agent_traces: a per-message key marking an error when truthy, on the assistant turn or its observations. |
| `error_invalid_calls` | bool | `false` | agent_traces: a call to a tool the trace does not declare, or with arguments that are not a JSON object, marks a failed turn. |
| `error_weight` | float | `1.0` | agent_traces: loss weight of a failed turn's tokens, regenerated and recorded (weights renormalized over the batch). Needs a failure signal above. |
| `first_error_weight` | float | `1.0` | agent_traces: loss weight of the first failed turn of a failed episode (replaces `error_weight` there). |
| `success_field` | str | `""` | agent_traces: row field telling whether the episode succeeded (falsy or <= 0 = failed); `first_error_weight` applies to failed episodes only. Empty = every episode. |
| `error_branches` | int | `0` | agent_traces: extra regenerated turns per trace at failed turns, the first one first. |
| `cutoff_logp` | float | `0.0` | agent_traces: teacher-reliability cutoff (`opd.traces.TeacherCutoff`, STRIDE): supervision stops after the first recorded turn where the statistic of per-turn mean teacher log-probs falls below this (< 0; 0 = off). Training only. |
| `cutoff_window` | int | `0` | 0: the statistic is the sum of per-turn means from the first turn (STRIDE); W >= 1: their mean over the last W turns. |
| `cutoff_tokens` | int | `0` | Each turn's mean covers its first this many tokens (0 = all). |
| `cutoff_weight` | float | `0.0` | Loss weight of the turns after the cutoff, recorded and regenerated (0 = dropped; < 1). |
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
| `backend` | str | `hf` | `hf` (the trainer's `generate`), `vllm` (in-process vLLM on the student's GPU, asleep while the trainer trains), `vllm_server` (a vLLM server fed weights over CUDA IPC; experimental), `sglang` (in-process SGLang engine, asleep while the trainer trains) or `sglang_server` (an SGLang server on this GPU or at `url`). See [Rollout engines](../guides/rl.md#rollout-engines-vllm-or-sglang). |
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
| `url` | str | `""` | vllm_server: a running server (started with `--weight-transfer-config '{"backend": "ipc"}'` on the trainer's GPU); sglang_server: a running SGLang server (started with `--skip-tokenizer-init`). |
| `sglang_args` | dict | `{}` | sglang backends: extra SGLang engine arguments (ServerArgs fields); those the options here set are refused. |
| `weight_sync` | str | `auto` | sglang_server: `tensor`, `distributed` or `disk`, as in `pgs rl`. |
| `weight_sync_dir` | str | `""` | `weight_sync: disk`: default `<output_dir>/rollout_weights`. |
| `weight_bucket_mb` | int | `512` | sglang backends: bf16 weights per update call. |

### loss

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | `8` | Teacher top-k for `topk_kl` and the xtok dense term. |
| `beta` | float | `1.0` | `topk_kl` and dense term: weight of the reverse KL (1 − beta: forward KL). |
| `is_low` / `is_high` | float | `0.5` / `2.0` | `sampled_rkl`/`xtok`: tokens whose importance ratio to the rollout policy falls outside the range get no gradient (ICE-POP). |
| `length_norm` | bool | `false` | Mean per sequence, then over sequences, instead of per token over the batch. |
| `xtok_spread` | str | `chunk` | `chunk`: every token of a chunk gets the chunk's advantage. `proportional`: token $t$ gets $A_c \log p(t) / \log p(c)$, its share of the chunk $c$'s advantage $A_c$. |
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
| `stop_at` | int | `0` | Stop after this step (0 = run to `steps`). The LR schedule stays the one of `steps`, so a run can be done in chunks (`stop_at: 50`, then `resume_from: auto` with `stop_at: 100`, …) and follow one schedule; the stop step is evaluated and saved as a resumable `step_<stop_at>`, and `final` is written only at `steps`. |

### logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_every` | int | `1` | Train-metric cadence. |
| `use_wandb` | bool | `false` | Mirror metrics to wandb; failures degrade to console, never kill the run. |
| `use_trackio` | bool | `false` | Mirror metrics to trackio. |
| `trackers` | list | `[]` | Further metric backends, `module:Class` or `path/to/file.py:Class` ([Observability](../guides/observability.md#your-own-backend)). |
| `project` | str | `palingenesis-opd` | Tracker project. |
| `run_name` | str | `""` | Run name (empty = auto). |
| `dump_trajectories` | int | `0` | Every N steps, write the batch's completions to `<output_dir>/rollouts/` (0 = never); read them with `pgs rollouts`. |
| `dump_groups` | int | `0` | Prompts (with all their rollouts) per dumped batch, drawn at random (0 = all). |
| `dump_content` | bool | `true` | `false`: counts and log-probs only, no prompts or completions. |
