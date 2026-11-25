# Preference optimisation (DPO)

Palingenesis trains DPO and its common variants with the same trainer as SFT:
set `dpo.enabled: true` and point `data.dataset` at preference pairs. The
optimizer, LR schedule, FSDP, checkpointing, best-model tracking and, above all,
the chat-template rendering and masking are the SFT ones, so a DPO run scores
exactly the tokens an SFT run on the same conversations would train on.

```bash
torchrun --standalone --nproc_per_node=1 -m palingenesis.train --config configs/dpo_qwen35_08b.yaml
```

## Data

Rows use the conversational preference format, with an explicit or an implicit prompt:

```json
{"prompt":   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen":   [{"role": "assistant", "content": "the answer", "reasoning": "the trace"}],
 "rejected": [{"role": "assistant", "content": "a worse answer"}],
 "chat_template_kwargs": {"enable_thinking": true}}
```

```json
{"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- **Completions** can be a single assistant message, several messages (a
  multi-turn continuation, e.g. tool calls and tool results) or a plain string.
  A string prompt becomes one user turn.
- **Reasoning** goes in `reasoning`, the OpenAI/vLLM field. The legacy
  `reasoning_content` and inline `<think>…</think>` are also read. As vLLM
  does, the trace is passed to the chat template under both `reasoning` and
  `reasoning_content`, so templates whose Jinja still reads
  `reasoning_content` (Qwen3.5's among them) render it.
- **`chat_template_kwargs`** is applied per row to every render, so one dataset
  can mix thinking and non-thinking pairs (e.g. `enable_thinking`).
- **Row fields** are renamed with `dpo.prompt_field`, `dpo.chosen_field` and
  `dpo.rejected_field`.

### What is scored

Each side is rendered as a whole conversation with the model's own chat
template, by the same code as SFT. Only assistant tokens are scored:

| Setting | Effect |
|---|---|
| `data.last_turn_only: true` | Score only the final assistant turn. Use it when the prompt contains earlier assistant turns that must not count. |
| `data.last_turn_only: false` | Score every assistant turn in the completion. |
| `data.train_on_reasoning: false` | Exclude reasoning tokens from the score, so only the final answer is compared. |

### Length

The chosen answer is never truncated. A pair whose chosen side does not fit
`data.max_seq_length` is dropped: a cut reference teaches the model to stop
mid-answer. The rejected side is truncated (its start kept) when
`dpo.truncate_rejected: true`, which suits degenerate rejected answers such as
repetition loops. With `false`, such pairs are dropped. Pairs whose two sides
render identically are dropped, since there is no preference to learn.

## Objectives

`delta` is the policy log-ratio of chosen minus that of rejected, both relative
to the reference. `delta_avg` uses per-token averages instead of sums.

| `loss_type` | Loss | Source |
|---|---|---|
| `sigmoid` | `-log σ(β·delta)` | DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) |
| `hinge` | `relu(1 - β·delta)` | SLiC-HF ([arXiv:2305.10425](https://arxiv.org/abs/2305.10425)) |
| `ipo` | `(delta_avg - 1/(2β))²` | IPO ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)) |
| `robust` | `((1-ε)·ℓ(delta) - ε·ℓ(-delta)) / (1-2ε)`, `ε = label_smoothing` | rDPO ([arXiv:2403.00409](https://arxiv.org/abs/2403.00409)) |
| `sigmoid_norm` | `-log σ(β·delta_avg)` | Length-normalised DPO |

Two modifiers combine with any objective:

- **`sft_weight`** adds `sft_weight × mean NLL of the chosen tokens` (as in RPO,
  [arXiv:2404.19733](https://arxiv.org/abs/2404.19733)). It keeps the chosen
  answer likely. Without
  it, the margin can grow purely by pushing the rejected answer down, which
  pure DPO is known to do. **`sft_scope: edits`** restricts that NLL to the
  chosen tokens where chosen and rejected differ (a token-level diff, widened by
  `sft_edit_window` tokens, default 8). Use it for minimal-edit pairs, whose
  chosen answer is mostly the policy's own text: plain RPO would then be SFT on
  that uncorrected text. The default `all` is RPO as published.
- **`ld_alpha < 1`** is LD-DPO ([arXiv:2409.06411](https://arxiv.org/abs/2409.06411)): tokens up to the length both answers share count
  fully, and the longer answer's tail counts `ld_alpha`. Leave it at 1.0 when the
  defect you are training against *is* the tail, e.g. a rejected answer that
  loops. Down-weighting the tail would then blunt the signal.

The reference is frozen: `dpo.reference_model`, or the starting policy when
empty. Dropout in the policy is disabled by default (`dpo.disable_dropout`).
Otherwise the policy's log-probabilities are noisy against a deterministic
reference.

## Precision: keep fp32 weights

Set `model.torch_dtype: float32`. The optimizer updates the weights in their
loaded dtype, without an fp32 master copy. A bf16 weight resolves about 0.4 % of
its magnitude, while an AdamW step at a DPO learning rate (~1e-6) is far smaller,
so in bf16 nearly every update rounds to zero. The policy then never leaves
the reference: rewards, margins and accuracy stay at exactly 0.
`Config.validate()` warns about this. Compute still runs in bf16 under
`train.bf16` autocast.

## Memory: no full logits

Every objective above is a function of per-sequence scores
`s_b = Σ_t a[b,t] · log π(y_t)`, so `∂L/∂log π(y_t) = (∂L/∂s_b) · a[b,t]`. The
step therefore runs in chunks along the sequence, reusing the chunked-CE
machinery:

1. **No-grad pass:** compute per-token log-probs for the policy and the reference.
2. **Weights:** evaluate the loss on the small `[2P]` score tensors and
   differentiate it to get per-sequence weights.
3. **Weighted backward:** backpropagate `Σ weight · log π` chunk by chunk and
   bridge the hidden-state gradient into the backbone.

The `[B, S, V]` logits are never materialised, nor is any other full-vocabulary
quantity for the whole sequence. On large vocabularies (Qwen3.5: 248k) these
would otherwise dominate memory. `tests/test_dpo.py` checks that the loss and parameter
gradients equal a naive full-logits implementation for every objective, with
and without `sft_weight` and `ld_alpha`, across chunk counts.

## Batching and normalisation

- **Micro-batches:** `train.per_device_batch_size` counts **pairs**. A
  micro-batch is `[2P, S]`, chosen rows first, and one policy forward scores
  both sides.
- **DPO term:** summed over pairs and divided by
  `pairs × world_size × grad_accum` (the micro-batch's pairs over all ranks). Accumulated and data-parallel gradients
  therefore equal the gradient of the mean loss over the whole effective batch
  (tested).
- **`sft_weight` term:** divided by the global count of chosen tokens × grad
  accum, like SFT's token denominator.

## Prompt-sharing packing (`dpo.packing`)

A padded micro-batch computes every pair's prompt twice (once under the chosen answer, once under the
rejected one), for the policy and again for the reference, and pads every row to the longest. With
`dpo.packing: tree` a micro-batch is one packed row instead
(the prefix-tree packing of `pgs rl`'s `train.packing: tree`):

```yaml
model:
  attn_implementation: sdpa   # routed to the variable-length flash kernel over packed rows
  compile: false              # compiled layers are refused with packed rows
dpo:
  enabled: true
  packing: tree               # padded (default) | varlen | tree
  micro_tokens: 16384         # optional: fill micro-batches by packed tokens instead of a pair count
```

- **Layout.** Every pair is a two-leaf prefix trie: the tokens its two sides share are forwarded once, then
  the chosen and the rejected tails, each attending to the shared prefix and to itself. What is shared is
  the common token prefix of the two rendered conversations: the prompt, earlier turns of a multi-turn
  prompt, and the first tokens the two answers have in common (minimal-edit pairs share up to the first
  edit). Pairs share nothing with each other, so a row has two levels whatever it holds: all the shared
  prefixes, then all the tails. Gated DeltaNet layers (Qwen3.5) run once per level, the tails starting from
  their prefix's final state. No row is padded. `varlen` lays the 2P sequences end to end: no padding, no
  sharing.
- **Policy and reference.** Both forwards use the packed row, so the prompt is computed twice per pair
  (policy, reference) instead of four times.
- **Same numbers.** Labels, score weights and the objectives stay in the `[2P, S]` layout; only the hidden
  states come from the packed row. Per-token and per-sequence log-probs, the loss and the gradients equal
  the padded path's up to floating-point order, for every `loss_type`, `ld_alpha`, `sft_weight` and
  `sft_scope`. Tokens scored inside a shared prefix (earlier assistant turns with
  `data.last_turn_only: false`) go through the output head once and carry both sides' weights.
- **When a row stays flat.** A tree level costs the recurrent layers one more kernel call. A micro-batch
  whose pairs together share fewer than 2,048 tokens is laid out as `varlen`. Watch
  `train/dpo/forwarded_share`: tokens forwarded per token of the pairs' sequences (1.0 = nothing shared;
  0.6 at an 80% prompt share with answers of equal length).
- **Token budget.** By default a micro-batch still holds `train.per_device_batch_size` pairs, so a tree run
  sees the same micro-batches as a padded run with the same seed. With `dpo.micro_tokens > 0`, pairs are
  taken in data order while their packed tokens (a shared prefix counted once) fit the budget;
  `per_device_batch_size` then does not apply, a pair above the budget is a micro-batch of its own, and
  the number of pairs varies per micro-batch. The preference term of a micro-batch is divided by its own
  pairs (summed over ranks) times grad accum, as SFT divides by a micro-batch's tokens. The LR schedule needs
  `train.max_steps` or a non-streaming dataset (the micro-batches are counted by one scan).
- **Memory.** The attention of a tail gathers its prefix's keys and values: at most three times the packed
  tokens per layer, transient, so there is no separate gather budget (unlike RL groups of many rollouts).
- **Works with** FSDP2, LoRA and `train.gradient_checkpointing` (none, selective, full), each checked against
  the padded path on CPU, and with Liger kernels (which do not touch the layout). The chunked passes over the
  vocabulary cover the scored tokens only. Held-out evaluation uses padded rows.
- **Refused by `Config.validate()`** with `dpo.packing: varlen | tree`: `model.attn_implementation` other
  than `sdpa`, `model.compile`, `multimodal.enabled`, `memory.activation_offload`. Refused at start-up: MoE
  policies (router balancing is not laid out on packed pairs), plugins with a loss term, and models whose
  layers a packed row cannot run exactly (sliding-window attention, recurrent layers other than Gated
  DeltaNet).

### Measured

`scripts/gpu_checks/dpo_tree.sh` measures one micro-step (reference forward, policy forward, loss, backward)
on synthesised pairs (answers of 100 to 700 tokens, the prompt sized to the share), against the fastest
padded layout that ran, compiled layers included. Pre-registered target: at least 1.5x at prompt shares of
70% and above; below 1.2x the feature is dropped. One A100 80GB (about 15 GB held by another job), fp32
weights, bf16 autocast, selective checkpointing, median of 3 steps:

| Model | Prompt share | Tree speed-up over the fastest padded layout |
|---|---|---|
| Qwen3.5-0.8B | 50% | 1.50x |
| Qwen3.5-0.8B | 80% | ~2.0x |
| Qwen3.5-0.8B | 95% | ~1.9x |
| Qwen3.5-2B | 50% | 1.57x |
| Qwen3.5-2B | 80% | not recorded |
| Qwen3.5-2B | 95% | 2.1x |

The target is met at every share measured at 70% or above. `varlen` alone (no padding, no sharing) gave
1.66x at a 50% share on the 0.8B model: with unsorted pairs, much of the gain at low prompt shares is the
padding that is no longer forwarded. Step times, pairs per micro-batch and peak memory per layout are in the
check's `throughput` lines.

Equality on the GPU (Qwen3.5-0.8B, bf16): per-token log-probs of the packed layouts differ from the padded
rows by 0.0025 on average, as two padded layouts of the same pairs do; the largest difference over all
scored tokens was 0.146 (varlen) and 0.150 (tree) against 0.131 between the two padded layouts. That maximum
is one draw from the tail of bf16 rounding noise, so the check judges the mean, the 99th percentile and the
signed mean of the differences, the loss and the gradient cosine (at least 0.999) against that measured
noise floor, and reports the maximum. The gradient cosine on the GPU and the 2B equality (which ran out of
memory in the first run; the check now sizes itself to the free memory) are pending a re-run. On CPU,
float64 tiny Qwen3 and Qwen3.5 models match the padded path to 1e-10 on log-probs and loss and 1e-6
(relative) on gradients (`tests/test_dpo_packing.py`).

## Metrics

| Metric | Meaning |
|---|---|
| `train/dpo/loss` | Preference loss (mean over pairs) |
| `train/dpo/sft_nll` | Mean NLL of chosen tokens (when `sft_weight > 0`) |
| `train/rewards/chosen`, `train/rewards/rejected` | `β · (log π - log π_ref)` per side |
| `train/rewards/margins` | Chosen minus rejected reward |
| `train/rewards/accuracies` | Share of pairs with chosen reward > rejected |
| `train/logps/chosen`, `train/logps/rejected` | Summed policy log-probs per sequence |
| `train/dpo/forwarded_share` | With `dpo.packing`: tokens forwarded per token of the pairs' sequences |
| `eval/loss`, `eval/rewards/*` | The same on `data.eval_dataset`. `eval/loss` drives best-checkpoint selection. |

The training metrics are averaged over the accumulation window and across ranks.
For evaluation, the reference scores of the fixed eval pairs are computed once
and cached.

Accuracy near 1.0 early in training usually means the pairs are too easy (e.g.
a clean answer against a long loop). The preference is then learned in a few
steps, and the margin grows by suppressing the rejected side. Watch
`logps/chosen`: if it falls, raise `sft_weight`.

## Not supported with `dpo.enabled`

`Config.validate()` rejects the following. Each assumes one SFT sequence per
row, or changes the token stream:

- `data.packing` (SFT document packing; `dpo.packing` is the preference layout), `data.sources`,
  `data.eval_sources`
- pretokenize, MSFT tracking, the seq-len curriculum, pretrain replay
- `preprocess.enabled`
- context parallel, gradient release
- the token-weighting plugins: DFT, CADFT, DEFT, InfoSFT and pre-RL

## Status

- **Tested on CPU:**
  - the losses, against their published definitions;
  - exact gradient equality with a naive implementation;
  - data rendering, including the official Qwen3.5 template;
  - the dataloader, grad-accumulation normalisation and evaluation;
  - a short end-to-end learning test;
  - `dpo.packing`: equality with the padded path (float64 Qwen3 and Qwen3.5 models; FSDP2, LoRA,
    activation checkpointing) and whole trainer runs. On one A100: 1.5x to 2.1x the padded step (see Measured); the GPU
    gradient-equality check is pending a re-run.
- **Run on one A100 80GB:** Qwen3.5-0.8B, fp32 weights, pairs up to 6,144
  tokens mixing thinking and non-thinking rows, one pair per micro-batch × 16
  accumulation, full activation checkpointing. Peak memory was about 27 GB.
  Over 20 steps, held-out preference loss fell from 0.693 (log 2, policy =
  reference) to 0.288.
- **Not yet run:** multi-GPU FSDP. The reference model is sharded like the
  policy and follows the same backbone + `lm_head` path as chunked SFT.
