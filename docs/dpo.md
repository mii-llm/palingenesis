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
  pure DPO is known to do.
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
  `pairs × world_size × grad_accum`. Accumulated and data-parallel gradients
  therefore equal the gradient of the mean loss over the whole effective batch
  (tested).
- **`sft_weight` term:** divided by the global count of chosen tokens × grad
  accum, like SFT's token denominator.

## Metrics

| Metric | Meaning |
|---|---|
| `train/dpo/loss` | Preference loss (mean over pairs) |
| `train/dpo/sft_nll` | Mean NLL of chosen tokens (when `sft_weight > 0`) |
| `train/rewards/chosen`, `train/rewards/rejected` | `β · (log π - log π_ref)` per side |
| `train/rewards/margins` | Chosen minus rejected reward |
| `train/rewards/accuracies` | Share of pairs with chosen reward > rejected |
| `train/logps/chosen`, `train/logps/rejected` | Summed policy log-probs per sequence |
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

- packing, `data.sources`, `data.eval_sources`
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
  - a short end-to-end learning test.
- **Run on one A100 80GB:** Qwen3.5-0.8B, fp32 weights, pairs up to 6,144
  tokens mixing thinking and non-thinking rows, one pair per micro-batch × 16
  accumulation, full activation checkpointing. Peak memory was about 27 GB.
  Over 20 steps, held-out preference loss fell from 0.693 (log 2, policy =
  reference) to 0.288.
- **Not yet run:** multi-GPU FSDP. The reference model is sharded like the
  policy and follows the same backbone + `lm_head` path as chunked SFT.
