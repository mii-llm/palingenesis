# Preference Optimization (DPO)

*Teach the model which of two answers is better, using the same trainer, chat templates and masking as SFT, without ever holding the full logits in memory.*

---

## How it fits

Preference training is a mode of `pgs train`, not a separate tool. Set `dpo.enabled: true` and point `data.dataset` at preference pairs. Everything else works as it does for SFT: the optimizer, LR schedule, FSDP, checkpointing, best-model tracking and, above all, the chat-template rendering and assistant masking.

```
┌────────────┐         ┌─────────────────────┐        ┌──────────────┐
│   PAIR     │ render  │  [chosen; rejected] │ score  │   POLICY     │──┐
│ prompt +   │────────▶│  same template and  │───────▶│  log-probs   │  │  -log σ(β·Δ)
│ chosen /   │  (SFT   │  masking as SFT     │        ├──────────────┤  ├──────────────▶ loss
│ rejected   │  path)  │                     │───────▶│  REFERENCE   │──┘
└────────────┘         └─────────────────────┘        │  (frozen)    │
                                                      └──────────────┘
```

A DPO run therefore scores exactly the tokens an SFT run on the same conversations would train on.

## Quickstart

```bash
pgs train --config configs/dpo_qwen35_08b.yaml
```

The minimum a config needs:

```yaml
model:
  name_or_path: ./checkpoints/my-sft/final   # policy, and the reference by default
  torch_dtype: float32                       # see "Precision" below

data:
  dataset: ./data/dpo/train.jsonl
  streaming: false
  max_seq_length: 6144
  last_turn_only: true
  eval_dataset: ./data/dpo/eval.jsonl        # held-out pairs (optional)

dpo:
  enabled: true
  beta: 0.1
  sft_weight: 0.2

train:
  per_device_batch_size: 2                   # PAIRS per micro-batch
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-7                      # full fine-tune: far below SFT rates
```

Any field can be overridden from the CLI, e.g. `--dpo.beta 0.05`.

## Data format

Each row is a preference pair. The prompt can be explicit:

```json
{"prompt":   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen":   [{"role": "assistant", "content": "the answer", "reasoning": "the trace"}],
 "rejected": [{"role": "assistant", "content": "a worse answer"}],
 "chat_template_kwargs": {"enable_thinking": true}}
```

or implicit, with both sides written out as full conversations:

```json
{"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- **Completions** can be a single assistant message, a multi-turn continuation (tool calls and tool results included) or a plain string. A string prompt becomes one user turn.
- **Reasoning** goes in `reasoning`, the field OpenAI-compatible servers such as vLLM use. The legacy `reasoning_content`, a `think` field and inline `<think>…</think>` are also read. Like vLLM, palingenesis passes the trace to the chat template under both `reasoning` and `reasoning_content`, so templates that still read `reasoning_content` in their Jinja (Qwen3.5's among them) render it.
- **`chat_template_kwargs`** applies per row, to every render of that row. One dataset can therefore mix thinking and non-thinking pairs.
- **Field names** are configurable: `dpo.prompt_field`, `dpo.chosen_field`, `dpo.rejected_field`.

### What gets scored

| Setting | Effect |
|---|---|
| `data.last_turn_only: true` | Only the final assistant turn is scored. Use it when the prompt contains earlier assistant turns that must not count. |
| `data.last_turn_only: false` | Every assistant turn in the completion is scored. |
| `data.train_on_reasoning: false` | Reasoning tokens are excluded, so the preference applies to the final answer only. |

### Long answers

!!! warning "The chosen answer is never truncated"
    A pair whose chosen side does not fit `data.max_seq_length` is **dropped**. A truncated reference would teach the model to stop mid-answer.

The rejected side is truncated (its start kept) when `dpo.truncate_rejected: true`, the default. That suits degenerate rejections such as repetition loops, which can run to thousands of tokens. With `false`, such pairs are dropped too. Pairs whose two sides render identically are dropped, since there is no preference to learn. Drop and truncation counts are logged when the eval set loads.

## Objectives

`Δ` is the policy's log-ratio of chosen minus that of rejected, both relative to the reference. `Δ_avg` uses per-token averages instead of sums.

| `loss_type` | Loss | Source |
|---|---|---|
| `sigmoid` | `-log σ(β·Δ)` | DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) |
| `hinge` | `max(0, 1 - β·Δ)` | SLiC-HF ([arXiv:2305.10425](https://arxiv.org/abs/2305.10425)) |
| `ipo` | `(Δ_avg - 1/(2β))²` | IPO ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)) |
| `robust` | `((1-ε)·ℓ(Δ) - ε·ℓ(-Δ)) / (1-2ε)`, `ε = label_smoothing` | rDPO ([arXiv:2403.00409](https://arxiv.org/abs/2403.00409)) |
| `sigmoid_norm` | `-log σ(β·Δ_avg)` | Length-normalised DPO |

Two modifiers combine with any of them:

**`sft_weight`** adds `sft_weight × mean NLL of the chosen answer` (as in RPO, [arXiv:2404.19733](https://arxiv.org/abs/2404.19733)). Pure DPO can widen the margin by pushing the rejected answer down while the chosen one falls too. The anchor keeps the chosen answer likely, so the objective becomes "produce this" and not merely "avoid that".

**`ld_alpha < 1`** is LD-DPO ([arXiv:2409.06411](https://arxiv.org/abs/2409.06411)). Tokens up to the length both answers share count fully, and the longer answer's tail counts `ld_alpha`. This counters DPO's bias towards longer answers.

!!! tip "When not to use `ld_alpha`"
    Leave it at `1.0` when the defect you are training against **is** the tail, for example a rejected answer that loops. Down-weighting the tail would blunt exactly the signal you want.

The reference model is frozen: `dpo.reference_model`, or the starting policy when that is empty. Dropout in the policy is disabled by default (`dpo.disable_dropout`). Otherwise its log-probabilities would be noisy against a deterministic reference, and the implicit reward would carry that noise.

## Precision: keep fp32 weights

!!! danger "Use `model.torch_dtype: float32`"
    The optimizer updates weights in their loaded dtype, without an fp32 master copy. A bf16 weight resolves about 0.4 % of its magnitude, while an AdamW step at a DPO learning rate (~1e-6) is thousands of times smaller. In bf16 nearly every update therefore rounds to zero. The policy never leaves the reference: rewards, margins and accuracy stay at exactly 0.

Compute still runs in bf16 under `train.bf16` autocast. `Config.validate()` warns if DPO is enabled with non-fp32 weights.

## Memory: no full logits

Every objective above is a function of per-sequence scores `s_b = Σ_t a[b,t] · log π(y_t)`, where `a` is the scored-token mask, LD-weighted when enabled. By the chain rule,

```
∂L / ∂log π(y_t)  =  (∂L / ∂s_b) · a[b,t]
```

and `∂L/∂s_b` comes cheaply from a no-grad pass. The step runs in chunks along the sequence, reusing the chunked cross-entropy machinery:

1. **No grad:** compute per-token log-probs for the policy and the reference, chunk by chunk.
2. **Scores:** evaluate the loss on the tiny `[2P]` score tensor and differentiate it to get per-sequence weights.
3. **With grad:** backpropagate `Σ weight · log π` chunk by chunk, then bridge the accumulated hidden-state gradient into the backbone.

The `[B, S, V]` logits are never materialised, and neither is any other full-vocabulary quantity for the whole sequence. On large vocabularies (Qwen3.5: 248k) these would otherwise dominate memory. The test suite checks that loss and parameter gradients equal a naive full-logits implementation, for every objective, with and without `sft_weight` and `ld_alpha`, across chunk counts.

**Measured:** Qwen3.5-0.8B with fp32 weights, pairs up to 6,144 tokens, one pair per micro-batch and full activation checkpointing used about **27 GB** peak on an A100.

## Batching and normalization

- **Micro-batches:** `train.per_device_batch_size` counts **pairs**. A micro-batch is `[2P, S]`, chosen rows first, so one policy forward and one reference forward score both sides.
- **Preference term:** summed over pairs and divided by `pairs × world_size × grad_accum`, so accumulated and data-parallel gradients are exactly the gradient of the mean over the effective batch.
- **`sft_weight` term:** divided by the global chosen-token count × grad accum, like SFT's token denominator.

## Metrics

| Metric | Meaning |
|---|---|
| `train/dpo/loss` | Preference loss, mean over pairs |
| `train/dpo/sft_nll` | Mean NLL of chosen tokens (when `sft_weight > 0`) |
| `train/rewards/chosen`, `train/rewards/rejected` | `β · (log π - log π_ref)` for each side |
| `train/rewards/margins` | Chosen reward minus rejected reward |
| `train/rewards/accuracies` | Share of pairs where chosen outscores rejected |
| `train/logps/chosen`, `train/logps/rejected` | Summed policy log-probs per sequence |
| `eval/loss`, `eval/rewards/*` | The same on `data.eval_dataset`. `eval/loss` selects the best checkpoint. |

Training metrics are averaged over the accumulation window and across ranks. The reference scores of the fixed eval pairs are computed once and cached.

### Reading them

- **Step 1:** `dpo/loss = log 2 ≈ 0.693` and a margin of exactly 0. The policy is the reference.
- **Accuracy at 1.0 within a few steps** means the pairs are easy, e.g. a clean answer against a long loop. The preference is learned quickly, and the margin then grows by suppressing the rejected side. Watch `logps/chosen`: if it falls, raise `sft_weight`.
- **`eval/loss` rising while train loss falls** is ordinary overfitting. The best checkpoint is kept.

## Not supported with DPO

`Config.validate()` rejects the following, because each assumes one SFT sequence per row or changes the token stream:

- packing, `data.sources`, `data.eval_sources`;
- pretokenize, MSFT tracking, the sequence-length curriculum, pretrain replay;
- `preprocess.enabled`;
- context parallel and gradient release;
- the token-weighting plugins: DFT, CADFT, DEFT, InfoSFT and pre-RL.

See the [`dpo` configuration reference](../reference/configuration.md#dpo) for every field.
