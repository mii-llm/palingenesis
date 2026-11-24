# Preference Optimization (DPO)

*Teach the model which of two answers is better, using the same trainer, chat templates and masking as SFT, without ever holding the full logits in memory.*

---

## How it fits

Preference training is a mode of `pgs train`, not a separate tool. Set `dpo.enabled: true` and point `data.dataset` at preference pairs. Everything else works as it does for SFT: the optimizer, LR schedule, FSDP, checkpointing, best-model tracking and, above all, the chat-template rendering and assistant masking.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 250" role="img" aria-labelledby="fig-dpo-title fig-dpo-desc">
<title id="fig-dpo-title">A DPO step: one batch of chosen and rejected rows, scored by the policy and the frozen reference</title>
<desc id="fig-dpo-desc">A preference pair, a prompt with a chosen and a rejected answer, is rendered with the same chat template and assistant masking as SFT into a micro-batch whose chosen rows come first. One policy forward and one reference forward score both sides. The summed log-probabilities of the scored tokens give Delta, the preference loss is computed from it, and the gradient reaches the policy only; the reference is frozen.</desc>
<defs>
<marker id="dpo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="dpo-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head-accent" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<!-- pair -->
<text class="t-label" x="10" y="62">pair</text>
<rect class="box" x="10" y="72" width="118" height="100" rx="4"/>
<text x="22" y="96">prompt</text>
<text x="22" y="124">chosen <tspan class="t-mono">y_w</tspan></text>
<text x="22" y="152">rejected <tspan class="t-mono">y_l</tspan></text>
<path class="line" d="M128,122 H160" marker-end="url(#dpo-arrow)"/>
<!-- batch -->
<text class="t-label" x="162" y="62">micro-batch [2P, S]</text>
<rect class="box-plain" x="162" y="72" width="150" height="100" rx="4"/>
<rect class="fill-soft" x="170" y="80" width="134" height="20" rx="2"/><text class="t-small" x="178" y="94">chosen rows</text>
<rect class="fill-raised" x="170" y="104" width="134" height="20" rx="2"/><text class="t-small" x="178" y="118">rejected rows</text>
<text class="t-muted t-small" x="170" y="146">SFT chat template</text>
<text class="t-muted t-small" x="170" y="161">and assistant masking</text>
<path class="line" d="M312,112 C330,112 330,60 352,60" marker-end="url(#dpo-arrow)"/>
<path class="line" d="M312,132 C330,132 330,184 352,184" marker-end="url(#dpo-arrow)"/>
<!-- models -->
<rect class="box-accent" x="354" y="36" width="128" height="48" rx="4"/>
<text x="418" y="57" text-anchor="middle">policy π</text><text class="t-small" x="418" y="73" text-anchor="middle">trained</text>
<rect class="box-quiet" x="354" y="160" width="128" height="48" rx="4"/>
<text x="418" y="181" text-anchor="middle">reference π_ref</text><text class="t-muted t-small" x="418" y="197" text-anchor="middle">frozen, no grad</text>
<!-- scores -->
<path class="line" d="M482,60 C510,60 510,110 536,114" marker-end="url(#dpo-arrow)"/>
<path class="line" d="M482,184 C510,184 510,134 536,130" marker-end="url(#dpo-arrow)"/>
<text class="t-muted t-small" x="494" y="44">Σ log π per side</text>
<text class="t-muted t-small" x="494" y="216">Σ log π_ref per side</text>
<circle class="box" cx="556" cy="122" r="18"/><text x="556" y="127" text-anchor="middle">Δ</text>
<path class="line" d="M574,122 H594" marker-end="url(#dpo-arrow)"/>
<rect class="box" x="596" y="100" width="96" height="44" rx="4"/>
<text x="644" y="121" text-anchor="middle">loss</text><text class="t-mono" x="644" y="136" text-anchor="middle">−log σ(βΔ)</text>
<!-- gradient -->
<path class="line-accent" d="M644,100 V18 H418 V34" marker-end="url(#dpo-arrow-a)" stroke-dasharray="4 3"/>
<text class="t-accent t-small" x="530" y="12" text-anchor="middle">gradient: policy only</text>
<text class="t-muted t-small" x="10" y="240">Only the scored (assistant) tokens enter the sums; the full logits are never materialised.</text>
</svg>
</div>
<figcaption>One DPO step. Both answers go through the SFT rendering path into one batch; the policy and the frozen reference each score both sides, and only the policy receives the gradient of the preference loss (here the default <code>sigmoid</code> objective).</figcaption>
</figure>

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
- **Both at once**, as in `HuggingFaceH4/ultrafeedback_binarized` (a `prompt` string, and `chosen`/`rejected` conversations that already start with that user turn): the completions are used as they are, so the prompt is not repeated.
- **Tool definitions** come from the row's `tools` field (`data.tools_field`) and are rendered into both sides, as in SFT.
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

### Contrastive twin pairs

Twins are prompts that differ by a small edit and need different answers: "the store closes at 18:00" and "the store closes at 17:00", a negated condition, a swapped unit, an added constraint. A model that misreads one twin as the other gives its sibling's answer. That answer is the most informative rejected side a prompt can have: chosen and rejected then differ exactly where the two prompts differ, and no length, style or fluency cue separates them for the model to learn instead. Written in both directions (P: A over A', and P': A' over A), the pair teaches reading the prompt, not preferring one answer.

```bash
# twins.jsonl: {"twin": "store-1", "prompt": "...", "answer": "Yes"}, rows sharing `twin` are siblings
pgs pairs build twins.jsonl -o twin_pairs.jsonl
pgs pairs build twins.jsonl -o misread.jsonl --prediction_field model_answer --only_misread
pgs pairs check dpo_rows.jsonl --require_balanced
```

`build` writes every ordered pair of siblings as a DPO row (`prompt`, `chosen`: its own answer, `rejected`: the sibling's, plus `twin`, `direction`, `prompt_edit_words`, and the row's `tools`, `chat_template_kwargs` and `sft_weight`). A pair is dropped and counted when the two answers are equal after whitespace and case folding, when the prompts are the same, or when the prompts differ by more than `--max_edit` (share of changed words and punctuation marks, default 0.5) or `--max_edit_words`. With `--prediction_field` (the model's own answer to each prompt, from your evaluation), a pair records `misread: true` when the model answered its prompt with the sibling's answer; `--only_misread` keeps those pairs only: on-policy pairs whose rejected side is the model's actual error.

`check` reads any DPO file (explicit or implicit prompts) and reports rows whose two sides are the same text, contradictions (a pair and its exact reverse on the same prompt), duplicates, the twin pairs it finds (a row whose rejected answer is the chosen answer of another row with a near-identical prompt; a twin is visible only when its sibling's answer is a chosen side somewhere in the file) and those without their reverse direction. Exit code 1 on identical sides or contradictions, and with `--require_balanced` on unbalanced twins.

The rejected side is a fluent answer that is right for the sibling prompt, so the margin can only grow by attending to the edited words. When the two answers are long and differ in one span ("the store is open" / "closed"), most of their tokens are shared and `sft_scope: edits` (below) puts the SFT anchor on the span that differs. Balance the directions, and the categories, as `check` reports: one-directional twins teach a preference for one answer instead of reading the prompt.

## Objectives

$\Delta$ is the policy's log-ratio of chosen minus that of rejected, both relative to the reference; $\Delta_{\text{avg}}$ uses per-token averages instead of sums:

$$
\Delta = \big(\log \pi(y_w) - \log \pi_{\text{ref}}(y_w)\big) - \big(\log \pi(y_l) - \log \pi_{\text{ref}}(y_l)\big)
$$

where $y_w$ is the chosen answer, $y_l$ the rejected one, $\pi$ the policy, $\pi_{\text{ref}}$ the frozen reference (log-probabilities summed over the answer's tokens), $\beta$ is `dpo.beta` and $\sigma$ the logistic function.
{: .pg-where }

| `loss_type` | Loss | Source |
|---|---|---|
| `sigmoid` | $-\log \sigma(\beta\Delta)$ | DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) |
| `hinge` | $\max(0,\ 1 - \beta\Delta)$ | SLiC-HF ([arXiv:2305.10425](https://arxiv.org/abs/2305.10425)) |
| `ipo` | $\big(\Delta_{\text{avg}} - \frac{1}{2\beta}\big)^2$ | IPO ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)) |
| `robust` | $\dfrac{(1-\varepsilon)\,\ell(\Delta) - \varepsilon\,\ell(-\Delta)}{1-2\varepsilon}$, with $\ell(x) = -\log\sigma(\beta x)$ and $\varepsilon$ = `label_smoothing` | rDPO ([arXiv:2403.00409](https://arxiv.org/abs/2403.00409)) |
| `sigmoid_norm` | $-\log \sigma(\beta\Delta_{\text{avg}})$ | Length-normalised DPO |

Two modifiers combine with any of them:

**`sft_weight`** adds `sft_weight × mean NLL of the chosen answer` (as in RPO, [arXiv:2404.19733](https://arxiv.org/abs/2404.19733)). Pure DPO can widen the margin by pushing the rejected answer down while the chosen one falls too. The anchor keeps the chosen answer likely, so the objective becomes "produce this" and not merely "avoid that".

A row may carry its own `sft_weight` field in [0, 1] (default 1), which scales the SFT term for that pair only. Set it to 0 for pairs whose chosen answer was written by another model rather than edited from the policy's own: they still teach the preference, but the policy is not trained to imitate the other model's text.

**`ld_alpha < 1`** is LD-DPO ([arXiv:2409.06411](https://arxiv.org/abs/2409.06411)). Tokens up to the length both answers share count fully, and the longer answer's tail counts `ld_alpha`. This counters DPO's bias towards longer answers.

!!! tip "When not to use `ld_alpha`"
    Leave it at `1.0` when the defect you are training against **is** the tail, for example a rejected answer that loops. Down-weighting the tail would blunt exactly the signal you want.

The reference model is frozen: `dpo.reference_model`, or the starting policy when that is empty. Dropout in the policy is disabled by default (`dpo.disable_dropout`). Otherwise its log-probabilities would be noisy against a deterministic reference, and the implicit reward would carry that noise.

## Precision: keep fp32 weights

!!! danger "Use `model.torch_dtype: float32`"
    The optimizer updates weights in their loaded dtype, without an fp32 master copy. A bf16 weight resolves about 0.4 % of its magnitude, while an AdamW step at a DPO learning rate (~1e-6) is thousands of times smaller. In bf16 nearly every update therefore rounds to zero. The policy never leaves the reference: rewards, margins and accuracy stay at exactly 0.

Compute still runs in bf16 under `train.bf16` autocast. `Config.validate()` warns if DPO is enabled with non-fp32 weights.

## Memory: no full logits

Every objective above is a function of per-sequence scores $s_b = \sum_t a_{b,t} \log \pi(y_t)$, where $a$ is the scored-token mask, LD-weighted when enabled. By the chain rule,

$$
\frac{\partial L}{\partial \log \pi(y_t)} = \frac{\partial L}{\partial s_b}\, a_{b,t}
$$

and $\partial L / \partial s_b$ comes cheaply from a no-grad pass. The step runs in chunks along the sequence, reusing the chunked cross-entropy machinery:

1. **No grad:** compute per-token log-probs for the policy and the reference, chunk by chunk.
2. **Scores:** evaluate the loss on the tiny `[2P]` score tensor and differentiate it to get per-sequence weights.
3. **With grad:** backpropagate $\sum_t w_{b,t} \log \pi(y_t)$, with the weights $w_{b,t} = \frac{\partial L}{\partial s_b} a_{b,t}$ from step 2, chunk by chunk, then bridge the accumulated hidden-state gradient into the backbone.

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
| `train/rewards/chosen`, `train/rewards/rejected` | $\beta\,(\log \pi - \log \pi_{\text{ref}})$ for each side |
| `train/rewards/margins` | Chosen reward minus rejected reward |
| `train/rewards/accuracies` | Share of pairs where chosen outscores rejected |
| `train/logps/chosen`, `train/logps/rejected` | Summed policy log-probs per sequence |
| `eval/loss`, `eval/rewards/*` | The same on `data.eval_dataset`. `eval/loss` selects the best checkpoint. |

Training metrics are averaged over the accumulation window and across ranks. The reference scores of the fixed eval pairs are computed once and cached.

### Reading them

- **Step 1:** `dpo/loss` $= \log 2 \approx 0.693$ and a margin of exactly 0. The policy is the reference.
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
