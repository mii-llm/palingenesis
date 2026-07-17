# On-Policy Distillation (OPD)

Distill a large teacher into a small student by scoring the student's **own samples** — not a fixed dataset of teacher outputs.

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl  # optional: annotate pool with teacher answers
pgs distill       --config configs/distill_opd.yaml
pgs distill       --config configs/distill_opd.yaml --train.learning_rate 5e-6 --train.steps 3000
```

Package: `palingenesis.opd` (`config.py`, `trainer.py`, `sources.py`, `token_bridge.py`, `formatting.py`, `pool.py`, `score_pool.py`).

**Architecture**: the *engine* (token bridge, on-policy sampling, reverse-KL scoring loss, checkpointing) is task-agnostic; everything task-shaped lives behind the `PromptSource` protocol (`sources.py`) — what conversation to roll out next, how to evaluate held-out prompts, which per-batch stats to log. Two sources ship, selected by `data.format`:

| `data.format` | prompts | dev metric |
|---|---|---|
| `mcqa` (default) | pool-row JSONL, shot regimes, fast/CoT templates | letter accuracy (`dev_acc`) |
| `messages` | JSONL of `{"messages": [...]}`, last turn = user | held-out reverse KL (`dev_kl`) + length |

A custom source is any object with `sample()` / `evaluate(engine)` / `batch_stats()`; pass it as `OPDTrainer(config, source=...)`.

## Generic chat distillation

```bash
pgs distill --config configs/distill_chat.yaml
```

This is OPD at its most valuable: on long-form generation (chat, reasoning, code, agentic traces), exposure bias compounds hardest, and on-policy correction is exactly the fix. Free-form answers can't be auto-graded, so the dev metric is the mean reverse KL to the teacher on held-out prompts — the trained quantity, measured out of sample. Rows whose last message is not a user turn are skipped with a warning. Practical knobs for long completions: small `train.score_micro_seqs` (4), `sampling.group_size 2` for rollout diversity, and a real `max_new_tokens` budget.

## Why on-policy

Offline distillation (teacher generates → student SFTs) trains the student on the *teacher's* trajectory distribution. At inference the student is on its own distribution, and every deviation compounds — the classic exposure-bias problem. OPD closes the gap:

1. Draw prompts from the pool, render with the **student's** chat template.
2. The student samples completions at temperature 1.0 with its **current** weights (exactly on-policy: one gradient step per batch, no importance sampling).
3. The teacher scores the same completions, conditioned on the same conversation rendered with the **teacher's** chat template.
4. Loss = full-distribution reverse KL over completion tokens:

   $$\mathcal{L} = \sum_{t}\sum_{v} p_\theta(v \mid x_{<t})\,\bigl(\log p_\theta(v \mid x_{<t}) - \log p_T(v \mid x_{<t})\bigr)$$

   Reverse KL is mode-seeking: the student concentrates on what the teacher considers likely *where the student actually goes*, instead of smearing mass over everything the teacher might say.

`train.loss_fn: sampled_rkl` switches to the sampled-token REINFORCE variant (per-token advantage = −sampled KL), which reproduces tinker-cookbook's `on_policy_distillation`. `full_kl` is a strictly lower-variance estimator and the default.

## The token bridge

Exact per-token KL requires both models to assign the *same ids* to the same text. The supported setup (`palingenesis/opd/token_bridge.py`):

- Student and teacher share a base vocabulary (ids `0..shared_vocab_size-1`, detected as the teacher's vocab size).
- The student may **extend** it with template/tool tokens the teacher cannot embed (e.g. a ChatML student adds `<|im_end|>` at 128256 on top of a 128256-id Llama-3 teacher vocab).
- Prompts are rendered per-model with each model's own chat template; only completion tokens are aligned.
- End-of-turn merge: the student's terminator and the teacher's terminator mean the same thing, so the student's terminator probability mass is added to the teacher's terminator slot before the KL. The teacher then supervises *when to stop*, not just what to say. Configure with:

```yaml
bridge:
  eos_map:
    "<|im_end|>": "<|eot_id|>"
  extra_stop_tokens: ["<|end_of_text|>"]
```

Set `eos_map` explicitly for base-model teachers: auto-detection uses the teacher's *configured* eos, which is often `<|end_of_text|>` rather than the conversational terminator.

`check_compatible()` runs before any weights load: probe texts must tokenize identically in both tokenizers (accented text, code, prose), swap sources must lie outside the shared vocab, swap targets inside it. A near-miss tokenizer pair silently degrades KL to noise, so this check raises instead of warning.

Sampled completions are truncated at the first stop token (**inclusive** — stopping is supervised too); any other student-only token (tool tokens etc.) truncates the completion before it, since the teacher has nothing to score there.

## Prompt pool and shot regimes

The pool is a JSONL of multiple-choice rows (`palingenesis.opd.pool`):

```json
{"question": "...", "options": [["A", "..."], ["B", "..."]], "answer": "A",
 "category": "storia", "source": "mmlu_italian"}
```

**Dedup against the target benchmark is not optional.** Training pools are often drawn from the same corpora a benchmark was curated from; hash every candidate question (`question_hash` — lowercased, accent-stripped, alphanumeric-only) against the benchmark set (`load_benchmark_hashes`) before it enters the pool. Adapters for common Italian MCQA sources ship in `pool.py` (`normalize_pinocchio`, `normalize_mmlu_pro_ita`, `normalize_mmlu_italian`).

The train/dev split is deterministic by question hash — stable across runs and input order.

## Teacher-correct filtering (`pgs distill-score`)

Reverse KL transfers the teacher's *errors* along with its knowledge: on a pool where the teacher is right half the time, half the supervision pulls the student toward wrong answers, and the teacher's accuracy is a hard ceiling. When the pool has verifiable answers, annotate it with the teacher's own answer first:

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
```

Each row is written back with `teacher_answer` (the option letter the teacher assigns the highest logit) and `teacher_correct`. The module only annotates — dropping incorrect rows, downweighting them, or rebalancing categories is downstream policy, same as the score-then-select flow of `pgs prepare` on the SFT side.

Scoring is one batched forward per row, no generation: the fast-mode prompt ends right before the answer, so the teacher's choice is read from the option-letter logits at the final prompt position. No decoding loop, no format parsing, no unparseable outputs. ~110k rows with 5-shot prompts score in about an hour on an A100.

Each training prompt is rendered with a randomized shot regime (`formatting.PromptRenderer`):

| Regime | Probability | Purpose |
|--------|-------------|---------|
| reference shots | `data.p_reference_shots` | exactly what the benchmark harness sends |
| 1–k pool shots | `data.p_pool_shots` | format generalization |
| zero-shot | remainder | robustness |

Prompt templates are **config, not code**: training against a benchmark means training on its *exact* prompt bytes, and that's policy. Set `data.fast_template` / `data.cot_template` / `data.system_message` to the benchmark's verbatim templates (placeholders: `{question}`, `{options}` required; `{topic}`, `{merged_letters}` optional — validated at startup). The library defaults are neutral English MCQA templates; `configs/distill_opd.yaml` carries ITALIC's verbatim Italian ones, byte-locked by a test.

## Memory

Student (fp32 + bf16 autocast, gradients) and teacher (bf16, frozen) share one GPU by default. Logits are never materialized for the full sequence — hidden states are gathered at completion positions first, keeping the big tensor at `N_completion_tokens × vocab`. Validated reference point: a 0.4B student + 3B teacher at `batch_prompts 32` with 5-shot prompts needs `score_micro_seqs 8` + `gradient_checkpointing true` (~24 GB on an A100); `score_micro_seqs 32` OOMs the same 80 GB card in the student scoring forward. The knobs:

| Knob | Effect |
|------|--------|
| `train.score_micro_seqs: 8` | fewer sequences per scoring forward (grad-accum keeps the math identical) |
| `model.gradient_checkpointing: true` | recompute student activations |
| `model.teacher_device: "cuda:1"` | move the teacher off the student's GPU |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fragmentation headroom for mixed generate/score phases |

## Monitoring

Every `logging.log_every` steps: `kl/tok` (should fall), `sampled_kl`, `residual_mass` (student probability on unmapped student-only tokens — should stay ≈0; if it grows, the student is drifting toward tokens the teacher can't see), `len` (mean completion length), `fmt_ok` (fraction of completions containing a valid option letter), and greedy dev accuracy every `train.eval_every` steps. `logging.use_wandb: true` mirrors everything to wandb; tracker failures degrade to console logging and never kill the run.

Gains typically saturate well before the KL stops falling — checkpoint often (`train.save_steps`) and pick by dev accuracy, not by KL.
