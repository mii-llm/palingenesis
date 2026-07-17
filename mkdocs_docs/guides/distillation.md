# On-Policy Distillation

*Shrink a teacher into a student by correcting the student where it actually goes — reverse KL on the student's own samples, across mismatched chat templates.*

---

## Offline vs on-policy

The usual way to distill is offline: the teacher generates a dataset, the student does SFT on it. That trains the student on the **teacher's** trajectory distribution — but at inference the student walks its own path, and every early deviation lands it in states the training data never covered.

On-policy distillation (OPD) flips the sampling:

```
┌────────────┐   sample    ┌──────────────┐   score    ┌────────────┐
│   PROMPT   │────────────▶│   STUDENT    │───────────▶│  TEACHER   │
│   pool     │  (its own   │  completions │  (same     │  log-probs │
└────────────┘   template) └──────────────┘   text)    └─────┬──────┘
                                 ▲                           │
                                 └────── reverse KL ─────────┘
```

Every step the student samples with its *current* weights and the loss is the full-distribution reverse KL to the teacher over exactly those tokens. One gradient step per batch, no importance sampling, no train/inference mismatch.

## Quickstart

```bash
pgs distill --config configs/distill_opd.yaml
```

Any config field is overridable from the CLI, same as `pgs train`:

```bash
pgs distill --config configs/distill_opd.yaml \
    --train.learning_rate 5e-6 \
    --sampling.cot_fraction 0.3
```

## Mismatched chat templates

OPD works across a student/teacher pair with **different chat templates** — e.g. a ChatML student distilled from a Llama-3-template teacher — as long as they share a base vocabulary. Prompts are rendered per-model with each model's own template; only completion tokens are aligned. The student's end-of-turn token is mapped onto the teacher's, so the teacher also supervises *when to stop*:

```yaml
bridge:
  eos_map:
    "<|im_end|>": "<|eot_id|>"    # student terminator -> teacher terminator
  extra_stop_tokens: ["<|end_of_text|>"]
```

A compatibility check runs before any weights load and raises if the two tokenizers diverge on probe texts — a near-miss vocabulary silently turns the KL into noise, so this is a hard error, not a warning.

!!! warning "Dedup your pool against the target benchmark"
    Training pools are often drawn from the same corpora a benchmark was curated
    from. `palingenesis.opd.pool` hashes every question (normalized: lowercased,
    accent-stripped, alphanumeric-only) so you can reject anything that appears
    in the benchmark before it enters the pool.

## Don't distill the teacher's mistakes

The teacher's accuracy is a hard ceiling for pure KL — and half of a mediocre teacher's supervision actively pulls the student toward wrong answers. When your pool has verifiable answers, score it with the teacher first and filter:

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
```

Every row comes back annotated with `teacher_answer` and `teacher_correct` — one batched forward per row (the answer is read from the option-letter logits, no generation, no parsing). What you do with the annotations — drop wrong rows, downweight them, rebalance — is your call, in the same score-then-select spirit as `pgs prepare`.

!!! note "Current scope"
    The OPD engine (token bridge, on-policy sampling, reverse-KL loss) is
    task-agnostic; the shipped data layer targets multiple-choice QA pools.
    Generic prompt sources (`messages` JSONL, pluggable dev metric) are the
    planned next step.

## What to watch

| Metric | Healthy |
|--------|---------|
| `kl/tok` | falls steadily |
| `residual_mass` | stays ≈ 0 (student mass on tokens the teacher can't see) |
| `fmt_ok` | rises toward 1 (completions contain a valid option letter) |
| `dev_acc` | the number that matters — checkpoint selection uses this |

Accuracy gains typically saturate **before** the KL stops falling: the KL keeps improving while dev accuracy plateaus once the transferable knowledge is exhausted. Keep `train.save_steps` small and pick the best checkpoint by dev accuracy.

## Memory

Student (fp32 + bf16 autocast) and frozen bf16 teacher share one GPU by default. If they don't fit: lower `train.score_micro_seqs` (gradient accumulation keeps the math identical), enable `model.gradient_checkpointing`, or move the teacher with `model.teacher_device: "cuda:1"`. With long CoT completions, also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Full reference: [`docs/on_policy_distillation.md`](https://github.com/mii-llm/palingenesis/blob/main/docs/on_policy_distillation.md) in the repository.
