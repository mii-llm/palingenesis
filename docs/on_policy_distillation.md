# On-Policy Distillation (OPD)

Distill a large teacher into a small student by scoring the student's **own samples** — not a fixed dataset of teacher outputs.

```bash
pgs distill --config configs/distill_opd.yaml
pgs distill --config configs/distill_opd.yaml --train.learning_rate 5e-6 --train.steps 3000
```

Package: `palingenesis.opd` (`config.py`, `trainer.py`, `token_bridge.py`, `formatting.py`, `pool.py`).

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

Each training prompt is rendered with a randomized shot regime (`formatting.PromptRenderer`):

| Regime | Probability | Purpose |
|--------|-------------|---------|
| reference shots | `data.p_reference_shots` | exactly what the benchmark harness sends |
| 1–k pool shots | `data.p_pool_shots` | format generalization |
| zero-shot | remainder | robustness |

The default templates are byte-identical to the ITALIC benchmark's `run_eval.py` (fast mode: answer with a bare letter; CoT mode: reason, then `Risposta: LETTERA`). Both are overridable per call for other benchmarks.

## Memory

Student (fp32 + bf16 autocast, gradients) and teacher (bf16, frozen) share one GPU by default. Logits are never materialized for the full sequence — hidden states are gathered at completion positions first, keeping the big tensor at `N_completion_tokens × vocab`. If you still OOM (typically with `cot_fraction > 0`, where completions are hundreds of tokens):

| Knob | Effect |
|------|--------|
| `train.score_micro_seqs: 8` | fewer sequences per scoring forward (grad-accum keeps the math identical) |
| `model.gradient_checkpointing: true` | recompute student activations |
| `model.teacher_device: "cuda:1"` | move the teacher off the student's GPU |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fragmentation headroom for mixed generate/score phases |

## Monitoring

Every `logging.log_every` steps: `kl/tok` (should fall), `sampled_kl`, `residual_mass` (student probability on unmapped student-only tokens — should stay ≈0; if it grows, the student is drifting toward tokens the teacher can't see), `len` (mean completion length), `fmt_ok` (fraction of completions containing a valid option letter), and greedy dev accuracy every `train.eval_every` steps. `logging.use_wandb: true` mirrors everything to wandb; tracker failures degrade to console logging and never kill the run.

Gains typically saturate well before the KL stops falling — checkpoint often (`train.save_steps`) and pick by dev accuracy, not by KL.
