# On-Policy Distillation

*Shrink a teacher into a student by correcting the student where it actually goes: the student samples, the teacher scores those exact tokens, and the student's distribution is pulled toward the teacher's. Works across tokenizers and with several teachers at once.*

---

## Offline vs on-policy

The usual way to distill is offline: the teacher generates a dataset, the student does SFT on it. That trains the student on the **teacher's** trajectories, but at inference the student walks its own path, and every early deviation lands it in states the training data never covered.

On-policy distillation (OPD) flips the sampling. Every step the student samples completions with its current weights, the teacher scores exactly those tokens, and the loss is the reverse KL from the student to the teacher on the student's own states:

```
 sources ──prompts──▶ rollout engine ──completions──▶ aligner ──teacher view──▶ teacher
 (per-source          (student's current weights:      (same vocabulary: ids;    (log-probs of the
  teacher)             HF generate or vLLM)             else byte chunks)          student's text)
                                      ▲                                                  │
                                      └──────── new weights ◀── loss (reverse KL) ◀──────┘
```

## Quickstart

Qwen3-1.7B into Qwen3-0.6B on GSM8K, non-thinking, on one A100 80GB:

```bash
uv sync --extra train --extra logging --extra vllm    # vLLM rollouts (Linux, driver >= 575)
```

```python
# data/gsm8k_*.jsonl: one chat prompt per line, with the reference answer for dev accuracy
import itertools, json
from datasets import load_dataset

PROMPT = "{q}\nSolve the problem step by step, then give the final answer as a number on the last line, after 'Answer:'."
def rows(split, n):
    for r in itertools.islice(load_dataset("openai/gsm8k", "main", split=split, streaming=True), n):
        yield {"messages": [{"role": "user", "content": PROMPT.format(q=r["question"])}],
               "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
for name, split, n in [("gsm8k_train", "train", 3000), ("gsm8k_test200", "test", 200)]:
    with open(f"data/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows(split, n))
```

```bash
pgs distill --config configs/distill_math.yaml
pgs distill --config configs/distill_math.yaml --train.learning_rate 1e-6 --logging.use_wandb true
```

Every option can be overridden on the command line, including those of a named teacher or source: `--teachers.qwen3_1_7b.backend vllm`, `--sources.gsm8k.max_new_tokens 384`.

## A step

1. The orchestrator draws `rollout.batch_prompts` prompts from the sources (by `weight`) and renders each with the student's chat template and with its teacher's.
2. The rollout engine samples `rollout.group_size` completions per prompt at `rollout.temperature`, and records each sampled token's log-probability under the policy that sampled it.
3. Each completion is cut at the first stop token (kept: stopping is supervised too), aligned with its teacher's view of the same text, and scored by that teacher.
4. The student scores its own completions (the only forward with gradient); each teacher's loss is summed over its samples and normalized by the batch's completion tokens. Logits are projected a slice of rows at a time, so `[tokens, vocabulary]` is never materialized.
5. Clipped AdamW step on fp32 master weights (bf16 autocast); the new weights go to the rollout engine.

## Rollout backends

| `rollout.backend` | What samples | Notes |
|---|---|---|
| `hf` (default) | the trainer's own `model.generate` | No extra dependency. Slow: padded batches, no paged KV cache. |
| `vllm` | an in-process vLLM engine on the student's GPU | Sleeps while the trainer trains (its weights and KV cache are released), wakes for the next batch; new weights are loaded into it in place after every step. Needs the `vllm` extra. |
| `vllm_server` | a separate `vllm serve` process on the same GPU | **Experimental.** Weights go over CUDA IPC through vLLM's native weight-transfer endpoints; generation runs concurrently with training (`rollout.max_staleness: 1`). Also needs `ray` installed (vLLM 0.26's IPC module imports it). |

`rollout.gpu_memory_utilization` is the vLLM engine's share of the GPU (weights and KV cache); with `vllm` it only holds that memory while it generates.

### Staleness and asynchrony

`rollout.max_staleness` is how many optimizer steps a batch may lag behind the weights it trains:

- `0` (default): batch k is generated with the weights of step k, strictly on-policy. The orchestrator thread still samples and renders the next prompts while the trainer trains.
- `1`: batch k+1 is generated (and scored by the teacher) while the trainer trains on batch k. The divergence losses need no correction for this: the student's distribution is computed with the current weights, only the visited states come from a one-step-old policy. The policy-gradient losses (`sampled_rkl`, `xtok`) weight each token by the importance ratio to the rollout policy's log-probabilities, and zero it outside `[loss.is_low, loss.is_high]`.

With `vllm` and `max_staleness: 1` the engine stays awake (its memory is held during training) and its kernels share the trainer's CUDA stream, so what overlaps is mostly host work: vLLM's scheduling and the trainer's Python. On a second stream the in-process engine raced with the trainer's kernels (illegal memory accesses that disappear with `CUDA_LAUNCH_BLOCKING=1`), so it is kept on one. With `vllm_server` the rollouts run in another process, and the teacher scoring on a stream of its own.

A batch older than `max_staleness` when it reaches the trainer is dropped and counted (`dropped_samples`); the orchestrator only starts a batch once the weights it may use are published, so this does not happen in normal operation. `staleness` is logged every step.

## Teachers and losses

A teacher is `hf` (in-process, frozen bf16, full distribution) or `vllm` (a vLLM server run prefill-only: `prompt_logprobs=k` returns the teacher's top-k log-probabilities and the actual token's at every completion position). A vLLM teacher is a separate server, launched on the student's GPU with `teachers.<name>.gpu_memory_utilization` or reached at `url`, because vLLM allows one sleep-mode engine per process and that one is the rollout engine. Each teacher has one loss, picked automatically unless `teachers.<name>.loss` is set:

| Loss | Needs | What it minimizes | Default for |
|---|---|---|---|
| `full_rkl` | hf teacher, shared vocabulary | exact reverse KL over the shared vocabulary at every completion token | hf teacher, shared vocabulary |
| `rs_kd` | hf teacher, shared vocabulary | forward KL to tokens drawn from the teacher's distribution, importance-weighted (Random Sampling KD, arXiv 2503.16870): unbiased, `loss.rs_rounds` ids per token | — |
| `topk_kl` | shared vocabulary | KL between coarse distributions: the teacher's top-k plus the realized token, and one tail bucket holding the rest of each side's mass (`loss.beta`: 1 reverse, 0 forward) | vllm teacher, shared vocabulary |
| `sampled_rkl` | shared vocabulary | REINFORCE on the per-token reward log p_T(y) − log p_S(y) (the sampled reverse KL), with an importance ratio to the rollout policy | — |
| `xtok` | any tokenizers | REINFORCE on text chunks: reward log p_T(chunk) − log p_S(chunk); optional top-k KL where a chunk is one token on each side | teacher with another tokenizer |

Asking for a shared-vocabulary loss with a teacher whose tokenizer differs is an error at startup, before any weights load.

A **shared vocabulary** means the teacher's vocabulary is a prefix of the student's and both tokenize a set of probe texts identically (Qwen3-0.6B and Qwen3-1.7B; or a ChatML student extending Llama 3's vocabulary with `<|im_end|>`, distilled from a Llama-3-template teacher). Prompts are rendered with each model's own template, and the student's end-of-turn token is scored against the teacher's (`teachers.<name>.eos_map`, automatic in the common cases), so the teacher also supervises when to stop.

### Across tokenizers

When the tokenizers differ, the completion's text is re-tokenized by the teacher and both token sequences are cut into **chunks** at the byte offsets where both end a token:

```
student (Qwen3)     |Un|belie|vably|,| |1|2|5|0| ducks|.|
teacher (SmolLM2)   |Un|belie|v|ably|,| |1|2|5|0| ducks|.|
chunks              [Un][belie][vably][,][ ][1][2][5][0][ ducks][.]      "vably" = "v" + "ably": one chunk
```

A chunk is the same text on both sides, so its total log-probability is comparable: the reward of chunk c is A_c = log p_T(c) − log p_S(c) (the sum of each side's token log-probabilities in the chunk), and the rewards of a completion add up to its log-likelihood ratio between the two models. Details that matter:

- The student's byte offsets come from the **sampled token ids** (each token's bytes), never from re-encoding the decoded text: a sampled sequence is often not the tokenizer's canonical encoding of its own text.
- The student's end-of-turn token pairs with the teacher's as the last chunk.
- Whitespace-only chunks carry no loss (`loss.mask_whitespace`): tokenizers disagree most on whitespace runs, and supervising them hurt code models badly in published work.
- Bytes after the last complete UTF-8 character (a completion cut inside an emoji) and special tokens inside a completion are left out.
- `loss.xtok_spread: chunk` gives every token of a chunk the chunk's advantage; `proportional` gives each token the share log p_S(t) / log p_S(c) of it.
- `loss.xtok_dense_weight > 0` adds a top-k KL at chunks of exactly one token on each side, over the teacher's top-k tokens mapped to student tokens that spell the same bytes (a tail bucket holds the rest).

### Several teachers

Each source names its teacher; every step mixes the sources by weight and each sample is scored by its own teacher with its own loss (one teacher per sample, no ensemble averaging):

```yaml
teachers:
  math: {model: Qwen/Qwen3-1.7B}                       # full_rkl
  chat: {model: HuggingFaceTB/SmolLM2-360M-Instruct}   # another tokenizer: xtok
sources:
  gsm8k: {path: data/gsm8k_train.jsonl, teacher: math, weight: 0.5}
  chat:  {path: data/chat_train.jsonl,  teacher: chat, weight: 0.5}
```

Metrics are reported per teacher (`kl/<teacher>`, `k1/<teacher>`, `tokens/<teacher>`) and dev metrics per source (`eval/dev_kl/<source>`). An hf teacher that does not fit next to the others can wait on CPU between scoring calls (`offload: true`) or live on another GPU (`device: cuda:1`).

## Measured

Student Qwen3-0.6B, non-thinking, on GSM8K train prompts; one A100 80GB (vLLM 0.26, torch 2.11, cu129). Every run: 64 prompts per step, 512 new tokens, temperature 1, 60 steps, learning rate 3e-6 (5 warmup steps, cosine), `configs/distill_math.yaml` with the changes listed. Dev metrics on 200 GSM8K test questions: `dev_kl` is the on-policy sampled reverse-KL estimate per student token (on the teacher's own tokens for xtok, whitespace chunks excluded), accuracy is greedy with the last number after "Answer:". Greedy accuracy of the models alone on the same questions: Qwen3-0.6B 64.5%, Qwen3-1.7B 79.0%, Qwen3.5-0.8B 52.0%, SmolLM2-360M-Instruct 10.0%.

| Run | Rollout | Teacher → loss | Rollout tok/s (median) | s / step (median) | dev_kl, step 0 → 60 | GSM8K %, step 0 / 20 / 40 / 60 |
|---|---|---|---:|---:|---|---|
| reference | `hf` | Qwen3-1.7B hf → `full_rkl` | 638 | 27.9 | 0.542 → 0.256 | 61.5 / 62.5 / 64.0 / 61.5 |
| default | `vllm` | Qwen3-1.7B hf → `full_rkl` | 7,503 | 5.6 | 0.538 → 0.248 | 63.0 / 64.0 / 61.0 / 59.5 |
| vLLM teacher | `vllm` | Qwen3-1.7B vllm (top-8) → `topk_kl` | 7,898 | 6.9 | 0.538 → 0.253 | 63.0 / 66.0 / 67.0 / 64.0 |
| sampled | `vllm` | Qwen3-1.7B hf → `sampled_rkl` | 7,529 | 5.4 | 0.538 → 0.282 | 63.0 / 67.5 / 62.5 / 59.5 |
| cross-tokenizer | `vllm` | Qwen3.5-0.8B hf → `xtok` | 8,557 | 5.7 | 0.446 → 0.204 | 63.0 / 58.0 / 48.5 / 53.5 |
| xtok, same vocabulary | `vllm` | Qwen3-1.7B hf → `xtok` | 7,448 | 5.4 | 0.509 → 0.242 | 63.0 / 67.5 / 64.0 / 64.5 |
| two teachers | `vllm` | GSM8K: Qwen3-1.7B `full_rkl`; chat: SmolLM2-360M `xtok` | 8,420 | 5.9 | GSM8K 0.538 → 0.277, chat 0.779 → 0.423 | 63.0 / 64.5 / 70.0 / 64.0 |
| async, in-process | `vllm`, `max_staleness: 1` | Qwen3-1.7B hf → `full_rkl` | 4,550 (while training) | 4.4 | 0.538 → 0.253 | 62.5 / 62.0 / 63.0 / 62.5 |
| async server | `vllm_server`, `max_staleness: 1` | Qwen3-1.7B hf → `full_rkl` | 4,345 (while training) | 4.7 | 0.524 → 0.260 | 61.0 / 65.5 / 66.0 / 61.5 |

What the numbers say:

- **Rollouts dominate with `hf`.** The same run takes 27.9 s per step with the trainer's `generate` and 5.6 s with the in-process vLLM engine (rollouts ~12× faster, steps ~5×); the KL curves match. In the vLLM step (medians), rollout is 2.15 s, the hf teacher 0.75 s, training 1.88 s, the weight update 0.14 s and sleep/wake 0.67 s; a vLLM teacher takes 1.9 s (prefill with top-k log-probs, returned as JSON).
- **Every loss reduces the on-policy KL by about half in 60 steps.** `full_rkl` and `topk_kl` (8 teacher tokens plus a tail bucket) track each other; the sampled estimators (`sampled_rkl`, `xtok`) are noisier.
- **Accuracy follows the teacher, within noise.** With 200 questions one standard error is ~3.4 points: runs toward Qwen3-1.7B peak 3–7 points above the start around steps 20–40, and end within noise of it. A longer run at learning rate 1e-6 (150 steps) has the same shape: 63.0 → 67.0 at step 50 → 60.5 at step 150, while its dev KL plateaus at 0.26 from step 50 on. Watch `eval/dev_acc` and keep the best checkpoint. Toward Qwen3.5-0.8B, which is worse than the student at GSM8K, the student drifts toward the teacher's 52%, as it should: distillation transfers the teacher's behaviour, not accuracy.
- **Across tokenizers, xtok behaves like the same-vocabulary losses.** Forced on the same-vocabulary teacher it matches `sampled_rkl`; aligning a batch of 64 completions costs ~0.05 s.
- **Asynchrony saves 15–20% on one GPU.** Generating the next batch while the trainer trains, 60 steps take 280 s (in-process engine) or 300 s (server) instead of 354 s; the two contend for the GPU (training 1.9 → 2.7–3.0 s, rollout 2.2 → 3.6–3.7 s), and the in-process engine also skips its per-step sleep/wake. Every batch was exactly one version old; none was dropped. The dev KL curves match the synchronous run's.
- **Weight sync is exact.** After an update, the rollout engine's log-probabilities of fixed probe sequences match the trainer's within bf16 noise (mean |Δ| 0.049 in-process, 0.056 over CUDA IPC, against 0.041 before any change and 1.56 after perturbing the trainer's weights without an update); Qwen3.5 (checkpoint names differ from the loaded model's) matches too (0.022).

### Qwen3.5-4B → Qwen3.5-0.8B, and where a step's time goes

`configs/distill_qwen35.yaml`: `full_rkl` from an hf teacher, 64 prompts × 512 new tokens per step, on-policy (`max_staleness: 0`). In 100 steps (11 minutes, five evaluations included) GSM8K accuracy goes 53.0 → 55.5 → 59.5 → 62.0 → 62.5% (steps 0/25/50/75/100; the teacher scores 87.5%) and dev KL 0.344 → 0.245. A step takes 5.5 s, down from 8.4 s before these changes (the same batch; timings are means over steps 3–8):

| Phase (s) | before | after | what changed |
|---|---:|---:|---|
| rollout (vLLM, 64 × ~375 tokens) | 2.58 | 2.46 | — (decode-bound: see below) |
| vLLM sleep/wake | 1.07 | 0 | `rollout.sleep: false`, `gpu_memory_utilization: 0.1`: the engine stays resident |
| weight sync | 0.23 | 0.03 | no weights to re-map after a sleep |
| teacher scoring | 1.92 | 1.46 | no output-head pass the loss does not use; Liger kernels |
| student forward/backward + optimizer | 2.57 | 1.67 | fused `full_rkl` (below); Liger kernels |

The fused `full_rkl` (`opd/fused_rkl.py`, used automatically for plain linear output heads and a shared vocabulary without end-of-turn remapping) computes both models' logits in fp32 straight from bf16 GEMMs, then makes two streaming passes over them in Triton: one for both log-sum-exps, the KL and its statistics, one for the analytic gradient q·(log q − log p + 1 − KL − S). Autograd through the definition, which the generic path uses, takes about a dozen elementwise passes over the [tokens, 248k] logits and recasts the output head's weight for every slice. The fused path is also more exact: against fp64 on the same inputs its loss is within 4e-6 and its gradients within 1.3% (norm), where the autocast path's are 9e-6 and 2.9%.

The fused path covers every head `logits.output_head` produces: a logit scale (Cohere, Granite, Falcon-H1) and Gemma 2's final-logit softcap are applied inside the kernels, with the softcap's derivative computed as 4σ(2x)σ(−2x) (no cancellation where the logits saturate). On real hidden states against fp64: Gemma-2-2B-it ← Gemma-2-2B (softcap 30, 256k vocabulary) loss error 2.6e-4 vs 4.8e-3 for the generic path, gradients 5–12× closer, 79 vs 197 ms; Granite-3.3-2B ← 8B (logit scales 1/8 and 1/16) 3.6e-5 vs 7.5e-5, 46 vs 61 ms.

Rollout time is set by decoding: 512 steps for the longest completion, about 5 ms each at 64 sequences (Qwen3.5's linear-attention layers read and write a recurrent state per sequence at every step). Larger batches decode more tokens per second: 128 prompts per step give 13.5k rollout tokens/s instead of 9.7k, and 4.9k trained tokens/s instead of 4.3k, at half the optimizer steps per token.

### Sparse teacher targets and token selection

Same setup (Qwen3.5-4B → 0.8B, 100 steps, one seed each; one standard error on 200 questions is ~3.4 points):

| Run | GSM8K %, step 0 / 25 / 50 / 75 / 100 | dev_kl, step 100 | s / step |
|---|---|---:|---:|
| `full_rkl` (baseline) | 53.0 / 55.5 / 59.5 / 62.0 / 62.5 | 0.243 | 5.5 |
| `token_weighting: sure`, α = 1 | 53.0 / 55.0 / 62.5 / 61.0 / 62.0 | 0.245 | 5.5 |
| `token_weighting: entropy`, keep 20% | 53.0 / 62.5 / 63.0 / 59.0 / 64.0 | 0.249 | 5.7 |
| `rs_kd`, 50 draws | 53.0 / 53.5 / 59.0 / 58.5 / 56.5 | 0.284 | 7.0 |

- **Token selection matches the baseline, within noise.** SuRe (arXiv 2608.25643) up-weights tokens the student found unlikely; `entropy` trains on only the 20% of tokens where the student is most uncertain (the forking tokens of arXiv 2506.01939, an RL result applied here to distillation). Training on a fifth of the tokens loses nothing measurable here, and neither beats the baseline by more than the noise.
- **Random Sampling KD is unbiased but noisier than top-k on-policy.** On a real batch, against the exact full forward-KL gradient: rs_kd with 50 draws is 13° off (25° with 12), top-12 with a tail bucket 2.6°, top-50 0.6°, and even renormalized top-12 (the paper's Top-K baseline) 4.4°. The paper (arXiv 2503.16870) reports the opposite ordering (top-12 58°, its method 4°) on pretraining text, where the teacher's distribution is broad; at the states an on-policy student visits the teacher is confident (4.3 distinct ids in 50 draws), so a top-k captures nearly all the mass and sampling only adds variance. rs_kd optimizes forward KL, which is mode-covering: completions grew to ~455 tokens (vs ~380) and accuracy trailed. It is also slower here (the teacher projects and samples the full vocabulary per token). Use it where sparse targets are the point — caching a teacher's targets offline, a teacher on another device — not in place of `full_rkl` on one GPU.

Also run, 5 steps each: an hf teacher with `offload: true` (moving Qwen3-1.7B on and off the GPU adds 1.4 s per step) and a vLLM teacher (Qwen3.5-0.8B) for `xtok` with the dense term. Runs are in the `palingenesis-validation` wandb project, named `opd2-*`.

## Multiple-choice pools

`format: mcqa` sources train on multiple-choice pools in a benchmark's exact prompt format (`configs/distill_opd.yaml` carries ITALIC's verbatim templates), with the reference shots, random pool shots and zero-shot mixed per prompt, and greedy letter accuracy as the dev metric.

!!! warning "Dedup your pool against the target benchmark"
    Training pools are often drawn from the same corpora a benchmark was curated
    from. `palingenesis.opd.pool` hashes every question (normalized: lowercased,
    accent-stripped, alphanumeric-only) so you can reject anything that appears
    in the benchmark before it enters the pool.

The teacher's accuracy is a hard ceiling for pure KL, and half of a mediocre teacher's supervision pulls the student toward wrong answers. Score the pool with the teacher first and filter:

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
```

Every row comes back annotated with `teacher_answer` and `teacher_correct` (one batched forward per row, the answer read from the option-letter logits).

## Custom sources

A source is any object with `sample()` (messages, max_new_tokens, meta), `evaluate(engine)` and `batch_stats(rollouts)`; pass it as `OPDTrainer(config, source=...)`. The engine a source evaluates with offers `greedy_generate(messages_list, max_new_tokens)` and `dev_kl(messages_list, max_new_tokens)`. The built-in sources in `palingenesis.opd.sources` are short templates.

## What to watch

| Metric | Meaning |
|--------|---------|
| `k1/<teacher>` | sampled reverse-KL estimate per student token, sum(log p_S − log p_T) / tokens: the same scale for every loss |
| `kl/<teacher>` | the loss's own KL (exact for `full_rkl`, coarse for `topk_kl`) |
| `eval/dev_kl/<source>` | `k1` on held-out prompts, sampled on-policy: the quantity being trained, out of sample |
| `eval/dev_kl_full/<source>` | exact per-token reverse KL (full_rkl teachers) |
| `eval/dev_acc/<source>` | greedy accuracy (messages rows with an `answer`, mcqa letters) |
| `residual_mass/<teacher>` | student mass on tokens the teacher cannot see; stays ≈ 0 |
| `is_dropped/<teacher>`, `abs_log_ratio/<teacher>` | policy-gradient losses: tokens outside the importance-ratio range, and the mean trainer/rollout log-prob gap |
| `dense_kl/<teacher>`, `dense_fraction/<teacher>` | xtok dense term: its KL per supervised position, and the share of tokens that have one |
| `rollout_tok_s`, `time/rollout`, `time/teacher`, `time/train`, `time/sync` | where a step's time goes |
| `staleness`, `dropped_samples` | how far behind the trained batch was |

Accuracy gains typically saturate before the KL stops falling. Keep `train.save_steps` small when checkpoint selection matters.

## Checkpoints and resume

Every `train.save_steps` steps the trainer writes `output_dir/step_N`: the student in Hugging Face format (loadable and servable as is) plus `trainer_state.pt` with the optimizer, the policy version and the random states. `final` is the model only. To continue an interrupted run, rerun the same command with `--train.resume_from auto` (the newest complete checkpoint in `output_dir`; a fresh start if there is none) or a checkpoint path. The student reloads from the checkpoint, the rollout engine syncs to it before its first rollout, the learning-rate schedule and wandb run continue where they were. Prompts continue from the sampler's saved state; the few drawn ahead of training when the checkpoint was written are skipped, not repeated.

## Memory

With `rollout.sleep: false` the vLLM engine keeps its `gpu_memory_utilization` share while the trainer trains, which saves the wake-up each step (about 1 s for Qwen3.5-0.8B) when it fits: models with little KV cache per token (hybrid linear-attention models, small models) need only a small share.

Student (fp32 master weights + bf16 autocast, AdamW) and frozen bf16 hf teachers share the GPU; the `vllm` rollout engine takes `rollout.gpu_memory_utilization` of it only while it generates. If it does not fit: lower `train.score_micro_seqs` (gradient accumulation keeps the math identical), enable `model.gradient_checkpointing`, move a teacher to another GPU (`device`), or offload it between uses (`offload`). vLLM's sleep mode is not compatible with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
