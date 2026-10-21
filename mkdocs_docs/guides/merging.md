# Model Merging

`palingenesis.merge` combines checkpoints that share an architecture, in weight space, without training. Use it for:
- **partial interpolation back:** pull an SFT, RL or distillation endpoint that overshot back toward its start;
- **adding an expert:** add what an expert learned on top of a shared base to another model (task arithmetic);
- **soups:** average same-origin checkpoints;
- **materializing:** make a partial training checkpoint loadable.

```bash
pgs merge interpolate runs/start/final runs/rl/final --lam 0.3 -o runs/rl_interp_0.3
pgs merge task-arithmetic --base Qwen/Qwen3.5-0.8B runs/agent/final:1 runs/expert/final:0.3 -o runs/agent_plus_expert
pgs merge soup runs/a/final runs/b/final runs/c/final -o runs/soup
pgs merge linear runs/a/final:0.7 runs/b/final:0.3 -o runs/mix
pgs merge materialize runs/sft/step-200/model --template runs/sft/final -o runs/sft_step200
```

```python
from palingenesis import merge

merge.interpolate("runs/start/final", "runs/rl/final", 0.3, "runs/rl_interp_0.3")
merge.task_arithmetic("Qwen/Qwen3.5-0.8B", ["runs/agent/final", "runs/expert/final"], [1.0, 0.3], "runs/agent_plus_expert")
report = merge.soup(["runs/a/final", "runs/b/final"], "runs/soup")
print(report)            # soup: 488 tensors -> runs/soup
```

## Operations

| Operation | Formula | Typical use |
|---|---|---|
| `interpolate(start, end, lam)` | `W = start + lam * (end - start)` | keep part of a fine-tuning's gain and drop its side effects (lam .3–.5); lam outside [0, 1] extrapolates |
| `linear(models, weights)` | `W = sum_i a_i * W_i`, `sum_i a_i = 1` | weighted mixes of siblings (`unnormalized=True` allows other sums) |
| `soup(models)` | `W = mean_i W_i` | averaging same-origin runs (seeds, sibling stages) |
| `task_arithmetic(base, models, weights)` | `W = base + sum_i l_i * (W_i - base)` | add each model's change from a shared base; with weights `[1, lam]`, model A keeps all of its own training and gains lam of B's |
| `model_stock(base, models)` | `W = base + t * (mean_i W_i - base)`, per tensor `t = k c / (1 + (k - 1) c)`, `c` = mean pairwise cosine of the task vectors `W_i - base` (clamped to [0, 1]) | several fine-tunes of one base: keeps what they agree on and shrinks toward the base where they disagree ([Model Stock](https://arxiv.org/abs/2403.19522)); `filter_wise=True` gives one `t` per output row |
| `ram(base, models, eps)` | per element: a change (`|W_i - base| > eps`) made by one model is kept whole; changes made by several are averaged | combining RL-trained experts, whose task vectors are sparse and nearly orthogonal ([RAM](https://arxiv.org/abs/2601.13572)): averaging would dilute each one's unique changes by `1/k` |
| `materialize(checkpoint, template)` | the checkpoint's tensors + the template's other tensors and files | intermediate training saves that hold only the trained (text) weights and a training config |

Why task arithmetic for adding an expert rather than a linear merge: `a * A + (1 - a) * B` also pulls A back toward the base
by `1 - a`, undoing part of A's training. `A + lam * (B - base)` adds only what B learned.

### Which method for which situation

Measure the task vectors before choosing. Two numbers decide: how many weights each model changed, and how aligned the
changes are. On our 0.8B RL runs from one base (LR 1e-6, 100–250 steps):

- **Sparsity.** 19–37% of weights changed by more than 1e-5, so the task vectors are sparse.
- **Separate experts are nearly orthogonal.** Pairwise cosine is .01–.13. Where two experts both changed a weight, their
  signs agree only 51–60% of the time.
- **Checkpoints of one run share a path.** Step 100 against step 250 has cosine .66 and 98% sign agreement.

| Situation | Suited | Not suited, and why |
|---|---|---|
| A run overshot (general skills drifted) | `interpolate` toward the start (WiSE-FT); `soup` or `model_stock` of its checkpoints (Model Stock reduces to a mild shrink of the checkpoint average, `t` ≈ .9) | `ram` and TIES: checkpoints share signs, so there is nothing to disentangle |
| Combining experts trained separately from one base | `task_arithmetic` (keeps each whole; interference is low when the vectors are orthogonal); `ram` (also averages the overlapping elements) | `linear`, `soup`, `model_stock`: they divide each expert by about `k` (Model Stock gave `t` = .20 for three experts). TIES sign election at ~50% agreement is a coin flip. DARE's random drop + rescale adds noise to already sparse vectors |

A merge cannot remove a regression that an expert already has: check each expert alone before merging.

## Guarantees

- **Tensors are matched by name**, so checkpoints sharded differently merge correctly. Every shape must match.
- **Math runs in float32**, one tensor at a time, so memory stays at one output shard plus one tensor per source. The result
  is saved in the template's dtype, or `dtype=` / `--dtype`.
- **The output mirrors the template:** its tensor set, its shard layout and index, and its non-weight files (config,
  tokenizer, chat template, generation and processor configs). Training-only files are not copied. The template defaults to
  the start (`interpolate`) or the first model; set it with `template=` / `--template`.
- **Missing tensors:** a tensor that some source lacks is an error by default. `missing="template"` copies the template's
  tensor unchanged, and the report counts how many were copied.
- **Extra tensors:** tensors present in a source but not in the template are an error by default; `extra="drop"` leaves
  them out.
- **Integer buffers** are copied, and must be equal across sources. Non-finite merged values are an error.
- **Provenance:** `merge_config.json` in the output records the method, sources, coefficients, base, template, dtype,
  rounding, seed, edit survival and counts.
- **Sources** can be local directories or Hugging Face model ids.

`pgs merge` refuses to write into a non-empty directory unless you pass `--overwrite`.

## Precision of small edits

Public checkpoints are stored in bf16, so their weights already sit on the bf16 grid. bf16 keeps 8 significant bits:
one ulp is 2^-7 to 2^-8 of a weight's magnitude. An edit far smaller than that, such as a merge with a small `lam`
between close siblings or a margin-surgery step (‖Δ‖ around 1e-3 to 1e-5 of ‖W‖), is computed correctly in float32. Saving it
with round-to-nearest then sends almost every element back to its original value, so the edit vanishes: in simulation
1–6% of it survives.

**Stochastic rounding** (SR) rounds each element up with probability equal to the fraction of an ulp it has moved, so the
saved value is unbiased: `E[SR(w + d)] = w + d`. A few elements move by a whole ulp and the rest stay. Summed over
millions of elements, the saved change points along the intended edit and has its size (projection ≈ 1). It is
noisier element by element (`norm_ratio` > 1). Edits computed from fp32 weights that are off the grid survive
round-to-nearest only on average. With bf16 sources nothing dithers them.

```bash
pgs merge interpolate runs/a/final runs/b/final --lam 0.05 --dtype bf16 --rounding stochastic --seed 0 -o runs/ab_0.05
```

```python
from palingenesis.checkpoint import save_hf_model
save_hf_model(model, tok, out, source_layout=True, dtype=torch.bfloat16, rounding="stochastic", seed=0)
```

- `--rounding {nearest,stochastic}` (default `nearest`) and `--seed` (default 0) exist on every `pgs merge` command, and
  `rounding=` / `seed=` on every merge function. Each tensor's noise is seeded from `(seed, name)`, so the output is
  reproducible and does not depend on tensor order or sharding. SR applies when the output dtype is bf16 or fp16.
  Casts to fp32 are exact anyway.
- **Measured survival:** every merge records `edit_survival` against the first model in `merge_config.json`:
  `projection` = ⟨saved − W₀, Δ⟩/‖Δ‖², `norm_ratio` and `changed_frac`. It is computed per tensor while merging, which
  adds one float64 pass per tensor. With nearest rounding to a 16-bit dtype, a projection below 0.5 logs a warning. Use
  `--rounding stochastic` then.
- `save_hf_model(..., dtype=, rounding=, seed=)` casts the saved state the same way (`palingenesis.precision.cast_state_dict`).
  Tied weights stay tied, and `config.json` records the saved dtype. In the source layout (Qwen3.5 multimodal exports),
  only the trained weights are cast. Tensors copied from the source checkpoint (the vision tower) keep their stored
  dtype.
- **Serving formats:** `palingenesis.precision.simulate_cast(W, fmt)` rounds to `bf16`, `fp16`, `fp8_e4m3` or
  `fp8_e5m2`, using a per-tensor amax scale for fp8 as common fp8 serving does. Check whether an edit survives serving with
  `q = simulate_cast(W, fmt); edit_survival(q, q + delta, simulate_cast(W + delta, fmt))`. fp8 steps are 16–32× coarser
  than bf16. A sub-ulp edit survives fp8 serving of off-grid (fp32 or bf16) weights only on average, through the weights'
  own offsets on the fp8 grid, so check large enough tensors. GPTQ / AWQ int4 are not simulated: their rounding depends on
  calibration data and error feedback.

## Choosing lam and adopting a merge

- **Gate every candidate:** evaluate a few lam values (e.g. .3 and .5) on the same battery as the endpoints, and adopt the
  largest lam that passes every gate against the start and keeps the gain you trained for. Merges cost seconds; evaluations
  are the cost.
- **Interpolating back beats stopping early.** In our small-model runs, interpolations of a finished SFT kept most of its
  task gain and recovered its instruction-following losses. Intermediate checkpoints were worse: early checkpoints loop most.
- **Build experts from the base** and add them with task arithmetic. Continue training from the merge, with the KL reference
  set to the merge, so later stages keep what it added.
- **What did not transfer at 0.8B:** deltas from continued pretraining alone, and "chat vector" transfers that work at 70B,
  collapsed instruction following. Re-check such recipes at your scale before relying on them.
- Merge only checkpoints of the same architecture and the same origin (shared base or shared initialization).
