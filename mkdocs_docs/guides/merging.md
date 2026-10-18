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
| `materialize(checkpoint, template)` | the checkpoint's tensors + the template's other tensors and files | intermediate training saves that hold only the trained (text) weights and a training config |

Why task arithmetic for adding an expert rather than a linear merge: `a * A + (1 - a) * B` also pulls A back toward the base
by `1 - a`, undoing part of A's training. `A + lam * (B - base)` adds only what B learned.

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
- **Provenance:** `merge_config.json` in the output records the method, sources, coefficients, base, template, dtype and
  counts.
- **Sources** can be local directories or Hugging Face model ids.

`pgs merge` refuses to write into a non-empty directory unless you pass `--overwrite`.

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
