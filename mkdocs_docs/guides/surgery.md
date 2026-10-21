# Margin Surgery

`palingenesis.surgery` repairs a decision whose overall level a training stage shifted, without training. It computes a
minimal-norm weight edit that moves the decision's margins on a few groups of prompts back to target values, then
writes the edit at one or more doses. It is experimental and single-device; read [Limits](#limits) before relying on it.

```bash
pgs surgery margins --model runs/before --model runs/after --twins data/cont_dev.jsonl --decision "<|im_end|>"
pgs surgery run --config surgery.yaml
```

```python
from palingenesis import surgery

report = surgery.run_surgery(surgery.SurgeryConfig.from_yaml("surgery.yaml"))
print(report["stop_reason"], report["edit"]["rel_reference"], report["doses"][0]["edit_survival"]["projection"])
```

## What it fixes

Training stages can move a decision along one axis. A typical case: after an agentic training stage a model ends
its turn after one tool call on requests that need several parallel calls. The log-odds of `<|im_end|>` right after
`</tool_call>` rises, while the model can still tell one-call requests from multi-call requests. This is a
**common-mode** shift: the level of the decision moved, its discrimination did not.

Surgery fixes the level. You give it prompts at the decision position, grouped, and a target per group: usually the
margins of a same-lineage reference model from before the shift. It solves for the smallest edit, inside the span of
the groups' margin gradients, that closes the gaps. Such edits can be very small (around 1% of the distance between
the shifted model and its reference), because the shift itself is low-dimensional.

Matching the reference's margins on the groups fixes the **direction** of the edit, not its amount: in validation, an
edit that brought the group margins back within a nat recovered only part of the behaviour, and a larger dose was
needed. **The dose has to be chosen on a behavioural set** (see [Choosing the dose](#choosing-the-dose)).

## When to use it, and when not

Use it when:

- a stage moved the **level** of one decision and the discrimination stayed. Check with `pgs surgery margins` on twins:
  the common margin m⁺ + m⁻ moved, the differential m⁺ − m⁻ did not;
- you have a same-lineage reference (the checkpoint before the stage) or a target value;
- you have a behavioural benchmark to choose the dose on, split in two halves.

Do not use it when:

- the discrimination is what was lost (the differential moved). Surgery moves levels; it does not teach distinctions;
- the decision is taken inside a thinking block (thinking-mode twins are refused);
- the behaviour is not a single-token decision at a known position (each element of a decision set must be one token).

## Diagnose first

```bash
# mean / SE / greedy share per model; with --prefix-side both also common and differential
pgs surgery margins --model runs/ref --model runs/shifted --twins data/twins_dev.jsonl --decision "<tool_call>" --prefix-side both

# is the target behaviour inside the span of K group gradients? (high for the decision, low for a control decision)
pgs surgery span --model runs/shifted --groups data/twins.jsonl --targets data/bfcl_like.jsonl --decision "<|im_end|>" --k 1,4,16
```

`span` prints the fraction ‖P_K t‖ / ‖t‖ of the target set's margin gradient t inside the span of the first K group
gradients. A useful pattern: high coverage for prompts of the decision you edit (rising with K), low coverage for an unrelated
decision. That says the groups represent the decision, and only it.

## Configuration

```yaml
model: runs/shifted/final          # θ0, the model to edit
reference: runs/before/final       # targets "reference" and trust_ref "reference_distance" need it
output_dir: runs/shifted_surgery
families:                          # constraint families; each gives groups x group_size sides
  cont:
    path: data/cont_twins.jsonl    # twins format (see Common-Mode Rejection): pos / neg sides, tools, prefix_calls
    side: pos
    decision: "<|im_end|>"         # one token, or a list (a token set: logsumexp over the set vs the rest)
    target: reference              # reference | hold | <number>
    constraint: equal              # equal | hinge_below | hinge_above
    groups: 16
    group_size: 8
  single:                          # one-call requests, where ending the turn is right: the edit must not make
    path: data/single_twins.jsonl  # the model end LESS readily than the reference (a hinge: a row only while violated)
    side: neg
    decision: "<|im_end|>"
    target: reference
    constraint: hinge_above        # margin must stay >= the reference's
    groups: 4
    group_size: 8
eval:                              # held-out twins, never in a group: reported for reference / start / end / doses
  cont_dev: {path: data/cont_dev.jsonl, side: both, decision: "<|im_end|>"}
iterations: 6
cutoff: 1.0e-2                     # drop Gram eigenvalues < cutoff * lambda_max
trust: 0.1                         # per-step |Δ| <= trust * |θ0 - θ_ref| (trust_ref: absolute -> absolute norm)
cap: 0.25                          # cumulative |θ - θ0| <= cap * |θ0 - θ_ref|
min_gap: 0.5                       # stop when |mean gap| < max(min_gap, stop_se * SE of the mean gap)
doses: [1, 2, 3]
params: {include: [".*"], exclude: []}   # regexes on parameter names
gram: auto                         # auto | device | cpu | recompute
master: device                     # device | cpu
work_dtype: null                   # default bf16 on CUDA, fp32 on CPU
save_dtype: null                   # default: the source's stored dtype
rounding: null                     # default stochastic for 16-bit saves
serving_formats: [fp8_e4m3]        # also report edit survival under simulated serving
micro_batch: 4
gradient_checkpointing: false
max_side_tokens: 2048              # drop longer sides (reported): padded long prompts exhaust memory
chat_template_kwargs: {enable_thinking: false}
seed: 0
```

The file may also nest everything under `surgery:`. Override keys on the command line with
`--set trust=0.2 --set families.cont.groups=8 --set "params.exclude=['embed']"` (values parsed as YAML).
`validate()` refuses: a reference target or reference-relative trust without `reference`, thinking-mode rendering,
non-positive doses, unknown modes and dtypes, invalid regexes; at load time, multi-token decisions and a decision
position inside an open `<think>` block; a reference with a different vocabulary or chat template.

## Targets and constraints

Each group is one row of the system. Its margin m̄ is the mean exact log-odds of the decision over its sides; groups,
not single prompts, because one prompt's margin is noise and the solve would chase it.

| target | value per side | use |
|---|---|---|
| `reference` | the reference model's margin on the same side (paired) | restore the pre-stage level |
| `hold` | θ0's own margin | keep another decision where it is while editing |
| a number | that value | no reference available (choose the dose on behaviour) |

| constraint | in the system | |
|---|---|---|
| `equal` | always | |
| `hinge_below` | only while m̄ > target | the margin must not exceed the target |
| `hinge_above` | only while m̄ < target | the margin must not fall below the target |

In our second application, call propensity was restored with equality groups on positive twins and `hinge_below`
groups on negative twins (calls must not become more likely than the reference's where no call is right).

## The solve

With g_i = ∇θ m̄_i for the active rows, A = G Gᵀ and gap = target − m̄, the minimal-norm step that closes the linearised
gaps is Δ = Gᵀ A⁺ gap. The group gradients of one decision are often nearly collinear (cosines of .9 and above are common), so A has
a few large eigenvalues and many tiny ones that are group noise: eigenvalues below `cutoff`·λmax are dropped
(c = V_k diag(1/λ_k) V_kᵀ gap, Δ = Gᵀc). The step is scaled to the trust radius and the cumulative cap, the margins are
re-measured, and the loop stops when |mean gap| < max(`min_gap`, `stop_se`·SE of the mean gap) over the equality rows
with no hinge active, or after `iterations` steps.

## Choosing the dose

The twin target fixes the direction, not the amount. Write several doses and choose on behaviour:

1. Split your behavioural benchmark in two halves by a hash of the item id.
2. Evaluate every dose on half A with your safety gates (over-calling, other categories, unrelated decisions).
3. Take the smallest dose within 1 SE of the best passing one; report it on half B; then run held-out tests once.

Always run the null baseline: a constant logit bias on the decision token, chosen on half A with the same rule. If the
bias does as well, the weight edit adds nothing for this decision. Also check contexts where the decision must not
move (for an end-of-turn edit: turns where ending after one call is right) with `pgs surgery margins`.

## Precision

The edit is often smaller than a bf16 step (one ulp is 2⁻⁸–2⁻⁷ of |w|), so how it is written decides whether it
exists:

- **fp32 master.** The forward and backward run on a working model in `work_dtype`; the edit accumulates in an fp32
  copy of the edited parameters only (`master: device`, or `cpu` for pinned host memory).
- **Working weights = saved weights.** After every update the working weights are re-derived from the master by the
  cast the save will use: `precision.cast_tensor(master, save_dtype, rounding, seed, name)` (per-tensor seed from the
  run seed and the tensor's name; tied tensors use the smallest alias, as `cast_state_dict` does), then cast to
  `work_dtype` as a loader would. Gaps, the stop rule and every reported margin are measured on these weights, never
  on the fp32 master. Doses are saved with `save_hf_model(..., source_layout=True, dtype=, rounding=, seed=)`, so the
  checkpoint holds exactly the measured weights (a test reloads it and reproduces the margins bit for bit).
- **Stochastic rounding for bf16-native sources.** Public checkpoints sit on the bf16 grid; rounding θ0 + Δ to nearest
  returns θ0 wherever |Δ_i| < ulp/2, so a small edit vanishes. `rounding` defaults to stochastic whenever the save dtype
  is 16-bit (unbiased: the edit survives in expectation). An fp32 source saved in fp32 is exact.
- **Edit survival** is reported per dose: read the **projection** ⟨saved − θ0, Δ⟩ / ‖Δ‖² (1 = the edit survives in
  direction and size, 0 = lost). The norm ratio is misleading: rounding moves the elements that cross a grid boundary by
  whole ulps, and that noise dominates the norm (9.8 for an fp32-sourced edit cast to bf16). `serving_formats` adds the same check under simulated serving
  (`fp8_e4m3`, `fp8_e5m2`; see [Model Merging](merging.md) for the precision helpers).
- **Not measured:** calibrated int4 quantizers (GPTQ, AWQ). Edits that survive bf16 can still revert under them:
  quantize the chosen dose and re-measure its margins.

## Memory

Compute is a few dozen group forward/backward passes: minutes. Memory is the constraint. With P_work the model's
parameters, P_edit the edited ones and K the number of rows:

| `gram` | device bytes | host bytes | backward passes per iteration |
|---|---|---|---|
| `device` | 2·P_work + 4·P_edit (master) + 4·P_edit (fp32 gradient accumulator) + **2·K·P_edit** + activations | θ0 of the edited parameters (source dtype) | K + 1 |
| `cpu` | 2·P_work + 4·P_edit + 4·P_edit + 2·P_edit (one flat gradient) | **2·K·P_edit** (pinned) + θ0 | K + 1 |
| `recompute` | 2·P_work + 4·P_edit + 4·P_edit + **4·P_edit** (two bf16 gradient buffers) | θ0 | K(K+1)/2 + 1 |

- `auto` picks `device` if 2·K·P_edit fits in free device memory after a headroom (max(4 GiB, 15%) + 8·P_edit), else
  `cpu` if it fits in 70% of available host RAM, else `recompute`. The choice and the estimates are logged and saved in
  the report.
- `recompute` is exact: for each row i it computes g_i once and recomputes every g_j, j > i, into a second buffer, so
  nothing scales with K. Recomputed bf16 gradients differ by about 1e-3 relative on GPU, far below the cutoff.
- The step never materialises the gradients: Δ = Σ c_i g_i is the gradient of Σ c_i m̄_i, one weighted backward over
  all active groups' sides. Gradients accumulate in fp32 through post-accumulate-grad hooks that free each bf16
  `.grad` as soon as it is produced.
- `master: cpu` moves the 4·P_edit master to pinned host memory; fp32 working weights saved in fp32 need no master.
- `params` restricts the edit: frozen parameters get no gradient and no master, and are saved bit for bit.
- Activations: `micro_batch` sides per forward (sorted by length to minimise padding), `gradient_checkpointing: true`,
  and `max_side_tokens`.
- The reference is loaded on the CPU, ‖θ0 − θ_ref‖ is computed tensor by tensor, then it is moved to the device to
  measure the targets and freed before θ0 goes to the device: the two models are never on the device together.

At 0.8B with K = 16 the `device` mode needs about 2·0.8 + 4·0.8 + 4·0.8 + 2·16·0.8 ≈ 34 GB plus activations; a 4B model
needs `cpu` (128 GB host) or `recompute` on one 80 GB GPU.

## Outputs

- `output_dir/dose_<α>/`: θ0 + α·Δ in the source checkpoint's layout (multimodal checkpoints keep their vision tower),
  the save dtype and rounding above, plus `surgery_config.json`.
- `output_dir/surgery_report.json`:
  - precision plan (source / work / save dtype, rounding), edited tensor and parameter counts, dropped sides;
  - memory: chosen `gram` mode, estimated bytes;
  - per iteration: mean gap, SE, threshold, per-group gaps, active hinges, rows, Gram spectrum (top 8, relative),
    kept eigenvalues, ‖c‖, step norm (measured and predicted √cᵀAc), scale, edit relative to ‖θ0 − θ_ref‖, mean
    group cosine, seconds;
  - stop reason, edit norm relative to ‖θ0 − θ_ref‖ and to ‖θ0‖;
  - per-family group means for the reference, the start and the end;
  - held-out margins (mean, SE, greedy share; common / differential for `side: both`) for the reference, the start, the
    end and every dose;
  - per dose: edit_rel, edit survival (and per serving format), gap, active hinges;
  - timings.

Lower-level pieces for research scripts: `load_sides`, `make_groups`, `group_margins`, `gram_matrix`,
`solve_coefficients`, `weighted_step`, `constraint_rows`, `span_coverage`.

## Limits

- **Twins must represent the target distribution's positives and hard negatives.** An edit can restore the group
  margins exactly and still move the behaviour only along the old axis: in one validation, restoring a tool-call
  propensity with twins whose negatives were easy (irrelevant tool sets) raised needed and unneeded calls together on a
  benchmark whose hard negatives are related tools with missing information. Build twins whose negatives look like the
  hard cases, and always check discrimination, not only the level, on a held-out behavioural set.
- **The dose is a free parameter.** Twin targets are a proxy; choose α on a behavioural split and confirm once.
- **Run the logit-bias baseline** before crediting the weight edit.
- **Thinking-mode decisions are unsupported**, and steering-like edits are known to break under chain of thought.
- **Calibrated int4 (GPTQ / AWQ) survival is unmeasured.**
- **Chained surgeries** accumulate edit norm; edit several decisions in one system (hold rows for the earlier ones)
  rather than one after another.
- **Status:** experimental; validated on one small model size so far. The ingredients (minimal-norm Gauss–Newton steps on a few
  outputs, gradient-span edits, stochastic rounding) are known; the combination with decision margins at templated
  positions, hinges and dose selection on a behavioural split is, as far as we know, new.

## Relation to CMR and merging

[Common-Mode Rejection](cmr.md) holds a decision's margin during training; surgery solves for it afterwards. Doses are
task arithmetic with a one-step task vector (θ0 + α·Δ, see [Model Merging](merging.md)).
