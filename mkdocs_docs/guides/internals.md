# Reading Model Internals

`palingenesis.internals` reads what a model represents, layer by layer, and checks whether the model uses it. It is
built for post-training: run it before and after a stage to see what the stage changed inside the model, not only in
its outputs.

- **Capture** the residual stream of every decoder layer at positions taken from the chat template (the last prompt
  token, the assistant / answer / reasoning tokens training puts loss on), streamed to sharded files on disk.
- **Probe** each layer with a linear classifier: grouped splits, a control task, calibrated probabilities, AUROC with
  cluster-bootstrap intervals, transfer to other datasets and languages. A probe is saved as one file that records
  exactly where it reads.
- **Lens** each layer through the model's own final norm and head (logit lens), or through trained per-layer
  translators (tuned lens): where a prediction or a decision forms.
- **Compare** two checkpoints per layer: debiased linear CKA, cosine, and probe transfer (train on A, test on B).
- **Intervene**: add a probe's direction (steering), remove it (ablation), or swap activations between two prompts
  (patching), and measure an exact decision log-odds against random-direction controls.
- **Use** a probe: as dataset labels, as a deployment gate with a calibrated threshold, or as an RL reward function.

It works on hybrid models (Qwen3.5 / Qwen3-Next: Gated DeltaNet linear-attention layers next to full attention), on
Llama, Qwen and Gemma models, through multimodal wrappers and the library's own wrappers (activation checkpointing,
layers compiled in place, FSDP2), with hooks only: no module is replaced or wrapped. The research behind each choice is
in [Interpretability Research](../architecture/interpretability.md).

```bash
pgs internals run --config internals.yaml       # capture + per-layer probes + checkpoint comparison
pgs internals lens --model runs/sft/final --data chats.jsonl --decision "<tool_call>"
pgs internals causal --model runs/sft/final --probe runs/internals/probes/sft/best.safetensors --data dev.jsonl --decision "I"
```

## The residual index

Every tool names a residual by `r`:

| r | what |
|---|---|
| 0 | the input of decoder layer 0 (the embeddings, including any embedding scale, e.g. Gemma's √d) |
| i + 1 | the output of decoder layer i |
| n (the number of layers) | the output of the last layer, **before** the final norm |

This matches `output_hidden_states` for every r < n. It does not for r = n: transformers 5 returns the final norm's
output as the last entry of `hidden_states`, so a logit lens built on `hidden_states` normalises the last layer twice
and a probe "at the last layer" reads a different object than at every other layer. Here every residual is raw, and
the lenses apply the final norm themselves.

In a hybrid model the residual stream is the same object after a Gated DeltaNet layer as after an attention layer.
Every report tags each residual with the mixer that wrote its last update (`embedding`, `linear_attention`,
`full_attention`, ...): read layer sweeps by mixer as well as by depth.

## Data

Rows are JSONL objects:

```json
{"id": "tqa-0193", "group": "Q:Who wrote Il Gattopardo", "label": 1,
 "messages": [{"role": "user", "content": "Who wrote Il Gattopardo?"}],
 "chat_template_kwargs": {"enable_thinking": false}}
```

- `messages` (required), `tools`, `prefix_calls` / `prefix_content` (a decision right after a tool call, as in the
  [twins format](cmr.md)), `chat_template_kwargs` (overrides the run's defaults key by key).
- `label`: 0 / 1, or a probability (for example the share of k sampled answers that were correct). Training uses soft
  labels as soft targets; evaluation binarises at `label_threshold` (0.5).
- `group`: the unit of independence. Rows of one group never land in two splits, and confidence intervals resample
  groups. Put the translations, paraphrases and repeated samples of one question in one group; without `group`, each
  row is its own (`id` if present).

## Positions

| position | the residual read |
|---|---|
| `last_prompt` | the last token of the prompt rendered with the generation prompt: the position whose next-token distribution is the first answer token. The prompt is the row's messages without a trailing assistant turn, rendered exactly as [margins](cmr.md) render a decision side. |
| `assistant_<r>` | the tokens training puts loss on (`ChatDataset`'s mask: text, tool calls, a reasoning block, the end-of-turn token) |
| `answer_<r>` | the same without the reasoning block |
| `reasoning_<r>` | the reasoning block only |

`<r>` is `mean` (one row per example, averaged in float32), `first`, `last` or `all` (one row per token; for lenses
and token-level analysis). Positions come from the same renders and masks as training, so a probe reads the same
tokens the trainer computes losses on. Batches are right-padded: causal attention and the recurrent state of linear
attention never see padding at a real position (a test checks that a row's residuals are identical alone and in a
padded batch on the hybrid model).

## Capture

```bash
pgs internals capture --model Qwen/Qwen3.5-0.8B --data known_train.jsonl --out acts/base_train \
    --position last_prompt --layers all --chat-template-kwargs '{"enable_thinking": false}'
```

A store is a directory:

| file | content |
|---|---|
| `meta.json` | model fingerprint (path, type, hidden size, depth, layer types, a hash of the weights, a hash of the chat template), position, residuals, mixers, storage dtype, chat template kwargs, dropped rows by reason, library version |
| `rows.jsonl` | one line per output row: source row, id, label, group, token count, token position |
| `shard_00000.safetensors` … | `{"r<k>": [rows, d]}` for every residual, one shard per 4,096 rows |

Memory is bounded: positions are gathered inside the hook (a batch's full `[B, T, d]` is never kept), the forward stops
after the deepest residual read, a shard is written every `shard_rows` rows, and a sweep loads one residual at a time.
The default storage dtype is bfloat16: residual streams can carry outlier features in the thousands, which float16 can
overflow, and probe statistics are far coarser than bfloat16's resolution.

## Probes

```bash
pgs internals probe --acts acts/base_train --out probes/base --test popqa=acts/base_popqa --test it=acts/base_it
```

Per residual, on the train store's grouped split (70 / 10 / 20 % of groups by default):

| | |
|---|---|
| **fit** | `logistic` (default): L2 logistic regression on standardised features, LBFGS, the strength chosen on validation from 1e-4 … 1. `diffmean`: the difference of class means. `lda`: the covariance-adjusted difference of means, with shrinkage chosen on validation. |
| **layer** | the best layer is chosen on **validation** AUROC; the test AUROCs of the other layers are not selection-free |
| **calibration** | Platt scaling on validation. Valid for that distribution only: recalibrate per domain and language |
| **thresholds** | logit thresholds flagging 1 %, 5 %, 10 % of validation negatives, for gates; every test set reports the TPR and FPR they give there |
| **test** | AUROC with a 95 % cluster-bootstrap interval (groups resampled), Brier, log loss, equal-mass ECE |
| **controls** | `control`: the same fit on random labels drawn per group (a control task); it must stay near 0.5, otherwise groups leak across splits. `selectivity` = AUROC − control AUROC. `dmean`: the diff-of-means AUROC next to the chosen fit. Residual 0 (the embeddings) is in every sweep: a high AUROC there means the label is lexical. `length-only AUROC`: the prompt length alone. |
| **transfer** | every `--test` store (other datasets, other languages) is scored by the train-split probe |

```
  r mixer              val   test [95% CI]   dmean   ctrl    sel        popqa           it
  0 embedding        0.612  0.598 [0.571,0.624]  0.571  0.507  0.091        0.553        0.520
 ...
 18 linear_attention 0.921  0.917 [0.903,0.930]  0.902  0.496  0.421        0.928        0.922
 ...
best layer (validation): 18   length-only AUROC: test 0.541, popqa 0.522, it 0.515
```

(Illustrative numbers.) Each probe is saved as `layer_<r>.safetensors` and the validation pick as `best.safetensors`,
with `sweep.json` next to them.

### The probe file

One `.safetensors` file: tensors `weight` (the classifier, raw residual units, standardisation folded in) and
`mean_diff` (μ⁺ − μ⁻), and in the header the kind, residual, position, bias, Platt (a, b), thresholds, mean residual
norm, the model fingerprint, chat template kwargs, label name, fit strength, train / validation sizes and every metric
of the sweep. `Probe.load(path)` gives `logit(x)`, `proba(x)`, `flag(x, at="fpr_0.05")` and `direction`.

The classifier and the intervention direction are kept apart on purpose. Classification uses the logistic weights.
Steering and ablation use the difference of means, which is the direction a model's behaviour responds to; logistic
weights also fit noise directions that separate the classes in the sample.

## Comparing checkpoints

Capture both checkpoints on the same rows and position, then:

```bash
pgs internals compare --a acts/base_train --b acts/sft_train
```

```
  r  cka 0.981  cos 0.974  norm x1.03  auroc A->A 0.917  A->B 0.889  B->B 0.920  dir cos 0.912
```

Per residual:

- `cka`: debiased linear CKA. The biased estimator drifts toward 1 when rows are not much more numerous than the hidden
  size, which is the usual case for a few thousand prompts.
- `cka_biased`: the biased estimator, for reference.
- `cosine`: the mean cosine of a row's residual in A and in B. Fine-tuning keeps the coordinate basis, so this is
  meaningful between checkpoints of one lineage.
- `norm_ratio`: the mean of |B| / |A|.
- `transfer`: A's probe on A's held-out rows, A's probe on B's rows, B's own probe, and the cosine between their
  diff-of-means directions.

Reading it:

| pattern | interpretation |
|---|---|
| CKA ≈ 1, A→B ≈ A→A, behaviour changed | the representation is unchanged: the stage changed what reads it (later layers, the head) |
| B→B ≈ A→A but A→B drops, direction cosine low | the information is still there but moved: a frozen probe from A is no longer valid on B |
| B→B drops | the stage removed (or blurred) the information at this layer |
| CKA drops only in the last layers | the usual signature of a post-training stage: the readout is rewritten, the middle is kept |

CKA is dominated by the top principal components and can move without the model's function changing, so it is never
reported without the probe measures next to it.

## Lenses

```bash
pgs internals lens --model runs/sft/final --data chats.jsonl --position answer_all
pgs internals lens --model runs/sft/final --data tool_prompts.jsonl --position assistant_first --decision "<tool_call>"
pgs internals tuned-lens --model runs/sft/final --data chats.jsonl --out lens.safetensors && \
pgs internals lens --model runs/sft/final --data chats.jsonl --tuned lens.safetensors
```

Per residual, averaged over every selected position: `KL` to the model's final distribution, `top1` agreement with the
final top token, `CE` of the actual next token, and with `--decision` the exact log-odds of the decision token set (the
[margin](cmr.md) of, for example, `<tool_call>` at the first assistant token). That last column shows at which depth a
decision forms, and whether a stage moved it there or only at the end.

Before reporting anything, the lens checks that `head(norm(residual n))` reproduces the model's logits (it fails loudly
on a wrong norm, a missing logit transform or the wrong submodule). The head includes the model's logit transform
(Gemma's softcap, Cohere / Granite scales).

With tied embeddings, early logit-lens layers mostly decode the input token: use the tuned lens there. The tuned lens
has one affine translator per residual r < n, zero-initialised as a residual map (it starts as the logit lens; weight
decay pulls it back toward it), trained with AdamW to minimise KL(final ‖ lens) on the post-training distribution
(assistant tokens by default), one backward per layer.

Language switching: give `--decision` a set of first tokens of Italian replies against the rest, on prompts in Italian,
and see where the Italian log-odds rise. Middle layers of multilingual models often decode the concept in English
first and switch to the output language late.

## Causal checks

A probe reads what is decodable. Whether the model uses it is a separate question.

```bash
pgs internals causal --model runs/sft/final --probe probes/sft/best.safetensors --data dev.jsonl \
    --decision "I" --alphas -2,-1,0,1,2 --n-random 8
```

```
layer 18, 400 prompts, base margin -6.214
  alpha -2.00: margin -7.902
  ...
  alpha +2.00: margin -2.117
slope 1.446 vs random 0.031 ± 0.052 (z 27.2)
ablation: margin change -0.884 vs random 0.004 ± 0.011
```

(Illustrative numbers.) The probe's diff-of-means vector is added at its residual with each `alpha`, in units of
|μ⁺ − μ⁻| (alpha = 1 moves the negative-class mean onto the positive one). The decision margin is measured at each
row's decision position. The same is done for random directions of the same norm, which are the null, since any large
push moves margins a little. Then the direction is ablated at every residual (h ← h − (h·û)û) and compared with
ablating random directions. A probe whose direction moves the decision far beyond the random null, in the expected
sign, is used. One that does not is only decodable.

Steering adds at every position, the prompt included. In hybrid models a decision is carried by the recurrent state
from the prefill, so steering only the generated tokens can do nothing. `pgs internals steer` generates with a dose, to
read what it does:

```bash
pgs internals steer --model runs/sft/final --probe probes/sft/best.safetensors --prompt "Who was George Rankin?" --alpha 0,-1,-2
```

**Activation patching** between the two sides of [twins](cmr.md), one residual and one position at a time:

```bash
pgs internals patch --model runs/sft/final --twins tool_twins.jsonl --decision "<tool_call>" --offsets -1,-2,-3,-4
```

`denoise` puts the `pos` side's residual into the `neg` side's run and tests sufficiency. `noise` does the reverse and
tests necessity. The effect is normalised as (patched − base) / (source − base): 1 means that this residual at this
position carries the whole difference. Positions count from the end of each prompt (−1 is the decision position).
Twins that differ by a minimal token replacement are the right corruption; noise-based corruptions push activations off
distribution. Start with single layers: windows of layers inflate effects.

## Using a probe

```python
from palingenesis.internals import ProbeScorer

scorer = ProbeScorer("probes/sft/best.safetensors")          # the model recorded in the probe, or model=, tok=
p = scorer.probabilities(rows)                               # [N] calibrated
abstain = scorer.gate(rows, at="fpr_0.05")
```

The scorer renders each row at the probe's position with the probe's chat template kwargs, reads only the probe's
residual and stops the forward there. It refuses a model of another shape. It warns when the weights or the chat
template differ from the probe's: moving a probe to another checkpoint is a hypothesis, which `compare` tests.

From safest to riskiest:

1. **Labels.** `pgs internals score --probe P --data rl_prompts.jsonl --out rl_prompts_scored.jsonl --column known`
   writes the probability into a column that rewards can read (every column reaches the reward functions). For a
   prompt-only probe (`last_prompt`) the policy cannot influence its label. Relabel with the current checkpoint between
   stages: one prefill per prompt instead of k sampled answers.
2. **Gate.** Abstain, ask or route when the probe fires at a calibrated operating point. Re-check the threshold after
   every training stage and per domain or language.
3. **Reward.** `palingenesis.internals.score:probe_reward` is a batched reward function, scored by a **frozen** copy
   (on CPU by default, so it does not compete with the policy and vLLM for GPU memory; no gradient flows through it):

   ```yaml
   rewards:
     known: {fn: palingenesis.internals.score:probe_reward, weight: 0.0, args: {probe: probes/sft/best.safetensors}}
   ```

   With weight 0 it is logged as `rewards/known` and acts as a monitor. Optimising a policy against a detector teaches
   it to evade the detector. A frozen copy keeps the policy from changing the detector's activations, but not from
   changing its text. Prefer prompt-only probes, which the policy cannot change, to condition another reward (pay for
   abstaining only where the probe says the model does not know). Prefer off-policy uses (labels, preference pairs).
   Keep a held-out probe, never used as a reward, as the monitor. Watch the gap between the frozen probe and a probe
   re-fitted on the policy's outputs.

## One job: `pgs internals run`

```yaml
models:
  base: Qwen/Qwen3.5-0.8B
  sft: runs/sft/final
train: data/known_en.jsonl          # labelled rows (grouped splits come from this file)
tests:
  popqa: data/known_popqa.jsonl     # another dataset: transfer
  it: data/known_it.jsonl           # another language: transfer
compare: [[base, sft]]
position: last_prompt
layers: all
probe_kind: logistic                # logistic | diffmean | lda
val: 0.1
test: 0.2
bootstrap: 1000
label_threshold: 0.5
control: true
chat_template_kwargs: {enable_thinking: false}
store_dtype: bfloat16
model_dtype: null                   # default bf16 on CUDA, fp32 on CPU
batch_size: 16
max_batch_tokens: 32768
max_tokens: null
limit: null
seed: 0
output_dir: runs/internals/known
```

The file may nest everything under `internals:`. Override keys on the command line with `--set layers=12-20` (values
parsed as YAML). `validate()` refuses unknown keys (with a suggestion), `*_all` positions for probes, comparisons of
models that are not listed, and invalid fractions, dtypes and sizes.

Output: `acts/<model>/<dataset>/` (stores, reused when they already exist with the same settings and weights),
`probes/<model>/` (one probe per layer, `best.safetensors`, `sweep.json`), `compare/<a>__<b>.json`, and `report.json`.

## Worked example: a "do I know this?" probe across layers, before and after a stage

The question: before answering, does the model's residual stream at the last prompt token say whether it knows the
answer? At which depth? Did a training stage change that representation, or only the way the model acts on it?

**1. Labels from the model's own answers.** For each question, sample k answers at T = 1 from the checkpoint you probe
and grade them. Then `label` = the share correct (a soft label; it is binarised at 0.5 for evaluation) and `group` = the
question id. A translated question keeps the group of its original. Label each checkpoint with its own answers: after
a stage, what a model knows is a property of that model.

**2. Two runs.** One config per checkpoint, each with its own labelled files: `train` = an English QA set, `tests` =
another dataset and the same questions in Italian. Then a comparison of the two checkpoints on one row file. Keep one
row order and one `max_tokens` so the stores hold the same rows.

```bash
pgs internals run --config known_base.yaml      # models: {base: ...}, train: known_en_base.jsonl, tests: {...}
pgs internals run --config known_sft.yaml       # models: {sft: ...},  train: known_en_sft.jsonl,  tests: {...}
pgs internals compare --a runs/internals/base/acts/base/train --b runs/internals/sft/acts/sft/train
```

`compare` fits both probes on the labels of `--a`, so the transfer columns ask whether the *same* property is read the
same way in both checkpoints. If one label file serves both checkpoints, a single run with `models: {base, sft}` and
`compare: [[base, sft]]` does all of this.

**3. Read the sweep, for each model.**

- Is the best layer's test AUROC above the generation-based baselines you already have (mean log-probability of the
  greedy answer, agreement between samples), at no generation cost?
- Is the control near 0.5, the embedding layer and the length baseline low? If so, the probe reads something the
  prompt's surface does not give away.
- Does `dmean` stay close to `logistic`? Then the signal is a simple shift along one direction.
- Do the transfer columns (another dataset, another language) hold up in the middle layers and fall in the first and
  last ones? A language-independent signal behaves this way. Each test set gets its own calibration before the
  probability is used.

**4. Read the comparison.** At the probe's layer, if `A->B` ≈ `A->A` and the CKA stays high, the stage kept the
representation. If abstention behaviour still changed, the stage changed its use. If `B->B` holds but `A->B` drops,
refit the probe on the new checkpoint before using it as a label or a gate.

**5. Is it used?** `pgs internals causal` with the decision tokens of an abstention ("I", or the first token of your
refusal template) on held-out prompts. If steering along the direction raises abstention far beyond random directions,
the direction is causal for the behaviour. If it does not, the model knows but does not act on it, and a
knowledge-aware reward or a gate has to connect the two.

## Python API

```python
from palingenesis.internals import (
    capture_dataset, ActivationStore, sweep, train_probe, evaluate, grouped_split, Probe,
    compare_stores, linear_cka, lens_report, fit_tuned_lens, check_readout,
    direction_effect, patching_sweep, generate, ProbeScorer, score_rows, probe_reward,
    Recorder, Selection, steer, ablate, patch, edit_residual, find_sites, active_hooks,
)
from palingenesis.internals.capture import run_backbone

store = capture_dataset(model, tok, rows, "acts/m", layers="all", position="last_prompt")
report = sweep(store, tests={"it": ActivationStore("acts/m_it")}, save_dir="probes/m")

with Recorder(model, [0, 12, 24], stop=True) as rec:     # any forward; Selection picks positions inside the hook
    run_backbone(model, input_ids, attention_mask)
    acts = rec.take()
with steer(model, {18: 2.0 * probe.mean_diff}):          # any generate / forward
    out = model.generate(**inputs)
assert active_hooks(model) == 0                          # every context removes its hooks, also on errors
```

Edits (`steer`, `ablate`, `patch`, `edit_residual`) run before readers on the same residual, so a `Recorder` sees the
edited value whatever the order the contexts were entered in. Hooks fire on every forward while active: keep the model
in eval mode (activation checkpointing in train mode runs a layer's forward twice).

## Limits and scope

- **Single device.** `device_map`-sharded and FSDP models are hooked correctly, but the lenses need the final norm
  and head unsharded.
- **Probes are linear and read one residual.** Per-head probes (ITI), the Gated DeltaNet recurrent state, multi-layer
  windows and ensembles are not implemented. Multi-layer ensembles are the recommended next step for a probe used as a
  reward, since evasion is costlier across many layers.
- **No attribution patching.** Patching is exact and costs layers × positions forwards per batch; use it on a few
  dozen twins.
- **No sparse autoencoders.** On supervised concepts, linear probes and the difference of means match or beat them,
  and an SAE must be retrained after each stage.
- **No CCS / unsupervised discovery.** Unsupervised methods find the most prominent binary feature, not necessarily
  the one you want.
- **Calibration does not transfer** across domains or languages; a threshold is valid where it was chosen.
- **The causal check needs a decision token.** Behaviours that are not a single-token decision at a known position need
  `steer` and reading the generations.
