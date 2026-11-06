# Off-Policy Distillation

*Train a student on fixed sequences (your data, or text a teacher wrote) against one or more teachers' full distributions at every trained token. A mode of `pgs train`: everything the SFT trainer does (data mixing, chat masking, packing, FSDP2, activation checkpointing, SeCO for long sequences, checkpoints and resume, plugins) works the same way.*

---

## Off-policy or on-policy?

| | Off-policy (`pgs train` + `distill.enabled`) | On-policy ([`pgs distill`](distillation.md)) |
|---|---|---|
| States the student trains on | the dataset's sequences (gold or teacher-written) | the student's own samples |
| Cost per token | one teacher forward (or none, with a cache) | sampling + a teacher forward |
| Teacher signal | full distribution, or a cached top-k | full distribution, top-k, or sampled |
| Scale | the SFT trainer's: FSDP2 over many GPUs, long contexts, packing | one GPU per run |
| Fits | compressing a large teacher on a big corpus; continued pretraining with a teacher; quantisation-aware training against the full-precision model; context distillation of system prompts / documents | fixing the student's own errors and loops (exposure bias); reasoning and agent behaviour |

The two compose: off-policy distillation on a large corpus first (cheap, dense, parallel), then on-policy distillation on the tasks where the student's own trajectories matter. On-policy distillation addresses exposure bias, which off-policy training cannot see (arXiv:2306.13649, 2306.08543); off-policy is what you can afford on billions of tokens.

## Quickstart

```yaml
# pgs init kd   writes this file
model:
  name_or_path: Qwen/Qwen3-0.6B
  torch_dtype: float32            # fp32 master weights, bf16 compute (train.bf16)
data:
  dataset: HuggingFaceH4/ultrachat_200k
  dataset_split: train_sft
  max_seq_length: 4096
distill:
  enabled: true
  divergence: forward_kl
  temperature: 1.0
  ce_weight: 0.1                  # a little cross-entropy on the gold tokens
  teachers:
    big: {model: Qwen/Qwen3-4B}   # online: runs on every micro-batch, frozen
train:
  output_dir: runs/kd
  max_steps: 2000
  learning_rate: 1.0e-5
```

```bash
pgs train --config kd.yaml
torchrun --nproc_per_node 8 -m palingenesis.train --config kd.yaml     # student and teacher sharded
```

The loss of each trained token t is

```
kd_weight · Σ_i w_i · D(p_i(·|t) ‖ q(·|t)) / Σ_i w_i  +  ce_weight · −log q(gold_t)
```

over the teachers i that cover t, weighted by the data's per-token weights and divided by the global trained-token count, exactly like the SFT loss (so gradient accumulation and data parallelism average correctly). Feature-matching and plugin terms are added on top.

## Teachers

### Online teachers

`teachers.<name>.model`: a frozen causal LM loaded next to the student. Its forward runs without gradient and keeps only the final hidden states at the trained positions; its logits are projected a slice of rows at a time inside the loss, so neither the student's nor the teacher's `[tokens, vocabulary]` logits ever exist in full.

| Key | Default | Meaning |
|---|---|---|
| `dtype` | `bfloat16` | the teacher's weights (`float16`, `float32`) |
| `placement` | `auto` | `shard`: FSDP2-sharded over the data-parallel ranks like the student (a teacher too large for one GPU's share); `replicate`: a full copy per rank; `auto`: shard when the student is sharded |
| `device` | the student's | single process: put the teacher on another GPU (`cuda:1`) |
| `attn_implementation` | the student's | e.g. `flash_attention_2` for a teacher whose student uses `sdpa` |
| `tokenizer` | the model's | must share the student's vocabulary (below) |

### Cached teachers (precomputed top-k)

A teacher you cannot afford to run on every step (a 70B teacher for a 1B student, or a teacher served elsewhere) can be scored once:

```bash
pgs distill-cache --config kd.yaml --teacher big --out caches/big --top_k 64              # HF teacher
torchrun --nproc_per_node 8 -m palingenesis.distill.build --config kd.yaml --teacher big   # 8 GPUs, one shard each
pgs distill-cache --config kd.yaml --teacher big --backend vllm --tensor_parallel_size 4   # vLLM prompt log-probs
```

then train against it: `teachers.big.cache: caches/big`. With both `model` and `cache` set, training reads the cache and the builder uses the model.

- **Same documents.** The builder walks the training config's own data (every row of every source once, unpacked, with its masking and its teacher view), so a cache matches the run it was built for.
- **Content-addressed.** Each document is keyed by a 128-bit hash of its token ids and its loss mask. The trainer finds a document's targets whatever the order, the shuffling, the packing, the number of ranks, the epoch or a resume. A document the cache does not hold is an error by default (`distill.missing: error`): the data or its masking changed since the build.
- **Verified.** The cache records the student tokenizer's vocabulary fingerprint, the teacher, its view, `top_k`, the temperature and the log-prob dtype; a different student tokenizer is refused at startup, and a build continuing a cache with other settings is refused.
- **Compact and memory-mapped.** Per trained position: k token ids (int32, already in the student's vocabulary) and their log-probs (float16 by default, `--lp_dtype float32`), plus the gold token's log-prob. k = 64 costs 390 bytes per trained token. The DataLoader workers map the shards and read only the rows of the documents they train on.
- **Resumable.** Shards become visible when their index is written (atomically, last); an interrupted build discards the unfinished shard and skips the finished documents. Every rank of a multi-GPU build writes its own shards.
- **vLLM.** The `vllm` backend runs the teacher prefill-only (`max_tokens=1`, `prompt_logprobs=k`) and reads the top-k and the gold token's log-prob at every position, at temperature 1. Much faster for large teachers; needs the `vllm` extra.

**Top-k is an approximation; how its tail is handled matters.** `distill.topk_mode: tail` (default) adds the gold token to the teacher's top-k and one bucket per side holding the rest of the mass, so the divergence is between the two full distributions coarse-grained on the same partition of the vocabulary (it equals the exact divergence when k covers the teacher's mass). `renorm` renormalises both sides over the support, the convention of the sparse-KD literature (arXiv:2503.16870 shows top-k renormalisation biases the student's calibration); use it to apply a temperature other than the cache's. `train/topk_mass/<teacher>` reports how much of the teacher's mass the cached top-k holds: at a well-trained teacher's states it is usually above 0.95 with k = 64.

### Several teachers

```yaml
data:
  sources:
    - {dataset: data/math.jsonl, name: math, weight: 0.5}
    - {dataset: data/chat.jsonl, name: chat, weight: 0.5}
distill:
  missing: skip                    # a token no teacher covers trains on CE only
  ce_weight: 0.1
  teachers:
    math_expert: {model: org/math-7b, sources: [math]}
    generalist:  {cache: caches/generalist-32b, weight: 2.0}     # every source
```

Per token, the teachers covering it are averaged with their `weight`s (a cached teacher covers the documents its cache holds; `sources` restricts a teacher to some data sources). Routing by source gives each domain its own expert teacher; averaging several teachers on the same tokens is ensemble distillation. Metrics come per teacher (`train/kd/<teacher>`).

### Teachers with another tokenizer

Token-level KD compares the two distributions id by id, so the teacher must share the student's vocabulary: an identical tokenizer, or a teacher vocabulary that is a prefix of the student's (a ChatML student extending Llama 3's vocabulary), with the end-of-turn token mapped (`eos_map`, automatic in the common cases; the same bridge as on-policy distillation, checked with probe texts before anything loads). Any other pair is refused with an error at startup. For a teacher of another family use on-policy distillation, whose `xtok` loss compares text chunks across tokenizers ([guide](distillation.md#across-tokenizers)).

## Losses

| `distill.divergence` | D(p ‖ q), p = teacher, q = student | Source |
|---|---|---|
| `forward_kl` | Σ p (log p − log q): mode-covering, the classic KD objective | Hinton et al., arXiv:1503.02531 |
| `reverse_kl` | Σ q (log q − log p): mode-seeking | MiniLLM, arXiv:2306.08543 |
| `jsd` | β KL(p‖m) + (1−β) KL(q‖m), m = βp + (1−β)q | generalized JSD, GKD, arXiv:2306.13649 |
| `skew_forward_kl` | KL(p ‖ βp + (1−β)q) | DistiLLM, arXiv:2402.03898 |
| `skew_reverse_kl` | KL(q ‖ βq + (1−β)p) | DistiLLM, arXiv:2402.03898 |
| `tvd` | ½ Σ \|p − q\|: bounded, robust to a teacher that is confidently wrong | |

- `temperature` softens both distributions; with `temperature_scale: true` (default) the divergence is multiplied by T², which keeps gradient magnitudes comparable across temperatures (Hinton et al.).
- `ce_weight` mixes in cross-entropy on the gold token; `kd_weight: 0` turns distillation into plain SFT plus whatever features / plugins add.
- The data's per-token weights apply (`data.turn_scaling`), and only trained tokens are distilled: chat masking decides what the teacher supervises (assistant turns, `include_observations`, `train_on_reasoning`, per-message `"loss": false`). Pretraining-mode sources (`mode: pretrain`) distill every token.
- `train/kd_loss` (per covered token), `train/gold_ce` (cross-entropy on the gold tokens, always reported), `train/kd_coverage`, `train/kd/<teacher>`.

## Context distillation (teacher views)

The teacher can read a different rendering of each row than the student: the student learns to answer without the context as the teacher answers with it (arXiv:2112.00861, 2209.15189).

```yaml
distill:
  teachers:
    informed:
      model: Qwen/Qwen3-4B
      privileged_field: reference        # a row field only the teacher sees
      privileged_template: "{content}\n\nReference: {privileged}"
      privileged_role: system            # or user: rewrite the last user turn
      # system_prompt: "You are ..."      # a system prompt only the teacher reads
      # prefix_messages: fewshot.jsonl    # few-shot examples before the conversation (never trained)
      # messages_field: teacher_messages  # the teacher's own copy of the conversation
      # chat_template_kwargs: {enable_thinking: true}
      # view_fn: my_module:make_view      # (row, messages) -> teacher messages
```

Both renderings are masked by the same rules, so their trained tokens are the same answer tokens; each row's alignment is checked token by token (up to the shared-vocabulary mapping) and every student position is paired with the teacher position that predicts the same token, however much longer the teacher's prefix is. A row whose answers differ in the two renderings, or whose teacher view exceeds `max_seq_length` (default twice `data.max_seq_length`), is dropped and counted (`Teacher views: N rows dropped`). The view is the same machinery on-policy self-distillation uses for privileged context (`with_privileged`).

With an online teacher, a view needs unpacked rows (each row has its own teacher sequence). A cached teacher bakes the view into the cache: the targets are stored at the student's positions, so packing works again.

## Feature matching

```yaml
distill:
  features:
    - {student: 11, teacher: 23, loss: mse, weight: 1.0}          # decoder-layer outputs
    - {student: model.layers.5.mlp, teacher: model.layers.10.mlp, loss: cosine, weight: 0.5, projection: linear}
  feature_lr: 1.0e-4               # learning rate of the projections (0: train.learning_rate)
```

Matches module outputs at the trained positions (FitNets, arXiv:1412.6550; layer-wise hidden-state targets of low-bit distillation, e.g. arXiv:2402.11295). `student` / `teacher`: a decoder-layer index (negative from the end) or a module path (matched against the end of module names, so wrappers do not matter). When the widths differ a learnable linear projection is created (`projection: auto`), trained with the student, checkpointed with it and left out of exported models. `loss`: `mse`, `nmse` (of L2-normalised vectors), `cosine`, or `lens_kl`: a logit-lens target, the forward KL between the two layers read through each model's own final norm and output head (frozen for this term, so only the layers below learn from it; useful for the last few layers, where intermediate states are already decodable; no projection needed). `via` picks the online teacher (default the first). Needs an online teacher; not available with SeCO.

Attention-score distillation is not built in: it needs the `[T, T]` attention maps, which fused attention kernels never materialise. A plugin can capture the q/k/v projections of one layer (`feature_requests` / module hooks) and compute relation matrices on a subsample of rows (MiniLM-style, arXiv:2012.15828); see [Plugins](../reference/plugins.md#trainer-plugins).

## Long contexts and many GPUs

- **No full logits, on either side.** The student's hidden states at the trained positions are projected to logits a slice at a time (`distill.rows_per_slice`, default ~512 MB of fp32 per side) and differentiated immediately; the teacher's logits for the same rows are computed in the same slice. Peak memory is one slice of each, whatever the sequence length.
- **Teacher memory.** An online teacher keeps its final hidden states for the micro-batch (`rows × seq × hidden` in bf16: 0.5 GB for 64k tokens of a 4096-wide teacher) and its weights (sharded with `placement: shard`). A cached teacher costs `seq × k × 6` bytes per row.
- **Packing** keeps documents apart in the student and the teacher (each model gets its own variable-length arguments when one flattens a packed batch and the other does not); cached targets follow their documents into the packed rows.
- **SeCO** (`memory.seco`): chunk-wise forward/backward for sequences far longer than activations allow on one GPU; the distillation loss runs per chunk on the trained rows inside it. The online teacher's forward runs once over the whole sequence (no gradient, so only one layer's activations at a time).
- **FSDP2.** The student is sharded as in SFT; the teacher shards with it under `placement: shard`. Gradients are reduced exactly like the SFT loss's (tested: two-process FSDP equals one process on the whole batch, student and teacher sharded, QAT plugin included).
- **Context parallel.** Under `parallel.context_parallel` each rank holds a slice of every row, and attention is exact across the group (see the multi-GPU guide). Cached teachers' targets are computed on whole rows and sharded like the inputs, with source tags and data weights. An online teacher runs on the student's slices, with its attention also routed through the group, so it reads the whole row. Its layers must be supported too: a recurrent / linear-attention teacher is refused. A teacher view (context distillation) with an online teacher is refused, because the view is a different rendering of the row and is not split like it. Cache such a teacher instead.
- **Resume** is deterministic like SFT's: the data stream is replayed to the next micro-batch, a cached teacher's targets are found by content, and plugin state is restored from the checkpoint.
- **Memory estimate**: `pgs profile --config kd.yaml` counts the teachers' weights (exact, divided by the ranks when sharded) and their per-micro-batch targets.

## Quantisation-aware training while distilling

The trainer's plugin API ([reference](../reference/plugins.md#trainer-plugins)) lets you change the student before it is sharded, add loss terms and gradient terms, set optimizer groups, ramp anything with training progress, match hidden states, and fold weights when the model is exported. `palingenesis.qat:TernaryQAT` is a complete reference built on it:

```yaml
distill:
  enabled: true
  temperature: 1.0
  teachers: {fp: {model: Qwen/Qwen3-0.6B}}      # the full-precision model
plugins:
  custom:
    - class: palingenesis.qat:TernaryQAT
      args:
        group_size: 128                          # absmean scale per 128 weights (0: per tensor, -1: per row)
        exclude: ["lm_head", "embed"]
        quantize_embedding: false                # true: the embedding (and a tied head) too
        strength: {start: 0.0, end: 0.1}         # ramp the quantisation in over the first 10%
        grid_weight: {start: 0.5, end: 1.0, start_value: 0.0, end_value: 1.0e-4}
        latent_lr: 2.0e-4
        latent_eps: 1.0e-12
        clamp: 1.5
        hidden_layers: [-1]                      # also match the last layer's output
        export: folded
```

What it does, hook by hook:

| Hook | Effect |
|---|---|
| `transform_model` | every matching `nn.Linear` becomes a `TernaryLinear` (same Parameter, same name: checkpoints, resume and tied embeddings unchanged); optional `TernaryEmbedding` / ternary head |
| forward | `W_q = s · clip(round(W/s), −1, 1)` with s = mean \|W\| per group (absmean, BitNet b1.58, arXiv:2402.17764), straight-through estimator, optional int8 per-token activations |
| `on_forward_start` | sets the quantisation `strength` (a buffer, so torch.compile does not recompile): `W + strength · (W_q − W)` |
| `feature_requests` | hidden-state targets from the full-precision teacher |
| `param_groups` | the latent weights in their own optimizer group (learning rate, Adam eps, no weight decay) |
| `on_before_optimizer_step` | adds the gradient of `grid_weight · Σ (W − W_q)²`, pulling latents to their grid points (elementwise on each rank's shard: exact under FSDP) |
| `on_after_optimizer_step` | clamps latents to `±clamp · s` |
| `metrics` | `codes_changed` (share of codes different from the start), `zero_share`, `strength`, `grid_reg` |
| `export_state_dict` | `folded`: the exported model holds `W_q` in the original layout (any loader reads it); `latent`: the latents as trained |

The plugin is also registered as the entry point `ternary-qat` (`plugins.custom: [ternary-qat]` runs it with its defaults). The defaults are a starting point, not a tuned recipe. Write your own plugin for other quantisers (learned scales, LSQ, 2-/4-bit grids, Hessian-weighted rounding), inserted modules (extra norms before output projections), calibration passes before training, or per-block reconstruction targets.

## Configuration reference

### `distill`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Off-policy distillation in `pgs train` |
| `teachers` | dict | `{}` | name → teacher settings (next table); `{name: model_id}` is a shorthand |
| `divergence` | str | `forward_kl` | `forward_kl`, `reverse_kl`, `jsd`, `skew_forward_kl`, `skew_reverse_kl`, `tvd` |
| `beta` | float | `0.5` | `jsd`: the teacher's weight in the mixture; `skew_*`: the skew α. In (0, 1) |
| `temperature` | float | `1.0` | Softmax temperature of both sides |
| `temperature_scale` | bool | `true` | Multiply the divergence by temperature² |
| `kd_weight` | float | `1.0` | Weight of the divergence |
| `ce_weight` | float | `0.0` | Weight of cross-entropy on the gold tokens |
| `topk_mode` | str | `tail` | Cached teachers: `tail` (top-k + gold + a tail bucket) or `renorm` |
| `missing` | str | `error` | A trained token with no teacher: `error`, or `skip` (CE only) |
| `features` | list | `[]` | Feature matching: `{student, teacher, weight, loss, projection, via, name}` |
| `feature_lr` | float | `0.0` | Learning rate of the feature projections (0 = `train.learning_rate`) |
| `rows_per_slice` | int | `0` | Rows projected per slice of the chunked loss (0 = automatic) |

### `distill.teachers.<name>`

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | str | `""` | Online teacher (HF id or path); with `cache`, the model the cache is built from |
| `cache` | str | `""` | A `pgs distill-cache` directory |
| `weight` | float | `1.0` | Mixing weight among the teachers covering a token |
| `sources` | list | `[]` | Data source names this teacher supervises (empty = all) |
| `dtype` | str | `bfloat16` | Online teacher's weights |
| `placement` | str | `auto` | `auto`, `replicate`, `shard` |
| `device` | str | `""` | Single process: the teacher's device |
| `attn_implementation` | str | `""` | Empty = the student's |
| `tokenizer` | str | `""` | Empty = the model's |
| `eos_map` | dict | `{}` | Student end-of-turn token → teacher's (shared-vocabulary bridge) |
| `probe_texts` | list | `[]` | Extra texts that must tokenize identically |
| `top_k` | int | `64` | Cache building: log-probs kept per position |
| `system_prompt` | str | `""` | View: a system prompt only the teacher reads |
| `prefix_messages` | list \| str | `[]` | View: messages (or a JSON/JSONL file) before the conversation; their answers are not trained |
| `privileged_field` | str | `""` | View: a row field placed by `privileged_template` |
| `privileged_template` | str | `{content}\n\n{privileged}` | Where the privileged text goes |
| `privileged_role` | str | `user` | `user` (last user turn) or `system` |
| `messages_field` | str | `""` | View: the teacher's own conversation field |
| `chat_template_kwargs` | dict | `{}` | View: the teacher's template options |
| `view_fn` | str | `""` | View: `module:function` (row, messages) → messages |
| `max_seq_length` | int | `0` | View token limit (0 = 2 × `data.max_seq_length`) |

Command-line overrides reach named teachers: `--distill.teachers.big.placement shard`, `--distill.teachers.cached.cache caches/big`.

**Not combinable** (refused by validation): `dpo.enabled`, the gated SFT objectives (`plugins.deft`, `dft`, `cadft`, `info_sft`, `pre_rl`), `memory.gradient_release`; feature matching with `memory.seco`; an online teacher view with `parallel.context_parallel`; an online teacher view with packing or pretokenize; teacher views with loop-recovery augmentation; `sources` routing of an online teacher with pretokenize.

## Examples

**Plain KD from a larger sibling** (the quickstart above): one online teacher, forward KL, a little CE.

**A large teacher, cached once:**

```yaml
distill:
  enabled: true
  teachers:
    big: {model: Qwen/Qwen3-32B, cache: caches/qwen3-32b-top64, top_k: 64}
```

```bash
pgs distill-cache --config kd.yaml --teacher big --backend vllm --tensor_parallel_size 4
torchrun --nproc_per_node 8 -m palingenesis.train --config kd.yaml
```

**Context distillation of a long system prompt** (the student learns to behave as if it had read it):

```yaml
distill:
  enabled: true
  divergence: reverse_kl
  teachers:
    self: {model: Qwen/Qwen3-4B, system_prompt: "<the 3,000-token policy document>"}
```

**Multi-teacher mix:** see [Several teachers](#several-teachers).

**Multi-GPU, long context, packed pretraining text with a sharded teacher:**

```yaml
data:
  sources: [{dataset: data/corpus.parquet, mode: pretrain, text_field: text}]
  packing: true
  max_seq_length: 32768
distill:
  enabled: true
  teachers: {big: {model: org/teacher-14b, placement: shard}}
train:
  gradient_checkpointing: full
```

**Ternary QAT while distilling:** see [above](#quantisation-aware-training-while-distilling).

## In Python

```python
from palingenesis.config import Config
from palingenesis.train import train
from palingenesis.qat import TernaryQAT

config = Config.from_yaml("kd.yaml")
result = train(config, plugins=[TernaryQAT(group_size=128, strength={"start": 0.0, "end": 0.1})])
```

`palingenesis.distill.build.build_cache(config, "big")` builds a cache in-process.

## Limitations

- Same-vocabulary teachers only (shared base vocabulary with an end-of-turn mapping); cross-tokenizer distillation is on-policy's `xtok`.
- A cached teacher at `topk_mode: tail` is exact only at the cache's temperature; another temperature needs `renorm` (which drops the tail) or a rebuild. vLLM caches are at temperature 1.
- Feature matching needs an online teacher and is not available under SeCO; with `model.compile`, the capture hooks may cause graph breaks.
- Attention-score / relation distillation is left to plugins (fused attention never materialises the maps).
- Evaluation (`data.eval_dataset`) still reports the student's cross-entropy, not the divergence to the teacher.
- Validated on CPU and small models in the test suite (losses against references, chunked equal to unchunked, two-process FSDP equal to one process, cache round trips, end-to-end runs); large-model GPU runs are pending.
