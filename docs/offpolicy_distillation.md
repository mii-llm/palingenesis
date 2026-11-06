# Off-Policy Distillation: Design Notes

How off-policy knowledge distillation is built into the trainer, and why. The user guide (configuration, examples, metrics) is [`mkdocs_docs/guides/offpolicy-distillation.md`](../mkdocs_docs/guides/offpolicy-distillation.md); the plugin API is in [plugins.md](plugins.md#trainer-plugins).

## 1. What existed (survey)

| Piece | Where | Reused for |
|---|---|---|
| On-policy distillation: HF and vLLM teachers, top-k prompt log-probs, `full_rkl` / `topk_kl` / `rs_kd` / `xtok` losses computed a slice of rows at a time | `palingenesis/opd/` | the slice-wise loss pattern; the shared-vocabulary bridge (`token_bridge`: prefix vocabularies, end-of-turn swap, probe-text identity check); `with_privileged` for privileged context; the vLLM prompt-logprob convention |
| Recorded-trace distillation (`trace_kd_weight`) | `opd/trace_trainer.py` | (an off-policy term limited to agent traces inside the on-policy trainer, one GPU) |
| SFT trainer: data mixing, chat masking, packing with per-document position ids, length grouping, pretokenized caches, FSDP2, activation checkpointing, compile, SeCO chunk-wise backward, chunked CE with FSDP-safe head handling, checkpoint / resume, best-model tracking | `train.py`, `data.py`, `loss.py`, `seco.py`, `distributed.py`, `checkpoint.py` | everything around the loss |
| `plugins.py` | research loss variants (DEFT, DFT, ...) behind booleans | (not an extension point) |

There was no offline distillation in the SFT trainer and no user extension point.

## 2. Decisions

**A mode of the SFT trainer, not a new trainer.** Off-policy KD is SFT with a different per-token target: the same fixed sequences, masks, packing, sharding, schedules, checkpoints. A parallel trainer would duplicate all of it and drift. `distill.enabled` switches the loss branch; teachers are built next to the student; the data pipeline gains three optional pieces (teacher views, source tags, a collate-time cache lookup).

**The loss is computed over the trained rows, a slice at a time** (`distill/chunked.py`). Only positions whose next token is trained enter the loss; for each slice the student's logits and the teacher's (projected from its hidden states, or gathered from its cached top-k) exist together and are differentiated at once. The student head's gradients accumulate through ordinary autograd (the FSDP-proven pattern of `chunked_cross_entropy_loss`: the root keeps the head gathered, one reduction), and the backbone gets its gradient through the same bridge node. Peak memory: one slice per side, independent of sequence length and vocabulary.

**Divergences on a common support** (`distill/losses.py`). Every divergence takes teacher and student log-probabilities over the same support with a mask: the full shared vocabulary (online teacher) or the cached top-k plus the gold token plus a tail bucket per side (`topk_mode: tail`). The tail construction keeps the quantity a divergence between two distributions on the same partition of the vocabulary; it is exact when the top-k holds the teacher's mass and degrades gracefully otherwise. Zero-probability entries contribute exactly zero (no `0 · inf`), so masked supports keep finite gradients.

**Content-addressed cache, looked up at collate time** (`distill/cache.py`). Keying documents by a hash of their token ids and loss mask makes the cache independent of everything that reorders or regroups data: shuffling, packing, length grouping, epochs, the number of ranks, resume. The lookup runs inside the DataLoader workers (`data._Collate`), splitting packed rows at position-id resets, so it costs the training loop nothing and works on the pretokenized path too. The format (per-shard binary arrays + a sorted index, memory-mapped) needs no database and resumes by shard.

**Teacher views align by trained tokens** (`distill/views.py`). The teacher's rendering is produced by the same `ChatDataset` masking as the student's, so both mark the same answer tokens as trained; alignment is "the i-th trained token of each rendering", verified token by token, with the student's sequence allowed to be a prefix (it may have been truncated). This survives any prefix change (system prompts, documents, few-shot examples, a different template) without reasoning about templates. Few-shot answers are marked `"loss": false` so they are context only. A cache built through a view stores targets at the student's positions, which is what lets cached context distillation pack.

**Routing by data source** uses per-token `source_ids` added by a thin dataset stage (`SourceTagged`) and carried through packing and collation; cached teachers are additionally routed by cache membership.

**Extension points are generic trainer hooks** (`palingenesis/hooks.py`), not distillation-specific, because the needs of quantisation-aware training (and of many other methods) are the same with or without a teacher. Model transforms run before float8 / activation checkpointing / FSDP / compile, so substituted modules and new parameters are sharded and checkpointed like the rest and parameter names stay those of the architecture (resume needs no special case). Exports go through `export_state_dict`, never resume checkpoints, so a plugin can fold weights for deployment while training state stays exact.

**Loss terms vs gradient terms.** A plugin loss term is scaled by `1 / (accumulation × data-parallel ranks)`, which is right for terms computed from this rank's activations. A weight regulariser computed on sharded parameters would need different scaling; the reference QAT plugin therefore adds its regulariser as a gradient in `on_before_optimizer_step`, elementwise on each rank's shard: exact under FSDP, free of activation memory, independent of the loss path.

## 3. One micro-batch

```
DataLoader worker: collate → CacheLookup (cached teachers: kd.<t>.ids / lp / gold / hit per position)
trainer:           shift labels; pack (flatten if a kernel needs it) or shard for CP; reshape the kd fields alike
                   Distiller.targets: rows = trained positions; per teacher
                       online: forward (its own packed layout or its view), keep final hidden states at rows (+ features)
                       cached: gather the rows of the kd fields
                       cover = routing & hits & view alignment
                   plugins.transform_targets
                   student forward (feature hooks capture)       ─┐
                   Distiller.loss: chunked over rows             │  or, under SeCO, per sequence chunk
                       slice: student logits, teacher logits / support, divergence, CE, token weights
                   + feature terms + plugin terms                ─┘
backward; at the last micro-batch: plugins.on_before_optimizer_step → clip → step → plugins.on_after_optimizer_step
```

## 4. Requirements from quantisation-aware distillation

The hook set was checked against the needs of recent low-bit training methods (read for requirements only; nothing here reproduces a specific recipe):

| Need | Methods that have it | Where it lands |
|---|---|---|
| Replace `nn.Linear` with a quantised module (STE, latent weights), keep names and ties | BitNet b1.58 (arXiv:2402.17764), ParetoQ (arXiv:2502.02631), TernaryLLM (arXiv:2406.07177), Hestia (arXiv:2601.20745) | `transform_model`; `TernaryLinear` reuses the Parameter |
| Insert modules (extra norms before output projections) | BitDistill (arXiv:2510.13998), BitNet b1.58 2B4T (arXiv:2504.12285) | `transform_model` |
| Quantise the embedding / a tied head | low-bit conversions that report whole-model bits | `TernaryEmbedding`, ternary head, tie preserved |
| Ramp quantisation in; anneal temperatures | Hestia (pressure ramp), continual QAT (arXiv:2502.11895) | `on_forward_start` + `Ramp`; step-dependent values as buffers (no recompiles) |
| Pull latents to the grid | latent QAT regularisers | `on_before_optimizer_step` gradient |
| Separate learning rate / eps / weight decay for latents and scales | ParetoQ, TernaryLLM, EfficientQAT (arXiv:2407.11062) | `param_groups` |
| Clamp latents after the step | latent QAT with bounded latents | `on_after_optimizer_step` |
| Logit KD with temperature, direction, JSD | BitDistill, LLM-QAT (arXiv:2305.17888), UPQ (arXiv:2506.09104) | built-in divergences inside the chunked loss (a plugin never needs full logits) |
| Hidden-state targets per layer | OneBit (arXiv:2402.11295), TernaryLLM | `features` / `feature_requests` |
| Attention-relation targets | BitDistill, MiniLMv2 (arXiv:2012.15828) | plugins on captured q/k/v at one layer (fused attention never materialises the maps) |
| Calibration before training (sensitivities, PTQ init) | Hestia, EfficientQAT | `on_train_start` (after resume); heavy calibration before training belongs in a script that writes the initial checkpoint |
| Fold quantised weights on export | all | `export_state_dict` (`folded` / `latent`) |
| Exactness under FSDP | — | DTensor-aware helpers in `palingenesis.qat`; two-process test |

## Quantisation-aware training while distilling

See the guide's section of the same name for the reference plugin, `palingenesis.qat.TernaryQAT`, and its hooks.

## 5. Tests

`tests/test_distill_losses.py` (divergences against reference formulas and `F.kl_div`, masks, tail and renorm supports, chunked == unchunked including gradients and downstream scaling, temperature², teacher mixing), `test_distill_cache.py` (format round trip, resume, refusals, padded and packed lookup, routing, HF builder against the teacher's own distribution, a view cache in student positions, training on a cache, misses), `test_distill_train.py` (end-to-end CPU runs of the trainer: online teacher, convergence towards the teacher, every divergence, views and their alignment, mismatch drops, routing with packing, SeCO equal to the full backward, validation, nested CLI overrides), `test_distill_distributed.py` (two gloo processes, FSDP student and sharded teacher, features and the QAT plugin, equal to one process), `test_qat.py`, `test_hooks.py`, `test_distill_memory.py`.

## 6. Open items

- Large-model GPU validation (throughput, memory against `pgs profile`, a quality comparison against SFT on teacher outputs).
- Cross-tokenizer off-policy KD (byte-chunk alignment as in on-policy `xtok`, or universal logit distillation) is not implemented.
- The SFT trainer's context parallelism shards inputs but does not dispatch attention to ring attention; distillation shards its fields consistently and will work once that path does.
- Per-teacher divergence and temperature (one setting for all teachers today).
- An evaluation of the divergence to the teacher on the held-out set (evaluation reports the student's CE).
