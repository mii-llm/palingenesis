# Architecture

*How the pieces fit together, and why each one is there.*

---

## The training step

Every optimizer step in palingenesis follows this sequence:

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 236" role="img" aria-labelledby="fig-step-title fig-step-desc">
<title id="fig-step-title">The training step, stage by stage</title>
<desc id="fig-step-desc">Input: data is read, tokenized with the chat template, packed and prefetched to the GPU. Step: forward, loss, backward, gradient clipping and the optimizer step. Record: metrics are logged and checkpoints written, then the next batch starts.</desc>
<defs><marker id="step-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<text class="t-label" x="10" y="47">input</text>
<rect class="box" x="100" y="20" width="120" height="44" rx="4"/>
<text x="160.0" y="39" text-anchor="middle">data</text>
<text class="t-muted t-mono" x="160.0" y="55" text-anchor="middle">data.py</text>
<rect class="box" x="244" y="20" width="120" height="44" rx="4"/>
<text x="304.0" y="39" text-anchor="middle">tokenize</text>
<text class="t-muted t-mono" x="304.0" y="55" text-anchor="middle">chat template</text>
<path class="line" d="M220,42 H242" marker-end="url(#step-arrow)"/>
<rect class="box" x="388" y="20" width="120" height="44" rx="4"/>
<text x="448.0" y="39" text-anchor="middle">pack</text>
<text class="t-muted t-mono" x="448.0" y="55" text-anchor="middle">PackedDataset</text>
<path class="line" d="M364,42 H386" marker-end="url(#step-arrow)"/>
<rect class="box" x="532" y="20" width="120" height="44" rx="4"/>
<text x="592.0" y="39" text-anchor="middle">prefetch</text>
<text class="t-muted t-mono" x="592.0" y="55" text-anchor="middle">CUDAPrefetcher</text>
<path class="line" d="M508,42 H530" marker-end="url(#step-arrow)"/>
<text class="t-label" x="10" y="119">step</text>
<rect class="box-accent" x="100" y="92" width="104" height="44" rx="4"/>
<text x="152.0" y="111" text-anchor="middle">forward</text>
<text class="t-muted t-mono" x="152.0" y="127" text-anchor="middle">model</text>
<path class="line" d="M592.0,64 V78 H152.0 V90" marker-end="url(#step-arrow)"/>
<rect class="box-accent" x="220" y="92" width="104" height="44" rx="4"/>
<text x="272.0" y="111" text-anchor="middle">loss</text>
<text class="t-muted t-mono" x="272.0" y="127" text-anchor="middle">loss.py</text>
<path class="line" d="M204,114 H218" marker-end="url(#step-arrow)"/>
<rect class="box-accent" x="340" y="92" width="104" height="44" rx="4"/>
<text x="392.0" y="111" text-anchor="middle">backward</text>
<text class="t-muted t-mono" x="392.0" y="127" text-anchor="middle">autograd</text>
<path class="line" d="M324,114 H338" marker-end="url(#step-arrow)"/>
<rect class="box-accent" x="460" y="92" width="104" height="44" rx="4"/>
<text x="512.0" y="111" text-anchor="middle">clip</text>
<text class="t-muted t-mono" x="512.0" y="127" text-anchor="middle">AdaGC / norm</text>
<path class="line" d="M444,114 H458" marker-end="url(#step-arrow)"/>
<rect class="box-accent" x="580" y="92" width="104" height="44" rx="4"/>
<text x="632.0" y="111" text-anchor="middle">step</text>
<text class="t-muted t-mono" x="632.0" y="127" text-anchor="middle">optimizer</text>
<path class="line" d="M564,114 H578" marker-end="url(#step-arrow)"/>
<text class="t-label" x="10" y="191">record</text>
<rect class="box" x="100" y="164" width="120" height="44" rx="4"/>
<text x="160.0" y="183" text-anchor="middle">log</text>
<text class="t-muted t-mono" x="160.0" y="199" text-anchor="middle">HealthMonitor</text>
<path class="line" d="M632.0,136 V150 H160.0 V162" marker-end="url(#step-arrow)"/>
<rect class="box" x="244" y="164" width="120" height="44" rx="4"/>
<text x="304.0" y="183" text-anchor="middle">checkpoint</text>
<text class="t-muted t-mono" x="304.0" y="199" text-anchor="middle">checkpoint.py</text>
<path class="line" d="M220,186 H242" marker-end="url(#step-arrow)"/>
<path class="line-quiet" d="M364,186 H404" marker-end="url(#step-arrow)"/>
<text class="t-muted t-small" x="412" y="190">next batch</text>
</svg>
</div>
<figcaption>Every optimizer step runs the same stages in this order; each stage is one module (table below), so any of them can be swapped without touching the others.</figcaption>
</figure>

Each stage is a module. They compose orthogonally — you can swap any part without touching the others. Here's the map:

| Stage | Module | What it decides |
|-------|--------|----------------|
| Data | `data.py` | Chat template masking, packing, multi-source mixing |
| Tokenize | HuggingFace chat template + `data.ChatDataset` | Which tokens are trained: whole assistant turns (tool calls and end-of-turn included), per the template's own turn markers |
| Pack | `data.PackedDataset` + `packing.PackedBatch` | Whole conversations per sequence, each attending only to itself |
| Prefetch | `perf.CUDAPrefetcher` | Overlaps PCIe transfer with previous step's compute |
| Forward | HuggingFace model + `torch.compile` | The neural network computation |
| Loss | `loss.py`, `plugins.py` | CE (plain, chunked, Cut CE) or a token-gated objective (DEFT, DFT, InfoSFT, CADFT), chunked |
| Backward | PyTorch autograd + `memory.GradientRelease` | Gradient computation, optional fused optimizer step |
| Clip | `perf.AdaGC` or `torch.nn.utils.clip_grad_norm_` | Per-tensor adaptive or global clipping |
| Step | Optimizer + `optim.Hyperball` + `optim.MONAAcceleration` | Weight update with optional norm projection |
| Log | `health.HealthMonitor` + `logging.Tracker` | Tiered diagnostics, wandb/trackio |
| Checkpoint | `checkpoint.py` | Sharded DCP (FSDP) or safetensors (single GPU) |

---

## Memory model

Each optimization removes a different category of memory:

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 332" role="img" aria-labelledby="fig-mem-title fig-mem-desc">
<title id="fig-mem-title">Where GPU memory goes in training, and what each option removes</title>
<desc id="fig-mem-desc">Five categories of GPU memory. Model weights in bf16 take a fixed 2 bytes per parameter. Optimizer states take 4 to 8 bytes per parameter with AdamW and 1 byte per parameter with lion8bit, 75 to 88 percent less. Gradients normally take 2 bytes per parameter and nothing with gradient release, which fuses the optimizer step into the backward pass. Activations kept for the backward drop to about 30 percent of naive with selective activation checkpointing. The loss logits, batch by sequence by vocabulary, can be enormous, for example 16 GB in fp32 for a batch of 4 sequences of 4K tokens with a 262K vocabulary; the chunked loss holds one N-th of them at a time and Cut Cross-Entropy none.</desc>
<text class="t-label" x="10" y="20">memory category</text>
<text class="t-label" x="236" y="20">without → with the option</text>
<text class="t-label" x="492" y="20">option · effect</text>
<text x="10" y="56">Model weights</text>
<text class="t-muted t-small" x="10" y="72">bf16: 2 B/param</text>
<rect class="box-plain" x="236" y="44" width="240" height="26" rx="2"/>
<rect class="fill-ink" x="236" y="44" width="240.0" height="26" rx="2"/>
<text x="492" y="56">no option</text>
<text class="t-muted t-small" x="492" y="72">fixed: 2 B/param</text>
<line class="rule" x1="10" y1="86" x2="690" y2="86"/>
<text x="10" y="108">Optimizer states</text>
<text class="t-muted t-small" x="10" y="124">AdamW: 2 states, 4–8 B/param</text>
<rect class="box-plain" x="236" y="96" width="240" height="26" rx="2"/>
<rect class="fill-accent" x="236" y="96" width="30.0" height="26" rx="2"/>
<text x="492" y="108">lion8bit</text>
<text class="t-muted t-small" x="492" y="124">1 B/param: 75–88% saved</text>
<line class="rule" x1="10" y1="138" x2="690" y2="138"/>
<text x="10" y="160">Gradients</text>
<text class="t-muted t-small" x="10" y="176">normally 2 B/param</text>
<rect class="box-plain" x="236" y="148" width="240" height="26" rx="2"/>
<text x="492" y="160">gradient release</text>
<text class="t-muted t-small" x="492" y="176">0: fused into backward</text>
<line class="rule" x1="10" y1="190" x2="690" y2="190"/>
<text x="10" y="212">Activations</text>
<text class="t-muted t-small" x="10" y="228">kept for the backward</text>
<rect class="box-plain" x="236" y="200" width="240" height="26" rx="2"/>
<rect class="fill-accent" x="236" y="200" width="72.0" height="26" rx="2"/>
<text x="492" y="212">selective AC</text>
<text class="t-muted t-small" x="492" y="228">~30% of naive</text>
<line class="rule" x1="10" y1="242" x2="690" y2="242"/>
<text x="10" y="264">Loss logits [B, S, V]</text>
<text class="t-muted t-small" x="10" y="280">4 × 4K × 262K in fp32 = 16 GB</text>
<rect class="box-plain" x="236" y="252" width="240" height="26" rx="2"/>
<rect class="fill-accent" x="236" y="252" width="30.0" height="26" rx="2"/>
<text x="492" y="264">chunked loss / CCE</text>
<text class="t-muted t-small" x="492" y="280">1/N at a time / none</text>
<rect class="box-plain" x="10" y="302" width="22" height="12" rx="2"/><text class="t-small t-muted" x="38" y="312">without the option (each row to its own scale)</text>
<rect class="fill-accent" x="330" y="302" width="22" height="12" rx="2"/><text class="t-small t-muted" x="358" y="312">what remains with it</text>
</svg>
</div>
<figcaption>Each option removes a different category of memory, so they compose. Bars are schematic, each row to its own scale.</figcaption>
</figure>

They add up: a Qwen3.5-4B fine-tune with its linear-attention layers frozen fits in about 16 GiB (`pgs profile` shows the breakdown for any config).

---

## Distributed model

For multi-GPU, the architecture adds three layers:

### FSDP2 (data parallelism + sharding)

Each transformer layer is wrapped with `fully_shard()`. Parameters are sharded across GPUs; during forward/backward, all-gathers bring each layer's params to full and reduce-scatters send gradients back. Communication overlaps with compute because each layer is an independent FSDP unit.

Palingenesis adds two optimizations over vanilla FSDP2:

1. **Last-layer skip**: the final transformer layer doesn't reshard after forward, because FSDP would immediately re-gather it for backward. Saves one reshard + one all-gather per step.

2. **Root kept gathered through the step**: the root unit (embeddings, final norm, output head, tied or not) does not reshard after forward. The chunked losses apply the head outside the model's forward, after a root forward (`logits.final_hidden_states`) has gathered it; the gradients equal single-process training (`tests/test_fsdp_trainer_path.py`).

### Context Parallel (sequence sharding)

For sequences longer than one GPU's memory can handle, every sequence is split over a group of GPUs (`palingenesis.context_parallel`). Each attention layer all-gathers the sequence's K/V from the group and attends its own queries to them, causally, at their global positions (exact; head-tail load balancing). FSDP shards over the data and context dimensions flattened. See the multi-GPU guide.

### Global valid-token normalization

The loss is

$$
\mathcal{L} = \frac{\sum_{t} \ell_t}{N_{\text{valid}}}
$$

where the sum runs over the rank's trained tokens $t$, $\ell_t$ is each one's loss, and $N_{\text{valid}}$ (`global_valid_tokens`) is the count of trained tokens all-reduced across all DP ranks. This ensures correct gradient scale even when different ranks have different amounts of padding.

---

## The optimizer stack

When all features are enabled, the optimizer step is:

1. **MONA** (`mona.apply()`) augments the gradients with curvature-aware acceleration: $\tilde G = G + \alpha A$, where $A$ is an exponential moving average of the gradient differences $G_k - G_{k-1}$ and $\alpha$ the acceleration coefficient.
2. **Base optimizer** (`optimizer.step()`: Muon, Lion, AdamW) makes its standard weight update from the augmented gradients $\tilde G$.
3. **Hyperball** (`hyperball.step()`) projects each constrained matrix back to its hypersphere, $W \leftarrow R\, W / \lVert W \rVert$, where $R$ is the matrix's norm at the first step.

MONA adds information (curvature). The base optimizer turns that into a direction. Hyperball removes the radial component, keeping only the angular movement. The composition is clean because each stage operates on a different aspect of the update.

---

## Checkpoint design

Two modes, chosen automatically:

| Scenario | Format | Extra memory | Resume speed |
|----------|--------|:---:|:---:|
| Single GPU | HF safetensors (sharded 2 GB) | 0 | Fast (memory-mapped) |
| FSDP (multi-GPU) | Distributed Checkpoint (DCP) | 0 | Fast (each rank loads its shard) |

Final export is always HF-format (gathered to rank 0) so the output works with `from_pretrained` anywhere.

The `BestModelTracker` maintains a shadow copy at `output/best/` — updated whenever eval loss improves. Never purged. This is the checkpoint you deploy.

---

## Module dependency graph

```
train.py
├── config.py          (flat YAML → typed dataclass)
├── data.py            (ChatDataset, turn markers, PackedDataset, collation)
├── packing.py         (forward arguments that isolate packed documents)
├── logits.py          (output head incl. logit transforms, FSDP-safe hidden states)
├── loss.py            (CE, chunked CE, CCE, DEFT bridge)
├── optim.py           (build_optimizer, schedulers, Hyperball, MONA, SAGE)
├── distributed.py     (FSDP2, mesh, Context Parallel)
├── memory.py          (GradientRelease, SelectiveDiff)
├── health.py          (HealthMonitor, entropy tracking)
├── perf.py            (Prefetch, GC, AdaGC, SpikeDetector, EMA, SLERP)
├── plugins.py         (DEFT, DFT, InfoSFT, CADFT, PreRL, SymNoise; chunked gated loss)
├── seco.py            (SeCO / SpaCO chunk-wise training)
├── dpo.py             (preference optimization)
├── checkpoint.py      (save/load, DCP, auto-purge, BestModelTracker)
├── kernels.py         (Liger kernel patching, activation checkpointing)
└── logging.py         (Tracker: wandb + trackio)
```

No circular dependencies. Each module imports only from modules above it in this list (with the exception of `config.py` which everything reads from).
