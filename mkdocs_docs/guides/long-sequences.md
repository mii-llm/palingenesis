# Long Sequences (SeCO)

*Train on sequences far longer than your GPU could hold. The model runs chunk by chunk, keeping one chunk's activations at a time, and still gets the exact gradient.*

---

## The problem

Backpropagation keeps every layer's activations for the whole sequence until the backward pass. Activation checkpointing shrinks that, but memory still grows with length. The chunked loss removes the logits from the picture, not the activations. A single agentic-coding trace of 200k tokens is therefore out of reach on one GPU, even for a small model.

## The idea

Transformers already process long inputs in chunks at inference time: prefill with a KV cache. SeCO does the same for training. The idea comes from [Li et al., arXiv:2505.16710](https://arxiv.org/abs/2505.16710); [ChunkFlow (Yuan et al., arXiv:2503.02356)](https://arxiv.org/abs/2503.02356) describes the same scheme, and is where packing short sequences into chunks and not recomputing the last chunk come from. The implementation and its validation are palingenesis'.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 372" role="img" aria-labelledby="fig-seco-title fig-seco-desc">
<title id="fig-seco-title">SeCO: a no-grad pass that fills the cache, then a reverse sweep that recomputes one chunk at a time</title>
<desc id="fig-seco-desc">Stage 1 runs chunks 1 to 4 left to right without gradients and stores each chunk's K/V in a cache. Stage 2 runs from chunk 4 back to chunk 1. The chunk being processed, here chunk 3, is recomputed with gradients from the cached prefix, backpropagates its own loss plus the gradient that chunk 4 already sent into its K/V, and adds the gradient of its attention into the gradient buffer of the K/V of chunks 1 and 2. Bounded-state layers relay the gradient of the state at each chunk boundary instead.</desc>
<defs>
<marker id="seco-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="seco-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head-accent" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>
<!-- Stage 1 -->
<text class="t-label" x="10" y="20">Stage 1 · no grad · left to right</text>
<text class="t-muted t-small" x="10" y="60">chunks</text>
<rect class="box" x="130" y="36" width="120" height="40" rx="4"/><text x="190" y="61" text-anchor="middle">chunk 1</text>
<rect class="box" x="270" y="36" width="120" height="40" rx="4"/><text x="330" y="61" text-anchor="middle">chunk 2</text>
<rect class="box" x="410" y="36" width="120" height="40" rx="4"/><text x="470" y="61" text-anchor="middle">chunk 3</text>
<rect class="box" x="550" y="36" width="120" height="40" rx="4"/><text x="610" y="61" text-anchor="middle">chunk 4</text>
<path class="line" d="M250,56 H268" marker-end="url(#seco-arrow)"/>
<path class="line" d="M390,56 H408" marker-end="url(#seco-arrow)"/>
<path class="line" d="M530,56 H548" marker-end="url(#seco-arrow)"/>
<path class="line" d="M190,76 V104" marker-end="url(#seco-arrow)"/>
<path class="line" d="M330,76 V104" marker-end="url(#seco-arrow)"/>
<path class="line" d="M470,76 V104" marker-end="url(#seco-arrow)"/>
<path class="line" d="M610,76 V104" marker-end="url(#seco-arrow)"/>
<!-- K/V cache -->
<text class="t-muted t-small" x="10" y="122">K/V cache</text>
<text class="t-muted t-small" x="10" y="137">kept to the end</text>
<rect class="box-accent" x="130" y="106" width="120" height="30" rx="3"/><text class="t-mono" x="190" y="125" text-anchor="middle">K/V 1</text>
<rect class="box-accent" x="270" y="106" width="120" height="30" rx="3"/><text class="t-mono" x="330" y="125" text-anchor="middle">K/V 2</text>
<rect class="box-accent" x="410" y="106" width="120" height="30" rx="3"/><text class="t-mono" x="470" y="125" text-anchor="middle">K/V 3</text>
<rect class="box-accent" x="550" y="106" width="120" height="30" rx="3"/><text class="t-mono" x="610" y="125" text-anchor="middle">K/V 4</text>
<line class="rule" x1="10" y1="168" x2="690" y2="168"/>
<!-- Stage 2 -->
<text class="t-label" x="10" y="192">Stage 2 · with grad · right to left</text>
<path class="line-quiet" d="M190,136 V150 H450 V206" marker-end="url(#seco-arrow)"/>
<path class="line-quiet" d="M330,136 V150"/>
<text class="t-muted t-small" x="442" y="194" text-anchor="end">reads the cached prefix</text>
<text class="t-muted t-small" x="10" y="232">chunks</text>
<rect class="box" x="130" y="208" width="120" height="44" rx="4"/><text class="t-muted" x="190" y="228" text-anchor="middle">chunk 1</text><text class="t-muted t-small" x="190" y="243" text-anchor="middle">waiting</text>
<rect class="box" x="270" y="208" width="120" height="44" rx="4"/><text class="t-muted" x="330" y="228" text-anchor="middle">chunk 2</text><text class="t-muted t-small" x="330" y="243" text-anchor="middle">waiting</text>
<rect class="box-accent" x="410" y="208" width="120" height="44" rx="4"/><text x="470" y="228" text-anchor="middle">chunk 3</text><text class="t-small" x="470" y="243" text-anchor="middle">recompute, backward</text>
<rect class="box-quiet" x="550" y="208" width="120" height="44" rx="4"/><text class="t-muted" x="610" y="228" text-anchor="middle">chunk 4</text><text class="t-muted t-small" x="610" y="243" text-anchor="middle">done, freed</text>
<path class="line-accent" d="M550,230 H532" marker-end="url(#seco-arrow-a)" stroke-dasharray="4 3"/>
<text class="t-accent t-small" x="610" y="188" text-anchor="middle">state gradient at</text>
<text class="t-accent t-small" x="610" y="201" text-anchor="middle">the chunk boundary</text>
<!-- gradient buffer -->
<text class="t-muted t-small" x="10" y="326">∂L / ∂(K/V)</text>
<text class="t-muted t-small" x="10" y="341">gradient buffer</text>
<rect class="box-plain" x="130" y="308" width="120" height="30" rx="3"/><text class="t-mono" x="190" y="327" text-anchor="middle">∂ K/V 1</text>
<rect class="box-plain" x="270" y="308" width="120" height="30" rx="3"/><text class="t-mono" x="330" y="327" text-anchor="middle">∂ K/V 2</text>
<rect class="box-accent" x="410" y="308" width="120" height="30" rx="3"/><text class="t-mono" x="470" y="327" text-anchor="middle">∂ K/V 3</text>
<rect class="box-plain" x="550" y="308" width="120" height="30" rx="3"/><text class="t-mono" x="610" y="327" text-anchor="middle">∂ K/V 4</text>
<path class="line-accent" d="M440,252 V282 H190 V306" marker-end="url(#seco-arrow-a)"/>
<path class="line-accent" d="M330,282 V306" marker-end="url(#seco-arrow-a)"/>
<text class="t-accent t-small" x="198" y="276">adds its attention gradient</text>
<path class="line-accent" d="M500,306 V254" marker-end="url(#seco-arrow-a)"/>
<text class="t-accent t-small" x="508" y="284">relayed in</text>
<text class="t-muted t-small" x="130" y="362">Kept throughout: the K/V cache and its gradient, plus the activations of the one chunk being recomputed.</text>
</svg>
</div>
<figcaption>SeCO on four chunks. Stage 1 fills the cache without gradients; stage 2 recomputes one chunk at a time from the end, backpropagating its loss plus the gradient later chunks already sent into its K/V (or its boundary state), and adding its own attention gradient into the prefix's buffer.</figcaption>
</figure>

1. **Stage 1:** a no-grad pass over the chunks with a cache. It stops before the last chunk: nothing reads what that chunk would add to the cache.
2. **Stage 2:** a reverse sweep, starting with the last chunk. Each chunk is run with gradients from the cache, its loss is backpropagated, and the gradient that later chunks sent into its cache entries is relayed backwards.

Chunks run in reverse, so every relayed gradient is complete when its chunk is recomputed. The result equals full backpropagation.

**What it costs.** Counting a forward as 1 and a backward as 2, per chunk:

| Schedule | Work per chunk | Against the same checkpointing without SeCO, k chunks |
|---|---|---|
| Backpropagation without checkpointing | 3 | |
| Backpropagation with full activation checkpointing | 4 | |
| SeCO, no checkpointing inside the chunk | 3, plus 1 for every chunk but the last | (4k − 1) / 3k: 1.33× for long sequences, 1× for one chunk |
| SeCO with `gradient_checkpointing: full` inside the chunk | 4, plus 1 for every chunk but the last | (5k − 1) / 4k: 1.25× for long sequences, 1× for one chunk |

This is a count of operations, not a measurement: the one measured point is in the [`pgs fit` example](single-gpu.md), where SeCO with 8k chunks ran 22% slower than full checkpointing at 16k tokens (two chunks; measured when stage 1 still ran the last chunk). A sequence that fits in one chunk costs what plain backpropagation costs. Where full activation checkpointing fits your sequences, it is the cheaper choice; SeCO is for the lengths where it does not.

## Quickstart

```yaml
model:
  torch_dtype: float32          # recommended; see below
  compile: false

data:
  max_seq_length: 300000        # as long as your data needs
  packing: false                # true: several documents per step, each trained on its own (see Packing below)

memory:
  seco: true
  seco_chunk_size: 4096         # activation memory is set by this, not by the sequence

train:
  per_device_batch_size: 1
  gradient_checkpointing: full  # composes with SeCO: checkpointing inside each chunk
```

At start-up the trainer checks, on your actual model, that running it chunk by chunk with a cache gives the same logits as one full forward. That is the only assumption SeCO rests on. The trainer refuses to train if the check fails.

## Rows of very different lengths: packing

Agent traces range from a few hundred tokens to hundreds of thousands. Without packing, one row is one step: the number of tokens per optimizer step varies by orders of magnitude, a short row uses a small part of the GPU, and with `per_device_batch_size` above 1 every row is padded to the longest and the padding is run through every chunk.

With `data.packing: true`, the data pipeline fills blocks of up to `max_seq_length` tokens with whole documents, as it does without SeCO. SeCO then takes each block apart again:

- **A document longer than a chunk** runs as its own chunk sequence, exactly as an unpacked row would.
- **Shorter documents** run in batches of right-padded rows, similar lengths together, each batch of at most `seco_chunk_size` padded tokens (rows × the longest row), in one forward and one backward. A batch holds no more activations than a chunk.
- **Tokens after a document's last trained position** are not run: under causal attention they reach no loss. A trace that ends with a long tool result loses it for free; so does the block's padding.

Every document is a row of its own, from position 0 with a fresh cache. There is no attention across documents, no recurrent or convolution state carried over, and positions restart, whatever the layers are. This does not rely on the kernels that packing otherwise needs for linear-attention hybrids (the `hybrid` extra), and it also works for the architectures that plain packing refuses. The gradient is the sum of the documents' own full backpropagations, which the test suite checks on every architecture above, and against the trainer's ordinary packed forward.

The price is the padding inside the batches of short documents. It depends on how many short documents a block holds: grouping random length mixes the way SeCO does (a simulation, log-uniform lengths) gave 2–8% padding with 64 or more short documents per block and up to 30% with 4 to 16. `ChunkwiseResult.padding` reports it, and `scripts/bench_seco.py` prints it. The time per token of packed against unpacked SeCO has not been measured on a GPU yet.

Not with packing: SpaCO, off-policy distillation, and MoE router balancing (the balancing losses need every token's routing before the first backward; set `moe.balance: false` or leave packing off).

## Which models

SeCO works from the cache classes of the installed transformers, not from per-model code. Every cache layer is one of two kinds:

| Kind | Layers | How SeCO relays the gradient |
|---|---|---|
| Append-only | full attention | one K/V leaf per layer; each chunk's prefix is a view of it |
| Bounded state | sliding-window / chunked attention, linear attention (Gated DeltaNet), short convolutions (LFM2), and hybrids | the layer's state at every chunk boundary is a Markov state; the gradient is relayed boundary to boundary |

The test suite builds tiny random models of every architecture below. It checks that, whenever the chunked-forward check passes, SeCO's gradients equal full backpropagation, and that whenever it fails the model is rejected. It does this with eager and SDPA attention, on transformers 5.12 and 5.15.

| Result on transformers 5.15 | Architectures |
|---|---|
| Exact | Llama, Mistral (with and without sliding window), Qwen2, Qwen3, Qwen3-MoE, **Qwen3.5**, Qwen3-Next, Gemma 2, Gemma 3, **LFM2**, GPT-OSS, Phi-3, OLMo-2, OLMo-3, SmolLM3, Granite, GraniteMoE-hybrid, Cohere2, Mixtral, GLM-4, Starcoder2, EXAONE-4, GPT-NeoX, BLOOM, GPT-2, Falcon-H1, Nemotron-H |
| Rejected: the model's own cached forward differs from its full forward | Bamba, Jamba, Mamba, RecurrentGemma |

On transformers 5.12, LFM2, Falcon-H1 and GraniteMoE-hybrid were also rejected, for the same reason; newer versions fixed their cached forward.

Dropout and router jitter are replayed: each chunk's RNG state is saved in stage 1 and restored for its recomputation. Gradients therefore stay exact in training mode.

## Efficiency

- **The prefix is never copied.** Each full-attention layer keeps the whole sequence's K/V in one buffer. A chunk attends to it block by block, and to its own tokens causally; the parts are merged by their log-sum-exp, the exact combination flash and ring attention use internally. Backward runs per block from the merged output, so the gradients are exact and land straight in the prefix's gradient buffer. Memory that grows with length is then the K/V plus its gradient, 2x the K/V, instead of 4x.
- **`memory.seco_kv_offload`.** The same block-by-block reads let the K/V (and the recurrent start states) live in pinned CPU memory, streamed to the GPU one block at a time. Only the gradient buffer stays resident: 1x the K/V. It costs PCIe transfers (measured 1.2x to 2.5x slower) and needs about the K/V size in host RAM.

    !!! warning "Offloading is bounded by host RAM, not GPU memory"
        The stores are pinned, so the host cannot swap them. palingenesis refuses up front to hold more than 40% of the machine's RAM in K/V and says how much it would need. On a 117 GiB host, 28 GiB of pinned K/V trained fine, while 56 GiB left the machine unusable and the GPU waiting on page faults.
- **Grouped-query attention.** K/V heads go to the kernel as they are, never expanded to all query heads.
- **Cache precision.** K/V are cached in the autocast dtype, exactly as attention consumes them.
- **Logits.** The loss per chunk is the chunked cross-entropy, so logits never exist for more than one slice at a time.
- **The last chunk runs once.** Stage 1 stops before it, and the K/V stores end where it starts. That saves one forward per sequence: a fifth to a quarter of the work for a sequence of one chunk, a few percent at eight chunks (see the cost table above; counted, not measured).

### Measured on one A100 80GB

fp32 weights, bf16 autocast, full activation checkpointing, 4k chunks, real text, one forward+backward per row. Times in seconds, peak memory in GiB. These runs predate packing and the single run of the last chunk; neither changes the memory that grows with length, and the times are at most one chunk's forward lower now.

**Qwen3.5-0.8B** (hybrid: only 6 of 24 layers keep a K/V cache):

| Tokens | Full backprop | SeCO, before this | **SeCO** | **SeCO + `seco_kv_offload`** |
|---|---|---|---|---|
| 131k | 440 s · 60 GiB | 152 s · 31 GiB | not run | not run |
| 262k | not run (~116 GiB) | 327 s · 38 GiB | 333 s · 31 GiB | 397 s · 27 GiB |
| 524k | not run | 857 s · 53 GiB | 809 s · 38 GiB | 1079 s · 30 GiB |
| **1M** | not run | out of memory | **2154 s · 53 GiB** | **3124 s · 36 GiB** |

**Qwen3-0.6B** (dense: every one of its 28 layers keeps K/V, like 4B–8B dense models):

| Tokens | SeCO, before this | **SeCO** | **SeCO + `seco_kv_offload`** |
|---|---|---|---|
| 65k | 22 s · 35 GiB | 24 s · 25 GiB | 54 s · 18 GiB |
| 131k | 78 s · 64 GiB | 84 s · 39 GiB | 198 s · 25 GiB |
| 262k | out of memory | 311 s · 67 GiB | 733 s · 39 GiB |
| 524k | out of memory | out of memory | refused: needs 56 GiB of host RAM |

Memory grows exactly as the K/V accounting predicts. Qwen3-0.6B stores 112 KB of K/V per token:

| Path | Measured growth | = |
|---|---|---|
| Before | ~460 KB/token | ~4x the K/V |
| K/V store | 224 KB/token | 2x (store + gradient) |
| Offload | 112 KB/token | 1x (gradient only) |

**Exactness** (4,096 real tokens for Qwen3.5-0.8B, 2,048 for Qwen3-0.6B), gradients against the fp64 ground truth:

| | Qwen3.5-0.8B | Qwen3-0.6B |
|---|---|---|
| full backprop, fp32 | 2.99e-4 | 3.17e-5 |
| SeCO, fp32 | 2.98e-4 | 3.11e-5 |
| SeCO + offload, fp32 | identical to SeCO | identical to SeCO |
| full backprop, bf16 | 1.89% | 6.9% |
| SeCO, bf16 | 1.91% | 5.7% |

Offload is numerically identical to the GPU store in fp32; in bf16 the two differ only by the flash backward's own non-determinism.

The Qwen3.5 rows above use transformers' pure-torch linear attention. With flash-linear-attention installed (the `train` extra), its Triton kernels are used instead. They are much faster, but not bit-exact in fp32. On 14,428 real tokens (Qwen3.5-0.8B, fp32, 4k chunks), SeCO against full backprop, both with those kernels:

| | Gradient cosine | Relative L2 error | Time | Peak memory |
|---|---|---|---|---|
| Full backprop | | | 8.9 s | 31.5 GiB |
| SeCO | 0.99998 | 5.5e-3 | 9.4 s | 19.1 GiB |

The startup check then allows 2e-3 instead of 1e-4 in fp32. The first step of a run is slower while Triton tunes its kernels for each new shape. On dense models (Qwen3-0.6B, 14,055 tokens, fp32) SeCO stays exact: cosine 1.00000, relative error 3.7e-5, 22.4 s vs 20.1 s, 26.3 vs 52.5 GiB.

## Choosing `seco_chunk_size`

- **Larger chunks** mean fewer passes and better GPU utilisation, but more activation memory per chunk.
- **Smaller chunks** use less memory, but each chunk re-reads the whole prefix's K/V.
- **With `gradient_checkpointing: full`**, only one layer's recomputation is ever materialised, so larger chunks become affordable.
- **With full checkpointing, 2k–4k chunks** are a good default. 4k is faster; 2k uses about 7 GB less.
- **Without checkpointing, keep chunks around 1k.** Each chunk's full activations stay alive (about 8 GB per 1k tokens here). This was the fastest setting at 524k.
- **Rule of thumb:** memory ≈ model + optimizer + one chunk's activations + ~30 KB per token (Qwen3.5-0.8B, from the table above: 7 GiB more from 262k to 524k tokens, 15 GiB more from 524k to 1M; it scales with the model's K/V width and recurrent-state size).
- **With packing**, the chunk size is also the size of a batch of short documents: larger chunks batch more of them per forward.

## SpaCO (approximate, faster)

`memory.spaco_budget: t` recomputes only `t` random chunks per sequence instead of all `k`. Each visited chunk's loss and relayed gradients are scaled by `k/t`. The paper's pseudocode scales only the relayed gradients. We measured that this leaves the expected gradient shrunk by `t/k`, and with both scaled the expectation matches the true gradient (scale 0.92–0.99, cosine 0.98–0.9999). It is a stochastic estimate, not the exact gradient. Use it only when backward compute, not memory, is the bottleneck.

## Measured and not measured

Measured (the tables above, one A100 80GB): memory and time against length for two models, exactness against fp64, the K/V accounting. Checked by the CPU test suite on tiny random models: that every path gives the gradient of full backpropagation (every architecture, packed rows, offload, activation checkpointing, dropout replay).

Not measured on a GPU yet, and stated above as counts or estimates only: the time saved by running the last chunk once, the throughput of packed against unpacked SeCO on a real length mix, and how much of a small chunk's time is spent outside the kernels. `scripts/bench_seco.py` measures all three (its header lists the configurations), and SeCO against full backpropagation at the same length:

```bash
python scripts/bench_seco.py --model Qwen/Qwen3.5-0.8B --documents 64 --min-length 256 --max-length 131072 \
    --block 131072 --modes rows,packed,full --trained-share 0.1 --chunks 4096,8192 --json seco.jsonl
```

## Limits

- **Single GPU only.** Multi-GPU SeCO is not validated. For multi-GPU long context use `parallel.context_parallel`.
- **Packing** takes the documents apart again (see above); it is not combined with SpaCO, off-policy distillation or MoE router balancing. Without packing, rows may be right-padded, and the padding is run.
- **The K/V gradient stays on the GPU**, also with `seco_kv_offload`: for dense models it is what limits the length.
- **Incompatible features.** Plugins that need full-sequence logits (DFT, DEFT, InfoSFT, pre-RL), `rl_readiness`, `sym_noise`, `gradient_release`, and DPO. `Config.validate()` rejects them.
- **`torch.compile` is not validated with SeCO.**
