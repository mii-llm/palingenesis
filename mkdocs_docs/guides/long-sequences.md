# Long Sequences (SeCO)

*Train on sequences far longer than your GPU could hold. The model runs chunk by chunk, keeping one chunk's activations at a time, and still gets the exact gradient.*

---

## The problem

Backpropagation keeps every layer's activations for the whole sequence until the backward pass. Activation checkpointing shrinks that, but memory still grows with length. The chunked loss removes the logits from the picture, not the activations. A single agentic-coding trace of 200k tokens is therefore out of reach on one GPU, even for a small model.

## The idea

Transformers already process long inputs in chunks at inference time: prefill with a KV cache. SeCO does the same for training. The idea comes from [Li et al., arXiv:2505.16710](https://arxiv.org/abs/2505.16710); the implementation and its validation are palingenesis'.

```
Stage 1 (no grad)      chunk 1 ──▶ chunk 2 ──▶ … ──▶ chunk k        builds the cache
                          │           │                 │
                        K/V₁        K/V₂              K/Vₖ          (+ recurrent states)

Stage 2 (backward)     chunk k ◀── … ◀── chunk 2 ◀── chunk 1         one chunk's graph at a time
                        recompute with grad, backprop its loss,
                        relay ∂L/∂(its K/V) into the chunks it attended to
```

1. **Stage 1:** a no-grad pass over all chunks with a cache.
2. **Stage 2:** a reverse sweep. Each chunk is recomputed with gradients from the cache, its loss is backpropagated, and the gradient that later chunks sent into its cache entries is relayed backwards.

Chunks run in reverse, so every relayed gradient is complete when its chunk is recomputed. The result equals full backpropagation, and the only cost is one extra no-grad forward.

## Quickstart

```yaml
model:
  torch_dtype: float32          # recommended; see below
  compile: false

data:
  max_seq_length: 300000        # as long as your data needs
  packing: false

memory:
  seco: true
  seco_chunk_size: 4096         # activation memory is set by this, not by the sequence

train:
  per_device_batch_size: 1
  gradient_checkpointing: full  # composes with SeCO: checkpointing inside each chunk
```

At start-up the trainer checks, on your actual model, that running it chunk by chunk with a cache gives the same logits as one full forward. That is the only assumption SeCO rests on. The trainer refuses to train if the check fails.

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

### Measured on one A100 80GB

fp32 weights, bf16 autocast, full activation checkpointing, 4k chunks, real text, one forward+backward per row. Times in seconds, peak memory in GiB.

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
- **Rule of thumb:** memory ≈ model + optimizer + one chunk's activations + ~60 KB per token (Qwen3.5-0.8B; it scales with the model's K/V width and recurrent-state size).

## SpaCO (approximate, faster)

`memory.spaco_budget: t` recomputes only `t` random chunks per sequence instead of all `k`. Each visited chunk's loss and relayed gradients are scaled by `k/t`. The paper's pseudocode scales only the relayed gradients. We measured that this leaves the expected gradient shrunk by `t/k`, and with both scaled the expectation matches the true gradient (scale 0.92–0.99, cosine 0.98–0.9999). It is a stochastic estimate, not the exact gradient. Use it only when backward compute, not memory, is the bottleneck.

## Limits

- **Single GPU only.** Multi-GPU SeCO is not validated. For multi-GPU long context use `parallel.context_parallel`.
- **No packing.** SeCO works on whole sequences. Rows may be right-padded.
- **Incompatible features.** Plugins that need full-sequence logits (DFT, DEFT, InfoSFT, pre-RL), `rl_readiness`, `sym_noise`, `gradient_release`, and DPO. `Config.validate()` rejects them.
- **`torch.compile` is not validated with SeCO.**
