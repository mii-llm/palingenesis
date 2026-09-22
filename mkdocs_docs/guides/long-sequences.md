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

- **Attention.** A chunk's queries attend to a longer prefix. For models configured with SDPA, SeCO attends through a lower-right causal bias with native grouped-query attention, which runs on the flash kernel with no mask. HF's generic path would build a `[chunk, prefix]` mask and expand K/V to every query head.
- **Cache precision.** K/V are cached in the autocast dtype, exactly as attention consumes them.
- **Logits.** The loss per chunk is the chunked cross-entropy, so logits never exist for more than one slice at a time.

### Measured on one A100 80GB

All runs use Qwen3.5-0.8B (hybrid: Gated DeltaNet + attention; torch kernels, no `fla`) with fp32 weights, bf16 autocast, real text and one sequence per step. Each is one forward+backward.

| Tokens | Full backprop + full checkpointing | SeCO, 2k chunks + full ckpt | SeCO, 4k chunks + full ckpt | SeCO, 1k chunks, no ckpt |
|---|---|---|---|---|
| 32k | 42 s · 19.3 GB | 42 s · 18.5 GB | 34 s · 27.5 GB | 44 s · 20.2 GB |
| 131k | 440 s · 64.6 GB | 175 s · 24.4 GB | 152 s · 33.0 GB | 185 s · 26.0 GB |
| 262k | not run (≈125 GB projected) | 363 s · 32.7 GB | 327 s · 40.3 GB | 336 s · 35.1 GB |
| 524k | not run | 871 s · 51.3 GB | 857 s · 56.5 GB | 713 s · 55.8 GB |
| 1M | not run | out of memory | out of memory | out of memory |

At 131k tokens SeCO is 2.4–2.9× faster than full backprop and uses 2.0–2.6× less memory, depending on the configuration. It trains 524k-token sequences. Full backprop's measured growth (≈0.46 MB per token) would need ≈125 GB already at 262k, beyond this 80 GB GPU; that length was not run. Memory still grows with length, because the attention K/V (and its gradient) and the per-chunk recurrent states are proportional to the sequence. For this model that is roughly 60 KB per token.

**Exactness** (same model, 4,096 real tokens):

- **fp64:** gradients match full backpropagation to 2–6·10⁻⁷ at every chunk size, which is the floor of the fp32 loss kernel.
- **fp32:** SeCO equals full backprop whenever both use the same loss chunk size.
- **bf16 autocast:** measured against the fp32 gradient, full backprop is off by 1.72% and SeCO by 1.73–1.93%. That is the same mixed-precision error.

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
