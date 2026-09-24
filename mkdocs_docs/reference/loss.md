# Loss Functions

*The loss function determines how much each trained token counts. The default is cross-entropy; token-weighting objectives are opt-in.*

---

## DEFT

Dynamic Entropy Fine-Tuning (arXiv:2602.11424). Each token's cross-entropy is weighted by the model's own confidence: where the model's prediction is concentrated and disagrees with the target, the token's weight shrinks; where the prediction is diffuse, the token keeps (almost) its full cross-entropy weight.

```yaml
plugins:
  deft: true
```

No hyperparameters. Standard CE (α→0) and DFT (α=1) are its two ends. The paper reports large gains fine-tuning **base** math models on NuminaMath (Qwen2.5-Math-1.5B, MATH-500: 41.7 with CE, 61.4 with DEFT).

!!! warning "Measured here: it hurt when refining a post-trained model"
    Qwen3-0.6B (post-trained, non-thinking), 30k NuminaMath-CoT conversations, the paper's recipe (1 epoch, LR 5e-5, cosine, warmup 0.1, fp32 master weights), greedy evaluation:

    | | MATH-500 | GSM8K |
    |---|---|---|
    | Base model | 52.8 | 64.0 |
    | Cross-entropy | 35.0 | 55.3 |
    | DEFT | 30.2 | 48.5 |
    | DFT | 5.0 | 2.6 (collapsed into repetition) |

    All three degraded a model that is already good at math at this learning rate, and the token-weighted ones more. Measure on your own model and data before enabling them.

The mechanism: each token's cross-entropy is multiplied by a trust gate `p_t^α_t`, where `p_t` is the model's probability of the target token and `α_t = Σ_v p_v²` (the collision probability, i.e. exponentiated Rényi-2 entropy) measures how concentrated the prediction is; the gate is a stop-gradient weight, so the gradient on the target logit is `-p^α (1 - p)`. A diffuse prediction (α near 0) keeps the full cross-entropy gradient; a confident one (α near 1) behaves like DFT's `p_t` gate, suppressing tokens the model confidently disagrees with.

### Chunked token-gated objectives

With `memory.chunked_loss: true` (the default), DEFT, DFT, InfoSFT and CADFT are computed chunk by chunk along the sequence, never materializing the full [B, S, V] logits; the result equals the full-logit definitions (tests/test_chunked_gated_losses.py). InfoSFT and CADFT need batch statistics (InfoSFT's mean weight, CADFT's per-sequence NLL z-scores), which one extra no-grad pass over the chunks provides.

---

## Standard cross-entropy

The baseline. `loss = -log(p_correct) / num_valid_tokens`, averaged across the sequence.

Active when no plugin is enabled. Uses `sum` reduction + global valid-token normalization for correct distributed training.

---

## Cut Cross-Entropy (Apple)

Computes CE without ever materializing the logit tensor. Custom Triton kernel that computes only: (1) the dot product for the correct token, (2) the log-sum-exp over all vocab entries on-the-fly in SRAM.

Memory: O(1) instead of O(B×S×V). For Gemma (262K vocab, seq 8K, batch 4): saves 16 GB.

```bash
pip install cut-cross-entropy
```

Auto-activated when available and no plugin needs full logits (DEFT, DFT, InfoSFT need logits → chunked CE is used instead).

---

## Chunked cross-entropy

Splits hidden states into N chunks along the sequence dimension. For each chunk: project through lm_head → compute CE → backward → free logits. Never holds more than 1/N of the logit tensor at once.

FSDP-aware: disables lm_head reshard during the chunk loop (avoids N redundant all-gathers), coalesces reduce-scatter into the final chunk.

```yaml
memory:
  chunked_loss: true
  loss_num_chunks: 8   # Auto-tuned based on seq_len × vocab_size
```

---

## Loss normalization

All loss functions use `sum` reduction divided by `global_valid_tokens`:

```python
loss = sum(per_token_losses) / global_valid_tokens
```

Where `global_valid_tokens` is all-reduced across DP ranks. This is critical for distributed correctness — without it, ranks with more padding get inflated gradients.


---

*For the research behind these loss functions and how they interact with other techniques, see [Architecture → Research](../architecture/research.md).*
