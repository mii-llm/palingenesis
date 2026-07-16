# Loss Functions

*The loss function determines which tokens the model pays attention to. The default (DEFT) is parameter-free and outperforms everything else.*

---

## DEFT (recommended)

Dynamic Entropy Fine-Tuning. Token weight is a function of the model's own confidence: tokens the model finds surprising get amplified, tokens it already knows get attenuated.

```yaml
plugins:
  deft: true
```

No hyperparameters. Subsumes standard CE (α→0) and DFT (α=1). The original paper (arxiv:2602.11424) reports gains on math and code reasoning, but these have not been independently reproduced.

The mechanism: `weight_t = -log(p_t)^α` where α is learned implicitly from the gradient dynamics. Hard tokens (low p_t) get exponentially more influence. Easy tokens (high p_t) contribute less noise.

### Chunked DEFT

When `memory.chunked_loss: true` and `plugins.deft: true`, palingenesis uses chunked DEFT — computes DEFT loss per sequence chunk without materializing the full [B,S,V] logit tensor. Identical numerical result, 1/N peak memory.

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
