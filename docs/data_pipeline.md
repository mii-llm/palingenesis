# Data Pipeline

## Overview

The data pipeline converts raw chat conversations into training-ready tensors with precise assistant-only loss masking. It handles the full path from HuggingFace dataset streaming to GPU-ready batches.

```
[Optional] pgs prepare --config <same config as training>
    │  Score with target-model perplexity → classify → filter → select
    │  Dump: prepared/scored_data.parquet + prepared_meta.json
    │  (training auto-consumes this when preprocess.enabled: true)
    ▼
HF Dataset / local .jsonl / .json / .parquet / prepared dir
    │
    │  .skip(rank).take_every(world_size)   ← interleaved rank sharding
    │  .skip(worker).take_every(num_workers) ← DataLoader worker sharding
    │
    ▼
ChatDataset._process(example)
    │
    │  tokenizer.apply_chat_template(
    │      messages,
    │      return_assistant_tokens_mask=True,  ← key: HF returns per-token mask
    │      return_dict=True,
    │      truncation=True,
    │      max_length=max_seq_length,
    │  )
    │
    │  labels = input_ids.clone()
    │  labels[~assistant_mask] = IGNORE_INDEX   ← only train on assistant
    │  labels[attention_mask == 0] = IGNORE_INDEX ← never train on padding
    │
    ▼
[Optional] PackedDataset
    │  Concatenate sequences end-to-end
    │  Chunk into fixed-length blocks
    │  (Labels preserve per-token masking from upstream)
    │
    ▼
DataLoader (collate_fn)
    │  Pad to longest-in-batch (dynamic padding)
    │  Pin memory for async H2D transfer
    │
    ▼
Batch: {input_ids, attention_mask, labels} on GPU
```

## Assistant-Only Masking

### Primary method: `return_assistant_tokens_mask`

Modern HuggingFace tokenizers support `{% generation %}` / `{% endgeneration %}` markers in their Jinja2 chat templates. When these markers wrap assistant content, `apply_chat_template` can return a boolean mask indicating which tokens are assistant-generated.

```jinja
{# Typical template structure #}
{% for message in messages %}
    {% if message.role == 'assistant' %}
        {% generation %}{{ message.content }}{% endgeneration %}
    {% else %}
        {{ message.content }}
    {% endif %}
{% endfor %}
```

The returned `assistant_tokens_mask` is a list of 0s and 1s aligned with `input_ids`. We convert this directly to labels:

```python
labels = input_ids.clone()
labels[~assistant_mask] = -100  # IGNORE_INDEX
```

### Fallback: Role-boundary heuristic

For models whose templates lack `{% generation %}` markers, we fall back to tokenizing progressively:

```python
for i, message in enumerate(messages):
    prefix_up_to_i = apply_chat_template(messages[:i+1], tokenize=False)
    prefix_len = len(tokenizer(prefix_up_to_i)["input_ids"])

    if message["role"] == "assistant":
        labels[current_pos:prefix_len] = input_ids[current_pos:prefix_len]

    current_pos = prefix_len
```

This is slower (multiple tokenization calls per sample) and less precise (may include assistant header tokens in the loss), but works for any model.

### What gets masked (concrete example)

For a Llama 3.1 conversation:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there! How can I help?<|eot_id|>
```

Masking result:
```
[MASKED MASKED MASKED ... MASKED] system + user content
[TRAINED TRAINED TRAINED ... TRAINED] "Hi there! How can I help?<|eot_id|>"
```

Only the actual assistant text tokens have loss computed. Special tokens (header tags, BOS) are masked.

## Sequence Packing

### When to use

Packing is beneficial when your dataset has many short conversations (< 1/4 of max_seq_length). Without packing, each batch wastes GPU cycles on padding tokens.

### How it works

```
Sample A: [tok tok tok tok] (200 tokens)
Sample B: [tok tok tok tok tok tok] (400 tokens)
Sample C: [tok tok] (100 tokens)

Packed (max_len=512):
Block 1: [A_tokens... B_tokens... C_tokens... remaining_of_next...]
Labels:  [A_labels... B_labels... C_labels... ...]
```

Labels preserve their per-token masking from the upstream ChatDataset — packed sequences correctly train only on assistant tokens regardless of which original sample they came from.

### When NOT to use packing

- Long agentic traces (already close to max_seq_length)
- When you need per-sample loss tracking
- With Context Parallel (CP needs the full sequence for correct attention)

## Streaming & Sharding

### Why streaming?

Large SFT datasets (millions of conversations) don't fit in memory. Streaming:
- Downloads data on-demand
- No disk space needed for the full dataset
- Instant startup (no preprocessing wait)

### Rank sharding

With `world_size=8`:
- Rank 0 sees examples 0, 8, 16, 24, ...
- Rank 1 sees examples 1, 9, 17, 25, ...
- ...

This is interleaved (not contiguous) to maximize diversity within each rank's data stream. Combined with the dataset's shuffle buffer (10k examples), this ensures ranks see different data at each step.

### Shuffling

Both streaming and map-style (non-streaming) datasets are shuffled with `train.seed` — streaming via a 10k-example buffer, map-style via a full index permutation.

The one exception: a prepared dataset with `preprocess.strategy: curriculum` is **not** shuffled, so the easy→hard ordering computed during preparation reaches the model intact.

### DataLoader worker sharding

With `num_workers=4` on rank 0:
- Worker 0: examples 0, 32, 64, ...
- Worker 1: examples 8, 40, 72, ...
- Worker 2: examples 16, 48, 80, ...
- Worker 3: examples 24, 56, 88, ...

This composes with rank sharding so that globally across all ranks and all workers, every example is processed exactly once per epoch.

## Dynamic Padding vs Fixed-Length

We use **dynamic padding** (pad to longest in batch) rather than fixed-length padding because:

1. Agentic conversations have highly variable lengths (100-8000+ tokens)
2. Fixed-length wastes compute on short conversations
3. Dynamic padding adapts to the actual data distribution
4. Combined with `drop_last=True`, batch shapes are consistent per step

The tradeoff: `torch.compile` may see shape variation across batches and need to handle dynamic shapes. With per-layer compilation and `fullgraph=True`, this is handled efficiently by inductor's symbolic shape support.

## Performance Characteristics

| Component | Overhead | Bottleneck |
|-----------|----------|-----------|
| Tokenization | ~1ms per sample | CPU-bound |
| Chat template | ~2ms per sample | Jinja2 rendering |
| Collation | ~0.1ms per batch | Memory allocation |
| H2D transfer | ~0.5ms per batch | PCIe bandwidth |
| Prefetch (2x) | Hides above | Overlaps with GPU |

With `num_workers=4` and `prefetch_factor=2`, the data pipeline runs entirely in parallel with GPU computation. The GPU should never wait for data.
