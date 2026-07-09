# Sequence Packing with FlexAttention

## Overview

Sequence packing eliminates padding waste by concatenating multiple conversations into single long sequences. FlexAttention provides the masking to prevent cross-document attention.

**Impact**: 30-60% throughput improvement for variable-length agentic traces (which range from 500 to 32K tokens).

## Current Status

Our `data.packing: true` config concatenates sequences end-to-end but relies on the model's standard causal attention. This means tokens from different documents CAN attend to each other — which is incorrect but often works in practice (the loss masking via `IGNORE_INDEX` prevents gradient flow).

For CORRECT packing, we need document-aware attention masking.

## Architecture (torchtitan approach)

torchtitan implements this via:

1. **Position tensor with resets**: `positions[i] = 0` marks the start of a new document
2. **`get_efficient_causal_mask_mod_for_packed_document(positions)`**: Creates a FlexAttention `mask_mod` that enforces document-level causal masking
3. **`create_block_mask()`**: Compiles the mask into a block-sparse structure
4. **`FlexAttention._compiled_flex_attn()`**: Runs the attention with the compiled mask

## Integration with HuggingFace Models

### Option A: `attn_implementation="flex_attention"` (transformers 5+)

Some HF models (Llama, Qwen, Gemma) support `flex_attention` natively:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    attn_implementation="flex_attention",
)
```

Then pass `position_ids` with resets at document boundaries:
```python
# Packed sequence: [doc1_tokens... | doc2_tokens... | doc3_tokens...]
# position_ids:    [0, 1, 2, ..., n | 0, 1, 2, ..., m | 0, 1, 2, ..., k]
```

### Option B: FlashAttention 2 with `cu_seqlens` (varlen)

Flash Attention 2 supports variable-length sequences natively via cumulative sequence lengths:

```python
# cu_seqlens = [0, len_doc1, len_doc1+len_doc2, total_len]
# Attention is computed per-document, no cross-document leakage
```

HF models with `attn_implementation="flash_attention_2"` can use this when inputs are properly prepared.

### Option C: Nested Tensors (PyTorch 2.6+)

PyTorch's nested tensors (`torch.nested.nested_tensor()`) represent variable-length sequences without padding. Combined with `torch.compile`, they can dispatch to FlexAttention automatically.

## Data Pipeline Changes for Packing

Our `PackedDataset` already concatenates sequences. To support document-aware attention, it needs to also produce:

1. **`position_ids`**: Reset to 0 at each document boundary
2. **`document_ids`** (optional): Integer ID per document for the mask function
3. **`cu_seqlens`** (for FA2): Cumulative lengths of each packed document

```python
# Current PackedDataset output:
{"input_ids": [tok1, tok2, ..., tokN],  # packed
 "labels": [lab1, lab2, ..., labN],      # with IGNORE at boundaries
 "attention_mask": [1, 1, ..., 1]}       # all ones (no padding)

# With document masking:
{"input_ids": [tok1, tok2, ..., tokN],
 "labels": [lab1, lab2, ..., labN],
 "attention_mask": [1, 1, ..., 1],
 "position_ids": [0, 1, 2, ..., a, 0, 1, 2, ..., b, 0, ...]}  # resets!
```

## Performance Comparison

| Method | Correctness | Speed | Compatibility |
|--------|------------|-------|---------------|
| No packing (padding) | ✓ | 1× (baseline, wastes compute on padding) | Universal |
| Packing without mask (current) | ~✓ (approximate) | 1.3-1.6× | Universal |
| Packing + FA2 varlen | ✓ | 1.4-1.8× | flash_attention_2 models |
| Packing + FlexAttention | ✓ | 1.5-2.0× | flex_attention models (PyTorch 2.5+) |

## Implementation Plan

1. **Phase 1** (done): Basic packing via `PackedDataset` — concatenation + loss masking
2. **Phase 2**: Add `position_ids` generation with document boundary resets
3. **Phase 3**: Support `attn_implementation="flex_attention"` with block mask creation
4. **Phase 4**: Benchmark and auto-select best backend per model

## When to Use Packing

- **USE** when: sequences vary widely in length (ratio > 3:1 between shortest and longest)
- **SKIP** when: all sequences are similar length (e.g., all ~8K agentic traces)
- **CAUTION**: Packing changes the effective batch composition — each "batch" contains variable numbers of documents
