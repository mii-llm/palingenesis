# Gemma 4 Training Guide

## Model Family Overview

| Model | Total | Effective | Architecture | Key Feature |
|-------|-------|-----------|--------------|-------------|
| E2B | 5.1B | 2.3B | Dense + PLE | Edge (phones, Jetson) |
| E4B | 8B | 4.5B | Dense + PLE | Consumer GPU (12-16GB) |
| 12B | 12B | 12B | Dense (standard) | Workstation |
| 26B-A4B | 26B | 4B | MoE | Efficient reasoning |
| 31B | 31B | 31B | Dense | Full power |

## Per-Layer Embeddings (PLE) — E2B/E4B

PLE is Gemma 4's unique architecture for small models. Instead of one embedding table:
- A **second embedding table** feeds each decoder layer a 256-dim signal
- Derived from token identity + evolving hidden state
- Each layer gets specialized per-token context
- The PLE table alone is ~4.7GB on E2B

**Training implications:**
- PLE params are like a second embedding — they're well-trained and fragile
- Aggressive LLRD protects PLE from being corrupted
- The "effective" param count (2.3B, 4.5B) is the compute per token
- The "total" count (5.1B, 8B) is what you need for memory planning (ALL params need gradients)

## The 262K Vocab Challenge

All Gemma 4 models use a 262,144 token vocabulary. This makes the logit tensor enormous:

| Batch | Seq | Logits (float32) | Chunks Needed |
|-------|-----|-------------------|---------------|
| 1 | 2048 | 2.1 GB | 2 |
| 1 | 4096 | 4.3 GB | 4 |
| 1 | 8192 | 8.6 GB | 8-16 |
| 2 | 4096 | 8.6 GB | 8-16 |
| 4 | 4096 | 17.2 GB | 16-32 |

**Chunked loss is MANDATORY** for Gemma 4. Without it, a single batch can OOM.

## Training Strategies by Goal

### 1. Style/Format Adaptation (minimal change)

For teaching output format (JSON, tool calls, structured responses) without
changing core capabilities:

```yaml
train:
  learning_rate: 5.0e-5   # Higher LR OK when only touching output layer
  epochs: 1
  llrd_decay: 1.0          # No LLRD needed
  freeze_non_attention: false
  # Alternative: freeze everything except lm_head
```

The "one tensor" approach (Gryphe's finding): training ONLY `lm_head` gives
style changes with near-zero forgetting. For our system, set
`llrd_decay: 0.0` and manually keep only lm_head trainable.

### 2. Agentic Capability (tool use, reasoning)

For teaching new behaviors (multi-step tool calls, agentic loops):

```yaml
train:
  learning_rate: 1.5e-5   # Conservative for E4B/12B
  epochs: 3                # Need multiple passes to learn complex behaviors
  llrd_decay: 0.9          # Protect early layers (general language)
plugins:
  info_sft: true           # Focus on informative tokens
  sym_noise: true          # Regularization
```

### 3. Long-Context Adaptation

Gemma 4 supports 128-256K context natively. For SFT on long agentic traces:

```yaml
data:
  max_seq_length: 16384    # Up to 32K on single A100
parallel:
  context_parallel: true    # For > 32K on multi-GPU
memory:
  loss_num_chunks: 32       # Must chunk aggressively with 262K vocab + long seq
```

## Model-Specific Configs

### E2B (5.1B total, edge model)

- Fits on 24GB GPU for inference, but training needs more (~45GB with optimizer)
- Use `cpu_offload: true` on single 24GB GPU
- Conservative: epochs=1, lr=5e-6, llrd=0.88
- PLE layers act like a second embedding — protect them

### E4B (8B total, consumer GPU model)

- Similar to training an 8B model
- Single A100-80GB works with selective AC + chunked loss
- LR: 1.5e-5 for agentic SFT
- The 262K vocab means CE is the memory bottleneck, NOT the model

### 12B (standard dense)

- Standard transformer — our normal optimization stack applies fully
- FSDP on 2+ GPUs recommended (24GB params + 96GB optimizer)
- LR: 1.5e-5, LLRD: 0.9
- Chunked loss: 16 chunks minimum (262K vocab!)
- Liger Kernel compatible (gemma2 architecture family)

## Key Differences from Llama/Qwen

| Feature | Llama 3.1 8B | Gemma 4 12B |
|---------|-------------|-------------|
| Vocab | 128,256 | 262,144 |
| Logit peak (B=1, S=8K) | 4.0 GB | 8.6 GB |
| Required chunks | 4-8 | 8-16 |
| Architecture | Standard transformer | Standard transformer |
| Context | 128K (RoPE) | 256K |
| Attention | GQA | GQA |
| Liger Kernel | Full support | gemma2 kernels |

| Feature | Qwen3-4B | Gemma 4 E4B |
|---------|----------|-------------|
| Total params | 4B | 8B |
| Effective | 4B | 4.5B |
| Architecture | Standard | Dense + PLE |
| Special | Thinking mode | PLE + multimodal |
| Vocab | 151,936 | 262,144 |

## Practical Tips

1. **Always use chunked loss**: 262K vocab is the #1 memory challenge
2. **Monitor PLE gradient norms**: If PLE embedding norms spike, reduce LR
3. **Gemma 4 is very good out-of-box**: Less SFT is often better. Start with 1 epoch and evaluate before doing more.
4. **Audio/vision tokens**: If your data includes multimodal content, ensure the model's vision/audio towers are properly initialized (use the `-it` variant)
5. **Apache 2.0**: No commercial restrictions — free to deploy

## Recommended Research-Backed Settings

Based on our training strategies document + model-specific considerations:

| Parameter | E2B | E4B | 12B |
|-----------|-----|-----|-----|
| LR | 5e-6 | 1.5e-5 | 1.5e-5 |
| LLRD | 0.88 | 0.9 | 0.9 |
| Epochs | 1-2 | 2-3 | 2-3 |
| Batch | 4 | 2 | 1 |
| Seq Length | 2048 | 4096 | 8192 |
| CE Chunks | 4 | 8 | 16 |
| SymNoise | alpha=7 | alpha=5 | alpha=5 |
| InfoSFT | yes | yes | yes |
| FSDP | no (single GPU) | optional | recommended |
