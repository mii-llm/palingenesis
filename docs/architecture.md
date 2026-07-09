# Architecture

## System Overview

palingenesis is a vertically-integrated SFT trainer built on PyTorch 2.12+. Every component is chosen for a single purpose: train agentic LLMs on long sequences as fast as possible with maximum memory efficiency.

```
                         ┌──────────────────────────────┐
                         │         torchrun             │
                         │  (elastic launch, multi-node) │
                         └──────────┬───────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────┐
              │              Process Group (NCCL)              │
              │         world_size = N_nodes * GPUs/node       │
              └────────┬────────────────────────┬─────────────┘
                       │                        │
            ┌──────────▼──────────┐   ┌────────▼──────────┐
            │   DeviceMesh "dp"   │   │  DeviceMesh "cp"  │
            │  (FSDP2 sharding)   │   │ (Context Parallel) │
            └──────────┬──────────┘   └────────┬──────────┘
                       │                        │
              ┌────────▼────────────────────────▼────────────┐
              │              Model (HuggingFace)              │
              │  ┌─────────────────────────────────────────┐ │
              │  │ Liger Kernel patches (RMSNorm, SwiGLU,  │ │
              │  │ RoPE, FusedCrossEntropy)                 │ │
              │  ├─────────────────────────────────────────┤ │
              │  │ Selective Activation Checkpointing       │ │
              │  │ (save SDPA + 50% matmuls, recompute rest)│ │
              │  ├─────────────────────────────────────────┤ │
              │  │ torch.compile per TransformerBlock       │ │
              │  │ (inductor backend, fullgraph=True)       │ │
              │  ├─────────────────────────────────────────┤ │
              │  │ FSDP2 fully_shard (bottom-up per layer)  │ │
              │  │ MixedPrecision(bf16 param, fp32 reduce)  │ │
              │  └─────────────────────────────────────────┘ │
              └──────────────────────┬───────────────────────┘
                                     │
              ┌──────────────────────▼───────────────────────┐
              │              Chunked CE Loss                  │
              │  Split [B,S,D] into N chunks along seq dim   │
              │  Per-chunk: lm_head projection + CE + bwd    │
              │  Accumulate grads, propagate through decoder  │
              │  Peak: O(B * S/N * V) instead of O(B * S * V)│
              └──────────────────────────────────────────────┘
```

## Data Flow

```
HF Dataset (streaming)
    │
    ▼
ChatDataset.__iter__()
    │  apply_chat_template(return_assistant_tokens_mask=True)
    │  Build labels: IGNORE_INDEX everywhere except assistant tokens
    │
    ▼
[Optional] PackedDataset
    │  Concatenate sequences, chunk into max_seq_length blocks
    │
    ▼
DataLoader (pin_memory, prefetch, drop_last)
    │  collate_fn: dynamic padding to longest in batch
    │
    ▼
[Optional] Context Parallel shard
    │  Split [B, S] -> [B, S/CP_degree] per rank
    │
    ▼
Forward pass (with autocast bf16)
    │
    ├─── Standard path: model(input_ids, attention_mask, labels) -> loss
    │
    └─── Chunked path: backbone(input_ids, attention_mask) -> hidden_states
                        ChunkedCELoss(hidden, labels, lm_head) -> loss
```

## Initialization Order

The order of operations during setup is critical:

```
1. setup_distributed()           # Process group, device assignment
2. apply_liger_kernel()          # BEFORE model load (monkey-patches HF classes)
3. AutoTokenizer.from_pretrained # Tokenizer
4. AutoModelForCausalLM.from_pretrained  # Model (uses patched classes)
5. apply_activation_checkpointing()      # BEFORE FSDP (wraps layers)
6. build_mesh() + apply_fsdp()           # Shard parameters
7. _compile_layers()                     # AFTER FSDP (compile sees sharded ops)
8. build_dataloader()            # Data pipeline
9. build_optimizer()             # Optimizer (sees FSDP-sharded params)
10. build_scheduler()            # LR schedule
```

This ordering is non-negotiable:
- Liger must patch before model instantiation (it replaces class definitions)
- AC must wrap before FSDP (FSDP needs to see the checkpoint wrappers)
- Compile must be after FSDP (it captures the distributed ops in the graph)
- Optimizer must be after FSDP (it sees DTensor parameters)

## Module Dependency Graph

```
train.py ─────┬──► config.py
              ├──► distributed.py ──► config.py
              ├──► context_parallel.py
              ├──► kernels.py
              ├──► data.py ──► config.py
              ├──► loss.py
              ├──► optim.py
              ├──► health.py
              ├──► checkpoint.py
              └──► logging.py ──► config.py
```

No circular dependencies. Each module is independently testable.
