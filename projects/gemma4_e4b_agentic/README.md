# Gemma 4 E4B Agentic SFT — Single A100-80GB

## Model Profile

- **Architecture**: Dense + PLE (Per-Layer Embeddings), 8B total params, ~4.5B effective
- **Vocab**: 262,144 tokens (VERY large — logit memory is the #1 challenge)
- **Context**: 128K tokens
- **Key trait**: PLE table is ~50% of params and is FRAGILE (needs aggressive LLRD)
- **License**: Apache 2.0

## Memory Budget (single A100-80GB)

```
Model (bf16):         8B × 2 = 16 GB
Muon optimizer:       8B × 4 = 32 GB  (trainable only, with Muon)
Gradients (bf16):     8B × 2 = 16 GB
Activations (sel AC): ~8 GB (batch=1, seq=4096)
CCE logits:           ~0 GB  (without CCE: 262K × 4096 × 4 = 4.3 GB PER SAMPLE!)
─────────────────────────────────────────
TOTAL with Muon+CCE:  ~72 GB → TIGHT fit on 80GB (batch=1)
TOTAL with AdamW:     ~88 GB → DOES NOT FIT (needs FSDP)
```

**CCE is NON-NEGOTIABLE for Gemma 4.** The 262K vocabulary makes logit materialization
the single largest memory consumer. Without CCE, even batch=1 at seq=4096 uses 4.3GB
just for logits.

## Key Design Decisions

1. **`optimizer: muon`** — Without Muon, 8B model doesn't fit on single A100 (needs FSDP).
   Muon saves 32GB by eliminating the v buffer. Makes single-GPU training possible.

2. **CCE (chunked_loss)** — 262K vocab × 4096 seq = 4.3 GB per sample JUST for logits.
   CCE reduces this to ~0. This is the #1 memory saver for Gemma 4.

3. **`llrd_decay: 0.9`** — PLE layers act like a second embedding. They're fragile and
   well-trained. Layer 0 gets 0.9^26 = 7% of peak LR. Protects PLE from corruption.

4. **`plugins.deft: true`** — Adaptive loss that naturally handles the heterogeneous
   difficulty of agentic traces. Subsumes InfoSFT and DFT.

5. **`data.include_observations: true`** — ECHO world modeling. Train on tool outputs.

6. **`train.base_merge: true` (SLERP)** — Anti-forgetting via periodic merge-back to
   pretrained weights. Critical because PLE drift is hard to recover from.

7. **`per_device_batch_size: 1`** — Memory is TIGHT at 72GB. Batch=1 with GA=16
   gives effective batch of 16 (sufficient for stable training).

8. **No `freeze_non_attention`** — Gemma 4 is NOT a hybrid model. All layers are
   standard attention. Full training with LLRD protection.

## Preprocessing

```bash
# 1. Score data with Gemma 4 E4B
palingenesis prepare \
    --model google/gemma-4-E4B-it \
    --data your-org/agentic-traces \
    --output projects/gemma4_e4b_agentic/prepared/ \
    --strategy optimal \
    --budget 8000 \
    --max_seq_length 4096

# 2. Train (single A100)
torchrun --standalone --nproc_per_node=1 \
    -m palingenesis.train \
    --config projects/gemma4_e4b_agentic/config.yaml
```

## Critical Notes for Gemma 4

- **262K vocab is the memory bottleneck** — always use CCE or chunked loss (num_chunks≥8)
- **PLE layers are NOT MoE** — they're dense per-layer embeddings. Train them gently.
- **The "4B effective" is misleading** — for training you need memory for ALL 8B params
- **Liger kernel supports Gemma 4** — via the gemma2 architecture family
- **SymNoise alpha=5** — standard for 8B models (not too aggressive)

## Expected Results

- Training time: ~4-8 hours on single A100 (8K samples, 3 epochs, batch=1, GA=16)
- Peak memory: ~72 GB (tight but fits)
- Val loss improvement: 20-30% reduction over base
- Tool-use: significant gains from ECHO observation loss
- General knowledge: preserved via LLRD + base-merge SLERP
