# Qwen3.5-4B Agentic SFT — Single A100-80GB

## Model Profile

- **Architecture**: Hybrid (3:1 DeltaNet + Attention), 4B dense params
- **Vocab**: 151,936 tokens
- **Context**: 128K native (DeltaNet enables O(n) long context)
- **Key trait**: Only attention layers should be adapted (DeltaNet is destructive to fine-tune)

## Memory Budget (single A100-80GB)

```
Model (bf16):         4B × 2 = 8 GB
Muon optimizer:       4B × 4 = 16 GB  (no v buffer!)
Gradients (bf16):     4B × 2 = 8 GB   (only ~25% trainable with freeze_non_attention)
Activations (sel AC): ~8 GB (batch=2, seq=4096)
CCE logits:           ~0 GB (never materialized)
─────────────────────────────────────────
TOTAL:                ~40 GB → FITS with room for batch=2-4
```

With `freeze_non_attention: true`, only ~25% of params train (attention + norms + lm_head),
reducing effective optimizer memory to ~4 GB and gradients to ~2 GB.

## Key Design Decisions

1. **`freeze_non_attention: true`** — From arxiv:2604.22127: adapting DeltaNet layers is
   DESTRUCTIVE (-14.8pp). Only the attention pathway should be trained.

2. **`optimizer: muon`** — 50% less memory than AdamW + 1.5× faster convergence.
   Combined with freeze, we only Muon-optimize the attention weight matrices.

3. **`plugins.deft: true`** — Parameter-free adaptive loss. Automatically handles the
   heterogeneous token difficulty in agentic traces.

4. **`data.include_observations: true`** — ECHO: train on tool outputs to learn world model.
   Critical for agentic tool-use (model learns to predict what tools return).

5. **`train.base_merge: true`** + SLERP — Periodic pull-back toward pretrained weights.
   Prevents the attention-only training from drifting too far.

6. **No LLRD** — With `freeze_non_attention`, LLRD is redundant (frozen params ignored).
   All trainable attention blocks get equal LR.

## Preprocessing

```bash
# 1. Score data with target model
palingenesis prepare \
    --model Qwen/Qwen3.5-4B \
    --data your-org/agentic-traces \
    --output projects/qwen35_4b_agentic/prepared/ \
    --strategy optimal \
    --budget 10000 \
    --max_seq_length 4096

# 2. Train
torchrun --standalone --nproc_per_node=1 \
    -m palingenesis.train \
    --config projects/qwen35_4b_agentic/config.yaml
```

## Expected Results

- Training time: ~2-4 hours on single A100 (10K samples, 3 epochs)
- Val loss improvement: expect 15-25% reduction over base
- Tool-use accuracy: significant gains from ECHO + DEFT combination
- No catastrophic forgetting: base_merge + freeze ensures preservation
