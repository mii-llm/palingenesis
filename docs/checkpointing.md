# Checkpointing & Resume

## Save Format

```
checkpoints/step-500/
  model/                              # HF-compatible (from_pretrained works)
    model-00001-of-00008.safetensors  # 2GB shards
    model-00002-of-00008.safetensors
    ...
    model.safetensors.index.json      # Shard index
    config.json                       # Model config
    tokenizer.json                    # Tokenizer
  optimizer/                          # Sharded optimizer state
    param_groups.json                 # Group metadata
    shard_0000.safetensors            # States for params 0-49
    shard_0050.safetensors            # States for params 50-99
    ...
  training_meta.json                  # Training position
  rng_state.safetensors               # RNG for reproducibility
```

## Low-Memory Design

### Saving
- Model saved via HF `save_pretrained(max_shard_size="2GB")` — sharded safetensors
- Optimizer saved in chunks of 50 params each — no single massive file

### Loading (Resume)
- Model loaded shard-by-shard via safetensors memory mapping
- Peak RAM: model_size + 2GB (not 2x model_size)
- Optimizer loaded shard-by-shard

### FSDP2 (Multi-GPU)
- Uses PyTorch Distributed Checkpoint (DCP)
- Each rank saves/loads its own shard
- `full_state_dict=True` + `cpu_offload=True` for final model save

## Resume

```bash
# Explicit checkpoint
torchrun ... --train.resume_from ./checkpoints/step-500

# Auto-find latest
torchrun ... --train.resume_from auto
```

**What gets restored**:
- Model weights (from safetensors shards)
- Optimizer states (momentum, variance)
- LR scheduler position
- RNG states (CPU + CUDA)
- Training position (step, epoch, micro_step)

**Data fast-forward**: On resume, the dataloader iterates past already-processed micro-steps. For streaming datasets this is O(N) — acceptable for checkpoints every 500 steps.

## Final Model

After training completes, `save_final()` produces a clean HF model:
```
output_dir/final/
  model-00001-of-00008.safetensors
  model.safetensors.index.json
  config.json
  tokenizer.json
  tokenizer_config.json
```

Loadable with:
```python
model = AutoModelForCausalLM.from_pretrained("output_dir/final")
```
