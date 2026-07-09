# Training Stability

Mechanisms that prevent training failures and ensure reliable convergence.

## Spike Detection (ZClip-Inspired)

**Paper**: "ZClip: Adaptive Spike Mitigation for LLM Pre-Training" (arxiv:2504.02507, Apr 2025)

**What**: Detects anomalous gradient norms via z-score and SKIPS the optimizer update entirely when a spike is detected.

**Why**: Loss spikes in SFT are caused by bad batches (corrupted data, extreme length outliers, formatting errors). A single spiked update can damage the model irreversibly. Detection + skip costs nothing (the forward/backward already ran) but saves the model.

**How**:
```
running_mean, running_var = EMA of gradient norms
z_score = (current_norm - running_mean) / std
if z_score > threshold: SKIP this update
```

**Config**:
```yaml
train:
  spike_detection: true       # Enable/disable
  spike_z_threshold: 5.0      # Z-score threshold (lower = more aggressive)
```

**Behavior**:
- First 50 steps: warmup (collects statistics, never skips)
- After warmup: skips if z_score > 5.0
- EMA decay: 0.99 (adapts to changing gradient scale)
- Only non-spike values update the running stats (prevents drift from outliers)

**When spikes are skipped**, the log shows:
```
step=1234 SPIKE SKIPPED (grad_norm=45.2, avg=1.85)
```

## Layer-wise Learning Rate Decay (LLRD)

**Paper**: "One LR Doesn't Fit All" (arxiv:2605.22297, NeurIPS 2025)

**What**: Each transformer layer gets a different learning rate. Early layers (general knowledge) get LOW LR. Late layers (task behavior) get HIGH LR.

**Why**: 
- Early layers learned robust language representations during pretraining
- Modifying them too much causes catastrophic forgetting
- Late layers need to adapt to the new task format
- Result: 1.5x faster convergence + less forgetting

**Formula**: `layer_lr = base_lr * decay^(num_layers - layer_idx)`

**Config**:
```yaml
train:
  llrd_decay: 0.9    # 1.0=off, 0.9=standard, 0.85=aggressive
```

**Effect on a 32-layer model with decay=0.9**:
| Component | LR Multiplier | Behavior |
|-----------|--------------|----------|
| Embedding | 0.04x | Nearly frozen |
| Layer 0 | 0.04x | Minimal change |
| Layer 8 | 0.10x | Slow adaptation |
| Layer 16 | 0.19x | Moderate |
| Layer 24 | 0.43x | Active learning |
| Layer 31 | 0.90x | Fast adaptation |
| lm_head | 1.00x | Full speed |

**When to use**:
- Always for SFT from instruct models (reduces forgetting)
- Especially useful for short runs (< 1 epoch) where forgetting is the main risk
- Set `0.85` for very aggressive anti-forgetting on small datasets

**When NOT to use**:
- Pretraining from scratch (all layers need equal updates)
- If model is very different from the SFT target (all layers need big changes)

## Gradient Clipping

Standard `max_grad_norm: 1.0` clips the total gradient L2 norm. This is the FIRST defense — prevents any single step from being too large. Spike detection is the SECOND defense — catches cases where the clipped norm is still anomalously high relative to recent history.

## Combined Flow

```
Gradient computed
    |
    v
[Clip to max_grad_norm=1.0]  -- hard cap, prevents numerical overflow
    |
    v
[Z-score check]              -- adaptive, catches relative anomalies
    |
    ├── Normal: optimizer.step()
    └── Spike: optimizer.zero_grad()  -- discard this update
```
