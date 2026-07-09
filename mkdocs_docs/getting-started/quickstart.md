# Quickstart

*From zero to a trained model in 5 minutes.*

---

## 1. Install

```bash
git clone https://github.com/your-org/palingenesis.git
cd palingenesis
uv pip install -e ".[train]"
```

## 2. Run

```bash
./run.sh configs/quickstart.yaml
```

The first run downloads the model (~6 GB) and starts streaming data. After 2-3 minutes of setup, you'll see:

```
step=1   loss=4.12 lr=0.00e+00 tok/s=0     grad_norm=1.23 dt=45.2s   ← first step: compile warmup
step=2   loss=3.89 lr=4.00e-06 tok/s=5842  grad_norm=0.89 dt=1.3s    ← normal speed
step=3   loss=3.74 lr=8.00e-06 tok/s=6011  grad_norm=0.76 dt=1.2s
...
```

!!! note "The first step is slow"
    `torch.compile` traces and compiles the model on the first forward pass. Steps 2+ are the real speed. This is normal and only happens once (cached for the session).

## 3. Use your own data

Edit `configs/quickstart.yaml`:

```yaml
model:
  name_or_path: your-org/your-model

data:
  dataset: path/to/your_data.jsonl
  messages_field: messages
```

Your data should be JSONL with chat messages:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

!!! tip "Better data in two commands"
    Before training on raw data, let palingenesis score and filter it with your target model — same config, one extra command:

    ```bash
    pgs prepare --config configs/quickstart.yaml
    pgs train --config configs/quickstart.yaml --preprocess.enabled true
    ```

    See the [Data Preparation guide](../guides/data.md) — it's the single highest-leverage step.

## 4. What happened?

Behind the scenes, palingenesis applied:

- **DEFT loss** — adaptive token weighting (harder tokens get more influence)
- **Power-decay LR** — theoretically optimal schedule
- **Chunked CE** — never OOMs on large vocabularies
- **Packing** — multiple conversations per sequence (2-3× throughput)
- **Best-model tracking** — saves the checkpoint with lowest eval loss

All without configuring anything.

## Next

- [Your first real training](first-training.md)
- [Single GPU optimization](../guides/single-gpu.md)
- [Autopilot mode](../guides/autopilot.md)
