# Your First Real Training

*The quickstart used a public dataset. Here's how to train on your own data with production settings.*

---

## How much data do you need?

This depends on your goal, but here are research-backed guidelines:

| Dataset size | Epochs | Expected result |
|:---:|:---:|---|
| 500–2,000 samples | 3–5 | Format learning. Model follows your template but capabilities are limited. |
| 2,000–10,000 samples | 2–3 | Skill acquisition. Model learns specific behaviors (tool-calling patterns, coding style). |
| 10,000–50,000 samples | 1–2 | Deep specialization. Model becomes genuinely competent at the domain. |
| 50,000+ samples | 1 | Diminishing returns unless data is highly diverse. Consider `pgs prepare` to select the best subset. |

A surprising finding from the research: 400 high-quality samples trained for 128 epochs outperforms 51,200 samples for 1 epoch. **Quality dominates quantity.** If you have fewer than 5,000 samples, that's fine — just train longer.

---

## Prepare your data

JSONL with chat messages:

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Multi-turn is supported. Only assistant tokens get loss.

## (Optional but recommended) Score and filter

Point your config's `data.dataset` at the raw data, add a `preprocess:` section, and run:

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
```

```yaml
preprocess:
  enabled: true          # training will auto-use the prepared output
  output_dir: ./prepared
  budget: 5000
  strategy: optimal
```

Takes ~10 minutes for 50K samples. Removes bad data, selects the optimal difficulty mix, and dumps `prepared/scored_data.parquet` plus a provenance manifest. With `enabled: true`, training picks it up automatically — no path editing needed. (The flag-based `pgs prepare --model ... --data ...` mode still exists; see the [Data guide](../guides/data.md).)

## Choose your config

| GPU | Command |
|-----|---------|
| RTX 4090 / A100-40GB | `./run.sh configs/qwen35_4b/a100_40gb.yaml` |
| A100-80GB | `./run.sh configs/qwen35_4b/a100_80gb.yaml` |
| H100-80GB | `./run.sh configs/qwen35_4b/h100_80gb.yaml` |
| 8× A100/H100 | `./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml` |

## Edit the config

```yaml
data:
  dataset: my_data.jsonl        # raw data (prepared output is used when preprocess.enabled)
  eval_dataset: my_data.jsonl
  eval_split: test
```

## Output

```
output/
├── best/       ← Lowest eval loss (USE THIS)
│   └── model/
├── final/      ← Last step
└── step-*/     ← Periodic (auto-purged)
```

## Packing: when to use it

Packing (`packing: true`, off by default) places several whole conversations in one sequence of up to `max_seq_length` tokens. A conversation is never split across two sequences, and each one only attends to itself: `position_ids` restart at every conversation and the trainer passes the arguments that make every attention and linear-attention layer respect them.

Without packing, batches are already cut to the length of their longest row, and length-grouped batching (`length_group_buffer`, on by default) puts rows of similar length together, so little compute goes to padding. On short chat data (Qwen3-0.6B, perfectblend) the two measured about the same (packing somewhat slower with `sdpa`, which needs an explicit block-diagonal mask), so packing is not a free speedup. It pays off when rows are few and of very different lengths, or with `attn_implementation: flash_attention_2`, whose variable-length kernel needs no mask.

Qwen3.5 and other linear-attention hybrids need the `hybrid` extra to pack (see [Installation](install.md)); the trainer refuses otherwise.

---

## Load your model

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("output/best/model")
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Loss NaN | Reduce `learning_rate` by 3× |
| Loss stuck | Run `pgs prepare` to filter bad data |
| OOM | Reduce `per_device_batch_size` |
| Slow | Set `model.compile: true` |
