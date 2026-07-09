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

Packing concatenates multiple short conversations into one long sequence (up to `max_seq_length`). This avoids wasting compute on padding tokens.

- **Use packing** (default, `packing: true`) when: most of your conversations are shorter than `max_seq_length`. This is almost always the case.
- **Disable packing** when: every conversation is already close to `max_seq_length` (e.g., long document summarization with 8K inputs). Packing provides no benefit when there's nothing to pack.

With packing, throughput typically improves 2-3× because the GPU processes useful tokens instead of padding.

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
