# Your First Real Training

*The quickstart used a public dataset. Here's how to train on your own data with production settings.*

---

## How much data do you need?

This depends on your goal. A rule of thumb (not a measured result; general chat ability tends to plateau after about a thousand samples, while math and code keep improving with more data, Dong et al. 2023):

| Dataset size | Epochs | Expected result |
|:---:|:---:|---|
| 500–2,000 samples | 3–5 | Format learning. Model follows your template but capabilities are limited. |
| 2,000–10,000 samples | 2–3 | Skill acquisition. Model learns specific behaviors (tool-calling patterns, coding style). |
| 10,000–50,000 samples | 1–2 | Deep specialization. Model becomes genuinely competent at the domain. |
| 50,000+ samples | 1 | Diminishing returns unless data is highly diverse. Consider `pgs prepare` to select the best subset. |

Repetition can beat new data at a fixed number of updates: Kopiczko et al. (2026, arXiv:2602.11149) trained base Qwen3/Olmo3 models on long chain-of-thought data and found 400 random samples for 128 epochs beat 51,200 samples for one epoch on AIME/GPQA, with gains saturating at 32–64 epochs. It is a result about repetition in long-CoT SFT of base models, not about quality versus quantity; for an instruct model, watch your regression evals when you repeat data.

---

## Prepare your data

JSONL with chat messages:

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Multi-turn is supported. Only assistant tokens get loss.

## (Optional) Score, filter and hold out an eval set

Point your config's `data.dataset` at the raw data, add a `preprocess:` section, and run:

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
```

```yaml
preprocess:
  enabled: true          # training will auto-use the prepared output
  output_dir: ./prepared
  budget: 5000           # a random subset of 5,000 (strategy: random, the default)
  eval_holdout: 200      # held-out eval set, never trained on
```

One forward pass over the data with the model you will train: 20,000 math conversations (10M tokens) took 7 minutes for Qwen3.5-0.8B on one A100, and the time grows with model size and token count. It drops outliers, records how familiar each response is to the model, and dumps `prepared/scored_data.parquet` plus a provenance manifest. It does not know which samples will teach the model most: no familiarity-based selection beat a random subset in our [measurements](../guides/data.md#what-the-measurements-say). With `enabled: true`, training picks it up automatically — no path editing needed. (The flag-based `pgs prepare --model ... --data ...` mode still exists; see the [Data guide](../guides/data.md).)

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

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 250" role="img" aria-labelledby="fig-pack-title fig-pack-desc">
<title id="fig-pack-title">Packing: three whole conversations in one sequence, each attending only to itself</title>
<desc id="fig-pack-desc">One packed sequence holds conversations A, B and C back to back, followed by padding. Position ids restart at 0 at the start of each conversation. Only the assistant tokens carry a training label. The attention mask is block-diagonal: each token attends causally to earlier tokens of its own conversation and never to another conversation.</desc>
<text class="t-label" x="10" y="20">one packed sequence (max_seq_length)</text>
<text class="t-muted t-small" x="10" y="61">conversation</text>
<text class="t-muted t-small" x="10" y="95">position_ids</text>
<text class="t-muted t-small" x="10" y="129">role</text>
<text class="t-muted t-small" x="10" y="163">trained</text>
<rect class="box" x="120" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="131.5" y="60" text-anchor="middle">A</text>
<text class="t-mono" x="131.5" y="95" text-anchor="middle">0</text>
<text class="t-mono t-muted" x="131.5" y="129" text-anchor="middle">u</text>
<rect class="box-quiet" x="120" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="146" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="157.5" y="60" text-anchor="middle">A</text>
<text class="t-mono" x="157.5" y="95" text-anchor="middle">1</text>
<text class="t-mono t-muted" x="157.5" y="129" text-anchor="middle">u</text>
<rect class="box-quiet" x="146" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="172" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="183.5" y="60" text-anchor="middle">A</text>
<text class="t-mono" x="183.5" y="95" text-anchor="middle">2</text>
<text class="t-mono t-muted" x="183.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="172" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="198" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="209.5" y="60" text-anchor="middle">A</text>
<text class="t-mono" x="209.5" y="95" text-anchor="middle">3</text>
<text class="t-mono t-muted" x="209.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="198" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="224" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="235.5" y="60" text-anchor="middle">A</text>
<text class="t-mono" x="235.5" y="95" text-anchor="middle">4</text>
<text class="t-mono t-muted" x="235.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="224" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="250" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="261.5" y="60" text-anchor="middle">B</text>
<text class="t-mono" x="261.5" y="95" text-anchor="middle">0</text>
<text class="t-mono t-muted" x="261.5" y="129" text-anchor="middle">u</text>
<rect class="box-quiet" x="250" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="276" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="287.5" y="60" text-anchor="middle">B</text>
<text class="t-mono" x="287.5" y="95" text-anchor="middle">1</text>
<text class="t-mono t-muted" x="287.5" y="129" text-anchor="middle">u</text>
<rect class="box-quiet" x="276" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="302" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="313.5" y="60" text-anchor="middle">B</text>
<text class="t-mono" x="313.5" y="95" text-anchor="middle">2</text>
<text class="t-mono t-muted" x="313.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="302" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="328" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="339.5" y="60" text-anchor="middle">B</text>
<text class="t-mono" x="339.5" y="95" text-anchor="middle">3</text>
<text class="t-mono t-muted" x="339.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="328" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="354" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="365.5" y="60" text-anchor="middle">C</text>
<text class="t-mono" x="365.5" y="95" text-anchor="middle">0</text>
<text class="t-mono t-muted" x="365.5" y="129" text-anchor="middle">u</text>
<rect class="box-quiet" x="354" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="380" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="391.5" y="60" text-anchor="middle">C</text>
<text class="t-mono" x="391.5" y="95" text-anchor="middle">1</text>
<text class="t-mono t-muted" x="391.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="380" y="148" width="23" height="22" rx="2"/>
<rect class="box" x="406" y="44" width="23" height="24" rx="2"/><text class="t-mono" x="417.5" y="60" text-anchor="middle">C</text>
<text class="t-mono" x="417.5" y="95" text-anchor="middle">2</text>
<text class="t-mono t-muted" x="417.5" y="129" text-anchor="middle">a</text>
<rect class="box-accent" x="406" y="148" width="23" height="22" rx="2"/>
<rect class="box-quiet" x="432" y="44" width="23" height="24" rx="2"/><text class="t-mono t-muted" x="443.5" y="60" text-anchor="middle">·</text>
<rect class="box-quiet" x="432" y="148" width="23" height="22" rx="2"/>
<rect class="box-quiet" x="458" y="44" width="23" height="24" rx="2"/><text class="t-mono t-muted" x="469.5" y="60" text-anchor="middle">·</text>
<rect class="box-quiet" x="458" y="148" width="23" height="22" rx="2"/>
<text class="t-muted t-small" x="10" y="200">u: user, system or tool result · a: assistant (trained)</text>
<text class="t-muted t-small" x="10" y="215">padding (·) carries no label</text>
<text class="t-label" x="500" y="20">attention mask</text>
<rect class="box-plain" x="500" y="44" width="154" height="154"/>
<rect class="fill-accent" x="500.5" y="44.5" width="10" height="10"/>
<rect class="fill-accent" x="500.5" y="55.5" width="10" height="10"/>
<rect class="fill-accent" x="511.5" y="55.5" width="10" height="10"/>
<rect class="fill-accent" x="500.5" y="66.5" width="10" height="10"/>
<rect class="fill-accent" x="511.5" y="66.5" width="10" height="10"/>
<rect class="fill-accent" x="522.5" y="66.5" width="10" height="10"/>
<rect class="fill-accent" x="500.5" y="77.5" width="10" height="10"/>
<rect class="fill-accent" x="511.5" y="77.5" width="10" height="10"/>
<rect class="fill-accent" x="522.5" y="77.5" width="10" height="10"/>
<rect class="fill-accent" x="533.5" y="77.5" width="10" height="10"/>
<rect class="fill-accent" x="500.5" y="88.5" width="10" height="10"/>
<rect class="fill-accent" x="511.5" y="88.5" width="10" height="10"/>
<rect class="fill-accent" x="522.5" y="88.5" width="10" height="10"/>
<rect class="fill-accent" x="533.5" y="88.5" width="10" height="10"/>
<rect class="fill-accent" x="544.5" y="88.5" width="10" height="10"/>
<rect class="fill-accent" x="555.5" y="99.5" width="10" height="10"/>
<rect class="fill-accent" x="555.5" y="110.5" width="10" height="10"/>
<rect class="fill-accent" x="566.5" y="110.5" width="10" height="10"/>
<rect class="fill-accent" x="555.5" y="121.5" width="10" height="10"/>
<rect class="fill-accent" x="566.5" y="121.5" width="10" height="10"/>
<rect class="fill-accent" x="577.5" y="121.5" width="10" height="10"/>
<rect class="fill-accent" x="555.5" y="132.5" width="10" height="10"/>
<rect class="fill-accent" x="566.5" y="132.5" width="10" height="10"/>
<rect class="fill-accent" x="577.5" y="132.5" width="10" height="10"/>
<rect class="fill-accent" x="588.5" y="132.5" width="10" height="10"/>
<rect class="fill-accent" x="599.5" y="143.5" width="10" height="10"/>
<rect class="fill-accent" x="599.5" y="154.5" width="10" height="10"/>
<rect class="fill-accent" x="610.5" y="154.5" width="10" height="10"/>
<rect class="fill-accent" x="599.5" y="165.5" width="10" height="10"/>
<rect class="fill-accent" x="610.5" y="165.5" width="10" height="10"/>
<rect class="fill-accent" x="621.5" y="165.5" width="10" height="10"/>
<text class="t-muted t-small" x="500" y="214">rows: queries · columns: keys</text>
<text class="t-muted t-small" x="500" y="229">causal, inside each conversation only</text>
</svg>
</div>
<figcaption>Packing three conversations into one sequence: position ids restart at every conversation, only assistant tokens are trained, and attention is block-diagonal, so no conversation sees another.</figcaption>
</figure>

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
