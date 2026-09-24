# CLI Reference

*All commands available through the `palingenesis` entry point.*

---

## Commands

### train

The main training command. Usually launched via `torchrun` for proper distributed setup.

```bash
pgs train --config configs/quickstart.yaml
```

Equivalent to:
```bash
torchrun --standalone --nproc_per_node=1 -m palingenesis.train --config configs/quickstart.yaml
```

### distill

On-policy distillation: the student samples with its current weights, each prompt's teacher scores those exact tokens, and the student is pulled toward the teacher's distribution (full-vocabulary reverse KL for a teacher sharing the student's vocabulary, byte-aligned chunks across different tokenizers). Rollouts come from the trainer's own `generate` or from vLLM (`rollout.backend`); teachers run in-process or on a vLLM server; several teachers can each take their own prompt sources. Overrides: `--section.option value`, and `--teachers.<name>.option` / `--sources.<name>.option` for the named mappings.

```bash
pgs distill --config configs/distill_math.yaml                 # Qwen3-1.7B -> Qwen3-0.6B, vLLM rollouts
pgs distill --config configs/distill_xtok.yaml                 # teacher with another tokenizer
pgs distill --config configs/distill_multi.yaml                # one teacher per prompt source
pgs distill --config configs/distill_math.yaml --train.learning_rate 1e-6 --teachers.qwen3_1_7b.backend vllm
```

See the [On-Policy Distillation guide](../guides/distillation.md).

### distill-score

Annotate a multiple-choice pool with its teacher's own answers (`teacher_answer`, `teacher_correct`) so you can filter or reweight before training — pure KL distills the teacher's errors too. One batched forward per row, no generation. Scores the first `mcqa` source (or `--source NAME`) with that source's teacher; accepts the same overrides as `distill`.

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl
```

### autopilot

Zero-config autonomous training. Profiles hardware, sweeps LR, trains to completion.

```bash
pgs autopilot --model Qwen/Qwen3.5-4B --dataset my_data.jsonl --output ./out
```

### prepare

Score, filter, and select training data by difficulty. Dumps parquet (default) or jsonl, plus a `prepared_meta.json` provenance manifest.

=== "Config-driven (recommended)"

    Reads the **same YAML as training** — scoring model, dataset, and the `preprocess:` section all come from one file, and the usual `--section.field` overrides apply:

    ```bash
    pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
    pgs prepare --config cfg.yaml --preprocess.budget 5000 --preprocess.strategy curriculum
    ```

    Afterwards, training consumes the output automatically:

    ```bash
    pgs train --config cfg.yaml --preprocess.enabled true
    ```

=== "Standalone flags"

    ```bash
    pgs prepare --model Qwen/Qwen3.5-4B --data raw.jsonl --output prepared/ \
        --strategy optimal --format parquet --split train
    ```

See the [Data Preparation guide](../guides/data.md) for the full workflow.

### prepare-multi

Multi-source preparation with per-source scoring.

```bash
pgs prepare-multi --model Qwen/Qwen3.5-4B --sources sources.yaml --output prepared/
```

### diagnose

Run pre/post-training health checks.

```bash
pgs diagnose --config config.yaml --mode pre --json
```

### inspect

Show tokenization and masking for sample data.

```bash
pgs inspect --config config.yaml --num_samples 5
```

### validate

Validate masking correctness across many samples.

```bash
pgs validate --config config.yaml --num_samples 200
```

### profile

Memory for a config: parameters, optimizer states and gradients exactly, activations roughly. `--measure` runs two real optimizer steps on the GPU and reports the true peak.

```bash
pgs profile --config config.yaml --gpu 80
pgs profile --config config.yaml --measure
```

### monitor

Check status of a running training from its log file.

```bash
pgs monitor --log_file outputs/train.log --brief
```

### loss

Analyze a training loss curve.

```bash
pgs loss --log_file outputs/train.log
```

### version

Show installed versions.

```bash
pgs version
```
