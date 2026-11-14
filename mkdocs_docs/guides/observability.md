# Observability: Rollouts, Trackers, Tracing

What a run does, beyond its loss curve: the trajectories RL and on-policy distillation sample, the metrics every trainer logs, and where to send both. None of it adds a dependency, and all of it costs nothing when it is off.

!!! abstract "In short"
    - **Trajectory dumps** (`logging.dump_trajectories`, RL and `pgs distill`): sampled rollouts written by a background thread to compressed JSONL. `pgs rollouts` reads them: how episodes end, rewards and their components, tool errors and latency, loops, the slowest turns and the worst trajectories, with alarms.
    - **Trackers**: `pgs train`, `pgs rl` and `pgs distill` share one tracker: wandb, trackio, and your own backends through `logging.trackers` (an [MLflow backend](#mlflow) in about 40 lines).
    - **`pgs monitor` / `pgs loss`** read the logs of every trainer (`step=N loss=...` from `pgs train`, `step N | k=v ...` from `pgs rl` and `pgs distill`).
    - **vLLM request tracing** over OTLP, for rollout engines and teacher servers: [below](#vllm-request-tracing-otlp).

## Trajectory dumps

```yaml
logging:
  dump_trajectories: 10   # every 10th step (0 = off, the default)
  dump_groups: 8          # 8 groups of the batch, drawn at random (0 = every group)
  dump_content: true      # false: counts, timings, rewards and flags only (no prompts, texts, arguments, results)
```

The same three options exist in `pgs rl` (a group is a prompt's `rollout.group_size` rollouts; zero-variance groups are dumped too, marked `kept: false`) and in `pgs distill` (a group is a prompt with its rollouts; for agent traces, a trace with its regenerated turns). `dump_groups: 0` writes the whole batch, which is what `dump_trajectories` alone did before.

**Cost.** The trainer hands the writer references to the batch it already holds; serialisation, JSON encoding and compression run on one background thread. The queue holds at most 4 batches: when the disk falls behind, a batch is dropped instead of waiting, and `logging/dump_dropped` (in the step metrics) counts the dropped trajectories. A queued batch is written before the trainer exits, on SIGTERM too. Sampling with `dump_groups` keeps both the disk and the writer's CPU time small; a dump of every group of every step of a long agentic run can grow by gigabytes a day.

**Files.** `<output_dir>/rollouts/steps_00000000-00000099.jsonl.gz`, one file per 100 steps (`.rank<r>` before `.jsonl.gz` on several ranks), each dumped batch appended as its own gzip member: a crash can cut only the last batch short, and a resumed run that redoes a step appends it again (readers keep the newest copy). JSONL because a trajectory is nested and irregular, gzip because it is in the standard library:

```bash
zcat outputs/rl/rollouts/steps_*.jsonl.gz | jq -c 'select(.kind == "trajectory" and .finish == "length") | {step, reward, tokens: ."gen_ai.usage.output_tokens"}'
```

**Records** (schema version 1). Fields follow OpenTelemetry's [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai) (`gen_ai.*`) where they name the same thing, so an exporter to a tracing backend is a mapping; the conventions are still marked "Development".

| Field | |
|---|---|
| `schema_version`, `kind`, `trainer`, `step`, `rank`, `trajectory_id` | `kind`: `trajectory`, or `timeline` (RL: the engine's sequences in flight per engine step) |
| `finish` | `stop`, `length` (token budget), `turns` (turn limit), `env_done` (the environment ended it), `error` (the rollout failed) |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | prompt and sampled tokens |
| `logprob_mean` | the sampler's mean log-prob of the sampled tokens |
| `loop`, `repeated_tool_calls` | a repetition loop in the text, or one call (same arguments) made 3+ times |
| `prompt` | the input messages (with `dump_content`) |
| `turns[]` | per assistant turn: `content`, `reasoning_content`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`, `duration_s`, `logprob_mean`, `tools_s`, `observations` (results that answer no call: a malformed call's error, a text-protocol reply) |
| `turns[].tool_calls[]` | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `error`, `duration_s` |
| RL | `group`, `kept`, `scored`, `trained`, `reward`, `rewards` (components), `advantage`, `turn_count`, `tool_calls`, `tool_errors`, `context_tokens`, `version`, `start_s`, `duration_s`, `info` (the environment's and rewards' extras) |
| `pgs distill` | `group`, `source`, `teacher`, `version`, `teacher_logprob_mean` and `k1_mean` (sampled reverse KL per token; same tokenizer only), `meta` (agent traces: `turn`, `onset`, `error`, `weight`) |

In Python: `palingenesis.rollouts.read_records(directory, steps=None)` yields the records.

### pgs rollouts

```bash
pgs rollouts --dir outputs/rl                    # the output_dir, or its rollouts/ directory
pgs rollouts --dir outputs/rl --since 200        # dumped steps >= 200
pgs rollouts --dir outputs/rl --json             # for an agent
pgs rollouts --dir outputs/rl --threshold loop_share=0.05 --threshold length_share=0.1
```

It reports the finish-reason shares, reward and component means, the zero-variance share (RL), the sampled KL to the teacher (`pgs distill`), tool calls per tool (calls, error rate, latency percentiles), the loop share, the slowest turns and tool calls, and the worst trajectories (lowest reward, longest, highest KL) with the tail of their last answer. It exits with 1 when a share crosses its threshold:

| Alarm | Default | Usually means |
|---|---|---|
| `length_share` | 0.2 | episodes hit the token budget: rambling, loops, or a budget too small for the task |
| `turns_share` | 0.2 | episodes hit the turn limit: the policy does not finish |
| `error_share` | 0.02 | rollouts failed (environment, sandbox, parser): `info.error` in the dump says why |
| `loop_share` | 0.1 | repetition loops (see the AGENTS.md decision tree) |
| `tool_error_rate` | 0.3 | bad tool names or arguments, timeouts, a broken tool: the per-tool table says which |
| `zero_variance_share` | 0.5 | most groups carry no gradient: tasks too easy or too hard for the policy |

## Trackers

The three trainers log through `palingenesis.logging.Tracker` on rank 0. For `pgs rl` and `pgs distill`:

```yaml
logging:
  use_wandb: true
  use_trackio: true
  project: my-project
  run_name: ""          # empty: the backend names the run
  trackers: []          # further backends, below
```

Metric names are the ones in the log line; evaluations are logged as `eval/<name>`, and every row carries the training step as `train/global_step` (the x-axis in wandb, so a resumed run's redone steps stay where they belong). The run id is saved with each resumable checkpoint, so a resumed run continues the same run in every backend. A backend that fails to start or to log is warned about and skipped; training continues.

### Your own backend

`logging.trackers` takes `module:Class` or `path/to/file.py:Class` entries, imported only when configured. A backend is a class with three methods:

```python
class MyBackend:
    def __init__(self, *, project, name, config, run_id, output_dir, resuming): ...
    def log(self, metrics: dict[str, float], step: int | None) -> None: ...
    def finish(self, exit_code: int) -> None: ...  # non-zero: the run failed
```

`run_id` is stable across resumes; `resuming` says whether this process continues a run; `config` is the run's configuration as a dictionary. `log` runs on the training thread, so a backend that talks to a server should queue or batch its calls (as MLflow's asynchronous logging does below).

### MLflow

!!! warning "Untested recipe"
    This backend is not part of palingenesis and has not been run against an MLflow server; it is written against MLflow 3's documented API. Install `mlflow-skinny` (the tracking client without the server, UI and SQL store) and point `MLFLOW_TRACKING_URI` at your server.

```python
# mlflow_tracker.py: logging.trackers: ["mlflow_tracker.py:MLflowTracker"]
import json
from pathlib import Path

import mlflow


def _flat(config: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = str(value)[:6000]  # MLflow's limit on a parameter value
    return out


class MLflowTracker:
    def __init__(self, *, project, name, config, run_id, output_dir, resuming):
        mlflow.set_experiment(project)
        state = Path(output_dir) / "mlflow_run.json"  # palingenesis run id -> MLflow run id
        known = json.loads(state.read_text()) if state.exists() else {}
        existing = known.get(run_id) if resuming else None
        mlflow.start_run(run_id=existing, run_name=name or f"run-{run_id}")
        if existing is None:
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps({**known, run_id: mlflow.active_run().info.run_id}))
            mlflow.log_params(_flat(config), synchronous=False)

    def log(self, metrics, step):
        numbers = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
        mlflow.log_metrics(numbers, step=step, synchronous=False)  # queued: no network call on the step

    def finish(self, exit_code):
        mlflow.flush_async_logging()
        mlflow.end_run(status="FAILED" if exit_code else "FINISHED")
```

### Registering an exported checkpoint in MLflow

!!! warning "Untested recipe"
    Written against MLflow 3's transformers flavour; the flavour loads the model on the CPU to log it.

Every trainer exports a Hugging Face directory (`<output_dir>/final`). To log it to the run and register a version:

```python
import json

import mlflow
from transformers import pipeline

run_id = list(json.load(open("outputs/rl/mlflow_run.json")).values())[-1]  # the run the backend above made
generator = pipeline("text-generation", model="outputs/rl/final", device="cpu")
with mlflow.start_run(run_id=run_id):
    mlflow.transformers.log_model(generator, name="model", registered_model_name="my-policy")
```

## vLLM request tracing (OTLP)

!!! warning "Untested recipe"
    vLLM (0.26) emits OpenTelemetry traces of its requests (queue time, time to first token, end-to-end latency) when given an OTLP endpoint; OpenTelemetry already ships with vLLM, so nothing is installed. The pass-through below is how palingenesis hands options to vLLM; the traces themselves have not been checked with these trainers.

Start a collector with a UI, for example Jaeger:

```bash
docker run --rm -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
export OTEL_SERVICE_NAME=pgs-rollouts OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
```

The colocated RL engine (`rollout.backend: vllm`) takes engine arguments through `rollout.vllm_args`:

```yaml
rollout:
  backend: vllm
  vllm_args:
    otlp_traces_endpoint: grpc://localhost:4317
    # collect_detailed_traces: [model]   # per-step model timing: slower, for a short diagnosis only
```

A vLLM teacher server of `pgs distill` takes server flags through `teachers.<name>.vllm_args`:

```yaml
teachers:
  big:
    backend: vllm
    vllm_args: ["--otlp-traces-endpoint", "grpc://localhost:4317"]
```

A server you start yourself (`rollout.url`, `teachers.<name>.url`) takes `--otlp-traces-endpoint` on its own command line. The engine's own view complements the dump: `engine/*` metrics and the dump's `timeline` records say how full the engine was, the traces say how long each request waited.
