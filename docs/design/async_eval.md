# Asynchronous in-training evaluation (design)

Status: approved with amendments and implemented (CPU-tested; GPU check `scripts/gpu_checks/async_eval.sh` not
run yet). Base: main `beb4882`.

## 0. Amendments and how it was built (read this first)

Owner's amendments:

1. **`transport: checkpoint` is in v1 and was built first.** `pgs eval watch <output_dir>` evaluates the run's
   checkpoints. The trainer only leases a due checkpoint (`.eval_pending`). DCP checkpoints are consolidated by the
   evaluator from the `model.*` keys. Push would need 2 bytes per parameter pinned on rank 0 (140 GB for 70B), so
   large models have an async path only through checkpoints.
2. **Cost is stated, never claimed zero.** At a due step, push copies the weights into the staging area on the
   training thread. In `pgs train` under FSDP it first gathers the full state to rank 0, as `save_final` does. That
   time is logged as `eval/async/snapshot_seconds` and measured by the GPU check.
3. **MIG is the only way to share a physical GPU**, with no reserved-budget mode. Metrics stay informational. With
   `transport: checkpoint`, `eval_tasks.select: <task>/<mode>/<metric>` writes `best_tasks.json` (higher is better)
   and the purge keeps that step. `final_timeout_s` defaults to 3600, and 0 returns at once. One service per run,
   with `service*.json` and per-step `result.json` left open to several. The host-memory refusal points to
   `transport: checkpoint`.
4. **One GPU:** async is refused with the message of section 3, which also names `pgs eval watch`.

Where the implementation differs from the sections below:

- **No push receiver refactored out of `EngineWorker`.** The evaluator's receiver is `palingenesis.eval.push.Receiver`
  (two host areas, newest wins). `EngineWorker` only reports its GPU UUIDs (`state()["gpus"]`).
- **No tee of the per-step gather in `pgs rl` under FSDP.** A due step gathers once more (`_served_weights`, the
  same path as `_push_weights`).
- **No side-stream overlap of the copy.** The snapshot is synchronous at due steps (`snapshot_seconds`). The overlap
  stays a possible optimisation if the GPU check shows the cost matters.
- **Results reach the trainer two ways:** from `step_<n>/result.json` on a shared file system, and as `result`
  frames over the push connection without one. The spec travels in the push `hello`, so `pgs eval serve` needs no
  training config.
- **The evaluator's engine runs in `rl.pool.worker` processes** (`EnginePool(updatable=True)`) for vLLM / SGLang,
  and in-process for `hf`. An OOM restarts it smaller twice, then the step fails; three failed steps in a row stop
  the evaluator.
- **Not built yet:** multi-role `role:` (async is refused with scenarios), several services per run, a per-rank shard
  push, and the NCCL link.
- **Prior art:** only the verified citations went into the public guide (TRL server mode, LLM Foundry `AsyncEval`,
  W&B `define_metric`). Section 13 below is the unverified design-time list.

Files as built: `eval/placement.py`, `eval/handoff.py`, `eval/service.py` (`watch`, `EvalService`),
`eval/push.py` (`serve`, `PushClient`, `Receiver`), `eval/async_eval.py` (`AsyncEval`, `full_weights`). Changes
are in `eval/config.py`, `eval/engines.py`, `eval/cli.py`, `rl/trainer.py`, `opd/trainer.py`, `train.py`,
`checkpoint.py`, `opd/trainer.py:prunable_checkpoints`, `rl/pool/buckets.py` and `rl/pool/worker.py`, plus
`scripts/train_slurm.sh` (`EVAL_NODES`) and `scripts/eval_slurm.sh`. Tests: `tests/test_async_eval_placement.py`,
`tests/test_async_eval.py`, `tests/test_async_eval_push.py` and `tests/test_async_eval_distributed.py`.

## 1. Problem and goals

`eval_tasks:` blocks training in every trainer today:

- `pgs rl` (`rl/trainer.py` `_eval_tasks`) and `pgs distill` (`opd/trainer.py` `_eval_tasks`) take the pipeline lock,
  sync the newest weights into the rollout engine, wake it, run `TrainingEval.run`, and put it back to sleep. No
  rollout and no optimizer step happens meanwhile. Multi-role scenarios skip `eval_tasks`.
- `pgs train` (`train.py`, `run_on_model`) generates with the model in training through `HFRollout`, on one process
  only. Multi-GPU runs skip evaluation with a warning.

Goal: an evaluator on GPUs or nodes that do not train, fed with the trainer's weights, that never makes a training
step wait and never touches the trainer's GPU memory. Configured in a few lines for one node, several nodes and
Slurm. The default (`placement: inline`) keeps today's behaviour exactly.

Non-goals for v1: evaluator results that drive training decisions (early stopping, best checkpoint); several
evaluator services per run; pipeline-parallel `pgs train`.

## 2. Decisions at a glance

| Question | Decision | Reason |
|---|---|---|
| Placement | `placement: inline` (default, today's code) or `async` | the default must not move |
| Where async runs | always a **separate process on GPUs the trainer does not use** | memory and SMs are never shared |
| Same GPU | **refused**, checked by GPU UUID; MIG instances count as distinct GPUs | a reserved memory budget does not stop SM contention: the run would slow by construction |
| Transport | `push` (default) in-memory weights over TCP reusing `rl/pool`; `checkpoint` (later phase) reads saved checkpoints | push works for every trainer and decouples the evaluation cadence from `save_steps`; FSDP `pgs train` checkpoints are DCP, not HF |
| Backlog | latest wins: at most one snapshot in flight from the trainer, one pending in the service | bounded memory, never blocks, newest weights evaluated |
| Process model | one `palingenesis.eval.service` process; launched by rank 0 (`devices:`) or by the user / Slurm (`pgs eval serve`, `service: auto`) | one code path, two ways to start it |
| Rendezvous | `<output_dir>/eval_tasks/service.json` on the shared filesystem, or an explicit `host:port` | Slurm jobs already need a shared `output_dir` |
| Results | the service writes `eval_tasks/step_<n>/` (same files as today) and returns metrics; rank 0 logs them at step `n` | the tracker already plots against `train/global_step` (`logging.py`, `define_metric`) |

## 3. Configuration

New keys of `EvalTasksConfig` (`eval/config.py`); every existing key keeps its meaning.

```yaml
eval_tasks:
  tasks: [gsm8k, my_task.yaml]
  every: 200
  limit: 200
  modes: [plain, think]
  placement: inline        # inline (default): today's behaviour | async
  # async: exactly one of devices / service
  devices: [6, 7]          # GPUs of rank 0's node, indices in the launch environment's view (CUDA_VISIBLE_DEVICES of
                           # the shell that ran torchrun, else nvidia-smi order); rank 0 starts the service on them
  service: ""              # "auto": a `pgs eval serve` found through <output_dir>/eval_tasks/service.json;
                           # "host:port": one at that address
  transport: push          # push | checkpoint (phase 5)
  engine:                  # the evaluator's engine, as `pgs eval`'s engine: section
    backend: vllm
    gpu_memory_utilization: 0.9
    max_model_len: 8192
    replicas: 1            # engines per evaluator GPU
    tensor_parallel_size: 1
  final_timeout_s: 3600    # after the last step: how long the trainer waits for the final evaluation (0: not at all)
  push_gbps: 0             # cap on the push rate (0: none); for links shared with the training's NCCL traffic
  role: ""                 # multi-role RL: the role whose weights are evaluated (default: the policy adapter)
```

Validation (`EvalTasksConfig.errors()` plus a start-up check, before any model load):

- `async` needs exactly one of `devices` / `service`; `inline` refuses both, and refuses `engine`, `transport`, `push_gbps`.
- `devices` must not intersect the GPUs of the trainer ranks on that node, the colocated rollout engine (same GPU as
  its rank) or a launched `vllm_server`. The check uses GPU UUIDs gathered from every rank at start-up
  (`all_gather_object` once), so `CUDA_VISIBLE_DEVICES` remapping, Slurm cgroups and `--virtual-local-rank` cannot fool
  it. `rollout.backend: remote` workers report their UUIDs in `hello` (a new `gpus` field in `EngineWorker.state()`),
  and a service on an engine's GPU is refused too: an engine that samples every step is a GPU in use.
- `transport: checkpoint` needs `train.save_steps > 0`, and `every` a multiple of it.
- `pgs train` with `parallel.pp_degree > 1`: refused in v1 (stages hold different parameters).

### Minimal configs per scenario

**One GPU.** Async is refused; inline stays the default:

```yaml
eval_tasks: {tasks: [gsm8k], every: 200, limit: 100}
```

`placement: async` here stops at start-up with:

```
eval_tasks.placement async needs a GPU the trainer does not use, and GPU 0 (GPU-3f2c...) trains.
On one GPU: placement: inline (training pauses for the evaluation), split the GPU with MIG and give the evaluator
the second instance, or evaluate the saved checkpoints afterwards with `pgs eval`.
```

**One node, 8 GPUs: 6 train, 2 evaluate.**

```yaml
eval_tasks: {tasks: [gsm8k, ifeval], every: 100, placement: async, devices: [6, 7]}
```

```bash
torchrun --nproc_per_node=6 -m palingenesis.train --config sft.yaml    # ranks take cuda:0..5
```

A `pgs rl` run with `rollout.backend: remote` engines on GPUs 4-5 trains on 0-3 and evaluates on 6-7 the same way;
the UUID check catches an overlap with the engines.

**Two nodes, node 1 evaluates** (one allocation, training on node 0):

```yaml
eval_tasks: {tasks: [gsm8k], every: 100, placement: async, service: auto}
```

```bash
# node 1
pgs eval serve --dir runs/rl            # all visible GPUs, one engine each; writes runs/rl/eval_tasks/service.json
# node 0
torchrun --nproc_per_node=8 -m palingenesis.rl --config rl.yaml
```

**Slurm.** `scripts/train_slurm.sh` gains `EVAL_NODES` (default 0): the last `EVAL_NODES` nodes of the allocation run
`srun --nodes=1 --nodelist=<node> pgs eval serve --dir <output_dir>` in the background, the others run torchrun as
today (`NNODES` shrinks accordingly). The config says `service: auto`.

```bash
EVAL_NODES=1 sbatch --nodes=3 scripts/train_slurm.sh rl.yaml        # 2 nodes train, 1 evaluates
sbatch scripts/eval_slurm.sh runs/rl                               # or a separate allocation (new script)
```

Without a shared filesystem: `pgs eval serve --port 7100 --results-dir <local dir>` and `service: host:7100`.

## 4. Process model

```
 trainer node                                               evaluator GPUs / node
 ┌──────────────────────────────────────────┐               ┌────────────────────────────────────────────┐
 │ rank 0 training thread                   │               │ palingenesis.eval.service                  │
 │  step n due: snapshot -> pinned staging  │               │  receiver thread: buckets -> host area     │
 │  (sync copy, one area)  ──hand-off──┐    │  TCP frames   │   (2 areas: incoming / ready, latest wins) │
 │  each step: drain results queue,    │    │  checksummed  │  eval loop: ready -> engine.update_weights │
 │  tracker.log(m, step=n)             │    │  buckets      │   -> Evaluator(tasks) -> step_<n>/ files   │
 │ AsyncEvalClient thread (own loop) ◄─┘────┼──────────────►│   -> "result" frame back                   │
 │  send, heartbeat, reconnect, results ◄───┼───────────────┤  engine: in-process (1 GPU) or EnginePool  │
 │ ranks 1..N: join the gather only (FSDP)  │               │   of rl.pool.worker (several GPUs)         │
 └──────────────────────────────────────────┘               └────────────────────────────────────────────┘
```

**One process type, two launchers.** The service is always another OS process: the trainer never imports vLLM for it,
never creates a CUDA context on an evaluator GPU, and vLLM's AutoConfig registrations (`hf_configs.py`) stay out of
the trainer.

- `devices:` rank 0 starts it with `subprocess.Popen([sys.executable, "-m", "palingenesis.eval.service", ...],
  start_new_session=True)` (as `eval/engines.py` `EnginePool` does), with `CUDA_VISIBLE_DEVICES` set to the evaluator
  GPUs and torchrun's variables removed (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_*`, `TORCHELASTIC_*`, `GROUP_RANK`),
  `os.nice(10)` and `OMP_NUM_THREADS` capped, so its CPU work yields to the trainer's data loaders. On Linux it sets
  `PR_SET_PDEATHSIG`; it also exits when the trainer's connection has been gone for `idle_timeout` (default 300 s).
  A service that dies is restarted up to 3 times with back-off (a Popen: no cost to the trainer).
- `service:` the user, Slurm or another torchrun starts `pgs eval serve`. It takes no training config: everything it
  needs (model path, tokenizer, chat-template arguments, stop tokens, the `eval_tasks` section with dataset paths made
  absolute, the seed, `output_dir`) arrives in the trainer's `hello`. The engine starts at the first `hello`
  (minutes, never on the training path) and is reused across reconnects of the same run.

**Rendezvous.** The service writes `service.json` atomically (`{host, port, token, gpus: [uuid], pid, started}`); the
trainer reads it at each due step until it connects (never waits for it: due steps before then are recorded as
skipped). The random `token` must come back in `hello`, so a stray trainer of another run cannot push. A `hello` also
carries the model signature (`protocol.model_signature`) and context length; a mismatch is refused, as for pool
workers.

**Lifecycle.** start -> `hello` (identity, UUID check, spec) -> engine ready -> step 0 (no push when the run starts
from `model.policy` without weight-transforming plugins: the service evaluates the checkpoint it loaded) -> due steps
-> end of training: the trainer stages the final weights (waiting for a free staging area here is allowed: training
is over), saves `final`, then waits up to `final_timeout_s` for the final result, logs it, sends `run_end` and, for a
service it launched, `shutdown`. A user-launched service keeps running unless started with `--exit-with-run`.

**Health.** A heartbeat (`state` frame) every 10 s. The trainer logs `eval/async/alive`, `lag_steps`, `skipped`,
`failed`, `push_seconds`, `snapshot_seconds`. Nothing in the client raises into the training thread: every failure
becomes a log line and a metric.

## 5. Weight transport

### Push (default)

Trainer side (`eval/async_eval.py`, `AsyncEvalClient`):

1. At a due step, rank 0 checks `client.ready()` (no snapshot in flight, service connected), non-blocking. Under
   FSDP the answer is broadcast to the ranks (one int, at due steps only), because the gather is a collective.
2. If ready, the served weights are copied into **one** pinned host staging area: the same iterator that feeds
   rollout engines and checkpoints (`plugins.rollout_weights`, adapter folding, `checkpoint_named_parameters`), cast
   to bf16, packed as checksummed buckets (`rl/pool/buckets.py`). If not ready: no gather, no copy, `skipped += 1`.
3. The sender thread sends the buckets over TCP (`protocol.send`), capped by `push_gbps`, with a push timeout
   (600 s). The area is free again once the service acknowledges receipt (not evaluation).

Cost to the trainer, all at due steps only:

| Trainer | Synchronous part | Extra GPU memory |
|---|---|---|
| `pgs rl` / `pgs distill`, FSDP | none extra: the step's own `_push_weights` gather is teed into the staging area (D2H copy of each tensor as it streams) | none |
| `pgs rl` / `pgs distill`, one rank / replicated | D2H of the weights, on a side stream overlapped with the next forward/backward; the optimizer step waits on its CUDA event (GPU-side) | none |
| `pgs train`, one process | as above | none |
| `pgs train`, FSDP | one gather pass (as `save_final`, one parameter at a time) + D2H | one full parameter at a time, as every save already needs |

Host memory: one staging area = 2 bytes per parameter (16 GB for an 8B model) of pinned memory on rank 0, allocated
once at start-up, after checking it fits in half of `MemAvailable` (refused otherwise, suggesting `transport:
checkpoint`). With `rollout.backend: remote` this adds to the pool's two areas. The NCCL link of `rl/pool` is **not**
used: it keeps the snapshot on the trainer's GPU, which is memory the trainer did not plan for.

Versioning: the version is the training step. Results carry it; the service refuses a push older than one it holds.

Service side: a `WeightReceiver` factored out of `EngineWorker` (`rl/pool/worker.py`: staging areas, bucket verify)
with two host areas, *incoming* and *ready*. A complete, verified push becomes *ready*, replacing an older ready one
that was not started (latest wins: its step is reported as `skipped`). Between two evaluations the eval loop loads
*ready* into the engine under the receiver's lock; an evaluation runs whole on one version. For several evaluator
GPUs, `EnginePool.update_weights` (now `NotImplementedError`) forwards to its local workers through a blocking
`RemotePool`, which is fine inside the service.

### Checkpoint (phase 5)

`transport: checkpoint`: the trainer only tells the service "step n is saved"; the service loads
`<output_dir>/step_<n>` (`pgs rl` / `pgs distill`: HF format; `pgs train` one process: `step-<n>/model`; `pgs train`
FSDP: DCP, consolidated by the service into a local HF directory from the model's keys only). Zero trainer cost
beyond the saves the user already pays, the evaluation follows `save_steps`, and it also serves `pgs eval watch
<output_dir>` (no trainer connection at all: a separate Slurm job evaluating checkpoints as they appear, as LLM
Foundry's async eval does). Purging (`train.keep_checkpoints`) must not delete a step the service has not loaded:
the trainer writes `step-<n>/.eval_pending`, the purge skips directories with one younger than 2 h, the service
removes it after loading. Backlog: `latest` (default) or `all` (checkpoints are on disk anyway).

## 6. Sequence of one evaluation (push)

```
trainer rank 0                         client thread                  service
step 400 done, due
 ready()? yes ─────────────────────►
 snapshot -> staging (0.2-2 s)
 hand-off (returns) ───────────────►  send buckets (paced) ──────────► receive into incoming, verify
step 401 ... trains                                                   ◄── ack "received v400" (area free)
                                                                     incoming -> ready (v400)
                                                                     loop: load ready into engine (s)
                                                                     Evaluator: tasks x modes (minutes)
step 437 ... trains                                                  write eval_tasks/step_400/*.jsonl,
                                      ◄──────────── result v400 ──── metrics.json, done
step 438: drain queue ◄──────────────  enqueue
 tracker.log(metrics, step=400)
 log eval/async/lag_steps=38
```

## 7. Failures

| Failure | What happens to training | What happens to evaluation |
|---|---|---|
| Service not up yet / never starts | nothing | due steps logged as skipped; a warning every 10 skips |
| Service slower than `every` | nothing (no gather or copy at skipped steps) | latest wins; `skipped` counts the steps passed over |
| Evaluator OOM (vLLM start or generate) | nothing (other GPUs) | the service restarts its engine with `max_num_seqs` halved, then `gpu_memory_utilization` -0.05, twice; then the step is `failed` and the next version tries again; after 3 failed versions the service reports `broken` and the trainer stops pushing |
| Task error (dataset, scorer, env) | nothing | as today: that task is switched off, logged; others go on |
| Service process dies | nothing; the in-flight push times out on the sender thread | `devices:`: relaunched (≤ 3); `service:` reconnect every 10 s; results of the version in flight lost (`lost` metric) |
| Evaluator node lost | nothing; sends fail at once or hit the push timeout | as above; the node's results already written stay |
| Push corrupted | nothing | checksum mismatch: dropped, the next due step pushes again |
| Trainer crashes / is preempted | n/a | a user-launched service finishes the version it holds and writes its files; a launched one exits after `idle_timeout` |
| Trainer resumes | nothing | steps with `eval_tasks/step_<n>/done` are not redone; results finished while the trainer was away are sent in the next `hello` and logged at their steps; the resume step is evaluated if due and not done |
| Same GPU as training or a rollout engine | the run does not start (`devices:`); the service is refused and training goes on (`service:`, discovered later) | an error naming both UUIDs and the alternatives |
| Host RAM too small for staging | the run does not start | message suggests `transport: checkpoint` |

The trainer's only shared resources with an evaluator are rank 0's host memory (bounded, allocated at start), its CPU
(one sender thread, a send per due step) and, across nodes, the network. The last is the one real risk: `push_gbps`
caps it, and the GPU check and the multi-node validation measure it.

## 8. Results, tracker, resume, decisions that read metrics

- Metrics keep today's names (`eval/tasks/<task>/<mode>/score`, ...) and are logged by rank 0 from the training
  thread (the tracker is not thread-safe) with `step=n`. W&B accepts this because the library already logs without an
  explicit W&B step and pins `train/global_step` as the x-axis (`logging.py`); trackio and other backends take the
  step argument. The final step's metrics enter `TrainResult.metrics`.
- Files: the service writes `eval_tasks/step_<n>/` exactly as `TrainingEval._tasks` does today (the code moves into a
  function both paths call), plus `metrics.json` and a `done` marker written last.
- Early stopping and best-checkpoint selection: today they read the eval loss only (`BestModelTracker`), never
  `eval_tasks`. v1 rule, enforced: async `eval_tasks` metrics are informational; no trainer decision reads them, since
  they arrive after the weights they describe are gone. A later "best by task score" needs `transport: checkpoint`
  (the weights exist on disk when the score arrives): the service would write `best_tasks.json` pointing at the step,
  and the purge would keep it.

## 9. Trainer coverage

| Trainer | v1 | Notes |
|---|---|---|
| `pgs rl` (colocated vLLM / SGLang / remote, one rank or FSDP) | yes | snapshot teed from the existing push path |
| `pgs distill` (OPD) | yes | same hook as `PublishedWeights.sync` |
| `pgs train`, one process | yes | inline also still available |
| `pgs train`, FSDP (+ TP) | yes | first multi-GPU `eval_tasks` for SFT; TP via `tp.canonical_state_dict` as the save does |
| `pgs train`, pipeline parallel | refused | later: gather across stages |
| Multi-role RL scenarios | phase 6 | `role:` picks one role's folded weights; inline stays skipped |

## 10. Validation

### CPU tests (`tests/test_async_eval.py`, fake engine via `--engine module:factory`, TCP on localhost)

1. Defaults: `placement` missing equals `inline`, and the inline path is called exactly as before (existing
   `eval_tasks` tests unchanged).
2. Never blocks: a service that never acknowledges, or sleeps inside an evaluation; 50 fake steps; each due step's
   hand-off returns within the snapshot copy time + 20 ms.
3. Latest wins: a slow service; due steps 10..100; evaluated steps are an increasing subset ending at 100, `skipped`
   accounts for the rest, at most one area allocated on each side.
4. Dead evaluator: kill the service mid-push and mid-evaluation; training finishes; `alive` drops to 0; a restarted
   service rejoins and later steps arrive.
5. Steps: a fake tracker records `(step, key)`; late results land at their own step, never the current one.
6. Same-GPU refusal: fake UUID sets (trainer ranks, rollout engines, MIG instances) -> refused / accepted with the
   expected message.
7. OOM: a fake engine raising `torch.OutOfMemoryError` once (retry smaller succeeds), always (step failed, then
   `broken`, the trainer stops pushing).
8. Resume: `done` markers skip steps; results finished during the outage are logged at their steps.
9. Final step: the final result is waited for up to `final_timeout_s`, then the run ends either way.
10. Corrupted push (`PGS_TEST_CORRUPT`, as the pool tests) is dropped; the next one applies.
11. The launched child's environment has no torchrun variables; config errors for every bad combination.
12. FSDP decision broadcast with 2 Gloo processes, where the existing CPU FSDP tests already run.

### GPU check (`scripts/gpu_checks/async_eval.sh` + `async_eval_check.py`, one A100 80GB)

The machine has one GPU, and sharing it is refused, so the script picks the split it can:

- **MIG** (two instances, e.g. `4g.40gb` + `3g.40gb`): trainer on one, service on the other.
- **Two or more GPUs**: `CUDA_VISIBLE_DEVICES` split.
- **One GPU without MIG** (the usual shared box): the service runs the `hf` engine on CPU. Every trainer-side cost
  (snapshot, staging, sender thread, results) is the same; only the evaluator's own speed differs.

Arms (Qwen3.5-0.8B, fixed seed, 300 steps, `every: 50`): `pgs train` SFT and `pgs rl` (colocated vLLM), each with no
`eval_tasks`, `inline`, `async`. Pre-registered pass criteria:

- mean step time of `async` within **2 %** of no evaluation over steps 10-300; at due steps the extra time is at
  most the logged `snapshot_seconds`, and that is ≤ 1 s for 0.8B;
- the training losses of `async` equal those of no evaluation step by step (bitwise with deterministic kernels,
  else within 1e-6 relative): evaluation must not touch the trainer's state or random streams;
- refusal: `devices` naming the trainer's GPU fails before any model load, with the message of section 3;
- OOM: the service with `PGS_TEST_EVAL_OOM=always` (fault injection in `generate`), and on MIG / a second GPU a real
  OOM (`gpu_memory_utilization: 0.99` next to a process holding memory): the trainer exits 0 with every step done.

### Needs multi-node

Pushes over a real network concurrent with FSDP NCCL traffic (the step-time criterion with and without
`push_gbps`), `service.json` discovery on a shared filesystem, the Slurm `EVAL_NODES` split, an evaluator node lost
(`scancel` of its step), and `devices:` on an 8-GPU node with torchrun ranks plus remote rollout engines.

## 11. Files

Add:
- `src/palingenesis/eval/service.py`: the service (`pgs eval serve`), eval loop, engine OOM retries, results.
- `src/palingenesis/eval/async_eval.py`: `AsyncEvalClient` (staging, sender thread, heartbeat, results queue, launch).
- `src/palingenesis/eval/placement.py`: GPU UUIDs, the same-GPU and host-memory checks, the child environment.
- `src/palingenesis/rl/pool/receiver.py`: `WeightReceiver`, factored out of `EngineWorker`.
- `scripts/eval_slurm.sh`, `scripts/gpu_checks/async_eval.sh`, `scripts/gpu_checks/async_eval_check.py`.
- `tests/test_async_eval.py`, `tests/test_async_eval_placement.py`.

Change:
- `eval/config.py` (keys, errors), `eval/training.py` (`_tasks` shared with the service; `submit` / `poll`),
  `eval/engines.py` (`EnginePool.update_weights`), `eval/cli.py` (`serve`, later `watch`).
- `rl/trainer.py`, `opd/trainer.py`, `train.py` (the async branch next to the inline one; the end-of-run wait).
- `rl/pool/worker.py` (uses `WeightReceiver`; GPU UUIDs in `state`).
- `checkpoint.py` (purge skips `.eval_pending`, phase 5).
- `scripts/train_slurm.sh` (`EVAL_NODES`).
- Docs: `mkdocs_docs/guides/evaluation.md`, `multi-node.md`, `multi-gpu.md`, `reference/configuration.md`,
  `AGENTS.md` (one line in the decision tree).

## 12. Phases and effort

| Phase | Content | Effort |
|---|---|---|
| 0 | config keys, validation, placement checks, CPU tests 1, 6, 11 | 2 days |
| 1 | service + receiver + client; `pgs rl` / `pgs distill` integration (one rank and FSDP tee); CPU tests 2-5, 7, 10 | 6 days |
| 2 | `pgs train` (one process, FSDP, TP), late logging, resume, final wait; tests 8, 9, 12 | 3 days |
| 3 | `devices:` launch, `pgs eval serve`, `service.json`, Slurm scripts, docs | 3 days |
| 4 | GPU check on the A100 and its fixes | 2 days |
| 5 | `transport: checkpoint`, `pgs eval watch`, DCP consolidation, purge lease | 3-4 days |
| 6 | multi-role `role:`, several services, per-rank shard push (no gather) if phase 4 shows the gather costs | open |

Phases 0-4 (about 16 days) deliver the main path; 5 and 6 are independent of each other.

## 13. Prior art (cited from memory; verify versions before the docs cite them)

- **verl** (`trainer.test_freq`, `val_before_train`), **OpenRLHF** (`eval_steps`), **NeMo-RL** (`val_period`) validate
  with the rollout engine, synchronously: today's `inline`, which we keep as the default.
- **TRL** `vllm_mode="server"` (`trl vllm-serve` on other GPUs, weights pushed from the trainer) and its async GRPO
  weight transfer: the pattern of a separate engine on disjoint GPUs fed by pushes, which `rl/pool` already follows.
- **TorchTitan** async checkpointing (`async_with_pinned_mem`): a snapshot staged in pinned host memory, I/O on a
  background thread, the training loop waiting only for the copy. Borrowed for the staging step.
- **Megatron-LM** and **TorchTitan** validation (`--eval-interval`, validator) are inline loss evaluation; Megatron's
  `--async-save` is the same staging idea for checkpoints.
- **LLM Foundry** `AsyncEval` callback and **AReaL**'s automatic evaluator launch evaluation jobs on saved
  checkpoints, and **OLMo** evaluates checkpoints offline next to small in-loop evals: the model for `transport:
  checkpoint` and `pgs eval watch`.
- **lm-evaluation-harness** in training loops (HF `Trainer` callbacks, NeMo) runs inline on the training model: what
  `pgs train` does today on one process.
- **W&B** `define_metric(step_metric=...)`: already used here; it is what lets late results plot at their own step.

## 14. Open questions for the owner

1. MIG as the only same-physical-GPU option: acceptable, or do you want a "reserved budget" mode despite the SM
   contention it cannot remove?
2. Should evaluation metrics ever drive training (best checkpoint by a task score, early stopping)? If yes,
   phase 5 becomes part of the main path.
3. Final evaluation: wait up to `final_timeout_s` (default 1 h), or return at once and let the service finish
   (results in files only, not in `TrainResult`)?
4. Several evaluator services per run (tasks split across nodes): needed in v1?
5. Host RAM: is a refusal at 50 % of `MemAvailable` for the pinned staging area the right bound on your nodes?
