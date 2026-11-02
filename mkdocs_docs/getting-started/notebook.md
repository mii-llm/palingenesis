# Training from a notebook

Every trainer is a plain Python object, so a Jupyter notebook (or any script) can run it in-process on **one GPU**.
Multi-GPU training needs one process per GPU, which a notebook kernel cannot provide: launch it with `torchrun`
(see the end of this page).

---

## Setup

Install into the environment the kernel runs (on Linux, the CUDA build of torch first: see [Installation](install.md)):

```bash
pip install "torch>=2.11" --index-url https://download.pytorch.org/whl/cu129
pip install "palingenesis[train,logging]" ipykernel
```

Start from a shipped example config, or any YAML in the repository's `configs/`:

```python
from palingenesis.examples import write_example

write_example("sft", "sft.yaml", force=True)   # also: "dpo", "distill", "rl" (pgs init --list)
```

---

## SFT and DPO

`palingenesis.train.train(config)` runs a whole training and returns a `TrainResult`. DPO is the same call with
`dpo.enabled: true` in the config.

```python
from palingenesis.config import Config
from palingenesis.train import train

config = Config.from_yaml("sft.yaml")
config.model.name_or_path = "Qwen/Qwen3-0.6B"
config.data.dataset = "data/train.jsonl"           # JSONL with {"messages": [...]}
config.data.streaming = False
config.train.max_steps = 100
config.train.output_dir = "./runs/sft-notebook"

result = train(config)
print(result.output_dir, result.global_step, result.metrics.get("train/loss"), result.best_eval_loss)
```

`Config.from_cli(["--config", "sft.yaml", "--train.max_steps", "100"])` applies the same dotted overrides as the
command line; pass the list explicitly (the kernel's own `sys.argv` is never read when you do).

What `train()` guarantees in-process:

- **Interrupting is safe.** Stopping the cell (KeyboardInterrupt) or an error closes the wandb / trackio run (marked
  failed) and tears down the process group, so the next `train()` call in the same kernel starts clean. Checkpoints
  saved so far stay on disk; re-running with the same `output_dir` resumes from the last complete one.
- **The result** has `output_dir`, `global_step`, `total_time_s`, `tokens_total`, `metrics` (the last value of every
  logged scalar, e.g. `train/loss`, `eval/loss`) and `best_eval_loss` / `best_step` when an eval set is configured.
- **No GPU, clear error.** Without a usable CUDA GPU it raises `RuntimeError("No usable CUDA GPU ...")` with the fix.

The model and optimizer are freed when `train()` returns. To give the memory back to other cells right away:

```python
import gc, torch
gc.collect(); torch.cuda.empty_cache()
```

---

## RL and on-policy distillation

`RLTrainer` and `OPDTrainer` run in-process the same way: build one from its config and call `train()`. The RL
trainer releases its rollout threads and event loop when `train()` returns or fails, so several runs in one kernel do
not accumulate threads.

```python
from palingenesis.logging import setup_logging
from palingenesis.rl.config import RLConfig
from palingenesis.rl.trainer import RLTrainer

setup_logging(rank=0)                     # step logs in the cell output
config = RLConfig.from_yaml("rl.yaml")    # pgs init rl
config.train.steps = 20
RLTrainer(config).train()
```

Rewards and environments can be plain Python objects defined in the notebook:
`RLTrainer(config, rewards={"correct": my_reward}, dataset=rows)`; see [Reinforcement Learning](../guides/rl.md).

```python
from palingenesis.opd.config import OPDConfig
from palingenesis.opd.trainer import OPDTrainer

config = OPDConfig.from_yaml("distill.yaml")   # pgs init distill
OPDTrainer(config).train()
```

With `rollout.backend: vllm`, vLLM runs inside the kernel and holds part of the GPU (`rollout.gpu_memory_utilization`).
Restart the kernel to release it completely before a different kind of run.

---

## Multi-GPU

FSDP, context parallelism and multi-GPU RL need one process per GPU. From a notebook cell, launch them as a
subprocess:

```bash
!torchrun --standalone --nproc_per_node=4 -m palingenesis.train --config sft.yaml
!torchrun --standalone --nproc_per_node=4 -m palingenesis.rl.trainer --config rl.yaml
```

See [Multi-GPU](../guides/multi-gpu.md).
