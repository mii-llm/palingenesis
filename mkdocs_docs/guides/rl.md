# Reinforcement Learning

`pgs rl` trains a model against rewards: a checked math answer, code that passes hidden tests, an LLM judge, or any Python function. The policy samples groups of rollouts with vLLM on the same GPU, each group is scored, and one policy-gradient step follows. Rollouts can be multi-turn: the policy calls tools (a Python sandbox, your functions, your environment) between turns.

```bash
pgs rl --config configs/rl_math.yaml
pgs rl --config configs/rl_math.yaml --rollout.group_size 16 --train.learning_rate 1e-6
```

```python
from palingenesis.rl import RLConfig, RLTrainer

def correct(completion, answer, **row):        # any dataset column is an argument
    return float(answer.strip() in completion)

RLTrainer(RLConfig.from_yaml("configs/rl_math.yaml"), rewards=[correct]).train()
```

## Data

A JSONL, JSON or parquet file, or a Hugging Face dataset id. The prompt is the first of `messages` (a chat), `prompt`, `question` or `problem` (a string becomes one user turn), or the column named by `data.prompt_field`. Every other column travels with the rollout to the rewards and the environment.

| Option | Default | |
|---|---|---|
| `data.dataset`, `data.split` | | Training rows |
| `data.eval_dataset` / `data.eval_size` | | Held-out rows (a file, or rows split off by content hash) |
| `data.system_prompt` | | Prepended when a row has no system message |
| `data.retire_above` | `0` | Stop drawing a prompt once its group's mean reward reaches this (0.9 with a 0/1 reward) |
| `data.format` | `auto` | `chat` (palingenesis rows) or `nemo_gym` (NVIDIA NeMo Gym / Nemotron-RL rows, prompt in `responses_create_params`), detected |
| `data.tools_field` | `tools` | A column of per-row tool schemas (OpenAI or Responses-API form), used when the environment has none |

NeMo Gym rows (all the `nvidia/Nemotron-RL-*` datasets) are converted at load: `instructions` + `input` become chat messages, Responses items become tool calls and tool results, flat tool schemas become chat-template ones; the verifier's columns (`ground_truth`, `expected_answer`, ...) stay as columns for the rewards. A row may carry a `verifier` column (a name or a list): it is then scored only by the rewards it names, so one mixed dataset can hold math, multiple-choice and code rows.

## Rewards

A reward is a function. It asks for what it needs by name, and gets only that:

```python
def correct(completion, answer):               # per sample: one float (or None)
    ...

def correct(prompts, completions, answer):     # batched (TRL's signature): one list
    ...

async def judged(prompt, completion, **row):   # async works too
    ...
```

Per sample it can ask for `completion` (the final answer, reasoning removed), `reasoning`, `prompt`, `messages` (the whole conversation, tool calls included), `completion_ids`, `finish_reason`, `env`, `sandbox`, and any dataset column. Batched functions get the plural names and every column as a list.

Return `None` when a reward does not apply to a sample: it is left out of the weighted sum. Raise `palingenesis.rl.SkipSample` when a sample cannot be scored for reasons that are not the policy's fault (a sandbox or judge outage): the rollout is kept out of the loss instead of teaching the policy that a right answer was wrong.

In the config, rewards are a named mapping; each is logged as `rewards/<name>`:

```yaml
rewards:
  correct: {fn: math, args: {field: answer}}
  concise: {fn: my_rewards.py:concise, weight: 0.1}
```

| Built-in | Scores |
|---|---|
| `math` | The last `\boxed{}` (else "Answer: ...", else the last number) against `field`; symbolic with `pip install 'palingenesis[rl]'` (math-verify) |
| `boxed_choice` | The option letter inside the last `\boxed{}` (multiple choice) |
| `qa_match` | Normalized exact match (`metric: em`) or token F1 (`f1`) against one or several reference answers |
| `exact`, `choice`, `regex` | Exact match, the last stated option letter, a regular expression |
| `code` | The last fenced Python block against the row's hidden tests (see below) |
| `judge` | A 0–10 quality score from an LLM judge behind any OpenAI-compatible endpoint (`url`, `model`, `template`) |
| `equivalence` | A YES/NO judge: is the final answer equivalent to the reference? (free-form answers no rule can check) |

## Code and the sandbox

`code` reads the row's tests in any common shape — APPS/TACO `{"inputs", "outputs", "fn_name"?}`, LiveCodeBench `[{"input", "output", "testtype"}]`, MBPP assert lists, HumanEval test source with an `entry_point` column — runs at most `max_tests` of them (longest input first, stopping at the first failure), and scores 1 when all pass. The reward is binary by default: partial credit (`partial: true`) invites programs that print guessed outputs.

Programs never see expected outputs: they get their inputs only, and results are compared outside the sandbox. Harnesses check that values returned by the program are plain built-in values, so objects that fake `__eq__` or `__bool__` fail, and a program that exits early never counts as passing.

| `sandbox.backend` | |
|---|---|
| `docker` (default) | A warm pool of long-lived containers: no network, read-only root, `nobody` user, no capabilities, memory and process limits; jobs are multiplexed over the containers' stdin, so there is no per-job start cost |
| `agent_sandbox` | [Kubernetes Agent Sandbox](https://agent-sandbox.sigs.k8s.io/) pods claimed once from a `SandboxWarmPool` (`sandbox.warmpool`, `sandbox.namespace`, `sandbox.connection`: `in_cluster`, `gateway:<name>`, or `url:<router>`) and reused; isolation is the pool's RuntimeClass (gVisor, Kata). `pip install 'palingenesis[agent-sandbox]'` |
| `subprocess` | The same limits, no isolation: development only (`sandbox.allow_unsafe: true`) |

A sandbox failure skips the sample (`skipped` in the logs); it is never scored as a wrong answer.

## Tools and environments

Stateless tools are functions:

```yaml
env:
  type: tools
  tools: [my_tools.py:search, my_tools.py:calculator]
  max_turns: 6
```

Anything with state is a class. Its public methods are the tools (schemas from type hints and the docstring), `reset(**row)` starts an episode, and `get_reward()` is an optional reward of its own:

```python
class PythonEnv:
    def __init__(self, sandbox):               # receives the trainer's sandbox if it asks
        self.sandbox = sandbox

    def reset(self, **row):
        self.calls = 0

    async def python(self, code: str) -> str:
        """Run a Python program and return what it prints.

        Args:
            code: The complete program.
        """
        ...
```

```yaml
env:
  type: examples/rl/python_env.py:PythonEnv
```

Tools that are not methods (HTTP routes, MCP servers) come from `tool_schemas()` and `call_tool(name, arguments)` instead, read after `reset` so they can change per episode. Setting `self.done = True` (e.g. in a `submit` tool) ends the episode; `get_reward(self, messages)` may verify from the transcript and may return components (`{"format": 0.1, "correct": 1.0}`, logged as `env/<name>`); `env.max_concurrent` caps the live instances (remote environments have a capacity).

Two adapters make other standards palingenesis environments, with no special case in the trainer:

| `env.type` | |
|---|---|
| `palingenesis.rl.envs.openenv:OpenEnvAdapter` | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) servers, remote over WebSocket (`args: {base_url}`) or the server class in-process (`args: {env_class}`); MCP tools are listed automatically, step environments get one tool taking their Action; `reset_fields` forwards the task selection so a group's rollouts see the same task |
| `palingenesis.rl.envs.nemo_gym:NemoGymAdapter` | NeMo Gym resources servers: `seed_session`, one route per tool, `verify` with the transcript; a masked sample is skipped. Runs any Nemotron-RL dataset against its own verifier |

Tool calls are parsed in the model's own format (Hermes JSON for Qwen2.5/Qwen3, XML for Qwen3-Coder/Qwen3.5), run concurrently with a timeout, and their results (sanitized: a tool cannot inject control tokens; cut to `env.max_tool_output_tokens`) are appended as tokens rendered by the model's own chat template. Nothing is ever re-tokenized: what the trainer scores is exactly what the sampler saw, and only sampled tokens are trained. A tool error is returned to the policy as the tool's answer.

## The objective

Rewards become group-relative advantages (reward minus the group mean, divided by the batch's standard deviation). Groups whose rollouts all got the same reward carry no gradient, so they are replaced by fresh prompts before the step (`rollout.max_refill`).

The default loss, `masked_is`, is a policy gradient anchored to the sampler's own log-probabilities: the importance ratio π_θ/μ covers both staleness and the numerical difference between vLLM and the trainer, capped at `loss.is_cap`, with PPO's asymmetric clip (`eps_low` 0.2, `eps_high` 0.28) applied as a mask, and a sequence mask (`loss.seq_mask`) that drops a trajectory whose tokens drift too far. No reference model and no KL term, as in current recipes (DAPO, Magistral, ScaleRL, Olmo 3). `cispo`, `icepop` and `gspo` are one option away. Losses are aggregated per prompt by default (`loss.aggregation`).

Every step logs what to watch: `reward` and `rewards/*`, `policy/abs_ratio_dev` and `policy/mismatch_k3` (sampler/trainer agreement; a rise precedes collapse), `policy/clipped`, `policy/seq_masked`, `policy/entropy`, `groups/zero_variance`, `truncated`, `skipped`, and the time split (`time/rollout`, `time/train`).

## Speed

| | |
|---|---|
| `rollout.max_staleness: 0` | On-policy: vLLM sleeps while the trainer steps, and the trainer waits for rollouts |
| `rollout.max_staleness: 1` | Rollouts, tools and rewards of step k+1 overlap the training of step k; the importance ratio corrects for the one-version lag |
| `rollout.prefix_caching` | A group's rollouts share their prompt's prefill |
| `train.micro_tokens` | Tokens per forward/backward micro-batch; logits are computed a slice at a time in fp32 and never materialized |

## How it is made fast

Measured on Qwen3.5-0.8B, GSM8K, one A100 (32 prompts x 8 rollouts per step): 92 s per step in the first version, about 20 s now, with the training pass at ~53% MFU.

- **Rollouts.** One engine call per batch (a flusher gathers every trajectory's pending turn; multi-turn trajectories never wait for the slowest tool of a lock-stepped batch). A group's identical prompts become one vLLM request with `n` samples: the prompt is prefilled once. Groups whose rewards would all be equal are anticipated: `missing / (1 - expected zero-variance rate)` groups are launched at once, instead of refilling in serial rounds that leave the engine half empty.
- **Policy step.** The log-probabilities of the sampled tokens come from bf16 tensor-core GEMMs with fp32 logits, a slice of rows at a time: `[tokens, vocab]` logits are never materialized, and the backward recomputes each slice from a saved log-sum-exp. Micro-batches hold a power-of-two number of rows, so Triton kernels (flash-linear-attention's included) autotune a handful of times instead of on every new batch size. Statistics stay on the GPU until one sync per step.
- **Overlap.** `rollout.max_staleness: 1` generates, runs tools and scores step k+1 while step k trains.
- **FP8 rollouts.** `rollout.kv_cache_dtype: fp8` / `rollout.quantization: fp8` speed up generation; the importance weights and the sequence mask absorb the extra sampler/trainer mismatch (watch `policy/abs_ratio_dev`).

## Multi-GPU, clusters, and memory

```bash
torchrun --nproc_per_node 8 -m palingenesis.rl.trainer --config configs/rl_math.yaml     # one node
sbatch examples/rl/slurm.sbatch configs/rl_math.yaml                                    # Slurm, many nodes
```

Every rank owns a GPU, its shard of the prompts and a colocated vLLM engine; the policy is sharded with FSDP2 (fp32 master weights, bf16 compute). The step is exact data parallelism: group baselines are rank-local, while the loss normalizers and the advantage std are all-reduced, so the gradient equals a single process's. Ranks run the same number of micro-batches (FSDP's gathers are collectives), and after each optimizer step the full weights are gathered one parameter at a time and loaded into every engine on the trainer thread. `rollout.batch_prompts` is the global batch. On `SIGUSR1`/`SIGTERM` (Slurm's warning before the time limit, preemption) every rank saves a resumable checkpoint at the same step and exits; `train.resume_from: auto` continues it.

| Memory option | Effect |
|---|---|
| `model.gradient_checkpointing: selective` | Keeps matmul outputs, recomputes the cheap operations: most of the activation memory for a few % of compute (`full`: everything; the default `auto` is `full` with `train.cpu_offload`, else off) |
| `train.micro_tokens` | Tokens per forward/backward micro-batch |
| `train.fsdp` | Shards parameters, gradients and optimizer state over the ranks (automatic under torchrun) |
| `train.cpu_offload` | FSDP2 offload: parameters, gradients and optimizer state in CPU memory (also on one GPU), for the largest model a GPU can train. Budget about 24 bytes of host RAM per parameter per node: 16 for the fp32 parameters, gradients and AdamW moments, plus pinned staging (a 4B policy on one GPU needs about 96 GB; tmpfs files count against it); the trainer warns at startup when it will not fit |

## Resuming

`train.save_steps` writes resumable checkpoints: the policy in Hugging Face format, the optimizer (sharded with DCP under FSDP), every rank's prompt sampler with its retired prompts, and the random states. `train.resume_from: auto` continues from the newest one, so the same command starts and resumes a run (with the same number of ranks).
