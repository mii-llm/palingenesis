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

**Which tools the policy gets.** Every public method is a tool unless the class pins its set, so a public helper (`grade()`, `check()`) would be callable by the policy, and RL finds such things. Pin them with a class attribute, `tools = ("python", "submit_answer")`, or for any environment (MCP and remote ones included) with `env.allowed_tools: [python, "search__*"]` (names or shell-style patterns; one matching nothing is an error). Only the exposed tools can be called: anything else is answered "unknown tool" before it reaches the environment. The run logs the list at startup. `configs/rl_agentic_math.yaml` is a complete example: GSM8K with `python` (sandboxed) and `submit_answer` (ends the episode; the environment grades it).

Two adapters make other standards palingenesis environments, with no special case in the trainer:

| `env.type` | |
|---|---|
| `palingenesis.rl.envs.openenv:OpenEnvAdapter` | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) servers, remote over WebSocket (`args: {base_url}`) or the server class in-process (`args: {env_class}`); MCP tools are listed automatically, step environments get one tool taking their Action; `reset_fields` forwards the task selection so a group's rollouts see the same task |
| `palingenesis.rl.envs.nemo_gym:NemoGymAdapter` | NeMo Gym resources servers: `seed_session`, one route per tool, `verify` with the transcript; a masked sample is skipped. Runs any Nemotron-RL dataset against its own verifier |
| `palingenesis.rl.envs.mcp:MCPEnv` | Any [MCP](https://modelcontextprotocol.io) server: its tools become the policy's tools (below) |

Tool calls are parsed in the model's own format (Hermes JSON for Qwen2.5/Qwen3, XML for Qwen3-Coder/Qwen3.5), run concurrently with a timeout, and their results (sanitized: a tool cannot inject control tokens; cut to `env.max_tool_output_tokens`) are appended as tokens rendered by the model's own chat template. Nothing is ever re-tokenized: what the trainer scores is exactly what the sampler saw, and only sampled tokens are trained. A tool error is returned to the policy as the tool's answer.

### MCP servers

Any [Model Context Protocol](https://modelcontextprotocol.io/specification/latest) server works as an environment: `tools/list` becomes the policy's tool schemas and each `tools/call` result its observation. A server takes a few lines with the official SDK (`pip install "palingenesis[mcp]"`):

```python
# shop_server.py
import uuid

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("shop")
PRICES, BASKETS = {"apple": 2, "pear": 3}, {}


@server.tool()
def create_basket() -> dict:
    """Create an empty basket."""
    basket_id = f"bsk_{uuid.uuid4().hex}"
    BASKETS[basket_id] = []
    return {"basket_id": basket_id}


@server.tool()
def add_item(basket_id: str, sku: str) -> str:
    """Add an item to the basket.

    Args:
        basket_id: The basket.
        sku: apple or pear.
    """
    if sku not in PRICES:
        raise ToolError(f"unknown sku {sku!r}: choose apple or pear")   # the policy reads this
    BASKETS[basket_id].append(sku)
    return f"added {sku}"


@server.tool()
def grade(basket_id: str, answer: str) -> dict:
    """Score the episode (hidden from the policy)."""
    return {"reward": float(sum(PRICES[s] for s in BASKETS[basket_id]) == 5)}


if __name__ == "__main__":
    server.run("streamable-http", host="0.0.0.0", port=8000)       # http://host:8000/mcp
```

```yaml
env:
  type: palingenesis.rl.envs.mcp:MCPEnv
  max_concurrent: 256              # episodes in flight: size it to the server
  max_turns: 8
  args:
    server: http://shop:8000/mcp   # or a command, run as a subprocess over stdio: [python, shop_server.py]
    state_tool: create_basket      # called by reset(): the episode's handle...
    state_arg: basket_id           # ...hidden from the policy and filled in on every call
    reward_tool: grade             # optional: the server scores the episode
```

- **One connection, every episode.** MCP (2026-07-28) is stateless: every request carries what the server needs, so one client serves all concurrent trajectories. Servers on the earlier session-based revisions are detected and supported. `servers: {search: <url>, code: <url>}` combines several servers; their tools are prefixed (`search__query`); `tools: [...]` keeps a subset.
- **Episode state lives behind a handle.** The protocol has no sessions, so a stateful server returns a handle from a creation tool and takes it on later calls, as in the spec's [stateful tools](https://modelcontextprotocol.io/specification/latest/server/tools#stateful-tools). With `state_tool`/`state_arg`, `reset()` creates it (with the row's `reset_fields` as arguments, so a group's rollouts get the same task) and the policy never sees it. Without them, the policy carries handles itself, as a deployed agent would.
- **Rewards.** Usually reward functions over the transcript. A server-side grader (`reward_tool`) receives the handle, and `answer` (the final assistant message) and/or `messages` when its schema has them; it returns a number, `{"reward": x}` or components (`{"format": 0.1, "correct": 1.0}`). An `isError` from the grader skips the sample instead of scoring it 0.
- **Errors are observations.** A result with `isError: true` goes back as `Error: ...` and counts in `tool_errors`. The Python SDK hides the message of an unexpected exception from clients: raise `ToolError` for messages the policy should read.
- **Serving it for training.** Rollouts call tools at the rate the engine generates: hundreds of calls in flight. Run the server with several workers or replicas behind one URL (the protocol is stateless, so any replica serves any request) and keep episode state in a shared store keyed by the handle. Pin down what varies (live web search, clocks): a group's rollouts are compared with each other, so the same call should give the same result within a group.

Text and embedded text resources reach the policy as text; images and audio become a placeholder. Tool descriptions are the server's; JSON-schema `title` annotations are dropped (prompt tokens in every rollout).

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

Measured on Qwen3.5-0.8B, GSM8K, one A100 (32 prompts x 8 rollouts per step): 92 s per step in the first version, about 18 s now: 11 s of rollouts (15.5k generated tokens/s) and 6 s for the policy step, which runs at 44% MFU (bf16 peak, model FLOPs: no recompute, no padding). Its matrix multiplies run at about 80% of peak; the rest is Qwen3.5's linear-attention kernels, norms and activations.

- **Rollouts.** With the colocated vLLM engine, one engine thread adds every request to the running batch the moment it arrives and returns it the step it finishes: a multi-turn trajectory whose turn ended runs its tools and comes back while the others keep decoding (continuous batching across turns; separate blocking calls would pace every turn by its slowest sequence: 1.7x slower on agentic GSM8K). A group's identical prompts become one vLLM request with `n` samples: the prompt is prefilled once. Groups whose rewards would all be equal are anticipated: `missing / (1 - expected zero-variance rate)` groups are launched at once, instead of refilling in serial rounds that leave the engine half empty.
- **Policy step.** The log-probabilities of the sampled tokens come from bf16 tensor-core GEMMs with fp32 logits, a slice of rows at a time: `[tokens, vocab]` logits are never materialized, and the backward recomputes each slice from a saved log-sum-exp. Everything else about a slice takes one pass over its logits (Triton): an online log-sum-exp yields the log-probs and the entropy together, and the backward writes the bf16 gradient directly, instead of separate max, exp, sum, gather, scatter and cast kernels over a 248k-wide vocabulary (34% → 40% MFU). On one GPU the model computes with bf16 parameters while AdamW steps fp32 master weights, and micro-batch gradients accumulate in fp32 (40% → 44%): fp32 weights under autocast would re-cast activations in every layer. Micro-batches hold a power-of-two number of rows, so Triton kernels (flash-linear-attention's included) autotune a handful of times instead of on every new batch size. Statistics stay on the GPU until one sync per step.
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
| `train.optimizer: adamw8bit` | bitsandbytes' 8-bit AdamW moments: 2 bytes per parameter instead of 8 (one GPU, without FSDP). `train.adam_eps` then defaults to 1e-8: 8-bit moments round small second moments to 0, and the fp32 default of 1e-15 would let those updates explode |
| `train.cpu_offload` | FSDP2 offload: parameters, gradients and optimizer state in CPU memory (also on one GPU), for the largest model a GPU can train. Budget about 24 bytes of host RAM per parameter per node: 16 for the fp32 parameters, gradients and AdamW moments, plus pinned staging (a 4B policy on one GPU needs about 96 GB; tmpfs files count against it); the trainer warns at startup when it will not fit |

## Resuming

`train.save_steps` writes resumable checkpoints: the policy in Hugging Face format, the optimizer (sharded with DCP under FSDP), every rank's prompt sampler with its retired prompts, and the random states. `train.resume_from: auto` continues from the newest one, so the same command starts and resumes a run (with the same number of ranks).
