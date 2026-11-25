# Reinforcement Learning

`pgs rl` trains a model against rewards: a checked math answer, code that passes hidden tests, an LLM judge, or any Python function. The policy samples groups of rollouts with vLLM (or [SGLang](#rollout-engines-vllm-or-sglang)) on the same GPU, each group is scored, and one policy-gradient step follows. Rollouts can be multi-turn: the policy calls tools (a Python sandbox, your functions, your environment) between turns.

!!! abstract "When to use it"
    - The task has a reward you can compute (a verifier, tests, a judge) and the model already solves it some of the time: groups where every rollout gets the same reward carry no gradient.
    - Start from an SFT checkpoint that still has output diversity: see [SFT → RL Transition](sft-to-rl.md).
    - To imitate a stronger model on the student's own samples instead of a reward, use [on-policy distillation](distillation.md); for preference pairs, [DPO](dpo.md).

    Jump to: [Data](#data) · [Rewards](#rewards) · [Tools and environments](#tools-and-environments) · [The objective](#the-objective) (what to watch) · [Sampler/trainer agreement](#samplertrainer-agreement) · [Rollout engines](#rollout-engines-vllm-or-sglang) · [Speed](#speed) · [Multi-GPU](#multi-gpu-clusters-and-memory) · [Resuming](#resuming)

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
| `data.revisit_after_steps` | `0` | Retry queue (0 = off, experimental): a prompt whose group had no success comes back this many policy versions (training steps) later instead of waiting for the next epoch. Delays count steps, not draws, so they do not move with refill or the world size |
| `data.revisit_partial_after_steps` | `-1` | Delay for partly solved groups (the ones that give gradient); -1 = half of `revisit_after_steps` (at least 1) |
| `data.revisit_max_failed` | `1` | Retries per prompt over the run after a group with no success (0 = replay only partly solved prompts: never-solved ones rarely turn mixed on a timer, and they are the costliest agentic draws) |
| `data.revisit_max_partial` | `3` | Replays per prompt over the run after a partly solved group |
| `data.revisit_partial_max_share` | `0.875` | A group solved at least this share (7/8) is not queued (little left to learn) |
| `data.revisit_share` | `0.5` | At most this share of all draws are revisits, so prompts never tried keep flowing |

With the queue on, success is the raw task reward (before `loss.group_shaping` and the overlong penalty) at or above `data.success_threshold`; a group where the environment reported `metric/verifier_failed` for most rollouts is not judged (`groups/verifier_failed` counts them). A row is never served twice at the same policy version, from the queue or the epoch order, nor while a group of it is still generating. The log adds `sampler/*` counters (rows tried, ever solved, `rows_solved_after_fail`: compare it with a no-learning null, since a row with $p = 0.1$ fails a group of 8 four times in ten; the sampler state keeps each row's (version, solved share) history for that), `sampler/revisit_share_drawn`, and `outcome/pass@1_fresh` / `outcome/pass@1_revisit`: the queue changes the prompt mix, so only fresh draws and the held-out evaluation compare with a run without it.

| Option | Default | |
|---|---|---|
| `data.format` | `auto` | `chat` (palingenesis rows) or `nemo_gym` (NVIDIA NeMo Gym / Nemotron-RL rows, prompt in `responses_create_params`), detected |
| `data.tools_field` | `tools` | A column of per-row tool schemas (OpenAI or Responses-API form), used when the environment has none |

### Screening the pool before the run

A group whose rollouts all succeed or all fail has zero advantages: rows the policy always or never solves cost rollouts and teach nothing. Dynamic sampling drops such groups during training, after paying for them. `pgs screen` spends k rollouts per row once, with the RL config's own rendering, environment and rewards, and keeps the rows the policy solves sometimes:

```bash
pgs screen --config configs/rl_math.yaml --k 8 --output pool_screened.jsonl
# then train on pool_screened.kept.jsonl:   --data.dataset pool_screened.kept.jsonl
pgs screen --config rl.yaml --k 16 --max_rate 0.9 --model runs/sft/final --rollout.backend hf
```

A row is kept when `--min_rate` < pass rate < `--max_rate` (defaults 0 and 1); a success is a reward at or above `data.success_threshold`. `<output>` holds every screened row with a `screen` record (samples, scored, successes, pass rate with its 95% Wilson interval, mean reward, finish reasons, mean completion tokens) and is appended chunk by chunk, so re-running the same command resumes; `<output>.kept.jsonl` holds the kept rows unchanged. The summary gives the rows per band (unsolved, partial, solved, failed: no rollout could be scored), the pass-rate histogram and, with `data.metrics_key`, the same per family. Other `--section.option` arguments override the config (`--rollout.backend hf` for a small model without vLLM; `--split eval` screens the held-out rows). Exit code 1 when nothing is kept.

Screen with the checkpoint you will train (`--model`), at the training temperature (the default, `rollout.temperature`). With k samples a row at true pass rate p looks unsolved with probability $(1-p)^k$: 43% for p = 0.1 at k = 8, so hard pools need a larger k, or keep a share of the unsolved rows for later stages. Tülu 3 removed the prompts its SFT model always solved before RLVR; Polaris and Skywork-OR1 trained on pass-rate bands. Not for multi-role scenarios (`pgs scenario audit` checks their rows) or multimodal policies yet.

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

Per sample it can ask for `completion` (the final answer, reasoning removed), `reasoning`, `prompt`, `messages` (the whole conversation, tool calls included), `completion_ids`, `raw_contents` (every assistant turn's content as sampled, before any strip), `finish_reason`, `env`, `sandbox`, and any dataset column. Batched functions get the plural names and every column as a list.

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
| `whitespace` | -1 when a reply starts with one space or has a line indented by exactly one space, else 0 (reads `raw_contents`) |

Add `whitespace` to every RL run, at weight 0.2:

```yaml
rewards:
  whitespace: {fn: whitespace, weight: 0.2}
```

The parser, the graders and the chat template all strip the text, so no other reward sees a reply that starts with a space (`" Hello!"`) or a stray space at a line start (`"\n def"`, which breaks pasted code). On a small model such a habit can drift in during RL and spread to every task and to thinking-mode answers; a 0.8B policy went from 0% to over 90% of its replies. The share of rollouts with a defect is logged on every run as `format/whitespace`; keep it under 0.02.

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

These adapters make other standards palingenesis environments, with no special case in the trainer:

| `env.type` | |
|---|---|
| `palingenesis.rl.envs.openenv:OpenEnvAdapter` | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) servers, remote over WebSocket (`args: {base_url}`) or the server class in-process (`args: {env_class}`); MCP tools are listed automatically, step environments get one tool taking their Action; `reset_fields` forwards the task selection so a group's rollouts see the same task |
| `palingenesis.rl.envs.nemo_gym:NemoGymAdapter` | NeMo Gym resources servers: `seed_session`, one route per tool, `verify` with the transcript; a masked sample is skipped. Runs any Nemotron-RL dataset against its own verifier |
| `palingenesis.rl.envs.mcp:MCPEnv` | Any [MCP](https://modelcontextprotocol.io) server: its tools become the policy's tools (below) |
| `palingenesis.rl.envs.harbor:HarborEnv` | [Harbor](https://harborframework.com) task directories: a fresh Docker container per episode, the task's tests as the reward; optionally the dialect of a real agent harness (opencode, pi, mini-swe-agent, OpenHands, qwen-code, goose, hermes, SWE-agent, terminus-2), one per GRPO group. See [Harbor Environments](harbor.md) and [Harness Profiles](harness-profiles.md) |

Tool calls are parsed in the model's own format (Hermes JSON for Qwen2.5/Qwen3, XML for Qwen3-Coder/Qwen3.5), run concurrently with a timeout, and their results (sanitized: a tool cannot inject control tokens; cut to `env.max_tool_output_tokens`) are appended as tokens rendered by the model's own chat template. Nothing is ever re-tokenized: what the trainer scores is exactly what the sampler saw, and only sampled tokens are trained. A tool error is returned to the policy as the tool's answer.

**Images.** With `model.multimodal: true` (Qwen-VL family) prompts may carry image parts and a tool may return `ToolOutput(text, images=[...])`, e.g. a screenshot. The options are below; the layouts, the chat templates that support them and the handling of old screenshots are in the [Multimodal guide](multimodal.md#multi-turn-rl-with-images).

| Option | Default | |
|---|---|---|
| `model.multimodal` | `false` | Load the policy with its vision tower; `model.train_vision_tower`, `model.train_projector` (both `false`), `model.min_pixels`, `model.max_pixels` as in [`multimodal`](../reference/configuration.md#multimodal). Engines: `hf`, in-process `vllm` |
| `env.image_results` | `tool` | `tool`: the image inside the tool message. `user`: text in the tool message, the images in one user message after the turn's tool results (what OpenAI-style chat completions allow) |
| `env.image_tool_note`, `env.image_user_note` | empty | `user` layout: the tool message's text when a tool returned only images, and the text that opens the user message |
| `env.max_images` | `0` | Keep the last N images in the context (0 = all); older ones become `env.image_placeholder` (`[image omitted]`). Turns sampled before a drop stay trained on the context they saw |
| `env.image_prune_slack` | `0` | Extra images allowed before dropping back to `max_images`: one context rewrite every slack + 1 images |

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

Rewards become group-relative advantages (reward minus the group mean, divided by the batch's standard deviation):

$$
A_{g,j} = \frac{r_{g,j} - \bar r_g}{\sigma_{\text{batch}} + \epsilon}
$$

where $r_{g,j}$ is the reward of rollout $j$ of group $g$, $\bar r_g$ the mean over the group and $\sigma_{\text{batch}}$ the standard deviation of the centred rewards $r_{g,j} - \bar r_g$ over the whole batch, all ranks included (`loss.advantage_std: batch`, the default; `group` divides by each group's own, `none` not at all). $\epsilon = 10^{-6}$.
{: .pg-where }

 Groups whose rollouts all got the same reward carry no gradient, so they are replaced by fresh prompts before the step (`rollout.max_refill`).

The default loss, `masked_is`, is a policy gradient anchored to the sampler's own log-probabilities: the importance ratio $\pi_\theta / \mu$ (trainer over sampler probability of each sampled token) covers both staleness and the numerical difference between vLLM and the trainer, capped at `loss.is_cap`, with PPO's asymmetric clip (`eps_low` 0.2, `eps_high` 0.28, i.e. the band $[1 - 0.2,\ 1 + 0.28]$) applied as a mask, and a sequence mask (`loss.seq_mask`) that drops a trajectory whose tokens drift too far. No reference model and no KL term, as in current recipes (DAPO, Magistral, ScaleRL, Olmo 3). `cispo`, `icepop` and `gspo` are one option away. Losses are aggregated per prompt by default (`loss.aggregation`).

### Choosing the algorithm

The estimator is four independent options: the objective (`loss.type`), the group baseline (`loss.baseline`), the advantage scale (`loss.advantage_std`) and the aggregation over tokens (`loss.aggregation`). Switch from the config or the command line, nothing else changes:

```bash
pgs rl --config configs/rl_math.yaml --loss.type cispo
pgs rl --config configs/rl_math.yaml --loss.type reinforce --loss.baseline leave_one_out --loss.advantage_std none
```

With ρ = π_θ/μ the ratio of the trained policy to the sampler on a token, A the advantage and S the sequence mask, every objective is a coefficient on A · ∇log π_θ:

| `loss.type` | Coefficient | Use it when | Source |
|---|---|---|---|
| `masked_is` (default) | S · min(ρ, `is_cap`), zero where PPO's clip would stop the token (A > 0 and ρ > 1 + `eps_high`, or A < 0 and ρ < 1 - `eps_low`) | The default: GRPO / DAPO's clipped surrogate, anchored to the sampler's log-probs | GRPO (2402.03300, Feb 2024), DAPO (2503.14476, Mar 2025) |
| `cispo` | S · min(ρ, `is_cap`), no mask | Every token should keep a gradient; the loss ScaleRL selected in its comparison (it found caps of 4, 5 and 8 equivalent) | MiniMax-M1 (2506.13585, Jun 2025), ScaleRL (2510.13786, Oct 2025) |
| `icepop` | S · ρ inside [`icepop_low`, `icepop_high`], zero outside | The sampler and the trainer disagree on a few tokens (MoE routing, quantized rollouts) | Ring-1T (2510.18855, Oct 2025) |
| `gspo` | one ratio per sequence (geometric mean of ρ), clipped at 1 ± `gspo_eps` | Token ratios are too noisy (long sequences, MoE); aggregation is then fixed per sequence | GSPO (2507.18071, Jul 2025) |
| `ipo` | S · min(ρ, `is_cap`) where \|π_θ - μ\| ≤ `ipo_eps`, zero outside | Stale or mismatched rollouts where a ratio mask misfires: it drops a rare token moving 0.01 → 0.06 and keeps a likely one falling 0.9 → 0.2. Newer and less tested than the rows above | prime-rl's IPO (its algorithm docs, Oct 2026), after DPPO (2602.04879, Feb 2026); the published form has no cap on ρ, `is_cap` is kept here because a bounded difference does not bound the ratio of a rare token |
| `reinforce` | S · 1 | On-policy runs where you want the plain score-function estimator, e.g. as a baseline for the others | REINFORCE; with `loss.baseline: leave_one_out`, RLOO (2402.14740, Feb 2024) |

`reinforce` has no ratio, so it is the policy gradient only on samples of the policy being trained. It is refused with `rollout.max_staleness > 0` (older versions sampled part of the batch; `cispo` is the same estimator with the ratio) and with `rollout.think_skip` (a forced token needs its importance weight), and warns with FP8 rollouts. With `max_staleness: 0` every batch is sampled by the weights that train on it, so what it leaves uncorrected is the numerical difference between vLLM and the trainer: `policy/mismatch_k3` still reports it and `loss.seq_mask` still drops the trajectories where it is large. On-policy the other token-level objectives reduce to the same gradient wherever ρ is 1; they differ in what they do with the mismatch. Unclipped REINFORCE on off-policy samples has been reported to collapse (2509.24203, Sep 2025), which is why staleness is an error here and not a warning.

| Option | Values | |
|---|---|---|
| `loss.baseline` | `mean` (default), `leave_one_out` | The mean reward of the group's scored rollouts, or of the other ones. For n scored rollouts the leave-one-out advantage is n / (n - 1) times the mean-baseline one: the same direction. It is a different estimator only where n differs between groups (failed rollouts) or the scale is kept (`advantage_std: none`); with `advantage_std: group` it changes nothing and is refused |
| `loss.advantage_std` | `batch` (default), `group`, `none` | Divide the advantage by the batch's std (ScaleRL, REINFORCE++ 2501.03262), the group's (GRPO), or not (Dr. GRPO 2503.20783: the group's std up-weights prompts that are nearly all right or all wrong) |
| `loss.aggregation` | `prompt` (default), `token`, `constant` | Token mean per prompt, then over prompts (ScaleRL); mean over all tokens (DAPO); sum over a constant budget (Dr. GRPO: no length bias) |
| `loss.pass_k` | `1` (off) | Train for pass@k: each rollout's reward is replaced by its pass@k credit before the baseline (PKPO 2505.15201, Pass@k Training 2508.10751); `2 <= pass_k < rollout.group_size`, see [below](#training-for-passk) |
| `loss.seq_mask` | `0.1` | Drop a trajectory whose mean \|ρ - 1\| exceeds this (Trust Region Masking 2512.23075, Dec 2025; its suggested range is 0.03 to 0.10) |
| `loss.kl_ref`, `kl_coef`, `kl_ref_reset` | off | KL to a frozen reference, optionally reset every N steps (DeepSeek-R1 keeps 0.001 with resets; ProRL 2505.24864 resets when progress stalls) |
| `loss.observation_weight`, `observation_exclude` | `0` (off), `[]` | The observation loss: λ × the NLL of the environment's output in the policy's own multi-turn trajectories, added to the policy loss (ECHO 2605.24517, which used λ = 0.05); see [below](#observation-loss) |

Published recipes as options: DAPO is `masked_is` with `aggregation: token` and `overlong_buffer`; ScaleRL is `cispo` with the defaults and `data.retire_above: 0.9`; Dr. GRPO is `advantage_std: none` with `aggregation: constant`; RLOO is `reinforce` with `baseline: leave_one_out` and `advantage_std: none`.

Why the defaults: ScaleRL, the one large comparison across recipes, found that the loss, the aggregation and the normalisation mostly change how fast a run learns, not where it ends, and that zero-variance filtering, prompt retirement and fp32 logits matter as much. `masked_is` keeps the mask because it bounds the update on stale and mismatched tokens at no cost on-policy; move to `cispo` when `policy/clipped` stays high, and to `ipo` or `icepop` when `policy/mismatch_k3` grows during the run.

What is verified: each objective's gradient is tested against a hand-computed value, and on samples of the policy itself all six give the exact policy gradient of the expected reward (`tests/test_rl.py`); the trainer runs end to end with each switch on CPU. `reinforce`, `ipo` and `leave_one_out` have no GPU benchmark behind them in this repository yet. Not implemented, because they need a second model or a different loop: learned or Monte Carlo value baselines (VinePPO 2410.01679: better credit assignment at 2 to 5 times the cost per step), a decoupled proximal policy for high staleness (AReaL 2505.24298), sequence-level tapered weights for offline data (TOPR 2503.14286).

Every step logs what to watch: `reward` and `rewards/*`, `policy/abs_ratio_dev` and `policy/mismatch_k3` (sampler/trainer agreement; a rise precedes collapse), `policy/clipped`, `policy/seq_masked`, `policy/entropy`, `groups/zero_variance`, `truncated`, `skipped`, and the time split (`time/rollout`, `time/train`). Rollout diagnostics: `engine/seqs_mean` (sequences decoding at once), `engine/tail_s` (time with under a quarter of the peak left), `engine/prefix_hit` and `engine/prefill_tokens`, `groups/continued`. `logging.dump_trajectories: N` writes every Nth batch (turns, tool calls and outputs, rewards and their components, advantages, per-turn and per-tool timing, zero-variance groups too; `logging.dump_groups` samples groups) and the engine's concurrency timeline to `<output_dir>/rollouts/`, on a background thread. Read it before tuning anything: `pgs rollouts --dir <output_dir>` reports finish reasons, tool errors and latency, loops and the worst trajectories ([Observability](observability.md)). `pgs monitor` and `pgs loss` read the step lines of the log; `logging.use_wandb`, `logging.use_trackio` and `logging.trackers` send the metrics to a tracker.

### Training for pass@k

```yaml
rollout:
  group_size: 16
loss:
  pass_k: 4          # 1 (the default): pass@1
```

GRPO-style training maximises pass@1. On prompts the policy solves only sometimes, it pushes every failed sample down as hard as a success is pushed up: exploration narrows, and pass@k at large k can fall while pass@1 rises (the sharpening signature: `outcome/pass@<G>` falling under a rising `outcome/pass@1`). `loss.pass_k: k` optimises the probability that at least one of k samples succeeds instead. Before the baseline, the reward of rollout i of a group of n scored rollouts becomes

$$
r'_i = \frac{n}{k} \cdot \frac{1}{\binom{n}{k}} \sum_{S \ni i,\ |S| = k} \max_{j \in S} r_j
$$

the mean best reward of the k-subsets of the group that contain it, times their share. The group mean of $r'$ is the unbiased estimate of the expected best of k samples (pass@k for 0 / 1 rewards), and $\sum_i r'_i \nabla \log \pi(y_i)$ is (n / k times) the unbiased estimate of its gradient: PKPO ([arXiv:2505.15201](https://arxiv.org/abs/2505.15201)). For 0 / 1 rewards with c successes it is the closed form of Pass@k Training ([arXiv:2508.10751](https://arxiv.org/abs/2508.10751)): a success scores 1 and a failure $1 - \binom{n-c-1}{k-1} / \binom{n-1}{k-1}$, so a failure next to successes costs less than under pass@1.
{: .pg-where }

The baseline, `advantage_std` and the aggregation then apply as usual. A group whose pass@k is certain (fewer than k failures) has equal $r'$ and carries no gradient, so dynamic sampling refills it; so does a group with at most k scored rollouts. Steps log `outcome/pass@<k>` (unbiased, at `data.success_threshold`) next to `outcome/pass@1`. Both papers report pass@k gains with pass@1 kept or improved when k is annealed back to 1 later in training: run a second stage with `pass_k: 1` (or resume with it). Neither the gain nor the schedule has been measured with this library yet; the transform is tested against brute force over subsets (`tests/test_evalstats.py`, `tests/test_rl_pass_k.py`). Not with a multi-role scenario.

### Observation loss

```yaml
loss:
  observation_weight: 0.05          # λ; 0 (the default) turns it off
  observation_exclude: ['\A\[hint\][^\n]*\n']   # optional: harness text in tool results, not a target
```

With a multi-turn environment, the policy is also trained to predict what its actions return: the tool results' tokens in its own trajectories get a cross-entropy term, added to the policy loss of any `loss.type`. Per trajectory the term is the summed NLL of its targets divided by its observation length (all its tool-result tokens, targets or not), and the step adds λ × the mean over its trajectories that have observations: ECHO's $\mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{Env}}$ ([arXiv:2605.24517](https://arxiv.org/abs/2605.24517); the paper does not state the batch average). The targets are the text of the tool results as the chat template renders it, never the template's markup or an image placeholder; the trainer's own messages (the answer to a malformed call, a format error), the truncation marker of `env.max_tool_output_tokens`, text the environment declares in `observation_notes` (HarborEnv: its harness profile's, e.g. terminus-2's warning prefix) and `loss.observation_exclude` matches count in the length but are not targets. Steps log `loss/observation` (mean NLL per target token) and `observation_tokens`. The paper reports gains from base models on terminal tasks and small ones from an SFT'd policy, at λ = 0.05 (0.01–0.05 worked, 0.1 degraded, 0.2 could collapse); nothing has been measured with it here. Details and the SFT counterpart: [Agentic training](agentic-training.md#observation-loss-include_observations).

## Sampler/trainer agreement

vLLM and the trainer compute the same policy with different kernels, so the log-probability of a sampled token differs slightly between them even with identical weights. The importance ratio $\rho = \pi_\theta / \mu$ absorbs the difference, but every unit of it is noise in the gradient, and it grows on long sequences and recurrent layers. The two agreement metrics are, over the trained tokens,

$$
\texttt{mismatch\_k3} = \mathbb{E}\big[\rho - 1 - \log \rho\big] \approx \mathrm{KL}(\mu \,\Vert\, \pi_\theta), \qquad \texttt{abs\_ratio\_dev} = \mathbb{E}\big[\lvert \rho - 1 \rvert\big]
$$

where $\mu$ is the sampler's (vLLM's) probability of the sampled token and $\pi_\theta$ the trainer's: the first is the per-token k3 estimate of the KL divergence. Both are 0 when the two agree exactly.
{: .pg-where }

Every step also logs them split by bucket, for the buckets that had tokens:

| Metric | Tokens |
|---|---|
| `policy/mismatch_k3/pos_0_1k`, `pos_1k_4k`, `pos_4k_plus` | by position in the completion (tokens after the prompt, tool results included): [0, 1024), [1024, 4096), 4096 and beyond |
| `policy/mismatch_k3/current` | sampled by the policy being trained: the numerical mismatch alone |
| `policy/mismatch_k3/stale` | sampled by an older policy version (`rollout.max_staleness > 0`): staleness on top. A turn whose request was added before a weight update (partial rollouts) counts as stale, so this is an upper bound: a request still queued at the update counts too |
| `policy/abs_ratio_dev/<bucket>` | the same buckets for $\mathbb{E}\big[\lvert \rho - 1 \rvert\big]$ |
| `policy/stale_tokens` | share of the trained tokens whose request started under an older version (an upper bound, as above) |

Tokens forced by think-skip (`rollout.think_skip`) carry an intended importance weight and are left out of the buckets. At step 0 of a fresh run the batch was sampled by the initial weights and is scored by the same weights before any update, so its mismatch is the two implementations' numerics alone: it is logged as `policy/mismatch_k3_step0`, and the trainer warns when it exceeds $10^{-2}$ (`MISMATCH_K3_STEP0_WARN` in `palingenesis.rl.trainer`). bf16 rounding differences stay well below that; above it the engine and the trainer compute different distributions (weights not loaded, a chat template, stop token or temperature difference, FP8 rollouts, kernels that disagree).

Options that bring the two closer:

| Option | Default | Effect |
|---|---|---|
| `rollout.fp32_head` | `false` | vLLM computes the logits in fp32 (a bf16 GEMM with fp32 output, as the trainer does) instead of rounding them to bf16 before the softmax, so the sampler's log-probs and the sampling itself use the trainer's logits. Uses vLLM's `head_dtype` (`hf_overrides`), available from vLLM 0.26.0: an older vLLM is a configuration error. Costs an fp32 `[sequences, vocab]` buffer per decoding step; the head must not be quantized |
| `rollout.ssm_state_dtype` | `float32` | The recurrent state of hybrid models (Gated DeltaNet, Mamba layers) between decoding steps, passed to vLLM as `mamba_ssm_cache_dtype`. The trainer's chunked kernels carry the state in fp32 over the whole sequence, so `float32` matches them; `bfloat16` rounds it at every generated token (less memory, a warning: more mismatch on recurrent layers); `auto` leaves vLLM's choice (the model config's `mamba_ssm_dtype` when it has one, as Qwen3.5 does; fp32 for NemotronH; else the model dtype). The `float32` default raises the engine's state memory for models whose `auto` was bf16 (Qwen3-Next). Models without recurrent layers ignore it |
| `model.match_vllm_norm` | `true` | The gated RMSNorm of linear-attention layers (Qwen3.5, Qwen3-Next, OLMo hybrid) in fp32 with one rounding, the operations of vLLM's `RMSNormGated`. transformers' pure-torch fallback rounds to bf16 after the normalization, again after the weight and once more after the gate (on random inputs about a third of its outputs differ from vLLM's by one bf16 step or more). With flash-linear-attention installed transformers uses its fused kernel, which already rounds once, and the patch changes nothing |

With `rollout.url` (a server you started) these options cannot reach the server: start it with `--hf-overrides '{"head_dtype": "float32"}'` and `--mamba-ssm-cache-dtype float32`. With the SGLang backends the same options set `enable_fp32_lm_head` and `mamba_ssm_dtype` ([Rollout engines](#rollout-engines-vllm-or-sglang)).

## Rollout engines: vLLM or SGLang

`rollout.backend` picks what samples the rollouts. The trainer, the losses and the metrics are the same whichever engine it is: every engine implements one protocol (`palingenesis.opd.rollout.RolloutEngine`: `generate`, `update_weights`, `sleep`, `wake`, and the streaming calls for continuous batching), returns the sampled token ids with each token's log-probability under the sampling distribution, and is checked against the trainer by the same [agreement metrics](#samplertrainer-agreement).

| `rollout.backend` | Engine | Weights after each step |
|---|---|---|
| `vllm` (default) | in-process vLLM on the policy's GPU; sleeps while the trainer trains | loaded in place (`load_weights`) |
| `vllm_server` | a `vllm serve` process on the policy's GPU, or `rollout.url` | CUDA IPC (vLLM's weight-transfer endpoints) |
| `sglang` | an `sglang.Engine` on the policy's GPU (its scheduler is a child process); sleeps while the trainer trains | CUDA IPC handles of bf16 copies, in buckets of `rollout.weight_bucket_mb` (`update_weights_from_tensor`) |
| `sglang_server` | an SGLang server launched on the policy's GPU, or one at `rollout.url` on another GPU or node | `rollout.weight_sync`: `tensor` (CUDA IPC, same GPU), `distributed` (NCCL broadcast from the trainer), `disk` (a checkpoint in `rollout.weight_sync_dir`, loaded by the server) |
| `remote` | `palingenesis.rl.pool.worker` processes on other GPUs or nodes (`--backend vllm`, `sglang` or `hf`) | staged in host memory, sent while the trainer steps ([multi-agent guide](multi-agent.md)) |
| `hf` | the trainer's own `model.generate` | none (it is the trained model); slow, for tests |

**Which one.** vLLM is the default and the most exercised path here: it is the one the speed figures below were measured with, it continues rollouts across weight updates (partial rollouts with `max_staleness > 0`), and it takes images. SGLang is the alternative when it is faster for the model and batch at hand (its radix cache shares long common prefixes, such as multi-turn agent contexts, across requests), when a model or kernel is available there first, or to keep the engine on another node with an NCCL weight push (`sglang_server` with `weight_sync: distributed`). Measure before switching: the [GPU checks](#checking-an-engine) report the throughput of both at an equal batch, and `policy/mismatch_k3_step0` of a few steps tells whether the engine's numerics agree with the trainer's.

| | vLLM (`vllm`, `vllm_server`) | SGLang (`sglang`, `sglang_server`) |
|---|---|---|
| Sampling: temperature, `top_p`, `top_k`, `presence_penalty`, stop ids, per-turn budgets | yes | yes; a budget that would pass the context is cut to fit (SGLang refuses such a request, vLLM stops there: the same completion) |
| Behaviour log-probs | after temperature and top-k / top-p (`processed_logprobs`) | after temperature; with top-k / top-p SGLang's renormalised `return_sampling_mask` log-probs (SGLang ≥ 0.5.12), else the untruncated ones, with a warning. RL training samples at plain temperature, where both are the same |
| A group's identical prompts | one request with `n` samples (prefilled once) | `n` requests; the radix cache shares the prefill |
| Continuous batching across turns | yes | yes (requests run as tasks on the engine's event loop while its scheduler decodes) |
| Weight update | in place (`vllm`); CUDA IPC (`vllm_server`) | bucketed bf16 CUDA IPC (`sglang`); CUDA IPC, NCCL or disk (`sglang_server`) |
| Prefix cache after an update | reset | flushed with the update's last bucket |
| Requests in flight at an update (`max_staleness > 0`) | continue with the new weights (partial rollouts; `policy/stale_tokens`) | the update waits for them (the engine finishes them before the first bucket is sent; a server's generate calls and updates exclude each other): every completion is sampled by one version |
| Sleep / wake (colocated, `max_staleness: 0`) | sleep level 2 | releases weights, KV cache and CUDA graphs (`release_memory_occupation`, via `torch_memory_saver`); the weights' memory comes back before the next update, the rest at wake |
| `rollout.fp32_head` | `head_dtype` (vLLM ≥ 0.26) | `enable_fp32_lm_head` |
| `rollout.ssm_state_dtype` (Qwen3.5, Qwen3-Next) | `mamba_ssm_cache_dtype` | `mamba_ssm_dtype` |
| `rollout.kv_cache_dtype`, `rollout.quantization` | yes | yes (`fp8` is SGLang's `fp8_e4m3`) |
| MoE routing replay (`rollout.routing_replay`) | `enable_return_routed_experts` (needs `prefix_caching`) | `enable_return_routed_experts`; SGLang returns every decoder layer, the MoE layers are kept. Not run on a GPU yet |
| LoRA (`lora.rank`) | the folded weights after every step; named adapters on `remote` workers | the same; named adapters on `remote` workers through `load_lora_adapter` (not run on a GPU yet) |
| Images (`model.multimodal`) | `vllm` | no: the SGLang engines take token ids (refused at startup) |
| Roles with their own adapter (scenarios) | `vllm_server` (one server per adapter) | no (refused: use `vllm_server`, `remote` or `hf`) |
| torchrun, one engine per rank | `vllm` | `sglang` (the engine on each rank's GPU; checked on one GPU only) |
| Determinism | `seed` | `random_seed`; batch-invariant kernels with `rollout.sglang_args: {enable_deterministic_inference: true}` |
| `pgs screen`, `run_episodes` | yes | `sglang` |
| The engine process dies | the trainer's own process (in-process engine) | `sglang`: the run ends with an error naming the scheduler (SGLang's own handler, which kills the whole process tree, is replaced), its other processes are stopped; `sglang_server`: the next request reports the server's exit code and log |
| AMD GPUs | vLLM's ROCm build | not tested |

**Installing SGLang.** SGLang pins its own torch and transformers, so it goes in an environment of its own, with palingenesis installed there: `pip install "sglang==0.5.17"` (the last release built on torch 2.11, which palingenesis' vLLM extra also uses; later releases need newer torch builds). Its PyPI wheels target CUDA 13; on a CUDA 12.9 driver swap in the cu129 builds as [SGLang's install guide](https://docs.sglang.ai/docs/get-started/install) shows (`scripts/gpu_checks/sglang.sh` does it in a fresh virtual environment). sglang is imported only when an SGLang backend is built.

| Option | Default | |
|---|---|---|
| `rollout.sglang_args` | `{}` | Further SGLang engine arguments ([ServerArgs](https://docs.sglang.ai/docs/advanced_features/server_arguments) fields), e.g. `{attention_backend: fa3}`, `{enable_deterministic_inference: true}`, `{rl_on_policy_target: fsdp}`. Arguments the options above set (`mem_fraction_static`, `context_length`, `disable_radix_cache`, ...) are refused with the option to use instead; unknown ones are an error naming the closest |
| `rollout.weight_bucket_mb` | `512` | Weights per update call: each bucket is one contiguous bf16 copy (one CUDA IPC handle); an update holds at most two buckets of copies at a time |
| `rollout.weight_sync` | `auto` | `sglang_server`: `tensor`, `distributed`, `disk`; `auto` is `tensor` for a server the trainer launches and `distributed` for one at `rollout.url` |
| `rollout.weight_sync_dir` | `<output_dir>/rollout_weights` | `weight_sync: disk`: where each version is written (a file system the server reads too); the previous version's files are deleted once the server has loaded the new one |

`rollout.gpu_memory_utilization` becomes SGLang's `mem_fraction_static`: the engine's weights and KV cache take about that fraction of the GPU memory that is free when the engine starts (SGLang keeps the rest of it as headroom), so next to a trainer that is already loaded the engine gets less than vLLM's fraction of the whole GPU; raise it if SGLang reports too little memory for its KV cache. `rollout.max_num_seqs` becomes `max_running_requests` (0: SGLang sizes it, also from the recurrent-state slots of hybrid models), `rollout.enforce_eager` disables CUDA graphs. A server you start yourself (`rollout.url`) keeps its own settings: start it with `--skip-tokenizer-init` and, as needed, `--enable-fp32-lm-head`, `--mamba-ssm-dtype float32` and `--enable-return-routed-experts`, and without `SGLANG_RETURN_ORIGINAL_LOGPROB` in its environment (SGLang would then report the log-probs of the unscaled logits; engines the trainer starts get an environment without it). With `--tp-size` above 1 the `tensor` sync sends one payload per rank.

The trainer stops what it started when a run ends: the in-process engine's scheduler process (which holds the GPU memory), a launched server and its process group, the NCCL group of `distributed`, the last checkpoint of `disk`. If the trainer is killed outright, SGLang's scheduler ends itself (it asks the kernel to be killed with its parent) and a launched server receives SIGTERM.

`weight_sync: distributed` joins the trainer (rank 0) and the server's ranks in an NCCL group of their own (SGLang's `init_weights_update_group`), then broadcasts each bucket while the server receives it. NCCL refuses two ranks on one GPU, so this is for a server on another GPU or node; on the trainer's own GPU use `tensor`. It has not run on a GPU yet (the GPU checks use one).

The weight sync and memory patterns follow [slime](https://github.com/THUDM/slime) / [Miles](https://github.com/radixark/miles) and [verl](https://github.com/volcengine/verl) (all Apache-2.0), reimplemented against the engine protocol: bucketed CUDA IPC pushes into a colocated engine, release and resume of the engine's memory around each training step (weights resumed before the update, KV cache after it), and a dedicated NCCL group to a remote engine. The engine side is SGLang's RL API ([update_weights_from_tensor / from_distributed / from_disk, release / resume_memory_occupation](https://docs.sglang.ai/docs/advanced_features/sglang_for_rl)).

### Checking an engine

`scripts/gpu_checks/sglang.sh` (one NVIDIA GPU, models from the Hugging Face cache) creates its own SGLang environment and prints one JSON line per check: generation and log-prob parity with vLLM on fixed prompts (mean |Δ log-prob| of each engine against the trainer), a weight update that changes the outputs and samples the new weights' log-probs (and restores the original outputs, with the radix cache on), sleep / wake memory release, 10-step `pgs rl` and `pgs distill` runs, and decode throughput against vLLM at an equal batch:

```bash
PGS_SRC=$PWD/src SGLANG_VENV=/scratch/sglang-venv WORKDIR=/scratch/sglang-checks \
  VLLM_PYTHON=$PWD/.venv/bin/python bash scripts/gpu_checks/sglang.sh
```

The SGLang backends are tested on CPU against a scripted engine (`tests/test_sglang_rollout.py`); their GPU validation is pending until that script's results are in.

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
- **Overlap.** `rollout.max_staleness: 1` generates, runs tools and scores step k+1 while step k trains. Groups still generating when a batch is full keep going and join the next batch (partial rollouts; new weights load between engine steps, and the importance ratio uses the log-probs each token was sampled with), so the engine does not drain waiting for a batch's slowest groups.
- **Budgets decide multi-turn speed.** A group is complete when its slowest rollout is, so a batch waits for the longest serial chain of turns. Size `rollout.max_completion_tokens` from what successful trajectories use (a row may carry its own `max_turns`, `max_completion_tokens` and `max_new_tokens`, positive integers checked at load; set `max_completion_tokens` with a raised `max_turns`, or the turns share the default budget: a warning names rows left under 256 tokens a turn): on agentic GSM8K with Qwen3.5-0.8B, halving it (cutting 1% of correct trajectories) made steps 1.7x faster. Measured and not worth it there: MTP speculative decoding (`rollout.vllm_args: {speculative_config: ...}`, slower at ~130 sequences in flight), smaller prefill chunks (no change), a bf16 recurrent state (`rollout.ssm_state_dtype: bfloat16`: a few % faster, 2.5x the sampler/trainer mismatch), and 64 groups per batch instead of 32 (the same trajectories per second: decoding is saturated). A profiled Qwen3.5-0.8B decode step at 128 sequences is 10 ms, 95% GPU kernels: the linear-attention state update (3 ms), attention (1.9 ms), GEMMs (2 ms) and the vocabulary softmax (0.7 ms).
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
| `lora.rank: 16` | Low-rank adapters on a frozen base instead of full weights: no gradients or optimizer state for the base, which is held in bf16 only. The engine gets the folded weights after every step; checkpoints are plain models plus `adapter/`. Tested on CPU only; validation on a GPU is pending ([LoRA in RL](lora.md#lora-in-rl)) |
| `train.optimizer: adamw8bit` | bitsandbytes' 8-bit AdamW moments: 2 bytes per parameter instead of 8 (one GPU, without FSDP). `train.adam_eps` then defaults to 1e-8: 8-bit moments round small second moments to 0, and the fp32 default of 1e-15 would let those updates explode |
| `train.optimizer: adamw_bf16` | The bf16 parameters are stepped directly ([AdamW bf16](../reference/optimizers.md#adamw-bf16)): no fp32 master copy and no fp32 gradients, bf16 moments with a bf16 Kahan compensation buffer (`train.bf16_rounding: kahan`, the default here: at RL learning rates stochastic rounding's noise outgrows the update). About 10 bytes per parameter for weights, gradients and optimizer state instead of 18. Gradients accumulate over micro-batches in bf16. One GPU, without FSDP or adapters; not yet measured against fp32 masters on a GPU |
| `train.cpu_offload` | FSDP2 offload: parameters, gradients and optimizer state in CPU memory (also on one GPU), for the largest model a GPU can train. Budget about 24 bytes of host RAM per parameter per node: 16 for the fp32 parameters, gradients and AdamW moments, plus pinned staging (a 4B policy on one GPU needs about 96 GB; tmpfs files count against it); the trainer warns at startup when it will not fit |

**Host memory.** After each optimizer step the trainer calls glibc's `malloc_trim(0)` (Linux with glibc only) and logs the process's resident memory as `memory/host_rss_gb` (OPD trainers too). Agentic runs allocate large transient buffers (tool outputs, verifier copies, a batch's trajectories) on many threads, and glibc gives each thread its own heap arena, up to 8 per CPU, keeping their pages after the buffers are freed: without a limit the RSS ratchets up at every new peak. For RL runs with many concurrent containers launch with `MALLOC_ARENA_MAX=4` (e.g. `MALLOC_ARENA_MAX=4 python -m palingenesis.rl.trainer --config ...`); a `memory/host_rss_gb` that keeps climbing after the first steps is worth a look.

## Common failures

| Symptom | What to check |
|---|---|
| `groups/zero_variance` near 1, reward flat | every rollout of a group gets the same reward: prompts too hard or too easy for the policy, or a reward that cannot tell rollouts apart; `data.retire_above` and the [retry queue](#data) change the prompt mix |
| `policy/mismatch_k3_step0` above 1e-2 (startup warning) | the engine and the trainer compute different distributions: weights, chat template, stop token, temperature, FP8 rollouts; see [Sampler/trainer agreement](#samplertrainer-agreement) |
| `policy/mismatch_k3` rising during the run | a rise precedes collapse; the options in [Sampler/trainer agreement](#samplertrainer-agreement) bring the two closer |
| Many `skipped` samples | sandbox, judge or environment outages (`SkipSample`, a sandbox failure); they are kept out of the loss |
| `format/whitespace` above 0.02 | add the [`whitespace` reward](#rewards) |
| Slow steps on agentic tasks | size `rollout.max_completion_tokens` and turn budgets ([How it is made fast](#how-it-is-made-fast)); read `logging.dump_trajectories` with `pgs rollouts` before tuning |
| `memory/host_rss_gb` keeps climbing | launch with `MALLOC_ARENA_MAX=4` ([Host memory](#multi-gpu-clusters-and-memory)) |

See also: [Multi-Agent RL](multi-agent.md) for episodes with several roles, [Harbor Environments](harbor.md) and [Harness Profiles](harness-profiles.md) for agent tasks, the [CLI reference](../reference/cli.md) for `pgs rl`, and the [configuration reference](../reference/configuration.md).

## Resuming

`train.save_steps` writes resumable checkpoints: the policy in Hugging Face format (with `lora.rank`: the folded policy, and its adapter in `adapter/`, which is what a resume loads on top of `model.policy`), the optimizer (sharded with DCP under FSDP), every rank's prompt sampler with its retired prompts, and the random states. `train.resume_from: auto` continues from the newest one, so the same command starts and resumes a run (with the same number of ranks).
