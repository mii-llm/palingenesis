# Training Agentic & Reasoning Models

*How to fine-tune models that think, use tools and recover from errors: the data format, what gets loss, thinking and non-thinking rows, long traces, and the checks to run before training.*

---

!!! abstract "When to use this guide"
    Your SFT data has reasoning, tool calls and tool results (multi-turn agent traces). The short path:

    1. Put the traces in the [data format](#data-format) below (`messages` + `tools`); ShareGPT-style rows are [normalised](#role-mapping-non-standard-formats).
    2. Decide the [thinking mode of each row](#reasoning-thinking-and-non-thinking-rows) if you train a model with both modes.
    3. Start from the [example config](#example-config-for-agentic-traces) (SeCO for long traces).
    4. [Check the masking](#debugging-masking) with `pgs inspect` and [validate the tool calls](#data-validation) before training.
    5. If the trained model loops or never stops calling tools, see [Repetition loops](#repetition-loops-and-runaway-episodes).

    Related: [Data Preparation](data.md), [Long Sequences (SeCO)](long-sequences.md), [On-Policy Distillation](distillation.md), [Reinforcement Learning](rl.md) with [Harbor](harbor.md) environments, [Multimodal](multimodal.md) (tools that return images), [Observability](observability.md) (inspecting rollouts).

## What makes agentic data different

Plain SFT data is one exchange: the user asks, the assistant answers, the loss is on the answer. An agent trace is a loop:

```
user → assistant thinks → calls a tool → tool result → thinks again → calls another tool → ... → final answer
```

The parts are not alike. The reasoning and the tool calls are the model's own output: it should learn to produce them. The tool results come from the environment: the model reads them but does not write them. Which tokens get loss decides what the model learns, so most of this guide is about that choice.

---

## Data format

```json
{"messages": [
  {
    "role": "system",
    "content": "You are a coding assistant with access to file editing tools."
  },
  {
    "role": "user",
    "content": "Fix the bug in the group_by function."
  },
  {
    "role": "assistant",
    "content": "",
    "reasoning": "Let me look at the implementation first. The issue says group_by only groups adjacent elements.",
    "tool_calls": [{"function": {"name": "view_file", "arguments": "{\"path\": \"/src/lib.rs\"}"}}]
  },
  {
    "role": "tool",
    "content": "pub fn group_by<K, F>(self, key: F) -> GroupBy<K, F, Self> ..."
  },
  {
    "role": "assistant",
    "content": "I've found the issue. The current implementation only tracks the current key...",
    "reasoning": "GroupBy keeps only the current key, not the keys seen before. A new ChunkBy struct with a map of seen keys fixes it."
  }
],
 "tools": [{"type": "function", "function": {"name": "view_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]}
```

- **Tools** go in a `tools` field next to `messages` (a list, or a JSON string: Arrow often cannot store heterogeneous JSON schemas natively). They are passed to the chat template as `tools=`, which renders them the way the model sees them at inference. The field name is `data.tools_field` (default `tools`).
- **Tool-call `arguments`** may be a dict or a JSON string (the OpenAI wire format); strings are parsed, since chat templates iterate over the arguments as a mapping. A whole conversation stored as a JSON string also works.
- **Reasoning** goes in `reasoning` (`reasoning_content` and `think` are read too). See [Reasoning](#reasoning-thinking-and-non-thinking-rows) for reasoning written inside the content.

### Per-message training flags

A message with `"loss": false` (or `"weight": 0`) stays in the context but gets no loss: a canned greeting, or a failed attempt kept so the model learns the recovery that follows it.

```json
{"role": "assistant", "content": "Hi! How can I help you today?", "loss": false}
```

Messages without the flag (or with `"loss": null`, which Arrow fills in for messages that lack the key) are trained.

### What gets loss

| Role | Field | Loss | Why |
|------|-------|:---:|-----|
| `system` | `content` | no | context the model does not write |
| `user` | `content` | no | input the model does not write |
| `assistant` | `content` | **yes** | the visible output |
| `assistant` | `reasoning` | **yes** (`data.train_on_reasoning: false` turns it off) | the thinking |
| `assistant` | `tool_calls` | **yes** | the actions: which tool, when, with what arguments |
| `tool` | `content` | no (default); yes with [`include_observations`](#observation-loss-include_observations) | environment output |

An assistant turn is trained as the template renders it: everything after the template's assistant header, up to and including the end-of-turn token. That covers the tool-call markup (`<tool_call>...</tool_call>` on Qwen), which templates render from `tool_calls` rather than from `content`, and the end-of-turn token after a call, which teaches the model to stop and wait for the result. An empty `<think></think>` block that a template inserts into a turn without reasoning is not trained.

`reasoning` is passed to the template as both `reasoning` and `reasoning_content` (as vLLM does), so Qwen3 / Qwen3.5 templates render it as `<think>...</think>`. Templates without a reasoning slot drop it: check with [`pgs inspect`](#debugging-masking).

---

## Reasoning: thinking and non-thinking rows

Models such as Qwen3 and Qwen3.5 have two modes, selected by the chat template (`enable_thinking`). If you train both, each row has to be rendered in the mode it was written in, or the model learns to think where it should not (or the reverse).

- **One mode for the dataset**: `data.chat_template_kwargs: {enable_thinking: true}`.
- **Both modes in one dataset**: a `chat_template_kwargs` column on each row overrides the dataset's setting key by key, so thinking rows (`{"enable_thinking": true}`, with `reasoning`) and non-thinking rows (`{"enable_thinking": false}`, without) mix in one file. Per source, `data.sources[i].chat_template_kwargs` sets a source's default.
- **Reasoning written inside the content** (`"content": "<think>...</think>The answer is 4."`): a *leading* block is taken as the turn's reasoning and rendered once by the template; tags later in the text stay text. Other delimiters (e.g. `[THINK]...[/THINK]`) are set with `data.think_tags`. The details are in [Data: reasoning fields and tags](data.md#reasoning-in-the-data-fields-baked-blocks-tags).
- **Train the answers but not the reasoning**: `data.train_on_reasoning: false`, for traces whose reasoning you do not want imitated.
- **Train only the last assistant turn**: `data.last_turn_only: true`, when earlier turns are context (for example, turns written by another model).

Keep the share of each mode in the data close to how the model will be used: a model fine-tuned only on thinking rows tends to degrade in the other mode.

---

## Observation loss (`include_observations`)

By default tool results are context only. The observation loss also trains the model to predict what the environment returns, as an auxiliary term next to its usual loss. It comes from ECHO ([Shrivastava et al., arXiv:2605.24517](https://arxiv.org/abs/2605.24517)), and palingenesis has it in two places:

```yaml
# pgs rl: the setting the paper tested (the policy's own rollouts)
loss:
  observation_weight: 0.05     # 0 = off (the default)

# pgs train: SFT on recorded traces (not tested by the paper)
data:
  include_observations: true
  observation_weight: 0.05     # the default
```

What the paper does: $\mathcal{L}_{\text{ECHO}} = \mathcal{L}_{\text{GRPO}}(\theta; \mathcal{A}) + \lambda \, \mathcal{L}_{\text{Env}}(\theta; \mathcal{O}')$, where $\mathcal{L}_{\text{Env}}$ is the negative log-likelihood of the observation targets $\mathcal{O}'$ of a rollout (the environment's output, without the harness's warning prefix), summed and divided by the rollout's whole observation length $|\mathcal{O}|$ ("so runs with different target subsets remain comparable"). The paper does not say how the per-sequence terms are averaged over a batch. It sweeps $\lambda \in \{0.001, \ldots, 0.2\}$: 0.01–0.05 worked, runs degraded at 0.1 and could collapse at 0.2; the main text uses a constant $\lambda = 0.05$ in all reported runs, while an appendix lists $\lambda \in \{0.02, 0.05\}$ ("SFT vs. base") without saying which run used 0.02.

| | ECHO (the paper) | `pgs rl`: `loss.observation_weight` | `pgs train`: `data.include_observations` |
|---|---|---|---|
| Training | GRPO, 16 rollouts per prompt | any `loss.type` (the term is added to the policy loss) | SFT |
| Whose observations | the policy's own rollouts | the policy's own rollouts | the traces' (someone else's actions) |
| Weight λ | 0.05 (0.01–0.05 worked) | `loss.observation_weight`, default 0 (off); 0.05 is the paper's | `data.observation_weight`, default 0.05 |
| Normalisation | per rollout: summed NLL over $\mathcal{O}'$ / $\lvert\mathcal{O}\rvert$; batch average not stated | per trajectory: summed NLL over its targets / its observation length $\lvert\mathcal{O}\rvert$, averaged over the step's trajectories that have observations | per conversation: λ × the mean NLL of its observation tokens, added to its assistant loss (below) |
| Which tokens | the environment's output, without the harness's warning prefix | the text of the tool results; not the chat template's markup, not the trainer's own messages, not text the harness declares as its own or `loss.observation_exclude` matches (below) | the text of tool / observation messages and tool-output blocks; not the chat template's markup |
| Measured | the results below | nothing yet: implemented and unit-tested, no training run | nothing yet: implemented and unit-tested, no training run |

What the paper reports (terminal tasks, Table 1): starting from base models, ECHO beats GRPO on every benchmark, e.g. TerminalBench-2.0 pass@1 2.70 → 5.17 (Qwen3-8B) and 5.17 → 10.79 (Qwen3-14B), where the paper puts the standard error near 1.5 points. Starting from a policy already fine-tuned on expert demonstrations, the gains are small (TerminalBench-2.0 pass@1 7.64 → 7.87, pass@3 slightly lower). Episodes got shorter for two of the three starting policies. The paper does not use the loss in SFT, and does not measure tool selection or error recovery.

### In RL (`loss.observation_weight`)

Each trajectory's term is the summed NLL of its observation targets divided by its observation length $\lvert\mathcal{O}\rvert$ (the tokens of the tool results, targets or not, as in the paper); the step adds λ × the mean of these terms over its trajectories that have observations. The NLL is the model's own (logits not scaled by the sampling temperature: the policy did not sample these tokens).

- **Targets**: the tokens of each tool result's text as the chat template renders it, never the template's markup (role headers, `<tool_response>` tags, the whitespace around the text). Image placeholders are never targets.
- **Counted in $\lvert\mathcal{O}\rvert$ but not targets**: the trainer's own messages (the error answering a malformed tool call, a text protocol's format error), the `...[N tokens truncated]...` marker `env.max_tool_output_tokens` writes, the text an environment declares in its `observation_notes` (regular expressions; HarborEnv takes its harness profile's: terminus-2's `Previous response had warnings:` prefix, the paper's case), and whatever `loss.observation_exclude` matches. The error of a tool that raised (`Error: ...`) is the environment's answer and stays a target; add a pattern to leave it out.
- It works with every `loss.type`, gradient accumulation, FSDP, prompt sharing, context rewrites (dropped screenshots: targets stay with the context they were read in), multimodal policies and MoE routing replay. It needs a fast tokenizer. Off (`0`), the train step is unchanged.
- Logged: `loss/observation` (the mean NLL per target token) and `observation_tokens` (targets in the step); `loss` includes the term. Trajectory dumps add `observation_tokens` and `observation_targets` per trajectory.

### In SFT (`data.include_observations`)

Per conversation the loss is its assistant loss plus λ × the mean NLL of its observation tokens; the trainer's token count (the loss's denominator) counts assistant tokens only, so long tool outputs no longer dilute the assistant's loss. Over a batch,

$$
\mathcal{L} = \frac{\sum_c \Big( \sum_{t \in A_c} w_t \, \ell_t \; + \; \lambda \, |A_c| \, \frac{1}{|O_c|} \sum_{t \in O_c} \ell_t \Big)}{\sum_c |A_c|}
$$

with $A_c$ the assistant tokens of conversation $c$, $O_c$ its observation tokens, $\ell_t$ the token's cross-entropy and $w_t$ the [turn-scaling](#turn-scaling) weight (1 by default): each conversation's observation term is weighted by its share of the assistant tokens, like its assistant loss. It is carried by per-token loss weights, so plain, chunked and Cut Cross-Entropy, the chunked DEFT-family objectives, packing, SeCO, context parallel, pre-tokenized caches and off-policy distillation apply it the same way; `plugins.pre_rl` and the DEFT family without `memory.chunked_loss` ignore the weights and are refused. A conversation whose only targets are observations is dropped. The held-out loss (`eval/loss`) measures the assistant's tokens.

!!! warning "Behaviour change"
    Before `data.observation_weight` existed, `include_observations: true` trained every token of the tool messages, wrapper tokens included, at the assistant's weight and counted them in the denominator. Now it trains the tool text at λ = 0.05 per conversation. To get the old weighting back (on the text only):

    ```yaml
    data:
      include_observations: true
      observation_normalization: token   # observation tokens like assistant tokens, counted in the denominator
      observation_weight: 1.0            # their per-token weight
    ```

Treat both as untested experiments: compare against the run without the loss on your own evals. When to try it, and when not:

- **Plausible**: environments whose outputs are mostly determined by the action (file contents, compiler errors, test output, structured API responses).
- **Avoid**: outputs that are noise to the model (random IDs, timestamps, web search results, very long dumps): loss on tokens the model cannot predict. Per-conversation normalisation keeps a long dump from outweighing the assistant's tokens, but it still sets the direction of the term.

---

## Turn scaling

```yaml
data:
  turn_scaling: progressive   # later turns get more weight
```

| Mode | Effect |
|------|--------|
| `uniform` (default) | every trained turn has the same weight |
| `progressive` | weight $w_k = \sqrt{k / K}$ for the $k$-th of $K$ trained turns |
| `last_heavy` | the final turn gets weight 2, the others 1 |

The rationale for `progressive` / `last_heavy` is that early turns of a long trace are often exploration and the late ones produce the solution. It is a heuristic, not a measured result: compare it with `uniform` on your evals. Weights are normalised to mean 1 per conversation, so the loss scale does not change. They are applied by the cross-entropy losses (plain, chunked, Cut Cross-Entropy, the chunked DEFT-family objectives) and by SeCO; DPO, `plugins.pre_rl` and the DEFT family without `memory.chunked_loss` refuse a non-uniform `turn_scaling`.

---

## Repetition loops and runaway episodes

A small student trained on a stronger teacher's traces often repeats a failed tool call, writes the same reasoning sentence over and over, or keeps calling tools after the task is done. The cause is the student's own uncertainty at hard decision points, amplified by repetition: once a call or sentence is in the context, repeating it becomes more likely ([arXiv:2512.12895](https://arxiv.org/abs/2512.12895)). Measure it before fixing it: the share of episodes with a loop and how episodes end (the model stops, hits the per-turn token limit, or hits the turn limit), at the sampling you deploy, in each thinking mode. For RL rollouts, [`pgs rollouts`](observability.md#pgs-rollouts) reports these shares from the trajectory dump.

Two levels of fix:

- **During SFT** (`data` keys, all opt-in): `loop_recovery` and `reasoning_loop_recovery` paste untrained copies of a call or a sentence so the model sees itself mid-loop and is trained on the gold way out; `repeat_unlikelihood` lowers the probability of continuing a repeat exactly where the gold escapes, and `stop_unlikelihood` lowers the probability of another tool call in the final reply. They need `memory.seco: true`. Watch `train/repeat_p` and `train/stop_p`: both should fall. These remove the loop forms they target; they do not teach recovery from states the teacher's traces never contain.

  ```yaml
  memory: {seco: true}
  data:
    loop_recovery: 0.2
    reasoning_loop_recovery: 0.2
    repeat_unlikelihood: 20.0
    stop_unlikelihood: 5.0
  ```

- **On the student's own episodes** (the cause): record the student's episodes and distil a teacher on them, with extra branches where the student starts to repeat itself, extra weight on failed calls, and a cutoff where the teacher stops being reliable. See [Distillation: agent traces](distillation.md#the-students-own-episodes-training-where-it-starts-to-repeat-itself).

At inference, `presence_penalty` (in rollout and eval sampling, e.g. `sampling_mix` entries of a distillation source) reduces repetition without training; it is a stopgap, not a fix.

---

## Example config for agentic traces

```yaml title="agentic config (Qwen3.5, long traces)"
model:
  name_or_path: Qwen/Qwen3.5-4B
  torch_dtype: float32           # fp32 master weights; compute runs in bf16 (see the SeCO guide)
  compile: false                 # SeCO is validated without torch.compile

data:
  dataset: ./agentic_traces.parquet   # messages (+ tools) per row
  max_seq_length: 65536          # whole traces; SeCO keeps activation memory set by the chunk
  packing: true                  # traces of very different lengths: short ones batched, long ones chunked
  include_observations: false    # true: observation loss (experimental in SFT, see above)

train:
  epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-5
  gradient_checkpointing: full

memory:
  seco: true
  seco_chunk_size: 4096
```

For traces that fit in memory without SeCO (up to a few tens of thousands of tokens on one GPU), drop the `memory:` section and use `gradient_checkpointing: full` or `selective`; at moderate lengths full activation checkpointing is cheaper than SeCO. `pgs fit --config ...` measures the options on your longest rows. The masking, `tools` and per-message flags work the same way in every setup.

---

## Debugging masking

Before training on agentic data, check the masking on a few rows:

```bash
pgs inspect --config configs/agentic_traces.yaml --num_samples 3
```

It shows which tokens get loss (highlighted) and which are masked. For an agent trace you should see:

- System prompt, including the tool definitions: **masked**
- User message: **masked**
- Assistant thinking (`<think>...</think>`): **loss** (masked with `train_on_reasoning: false`)
- Assistant tool calls, and the end-of-turn token after them: **loss**
- Tool responses: **masked** (or **loss** with `include_observations`)
- Assistant final answer: **loss**
- Turns flagged `"loss": false`: **masked**

`pgs validate --config ...` runs the same check over many rows and reports the ones that fail.

| What you see | Likely cause | Fix |
|---|---|---|
| User or system tokens have loss | roles not recognised | spell the roles as in [Role mapping](#role-mapping-non-standard-formats), or extend `ROLE_MAP` |
| Thinking tokens missing | the reasoning is not in `reasoning` / `reasoning_content` / `think`, or the row renders in non-thinking mode | move it into one of those fields; set the row's `chat_template_kwargs` |
| Thinking tokens masked | `data.train_on_reasoning: false` | set it to `true` |
| Tool responses have loss | `data.include_observations: true` | [observation loss](#observation-loss-include_observations); set it to `false` for standard masking |
| A whole turn masked | the message carries `"loss": false` or `"weight": 0`, or `last_turn_only: true` | [per-message flags](#per-message-training-flags) |
| Tool definitions missing from the system prompt | the tools are in another column | set `data.tools_field` |

---

## How much reasoning data do you need?

There is no single number; what is known depends on the model size and the source of the traces.

- **Large models, curated data**: a few hundred to about a thousand carefully chosen long reasoning traces were enough to make 32B instruction models strong on competition math (s1, [arXiv:2501.19393](https://arxiv.org/abs/2501.19393), 1,000 examples; LIMO, [arXiv:2502.03387](https://arxiv.org/abs/2502.03387), 817). Selection by difficulty and quality mattered more than volume there.
- **Small models need more, and shorter, traces**: DeepSeek-R1's distilled small models used about 800k samples ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)), and small students learn better from shorter chains or a mix of short and long ones than from a large teacher's longest traces alone ([arXiv:2502.12143](https://arxiv.org/abs/2502.12143)).
- **Agent traces**: verified outcomes matter most: keep traces whose task was solved (tests passed, the checker accepted the answer), with clean endings and few errors, and replay the model's own answers to general prompts to limit forgetting. Measure on held-out tasks, not on the tasks you trained on.

`pgs prepare` removes duplicate and unusable traces, caps the trajectories per task (`--max_per_prompt`) and reports how many traces are cut to `max_seq_length`. It does not rank traces by value: filter on outcome before preparing.

---

## Data validation

Before training, check that the tool calls match the declared tools:

```python
import json

from palingenesis.validate_data import validate_dataset

with open("data.jsonl") as f:
    samples = [json.loads(line) for line in f]

report = validate_dataset(samples, messages_field="messages", tools_field="tools")
print(f"Valid: {report['valid']}/{report['total']}")
for idx, err in report["errors"][:10]:
    print(f"  sample {idx}: {err}")
if report["tools_never_called"]:
    print(f"Tools declared but never called: {report['tools_never_called']}")
```

The validator checks that every `tool_calls[].function.name` exists in the row's `tools`, that every argument exists in that tool's schema, and that the `required` arguments are present. Catching these before training keeps the model from learning to call tools that do not exist or to pass wrong arguments.

---

## Role mapping (non-standard formats)

Conversation lists in other common layouts are normalised during training:

| Your format | Normalised to |
|-------------|:---:|
| `{"from": "human", "value": "..."}` | `{"role": "user", "content": "..."}` |
| `{"from": "gpt", "value": "..."}` | `{"role": "assistant", "content": "..."}` |
| `{"role": "bot", "content": "..."}` | `{"role": "assistant", "content": "..."}` |
| `{"role": "function", "content": "..."}` | `{"role": "tool", "content": "..."}` |
| Field `"think"` on an assistant message | `reasoning` |
| Field `"function_call"` on an assistant message | `tool_calls` |
| Conversation in a `conversations`, `conversation`, `chat`, `dialogue` or `turns` field | found automatically |

So ShareGPT-style datasets work without conversion. Flat single-turn rows (Alpaca's `instruction` / `input` / `output` columns) are not a conversation and are not converted: turn them into `messages` first, for example `[{"role": "user", "content": instruction + "\n\n" + input}, {"role": "assistant", "content": output}]`.

For your own role names, extend the map:

```python
from palingenesis.validate_data import ROLE_MAP

ROLE_MAP["instructor"] = "user"
ROLE_MAP["learner_response"] = "assistant"
```
