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

```yaml
data:
  include_observations: true   # tool / observation messages also get loss
```

By default tool results are context only. With `include_observations: true` they get loss too, so the model is also trained to predict what the environment returns.

The idea comes from ECHO ([Shrivastava et al., arXiv:2605.24517](https://arxiv.org/abs/2605.24517)), with one difference to keep in mind. ECHO adds the observation-prediction loss to **reinforcement learning** (GRPO): the policy predicts the observations its own actions produce, which turns failed rollouts into supervision. The paper reports GRPO's pass@1 on TerminalBench-2.0 roughly doubled (Qwen3-8B 2.7% → 5.2%, Qwen3-14B 5.2% → 10.8%). Here the option applies the same loss to **SFT** traces, where the observations follow someone else's actions. Whether it helps in SFT has not been measured by us or, as far as we know, published: treat it as an experiment and compare against the default on your own evals.

When to try it, and when not:

- **Plausible**: environments whose outputs are mostly determined by the action (file contents, compiler errors, test output, structured API responses).
- **Avoid**: outputs that are noise to the model (random IDs, timestamps, web search results, very long dumps). They add loss on tokens the model cannot predict, and on long tool outputs they can outweigh the assistant's own tokens.

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

The rationale for `progressive` / `last_heavy` is that early turns of a long trace are often exploration and the late ones produce the solution. It is a heuristic, not a measured result: compare it with `uniform` on your evals. Weights are normalised to mean 1 per conversation, so the loss scale does not change. They are applied by the cross-entropy losses (plain, chunked, Cut Cross-Entropy); SeCO, DPO and the other objectives refuse a non-uniform `turn_scaling`.

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
