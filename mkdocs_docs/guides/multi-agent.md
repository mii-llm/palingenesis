# Multi-Agent RL (Scenarios)

A **scenario** is a task played by several roles: two players of a game, a lead and its workers, a solver and a verifier. `pgs rl` trains on scenarios when its config has a `scenario:` section. Each role is a model: the trained policy, a LoRA adapter of its own on the policy's base, or a frozen partner. Without the section, `pgs rl` is single-agent RL exactly as before.

!!! abstract "When to use it"
    You want a skill that only shows up between agents: writing for a reader with no context, checking someone else's answer, delegating, playing against an opponent. Run the [verifier audit](#audit-a-verifier-before-training) first, and always train a single-agent control next to it: published multi-agent gains often disappear against a single agent given the same number of tokens (see [What to compare against](#what-to-compare-against)).

    Everything on this page is tested on CPU with tiny models. Nothing has been run on a GPU yet.

## Quick start

You already have a dataset and a reward for `pgs rl`. Add a `scenario:` section that names a **pattern**, check it, train:

```yaml
# rows as for single-agent RL: {"question": "...", "answer": "72"}
data:     {dataset: data/gsm8k_train.jsonl}
rewards:  {correct: {fn: math}}             # the same section as for single-agent RL
scenario: {type: solve_check}               # a solver and a checker
loss:     {advantage_std: role, advantage_std_floor: 0.05}
```

```bash
pgs scenario check --config rl.yaml         # your rows, your rewards, the prompts of each role, the audit
pgs rl --config rl.yaml
```

No Python and no generated data: see [Where the data comes from](#where-the-data-comes-from). The library also ships scenarios with a generator of their own:

```bash
pgs scenario generate take_away --n 4000 --seed 0 --out data/take_away.jsonl
pgs scenario audit take_away                      # the verifier pays oracles and no degenerate policy
pgs rl --config configs/rl_scenario_take_away.yaml
```

```yaml
# configs/rl_scenario_take_away.yaml (excerpt)
scenario:
  type: take_away
roles:
  first: {}                                 # the trained policy
  second: {partner: "scripted:oracle"}      # a perfect opponent, frozen
loss:
  advantage_std: role
  advantage_std_floor: 0.05
```

## Where the data comes from

!!! note "What is a multi-agent row"
    A row is a **single-agent task**: a prompt and whatever checks the answer (a gold answer, hidden tests, a task directory). Nothing in it says who talks to whom.

    - The **pattern** (or scenario) supplies the protocol: the roles, what each one sees, the prompts, who is paid for what.
    - The **rewards** supply the verifier: the functions of the `rewards:` section score a final answer exactly as in single-agent RL.

    So one file serves single-agent RL, every pattern, every role assignment and every prompt revision.

There are three ways to get rows, from least to most work.

| Path | Rows | Verifier | You write |
|---|---|---|---|
| [(a) A built-in generator](#a-a-built-in-generator) | `pgs scenario generate` | the scenario's own | nothing |
| [(b) Your RL dataset plus a pattern](#b-your-rl-dataset-plus-a-pattern) | the JSONL / parquet / Hub dataset you use for `pgs rl` | your `rewards:` section | YAML |
| [(c) Your own scenario](#c-your-own-scenario) | anything | your code | a Python class |

### (a) A built-in generator

`pgs scenario generate <name> --n N --seed S --out rows.jsonl` writes the rows of a [starter scenario](#the-starter-scenarios): synthetic tasks with a known solution, for learning the machinery and for tests. Use a different seed for held-out rows (`data.eval_dataset`).

```json
{"scenario": "ledger_sum", "id": "ledger_sum-0-3", "language": "it", "item": "pere",
 "ledgers": {"nord": {"pere": 12, "mele": 40}, "sud": {"fichi": 7}}, "answer": 12}
```

### (b) Your RL dataset plus a pattern

A pattern is a scenario with no generator and no verifier of its own. It reads a row's prompt the way `pgs rl` does (`data.prompt_field`, else the first of `messages`, `prompt`, `question`, `problem`; `data.system_prompt` applies) and scores every final answer with the run's `rewards:` functions and weights.

| Pattern | Roles | What happens | Rows need |
|---|---|---|---|
| `solve_check` | `solver`, `checker` | The solver answers. The checker judges two candidates per episode, one right and one wrong, and corrects the wrong one. | a prompt, a gold answer |
| `handoff` | `writer`, `reader` | The writer sees a text and writes a note without knowing the question. Readers answer from the note alone. | a text (`context`), a question, a gold answer |
| `lead_workers` | `lead`, `worker` | The lead splits the task into `SUBTASK:` lines, workers answer them, the lead merges into the final answer. | a prompt, a gold answer |
| `plan_execute` | `planner`, `executor` | The planner reads a [Harbor](harbor.md) task and writes a plan. The executor works in the task's container; the task's tests are the reward. | `task_dir`, `messages` (`pgs harbor rows`) |

`pgs scenario patterns` prints this list with every option, and the row fields under the names the options use.

**A worked example.** `data/gsm8k_train.jsonl` holds single-agent rows; the `answer` column is the final number:

```json
{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"}
```

The config is the single-agent one plus `scenario:` (the whole file is `configs/rl_scenario_solve_check.yaml`):

```yaml
model:
  policy: <your model>
data:
  dataset: data/gsm8k_train.jsonl
  eval_dataset: data/gsm8k_test.jsonl
rewards:
  correct: {fn: math, args: {field: answer}}     # the built-in math reward, as in configs/rl_math.yaml
scenario:
  type: solve_check
  args:
    gold_field: answer                           # the column the oracle partners and the right candidate read
rollout:
  group_size: 4
  max_new_tokens: 1024
loss:
  advantage_std: role
  advantage_std_floor: 0.05
train:
  output_dir: runs/solve_check
```

Check it before training:

```bash
pgs scenario check --config rl.yaml --n 200
```

```text
scenario    solve_check: roles solver, checker; outcome solver
rewards     correct (math, x1)
rows        7473 in data/gsm8k_train.jsonl; 200 checked, 200 fit the scenario

--- prompt of solver (en) ---
[user] Natalia sold clips to 48 of her friends in April, ...

End your reply with exactly one line of the form:
ANSWER: <your final answer>

--- prompt of checker (en) ---
[user] A colleague was given this task:
...
Their final answer:
72

Check the answer yourself. Reply with exactly one line `VERDICT: ACCEPT` if it is right, or `VERDICT: REJECT` ...

oracle      solver                   +1.000
oracle      checker                  +1.000
degenerate  solver/silent            +0.000
degenerate  checker/always_accept    +0.000
degenerate  checker/always_reject    -0.250
...
scenario check: passed
```

It does four things, and exits 1 if any fails:

1. **Fields.** Every sampled row must have what the pattern reads. A miss names the row, its columns and the option that fixes it (`scenario.args.prompt_field`, `gold_field`, `material_field`, ...). `pgs rl` makes the same check on its first 200 rows and stops before the first rollout.
2. **Rewards.** The `rewards:` section must resolve (a built-in name, or `module:function`).
3. **Prompts.** One episode per language is played with the oracle partners and each role's opening prompt is printed, so you read what the model will read.
4. **The audit, on your rows.** The oracle partners must earn the reward and no degenerate partner may ([Audit a verifier before training](#audit-a-verifier-before-training)). The oracle answers with the gold column, so this also tests your data against your reward: if the column holds a worked solution (`"... #### 72"`) and the reward compares the whole string, the oracle is not paid and the check says so.

Then train, and train the single-agent control on the same rows next to it:

```bash
pgs rl --config rl.yaml
pgs rl --config configs/rl_math.yaml        # the same data and reward, one agent
```

**The final answer.** By default a pattern asks every answering role to end with one line `ANSWER: ...`, and gives the rewards the text of that line as the completion. A reply without the line, with two of them, or cut by a token limit has not answered and scores 0. Rewards that parse a format of their own (a `\boxed{}` answer, a fenced code block) want `answer_tag: ""`: the prompt is then the row's own, untouched, and the whole reply is the completion.

**Options every pattern takes** (`scenario.args`):

| Option | Default | Meaning |
|---|---|---|
| `prompt_field` | `""` | the prompt's column; empty: `data.prompt_field`, else the first of `messages`, `prompt`, `question`, `problem` |
| `gold_field` | `answer` | the reference answer's column (a string, or a list whose first entry is used), for the oracle partners and for the right candidate of `solve_check`. Rewards read their own column (`rewards.<name>.args.field`) |
| `answer_tag` | `ANSWER` | the tag of the final-answer line; `""`: no protocol, the whole reply is the answer |
| `language_field`, `language` | `language`, `en` | prompts are written in English and Italian and chosen per row from this column (`en` / `it`); any other value, or no column, uses `language` |
| `threshold` | `data.success_threshold` | an answer is "right" from this reward on. It does not change what a session is paid; `solve_check` uses it to tell its right candidate from its wrong one |
| `weights` | per pattern | `{role: {component: weight}}`, as for every scenario |

`plan_execute` takes `prompt_field`, `language_field`, `language` and `weights` only.

#### `solve_check`

In every episode the checker judges two candidates in two separate sessions: one that your rewards score as right and one that they score as wrong. One of them is the solver's own answer, the other its twin of the opposite class. So "always accept" earns +1 −1 = 0 in every episode and "always reject" at most −0.5, whatever the solver's accuracy.

- Solver right: the wrong twin is the wrong answer of another solver attempt of the same episode (`solver_samples`, default 2); else a numeric gold one slip away (±1, ±10, digits reversed); else an entry of the row's `distractors` column (`distractor_field`); else the gold answer of another row.
- Solver wrong: the right twin is the gold answer.
- Every twin is scored by your rewards before it is shown. If no pair of one right and one wrong answer exists, no checker session runs in that episode (`role/solver/rewards/metric/checked`).

| Role | Component | Value |
|---|---|---|
| solver | `correct` | the rewards' weighted sum on its final answer |
| checker | `verdict` | right accept +1, wrong accept −1, right reject −1, wrong reject +0.5, +1 with a right correction, no verdict −1 |

Other options: `checker_samples` (2), `max_candidate_chars` (400). The candidates are final answers only, never the reasoning. One cue remains: a gold answer is written the way the dataset writes it, a solver's answer the way the model does. Watch `metric/twin_gold` against `metric/twin_slip`, `twin_distractor` and `twin_other_row`; the module docstring has the full analysis.

#### `handoff`

```yaml
rewards:  {f1: {fn: qa_match, args: {metric: f1}}}
scenario:
  type: handoff
  args: {material_field: context, max_note_chars: 400}
```

The writer is given the text (`material_field`, default `context`) and not the question. The reader is given the note (cut to `max_note_chars`, default 600) and the question (the row's prompt; `question_field` names another column), never the text. `reader_samples` readers (2) answer from the same note. With `control: true` (the default) one more reader answers without a note; its session is logged and not trained.

| Role | Component | Value |
|---|---|---|
| writer | `lift` | its readers' mean reward minus the control's: what the note added |
| writer | `readers` | its readers' mean reward (weight 0; the writer's reward with `control: false`) |
| reader | `correct` | the rewards' weighted sum on its final answer |

A constant note earns a lift of 0, and so does a note for a question the reader can answer alone. What the reward cannot see: a note that copies a text shorter than the limit (the check warns), and a private code between a writer and a reader trained together (train against `roles.reader.train: false` or `partner: base`).

#### `lead_workers`

The lead writes at most `max_workers` (3) lines `SUBTASK: ...`; each goes to one worker, which sees nothing else. The lead's conversation then continues with the workers' replies and it gives the final answer. The work and merge stages run `team_samples` (2) times from one plan, and each team's merge `merge_samples` (2) times from the same replies. With `control: true` (the default) one more session is the lead answering alone.

| Role | Session | Component | Compared with |
|---|---|---|---|
| lead | plan | `total`: the mean reward of the merges that followed it. `lift` (weight 0): `total` minus the control's | the plans of the row's other episodes |
| lead | merge | `correct`: the rewards' weighted sum on its final answer | the other merges of the same team, which read the same replies |
| worker | | `team`: the mean reward of its team's merges | the workers of the other teams that got the same subtask |

These follow the [credit rules](#credit): a session is compared only with sessions that could have been it. A sub-answer cannot be checked by itself (the dataset has gold for the task, not for its parts), so a worker shares its team's reward and its credit is noisy with few teams. A lead that ignores its workers and solves alone is paid in full: `metric/control` measures exactly that, inside the run.

#### `plan_execute`

```yaml
data:     {dataset: data/harbor_rows.jsonl, prompt_field: messages}
scenario:
  type: plan_execute
  timeout: 1800
  args: {executor_samples: 2, max_turns: 16, max_concurrent: 16}
roles:
  executor: {max_completion_tokens: 8192}
```

Rows are [Harbor rows](harbor.md) (`pgs harbor rows`); there is no `rewards:` and no `env:` section. The planner gets the task's instruction and no tools. Each of `executor_samples` executors gets the row's own conversation with the plan appended (cut to `max_plan_chars`, default 4000), and a `HarborEnv` of its own: the same container, tools and tests as single-agent RL on Harbor. `env_args` are HarborEnv's arguments; `env` names another environment class with the same interface.

| Role | Component | Value |
|---|---|---|
| planner | `executed` | the mean reward of its executors |
| planner | `lift` | `executed` minus a plan-less executor's reward (weight 0; needs `control: true`, one more container per episode) |
| executor | `tests` | the task's tests |

`pgs scenario check --config ... --n 5` starts containers: its oracle planner writes the task's `solution/solve.sh` as the plan and its oracle executor runs it.

### (c) Your own scenario

When no pattern has the shape you need, write the protocol and the verifier as a class ([What a scenario is made of](#what-a-scenario-is-made-of)) and use it with `scenario.type: "my_file.py:Handoff"`. Give it rows any way you like (a `generate` classmethod makes `pgs scenario generate` and `pgs scenario audit` work), and write an oracle and the obvious degenerate partners. To score with the run's rewards instead of your own verifier, subclass `palingenesis.rl.scenarios.library.patterns.Pattern`: `await self.pay(session, row)` grades a session's final answer with the `rewards:` functions.

### From rollouts back to datasets

Multi-agent rollouts become ordinary single-agent datasets again. With `logging.dump_trajectories: N`, every trained session is one record with its `role`, `model`, `adapter`, `episode`, `session` and `comparison` group; the first record of an episode also holds the partners' sessions and every channel message, as sent and as delivered. [`pgs rollouts`](observability.md#pgs-rollouts) reports finish reasons, loops and rewards per role.

```bash
pgs scenario export runs/x/rollouts --format sft --out sft.jsonl --roles lead --min-reward 1
pgs scenario export runs/x/rollouts --format dpo --out pairs.jsonl --margin 0.5
pgs scenario export runs/x/rollouts --format traces --out traces.jsonl
```

| Format | Rows | For |
|---|---|---|
| `sft` | `{"messages", "tools"}` of sessions that earned `--min-reward`: one conversation per session, as that role saw it | `pgs train`: the system distilled into one model |
| `dpo` | `{"prompt", "chosen", "rejected"}` from comparison groups whose sessions answered the same prompt, best against worst | `pgs train` with DPO |
| `traces` | every scored session, failures included | on-policy distillation (`sources.<name>.format: agent_traces`) |

Each row keeps its `role`, `reward` and source (step, group, episode, session). For example, `--roles checker --min-reward 1` of a `solve_check` run is a dataset of verified checks, and `--roles solver --format dpo` pairs a right and a wrong answer to the same question.

## What a scenario is made of

A scenario is `async def run(episode, row)`. It uses five things:

| Thing | What it is | In code |
|---|---|---|
| role | a name the config maps to a model | `roles.<name>` |
| session | one context window of one role | `await episode.run(role, messages)`; or `s = await episode.open(role, messages)`, then `await s.say()` / `await s.say(reply)` |
| channel | text handed from one session to another: size-limited, redacted, logged | `episode.channel(name, max_chars, redact).send(text)` |
| branch | a fork: *n* sub-episodes that share everything before it | `await episode.branch(n, fn)` |
| credit | named reward components per session | `session.reward(name, value)` |

Two more calls cover work that does not block its caller: `episode.spawn(coroutine)` starts a task the episode waits for before scoring, and `live.notify(text)` leaves a message a running session reads at its next turn boundary. A tool ends its own session by raising `EndSession`.

A session of the trained policy is **one token-exact trajectory**, however many turns it has. Another role's message arrives as the next user message (or as a tool result), is context, and is never trained. The trainer scores exactly the tokens the sampler produced, as in single-agent RL.

```python
from palingenesis.rl.scenarios import Scenario
from palingenesis.rl.scenarios.base import user

class Handoff(Scenario):
    name, roles, outcome = "handoff", ("writer", "reader"), "reader"

    @classmethod
    def generate(cls, n, seed=0):
        ...                                     # rows: the task alone, with its gold answer

    async def run(self, episode, row):
        writer = await episode.run("writer", [user(row["material"] + "\nWrite notes for a colleague.")])
        note = episode.channel("note", max_chars=600, redact=(str(row["answer"]),)).send(writer.text)

        async def read(sub):                    # two readers of the same note: one comparison group
            reader = await sub.run("reader", [user(f"Notes:\n{note}\n\n{row['question']}")])
            reader.reward("correct", check(reader.text, row["answer"]))
            return reader

        readers = await episode.branch(2, read)
        writer.reward("readers", sum(r.rewards["correct"] for r in readers) / len(readers))
```

Use it with `scenario.type: "my_file.py:Handoff"`.

## Agent-harness patterns

Agent harnesses compose agents in a handful of ways: a main agent that calls subagents as tools, background agents that report when they finish, mailboxes, handoffs, group chats, pipelines. A model that will run inside one should be trained on episodes shaped the same way. This is how each pattern is written with the runtime, and what it cannot do yet.

| Pattern | Status | How |
|---|---|---|
| Subagent as a tool (fresh context, final text returned as the tool result; nested) | supported | a tool that runs a session: `async def task(brief): return (await episode.run("worker", [user(brief)])).text` |
| Parallel subagents | supported | the tool calls of one turn run concurrently; or `asyncio.gather` / `episode.branch` in the scenario |
| Background subagent with a completion notification | supported | the tool calls `episode.spawn(work())` and returns at once; `work` ends with `lead.notify(text)`, which the lead reads with its next tool result |
| Messages between agents, a child reporting to its parent, steering a running agent | supported at turn boundaries | `live.notify(text)` from any code that holds the session; mailboxes are queues in `episode.shared` |
| Resumable subagent (continue with its context intact) | supported | keep the `episode.open(...)` handle and `say(message)` again: one trajectory |
| A tool that ends its session (submit, finish) | supported | the tool raises `EndSession(result)` |
| Handoff (control moves to another agent) | supported | a tool raises `EndSession`, the scenario then runs the next role on the conversation so far (`session.messages`) |
| Prompt chaining, routing, orchestrator-workers, evaluator-optimizer | supported | plain scenario code over `run` and `open` |
| Role-play pairs, debate, voting, mixture of agents | supported | `open` sessions relayed by the scenario (`take_away`); `branch` and a merge function |
| Group chat (selector, round robin) | supported | several open sessions; the scenario or a selector session picks the speaker and passes on the unseen messages |
| Shared task board, blackboard | supported | a dict in `episode.shared` behind tools |
| Verification gates (reviewer, test runner, stop hook) | supported | a session or a Python tool; a stop hook answers an open session's last message instead of closing it |
| Workflow scripts and pipes | supported | a scenario is a workflow script; a channel is a pipe |
| MCP tools, A2A tasks | by wrapping | MCP tools as async functions; an A2A task is a spawned session with notifications |
| Fork (a child that inherits the running parent's context) | **not supported** | a tool cannot read its caller's live transcript; a *finished* session can be continued from its `messages` |
| Interrupting a session in the middle of a turn | **not supported** | delivery is at turn boundaries only |
| A shared filesystem, worktrees, containers | **not supported** | scenario tools are Python functions; no sandbox per episode yet |
| A token budget shared by a whole delegation tree | **not supported** | limits are per session, plus `scenario.timeout` |
| Anything that outlives an episode (schedules, long waits) | **not supported** | an episode is the unit of training |

When the *model* decides the shape (how many subagents, with which briefs), sessions of one role are no longer interchangeable: give children a `slot` (their depth, or their index) so that only comparable sessions share a baseline, or pay them on a result that can be checked by itself, as `ledger_sum` does.

## The starter scenarios

These are the scenarios of [path (a)](#a-a-built-in-generator). Each has a seeded generator, a programmatic verifier, an oracle partner and degenerate partners for the audit, and prompts written in English and Italian (the generator alternates them).

| Scenario | Roles | The task | Reward |
|---|---|---|---|
| `take_away` | `first`, `second` | A subtraction game with a known solution; heap, largest move and variant change per row; half of the rows are lost for the first player. Each player answers `TAKE: n`. | `win` +1 / −1. Breaking the protocol forfeits: −1, and 0 for the opponent. `optimal`: share of moves that kept a won position (weight 0, logged). |
| `ledger_sum` | `lead`, `worker` | The lead must total one item over regional ledgers it cannot see. It asks each region's clerk through an `ask(region, question)` tool; the clerks run in parallel. | lead `total` 1 / 0; `coverage` (weight 0). worker `value` +1 / −1 on **the question it received**; unjudgeable questions stay out of the loss. |
| `checked_answer` | `solver`, `verifier` | The solver answers an arithmetic question. The verifier judges two candidates per episode, one right and one wrong (the solver's answer and its opposite-class twin). | solver `correct` 1 / 0 / −0.5 (no answer). verifier `verdict`: right accept +1, wrong accept −1, wrong reject −1, right reject +0.5, +1 with the right correction. |

Three properties are checked by tests, not asserted:

- In `take_away`, the two players' rewards never add up to more than 0, for every pair of scripted policies. Cooperation cannot be paid.
- In `checked_answer`, "always accept" earns exactly 0 in every episode and "always reject" −0.25, whatever the solver's accuracy.
- In `ledger_sum`, a lead that does not ask cannot reach the total, and "always 0" earns a worker −0.5 on average.

## Roles

```yaml
roles:
  lead:   {share: 2.0, max_turns: 8}
  worker: {adapter: clerk}                        # its own LoRA adapter (needs lora.rank)
```

| Option | Default | Meaning |
|---|---|---|
| `adapter` | `""` | a LoRA adapter of its own on the policy's frozen base; roles naming the same adapter share it. Empty: the policy's own weights |
| `partner` | `""` | a frozen partner: `base` (the policy's base with its adapters off), `scripted:<name>`, an OpenAI-compatible endpoint URL, or `module:callable` |
| `partner_args` | `{}` | arguments for an endpoint or a class partner (`model`, `api_key`, `temperature`, `max_tokens`) |
| `train` | `true` | `false`: the policy plays the role and its tokens do not train |
| `system_prompt` | `""` | put before the scenario's own instructions |
| `max_turns`, `max_new_tokens`, `max_completion_tokens` | `0` | the role's limits; 0 = the scenario's or the rollout's |
| `chat_template_kwargs` | `{}` | e.g. `{enable_thinking: true}` for one role only |
| `share` | `1.0` | the role's part of the loss, relative to the other trained roles |

A role the config does not mention is the trained policy. One policy playing every role (told apart by its prompts) is the default and costs nothing extra.

!!! warning "Token budgets"
    `rollout.max_completion_tokens` bounds a whole session, all its turns. Its default is one turn's worth (`rollout.max_new_tokens`), which ends a multi-turn session after its first turn. Set it (or `roles.<name>.max_completion_tokens`) for games and for leads that call tools.

### Several policies on one GPU

With `lora.rank` and `roles.<name>.adapter`, each named adapter is a separate policy on one frozen base: megabytes each, one copy of the base in the trainer. Each adapter has its own optimizer group and is **clipped by itself** (a loud role cannot shrink another's update), and all of them take one optimizer step per training step.

The rollout engine needs each policy as a model:

- `rollout.backend: hf` generates with the trainer's own model, one adapter per call. Slow; tests and debugging.
- `rollout.backend: vllm_server` starts one vLLM server per adapter (and one for `partner: base`), each fed that adapter's folded weights after every step. Set `rollout.gpu_memory_utilization` per server. This path is **not validated on a GPU**.
- The in-process `vllm` backend holds one model, so it is refused with several adapters.

## Credit

A baseline is only valid among sessions that could have been each other. Sessions are compared within a **comparison group**: the same role, at the same place of the episode, in the episodes of one row. A branch makes a smaller group: the *n* sessions that started from the same state.

$$
A = \frac{r - b_{\text{group}}}{\max(\sigma_{\text{role}},\ \text{floor})}
$$

where $r$ is the session's reward, $b_{\text{group}}$ the mean reward of its comparison group's episodes (`loss.baseline: leave_one_out`: of the other episodes), and $\sigma_{\text{role}}$ the standard deviation of the centred rewards of the same role over the batch (`loss.advantage_std: role`), never less than `loss.advantage_std_floor`.
{: .pg-where }

- **Per role.** With one scale for the whole batch (`batch`, the single-agent default), the role with the widest reward sets the size of every role's gradient.
- **The floor.** A role whose rewards hardly differ in a batch would otherwise have that noise scaled to unit size.
- **Episodes, not sessions.** When one episode has several sessions in a group (the workers of one lead), the baseline counts each episode once.
- **Shares.** With `loss.aggregation: prompt` each comparison group weighs the same, so a role with more groups per row would get that many times the loss, by accident of the scenario's shape. The step's loss is $\sum_\rho s_\rho L_\rho$ with $s_\rho$ the role's `share` over the sum of the shares of the roles present.
- **Zero variance.** A comparison group whose sessions all got the same reward carries no gradient and is dropped; a row is kept if any of its groups has something to train.

A reward trains only the sessions it was given to. A scenario's verifier decides who gets what: in `checked_answer` the solver is not paid for the verifier's verdicts, and the verifier is not paid for the solver's accuracy.

## Asynchrony and speed

- **One event loop, one batch per engine.** Sessions are coroutines. A session waiting for another role's message holds no thread and no engine slot. Whatever turns are ready, of any role, episode or row, join their engine's running batch at its next step.
- **Prefix reuse.** A session's turns extend one token sequence, and the sessions of a branch start from the same tokens: the engine's prefix cache serves both. `engine/prefix_hit` reports it.
- **Rollouts overlap training.** `rollout.max_staleness: 1` lets the next batch's episodes run while the trainer steps. Every turn records the policy version that sampled it; the loss's importance ratio corrects for the lag; a row older than the bound is dropped whole.
- **One optimizer step per training step**, for every adapter. Nothing is stepped once per role instance.
- **Failures stay local.** An episode that raises or exceeds `scenario.timeout` is cancelled and left out of training; the row's other episodes train. `episode/failed` counts them.

### Several GPUs

Episodes never leave their rank: every GPU runs a trainer rank, its own engine and its share of the rows, and sessions talk through in-process queues. Two layouts:

| Layout | How | Collectives per step |
|---|---|---|
| FSDP (default under torchrun) | the policy is sharded; gathered weights go to each rank's engine | per layer, in every forward and backward |
| `train.replicate_base: true` (with `lora.rank`) | every rank holds the whole frozen base and a copy of the adapters; only the adapters' gradients are summed | one, of a few megabytes, plus the scalar normalisers |

With a replicated base each rank folds and loads its own engine's weights locally, on the rollout thread. Every 50 steps the ranks compare a checksum of their adapters (`replica/drift`, 0 when they agree) and take rank 0's on a mismatch. A two-rank run is tested on CPU; nothing on GPUs.

### Engines on other GPUs or nodes

`rollout.backend: remote` takes the engines out of the trainer's process and off its GPUs. Each engine is a worker started with `python -m palingenesis.rl.pool.worker --model <policy> --port <port>` on its own GPU (a worker over several GPUs with vLLM's tensor parallelism is planned for by `pgs scenario plan --engine-tp` but has never been run); `rollout.engines` lists their addresses. Under torchrun, rank *r* generates with `engines[r::ranks]` and rank 0 sends the weights to all of them.

- **A weight update never waits for the engines.** The trainer copies what has to travel into host staging buffers (pinned with a GPU), which is the only part under its weights lock, and goes on with its next step. The bytes leave on the pool's own thread.
- **An engine never samples from half an update.** It receives a push into its own staging area while it keeps generating with the weights it has, applies the complete push between two of its steps, and tags everything it samples from then on with the new version. A push whose sender dies in the middle is dropped whole.
- **Two staging areas alternate** on the trainer. A new snapshot goes to the one no push is being sent from; when both are still travelling it is skipped, and the next update carries newer weights. An engine still receiving version *v* gets the newest staged version next.
- **Staleness decides what trains.** Every turn carries the version that sampled it; a row older than `rollout.max_staleness` is dropped. With `max_staleness: 0` an update waits instead: rank 0 for every engine it sends to, each other rank for its own engines, so no rank collects rows of the old version. If the engines do not confirm in 30 minutes the run stops with an error that names them; an update a worker could not apply stops it at once with the worker's reason.
- **The first update of a run always waits**, whatever `max_staleness` is, and until it has arrived a worker tags what it samples with version -1, which is never trained. A worker that outlived a crashed or finished run therefore never contributes rows sampled by that run's weights. Rows whose version is above the trainer's own are dropped too.
- **A worker is checked when the trainer connects.** The trainer compares its model's architecture, stop tokens, chat-template arguments, context length and head precision with the worker's and refuses a worker that was started with others. `pgs scenario plan --engine-gpus` prints launch lines that carry them.
- **A lost worker ends the run.** Requests it was running are sent again to the other engines of the same trainer rank, so no rollout waits for it (a rank whose only engine is gone fails its rollouts at once); the next weight update raises an error that names the worker. Restart it and resume from the last checkpoint: a restarted worker holds none of the run's weights, and the first update of a run sends everything an engine needs. The trainer does not reconnect on its own.

What travels (`rollout.push`):

| Push | Sent per update | Engine side | Use |
|---|---|---|---|
| `full` | every weight, 2 bytes per parameter | loaded as is | full-weight training; adapters under FSDP |
| `fold` | the adapters' factors, megabytes (and the base, once) | computes $W + sBA$ against the base it was sent | one adapter; any engine |
| `native` | every named adapter's factors (and the base, once) | loaded as named LoRA adapters of the engine; a request names its adapter and adapters share the batch | roles with their own adapter, or a role on the frozen base (`partner: base`) |

`auto` picks `native` when a role has its own adapter or plays the frozen base, and `fold` otherwise; `fold` is refused in those cases, because an engine that folds holds the trained policy alone. `native` is refused with `lora.dora` (the magnitudes cannot be loaded as a plain adapter). Under FSDP adapters travel folded into the full weights.

The first update of a run also carries the frozen base, so that an engine never folds into, or serves adapters next to, weights of another run. The engine applies the base together with the factors that follow it, never before. That first update is as large as a full one.

**Memory on the hosts.** With `fold` each worker keeps the base in host memory to fold against, next to the copy on its GPU: 2 bytes per parameter for a bf16 model, so 8 GB per worker for 4B parameters, 16 GB for 8B, 64 GB for 32B; eight workers of a 4B policy on one node hold 64 GB of host memory. The trainer needs the same amount once, while the first update is staged (freed when it has been sent). `native` keeps no base on the host. `full` stages every update on both sides: up to two snapshots on the trainer (2 x 2 bytes per parameter) and one on each worker.

`native` is how several roles' policies are served by **one** engine pool: vLLM loads LoRA adapters next to the base, so no server per adapter is needed. Each pushed version is an adapter of its own to vLLM (its own name, id and directory): requests in flight keep the version they started with, and the prefix cache is reset on every update. Start the workers with `--max-loras` at least twice the number of adapters (the planner's launch lines do). With the `hf` worker the adapters are this library's own layers, which is what the tests run. The trainer's default adapter is the adapter `policy` there: a request that names no adapter samples it, not the bare base. Each update writes the adapters to `--adapter-dir` (a memory file system) in the PEFT layout and deletes the versions before the last two. The vLLM side has not been run: whether vLLM's LoRA loader accepts the module names as written, and whether the base loads by name into an engine built with `enable_lora`, are the first things to check on a GPU. The handshake covers the model's architecture, stop tokens, chat-template arguments, context length, head precision and the LoRA rank served; other engine settings (`kv_cache_dtype`, quantization, `--vllm-args`) are the operator's to keep equal.

The transport is a TCP stream per engine, not an NCCL collective. Two reasons, both derived, not measured:

- The trainers' process group is theirs: FSDP's gathers and the batch normalisers run on it, so engines in that group would have to take part in every collective, and a second NCCL communicator driven from a background thread could not be exercised without GPUs.
- The traffic does not call for it. With adapters an update is megabytes: about 160 MB for 4B parameters at rank 16, 0.05 s on a 25 Gbit link. A full push of 4B parameters is 8 GB: about 2.6 s at 25 Gbit, 6.5 s at 10 Gbit, travelling while the trainer steps. NCCL over InfiniBand would cut that several times, which matters only when full-weight updates take longer than a training step.

`pool/push_mb`, `pool/push_s`, `pool/lag` (versions the slowest engine is behind), `pool/skipped` and `pool/engines` (workers still generating) are logged every step.

The wire format is a length-prefixed JSON header followed by the raw bytes of the tensors, in the hosts' native byte order: trainer and workers must have the same endianness. A tensor of a dtype the format does not know is refused by name.

`pgs scenario plan --config rl.yaml --params-b 4 --nodes 2 --gpus-per-node 8 --engine-gpus 8 --hosts node1,node2` prints this layout for a cluster: the trainers' GPUs, one launch line per engine worker, the `rollout.engines` list, the estimated size of an update and the time until the last engine has it (the engines of other nodes share rank 0's link), and the trainer's launch command. Without `--engine-gpus` it sizes engines on the trainers' own GPUs: ranks, estimated trainer memory, the memory share per engine, the launch command, and what does not fit.

A two-rank trainer sharing two engine workers, pushes overlapping training, the staleness drop, a sender dying mid-push, a worker dying or restarting, and a role on the frozen base are tested across processes on CPU. Nothing on GPUs.

Roles with their own adapter need `rollout.backend: remote` (named adapters of one engine pool) to use several GPUs; with `vllm_server` they stay on one.

## Audit a verifier before training

```bash
pgs scenario audit checked_answer --n 500
```

It plays the scenario's scripted partners through the verifier: the oracles together (they must earn the reward), then each degenerate partner with oracles in the other roles (none may earn more than 0 on average). Exit code 1 on a failure. A [pattern](#b-your-rl-dataset-plus-a-pattern) is audited on your own rows and rewards by `pgs scenario check --config rl.yaml`, which runs the same audit. A partner that declares a `row` parameter gets the episode's row, which is how an oracle reads the gold answer. Write an oracle and the obvious degenerate policies for every scenario of your own (`Scenario.partners`); a small policy finds what the audit did not try.

## Metrics

| Metric | Meaning |
|---|---|
| `role/<r>/reward`, `role/<r>/rewards/<component>` | per role, over the batch's kept rows |
| `role/<r>/advantage_std`, `role/<r>/advantage_abs` | the role's reward spread and advantage size |
| `role/<r>/policy/mismatch_k3`, `abs_ratio_dev`, `clipped`, `entropy`, `kl_ref` | the policy statistics of the RL guide, over that role's trained tokens (`role/<r>/trained_tokens`) |
| `grad_norm/<adapter>` | each adapter's gradient norm before its own clip |
| `pool/push_mb`, `pool/push_s`, `pool/lag`, `pool/skipped` | remote engines: an update's size and travel time, versions the slowest engine is behind, snapshots skipped |
| `role/<r>/sessions`, `turns`, `completion_len`, `truncated`, `skipped` | what the role's sessions looked like |
| `episode/sessions`, `episode/messages`, `episode/failed` | per episode |
| `reward`, `outcome/pass@1` | of the outcome role (`scenario.outcome`) |
| `role/<r>/rewards/metric/control`, `.../lift` | patterns: the control session's reward (a reader without a note, a lead alone, an executor without a plan) and what the role added to it |
| `role/solver/rewards/metric/checked`, `metric/twin_<source>`, `metric/team` | `solve_check`: episodes with a balanced pair, where the twin came from, and whether the pair's final answer was right |
| `replica/drift` | replicated base: 0 when the ranks hold the same adapters |

## What to compare against

Multi-agent training has a poor record against simple controls. Before trusting a gain, run:

- a **single agent with the same number of sampled tokens** (a longer budget, or self-consistency voting);
- for a verifier or evaluator, a **fixed rule** in its place (always reject, retry when truncated);
- the trained role **with an untrained partner** and the reverse: a gain that needs both trained sides may be a private convention;
- several seeds, and a held-out set that was not used to pick the checkpoint.

## Limits

- No GPU run has been made: the vLLM paths (a server per adapter, folded pushes, replicated ranks) are untested.
- Roles with their own adapter: not with DoRA, LoRA+, FSDP or `kl_ref_reset`. Several GPUs need `rollout.backend: remote` (with `train.replicate_base` for several trainer ranks); with `vllm_server` they stay on one.
- `set_adapter` (what the trainer selects a micro-batch's adapter with) is process-wide, so that a forward recomputed under gradient checkpointing uses the right adapter; rollout threads select theirs with `use_adapter`, which wins inside its block.
- Scenario sessions have no sandbox of their own: tools are Python functions of the scenario. `plan_execute` gives each executor session a Harbor container through `HarborEnv`; a container shared between roles is not implemented, and harness profiles (`env_args.profile` / `profiles`) are refused, because a session's conversation and text protocol are the scenario's. `plan_execute` is tested with a stand-in environment and has not been run against Docker.
- Patterns read rows whose prompt ends with a user message of text: no images, and a row's `tools` column is not passed on. Their prompts exist in English and Italian only.
- In `lead_workers`, a sub-answer is not checked by itself: workers share their team's reward.
- `loss.overlong_buffer`, `loss.group_shaping`, `loss.observation_weight`, `rollout.think_skip` and `model.multimodal` are refused with a scenario.
- Remote engines take no images and no MoE routing replay yet, and their weights travel over TCP (no NCCL or RDMA transport).
