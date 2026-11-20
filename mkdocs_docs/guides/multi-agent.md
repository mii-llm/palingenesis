# Multi-Agent RL (Scenarios)

A **scenario** is a task played by several roles: two players of a game, a lead and its workers, a solver and a verifier. `pgs rl` trains on scenarios when its config has a `scenario:` section. Each role is a model: the trained policy, a LoRA adapter of its own on the policy's base, or a frozen partner. Without the section, `pgs rl` is single-agent RL exactly as before.

!!! abstract "When to use it"
    You want a skill that only shows up between agents: writing for a reader with no context, checking someone else's answer, delegating, playing against an opponent. Run the [verifier audit](#audit-a-verifier-before-training) first, and always train a single-agent control next to it: published multi-agent gains often disappear against a single agent given the same number of tokens (see [What to compare against](#what-to-compare-against)).

    Everything on this page is tested on CPU with tiny models. Nothing has been run on a GPU yet.

## Quick start

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

Each has a seeded generator, a programmatic verifier, an oracle partner and degenerate partners for the audit, and prompts written in English and Italian (the generator alternates them).

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

`rollout.backend: remote` takes the engines out of the trainer's process and off its GPUs. Each engine is a worker started with `python -m palingenesis.rl.pool.worker --model <policy> --port <port>` on its own GPU (several GPUs with vLLM's tensor parallelism); `rollout.engines` lists their addresses. Under torchrun, rank *r* generates with `engines[r::ranks]` and rank 0 sends the weights to all of them.

- **A weight update never waits for the engines.** The trainer copies what has to travel into host staging buffers (pinned with a GPU), which is the only part under its weights lock, and goes on with its next step. The bytes leave on the pool's own thread.
- **An engine never samples from half an update.** It receives a push into its own staging area while it keeps generating with the weights it has, applies the complete push between two of its steps, and tags everything it samples from then on with the new version. A push whose sender dies in the middle is dropped whole.
- **Two staging areas alternate** on the trainer. A new snapshot goes to the one no push is being sent from; when both are still travelling it is skipped, and the next update carries newer weights. An engine still receiving version *v* gets the newest staged version next.
- **Staleness decides what trains.** Every turn carries the version that sampled it; a row older than `rollout.max_staleness` is dropped. With `max_staleness: 0` an update waits instead: rank 0 for every engine it sends to, each other rank for its own engines, so no rank collects rows of the old version. If the engines do not confirm in 30 minutes the run stops with an error that names them.
- **A worker is checked when the trainer connects.** The trainer compares its model's architecture, stop tokens, chat-template arguments, context length and head precision with the worker's and refuses a worker that was started with others. `pgs scenario plan --engine-gpus` prints launch lines that carry them.
- **A lost worker ends the run.** Requests it was running are sent again to the engines that remain, so no rollout waits for it; the next weight update raises an error that names the worker. Restart it and resume from the last checkpoint: a restarted worker holds none of the run's weights, and the first update of a run sends everything an engine needs. The trainer does not reconnect on its own.

What travels (`rollout.push`):

| Push | Sent per update | Engine side | Use |
|---|---|---|---|
| `full` | every weight, 2 bytes per parameter | loaded as is | full-weight training; adapters under FSDP |
| `fold` | the adapters' factors, megabytes (and the base, once) | computes $W + sBA$ against the base it was sent | one adapter; any engine |
| `native` | every named adapter's factors (and the base, once) | loaded as named LoRA adapters of the engine; a request names its adapter and adapters share the batch | roles with their own adapter, or a role on the frozen base (`partner: base`) |

`auto` picks `native` when a role has its own adapter or plays the frozen base, and `fold` otherwise; `fold` is refused in those cases, because an engine that folds holds the trained policy alone. `native` is refused with `lora.dora` (the magnitudes cannot be loaded as a plain adapter). Under FSDP adapters travel folded into the full weights.

The first update of a run also carries the frozen base, so that an engine never folds into, or serves adapters next to, weights of another run. The engine applies the base together with the factors that follow it, never before. That first update is as large as a full one.

**Memory on the hosts.** With `fold` each worker keeps the base in host memory to fold against, next to the copy on its GPU: 2 bytes per parameter for a bf16 model, so 8 GB per worker for 4B parameters, 16 GB for 8B, 64 GB for 32B; eight workers of a 4B policy on one node hold 64 GB of host memory. The trainer needs the same amount once, while the first update is staged (freed when it has been sent). `native` keeps no base on the host. `full` stages every update on both sides: up to two snapshots on the trainer (2 x 2 bytes per parameter) and one on each worker.

`native` is how several roles' policies are served by **one** engine pool: vLLM loads LoRA adapters next to the base and reloads one in place under the same name while requests run (`load_inplace`), so no server per adapter is needed. With the `hf` worker the adapters are this library's own layers, which is what the tests run. The trainer's default adapter is the adapter `policy` there: a request that names no adapter samples it, not the bare base. Each update writes the adapters to `--adapter-dir` (a memory file system) in the PEFT layout and deletes the versions before the last two. The vLLM side has not been run.

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

It plays the scenario's scripted partners through the verifier: the oracles together (they must earn the reward), then each degenerate partner with oracles in the other roles (none may earn more than 0 on average). Exit code 1 on a failure. Write an oracle and the obvious degenerate policies for every scenario of your own (`Scenario.partners`); a small policy finds what the audit did not try.

## Data

`pgs scenario generate <name> --n N --seed S --out rows.jsonl` writes rows. A row is **the task alone**: its numbers, texts and gold answers, plus `scenario`, `id` and `language`. Prompts are built per role by the scenario, so one file serves every role assignment and prompt revision.

```json
{"scenario": "ledger_sum", "id": "ledger_sum-0-3", "language": "it", "item": "pere",
 "ledgers": {"nord": {"pere": 12, "mele": 40}, "sud": {"fichi": 7}}, "answer": 12}
```

Use a different seed for held-out rows (`data.eval_dataset`).

### What a run logs, and exports

With `logging.dump_trajectories: N`, every trained session is one record with its `role`, `model`, `adapter`, `episode`, `session` and `comparison` group; the first record of an episode also holds the partners' sessions and every channel message, as sent and as delivered. [`pgs rollouts`](observability.md#pgs-rollouts) reports finish reasons, loops and rewards per role.

```bash
pgs scenario export runs/x/rollouts --format sft --out sft.jsonl --roles lead --min-reward 1
pgs scenario export runs/x/rollouts --format dpo --out pairs.jsonl --margin 0.5
pgs scenario export runs/x/rollouts --format traces --out traces.jsonl
```

| Format | Rows | For |
|---|---|---|
| `sft` | `{"messages", "tools"}` of sessions that earned `--min-reward` | `pgs train`: the system distilled into one model |
| `dpo` | `{"prompt", "chosen", "rejected"}` from comparison groups whose sessions answered the same prompt, best against worst | `pgs train` with DPO |
| `traces` | every scored session, failures included | on-policy distillation (`sources.<name>.format: agent_traces`) |

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
| `replica/drift` | replicated base: 0 when the ranks hold the same adapters |

## What to compare against

Multi-agent training has a poor record against simple controls. Before trusting a gain, run:

- a **single agent with the same number of sampled tokens** (a longer budget, or self-consistency voting);
- for a verifier or evaluator, a **fixed rule** in its place (always reject, retry when truncated);
- the trained role **with an untrained partner** and the reverse: a gain that needs both trained sides may be a private convention;
- several seeds, and a held-out set that was not used to pick the checkpoint.

## Limits

- No GPU run has been made: the vLLM paths (a server per adapter, folded pushes, replicated ranks) are untested.
- Roles with their own adapter: one GPU; not with DoRA, LoRA+, FSDP or `kl_ref_reset`.
- Scenario sessions have no sandbox yet: tools are Python functions of the scenario. Harbor containers shared between roles are not implemented.
- `loss.overlong_buffer`, `loss.group_shaping`, `loss.observation_weight`, `rollout.think_skip` and `model.multimodal` are refused with a scenario.
- Remote engines take no images and no MoE routing replay yet, and their weights travel over TCP (no NCCL or RDMA transport).
