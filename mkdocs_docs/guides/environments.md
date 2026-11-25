# Environments (the catalogue)

The **environment catalogue** is where RL and evaluation environments are found, audited and run. Every entry is a typed record: the class that implements the [environment protocol](rl.md#tools-and-environments), where its tasks come from, what its verifier checks, what it needs to run, its licence, and the scripted policies that audit it. Fourteen environments ship with the library; other packages add theirs through a Python entry point.

!!! abstract "When to use it"
    You want a ready environment to evaluate a checkpoint on or to train with `pgs rl`; you are writing an environment and want to know whether its reward can be gamed before a model finds out; or you publish environments for others. For the protocol itself (reset, tools, `get_reward`) see [Reinforcement Learning](rl.md#tools-and-environments); for Docker task directories see [Harbor Environments](harbor.md).

```bash
pgs env list                                   # the catalogue, by family
pgs env info text-to-sql                       # one entry: data, tools, verifier, sandbox, licence, an example prompt
pgs env test text-to-sql                       # the audit: an oracle must be paid, degenerate policies must not
pgs env run  text-to-sql --model runs/sft/final --k 4      # pass@1, pass@4, episodes/s
pgs env rows text-to-sql --n 2000 -o rows.jsonl            # rows for pgs rl, with the config lines to use
pgs env new  acme/my-env                       # a package skeleton with a test
```

## What an entry holds

An entry is an `EnvSpec` (`palingenesis.rl.catalog`), validated when it is built: a spec with an unknown family, no licence, no verifier description or no way to audit it does not register.

| Field | Meaning |
|---|---|
| `name`, `version` | `wordle`, or `<publisher>/<name>` for third-party packages. Bump the version when rows, prompts or the verifier change: results across versions do not compare. |
| `family` | `math`, `code`, `tool-calling`, `instruction-following`, `sql`, `extraction`, `games`, `search`, `shell`, `other` |
| `env`, `env_args` | The environment class (or `module:Class`) and its constructor arguments |
| `datasets` | One or more `DatasetSpec`: a **generator** (`(n, seed, **options) -> rows`, nothing downloaded), a **Hugging Face dataset** pulled at run time with a `convert` function per row, or a **file**. Each with its own licence and citation. Nothing is vendored. |
| `verifier` | What the reward checks, in words |
| `sandbox` | `none`, `process` (runs model-written code: resource limits) or `docker` |
| `difficulty` | The generator option that sets it and its levels, and the row field that records it |
| `license`, `citation` | Of the environment's code; `spec.permissive` is true when code and every declared dataset are permissively licensed |
| `trainable` | `false` = evaluation only: the verifier is not safe to optimize against (see [instruction following](#the-built-in-environments)) |
| `reference` | Measured results of reference models (`{model: {"pass@1": ...}}`). Empty until measured: `scripts/gpu_checks/envs.sh` produces them. |
| `max_turns`, `max_new_tokens`, `success_threshold`, `chance` | Episode limits, the reward that counts as a pass, and the pass rate a policy can reach honestly without solving anything (guessing, abstention rows) |
| `oracle`, `degenerate` | The audit's scripted policies (below) |

A policy is a function `policy(row, messages)` called before every assistant turn. It returns the turn: text (a final reply), a list of `(tool, arguments)` calls, or `{"content": ..., "calls": [...]}`.

## The built-in environments

Every environment below generates its own rows (seeded, unlimited, licence Apache-2.0); some also read a public dataset.

| Environment | Family | Turns | What is verified | Other datasets |
|---|---|---|---|---|
| `math-boxed` | math | 1 | The last `\boxed{}` equals the gold: exactly as a rational for numbers and tuples, by sympy equivalence for expressions. The answer must be in the form asked (a number literal, an expanded expression), so the unevaluated question does not pass. | `gsm8k` (MIT) |
| `code-unit-tests` | code | 1 | The fenced Python function passes hidden tests in a sandbox job. Call-shaped tests return values as JSON and are compared outside the sandbox. | `mbpp` (CC-BY-4.0), `humaneval` (MIT) |
| `code-stdio` | code | 1 | A stdin/stdout program prints the expected output for every hidden input; all inputs run in one process, expected outputs never enter the sandbox. | your rows |
| `tools-function-call` | tool-calling | 1 | The first turn's calls are exactly the acceptable ones (BFCL style): names, arguments, values; irrelevance rows are right only without a call. | `from_bfcl` converter |
| `tools-filesystem`, `tools-calendar`, `tools-shop` | tool-calling | up to 16 | The simulated world after the episode equals the goal world, and questions end with `ANSWER: <value>`. Outcome, not path. | |
| `instruction-following` | instruction-following | 1 | 19 checks of form (IFEval style): lengths, keywords, case, bullets, sections, endings. **Evaluation only.** | `ifeval` (Apache-2.0) |
| `text-to-sql` | sql | up to 8 | The final query returns the gold query's rows on the database and on a hidden twin with other data. Read-only tool, bounded execution. | `gretel` (Apache-2.0) |
| `json-extraction` | extraction | 1 | The JSON validates against the row's JSON Schema and every value equals the gold. | |
| `wordle`, `number-guess` | games | up to 8 / 13 | The secret was played as a guess within the budget. Native tool calls, or `<guess>...</guess>` in text. | |
| `corpus-qa` | search | up to 10 | `ANSWER:` line equals the gold by normalized exact match; the corpus is fictional, so the answer must be retrieved (BM25 `search`, `read`). | your corpus |
| `harbor` | shell | up to 40 | The task's own `tests/test.sh` in a clean container ([Harbor](harbor.md), with [harness profiles](harness-profiles.md)). | your task directories |

`instruction-following` is evaluation-only by design: its checks read form, not content, so text that says nothing and satisfies every constraint gets the full reward. As an evaluation of a model that is trying to answer, the pass rate measures instruction following; as an RL reward on its own it teaches filler. `pgs env rows` refuses it unless you pass `--allow-eval-only`; pair it with a content reward (a judge) if you train on it.

Hub datasets are downloaded by `datasets` the first time they are used (`pgs env run math-boxed --dataset gsm8k`), converted row by row, and rows the environment cannot verify are dropped (IFEval instructions without a check here; SQL that does not run in SQLite). `ifeval` uses this library's checks, not the official scorer: do not compare its numbers with published IFEval results. `gretel` has no hidden twin, so a result read off the data and hard-coded passes: evaluate on it, do not use it as an RL reward.

## Auditing reward holes

A verifier is a program a policy optimizes against. A small model finds its holes within tens of RL steps: a reward that pays an empty reply, the prompt repeated, every tool called once, or a confident claim of success, teaches exactly that. `pgs env test` plays scripted policies through the environment, with the rollout loop's own semantics (tool errors as observations, unknown tools refused, `done`, the turn limit), and no model:

| Policy | Must | A failure means |
|---|---|---|
| **oracle**: the entry's solver, reading the row's gold | pass every row | a broken task, a wrong gold, or a verifier that rejects the intended answer |
| **degenerate**: the entry's own attacks, plus `empty`, `refusal`, `echo_prompt` and `tool_spam` on every environment | pass no more than `chance` of the rows | a **reward-hacking hole**: reported with the policy, the first row it passes, and its reply |
| **reuse**: the oracle again after the attacks, on the pool's reused instances | pass | the environment carries state from one episode into the next |

An environment that raises on a degenerate input, instead of answering `Error: ...` or scoring 0, fails too. A degenerate policy that stays under `chance` but collects a mean reward above it is a warning: partial credit a policy can harvest.

```text
$ pgs env test text-to-sql
text-to-sql 1.0.0  dataset synthetic  rows 32
  oracle                              pass  32/32  mean reward  1.000     852.8 episodes/s    1705.7 steps/s
  degenerate select_star              pass   0/32  mean reward  0.000 ...
  degenerate hardcode_visible         pass   0/32  mean reward  0.000 ...
  degenerate gold_then_drop           pass   0/32  mean reward  0.000 ...
  degenerate runaway                  pass   0/32  mean reward  0.000 ...
  reuse      oracle again             pass   8/8   mean reward  1.000 ...
  audit: passed
```

The exit code is 1 on any failure, `--json` gives the report, `--all` audits the whole catalogue, and `--dataset` audits another dataset of the entry or your own file of rows. The same audit from Python:

```python
from palingenesis.rl.catalog import audit, get

report = audit(get("text-to-sql"), n=64)
assert report["ok"], report["failures"]          # report["holes"]: [{"policy", "passed", "row", "reply"}, ...]
```

What the built-in attacks are, so you can write yours in the same spirit:

| Environment | Attacks that must score 0 |
|---|---|
| `math-boxed` | the question's own expression in the box; every number of the prompt boxed; the gold boxed but not last |
| `code-*` | `sys.exit(0)`, `os._exit(0)`, a function returning an object whose `__eq__` says yes, the shown example hard-coded, echoing the input, reading every file next to the program, writing the result file, rewriting the harness from inside |
| `tools-function-call` | the call described in prose; the right name without arguments; the gold call plus a spare one; the gold call with other values |
| `tools-*` (stateful) | claiming it is done; looking and claiming; listing 40 candidate answers; deleting / cancelling everything |
| `text-to-sql` | `SELECT *`; the visible result hard-coded as literals; `DROP TABLE` through the tool and stacked on the gold query; a runaway recursive query |
| `json-extraction` | `{}`; a schema-valid object of blanks; the schema itself; the gold with one extra key; the gold with numbers as strings |
| games | stating the secret without playing it; one guess repeated; random guesses; counting up |
| `corpus-qa` | answering without searching; the question's subject as the answer; pasting the search results; listing every candidate |

!!! warning "A hole the audit found in assert-based code tests"
    Tests written as a `check(candidate)` function or as asserts on expressions (HumanEval) run through the library's assert harness (`palingenesis.rl.grading`), in the same process as the program. That harness is sound against early exits and always-equal objects, but a program written to rewrite the harness from inside (replace its comparison function, or read its nonce off the stack and report success) scores 1: on rows of that shape the audit reports it as `degenerate rewrite_harness` (`pgs env test code-unit-tests --dataset humaneval`). No in-process assert runner can close this. Tests that are calls with literal arguments and a literal expected value (the generated rows, MBPP's asserts, the APPS / LiveCodeBench shapes) do not have the hole: `code-unit-tests` runs them as calls and compares the returned values outside the sandbox. Prefer that shape for anything you train on; use `humaneval` for evaluation.

The audit shows what it tests, not more. An oracle that passes proves the rows are solvable by the intended route; attacks that fail prove those attacks fail. Add an attack for every shortcut you can think of in your task, and when a trained policy finds a new one, add it to the entry.

## Evaluating a model

```bash
pgs env run tools-shop --model runs/sft/final --k 4 --n 200 --thinking off --output shop.jsonl
```

runs k episodes per row through the RL rollout loop (`palingenesis.rl.episodes.run_episodes`): the same prompt rendering, tool-call parsing, limits and environment rewards as training, so the evaluation scores what training would see. It reports pass@1 (the mean over rows of the share of passing samples), pass@k (rows with at least one passing sample), the mean reward, how episodes ended (`stop`, `length`, `turns`, `env_done`), the tool error rate and episodes/s. `--output` writes one line per row (`{"id", "scores", "rewards", "finishes"}`), which [`pgs compare`](../reference/cli.md#compare) reads for intervals and paired tests between two checkpoints.

Options: `--backend vllm | sglang | hf`, `--temperature`, `--top_p`, `--top_k`, `--thinking on | off | default`, `--dataset`, `--option key=value` (a generator option, e.g. `--option difficulty=3`), `--max_turns`, `--max_new_tokens`, `--env_args '{...}'`.

Environments that run model-written code (`code-*`) use the Docker sandbox pool by default; `--sandbox process` runs the programs on your machine with resource limits and no isolation, and says so.

`scripts/gpu_checks/envs.sh` runs every built-in environment with one model at k = 4 on one GPU and prints pass@1, pass@4 and episodes/s per environment:

```bash
PGS_PY=/path/to/venv/bin/python PGS_SRC=/path/to/palingenesis/src bash scripts/gpu_checks/envs.sh
```

## Training on an environment

Catalogue environments own their reward (`get_reward`), so an RL config needs no `rewards:` section:

```bash
pgs env rows text-to-sql --n 2000 --seed 1 -o data/sql_rows.jsonl      # prints the lines below
```

```yaml
env:
  type: palingenesis.rl.catalog.envs.sql:SqlEnv
  max_turns: 8
data:
  dataset: data/sql_rows.jsonl
  success_threshold: 1
rollout:
  max_new_tokens: 768
  max_completion_tokens: 6144
```

Then the usual path: [`pgs screen`](rl.md#screening-the-pool-before-the-run) to keep the rows the policy solves sometimes, `pgs rl`, and [`pgs rollouts`](observability.md) to read what the policy does. Use a different seed for the rows you evaluate on. The components an environment reports are logged as `env/<name>`; `metric/...` components are logged and not added to the reward (`env/metric/valid`, `env/metric/guesses`).

## Writing an environment

```bash
pgs env new acme/price-check --family tool-calling
```

writes `price-check/` with a small working environment (one tool, a final answer), its generator, an oracle, two degenerate policies, the `EnvSpec`, a `pyproject.toml` that registers it, and a test that runs the audit. `pgs env test price-check/price_check.py:SPEC` audits it without installing anything. Replace the task and keep the shape:

```python
from palingenesis.rl.catalog import DatasetSpec, EnvSpec, scripted

class PriceCheckEnv:
    tools = ("lookup",)                              # the methods the policy may call

    def reset(self, records=None, answer="", **row): # one instance is reused: set everything here
        self.records, self.answer = dict(records or {}), str(answer)

    def lookup(self, key: str) -> str:
        """Look up the value stored under a key.

        Args:
            key: The key to look up.
        """
        return self.records[key]                     # an exception reaches the policy as "Error: ..."

    def get_reward(self, messages):                  # from the transcript, and from the state you kept
        lines = [l.strip() for l in (messages[-1].get("content") or "").splitlines() if l.strip()]
        return {"correct": float(bool(lines) and lines[-1] == f"ANSWER: {self.answer}")}

SPEC = EnvSpec(
    name="acme/price-check", version="0.1.0", family="tool-calling",
    summary="Look a value up with a tool and report it.",
    env=PriceCheckEnv, license="Apache-2.0",
    verifier='The last line of the final message is "ANSWER: <the stored value>".',
    datasets={"synthetic": DatasetSpec(generator=generate, license="Apache-2.0")},
    max_turns=4,
    oracle=scripted(),                               # plays row["solution"], then replies row["solution_answer"]
    degenerate={"guess": guess, "lookup_everything": lookup_everything},
)
```

What makes a verifier hold, from the built-in environments:

- **Score the outcome, not the transcript.** Compare the final state with a goal state; never search the reply for a claim. Where the answer is text, require it in one committed place (`\boxed{}`, an `ANSWER:` line, the last code block) and compare it exactly: a reply that lists candidates then scores 0.
- **Keep the gold out of reach.** Expected outputs stay outside the sandbox; a database the policy can query needs a hidden twin; a corpus the policy could know needs to be fictional.
- **Let the generator compute the gold with the environment itself** (run the solution's calls on the simulator, the gold query on the database), and reject rows a cheap reply would pass: an empty result, a constant, the example shown in the prompt.
- **Say the form the answer must have** when equivalence is checked (a number literal, an expanded expression): the unevaluated question is equivalent to its answer.
- **Report every component on every episode** (0 when it does not apply), and prefix what is only a statistic with `metric/`.
- **Mark it evaluation-only** when the check cannot see what you care about.

### Speed

Episodes are coroutines; tools and `get_reward` may be `async`, and synchronous ones run in threads, so a slow verifier never stalls rollouts. Three pieces keep environments cheap:

- **Instances are pooled** (`EnvPool`: construct once, `reset` per episode). Keep what is expensive across episodes at module level and key it by content: `text-to-sql` restores each database from a cached serialized image, `corpus-qa` builds a corpus's BM25 index once.
- **CPU-heavy verification goes to a process pool in batches.** `await verify(fn, *args, timeout=...)` (`palingenesis.rl.catalog.verify`) runs a module-level function in a worker process; calls made in the same loop iteration travel as one batch per worker, and each item has its own time limit (a check that hangs fails alone). `math-boxed` sends only the sympy comparisons there; numbers are decided in-process. `PGS_VERIFY_WORKERS` sets the workers (default up to 4; 0 = threads).
- **One sandbox process per submission, not per test.** `code-stdio` points the process's real stdin and stdout at a fresh file per test and executes the program again in the same process; `code-unit-tests` runs all of a row's calls in one job. The sandbox itself is the RL config's warm pool (Docker containers that multiplex jobs, no container start per job).

Scripted-oracle throughput on a laptop CPU (Apple M-series, 32 concurrent episodes; `tests/test_rl_catalog_envs.py` measures it and prints the table with `-s`):

| Environment | episodes/s | steps/s |
|---|---|---|
| `tools-function-call` | 8600 | 18700 |
| `tools-filesystem` / `tools-calendar` / `tools-shop` | 3000 / 4800 / 3800 | 18600 / 13700 / 14200 |
| `wordle` / `number-guess` | 8600 / 1900 | 17100 / 19800 |
| `corpus-qa` | 4000 | 16200 |
| `instruction-following` / `json-extraction` | 11000 / 7300 | 11000 / 7300 |
| `math-boxed` | 6700 | 6700 |
| `text-to-sql` | 730 | 1460 |
| `code-unit-tests` / `code-stdio` (process sandbox) | 113 / 110 | 113 / 110 |

A step is a tool call or the final reply the verifier scores. With a model in the loop, generation dominates everything but the code environments.

## Registering environments from your package

```toml
[project.entry-points."palingenesis.environments"]
price_check = "price_check:SPEC"       # an EnvSpec, a list of them, or a function returning either
```

After `pip install`, the entry is in `pgs env list` (tagged with your distribution's name) and works with every command. Entry points are loaded on first use; one that fails to import or validate is reported under the listing and skipped, so a broken package never hides the rest. Third-party names cannot replace built-in ones: publish as `<publisher>/<name>`. In a process, `palingenesis.rl.catalog.register(spec)` does the same without packaging.

## Importing environments from other ecosystems

`palingenesis.rl.catalog.adapters` turns environments written for other libraries into entries (experimental):

```python
from palingenesis.rl.catalog import register
from palingenesis.rl.catalog.adapters import openenv_spec, verifiers_spec

register(verifiers_spec("acme/reverse-text", "reverse_text:load_environment", license="MIT",
                        summary="Reverse a string."))
register(openenv_spec("acme/echo", base_url="http://localhost:8001", license="BSD-3-Clause",
                      summary="An OpenEnv server."))
```

| Source | What is imported | Limits |
|---|---|---|
| [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) (the Prime Intellect Environments Hub's library) | The environment's dataset, system prompt, Python tools and rubric (each reward function called with what it declares among `prompt`, `completion`, `answer`, `state`, `task`, `info`, `parser`) | `SingleTurnEnv` and `ToolEnv` only. A `MultiTurnEnv` with its own `env_response`, and stateful or sandbox tool environments, are refused: write those against the protocol. Tested against a stand-in of the interface, not the package. |
| [OpenEnv](https://github.com/meta-pytorch/OpenEnv) servers | Remote or in-process, through [`OpenEnvAdapter`](rl.md#tools-and-environments); tasks selected by a seed per row (or your rows) | The server's own rewards; GRPO groups need the task selected by the row |
| NeMo Gym, MCP | Not catalogue adapters: use `palingenesis.rl.envs.nemo_gym` / [`envs.mcp`](rl.md#mcp-servers) as `env.type` | |

**Licences.** An imported environment keeps its own licence, which the library cannot read off the object: `license=` is required, and a licence that is not permissive is refused unless you pass `allow_non_permissive=True` (the entry then shows as not permissive in `pgs env list`; `--permissive` filters). **Audits.** Imported environments come without a solver, so `pgs env test` runs the degenerate policies only and says so; pass `oracle=` for the full audit. A rubric that pays an empty reply is a hole whoever wrote it.

## Limits

- `pgs env run` and the GPU check have been exercised on CPU only (a scripted engine in the tests, a 0.6B model through the `hf` backend); `scripts/gpu_checks/envs.sh` is the validation on a GPU with vLLM, and `reference` results are empty until it has run.
- Hub dataset converters are tested on rows of the published shape, not against the downloaded datasets.
- The process sandbox limits CPU time, memory, output and open files and kills the process group; it does not isolate. Train with `sandbox.backend: docker`.
- Multi-role [scenarios](multi-agent.md) have their own registry and audit (`pgs scenario`); they are not catalogue entries.
