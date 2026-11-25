# Evaluation

*`pgs eval`: tasks defined in YAML, run through the code that trains.*

An evaluation harness that renders prompts, parses tool calls or scores answers differently from the trainer measures a slightly different model than the one you trained. `pgs eval` renders, parses and scores with the trainer's functions: the chat template and its arguments (thinking modes), the tool-call parser, the environments (tools, Harbor, harness profiles), the sandboxes and the rewards are the ones `pgs rl`, `pgs screen` and `pgs distill` use. Tasks with an environment run as episodes of the RL pipeline; single-turn tasks run in a leaner executor built from the same functions (a test compares the two paths record by record). Training and evaluation agree by construction.

!!! abstract "In short"
    - **A task is a YAML file**: a dataset (local file or Hub id), a field mapping, a prompt, a scorer (a built-in, any RL reward, or your function), k samples, sampling, and an environment for agentic tasks.
    - **Built-in tasks and suites** (`configs/eval/`, datasets pulled from the Hub at run time): `pgs eval --model X --suite core` runs math, knowledge, instruction following, reasoning, sandboxed code, function calling and Italian on one engine.
    - **Tasks of lm-evaluation-harness run as they are**: `--tasks path/to/lm_eval/tasks/gsm8k/gsm8k.yaml`. Same prompts, few-shot examples, filters and metrics.
    - **Built for speed**: one engine per run, every request submitted at once, n = k sampling, prefix caching, scoring in other processes while the engine generates, multiple choice with one prefill per question, a generation cache. See [Speed](#speed).
    - **Each thinking mode is a separate result**: `--modes plain,think` writes `<task>.plain.jsonl` and `<task>.think.jsonl`.
    - **Results are `pgs compare` files**: one row per item with its k scores, so `pgs compare base.jsonl candidate.jsonl --require not_worse` is the whole regression gate.
    - **`eval_tasks:`** in a `pgs train` / `pgs rl` / `pgs distill` config runs tasks during training, on the trainer's rollout engine.

```bash
pgs eval --model runs/sft/final --suite core
pgs eval --model runs/sft/final --tasks gsm8k,math500 --k 4 --modes plain,think --out eval/sft
pgs eval --config eval.yaml
pgs eval --list                      # the built-in tasks and suites
```

```text
┌───────┬───────┬───────┬─────────────────┬────────────────┬─────────────────┬────────┬───────┬───────┬─────────┬───────┬───────┐
│ task  │ mode  │ items │           score │         95% CI │          pass@k │ tokens │ trunc │ loops │ items/s │ tok/s │  time │
├───────┼───────┼───────┼─────────────────┼────────────────┼─────────────────┼────────┼───────┼───────┼─────────┼───────┼───────┤
│ gsm8k │ plain │  1319 │ 0.xxxx ± 0.xxxx │ [0.xxx, 0.xxx] │ 0.xxxx (k=4)    │    ... │  ...% │  ...% │     ... │   ... │  ...s │
```

The table's score is the mean over items with its standard error; `report.md` and `summary.json` in the output directory hold the same, with per-category scores (`--categories` prints them too). `--push_to_hub me/eval-results` uploads the directory to a private dataset repository.

The engine is vLLM in-process by default (`--backend sglang` or `hf` otherwise; `hf` is `model.generate`, for CPU tests and tiny models). One engine serves every task and mode of a run.

## A task

```yaml
# my_task.yaml
name: store_math                    # names the result files (default: the file's name)
dataset: data/store_math.jsonl      # .jsonl / .json / .parquet, or a Hugging Face dataset id
split: test                         # Hub datasets (and {split: rows} JSON files); subset: <config name>
limit: 500                          # a fixed-seed sample of the items, the same for every checkpoint (0 = all)
id_field: qid                       # the item's id column (default: its row number)
fields:                             # what the prompt and the scorer read, from the dataset's columns
  answer: {from: solution, regex: '####\s*(.+)$', replace: {",": ""}}
prompt: |
  {{question}}

  Give the final number as "Answer: <number>".
scorer: numeric
scorer_args: {field: answer}
k: 4                                # samples per item (0 = the run's --k)
sampling: {temperature: 0.7, top_p: 0.8, top_k: 20, max_tokens: 1024}
```

```bash
pgs eval --model runs/sft/final --tasks my_task.yaml,gsm8k --out eval/sft
```

**Data.** `dataset` is a local file or a Hub dataset id (with `subset`, `split`, `revision`). `data_files` reads one file of a Hub repository instead of the dataset's loader (JSON lines, JSON or parquet), and `join: {data_files: answers.json, key: id}` adds the columns of a second file matched by a key (BFCL keeps questions and accepted answers apart). Offline (`HF_HUB_OFFLINE=1`), a dataset is read from the `datasets` cache or, failing that, from its files in the Hub cache.

**Fields.** Every dataset column reaches the scorer and the environment, as in RL; `fields:` adds columns computed from them. An entry is a column name (dotted paths reach into mappings and lists: `question.0.content`) or a mapping with one source and transforms, applied in this order:

| Key | |
|---|---|
| `from: column` / `template: "{{a}} {{b}}"` / `value: x` | the source: a column, text over the columns, or a constant |
| `default: x` | used when the column is missing or null (otherwise that is an error naming the row) |
| `json: true` | parse a JSON string |
| `regex`, `group` | the last match's group (default 1); no match is an error |
| `replace: {",": ""}` | text replacements |
| `strip: false` | strings are stripped unless this is false |
| `map: {yes: A, no: B}` | a lookup; an unknown value is an error |
| `choices: true` | a list of options becomes `A. first` / `B. second` lines |
| `letter: true` | a 0-based option index becomes its letter |
| `tools: true` | function specs (`{name, description, parameters}`) become tool schemas; `dict` / `float` / `tuple` types become JSON Schema's |

**Prompt.** `prompt` is one user turn, `messages` a chat (`[{role, content}]`), `prompt_field` a column that already holds one. `{{name}}` is replaced by a field or column; a name that is neither is an error. Single braces are left alone, so `\boxed{}` and LaTeX need no escaping. `system_prompt` is prepended when the chat has no system message. A `tools` column (tool schemas) is rendered by the chat template, and the model's calls are parsed by the trainer's parser.

**Items and ids.** An item's id is `<task>/<id_field value>`; ids must be unique, because `pgs compare` pairs two runs by id. `limit` draws a fixed-seed sample (`sample_seed`) kept in file order, not the first rows: benchmark files are often sorted by subject. `cluster_field` names a column whose equal values are not independent items (the questions of one passage, the paraphrases of one question): they are resampled together by every bootstrap.

## Scorers

A scorer is an RL reward function: it declares the arguments it wants (`completion`, `reasoning`, `messages`, `prompt`, `finish_reason`, `sandbox`, `env`, any dataset column) and returns a number, or `None` when it does not apply. The completion it sees is the final assistant message with the reasoning removed, as in training.

| `scorer` | Passes when | `scorer_args` |
|---|---|---|
| `exact_match` | the answer text equals the gold text (or one of a list), whitespace-normalised | `field`, `extract` (`full`, `last_line`, `boxed`, `answer`, `regex` + `pattern`), `ignore_case`, `ignore_punctuation` |
| `numeric` | the final number equals the gold number | `field`, `extract` (`auto`: `\boxed{}`, else "Answer:", else the last number; `boxed`; `last_number`), `rel_tol`, `abs_tol` |
| `math` | the last `\boxed{}` is equivalent to the gold answer (symbolically with `math-verify`, from `palingenesis[rl]`) | `field` |
| `mcq` | the option letter the completion concludes with is the gold letter (or index): the last "answer is (X)" / "Answer: X", else a boxed letter, else the last standalone letter | `field`, `letters` |
| `regex` / `regex_match` | a pattern matches / its capture equals a gold column | `pattern`, `field`, `group` |
| `json_schema` | the completion holds JSON (fenced or bare) that validates against a schema, and equals an expected value when asked | `schema` or `schema_field`, `expected_field` |
| `function_call` | the final message's tool calls are the expected ones: `[{name: {argument: [accepted values]}}]`, `""` among the values makes an argument optional, unlisted arguments fail, `[]` expects no call | `field`, `ordered` |
| `ifeval` | the response follows every instruction of an IFEval item | `loose`, `level` (`prompt` / `instruction`), `checker` |
| `code` | the last fenced Python block passes the row's hidden tests in the sandbox (`sandbox:` section, as in RL) | `field`, `timeout`, `max_tests`, `partial` |
| `judge`, `equivalence` | an LLM judge behind an OpenAI-compatible endpoint | `url`, `model`, `template`, `field` |
| `module:function`, `path/to/file.py:function` | your reward function | its arguments |

Any other RL reward works by name (`exact`, `choice`, `boxed_choice`, `qa_match`, ...). Several weighted rewards go under `rewards:` exactly as in an RL config, instead of `scorer`. A sample passes (pass@k) at or above `threshold` (0.5).

`json_schema` uses the `jsonschema` package when it is installed and a built-in validator of the common keywords otherwise.

### LLM judges must be validated

A judge that has not been compared with human labels measures the judge. A task whose scorer is `judge` or `equivalence` (or any scorer with a `judge_validation:` section) runs the judge on labelled examples before the evaluation and refuses to start below the agreement you ask for:

```yaml
scorer: equivalence
scorer_args: {url: http://localhost:8001/v1, model: judge-model, field: answer}
judge_validation:
  file: data/judge_labels.jsonl     # rows: question (or prompt), completion, the reference column, label (0 / 1)
  min_agreement: 0.9                # default 0.8
  min_examples: 50                  # default 20: agreement on a handful of labels is noise
```

The summary records the agreement, Cohen's kappa and the number of examples. `judge_validation: {skip: true}` uses the judge unvalidated and says so in the summary. Label examples from the model family you evaluate: a judge validated on fluent answers can still fail on a small model's malformed ones.

## Built-in tasks

| Task | Dataset (license) | Items | Score |
|---|---|---|---|
| `gsm8k` | `openai/gsm8k`, test (MIT) | 1,319 | `numeric`: the final number |
| `math500` | `HuggingFaceH4/MATH-500` (MIT) | 500 | `math`: the last `\boxed{}`, symbolic equivalence with `math-verify` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`, test (MIT) | 12,032 | `mcq`: the letter, up to ten options |
| `ifeval` | `google/IFEval` (Apache-2.0) | 541 | `ifeval`: prompt-level strict accuracy |
| `bfcl_simple` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, v3 simple (Apache-2.0) | 400 | `function_call`: BFCL-style accepted values |
| `bbh` | `lukaemon/bbh`, 27 tasks (MIT) | 6,511 | `exact_match` on the line after "Answer:", zero-shot; per-task scores |
| `humaneval` | `openai/openai_humaneval` (MIT) | 164 | `code`: the function passes the hidden tests in the sandbox |
| `mbpp` | `google-research-datasets/mbpp`, sanitized test (CC-BY-4.0) | 257 | `code`: the three asserts pass in the sandbox |
| `mmlu` | `cais/mmlu`, test, 57 subjects (MIT) | 14,042 | log-likelihood of " A" ... " D", 5 dev examples per subject (lm-eval's `mmlu`) |
| `mmlu_pro_mc` | `TIGER-Lab/MMLU-Pro`, test (MIT) | 12,032 | log-likelihood of the letter, zero-shot |
| `mmmlu_it` | `openai/MMMLU`, `IT_IT` (MIT) | 14,042 | `mcq`, Italian prompt; per-subject scores |
| `belebele_it` | `facebook/belebele`, `ita_Latn` (CC-BY-SA-4.0) | 900 | `mcq`, Italian prompt; questions of one passage are a cluster |
| `gpqa_diamond` | `Idavidrein/gpqa` (CC-BY-4.0, gated) | 198 | `mcq`, options shuffled once per question |

Every task file starts with a comment that states the prompt format, the license and what the score is and is not comparable with. `gpqa_diamond` is not part of a suite: the dataset is gated and its authors ask that examples are not published, so keep its sample files private.

| Suite | Tasks |
|---|---|
| `core` | `gsm8k`, `math500`, `mmlu_pro` (2,000 items), `ifeval`, `bbh` (1,350), `humaneval`, `mbpp`, `bfcl_simple`, `mmmlu_it` (1,000), `belebele_it` |
| `quick` | 200 items each of `gsm8k`, `math500`, `mmlu_pro`, `ifeval` |
| `multilingual` | `mmmlu_it` (2,000), `belebele_it` |
| `loglikelihood` | `mmlu`, `mmlu_pro_mc` |

A suite is a YAML file with a `tasks:` list (`--suite my_suite.yaml`); limits are fixed-seed samples, the same items for every checkpoint.

Nothing is vendored: the datasets are downloaded from the Hub on first use and cached. Override a task without copying its file: `tasks: [{task: mmlu_pro, limit: 2000}]`.

- **`math500` without `math-verify`** falls back to a normalised text and numeric comparison, which marks equivalent forms (`\frac{1}{2}` and `0.5`) wrong. Install `palingenesis[rl]` and keep it the same across the runs you compare.
- **`ifeval`** uses the benchmark's reference checker when it is installed (`instruction_following_eval`, or lm-eval's copy) and otherwise a port of its checks (`palingenesis.eval.ifeval`, Apache-2.0 like the reference). The port differs in two places: sentence and capital-word counts use NLTK only when it is installed with its data (a rule-based splitter otherwise), and the 31 prompts with `language:response_language` need `pip install langdetect`; without it they are left out of the score and counted in `unscored_items`. Compare runs made in the same environment.
- **`bfcl_simple`** passes the functions as tools through the chat template and reads the call with the RL tool-call parser. Its scorer follows BFCL's rules (function name, every argument among its accepted values, optional arguments may be left out) but is not the leaderboard's checker, so its numbers are not leaderboard numbers. The other single-turn categories are a three-line task away:

    ```yaml
    tasks:
      - task: bfcl_simple
        name: bfcl_parallel
        data_files: BFCL_v3_parallel.json
        join: {data_files: possible_answer/BFCL_v3_parallel.json, key: id}
    ```

    Your own function-calling data needs a `tools` column (schemas or function specs with `tools: true`) and a column of expected calls: `[{"get_weather": {"city": ["Rome", "Roma"], "unit": ["celsius", ""]}}]`.

## Multiple choice by log-likelihood

A task with `output: choices` generates nothing: each row has a list of continuations (`choices_field`, default `choices`) and the index of the right one (`gold_field`, default `gold`), and the answer is the choice the model gives the highest log-probability after the prompt (`target_delimiter`, default a space, goes between them). This is how lm-evaluation-harness scores MMLU, ARC or HellaSwag.

Other harnesses send one request per choice, each a full prefill of the prompt with vLLM's `prompt_logprobs`. `pgs eval` folds an item's choices into a trie over their tokens and reads each node with a one-token request that returns the next token's top log-probs: four single-token choices (" A" to " D") cost one request and one prefill; longer choices share their common prefix and reuse the prompt from the prefix cache. A wanted token outside the engine's top candidates (`engine.max_logprobs`, default 32) is read with a second request restricted to it and an anchor token, whose known log-prob restores the scale, so every value is the model's exact log-prob. With `--backend hf` the model is in the process and choices are scored with padded forward passes.

The summary reports `acc` (the score) and `acc_norm` (log-likelihood per character, as lm-eval), and the sample file each choice's log-likelihood. Chat models can be scored after their chat-templated prompt (`template: chat`); few-shot text prompts are `template: completion`.

## Tasks of lm-evaluation-harness

```bash
pgs eval --model M --tasks path/to/lm_eval/tasks/gsm8k/gsm8k.yaml              # as it is
pgs eval --model M --tasks 'path/to/lm_eval/tasks/mmlu/default/mmlu_*.yaml' --num_fewshot 5
pgs eval --model M --tasks path/to/task.yaml --template chat                    # through the chat template
```

A task file in lm-eval's format is read directly (`palingenesis.eval.lmeval`). The supported subset: `dataset_path` / `dataset_name`, the split keys, `output_type: generate_until | multiple_choice`, `doc_to_text` / `doc_to_target` / `doc_to_choice` (a column, a Jinja template, a literal or `!function`), `description`, `target_delimiter`, `fewshot_delimiter`, `num_fewshot`, `fewshot_config` (`sampler: default | first_n`), `generation_kwargs` (`until`, `max_gen_toks`, `temperature`), `filter_list` (`regex`, `take_first`, `remove_whitespace`, `lowercase`, `uppercase`), `metric_list` (`exact_match` with its options; `acc`, `acc_norm`; any metric a `process_results` function returns as a number), `process_docs`, `include`, `metadata.version`. Anything else is refused with an error naming the key, never silently approximated.

The prompt a harness builds is reproduced exactly: description, examples (text, delimiter, target), the question; the default sampler draws with `random.Random(1234)` in document order; `first_n` takes the first examples; continuations are tokenized as lm-eval does (`encode_pair`). Tasks run in completion mode (raw text, no chat template) unless `--template chat`. Each (metric, filter) is reported under lm-eval's name (`exact_match,strict-match`); the first is the item's score. `!function` runs Python from the task's directory, as lm-eval does: read a task before you run it.

One extension: `fewshot_config.group_by: <column>` takes a question's examples from the few-shot rows with the same value (the built-in `mmlu` is one file for 57 subjects).

**Agreement is measured, not assumed.** `python -m palingenesis.eval.conformance --pgs <result.jsonl> --lmeval <lm-eval output dir> --task gsm8k --metric exact_match,strict-match` reports both accuracies, the difference in standard errors, and how many documents the two harnesses score alike; `scripts/gpu_checks/eval_bench.sh` runs it on GSM8K and a multiple-choice task after timing the harnesses.

## Speed

| What | How |
|---|---|
| one engine per run | every task and mode reuses it; scorer processes start while it loads |
| everything at once | a task's requests enter the engine's continuous batch together, in prefix order (shared few-shot examples, system prompts and tool schemas are prefilled once: `prefix_caching` is on) |
| n = k | the k samples of an item are one request (one prefill); a greedy item is one request whatever k |
| tokenization | prompts are rendered once per task and mode and tokenized in one batched call |
| scoring | runs while the engine generates: plain scorers in `run.score_workers` processes (automatic from 256 samples), so a symbolic math check never takes the interpreter from the engine's thread; judges and sandboxes on threads |
| multiple choice | one prefill per question instead of one per choice ([above](#multiple-choice-by-log-likelihood)) |
| engine settings | `max_num_seqs` 512, `max_num_batched_tokens` 16384, no engine-side detokenization unless a task has stop strings |
| several GPUs | one engine per visible GPU, automatically (`--devices 0,1`); `engine.tensor_parallel_size` for a model that needs several |
| small models | `--replicas N` puts N engines on each GPU: a sub-2B model leaves a large GPU idle behind the engine's single scheduling thread |
| cache | what the engine returned is cached by (checkpoint fingerprint, prompt tokens, sampling, k, seed) in `~/.cache/palingenesis/eval` (`--cache_dir`, `--no_cache`): a second run of the same checkpoint generates nothing, and a changed scorer re-scores |
| resume | `--resume` keeps the finished items whose request (prompt tokens, sampling, k) is unchanged |
| speculative decoding | `engine.speculative_config: {...}` is passed to vLLM |

Each summary has `seconds`, `items_per_second`, `tokens_per_second`, `time_to_first_result`, `engine_requests`, `cache_hits` and, where `nvidia-smi` is present, `gpu_util_mean` / `gpu_util_max` / `gpu_memory_max_gb`. If `gpu_util_mean` is well under 90% on a small model, raise `--replicas`.

!!! note "Measured where?"
    The comparison with lm-evaluation-harness and lighteval is `scripts/gpu_checks/eval_bench.sh` (same model, tasks, sampling, GPU and vLLM build; pre-registered targets in its header). Numbers belong in this guide only once that script has produced them on your hardware.

## Contamination

```bash
pgs eval --model M --tasks gsm8k --contamination data/sft.jsonl,data/rl/
```

Flags the items whose prompt shares a 13-gram of words (`run.contamination_ngram`) with the training data (`.jsonl`, `.json`, `.parquet`, `.txt`; directories are walked), and reports the score on the flagged and on the clean items in each summary (`contamination`). The training files are streamed once; memory does not grow with them.

## Agentic tasks

A task with an `env:` section is an episode, not a completion. The section is the RL config's own `env:` (and `sandbox:`), so an environment written for training evaluates as it trains: tool calls are executed, results are truncated and returned, the turn and token budgets apply, and the environment's `get_reward` scores the episode.

```yaml
name: harbor_dev
dataset: data/harbor_dev.jsonl
prompt_field: messages
env:
  type: palingenesis.rl.envs.harbor:HarborEnv
  args: {profiles: {opencode: 1, pi: 1}}    # harness profiles, as in training
  max_turns: 16
  max_concurrent: 32
  tool_timeout: 120
k: 4
sampling: {max_tokens: 1024, max_completion_tokens: 6144}   # per turn, and for the whole episode
```

`max_tokens` is one assistant turn; `max_completion_tokens` is the whole episode (default: `max_tokens` × `env.max_turns`). The summary's `finish` shares say how episodes end (`stop`, `length`, `turns`), and the samples file keeps every turn. For lower-level control (replaying a recorded prefix, stop resampling), `palingenesis.rl.episodes.run_episodes` is the same loop as a function.

## Thinking modes and sampling

```bash
pgs eval --model m --tasks gsm8k --modes plain,think
```

| Mode | Chat template arguments |
|---|---|
| `default` | none: the template's own default |
| `plain` | `enable_thinking: false` |
| `think` | `enable_thinking: true` |

Each mode is run and reported on its own: a model can gain in one and lose in the other, and an average hides both. Define others (a template with different switches, a mode with its own sampling) under `modes:`:

```yaml
modes:
  think:  {chat_template_kwargs: {enable_thinking: true},  sampling: {temperature: 0.6, top_p: 0.95, top_k: 20, max_tokens: 8192}}
  plain:  {chat_template_kwargs: {enable_thinking: false}, sampling: {temperature: 0.7, top_p: 0.8,  top_k: 20}}
```

**Sampling** options are `temperature`, `top_p`, `top_k`, `presence_penalty`, `max_tokens` (one turn, thinking included) and `max_completion_tokens`. An option left unset falls through: command line over mode over task over the run's `sampling:`. A temperature unset everywhere is the model's own `generation_config.json`, the sampling a server uses by default (with its `top_p` and `top_k`); without one, greedy at k = 1 and temperature 1 at k > 1. The summary records what was used and where the temperature came from.

## Outputs and statistics

Per task and mode, in the output directory:

| File | |
|---|---|
| `<task>.<mode>.jsonl` | one row per item: `{"id", "scores": [k numbers], "cluster"?}` plus per-sample `finish`, `tokens`, `loop` lists. This is the `pgs compare` format. |
| `<task>.<mode>.samples.jsonl` | one row per sample: score, reward components, finish reason, tokens, the completion and its reasoning, the prompt (first sample), the turns of an episode (`--no_samples` to skip) |
| `<task>.<mode>.summary.json` | the summary below; `summary.json` collects the run's |

| Summary field | |
|---|---|
| `mean_score` | the mean over items of each item's mean score, with a percentile bootstrap interval over items (clusters) |
| `pass@k` | the unbiased estimator (Chen et al. 2021) for k = 1, 2, 4, ... up to the samples drawn, with bootstrap intervals |
| `wilson` | the Wilson interval of the first sample's pass rate (one binomial draw per item) |
| `completion_tokens` | mean, median, 95th percentile and maximum of the generated tokens |
| `finish`, `truncated_share`, `turn_limit_share`, `error_share` | how samples ended; `truncated_share` is the share cut by the token budget |
| `loop_share` | the share of samples the loop detector of the rollout dumps flags (a repeated span in the answer, or one tool call made three times) |
| `unscored_samples`, `unscored_items` | samples no reward could score (a sandbox or judge outage, an unsupported check); items with none scored are left out of the result file |
| `stderr`, `metrics`, `categories` | the standard error of the mean score; each reward component's mean (a task with several rewards, lm-eval metrics per filter); the score per value of the task's `category_field` |
| `sampling`, `chat_template_kwargs`, `seconds`, `tokens_per_second`, `items_per_second`, `time_to_first_result` | what was run, and how long generation and scoring took |

The intervals come from `palingenesis.evalstats`, the code behind `pgs compare`. `--resume` keeps the items already in the result files whose request is unchanged (after an interruption); without it the files start over. Result files are appended item by item while a task runs and rewritten in item order when it ends.

## Comparing checkpoints

Run the same tasks, modes, k and sampling on both checkpoints, then pair them by item:

```bash
pgs eval --model runs/base       --tasks gsm8k,ifeval --modes plain,think --k 4 --out eval/base
pgs eval --model runs/candidate  --tasks gsm8k,ifeval --modes plain,think --k 4 --out eval/candidate

pgs compare eval/base/gsm8k.think.jsonl eval/candidate/gsm8k.think.jsonl
pgs compare eval/base/ifeval.plain.jsonl eval/candidate/ifeval.plain.jsonl --require not_worse --margin 0.02
```

`pgs compare` reports the difference of the mean score and of every pass@k with a paired bootstrap interval, a permutation p-value and, at k = 1, the exact McNemar test; `--require better` / `not_worse` turn it into an exit code. Pairing matters: two accuracies with overlapping intervals can still differ significantly on the same items, and the reverse. A difference whose interval contains 0 is not an improvement, whatever the two means say.

Two modes of one checkpoint compare the same way (`gsm8k.plain.jsonl` against `gsm8k.think.jsonl`): is thinking worth its tokens on this task?

## During training

```yaml
# in a pgs train, pgs rl or pgs distill config
eval_tasks:
  tasks: [gsm8k, my_task.yaml]     # built-in names, task files or inline tasks
  every: 200                       # steps (0 = the trainer's eval_every: data.eval_every / train.eval_every)
  limit: 200                       # items per task: the same fixed sample at every evaluation (0 = all)
  modes: [plain, think]            # empty = the trainer's own model.chat_template_kwargs
  k: 1
  sampling: {max_tokens: 1024}
```

| Trainer | Engine |
|---|---|
| `pgs rl`, `pgs distill` | the rollout engine that samples the training rollouts (vLLM, SGLang or hf), after loading the newest weights: no second copy of the model. |
| `pgs train` | the model itself (`model.generate`, batched), on a single process. Multi-GPU runs skip the tasks with a warning: run `pgs eval` on the saved checkpoints. |

Metrics are logged as `eval/tasks/<task>/<mode>/score`, `pass@k`, `truncated`, `loops` and `tokens`. The result files of each evaluation go to `<output_dir>/eval_tasks/step_<n>/`, in the compare format, so two steps of a run compare directly:

```bash
pgs compare runs/rl/eval_tasks/step_0/gsm8k.plain.jsonl runs/rl/eval_tasks/step_400/gsm8k.plain.jsonl
```

The tasks run every `every` steps and at the last step (`final: false` to skip it); `pgs rl` and `pgs distill` also run them before the first step, which gives the baseline to compare with. They take training time: on an RL run the engine evaluates instead of sampling, so keep `limit` small and the full evaluation for checkpoints. An evaluation that fails (a dataset that cannot be loaded, an environment error) is logged and switched off; it never stops the run. Multi-role scenario runs skip `eval_tasks`.

## Pitfalls

**Contamination.** A benchmark's test split that is in the training data measures memory. Before trusting a gain, check the training sets for the benchmark's items: exact and near-duplicate questions, and derived sets (many public instruction and reasoning datasets contain GSM8K, MATH or MMLU items, sometimes paraphrased or translated). `pgs prepare --mode pretrain` has n-gram decontamination for raw text; for chat data, compare normalised questions. A model that scores far above its size class on one public benchmark and not on a fresh set of the same kind is the usual sign. Keep a private task of your own next to the public ones: it cannot leak.

**Sampling settings.** Scores depend on them, sometimes more than on the checkpoint. Compare runs made with identical sampling, and evaluate at the sampling the model is deployed with: greedy decoding hides the loops a sampled deployment shows, and some thinking models degrade or loop under greedy decoding. With k > 1 at temperature 0 the k samples are identical and pass@k equals pass@1 (the config warns). `max_tokens` is part of the setting: a budget too small for the thinking mode turns reasoning into truncation, which reads as a lower score. Check `truncated_share` before reading the accuracy.

**Thinking modes.** Evaluate every mode you ship. Training in one mode moves the other (distillation with thinking on can shorten or lengthen answers with thinking off), and `default` is whatever the chat template does, which changes between model versions. Give the thinking mode its own token budget, and read `completion_tokens` with the score: an accuracy gain paid for with three times the tokens is a different result.

**One sample is one draw.** At temperature > 0, k = 1 on a few hundred items has an interval of several points. Use k ≥ 4 for decisions, report the interval, and decide with `pgs compare` on paired items, not by eye.

**Answer extraction is part of the benchmark.** `numeric` and `mcq` accept common answer formats; a model that changes its format between checkpoints can move the score without changing its ability. Read the samples file of a surprising result (`score: 0` rows with a correct answer in another format), and keep the task's prompt and scorer fixed across the runs you compare.

**Unscored samples are not failures.** A sandbox outage or an unreachable judge leaves samples unscored instead of scoring them 0; a large `unscored_samples` means the evaluation did not measure what you think. Fix the infrastructure and `--resume`.

**Environment differences.** Optional packages change scorers (`math-verify`, `langdetect`, the IFEval reference checker). Run the base and the candidate in the same environment.

## In Python

```python
from palingenesis.eval import EvalConfig, load_task
from palingenesis.eval.cli import run

config = EvalConfig()
config.model.path = "runs/sft/final"
config.run.k, config.run.modes, config.run.output_dir = 4, ["plain", "think"], "eval/sft"
config.engine = {"max_model_len": 8192}
config.tasks = [load_task("gsm8k"), load_task("my_task.yaml")]
config.validate()
report = run(config)                       # starts the engine, runs every task and mode, writes the files
for summary in report["results"]:
    print(summary["task"], summary["mode"], summary["mean_score"], summary["pass@4"], summary["truncated_share"])
```

With a rollout engine you already hold (a trainer's, a server's), `Evaluator(tokenizer, engine, model).run(load_task("gsm8k"), "think", k=4, out_dir="eval/sft")` evaluates one task in one mode and returns its summary. `palingenesis.eval.scorers` has the scorers as plain functions (`numeric("Answer: 18", answer="18")`).

## Troubleshooting

| Symptom | Cause |
|---|---|
| `unknown option task.X (did you mean Y?)` | a misspelt task option: options are checked, nothing is ignored silently |
| `placeholder {{x}} is not a field or a column` | the prompt names a column the dataset does not have; the message lists the row's columns |
| `fields.answer: regex ... does not match` | a row whose gold column has another format: fix the regex, or add `default:` |
| `cannot load <dataset>` offline | the dataset is in neither cache: run once with network, or point `dataset` at a local file |
| every sample `finish: length` | `max_tokens` too small for the mode (thinking), or the model loops: read the samples file |
| `unscored_items` > 0 on `ifeval` | `langdetect` is not installed (31 prompts need it) |
| `the judge agrees with the labels on N%` | the judge fails its validation: fix its prompt or model before using its scores |
| vLLM runs out of memory next to another job | `--engine.gpu_memory_utilization 0.4` (the default is 0.9: evaluation alone on the GPU) |
