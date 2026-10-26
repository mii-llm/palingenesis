# Harness Profiles

Small agents are deployed inside **agent harnesses**: opencode, pi, mini-swe-agent, OpenHands, qwen-code, goose, hermes, terminus-2, and others. Each one speaks its own dialect:

- the system prompt and how the request is wrapped;
- tool names, argument names and JSON schemas;
- native function calling, or a text protocol (fenced commands, JSON actions, keystrokes);
- how tool results and errors are worded and truncated;
- how an episode ends (a turn without a tool call, a submit tool, a sentinel command, a completion flag);
- where the final answer comes from.

A model trained in a single dialect learns that dialect, and it degrades when a user runs it under a different harness. A **harness profile** reproduces one harness, at a pinned version, inside [`HarborEnv`](harbor.md). It is token-exact and offline: no harness process, no endpoint. That lets you train and evaluate under many dialects on the same tasks, so the model holds up under whatever harness a user picks.

```yaml
env:
  type: palingenesis.rl.envs.harbor:HarborEnv
  args:
    profiles: {palingenesis-default: 1, opencode: 1, pi: 1, mini-swe-agent: 1, openhands-sdk: 1,
               qwen-code: 1, goose: 1, hermes: 1, swe-agent: 1}
    profile_seed: 0                                           # optional
    profile_chat_kwargs: {opencode: {enable_thinking: true}}  # optional, per profile
```

`profile: opencode` runs a single profile. Without `profile`/`profiles`, HarborEnv behaves exactly as it always has (the `palingenesis-default@1` dialect, tested byte-identical). `pgs harness profiles` lists the built-in profiles with the sources each one was written from.

## How a profile is chosen: one per GRPO group

Before each `reset`, the trainer sets `env.episode_group` to the GRPO group id. The profile is then drawn by weight from `sha256(seed, group, task_dir)`:

- **All samples of a group share the profile.** The advantage compares policy samples under the same harness, not one harness against another.
- **Successive groups vary**, so a batch mixes harnesses in proportion to the weights.
- A row's own `harness_profile` field forces a profile (`"harness_profile": "pi@1.3.0"`), e.g. to pin a held-out evaluation split.
- Outside the trainer (no group id), the row's content stands in for the group, so the draw stays deterministic.

The drawn profile is recorded per rollout in `info["harness_profile"]` (the rollout dumps) and in the rewards as:

| Metric | Meaning |
|---|---|
| `env/metric/profile/<name>` | 1 for trajectories of that profile, 0 otherwise: the batch share |
| `env/metric/profile/<name>/reward` | The reward of that profile's trajectories only: per-harness learning curves |

Profile names are validated when the environment is built, not mid-run.

## Built-in profiles

| Profile | Protocol | Tools the policy sees | Episode ends | Answer |
|---|---|---|---|---|
| `palingenesis-default@2` | native | read(path, offset, limit), write, edit(path, old, new, all), glob(pat, path), grep(pat, path), bash(cmd) | turn without a tool call | last text |
| `palingenesis-default@1` | native | bash, read_file, write_file, str_replace, submit | submit | last text |
| `mini-swe-agent@2.4.6` | native | bash(command), `/bin/sh` per call, 30 s | `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` | output after the sentinel |
| `mini-swe-agent-text@2.4.6` | ```` ```mswea_bash_command ```` block, exactly one per turn | same | same | same |
| `opencode@1.19.2` | native | bash(command, timeout ms, workdir), edit(filePath, oldString, newString, replaceAll), glob, grep, read, write | turn without a tool call | last text |
| `pi@1.3.0` | native | read, bash(command, timeout s), edit(path, edits[{oldText, newText}]), write | turn without a tool call | last text |
| `openhands-sdk@1.55.1` | native | terminal (persistent), file_editor, finish(message), think | finish, or a turn without a call | finish message, else last text |
| `swe-agent@main-3ea751c` | native | bash (persistent), str_replace_editor, submit | submit twice (a review step showing the diff first) | last text |
| `qwen-code@0.25.2` | native | run_shell_command, read_file, write_file (read first), edit(file_path, old_string, new_string, replace_all), glob, grep_search | turn without a tool call | last text |
| `goose@1.55.0` | native | shell, write, edit(path, before, after), tree | turn without a tool call | last text |
| `hermes@main-e6899c8` | native | terminal (persistent), read_file, write_file (read first), patch, search_files; JSON results | turn without a tool call | last text |
| `kimi-cli@1.52.0` | native | Shell, ReadFile, Glob, Grep, WriteFile, StrReplaceFile; `<system>` status tags | turn without a tool call | last text |
| `terminus-2@harbor-0.24.0` | text: one JSON object {analysis, plan, commands[{keystrokes, duration}], task_complete} | keystrokes to a persistent tmux pane | task_complete twice in a row | none |

A bare name selects the registered default version (`palingenesis-default` is `@2`). `name@version` pins one.

**What is exact and what is paraphrased.** Parameter schemas, tool names and order, result strings and error strings are exact wherever a recording or the source code shows them. System prompts and long tool descriptions are concise paraphrases, never the long upstream text. Real harnesses send much more, e.g. qwen-code about 11k tokens, OpenHands about 5k, kimi-cli about 4k. The profiles' first prompts are 0.5–2.7k tokens with the Qwen3.5 template. Strings seen in neither a recording nor the source are marked `approx.` in the code.

**How each profile was built.** Recorded with [`pgs harness capture`](#recording-a-real-harness-pgs-harness): opencode and mini-swe-agent (through the Harbor 0.24 CLI, the real agent installed in the task container), qwen-code and kimi-cli (run locally against `pgs harness serve`). Read from the source at the pinned revision: pi, OpenHands (schemas dumped from the installed SDK), SWE-agent, goose, hermes, terminus-2 (Harbor's parser, ported). palingenesis-default@2 follows nanocode; its file tools execute nanocode's own functions.

### palingenesis-default@2 (nanocode)

The default dialect: the smallest harness that works, modelled on nanocode, a minimal Claude Code alternative.

- **System prompt:** `Concise coding assistant. cwd: /workdir` (the task's working directory).
- **Tools** (`number` parameters are sent as `integer`, as nanocode's `make_schema` does):

| Tool | Description sent to the model | Parameters |
|---|---|---|
| `read` | Read file with line numbers (file path, not directory) | `path`, `offset?`, `limit?` |
| `write` | Write content to file | `path`, `content` |
| `edit` | Replace old with new in file (old must be unique unless all=true) | `path`, `old`, `new`, `all?` |
| `glob` | Find files by pattern, sorted by mtime | `pat`, `path?` |
| `grep` | Search files for regex pattern | `pat`, `path?` |
| `bash` | Run shell command | `cmd` |

- **Results:** `read` numbers lines `   1| text` (`offset` is 0-based). `write` and `edit` return `ok`. A search with no hits returns `none`, and a command with no output returns `(empty)`. Errors are `error: <exception>`, e.g. `error: [Errno 2] No such file or directory: 'x.txt'`, `error: old_string not found`, `error: old_string appears 2 times, must be unique (use all=true)`. A missing argument gives `error: 'path'`, and an unknown tool gives `error: 'name'`.
- **How it runs:** `read`, `write`, `edit`, `glob` and `grep` execute nanocode's own Python functions with the container's `python3`, in the working directory. Results and exception wording are therefore exactly nanocode's: `glob` returns paths like `./src/a.py` sorted newest first, `grep` returns at most 50 `path:line:text` hits, and `write` does not create parent directories. `bash` runs under `/bin/sh` with stderr merged into stdout and a 30 s timeout (`(timed out after 30s)`).
- **Ending:** the episode ends at a reply without tool calls, and that reply is the answer. There is no submit tool, so the answer and the end of the episode are the same act.
- **Requirements and gaps:** the task image needs `python3`; without it the file tools answer `error: python3 is not installed in the container`. The partial output of a timed-out command is not kept. There is no output cap of its own (the trainer's `env.max_tool_output_tokens` applies).

### Per-harness notes

- **mini-swe-agent** answers a turn without a command as a format error. Upstream drops the rejected turn from its history; here it stays, because rollouts are token-exact. The submission is the command output after the sentinel line, and there is no separate answer channel.
- **opencode:** Harbor's adapter passes the instruction quoted (`"<instruction>\n"`, as recorded). Tool names are lowercased before lookup (`Bash` runs `bash`), and unknown tools go through its `invalid` tool. The recorded schemas, including a `$schema` key and opencode's own bounds, are sent as they are. Its skill, task, todowrite and webfetch tools are left out.
- **OpenHands:** legacy and malformed names are repaired as in `normalize_tool_call` (bash → terminal, str_replace_editor → file_editor). Undo is single level, and the terminal cannot keep a process running between calls. task_tracker is left out.
- **SWE-agent** expects one call per turn and drops rejected turns; here every call runs. Its submit runs `git add -A && git diff --cached` and asks for confirmation once (the review step).
- **qwen-code and destructive actions:** its prompt asks for confirmation before destructive or hard-to-reverse actions *when the current interaction mode can receive it; otherwise stop and report the blocker*. It also says authorisation *stands for the scope specified*. Under Harbor the task instruction is the user's request, so a destructive action the task explicitly asks for (delete these files, stop that process) is in scope and is scored normally. Destructive actions beyond the request are what the verifiers' collateral checks penalise.
- **goose:** the developer extension's shell, write, edit and tree tools. Its toolshim is not emulated.
- **hermes** tracks main (Harbor installs main) and is pinned to e6899c8. It uses a compact subset of the hermes-cli toolset with the advanced parameters left out, JSON result shapes following its result classes, and exact-match patching (no fuzzy strategies).
- **kimi-cli 1.52.0** is the final, archived release. Its console entry point is deprecation-gated (every command prints a notice and exits), so Harbor 0.24's kimi-cli adapter no longer runs an agent. The profile emulates the 1.52.0 agent as recorded through `python -m kimi_cli.cli`. Its StrReplaceFile replaces the first occurrence of a non-unique string, and Grep skips `.env*` files.
- **terminus-2** drives a persistent tmux pane, so the task image needs `tmux` (see below). Its parser (Harbor's, Apache-2.0) tolerates small-model JSON mistakes. Extra text is a warning. A missing closing brace, or an object inside prose, is auto-corrected with a warning. Anything else executes nothing and gets terminus's own feedback (*Previous response had parsing errors: ...*). Waits are deterministic: after the keystrokes, the pane is polled until its shell has no child process and the screen is stable, for at most `duration` (≤ 60) seconds.

### Images: tmux for terminus-2

terminus-2 needs `tmux` in the container. Build a variant of your task image and map it, without touching the tasks:

```dockerfile
FROM my-base:1
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
```

```yaml
env:
  args:
    profiles: {opencode: 1, terminus-2: 1}
    images: {"my-base:1": "my-base:1-tmux"}     # task docker_image -> image used for the episode
```

`images` applies to every profile, so give the variant the same verifier as the base.

### Answer tasks and harnesses without an answer channel

mini-swe-agent, SWE-agent and terminus-2 have no channel for a final reply. Tasks graded on the reply (*how many ERROR lines are there?*) cannot be solved under them. Keep those task families out of their groups (a row's `harness_profile`, or separate data with separate weights), or accept a 0. Under `palingenesis-default@1`, a model that calls `submit` before writing its answer also gives an empty reply. Profiles that end on a reply without tool calls avoid that by construction.

## Thinking per profile

`HarnessProfile.chat_template_kwargs` sets the chat-template kwargs of that profile's episodes. It is None for the built-ins, which means the model's default: that is what a harness gets when it sends no `chat_template_kwargs` to vLLM. Override per profile with `profile_chat_kwargs`:

```yaml
env:
  args:
    profiles: {opencode: 1, pi: 1}
    profile_chat_kwargs: {opencode: {enable_thinking: true}, pi: {enable_thinking: false}}
```

The kwargs reach the episode through `episode_info["chat_template_kwargs"]`. Precedence: `model.chat_template_kwargs` < profile < the row's own `chat_template_kwargs`.

## Unknown tools

A call to a tool the episode does not expose is answered the way that harness answers it, through the environment's optional `unknown_tool(name, arguments)` hook:

| Profile | Answer, or repair |
|---|---|
| palingenesis-default@2 | `error: 'name'` |
| opencode | Lowercases the name and retries (`Bash` → `bash`); otherwise its `invalid` tool |
| OpenHands | Repairs legacy names (`bash` → `terminal`, `str_replace_editor` → `file_editor`); otherwise `Tool 'x' not found. Available: [...]` |
| pi | `Tool x not found` |
| others | The harness's own message (see the source of each profile) |

Environments without the hook keep the trainer's message: `Error: unknown tool 'x'. Available tools: ...`.

## Evaluating under a profile

`palingenesis.rl.episodes.run_episodes` runs episodes outside training through the pipeline's own rollout loop (see [Harbor Environments](harbor.md#evaluating-and-calibrating-outside-training)):

```python
groups = run_episodes(engine, tok, rows, HarborEnv,
                      {"profiles": {"opencode": 1, "pi": 1, "mini-swe-agent": 1},
                       "images": {"my-base:1": "my-base:1-tmux"}},
                      k=4, temperature=0.7, chat_template_kwargs={"enable_thinking": False})
by_profile = {}
for group in groups:
    for t in group:
        by_profile.setdefault(t.info["harness_profile"], []).append(t.reward)
print({p: sum(r) / len(r) for p, r in by_profile.items()})
```

Report pass@1 per profile, not only the mix: a model can be strong under one harness and broken under another (e.g. never emitting mini-swe-agent's sentinel). Check feasibility first with one strong model under one profile; train under the mix.

## Writing your own profile

A profile is data plus small bindings. Each tool is a `ToolSpec(schema, bind)`. `bind(session, args)` runs ops on the episode container (`palingenesis.rl.envs.harness.ops`) and returns the harness's own output text:

```python
# my_profiles.py
from palingenesis.rl.envs.harbor import HarborEnv  # noqa: F401 (env.type: my_profiles.py:HarborEnv)
from palingenesis.rl.envs.harness import HarnessProfile, ToolSpec, ops, register
from palingenesis.rl.envs.harness.profile import fn_schema
from palingenesis.rl.envs.harness.rules import LastText, NoCall


def _shell(s, a):
    r = ops.exec_(s.ws, str(a.get("command", "")), timeout=60, workdir=s.workdir)
    s.last_exec = r
    return f"$ exit {r.exit_code}\n{r.output}" if r.output else f"$ exit {r.exit_code}"


def _cat(s, a):
    try:
        return ops.read(s.ws, s.ws.abspath(a.get("path", "")))
    except ops.OpError as e:
        return f"cat: {a.get('path')}: {'Is a directory' if e.kind == 'is_dir' else 'No such file or directory'}"


def my_agent() -> HarnessProfile:
    return HarnessProfile(
        name="my-agent", version="0.3.0",
        system="You are my-agent. Work in {workdir}. Answer in one line when done.",
        tools=[ToolSpec(fn_schema("shell", "Run a command.", {"command": {"type": "string"}}, ["command"]), _shell),
               ToolSpec(fn_schema("cat", "Print a file.", {"path": {"type": "string"}}, ["path"]), _cat)],
        stop=NoCall(), answer=LastText(),
        unknown_tool=lambda name, names, args: f"no such tool: {name}",
        emulates="my-agent 0.3.0 src/tools.py")


register("my-agent", "0.3.0", my_agent)
```

```yaml
env:
  type: my_profiles.py:HarborEnv           # importing the module registers the profile
  args: {profiles: {my-agent: 1, opencode: 1}}
```

The pieces:

| Field | |
|---|---|
| `system`, `user` | Templates. `{instruction}`, `{workdir}`, `{tools}` (a text protocol's tool list), `{system_information}` (the container's `uname`), `{date}`, `{model}`, `{git_repo}`, `{git_root}` are filled by plain replacement, so JSON examples need no brace escaping |
| `tools` | `ToolSpec(schema, bind)`, in the order the harness sends them |
| `protocol` | `Native()` (function calling), `FencedCommand` (mini text mode), `XmlTool`, `ReAct`, `JsonBlock`, `Tags`, `TerminusJSON` |
| `stop` | `NoCall()`, `Sentinel(text)`, `Submit(tool, confirm)`, `Finish(tool, field)`, `Flag(confirm)` |
| `answer` | `LastText()`, `StopAnswer(fallback)` (what the stop rule captured: a finish message, a sentinel submission) |
| `format_error`, `unknown_tool`, `alias` | The harness's wording for a malformed turn and an unknown tool, and its own name repair |
| `start` | Per-episode setup (e.g. a tmux session) returning extra template values for the first turn |
| `exec_timeout`, `runtime_max_output` | Default shell timeout; the runtime's own output cap (the profile truncates first; the runtime reads at most 4 MiB per stream, so a file read over about 3 MiB is refused as too large) |
| `chat_template_kwargs` | Thinking on or off for this harness |
| `context` | The harness's compaction policy, recorded (not emulated: rollouts are token-exact) |

**Ops.** Every tool reduces to a closed set of ops on the container:

| Op | |
|---|---|
| `exec_` | A command, stateless or persistent (`cd` and exported variables carry over); `shell="sh"` for `/bin/sh -c` |
| `read`, `list_dir` | A whole file (the profile slices, numbers and truncates it); a directory listing to a depth |
| `write` | Create or overwrite (parent directories created; an existing file keeps its mode and owner) |
| `replace`, `multi_replace`, `insert`, `undo_edit` | Exact replacement (unique, or all); several against the original text; insert after a line; single-level undo |
| `glob`, `grep` | File search by pattern (newest first); content search by extended regex |
| `Tmux`, `keys` | A persistent pane driven by keystrokes, with deterministic waits |

Ops raise `OpError(kind, **data)` on the policy's mistakes (`not_found`, `is_dir`, `no_match`, `multiple`, `identical`, `empty_old`, ...), and each profile words the error itself. One op serves opencode's `edit(filePath, oldString, newString)`, pi's `edit(path, edits[])` and OpenHands' `file_editor(str_replace)`, each with its own messages. Truncation helpers in `harness/context.py` (`head_tail_chars`, `head_lines`, `tail_lines`, `middle_clip`) reproduce each harness's wording.

## Recording a real harness: `pgs harness`

A profile is only as good as its match to the real harness. `pgs harness` records what a real harness sends to its model, so a profile can be diffed against it:

```bash
pgs harness profiles
pgs harness serve   --script turns.json --out requests.jsonl [--host 127.0.0.1] [--port 8000] [--model NAME]
pgs harness capture <agent> --task <task_dir> --script turns.json --out <dir> \
                    [--model NAME] [--harbor 0.24.0] [--host-ip IP] [--ak key=value ...] [--ae KEY=VALUE ...]
```

- **`serve`** is a stub OpenAI-compatible endpoint (`/v1/chat/completions`, streaming or not, plus `/v1/models`). It answers from a script and appends every request body, exactly as received, to a JSONL file.
- **`capture`** starts the stub on the host, then runs the real agent once through the Harbor CLI against it (`uv run --with harbor==0.24.0 harbor run -p <task> -a <agent> ...`). The agent is installed in the task container, which needs network for the install: use a copy of the task in Harbor's `[[steps]]` layout with `allow_internet = true`. `--ak` and `--ae` pass extra agent kwargs and environment variables to Harbor. `<dir>` receives `requests.jsonl`, `command.json` and `harbor.log`.

The stub points each agent at itself the way that agent accepts a custom model: opencode through a provider config (`@ai-sdk/openai-compatible`), pi with `model_api=openai-completions`, OpenHands through `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`, and the rest through `OPENAI_BASE_URL` / `OPENAI_API_KEY` with `-m openai/<model>`.

**Script format** (`turns.json`): a list of assistant turns, each `{"tool_calls": [{"name": "...", "arguments": {...}}]}` or `{"content": "..."}` (for text protocols, the text the harness parses).

- Turns advance per conversation, identified by the first two messages. A harness that drops a rejected turn from its history still advances.
- A side request without tools (a title, a summary) gets a short text.
- Past the end of the script the stub answers `Done.`.
- Responses-API and Anthropic Messages requests are recorded and refused (501), because codex and claude-code are not emulated.

Recordings are test fixtures. Conformance tests replay each recorded tool call through the profile on the same task and compare the schemas and results. Keep recordings local: some harness prompts are not yours to redistribute.

## Known gaps

Not emulated:

- compaction and summarisation (every harness but mini-swe-agent), because rollouts are token-exact;
- opencode's fuzzy edit fallbacks, pi's whitespace-normalised matching, hermes' fuzzy patch strategies (exact match here);
- goose's toolshim;
- multi-level undo (OpenHands, SWE-agent) and long-running terminal processes;
- SWE-agent's one-call-per-turn enforcement;
- `apply_patch` (codex, opencode with gpt-* models);
- codex (OpenAI Responses API only) and claude-code (Anthropic Messages only);
- openclaw (its custom-model support is partial).
