# Harness profiles (HarborEnv)

> The user guide is on the docs site: `mkdocs_docs/guides/harness-profiles.md` (and `guides/harbor.md` for the
> environment itself). This page keeps the design notes.

Small agents are deployed inside agent harnesses (opencode, pi, mini-swe-agent, OpenHands, ...), and each harness
speaks its own dialect: system prompt, tool names and argument names, observation format, error wording, how an
episode ends and where the final answer comes from. A **harness profile** reproduces one harness at a pinned version
inside `HarborEnv`, token-exact and offline, so the policy can be trained and evaluated under several dialects on the
same Harbor tasks. Design: `research/tiny-reliable-agent/harness/REPORT.md` §4.

```yaml
env:
  type: palingenesis.rl.envs.harbor:HarborEnv
  args:
    profiles: {palingenesis-default: 1, opencode: 1, pi: 1, mini-swe-agent: 1, openhands-sdk: 1}
    profile_seed: 0                                        # optional
    profile_chat_kwargs: {opencode: {enable_thinking: true}}   # optional, per profile
    images: {"harbor3-base:1": "harbor3-base:3"}            # terminus-2 needs tmux
```

`profile: opencode` runs a single profile. Without `profile`/`profiles` HarborEnv behaves exactly as before
(the `palingenesis-default@1` profile, tested byte-identical). The name `palingenesis-default` alone is version 2,
nanocode's dialect (below); a mix of `@1` and `@2` logs both under `metric/profile/palingenesis-default`. A row's `harness_profile` field forces a profile.
`pgs harness profiles` lists them.

## Profiles

| profile | protocol | tools | stop | answer | prompt* |
|---|---|---|---|---|---|
| `palingenesis-default@1` | native | bash, read_file, write_file, str_replace, submit | submit / no call | last text | — |
| `palingenesis-default@2` | native | read(path, offset, limit), write, edit(path, old, new, all), glob(pat, path), grep(pat, path), bash(cmd); nanocode's own functions via the container's python3, `/bin/sh`, 30 s | turn without a tool call | last text | 1 line |
| `mini-swe-agent@2.4.6` | native | bash(command), `/bin/sh` per call, 30 s | `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` | output after the sentinel (no answer channel) | 0.8k |
| `mini-swe-agent-text@2.4.6` | ```` ```mswea_bash_command ```` block, exactly one | same | same | same | 0.6k |
| `opencode@1.19.2` | native | bash(command, timeout ms, workdir), edit(filePath, oldString, newString, replaceAll), glob, grep, read, write | turn without a tool call | last text | 2.2k |
| `pi@1.3.0` | native | read, bash(command, timeout s), edit(path, edits[{oldText, newText}]), write | turn without a tool call | last text | 1.3k |
| `openhands-sdk@1.55.1` | native | terminal (persistent), file_editor, finish(message), think | finish, or a turn without a call | finish message, else last text | 2.6k |
| `swe-agent@main-3ea751c` | native | bash (persistent), str_replace_editor, submit | submit twice (a review step with the diff first) | last text | 1.3k |
| `qwen-code@0.25.2` | native | run_shell_command, read_file, write_file (read first), edit(file_path, old_string, new_string, replace_all), glob, grep_search | turn without a tool call | last text | 2.4k |
| `goose@1.55.0` | native | shell, write, edit(path, before, after), tree | turn without a tool call | last text | 1.0k |
| `hermes@main-e6899c8` | native | terminal (persistent), read_file, write_file (read first), patch, search_files; JSON results | turn without a tool call | last text | 1.6k |
| `kimi-cli@1.52.0` | native | Shell, ReadFile, Glob, Grep, WriteFile, StrReplaceFile; `<system>` status tags | turn without a tool call | last text | 2.7k |
| `terminus-2@harbor-0.24.0` | text: one JSON object {analysis, plan, commands[{keystrokes, duration}], task_complete} | keystrokes to a persistent tmux pane | task_complete twice in a row | none | 0.5k |

\* rendered first prompt with the Qwen3.5 template, tokens. Real harnesses send much more (qwen-code ≈ 11k tokens,
kimi-cli ≈ 4k, OpenHands ≈ 5k): system prompts and long tool descriptions are concise paraphrases, never the long
upstream text; parameter schemas, tool names and order, and result / error strings are exact where recorded.

**Fidelity sources.** Recorded with `pgs harness` (fixtures, local only, in
`research/tiny-reliable-agent/harness/fixtures/`): opencode and mini-swe-agent through the Harbor 0.24 CLI on the box
(the real agent installed in the task container), qwen-code and kimi-cli run locally against `pgs harness serve`.
From source at the pinned revision: pi, OpenHands (schemas dumped from the installed SDK; file_editor strings checked
against it), SWE-agent, goose, hermes, terminus-2 (parser ported). Strings seen neither in a capture nor in the source
are marked `approx.` in `profiles_more.py`. Tests (`tests/test_harness_more.py`) replay each recorded tool call through
the profile and compare results (qwen-code, kimi-cli locally; opencode and mini-swe-agent in Docker on the same task).

**Per-harness notes.**
- *mini-swe-agent* drops a rejected turn from its history and answers it as a user message; here the turn stays (a
  format error is a user turn, the rollout is token-exact).
- *opencode*: Harbor's adapter passes the instruction quoted (`"<instruction>\n"`, recorded); tool names are lowercased
  before lookup (`Bash` runs `bash`); unknown tools go through its `invalid` tool. skill, task, todowrite and webfetch
  (sent by the real agent) are left out.
- *OpenHands*: legacy names are repaired as in `normalize_tool_call` (bash → terminal, str_replace_editor →
  file_editor, malformed names); undo is single level; the terminal cannot keep a process running between calls.
- *SWE-agent* expects exactly one call per turn and drops rejected turns; here every call runs. Its submit runs
  `git add -A && git diff --cached` (a non-repository gives an empty diff; upstream would fail).
- *qwen-code and destructive actions*: its prompt asks for confirmation before destructive or hard-to-reverse actions
  "when the current interaction mode can receive it; otherwise stop and report the blocker", and says authorisation
  "stands for the scope specified". Under Harbor the task instruction is the user's request, so a destructive action
  the task explicitly asks for (delete these files, kill that process) is in scope and is scored as usual (doing it:
  the verifier's reward; refusing: 0); destructive actions beyond the request are what the verifiers' collateral
  checks penalise. The paraphrase keeps both halves of the rule.
- *kimi-cli 1.52.0* is the final, archived release: its console entry point is deprecation-gated (every command
  prints a notice and exits), so Harbor 0.24's kimi-cli adapter no longer runs an agent. The profile emulates the
  1.52.0 agent (recorded via `python -m kimi_cli.cli`); its successor (Kimi Code CLI, TypeScript) is not emulated. Its
  StrReplaceFile replaces the first occurrence of a non-unique string (recorded), and Grep skips `.env*` files.
- *hermes* tracks main (Harbor installs main): pinned to e6899c8; a compact subset of the hermes-cli toolset (terminal,
  read_file, write_file, patch, search_files) with the advanced parameters left out; result JSON shapes follow its
  result classes.
- *terminus-2* needs tmux in the image: `research/tiny-reliable-agent/harness/Dockerfile.tmux` builds
  `harbor3-base:3` = `:1` + tmux (same verifier); pass `images={"harbor3-base:1": "harbor3-base:3"}`. Its parser
  (Harbor's, Apache-2.0) tolerates small-model JSON mistakes: extra text is a warning, a missing closing brace or an
  object inside prose is auto-corrected with a warning, anything else executes nothing and gets terminus's own
  feedback ("Previous response had parsing errors: ..."). Waits are deterministic: after the keystrokes the pane is
  polled until its shell has no child process and the screen is stable, at most `duration` (≤ 60) seconds.

## How it works

- `harness/ops.py`: the closed op set on the episode container (Exec stateless/persistent/`sh`, Read, View/list,
  Write, Replace(all), MultiReplace, Insert, UndoEdit (single level), Glob, Grep, End, Tmux/Keys; ApplyPatch is not
  implemented). Ops raise `OpError(kind, ...)`; each profile words the error itself.
- `harness/profile.py`: `ToolSpec(schema, bind)`, `HarnessProfile(name@version, system, user, tools, protocol, stop,
  answer, context, alias, unknown_tool, start, chat_template_kwargs)`, `Session` (one episode), `HarnessMix` (the
  draw).
- `harness/protocols.py`: `Native`, `FencedCommand` (mini text), the generic `XmlTool`, `ReAct`, `JsonBlock`, `Tags`;
  `harness/terminus.py`: `TerminusJSON`.
- `harness/rules.py`: stop rules `NoCall`, `Sentinel`, `Submit(confirm)`, `Finish(field)`, `Flag(confirm)`; answer
  rules `LastText`, `StopAnswer(fallback)`.
- `harness/context.py`: truncation with harness wording and the `Context` record (compaction is not emulated:
  rollouts are token-exact).
- `harness/profiles.py` (phase 1), `profiles_more.py` (phase 2), `terminus.py`; `schemas.py` holds recorded
  parameter schemas; `capture.py` is `pgs harness`.

Through the ordinary environment protocol: `reset()` returns the profile's conversation (system + rendered user turn),
`tool_schemas()` the native schemas (none for a text protocol), `text_actions()` the text protocol's actions or the
harness's format error (mini answers a turn without a call), `observation_text()` the next user turn, `done` the stop
rule, `unknown_tool()` the harness's answer to a tool it does not have (or the call its alias repair makes), and
`get_reward()` passes the answer rule's result to the verifier as `/logs/agent/final_message.md`.

**One profile per GRPO group.** The trainer sets `env.episode_group` (the group id) before `reset`; the profile is
drawn from sha256(seed, group, task_dir) by weight, so all samples of a group share it (advantages compare policies,
not harnesses) and groups vary. The drawn profile is in the rollout's `info["harness_profile"]` and in the rewards as
`env/metric/profile/<name>` (share, every trajectory) and `env/metric/profile/<name>/reward` (mean reward of that
profile's trajectories only).

**Thinking per profile.** `HarnessProfile.chat_template_kwargs` (None for the built-ins: the model's default, which is
what a harness that sends no `chat_template_kwargs` to vLLM gets) or `HarborEnv(profile_chat_kwargs={name: {...}})`
reach the episode through `episode_info["chat_template_kwargs"]`: model kwargs < profile < the row's own.

Set `env.max_tool_output_tokens` above the profiles' own limits so the harness wording is what the policy sees.

## Evaluation under a profile

`palingenesis.rl.episodes.run_episodes(engine, tok, rows, HarborEnv, env_args, k=...)` runs episodes outside training
through the pipeline's own rollout loop. `research/tiny-reliable-agent/scripts/eval/harbor_eval.py` uses it:

    python harbor_eval.py --profile opencode --k 4 small=Qwen/Qwen3.5-0.8B
    python harbor_eval.py --profiles 'opencode:1,pi:1,mini-swe-agent:1' --k 4 small=...
    python harbor_eval.py --profile terminus-2 --images harbor3-base:1=harbor3-base:3 small=...

Results are tagged by profile (per episode, and `by_profile` pass@1 in the summary).

## `pgs harness` (recording real harnesses)

    pgs harness serve   --script turns.json --out requests.jsonl [--port 8000]
    pgs harness capture <agent> --task <task_dir> --script turns.json --out <dir> [--host-ip IP]

`serve` is a stub OpenAI-compatible endpoint (`/v1/chat/completions`, streaming or not; `/v1/models`) that answers from
a script of assistant turns (tool calls or text; turns advance per conversation; side requests without tools, such as
a title, get a short text) and appends every request body, exactly as received, to a JSONL file. `capture` starts it
on the host and runs the real agent once with the Harbor CLI (`uv run --with harbor==0.24.0 harbor run -p <task> -a
<agent> ...`, the agent installed in the task container, which needs network for the install: use a copy of the task in
Harbor's `[[steps]]` layout with `allow_internet = true`). The recorded requests are the fixtures the conformance tests
diff a profile against; they stay local (some harness prompts are not ours to redistribute). Responses-API and
Anthropic-Messages requests are recorded and refused (codex, claude-code: not emulated).

## Known gaps

Not emulated: compaction (all but mini), opencode's fuzzy edit fallbacks, pi's whitespace-normalised matching, goose's
toolshim, hermes' fuzzy patch strategies (exact match here), multi-level undo (OpenHands, SWE-agent), long-running
terminal processes, one-call-per-turn enforcement (SWE-agent), apply_patch (codex / opencode with gpt-* models).
Answer families (du_answer, port_who, ...) cannot be solved under mini-swe-agent, SWE-agent or terminus-2 (no answer
channel): keep them out of those groups or accept a 0.
