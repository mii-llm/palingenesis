# Harness profiles (HarborEnv)

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
    profile_seed: 0           # optional
```

`profile: opencode` runs a single profile. Without `profile`/`profiles` HarborEnv behaves exactly as before
(the `palingenesis-default` profile, tested byte-identical). A row's `harness_profile` field forces a profile.

## Profiles (phase 1)

| profile | protocol | tools | stop | answer |
|---|---|---|---|---|
| `palingenesis-default@1` | native | bash, read_file, write_file, str_replace, submit | submit / no call | last text |
| `mini-swe-agent@2.4.6` | native | bash(command), `/bin/sh` per call, 30 s | `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` | output after the sentinel (no answer channel) |
| `mini-swe-agent-text@2.4.6` | ```` ```mswea_bash_command ```` block, exactly one | same | same | same |
| `opencode@1.19.2` | native | bash(command, timeout ms, workdir), read, write, edit(filePath, oldString, newString, replaceAll), glob, grep | turn without a tool call | last text |
| `pi@1.3.0` | native | read, bash(command, timeout s), edit(path, edits[{oldText, newText}]), write | turn without a tool call | last text |
| `openhands-sdk@1.55.1` | native | terminal (persistent), file_editor, finish(message), think | finish, or a turn without a call | finish message, else last text |

Tool parameter schemas are copied exactly from the harness source at that version (they are interface contracts);
observation and error strings follow the source (mini's observation templates and OpenHands' file_editor strings are
checked against the upstream implementations in the tests). System prompts and long tool descriptions are concise
paraphrases, never the long upstream text. Rendered prompt length with the Qwen3.5 template: mini ≈ 0.8k tokens,
mini-text 0.6k, pi 1.3k, opencode 2.0k, OpenHands 2.6k.

## How it works

- `harness/ops.py`: the closed op set on the episode container (Exec stateless/persistent, Read, View/list, Write,
  Replace(all), MultiReplace, Insert, UndoEdit (single level), Glob, Grep, End; Keys/tmux and ApplyPatch are phase 2
  stubs). Ops raise `OpError(kind, ...)`; each profile words the error itself.
- `harness/profile.py`: `ToolSpec(schema, bind)`, `HarnessProfile(name@version, system, user, tools, protocol, stop,
  answer, context)`, `Session` (one episode), `HarnessMix` (the draw).
- `harness/protocols.py`: `Native`, `FencedCommand` (mini text), and the generic `XmlTool`, `ReAct`, `JsonBlock`,
  `Tags` text protocols (parse / render / observation).
- `harness/rules.py`: stop rules `NoCall`, `Sentinel`, `Submit(confirm)`, `Finish(field)`, `Flag(confirm)`; answer
  rules `LastText`, `StopAnswer(fallback)`.
- `harness/context.py`: truncation with harness wording (head/tail characters, head/tail lines within a byte budget,
  middle clip) and the `Context` record (compaction is not emulated: rollouts are token-exact).

Through the ordinary environment protocol: `reset()` returns the profile's conversation (system + rendered user turn),
`tool_schemas()` the native schemas (none for a text protocol), `text_actions()` the text protocol's actions or the
harness's format error (mini answers a turn without a call), `observation_text()` the next user turn, `done` the stop
rule, and `get_reward()` passes the answer rule's result to the verifier as `/logs/agent/final_message.md`.

**One profile per GRPO group.** The trainer sets `env.episode_group` (the group id) before `reset`; the profile is
drawn from sha256(seed, group, task_dir) by weight, so all samples of a group share it (advantages compare policies,
not harnesses) and groups vary. The drawn profile is in the rollout's `info["harness_profile"]` and in the rewards as
`env/metric/profile/<name>` (share, every trajectory) and `env/metric/profile/<name>/reward` (mean reward of that
profile's trajectories only).

Set `env.max_tool_output_tokens` above the profiles' own limits so the harness wording is what the policy sees.

## Known gaps

Not emulated: compaction (opencode, pi, OpenHands condenser), opencode's fuzzy edit fallbacks and pi's
whitespace-normalised matching (exact match here), OpenHands' multi-level undo and long-running terminal processes
(`is_input` answers that nothing runs), OpenHands' task_tracker. opencode's Effect-Schema JSON schemas are
reconstructed without `$schema`. `chat_template_kwargs` (e.g. `enable_thinking`) is recorded per profile but not yet
applied. Answer families (du_answer, port_who, ...) cannot be solved under mini-swe-agent (no answer channel): keep
them out of mini groups or accept a 0.

## `pgs harness capture` (design, not implemented)

Goal: measure each profile's fidelity against the real harness instead of trusting the paraphrase.

1. `pgs harness capture <agent> --version X --task <task_dir>` starts a recording stub OpenAI-compatible endpoint
   (chat completions; Responses and Anthropic Messages later) that logs each request body and answers from a script
   (a tool call per turn, then a final text), installs the pinned harness in the task image (or a pre-baked
   `harbor3-agents` image) and runs it once through the Harbor CLI adapter against the stub.
2. The captured requests (system prompt, tools, every observation and error string the harness sent back) are saved
   as local fixtures under `~/.cache/palingenesis/harness_captures/<agent>@<version>/`, never committed or
   redistributed (licences: some prompts are proprietary).
3. A conformance test per profile replays the same scripted calls through `Session` and diffs: tool schemas (must be
   equal), observation and error strings (must be equal up to paths and timestamps), and the system prompt (reported
   as a similarity score: profiles paraphrase it on purpose).
4. The same capture run gives the emulation gap check: real-agent pass@1 via the Harbor CLI vs emulated pass@1 on
   held-out tasks.
