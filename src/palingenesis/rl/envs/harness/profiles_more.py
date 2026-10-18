"""Phase 2 harness profiles (terminus-2 lives in terminus.py).

    swe-agent@main-3ea751c   bash (persistent), str_replace_editor, submit (twice: a review step with the diff first);
                             one call per turn; "OBSERVATION:" tool messages; no answer channel (last text)
    qwen-code@0.25.2         run_shell_command, read_file, write_file, edit, glob, grep_search (the coding subset of its
                             10 tools); stops at a turn without a tool call
    goose@1.55.0             the developer extension: shell, write, edit(path, before, after), tree
    hermes@main-e6899c8      a compact subset of hermes-cli: terminal (persistent), read_file, write_file, patch,
                             search_files; JSON tool results
    kimi-cli@1.52.0          Shell, ReadFile, Glob, Grep, WriteFile, StrReplaceFile; <system> status tags

Sources and fidelity: qwen-code and kimi-cli parameter schemas and most result strings were recorded from the real
CLIs against `pgs harness serve` (fixtures in research/tiny-reliable-agent/harness/fixtures/); swe-agent, goose and
hermes are written from their source at the pinned revision. Strings not observed in a capture or the source are
marked "approx." where they are produced. Prompts and long tool descriptions are concise paraphrases.

qwen-code and destructive actions: its prompt asks for confirmation before destructive or hard-to-reverse actions
"when the current interaction mode can receive it; otherwise stop and report the blocker", while authorisation
"stands for the scope specified". Under Harbor (non-interactive --prompt) the task instruction is the user's request,
so a destructive action the task explicitly asks for (delete these files, kill that process) is in scope and is
scored as usual (doing it: verifier reward; refusing: 0); destructive actions beyond the request are what the
verifiers' collateral checks penalise. The paraphrase keeps both halves of the rule.

kimi-cli: 1.52.0 is the final, archived release; its console entry point is deprecation-gated (every command prints a
notice and exits), so Harbor 0.24's kimi-cli adapter no longer runs an agent. The profile emulates the 1.52.0 agent
(captured through `python -m kimi_cli.cli`), the last kimi-cli people deployed; its successor (Kimi Code CLI) is not
emulated.
"""

from __future__ import annotations

import difflib
import json
import posixpath
import shlex

from palingenesis.rl.envs.harness import ops
from palingenesis.rl.envs.harness.context import Context
from palingenesis.rl.envs.harness.ops import OpError
from palingenesis.rl.envs.harness.profile import HarnessProfile, ToolSpec, fn_schema, register
from palingenesis.rl.envs.harness.rules import LastText, NoCall, StopRule
from palingenesis.rl.envs.harness.schemas import KIMI_PARAMETERS, QWEN_CODE_PARAMETERS


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ================================================================== SWE-agent (main @ 3ea751c, as Harbor clones it)

SWE_TIMEOUT = 30
SWE_SYSTEM = "You are a helpful assistant that can interact with a computer to solve tasks."
SWE_USER = (
    "<uploaded_files>\n{workdir}\n</uploaded_files>\n"
    "I've uploaded a python code repository in the directory {workdir}. Consider the following PR description:\n\n"
    "<pr_description>\n{instruction}\n</pr_description>\n\n"
    "Can you help me implement the necessary changes to the repository so that the requirements specified in the "
    "<pr_description> are met?\nI've already taken care of all changes to any of the test files described in the "
    "<pr_description>, so you don't have to modify the testing logic or any of the tests.\n"
    "Make the minimal changes to non-test files in the {workdir} directory so that the <pr_description> is satisfied.\n"
    "Follow these steps:\n1. Find and read the code relevant to the <pr_description>\n"
    "2. Create a script that reproduces the error and run it with `python <filename.py>` using the bash tool\n"
    "3. Edit the source code of the repo to resolve the issue\n4. Rerun your reproduce script and confirm the error is "
    "fixed\n5. Think about edge cases and make sure your fix handles them\nYour thinking should be thorough, so it's "
    "fine if it's very long."
)

SWE_TOOLS = {
    "bash": fn_schema(
        "bash",
        "runs the given command directly in bash",
        {"command": {"type": "string", "description": "The bash command to execute."}},
        ["command"],
    ),
    "str_replace_editor": fn_schema(
        "str_replace_editor",
        "Custom editing tool for viewing, creating and editing files * State is persistent across command calls and "
        "discussions with the user * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` "
        "is a directory, `view` lists non-hidden files and directories up to 2 levels deep * The `create` command "
        "cannot be used if the specified `path` already exists as a file * If a `command` generates a long output, it "
        "will be truncated and marked with `<response clipped>` * The `undo_edit` command will revert the last edit "
        "made to the file at `path`\nNotes for using the `str_replace` command: * The `old_str` parameter should match "
        "EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces! * If the `old_str` "
        "parameter is not unique in the file, the replacement will not be performed. Make sure to include enough "
        "context in `old_str` to make it unique * The `new_str` parameter should contain the edited lines that should "
        "replace the `old_str`\n",
        {
            "command": {
                "type": "string",
                "description": "The commands to run. Allowed options are: `view`, `create`, "
                "`str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
            },
            "path": {
                "type": "string",
                "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
            },
            "file_text": {
                "type": "string",
                "description": "Required parameter of `create` command, with the content of the file to be created.",
            },
            "old_str": {
                "type": "string",
                "description": "Required parameter of `str_replace` command containing the "
                "string in `path` to replace.",
            },
            "new_str": {
                "type": "string",
                "description": "Optional parameter of `str_replace` command containing the new "
                "string (if not given, no string will be added). Required parameter of `insert` command "
                "containing the string to insert.",
            },
            "insert_line": {
                "type": "integer",
                "description": "Required parameter of `insert` command. The `new_str` "
                "will be inserted AFTER the line `insert_line` of `path`.",
            },
            "view_range": {
                "type": "array",
                "description": "Optional parameter of `view` command when `path` points to a "
                "file. If none is given, the full file is shown. If provided, the file will be shown in the "
                "indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. "
                "Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                "items": {"type": "integer"},
            },
        },
        ["command", "path"],
    ),
    "submit": {
        "type": "function",
        "function": {"name": "submit", "description": "submits the current file", "parameters": {"type": "object"}},
    },
}

SWE_REVIEW = (
    "Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.\n\n"
    "1. If you made any changes to your code after running the reproduction script, please run the reproduction "
    "script again.\n  If the reproduction script is failing, please revisit your changes and make sure they are "
    "correct.\n  If you have already removed your reproduction script, please ignore this step.\n"
    "2. Remove your reproduction script (if you haven't done so already).\n"
    "3. If you have modified any TEST files, please revert them to the state they had before you started fixing the "
    "issue.\n  You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you "
    "need to revert.\n4. Run the submit command again to confirm.\n\nHere is a list of all of your changes:\n\n"
    "<diff>\n{diff}\n</diff>\n"
)
SWE_TRUNCATED = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should "
    "retry this tool after you have searched inside the file with `grep -n` in order to find the line "
    "numbers of what you are looking for.</NOTE>"
)


def swe_observation(output: str, limit: int = 100_000) -> str:
    """next_step_template / next_step_no_output_template / next_step_truncated_observation_template."""
    if output.strip() == "":
        return "Your command ran successfully and did not produce any output."
    if len(output) > limit:
        return (
            f"Observation: {output[:limit]}<response clipped><NOTE>Observations should not exceeded {limit} "
            f"characters. {len(output) - limit} characters were elided. Please try a different command that "
            "produces less output or use head/tail/grep/redirect the output to a file. Do not use interactive "
            "pagers.</NOTE>"
        )
    return f"OBSERVATION:\n{output}"


def swe_format_error(reason: str, n: int) -> str:
    if "No tool calls" in reason:
        return (
            "Your last output did not use any tool calls!\nPlease make sure your output includes exactly _ONE_ "
            "function call!\nYou must invoke the function directly using the function call format.\nYou cannot "
            "invoke commands with ```, you have to use the function call format.\nIf you think you have already "
            "resolved the issue, please submit your changes by running the `submit` command.\nIf you think you "
            "cannot solve the problem, please run `exit_forfeit` (if available) or `submit`.\nElse, please "
            "continue with a new tool call!"
        )
    return f"Your action could not be parsed properly: {reason}.\n"


def _swe_cat_n(text: str, desc: str, start: int = 1) -> str:
    """_make_output: maybe_truncate, expandtabs, every "\\n"-split line numbered (a trailing empty line too)."""
    if len(text) > 16000:
        text = text[:16000] + SWE_TRUNCATED
    text = text.expandtabs()
    body = "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(text.split("\n")))
    return f"Here's the result of running `cat -n` on {desc}:\n{body}\n"


def _swe_editor(s, a) -> str:
    command, path = a.get("command"), str(a.get("path", ""))
    out = _swe_editor_raw(s, command, path, a)
    return swe_observation(out.rstrip("\n"))


def _swe_editor_raw(s, command, path, a) -> str:
    if not path.startswith("/"):
        return (
            f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant "
            f"{posixpath.join(s.ws.workdir, path)}?"
        )
    k = s.ws.kind(path)
    if command not in ("view", "create", "str_replace", "insert", "undo_edit"):
        return (
            f'Unrecognized command {command}. The allowed commands for the str_replace_editor tool are: "view", '
            '"create", "str_replace", "insert", "undo_edit"'
        )
    if k is None and command != "create":
        return f"The path {path} does not exist. Please provide a valid path."
    if command == "create":
        if k == "file":
            return f"File already exists at: {path}. Cannot overwrite files using command `create`."
        if k == "dir":
            return f"The path {path} is a directory and only the `view` command can be used on directories"
        if a.get("file_text") is None:
            return "Parameter `file_text` is required for command: create"
        if s.ws.kind(posixpath.dirname(path)) != "dir":
            return f"The parent directory {posixpath.dirname(path)} does not exist. Please create it first."
        ops.write(s.ws, path, str(a["file_text"]))
        return f"File created successfully at: {path}"
    if k == "dir" and command != "view":
        return f"The path {path} is a directory and only the `view` command can be used on directories"
    if command == "view":
        if k == "dir":
            if a.get("view_range"):
                return "The `view_range` parameter is not allowed when `path` points to a directory."
            r = s.ws.sh(f"find {path} -maxdepth 2 -not -path '*/\\.*'")
            return (
                f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{r.stdout}\n"
            )
        text = ops.read(s.ws, path)
        vr = a.get("view_range")
        if not vr:
            return _swe_cat_n(text, path)
        if not (isinstance(vr, list) and len(vr) == 2 and all(isinstance(x, int) for x in vr)):
            return "Invalid `view_range`. It should be a list of two integers."
        lines, (first, last) = text.split("\n"), vr
        n = len(lines)
        if first < 1 or first > n:
            return (
                f"Invalid `view_range`: {vr}. Its first element `{first}` should be within the range of lines of "
                f"the file: {[1, n]}"
            )
        if last > n:
            return (
                f"Invalid `view_range`: {vr}. Its second element `{last}` should be smaller than the number of "
                f"lines in the file: `{n}`"
            )
        if last != -1 and last < first:
            return (
                f"Invalid `view_range`: {vr}. Its second element `{last}` should be larger or equal than its "
                f"first `{first}`"
            )
        body = lines[first - 1 :] if last == -1 else lines[first - 1 : last]
        return _swe_cat_n("\n".join(body), path, first)
    if command == "str_replace":
        if a.get("old_str") is None:
            return "Parameter `old_str` is required for command: str_replace"
        old, new = str(a["old_str"]).expandtabs(), str(a.get("new_str") or "").expandtabs()
        text = ops.read(s.ws, path).expandtabs()
        n = text.count(old)
        if n == 0:
            return f"No replacement was performed, old_str `{old}` did not appear verbatim in {path}."
        if n > 1:
            lines = [i + 1 for i, line in enumerate(text.split("\n")) if old in line]
            return (
                f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines {lines}. Please "
                "ensure it is unique"
            )
        if new == old:
            return f"No replacement was performed, old_str `{old}` is the same as new_str `{new}`."
        after = text.replace(old, new)
        ops.write(s.ws, path, after)
        line = text.split(old)[0].count("\n")
        start, end = max(0, line - 4), line + 4 + new.count("\n")
        snippet = "\n".join(after.split("\n")[start : end + 1])
        return (
            f"The file {path} has been edited. "
            + _swe_cat_n(snippet, f"a snippet of {path}", start + 1)
            + "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )
    if command == "insert":
        line, new = a.get("insert_line"), a.get("new_str")
        if line is None:
            return "Parameter `insert_line` is required for command: insert"
        if new is None:
            return "Parameter `new_str` is required for command: insert"
        text = ops.read(s.ws, path).expandtabs()
        lines, new_lines = text.split("\n"), str(new).expandtabs().split("\n")
        if not isinstance(line, int) or line < 0 or line > len(lines):
            return (
                f"Invalid `insert_line` parameter: {line}. It should be within the range of lines of the file: "
                f"{[0, len(lines)]}"
            )
        ops.write(s.ws, path, "\n".join(lines[:line] + new_lines + lines[line:]))
        snippet = lines[max(0, line - 4) : line] + new_lines + lines[line : line + 4]
        return (
            f"The file {path} has been edited. "
            + _swe_cat_n("\n".join(snippet), "a snippet of the edited file", max(1, line - 4 + 1))
            + "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, "
            "etc). Edit the file again if necessary."
        )
    try:  # undo_edit
        e = ops.undo_edit(s.ws, path)
    except OpError:
        return f"No edit history found for {path}."
    return f"Last edit to {path} undone successfully. {_swe_cat_n(e.after, path)}"


def _swe_bash(s, a) -> str:
    cmd = str(a.get("command", ""))
    r = ops.exec_(s.ws, cmd, SWE_TIMEOUT, persistent=True)
    s.last_exec = r
    if r.timed_out:
        return (
            f"The command '{cmd}' was cancelled because it took more than {SWE_TIMEOUT} seconds. Please try a "
            "different command that completes more quickly. Note: A common source of this error is if the command "
            "is interactive or requires user input (it is impossible to receive user input in the current "
            "environment, so the command will never complete)."
        )
    return swe_observation(r.output.rstrip("\n"))


def _swe_submit(s, a) -> str:
    q = s.ws.workdir
    r = s.ws.sh(
        f"cd {q} && git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git add -A >/dev/null 2>&1 && "
        "git diff --cached",
        timeout=60,
    )
    patch = r.stdout if r.exit_code == 0 else ""  # not a git repository: an empty diff (upstream would fail)
    if not s.flags.get("swe_reviewed"):
        s.flags["swe_reviewed"] = True
        return swe_observation(SWE_REVIEW.replace("{diff}", patch).rstrip("\n"))
    s.finish(None)
    return patch


class SweSubmit(StopRule):
    """submit ends the episode on its second call (review_on_submit_m); a turn without a call is a format error."""

    requires_action = True


def swe_agent() -> HarnessProfile:
    binds = {"bash": _swe_bash, "str_replace_editor": _swe_editor, "submit": _swe_submit}
    return HarnessProfile(
        name="swe-agent",
        version="main-3ea751c",
        system=SWE_SYSTEM,
        user=SWE_USER,
        tools=[ToolSpec(SWE_TOOLS[n], binds[n]) for n in ("bash", "str_replace_editor", "submit")],
        stop=SweSubmit(),
        answer=LastText(),
        format_error=swe_format_error,
        exec_timeout=SWE_TIMEOUT,
        unknown_tool=lambda name, names, args: (
            f"Your action could not be parsed properly: Command '{name}' not found in list of available commands..\n"
        ),
        context=Context("none", note="history processors: cache_control only"),
        emulates="SWE-agent main@3ea751c (2026-07-16; Harbor 0.24 clones main): config/default.yaml, "
        "tools/{registry,edit_anthropic,review_on_submit_m}, sweagent/tools/parsing.py FunctionCallingParser; "
        "one call per turn is not enforced here (all calls run)",
    )


# ================================================================== qwen-code 0.25.2

QWEN_SYSTEM = (
    "You are Qwen Code, a non-interactive CLI agent developed by Alibaba Group, specializing in software engineering "
    "tasks. Your primary goal is to help users safely and efficiently, following these instructions and using your "
    "tools.\n\n"
    "# Core Mandates\n"
    "- **Conventions:** never assume file contents; read the relevant code and configuration before changing it, and "
    "follow the project's style and structure.\n"
    "- **Comments:** default to none.\n"
    "- **Proactiveness:** do what was asked, including directly implied follow-up, and nothing beyond it.\n\n"
    "# Primary Workflows\n"
    "For engineering tasks: plan briefly when the work is complex, implement with the search and editing tools, adapt "
    "when you learn something new, and verify the change with the project's own checks before reporting it complete. "
    "Report outcomes faithfully: never claim a check passed when it did not.\n\n"
    "# Operational Guidelines\n"
    "Be concise and direct (CLI). Before your first tool call, briefly say what you are about to do. Use absolute "
    "paths with the file tools. Before running a shell command that modifies files or system state, explain its "
    "purpose and impact briefly. Prefer the dedicated tools to shell commands for reading, writing and searching "
    "files.\n\n"
    "# Executing actions with care\n"
    "Consider the reversibility and blast radius of actions. Local, reversible actions (editing files, running tests) "
    "are fine. For actions that are hard to reverse or destructive (deleting files, killing processes, rm -rf, "
    "git reset --hard, overwriting uncommitted changes), obtain confirmation when the interaction mode can receive it; "
    "otherwise stop and report the blocker. Authorization stands for the scope specified: an action the user "
    "explicitly requested is authorized for that scope, nothing beyond it. Match the scope of your actions to what "
    "was actually requested.\n\n"
    "# Environment\n"
    "Working directory (project root): {workdir}\nPlatform: linux\nToday's date: {date}"
)

QWEN_DESCRIPTIONS = {
    "run_shell_command": (
        "Executes a given shell command (as `bash -c <command>`) in a subprocess with optional timeout. This tool is "
        "for terminal operations like git, npm, docker: do not use it to read, write, edit or search files (use the "
        "dedicated tools). The optional timeout is in milliseconds (default 120000, max 600000); set is_background for "
        "long-running processes. The result reports Command, Directory, Output (stdout and stderr), Error, Exit Code, "
        "Signal and Process Group PGID."
    ),
    "read_file": (
        "Reads and returns the content of a specified file. The file_path argument MUST be an absolute path (project "
        "root + relative path). Large files are truncated, and the response says how to read more with the 'offset' "
        "(0-based) and 'limit' parameters."
    ),
    "write_file": (
        "Writes content to a specified file, creating it or overwriting it. For an existing file whose content has not "
        "been established in this session, read it with read_file first: blind overwrites are rejected. The "
        "file_path argument MUST be an absolute path."
    ),
    "edit": (
        "Replaces text within a file. By default replaces a single occurrence; set `replace_all` to change every "
        "instance of `old_string`. Read the file first with read_file. `file_path` MUST be absolute; `old_string` must "
        "be the exact literal text (with whitespace and indentation) and, for a single replacement, unique (include "
        "context); `new_string` is the exact replacement."
    ),
    "glob": (
        "Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like "
        '"**/*.js" or "src/**/*.ts"\n- Returns matching file paths sorted by modification time\n- Use this '
        "tool when you need to find files by name patterns"
    ),
    "grep_search": (
        'A powerful search tool built on ripgrep: full regex syntax (e.g. "log.*Error"), filter files '
        'with the glob parameter (e.g. "*.js"). Use it instead of grep or rg in the shell.'
    ),
}
QWEN_TIMEOUT_MS = 120_000


def _qwen_shell(s, a) -> str:
    cmd = str(a.get("command", ""))
    timeout = a.get("timeout")
    ms = int(timeout) if isinstance(timeout, int) and timeout > 0 else QWEN_TIMEOUT_MS
    directory = a.get("directory")
    wd = s.ws.abspath(directory) if directory else None
    r = ops.exec_(s.ws, cmd, min(ms, 600_000) / 1000, workdir=wd)
    s.last_exec = r
    out = r.output.rstrip("\n")
    err = f"Command timed out after {ms}ms" if r.timed_out else "(none)"  # approx. (not captured)
    return (
        f"Command: {cmd}\nDirectory: {directory or '(root)'}\nOutput: {out or '(empty)'}\nError: {err}\n"
        f"Exit Code: {'(none)' if r.timed_out else r.exit_code}\nSignal: {'SIGKILL' if r.timed_out else '(none)'}\n"
        "Process Group PGID: (none)"
    )


def _qwen_abs_error(path: str) -> str | None:
    if not path.startswith("/"):
        return f"File path must be absolute, but was relative: {path}. You must provide an absolute path."  # approx.
    return None


def _qwen_read(s, a) -> str:
    path = str(a.get("file_path", ""))
    if (e := _qwen_abs_error(path)) is not None:
        return e
    k = s.ws.kind(path)
    if k is None:
        return f"File not found: {path}"
    if k == "dir":
        return f"Path is a directory, not a file: {path}"  # approx.
    text = ops.read(s.ws, path)
    s.flags.setdefault("qwen_read", set()).add(path)
    offset, limit = a.get("offset"), a.get("limit")
    if offset is None and limit is None:
        return text
    lines = text.split("\n")
    start = max(0, int(offset or 0))
    end = min(len(lines), start + int(limit or 2000))
    body = "\n".join(lines[start:end])
    if start == 0 and end >= len(lines):
        return text
    return (
        "IMPORTANT: The file content has been truncated.\n"  # approx. (gemini-cli lineage wording)
        f"Status: Showing lines {start + 1}-{end} of {len(lines)} total lines.\n"
        "Action: To read more of the file, you can use the 'offset' and 'limit' parameters in a subsequent "
        f"'read_file' call. For example, to read the next section of the file, use offset: {end}.\n\n"
        f"--- FILE CONTENT (truncated) ---\n{body}"
    )


def _qwen_write(s, a) -> str:
    path = str(a.get("file_path", ""))
    if (e := _qwen_abs_error(path)) is not None:
        return e
    k = s.ws.kind(path)
    if k == "file" and path not in s.flags.get("qwen_read", set()):
        return (
            f"File {path} exists but has not been read in this session. Use read_file to read it before "
            "overwriting it."
        )  # approx. (prior-read enforcement)
    created = ops.write(s.ws, path, str(a.get("content", "")))
    s.flags.setdefault("qwen_read", set()).add(path)
    return (
        f"Successfully created and wrote to new file: {path}." if created else f"Successfully overwrote file: {path}."
    )  # (the second: approx.)


def _qwen_edit(s, a) -> str:
    path = str(a.get("file_path", ""))
    if (e := _qwen_abs_error(path)) is not None:
        return e
    old, new = str(a.get("old_string", "")), str(a.get("new_string", ""))
    if s.ws.kind(path) is None:
        if old == "":
            ops.write(s.ws, path, new)
            return f"Created new file: {path} with provided content."  # approx.
        return f"File not found: {path}"  # approx.
    try:
        e = ops.replace(s.ws, path, old, new, bool(a.get("replace_all")))
    except OpError as err:
        if err.kind == "multiple":
            return (
                f"Failed to edit. Found {err.data['count']} occurrences for old_string in {path} but replace_all "
                "was not enabled."
            )
        if err.kind == "identical":
            return f"No changes to apply. The old_string and new_string are identical in file: {path}"  # approx.
        return (
            f"Failed to edit, 0 occurrences found for old_string in {path}. No edits made. The exact text in "
            "old_string was not found. Ensure you're not escaping content incorrectly and check whitespace, "
            "indentation, and context. Use read_file tool to verify."
        )
    lines = e.after.split("\n")
    n = len(lines)
    if n <= 50:
        first, last = 1, n
    else:  # approx.: a window around the change
        first, last = max(1, e.line - 10), min(n, e.line + new.count("\n") + 10)
    s.flags.setdefault("qwen_read", set()).add(path)
    return (
        f"The file: {path} has been updated. Showing lines {first}-{last} of {n} from the edited file:\n\n---\n\n"
        + "\n".join(lines[first - 1 : last])
    )


def _qwen_glob(s, a) -> str:
    pattern = str(a.get("pattern", ""))
    try:
        files = ops.glob(s.ws, pattern, a.get("path") or None, limit=1000)
    except OpError:
        files = []
    if not files:
        return f'No files found matching pattern "{pattern}" in the workspace directory'  # approx.
    return (
        f'Found {len(files)} file(s) matching "{pattern}" in the workspace directory, sorted by modification time '
        "(newest first):\n---\n" + "\n".join(files)
    )


def _qwen_grep(s, a) -> str:
    pattern = str(a.get("pattern", ""))
    limit = a.get("limit") if isinstance(a.get("limit"), int) and a["limit"] > 0 else 1000
    try:
        rows = ops.grep(s.ws, pattern, a.get("path") or None, a.get("glob") or None, limit=limit)
    except OpError:
        rows = []
    if not rows:
        return f'No matches found for pattern "{pattern}" in the workspace directory.'  # approx.
    what = "match" if len(rows) == 1 else "matches"
    return f'Found {len(rows)} {what} for pattern "{pattern}" in the workspace directory:\n---\n' + "\n".join(
        f"{p}:{n}:{t}" for p, n, t in rows
    )


def qwen_unknown_tool(name: str, names: list[str], args: dict) -> str:
    near = sorted(names, key=lambda n: (levenshtein(name, n), n))[:3]
    return (
        f'Tool "{name}" not found in registry. Tools must use the exact names that are registered. Did you mean '
        f"one of: {', '.join(chr(34) + n + chr(34) for n in near)}?"
    )


def qwen_code() -> HarnessProfile:
    binds = {
        "run_shell_command": _qwen_shell,
        "read_file": _qwen_read,
        "write_file": _qwen_write,
        "edit": _qwen_edit,
        "glob": _qwen_glob,
        "grep_search": _qwen_grep,
    }
    order = ("edit", "glob", "grep_search", "read_file", "run_shell_command", "write_file")  # as the CLI lists them
    return HarnessProfile(
        name="qwen-code",
        version="0.25.2",
        system=QWEN_SYSTEM,
        tools=[ToolSpec(_tool(n, QWEN_DESCRIPTIONS[n], QWEN_CODE_PARAMETERS[n]), binds[n]) for n in order],
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=QWEN_TIMEOUT_MS / 1000,
        unknown_tool=qwen_unknown_tool,
        values={"date": "Friday, October 16, 2026"},
        context=Context("compaction", emulated=False, note="chat compression near the limit"),
        emulates="@qwen-code/qwen-code 0.25.2 recorded with pgs harness serve (--yolo --auth-type openai --prompt): "
        "6 of its 10 tools (notebook_edit, skill, tool_call, tool_search left out), the ~34 KB system prompt "
        "and the skills/memory system reminders summarised",
    )


# ================================================================== goose 1.55.0 (developer extension)

GOOSE_TIMEOUT = 300  # DEFAULT_EXTENSION_TIMEOUT
GOOSE_SYSTEM = (
    "You are a general-purpose AI agent called goose, created by AAIF (Agentic AI Foundation).\n"
    "goose is being developed as an open-source software project.\n\n"
    "# Extensions\n\nExtensions provide tools and context; the active ones are below and their tools are in your tool "
    "specification.\n\n## developer\n\n### Instructions\n"
    "Use the developer extension to build software and operate a terminal.\n\n"
    "Use the tools efficiently: read what you need in as few turns as possible, then make the edits or run the "
    "commands; you manage your context window.\n\n"
    "For editing software, use tree to understand the codebase structure and file sizes, rg to search, cat or sed "
    "to read (always read before editing), then write and edit to make changes. Test and verify as appropriate.\n\n"
    "When running Python scripts or commands, always use `python3` instead of `python`.\n\n"
    "# Response Guidelines\n\nUse Markdown formatting for all responses."
)

_SCHEMA_2020 = "https://json-schema.org/draft/2020-12/schema"
GOOSE_TOOLS = {
    "write": _tool(
        "write",
        "Create a new file or overwrite an existing file. Creates parent directories if needed.",
        {
            "$schema": _SCHEMA_2020,
            "title": "FileWriteParams",
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    "edit": _tool(
        "edit",
        "Edit a file by finding and replacing text. The before text must match exactly and uniquely. "
        "Use empty after text to delete.",
        {
            "$schema": _SCHEMA_2020,
            "title": "FileEditParams",
            "type": "object",
            "properties": {"path": {"type": "string"}, "before": {"type": "string"}, "after": {"type": "string"}},
            "required": ["path", "before", "after"],
        },
    ),
    "shell": _tool(
        "shell",
        "Execute a shell command in the current dir. Commands run under `bash` (set GOOSE_SHELL to "
        "override) - write command strings in that shell's syntax. Returns an object with stdout and stderr "
        "as separate fields. The output of each stream is limited to up to 2000 lines, and longer outputs "
        "will be saved to a temporary file.",
        {
            "$schema": _SCHEMA_2020,
            "title": "ShellParams",
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_secs": {
                    "description": "Maximum time in seconds to allow the command to "
                    "run before it is killed.\nIf omitted, defaults to "
                    "DEFAULT_EXTENSION_TIMEOUT.",
                    "type": ["integer", "null"],
                    "format": "uint64",
                    "minimum": 0,
                },
            },
            "required": ["command"],
        },
    ),
    "tree": _tool(
        "tree",
        "List a directory tree with line counts. Traversal respects .gitignore rules.",
        {
            "$schema": _SCHEMA_2020,
            "title": "TreeParams",
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "depth": {"type": "integer", "format": "uint32", "minimum": 0, "default": 2},
            },
            "required": ["path"],
        },
    ),
}


def _goose_shell(s, a) -> str:
    cmd = str(a.get("command", ""))
    if not cmd.strip():
        return "Command cannot be empty."
    t = a.get("timeout_secs")
    limit = int(t) if isinstance(t, int) and t > 0 else GOOSE_TIMEOUT
    r = ops.exec_(s.ws, cmd, limit)
    s.last_exec = r
    out = r.output.rstrip("\n") if r.output else ""
    text = out if out else "(no output)"
    lines = r.output.split("\n")
    notice = ""
    if len(lines) > 2000 or len(r.output) > 50_000:
        path = f"{s.ws.scratch}/goose-output-{len(lines)}.txt"
        try:
            s.ws.store(path, r.output)
        except OpError:
            pass
        text = "\n".join(lines[-50:])[-8000:]
        reason = (
            f"Output exceeded 2000 line limit ({len(lines)} lines total)."
            if len(lines) > 2000
            else f"Output exceeded 50000 byte limit ({len(r.output)} bytes total)."
        )
        notice = (
            f"[{reason} Full output saved to {path}. Read it with shell commands like `head`, `tail`, or "
            "`sed -n '100,200p'` up to 2000 lines at a time.]"
        )
    if r.timed_out:
        text += f"\n\nCommand timed out after {limit} seconds"
    elif r.exit_code != 0:
        text += f"\n\nCommand exited with code {r.exit_code}"
    return text + (f"\n{notice}" if notice else "")


def _goose_write(s, a) -> str:
    shown = str(a.get("path", ""))
    content = str(a.get("content", ""))
    try:
        created = ops.write(s.ws, s.ws.abspath(shown), content)
    except OpError as e:
        return f"Failed to write {shown}: {e.data.get('detail', 'Is a directory (os error 21)')}"
    return f"{'Created' if created else 'Wrote'} {shown} ({len(content.splitlines())} lines)"


def _goose_edit(s, a) -> str:
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown)
    before, after = str(a.get("before", "")), str(a.get("after", ""))
    if s.ws.kind(path) != "file":
        return f"Failed to read {shown}: No such file or directory (os error 2)"
    text = ops.read(s.ws, path)
    hits = text.count(before) if before else 0
    lines = text.splitlines()
    if hits == 0:
        msg = "No match found for the specified text."
        first = before.splitlines()[0].strip() if before.splitlines() else ""
        if first:
            for i, line in enumerate(lines):
                if first in line or line.strip() in first:
                    msg += "\n\nDid you mean:\n```\n" + "\n".join(lines[max(0, i - 2) : i + 3]) + "\n```"
                    break
        preview = (
            "(file is empty)" if not text else "\n".join(f"{i + 1:>4}: {line}" for i, line in enumerate(lines[:20]))
        )
        if len(lines) > 20:
            preview += f"\n... ({len(lines) - 20} more lines)"
        return msg + f"\n\nFile preview:\n```\n{preview}\n```"
    if hits > 1:
        msg = f"Found {hits} matches. Please provide more context to identify a unique match:\n"
        pos = -1
        for k in range(min(hits, 2)):
            pos = text.find(before, pos + 1)
            n = text.count("\n", 0, pos) + 1
            msg += f"\nMatch {k + 1} (line {n}):\n```\n" + "\n".join(lines[max(0, n - 2) : n + 1]) + "\n```"
        if hits > 2:
            msg += f"\n\n...and {hits - 2} more"
        return msg
    ops.write(s.ws, path, text.replace(before, after, 1))
    return f"Edited {shown} ({len(before.splitlines())} lines -> {len(after.splitlines())} lines)"


def _goose_tree(s, a) -> str:
    root = s.ws.abspath(str(a.get("path", "")))
    depth = a.get("depth", 2)
    k = s.ws.kind(root)
    if k is None:
        return f"Path does not exist: {root}"
    if k != "dir":
        return f"Path is not a directory: {root}"
    maxd = "" if depth == 0 else f"-maxdepth {int(depth) + 1}"
    q = shlex.quote(root)
    r = s.ws.sh(
        f"cd {q} && find . -mindepth 1 {maxd} -not -path '*/.*' -type d | sed 's|^\\./|d |'; "
        f"find . -mindepth 1 {maxd} -not -path '*/.*' -type f | sed 's|^\\./|f |'; "
        f"find . -mindepth 1 {maxd} -not -path '*/.*' -type f -exec wc -l {{}} + 2>/dev/null"
    )
    tree: dict = {}
    counts: dict[str, int] = {}
    kinds: list[tuple[str, str]] = []
    for line in r.stdout.splitlines():
        if line[:2] in ("d ", "f "):
            kinds.append((line[0], line[2:]))
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1] != "total":
                counts[parts[1][2:] if parts[1].startswith("./") else parts[1]] = int(parts[0])
    for kind, rel in kinds:
        node = tree
        comps = rel.split("/")
        for c in comps[:-1]:
            node = node.setdefault(c + "/", {})
        if kind == "d":
            node.setdefault(comps[-1] + "/", {})
        else:
            node[comps[-1]] = counts.get(rel, 0)

    def total(node) -> int:
        return sum(total(v) if isinstance(v, dict) else v for v in node.values())

    def fmt(n: int) -> str:
        return f"[{n // 1000}K]" if n >= 1000 else f"[{n}]"

    out: list[str] = []

    def render(node, d):
        for name in sorted(k for k in node if k.endswith("/")):
            out.append(f"{'  ' * d}{name}  {fmt(total(node[name]))}")
            render(node[name], d + 1)
        for name in sorted(k for k in node if not k.endswith("/")):
            out.append(f"{'  ' * d}{name}  {fmt(node[name])}")

    render(tree, 0)
    return "\n".join(out) + "\n" if out else "(empty directory)"


def goose() -> HarnessProfile:
    binds = {"write": _goose_write, "edit": _goose_edit, "shell": _goose_shell, "tree": _goose_tree}
    return HarnessProfile(
        name="goose",
        version="1.55.0",
        system=GOOSE_SYSTEM,
        tools=[ToolSpec(GOOSE_TOOLS[n], binds[n]) for n in ("write", "edit", "shell", "tree")],
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=GOOSE_TIMEOUT,
        unknown_tool=lambda name, names, args: f"Tool not found: {name}",  # approx.
        context=Context("compaction", emulated=False, note="auto-compaction near the limit"),
        emulates="goose v1.55.0 (Harbor installs the stable release with the builtin developer extension): "
        "crates/goose/src/prompts/system.md, agents/platform_extensions/developer/{mod,edit,shell,tree}.rs "
        "(read_image left out); schemas as schemars 1.0 renders them (unconfirmed by a capture)",
    )


# ================================================================== Hermes Agent (main @ e6899c8, compact subset)

HERMES_SYSTEM = (
    "You are Hermes Agent, built by Nous Research. Be direct: match the length of your reply to the weight of the ask; "
    "finished work gets a short report of what changed, what is verified and what is left. No filler, no restating "
    "the request, no narrating tool calls. When unsure, say so plainly.\n\n"
    "# Environment\nShell: bash on linux (local backend). Working directory: {workdir}. The working directory, files "
    "and exported variables persist between terminal calls.\n\n"
    "Use read_file, search_files, patch and write_file instead of cat/head/tail, grep/find/ls, sed/awk and "
    "echo/heredoc in the terminal."
)

HERMES_TOOLS = {
    "terminal": fn_schema(
        "terminal",
        "Execute shell commands on the OS, shell and backend named in your environment section. The working directory, "
        "files and exported variables persist between calls: activate a virtualenv or export once, not before every "
        "command.\n\nUse read_file, search_files, patch and write_file instead of cat/head/tail, grep/find/ls, sed/awk "
        "and echo/heredoc. Output is auto-truncated with the full text saved to a file; never pipe through head/tail "
        "to shorten it.\n\nForeground (default) returns as soon as the command finishes. When a command changes "
        'directory, trust the result\'s "cwd" instead of prefixing later commands with cd.\n',
        {
            "command": {"type": "string", "description": "The shell command to execute"},
            "timeout": {
                "type": "integer",
                "description": "Max seconds (default 180, foreground max 600). Finishing early "
                "returns at once, so set it generously.",
                "minimum": 1,
            },
            "workdir": {"type": "string", "description": "Absolute working directory for this command only."},
        },
        ["command"],
    ),
    "read_file": fn_schema(
        "read_file",
        "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output "
        "format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files.",
        {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default: 1)",
                "default": 1,
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 2000, max: 2000). Reads "
                "are additionally capped at a ~100K-character budget with a next_offset continuation.",
                "default": 2000,
                "maximum": 2000,
            },
        },
        ["path"],
    ),
    "write_file": fn_schema(
        "write_file",
        "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in "
        "terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted "
        "edits. For an EXISTING file, call read_file first: write_file refuses when this task has no current full read "
        "of the file.",
        {
            "path": {
                "type": "string",
                "description": "Path to the file to write (will be created if it doesn't exist, "
                "overwritten if it does)",
            },
            "content": {"type": "string", "description": "Complete content to write to the file"},
        },
        ["path", "content"],
    ),
    "patch": fn_schema(
        "patch",
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. Uses fuzzy matching (9 "
        "strategies) so minor whitespace/indentation differences won't break it. Returns a unified diff. Auto-runs "
        "syntax checks after editing. Finds a unique string and replaces it.",
        {
            "path": {"type": "string", "description": "File path to edit."},
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace. Must be unique in the file "
                "unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "Changed replacement text; it must differ from old_string. "
                "Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
        },
        ["path", "old_string", "new_string"],
    ),
    "search_files": fn_schema(
        "search_files",
        "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Content search "
        "(target='content'): regex search inside files. File search (target='files'): find files by glob pattern "
        "(e.g., '*.py', '*config*').",
        {
            "pattern": {
                "type": "string",
                "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search",
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
                "description": "'content' searches inside file contents, 'files' searches for files by name",
                "default": "content",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current working directory)",
                "default": ".",
            },
            "file_glob": {
                "type": "string",
                "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 50)",
                "default": 50,
            },
        },
        ["pattern"],
    ),
}


def _hj(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def _hermes_terminal(s, a) -> str:
    cmd = str(a.get("command", ""))
    t = a.get("timeout")
    limit = min(int(t), 600) if isinstance(t, int) and t > 0 else 180
    wd = a.get("workdir")
    before = s.flags.get("hermes_cwd", s.ws.workdir)
    if wd:
        r = ops.exec_(s.ws, cmd, limit, workdir=s.ws.abspath(wd))
    else:
        r = ops.exec_(s.ws, cmd, limit, persistent=True)
    s.last_exec = r
    out = r.output.strip()
    if r.timed_out:
        out = (out + "\n" if out else "") + f"[Command timed out after {limit}s]"  # approx.
    d = {"output": out, "exit_code": 124 if r.timed_out else r.exit_code, "error": None}
    if not wd and r.cwd and r.cwd != before:
        d["cwd"] = r.cwd
        s.flags["hermes_cwd"] = r.cwd
    return _hj(d)


def _hermes_read(s, a) -> str:
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown, s.flags.get("hermes_cwd"))
    k = s.ws.kind(path)
    if k != "file":
        return _hj({"error": f"File not found: {shown}"})  # approx.
    text = ops.read(s.ws, path)
    s.flags.setdefault("hermes_read", set()).add(path)
    lines = ops.lines_of(text)
    offset = max(1, int(a.get("offset") or 1))
    limit = min(2000, int(a.get("limit") or 2000))
    body = lines[offset - 1 : offset - 1 + limit]
    d = {
        "content": "\n".join(f"{i + offset}|{line}" for i, line in enumerate(body)),
        "total_lines": len(lines),
        "file_size": len(text.encode()),
        "truncated": offset - 1 + limit < len(lines),
    }
    if d["truncated"]:
        d["hint"] = (
            f"Use offset={offset + limit} to continue reading (showing {offset}-{offset + len(body) - 1} of "
            f"{len(lines)} lines)"
        )
    d.update({"is_binary": False, "is_image": False})
    return _hj(d)


def _hermes_write(s, a) -> str:
    path = s.ws.abspath(str(a.get("path", "")), s.flags.get("hermes_cwd"))
    if s.ws.kind(path) == "file" and path not in s.flags.get("hermes_read", set()):
        return _hj(
            {
                "error": "write_file refused: this task has no current full read of the file. Call read_file, "
                "merge your changes, then retry."
            }
        )  # approx.
    content = str(a.get("content", ""))
    parent_missing = s.ws.kind(posixpath.dirname(path)) is None
    try:
        ops.write(s.ws, path, content)
    except OpError as e:
        return _hj({"bytes_written": 0, "dirs_created": False, "error": str(e.data.get("detail", e.kind))})
    s.flags.setdefault("hermes_read", set()).add(path)
    return _hj({"bytes_written": len(content.encode()), "dirs_created": parent_missing, "verified": True})


def _hermes_patch(s, a) -> str:
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown, s.flags.get("hermes_cwd"))
    old, new = str(a.get("old_string", "")), str(a.get("new_string", ""))
    if s.ws.kind(path) != "file":
        return _hj({"success": False, "error": f"File not found: {shown}"})  # approx.
    try:
        e = ops.replace(s.ws, path, old, new, bool(a.get("replace_all")))
    except OpError as err:
        msg = {
            "multiple": f"Found {err.data.get('count')} matches for old_string. Include more surrounding context to "
            "make it unique, or set replace_all=true.",
            "identical": "new_string must differ from old_string.",
            "empty_old": "old_string must not be empty.",
        }.get(err.kind, "Could not find a match for old_string in the file.")  # approx.
        return _hj({"success": False, "error": msg})
    diff = "".join(
        difflib.unified_diff(e.before.splitlines(True), e.after.splitlines(True), f"a/{shown}", f"b/{shown}")
    )
    return _hj({"success": True, "diff": diff, "files_modified": [shown]})


def _hermes_search(s, a) -> str:
    pattern, target = str(a.get("pattern", "")), a.get("target") or "content"
    root = s.ws.abspath(str(a.get("path") or "."), s.flags.get("hermes_cwd"))
    limit = int(a.get("limit") or 50)
    try:
        if target == "files":
            files = ops.glob(s.ws, pattern, root, limit=limit)
            return _hj({"total_count": len(files), "files": files} if files else {"total_count": 0})
        rows = ops.grep(s.ws, pattern, root, a.get("file_glob") or None, limit=limit)
    except OpError as e:
        return _hj({"total_count": 0, "error": f"Path not found: {e.data.get('path')}"})  # approx.
    d: dict = {"total_count": len(rows)}
    if rows:
        d["matches"] = [{"path": p, "line": n, "content": t} for p, n, t in rows]
    return _hj(d)


def hermes() -> HarnessProfile:
    binds = {
        "terminal": _hermes_terminal,
        "read_file": _hermes_read,
        "write_file": _hermes_write,
        "patch": _hermes_patch,
        "search_files": _hermes_search,
    }
    return HarnessProfile(
        name="hermes",
        version="main-e6899c8",
        system=HERMES_SYSTEM,
        tools=[
            ToolSpec(HERMES_TOOLS[n], binds[n])
            for n in ("terminal", "read_file", "write_file", "patch", "search_files")
        ],
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=180,
        unknown_tool=lambda name, names, args: _hj({"error": f"Unknown tool: {name}"}),  # approx.
        context=Context("compaction", emulated=False, note="context compression"),
        emulates="hermes-agent main@e6899c8 (2026-10-16; Harbor 0.24 installs main): tools/terminal_tool.py "
        "TERMINAL_SCHEMA, tools/file_tools.py READ_FILE/WRITE_FILE/PATCH/SEARCH_FILES schemas (terminal: "
        "background/pty/notify/heartbeat/persist_on_release left out; search_files: offset/order/output_mode/"
        "context left out), result JSON shapes from terminal_tool_result.py and file_operations_common.py; "
        "the rest of the hermes-cli toolset (browser, memory, delegation, cron, ...) left out",
    )


# ================================================================== kimi-cli 1.52.0

KIMI_SYSTEM = (
    "You are Kimi Code CLI, an interactive general AI agent running on a user's computer.\n\n"
    "Your primary goal is to help users with software engineering tasks by taking action: use the tools available to "
    "make real changes on the user's system. You should also answer questions when asked.\n\n"
    "# Prompt and Tool Use\n\n"
    "Read the user's request and do what was asked. For simple questions you may reply directly; otherwise default "
    "to taking action with tools (when a request could be a question or a task, treat it as a task). When creating, "
    "modifying or running code or files, use the tools (WriteFile, Shell, ...) to make the actual changes; do not "
    "just describe them. Tool calls should be self-explanatory; follow each tool's description. Make independent "
    "tool calls in parallel. The system may add information in `<system>` tags within user or tool messages.\n\n"
    "# Working Environment\n\nOperating system: Linux. Working directory: {workdir}.\n\n"
    "# Ultimate Reminders\n\nBe HELPFUL, CONCISE and ACCURATE; test what you build and verify what you change. Never "
    "diverge from the task, never give more than asked, keep it simple, and do not give up too early. When the task "
    "requires creating or modifying files, always use tools to do so."
)

KIMI_DESCRIPTIONS = {
    "Shell": (
        "Execute a bash (`/bin/bash`) command. Use this tool to explore the filesystem, edit files, run scripts, "
        "get system information, etc.\n\n**Output:** stdout and stderr combined, possibly truncated; if the "
        "command failed, the exit code is given in a system tag. Set a timeout for long commands; "
        "run_in_background starts a background task (with a short description)."
    ),
    "ReadFile": (
        "Read text content from a file. A `<system>` tag precedes the content; lines are numbered like "
        "`cat -n`. Use `line_offset` and `n_lines` for large files (max 1000 lines per call). Prefer Grep to "
        "search; use Glob or `ls` to list directories."
    ),
    "Glob": (
        "Find files and directories using glob patterns (`*`, `?`, `**`), e.g. `*.py`, `src/**/*.js`, "
        "`*.config.{js,ts}`. Patterns starting with `**` are rejected (they would search everything)."
    ),
    "Grep": (
        "A powerful search tool based-on ripgrep. ALWAYS use it instead of running `grep` or `rg` with Shell. Use "
        "ripgrep syntax (escape braces). Hidden files are searched; sensitive files (such as `.env`) are always "
        "skipped."
    ),
    "WriteFile": (
        "Write content to a file. `mode` defaults to `overwrite`; for long content (> 100 lines) write in "
        "several calls, `overwrite` first and then `append`."
    ),
    "StrReplaceFile": (
        "Replace specific strings within a specified file. Only for text files; multi-line strings are "
        "supported; one edit or a list of edits per call. Prefer it to WriteFile and Shell `sed`."
    ),
}


def _sys(text: str) -> str:
    return f"<system>{text}</system>"


def _kimi_path(s, path: str) -> str:
    return s.ws.abspath(path)


def _kimi_shell(s, a) -> str:
    cmd = str(a.get("command", ""))
    t = a.get("timeout", 60)
    limit = int(t) if isinstance(t, int) and t >= 1 else 60
    r = ops.exec_(s.ws, cmd, limit)
    s.last_exec = r
    if r.timed_out:
        return _sys(f"ERROR: Command killed by timeout ({limit}s)") + r.output
    if r.exit_code != 0:
        return _sys(f"ERROR: Command failed with exit code: {r.exit_code}.") + r.output
    return _sys("Command executed successfully.") + r.output


def _kimi_read(s, a) -> str:
    shown = str(a.get("path", ""))
    path = _kimi_path(s, shown)
    k = s.ws.kind(path)
    if k is None:
        return _sys(f"ERROR: `{shown}` does not exist.")
    if k == "dir":
        return _sys(f"ERROR: `{shown}` is not a file.")  # approx.
    lines = ops.lines_of(ops.read(s.ws, path))
    offset = int(a.get("line_offset") or 1)
    n = min(int(a.get("n_lines") or 1000), 1000)
    start = max(0, len(lines) + offset) if offset < 0 else offset - 1
    chunk = lines[start : start + n]
    eof = start + len(chunk) >= len(lines)
    head = f"{len(chunk)} lines read from file starting from line {start + 1}. Total lines in file: {len(lines)}." + (
        " End of file reached." if eof else ""
    )
    return _sys(head) + "".join(f"{i + start + 1:6}\t{line}\n" for i, line in enumerate(chunk))


def _kimi_write(s, a) -> str:
    path = _kimi_path(s, str(a.get("path", "")))
    content = str(a.get("content", ""))
    mode = a.get("mode") or "overwrite"
    try:
        if mode == "append" and s.ws.kind(path) == "file":
            content = ops.read(s.ws, path) + content
        ops.write(s.ws, path, content)
    except OpError as e:
        return _sys(f"ERROR: Failed to write to {path}: {e.data.get('detail', e.kind)}")  # approx.
    verb = "appended to" if mode == "append" else "overwritten"
    return _sys(f"File successfully {verb}. Current size: {len(content.encode())} bytes.")


def _kimi_replace(s, a) -> str:
    shown = str(a.get("path", ""))
    path = _kimi_path(s, shown)
    if s.ws.kind(path) != "file":
        return _sys(f"ERROR: `{shown}` does not exist.")
    edits = a.get("edit")
    edits = [edits] if isinstance(edits, dict) else (edits or [])
    text = ops.read(s.ws, path)
    total = 0
    for e in edits:
        old, new = str(e.get("old", "")), str(e.get("new", ""))
        n = text.count(old) if old else 0
        if n == 0:
            continue
        if e.get("replace_all"):
            text, total = text.replace(old, new), total + n
        else:  # the first occurrence (str.replace(old, new, 1)), even when it is not unique
            text, total = text.replace(old, new, 1), total + 1
    if total == 0:
        return _sys("ERROR: No replacements were made. The old string was not found in the file.")
    ops.write(s.ws, path, text)
    return _sys(f"File successfully edited. Applied {len(edits)} edit(s) with {total} total replacement(s).")


def _kimi_glob(s, a) -> str:
    pattern = str(a.get("pattern", ""))
    if pattern.startswith("**"):
        return _sys(
            f"ERROR: Pattern `{pattern}` starts with '**', which is not allowed because it would recursively "
            "search all directories. Use a more specific pattern."
        )  # approx.
    root = s.ws.abspath(a.get("directory") or s.ws.workdir)
    try:
        files = ops.glob(s.ws, pattern, root, limit=1000)
    except OpError:
        files = []
    if a.get("include_dirs", True):
        entries, _ = ops.list_dir(s.ws, root, 6, hidden=True) if s.ws.kind(root) == "dir" else ([], 0)
        files += [
            posixpath.join(root, d.rstrip("/"))
            for d in entries
            if d.endswith("/") and ops.glob_match(pattern, d.rstrip("/"))
        ]
    rel = sorted(posixpath.relpath(f, root) for f in files)
    if not rel:
        return _sys(f"No matches found for pattern `{pattern}`.")  # approx.
    return _sys(f"Found {len(rel)} matches for pattern `{pattern}`.") + "\n".join(rel)


def _kimi_grep(s, a) -> str:
    pattern = str(a.get("pattern", ""))
    root = s.ws.abspath(str(a.get("path") or "."))
    try:
        rows = ops.grep(s.ws, pattern, root, a.get("glob") or None, limit=10_000)
    except OpError:
        rows = []
    rows = [r for r in rows if not posixpath.basename(r[0]).startswith(".env")]  # sensitive files are skipped

    def rel(p):
        return "./" + posixpath.relpath(p, root)

    mode = a.get("output_mode") or "files_with_matches"
    if mode == "content":
        lines = [f"{rel(p)}:{n}:{t}" if a.get("-n", True) else f"{rel(p)}:{t}" for p, n, t in rows]
    elif mode == "count_matches":
        counts: dict[str, int] = {}
        for p, _, _ in rows:
            counts[rel(p)] = counts.get(rel(p), 0) + 1
        lines = [f"{p}:{c}" for p, c in counts.items()]  # approx.
    else:
        lines = list(dict.fromkeys(rel(p) for p, _, _ in rows))
    head = a.get("head_limit", 250)
    if isinstance(head, int) and head > 0:
        lines = lines[:head]
    return "\n".join(lines) if lines else _sys("No matches found.")  # (the second: approx.)


def kimi_cli() -> HarnessProfile:
    binds = {
        "Shell": _kimi_shell,
        "ReadFile": _kimi_read,
        "Glob": _kimi_glob,
        "Grep": _kimi_grep,
        "WriteFile": _kimi_write,
        "StrReplaceFile": _kimi_replace,
    }
    order = ("Shell", "ReadFile", "Glob", "Grep", "WriteFile", "StrReplaceFile")
    return HarnessProfile(
        name="kimi-cli",
        version="1.52.0",
        system=KIMI_SYSTEM,
        tools=[ToolSpec(_tool(n, KIMI_DESCRIPTIONS[n], KIMI_PARAMETERS[n]), binds[n]) for n in order],
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=60,
        unknown_tool=lambda name, names, args: _sys(f"ERROR: Tool `{name}` not found"),
        context=Context("compaction", emulated=False, note="context compaction"),
        emulates="kimi-cli 1.52.0 (final release) recorded with pgs harness serve through python -m kimi_cli.cli "
        "--print --yolo: 6 of its 16 tools (Agent, AskUserQuestion, SetTodoList, Task*, ReadMediaFile, "
        "FetchURL, plan mode left out), ~15 KB system prompt summarised",
    )


for _f, _n, _v in (
    (swe_agent, "swe-agent", "main-3ea751c"),
    (qwen_code, "qwen-code", "0.25.2"),
    (goose, "goose", "1.55.0"),
    (hermes, "hermes", "main-e6899c8"),
    (kimi_cli, "kimi-cli", "1.52.0"),
):
    register(_n, _v, _f)
