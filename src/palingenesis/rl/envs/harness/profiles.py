"""Built-in harness profiles (phase 1). Each emulates one pinned harness version: its tool contracts (parameter
schemas copied exactly: they are interface contracts), its observation and error wording (from the source at that
version), its stop and answer rules; system prompts and long tool descriptions are concise paraphrases of the
originals (summarised in research/tiny-reliable-agent/harness/profiles/), never the long upstream text.

    palingenesis-default@1   HarborEnv's own prompt and tools (bash, read_file, write_file, str_replace, submit);
                             rendered byte-identically to HarborEnv without profiles
    palingenesis-default@2   nanocode's dialect (what "palingenesis-default" names): read(path, offset, limit) with
                             "   1| " line numbers, write, edit(path, old, new, all), glob(pat, path), grep(pat, path),
                             bash(cmd); "ok" / "error: ..." / "none" / "(empty)"; stops at a turn without a tool call
    palingenesis-default@3   @2 except that an edit or write leaving the file unchanged is an error, not "ok" (ours,
                             not nanocode's); opt-in by id: the bare name stays @2, so results stay comparable
    mini-swe-agent@2.4.6     native bash(command) only, JSON {returncode, output} observations (head/tail over 10k
                             characters), /bin/sh per command, sentinel submit, no answer channel (the submission is
                             the output after the sentinel line)
    mini-swe-agent-text@2.4.6   the same with ```mswea_bash_command fenced blocks, exactly one per turn
    opencode@1.19.2          bash(command, timeout ms, workdir), read, write, edit(filePath, oldString, newString,
                             replaceAll), glob, grep; stops at a turn without a tool call; answer = last text
    pi@1.3.0                 read, bash(command, timeout s), edit(path, edits[{oldText, newText}]), write; stops at
                             a turn without a tool call; answer = last text
    openhands-sdk@1.55.1     terminal (persistent), file_editor (view/create/str_replace/insert/undo_edit),
                             finish(message), think; stops at finish (or a turn without a call); answer = the finish
                             message, else the last text

Known gaps (phase 1): opencode/pi compaction and OpenHands' condenser are not emulated (rollouts are token-exact);
opencode's fuzzy edit fallbacks and pi's whitespace-normalised matching are exact-match here; pi's schema validation
(typebox, "Validation failed for tool ...") is not emulated (edit arguments it rejects get edit.js's own message);
OpenHands' undo is single level and its terminal cannot keep a process running between calls (is_input answers that
nothing is running); OpenHands' task_tracker is left out; opencode's skill, task, todowrite and webfetch tools are left out (Harbor does not
disable them). opencode's parameter schemas are the ones recorded with `pgs harness capture opencode` (Harbor 0.24 CLI);
tests/test_harness_more.py diffs the profiles against the recorded requests.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
import uuid

from palingenesis.rl.envs.harness import context as ctx
from palingenesis.rl.envs.harness import ops
from palingenesis.rl.envs.harness.context import Context
from palingenesis.rl.envs.harness.ops import OpError
from palingenesis.rl.envs.harness.profile import HarnessProfile, ToolSpec, fn_schema, register
from palingenesis.rl.envs.harness.protocols import FencedCommand, Native
from palingenesis.rl.envs.harness.rules import Finish, LastText, NoCall, Sentinel, StopAnswer, Submit
from palingenesis.rl.envs.harness.schemas import OPENCODE_PARAMETERS

MODEL = "qwen3.5-0.8b"  # the served model name a harness would report (opencode's environment block)
DATE = "Fri Oct 16 2026"  # fixed so prompts are reproducible (harnesses print today's date)


fn = fn_schema


def _spill(s, text: str, path: str) -> str:
    """Save a full output where the harness would (so the policy can read it back); returns the path."""
    try:
        s.ws.store(path, text)
    except OpError:
        pass
    return path


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ================================================================== palingenesis-default


def palingenesis_default() -> HarnessProfile:
    from palingenesis.rl.envs.harbor import SYSTEM, TOOL_SCHEMAS

    return HarnessProfile(
        name="palingenesis-default",
        version="1",
        system=SYSTEM,
        tools=[ToolSpec(s) for s in TOOL_SCHEMAS.values()],
        stop=Submit("submit"),
        answer=LastText(),
        legacy=True,
        emulates="HarborEnv's own prompt and tools (the row's conversation, HarborEnv.call_tool)",
    )


# palingenesis-default@2: nanocode's dialect (a minimal Claude Code alternative): a one-line system prompt, six tools
# with terse schemas, "ok" / "error: ..." / "none" / "(empty)" observations, ends at a turn without a tool call. The
# file tools run nanocode's own functions with the container's python3 (same results and Python exception wording).

NANO_SYSTEM = "Concise coding assistant. cwd: {workdir}"
NANO_TIMEOUT = 30
_NANO_SPEC = (  # name, description, {param: "type" | "type?"} (nanocode's TOOLS; make_schema maps number -> integer)
    (
        "read",
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
    ),
    ("write", "Write content to file", {"path": "string", "content": "string"}),
    (
        "edit",
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
    ),
    ("glob", "Find files by pattern, sorted by mtime", {"pat": "string", "path": "string?"}),
    ("grep", "Search files for regex pattern", {"pat": "string", "path": "string?"}),
    ("bash", "Run shell command", {"cmd": "string"}),
)


def _nano_schema(name: str, description: str, params: dict) -> dict:
    props = {k: {"type": "integer" if t.rstrip("?") == "number" else t.rstrip("?")} for k, t in params.items()}
    return fn(name, description, props, [k for k, t in params.items() if not t.endswith("?")])


NANO_TOOLS = {n: _nano_schema(n, d, p) for n, d, p in _NANO_SPEC}

_NANO_FUNCS = r"""
import glob as globlib, json, os, re, sys
def read(args):
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))
def write(args):
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"
def edit(args):
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"
def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(files, key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0, reverse=True)
    return "\n".join(files) or "none"
def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"
"""
_NANO_MAIN = r"""
TOOLS = {"read": read, "write": write, "edit": edit, "glob": glob, "grep": grep}
path = sys.argv[1]
with open(path) as f:
    name, args = json.load(f)
os.remove(path)
try:
    out = TOOLS[name](args)
except Exception as err:
    out = f"error: {err}"
sys.stdout.write(out)
"""
_NANO_PY = _NANO_FUNCS + _NANO_MAIN
# palingenesis-default@3: nanocode with two answers changed, a no-op edit or write is an error instead of "ok" (a
# small agent repeats an identical edit while the harness keeps saying ok); everything else is @2's
_NANO3_FUNCS = r"""
def write(args):
    try:
        with open(args["path"], newline="") as f:
            same = f.read() == args["content"]
    except Exception:
        same = False
    if same:
        return "error: content identical to the current file, nothing changed"
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"
def edit(args):
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    if replacement == text:
        return "error: old and new are identical, nothing changed"
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"
"""
_NANO3_PY = _NANO_FUNCS + _NANO3_FUNCS + _NANO_MAIN


def _nano_py(s, name: str, a: dict, script: str = _NANO_PY) -> str:
    """nanocode's run_tool for a file tool, run with the container's python3 in the working directory (arguments
    staged as a file: an argv entry is capped at 128 KiB)."""
    tmp = f"{s.ws.scratch}/.pgs_nano_{uuid.uuid4().hex[:12]}.json"
    s.ws.rt.put_files({tmp: json.dumps([name, a])}, "/", as_user=True)
    r = s.ws.rt.exec(f"python3 -c {shlex.quote(script)} {shlex.quote(tmp)}", 60, workdir=s.workdir)
    if r.exit_code == 127:
        return "error: python3 is not installed in the container"
    if r.exit_code != 0 or r.timed_out:  # the interpreter itself failed (run_tool's own errors exit 0)
        lines = (r.stderr or r.stdout).strip().splitlines()
        return f"error: {lines[-1] if lines else 'tool failed'}"
    return r.stdout


def _nano_bash(s, a):
    if "cmd" not in a:
        return "error: 'cmd'"
    r = ops.exec_(s.ws, str(a["cmd"]), NANO_TIMEOUT, workdir=s.workdir, shell="sh")  # Popen(shell=True): /bin/sh
    s.last_exec = r
    out = r.output + ("\n(timed out after 30s)" if r.timed_out else "")
    return out.strip() or "(empty)"


def _nano_profile(version: str, script: str, emulates: str) -> HarnessProfile:
    tools = [
        ToolSpec(NANO_TOOLS[n], _nano_bash if n == "bash" else (lambda s, a, n=n: _nano_py(s, n, a, script)))
        for n, _, _ in _NANO_SPEC
    ]
    return HarnessProfile(
        name="palingenesis-default",
        version=version,
        system=NANO_SYSTEM,
        tools=tools,
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=NANO_TIMEOUT,
        unknown_tool=lambda name, names, args: f"error: {name!r}",  # TOOLS[name] KeyError
        emulates=emulates,
    )


_NANO_EMULATES = (
    "nanocode (minimal Claude Code alternative): system prompt, TOOLS and make_schema, run_tool's 'error: {err}' "
    "wording, the agentic loop's stop at a reply without tool calls; bash under /bin/sh with stderr merged, a 30 s "
    "timeout (the partial output of a timed-out command is not kept), no output cap of its own"
)


def palingenesis_default_2() -> HarnessProfile:
    return _nano_profile("2", _NANO_PY, _NANO_EMULATES)


def palingenesis_default_3() -> HarnessProfile:
    return _nano_profile(
        "3",
        _NANO3_PY,
        _NANO_EMULATES + "; changed on purpose: an edit or write that leaves the file unchanged is an error",
    )


# ================================================================== mini-swe-agent 2.4.6

MINI_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_ENV = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
MINI_TIMEOUT = 30  # LocalEnvironmentConfig.timeout

MINI_BASH = fn(
    "bash",
    "Execute a bash command",
    {"command": {"type": "string", "description": "The bash command to execute"}},
    ["command"],
)

MINI_SYSTEM = "You are a helpful assistant that can interact with a computer."
MINI_SYSTEM_TEXT = (
    "You are a helpful assistant that can interact with a computer.\n\n"
    "Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\n"
    "Before the command, write a THOUGHT section explaining your reasoning.\n"
    "Format your response as shown in <format_example>.\n\n"
    "<format_example>\nYour reasoning and analysis here. Explain why you want to perform the action.\n\n"
    "```mswea_bash_command\nyour_command_here\n```\n</format_example>\n\n"
    "Responses that break these rules are rejected."
)

_MINI_WORKFLOW = (
    "Please solve this issue: {instruction}\n\n"
    "You can execute bash commands and edit files to implement the necessary changes.\n\n"
    "## Recommended Workflow\n\n"
    "Work step by step so that you can iterate on your changes:\n\n"
    "1. Find and read the relevant files\n"
    "2. Write a script that reproduces the issue\n"
    "3. Edit the source code to resolve it\n"
    "4. Run your script again to verify the fix\n"
    "5. Test edge cases\n"
    f"6. Submit and finish your work with the command `echo {MINI_SENTINEL}`.\n"
    "   Do not combine it with any other command. <important>After this command you cannot continue working on this "
    "task.</important>\n\n"
)
_MINI_TAIL = (
    "<system_information>\n{system_information}\n</system_information>\n\n"
    "## Useful command examples\n\n"
    "### Create a new file:\n\n{fence}\ncat <<'EOF' > newfile.py\nhello = \"world\"\nprint(hello)\nEOF\n```\n\n"
    "### Edit files with sed:\n\n{fence}\n# Replace all occurrences\nsed -i 's/old_string/new_string/g' filename.py\n\n"
    "# Replace only the first occurrence\nsed -i 's/old_string/new_string/' filename.py\n```\n\n"
    "### View file content:\n\n{fence}\n# View specific lines with numbers\nnl -ba filename.py | sed -n '10,20p'\n```\n"
)

MINI_USER = (
    _MINI_WORKFLOW + "## Command Execution Rules\n\n"
    "You issue at least one command, the system runs it in a subshell, you see the result, then you write your next "
    "command(s).\n\n"
    "Each response should include:\n\n1. **Reasoning text** explaining your analysis and plan\n"
    "2. At least one tool call with your command\n\n"
    "**CRITICAL REQUIREMENTS:**\n\n"
    "- Your response MUST include AT LEAST ONE bash tool call\n"
    "- Directory or environment variable changes are not persistent: every action runs in a new subshell (prefix an "
    "action with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...`, or write and load variables from files)\n"
    f"- Submit and finish with `echo {MINI_SENTINEL}`, alone.\n\n"
    "Example of a CORRECT response:\n<example_response>\nI need to understand the structure of the repository first. "
    'Let me list the files in the current directory.\n\n[Makes bash tool call with {"command": "ls -la"} as '
    "arguments]\n</example_response>\n\n" + _MINI_TAIL.replace("{fence}", "```bash")
)

MINI_USER_TEXT = (
    _MINI_WORKFLOW + "## Important Rules\n\n"
    "1. Every response must contain exactly one action\n"
    "2. The action must be enclosed in triple backticks\n"
    "3. Directory or environment variable changes are not persistent: every action runs in a new subshell (prefix an "
    "action with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...`, or write and load variables from files)\n\n"
    "## Formatting your response\n\nHere is an example of a correct response:\n\n<example_response>\n"
    "THOUGHT: I need to understand the structure of the repository first. Let me list the files in the current "
    "directory.\n\n```mswea_bash_command\nls -la\n```\n</example_response>\n\n"
    + _MINI_TAIL.replace("{fence}", "```mswea_bash_command")
)


def _tojson(value) -> str:
    """Jinja2's |tojson (htmlsafe json.dumps), as mini-swe-agent's observation template renders strings."""
    return (
        json.dumps(value, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def mini_observation(output: str, returncode: int, exception_info: str = "") -> str:
    """mini.yaml observation_template (tool message)."""
    exc = f', "exception_info": {_tojson(exception_info)}' if exception_info else ""
    cut = ctx.head_tail_chars(output, 10000, 5000, 5000)
    if cut is None:
        return f'{{\n  "returncode": {returncode},\n  "output": {_tojson(output)}{exc}\n}}'
    head, tail, elided = cut
    return (
        f'{{\n  "returncode": {returncode},\n  "output_head": {_tojson(head)},\n  "output_tail": {_tojson(tail)},'
        f'\n  "elided_chars": {elided},\n  "warning": "Output too long."{exc}\n}}'
    )


MINI_TEXT_WARNING = (
    "<warning>\nThe output of your last command was too long.\nPlease try a different command that produces less "
    "output.\nIf you're looking at a file you can try use head, tail or sed to view a smaller number of lines "
    "selectively.\nIf you're using grep or find and it produced too much output, you can use a more selective search "
    "pattern.\nIf you really need to see something from the full command's output, you can redirect output to a file "
    "and then search in that file.\n</warning>"
)


def mini_text_observation(output: str, returncode: int, exception_info: str = "") -> str:
    """mini_textbased.yaml observation_template (user message)."""
    head = f"<exception>{exception_info}</exception>\n" if exception_info else ""
    head += f"<returncode>{returncode}</returncode>\n"
    cut = ctx.head_tail_chars(output, 10000, 5000, 5000)
    if cut is None:
        return f"{head}<output>\n{output}</output>"
    h, t, elided = cut
    return (
        f"{head}{MINI_TEXT_WARNING}<output_head>\n{h}\n</output_head>\n<elided_chars>\n{elided} characters "
        f"elided\n</elided_chars>\n<output_tail>\n{t}\n</output_tail>"
    )


def mini_format_error(reason: str, n: int) -> str:
    return (
        "Tool call error:\n\n<error>\n" + reason + "\n</error>\n\n"
        "Here is general guidance on how to submit correct toolcalls:\n\n"
        "Every response needs to use the 'bash' tool at least once to execute commands.\n\n"
        "Call the bash tool with your command as the argument:\n- Tool: bash\n"
        '- Arguments: {"command": "your_command_here"}\n\n'
        f"If you want to end the task, please issue the following command: `echo {MINI_SENTINEL}`\n"
        "without any other command."
    )


def mini_unknown_tool(name: str, names: list[str], args: dict) -> str:
    """parse_toolcall_actions: a call of another tool is a format error (the whole turn, upstream)."""
    reason = f"Unknown tool '{name}'." + ("" if "command" in args else "Missing 'command' argument in bash tool call.")
    return mini_format_error(reason, 0)


def mini_text_format_error(reason: str, n: int) -> str:
    return (
        "Format error:\n\n<error>\n" + reason + "\n</error>\n\n"
        "Here is general guidance on how to format your response:\n\n"
        f"Please always provide EXACTLY ONE action in triple backticks, found {n} actions.\n\n"
        "Please format your action in triple backticks as shown in <response_example>.\n\n"
        "<response_example>\nHere are some thoughts about why you want to perform the action.\n\n"
        "```mswea_bash_command\n<action>\n```\n</response_example>\n\n"
        "If you have completed your assignment, please consult the first message about how to\n"
        "submit your solution (you will not be able to continue working on this task after that)."
    )


def _mini_bash(text_mode: bool):
    def bind(s, a):
        cmd = str(a.get("command", ""))
        r = ops.exec_(s.ws, cmd, MINI_TIMEOUT, shell="sh", env=MINI_ENV)
        exc = ""
        if r.timed_out:  # subprocess.TimeoutExpired, as LocalEnvironment reports it
            exc = (
                f"An error occurred while executing the command: Command '{cmd}' timed out after {MINI_TIMEOUT} seconds"
            )
            r = ops.ExecOut(r.output, -1, True)
        s.last_exec = r
        return (mini_text_observation if text_mode else mini_observation)(r.output, r.exit_code, exc)

    return bind


def mini_swe_agent() -> HarnessProfile:
    return HarnessProfile(
        name="mini-swe-agent",
        version="2.4.6",
        system=MINI_SYSTEM,
        user=MINI_USER,
        tools=[ToolSpec(MINI_BASH, _mini_bash(False))],
        protocol=Native(),
        stop=Sentinel(MINI_SENTINEL, "bash"),
        answer=StopAnswer(None),
        context=Context("none"),
        format_error=mini_format_error,
        exec_timeout=MINI_TIMEOUT,
        unknown_tool=mini_unknown_tool,
        emulates="minisweagent 2.4.6 config/mini.yaml, models/utils/actions_toolcall.py, environments/local.py",
    )


def mini_swe_agent_text() -> HarnessProfile:
    return HarnessProfile(
        name="mini-swe-agent-text",
        version="2.4.6",
        system=MINI_SYSTEM_TEXT,
        user=MINI_USER_TEXT,
        tools=[ToolSpec(MINI_BASH, _mini_bash(True))],
        protocol=FencedCommand("mswea_bash_command", "bash", "command"),
        stop=Sentinel(MINI_SENTINEL, "bash"),
        answer=StopAnswer(None),
        context=Context("none"),
        format_error=mini_text_format_error,
        exec_timeout=MINI_TIMEOUT,
        emulates="minisweagent 2.4.6 config/mini_textbased.yaml, models/utils/actions_text.py",
    )


# ================================================================== opencode 1.19.2

OC_MAX_LINES, OC_MAX_BYTES, OC_TIMEOUT_MS = 2000, 50 * 1024, 120_000

OPENCODE_SYSTEM = (
    "You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions "
    "below and the tools available to you to assist the user.\n\n"
    "IMPORTANT: Never generate or guess URLs unless you are confident they help the user with programming.\n\n"
    "# Tone and style\n"
    "Be concise, direct and to the point. Your output is shown in a terminal (GitHub-flavored markdown, monospace). "
    "Text you write outside tool calls is shown to the user; use tools only to do the work, never to talk to the user "
    "(no echo, no code comments for that). Answer in fewer than 4 lines unless the user asks for detail, with no "
    "preamble or postamble: one-word answers are best (user: 2 + 2 / assistant: 4). No emojis unless asked.\n\n"
    "# Proactiveness\n"
    "Do what was asked, including reasonable follow-up actions, without surprising the user; do not explain your "
    "changes unless asked.\n\n"
    "# Following conventions\n"
    "Understand the code's conventions before changing it; never assume a library is available; follow security best "
    "practices. Do not add comments unless asked.\n\n"
    "# Doing tasks\n"
    "Search to understand the codebase and the request, implement the solution with the tools, then verify it (run "
    "the project's tests, lint and type checks when they exist). Never commit unless the user asks.\n\n"
    "# Tool usage policy\n"
    "Prefer the dedicated tools (read, edit, write, glob, grep) to shell commands for files, and make independent tool "
    "calls in parallel in one message.\n\n"
    "# Code References\n"
    "When referring to code, use the pattern `file_path:line_number`.\n"
    "You are powered by the model named {model}. The exact model ID is vllm/{model}\n"
    "Here is some useful information about the environment you are running in:\n"
    "<env>\n  Working directory: {workdir}\n  Workspace root folder: {git_root}\n  Is directory a git repo: {git_repo}\n"
    "  Platform: linux\n  Today's date: {date}\n</env>"
)

OPENCODE_TOOLS = {
    "bash": fn(
        "bash",
        "Executes a given bash command in a persistent shell session with optional timeout.\n\n"
        "All commands run in the current working directory by default; use the `workdir` parameter instead of "
        "`cd <directory> && <command>`. This tool is for terminal operations (git, npm, docker, ...): do not use it "
        "to read, write, edit or search files, use the dedicated tools. Quote paths that contain spaces.\n\n"
        f"Without a timeout, commands time out after {OC_TIMEOUT_MS}ms. If the output exceeds {OC_MAX_LINES} lines "
        f"or {OC_MAX_BYTES} bytes, it is truncated and the full output is written to a file you can read with "
        "offset/limit or search with grep. Run independent commands as parallel tool calls; chain dependent ones with "
        "'&&'; do not separate commands with newlines.",
        {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"type": "integer", "exclusiveMinimum": 0, "description": "Optional timeout in milliseconds"},
            "workdir": {
                "type": "string",
                "description": "The working directory to run the command in. Defaults to the "
                "current directory. Use this instead of 'cd' commands.",
            },
        },
        ["command"],
    ),
    "read": fn(
        "read",
        "Read a file or directory from the local filesystem; a missing path is an error. filePath should be absolute. "
        f"By default returns up to {OC_MAX_LINES} lines from the start; offset is the 1-indexed line to start from "
        "(call again with a larger offset for later sections). Each line is prefixed with its number as `<line>: "
        "<content>`; a directory lists its entries one per line, subdirectories with a trailing `/`. Lines longer than "
        "2000 characters are truncated. Use grep to find content in large files and glob to find file names.",
        {
            "filePath": {"type": "string", "description": "The absolute path to the file or directory to read"},
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "The line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "minimum": 0,
                "description": "The maximum number of lines to read (defaults to 2000)",
            },
        },
        ["filePath"],
    ),
    "write": fn(
        "write",
        "Writes a file to the local filesystem, overwriting an existing file. Read an existing file before "
        "overwriting it; prefer editing existing files; do not create documentation files unless asked.",
        {
            "content": {"type": "string", "description": "The content to write to the file"},
            "filePath": {
                "type": "string",
                "description": "The absolute path to the file to write (must be absolute, not relative)",
            },
        },
        ["content", "filePath"],
    ),
    "edit": fn(
        "edit",
        "Performs exact string replacements in files. Read the file first and keep its exact indentation (what follows "
        "the `N: ` line prefix of read; never include the prefix in oldString or newString). The edit fails if "
        "oldString is not found, or is found more than once: give more surrounding context, or set replaceAll to "
        "change every instance (e.g. renaming a variable).",
        {
            "filePath": {"type": "string", "description": "The absolute path to the file to modify"},
            "oldString": {"type": "string", "description": "The text to replace"},
            "newString": {
                "type": "string",
                "description": "The text to replace it with (must be different from oldString)",
            },
            "replaceAll": {"type": "boolean", "description": "Replace all occurrences of oldString (default false)"},
        },
        ["filePath", "oldString", "newString"],
    ),
    "glob": fn(
        "glob",
        "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like "
        '"**/*.js" or "src/**/*.ts"\n- Returns matching file paths\n- Use it to find files by name patterns; '
        "batch several speculative searches in one response",
        {
            "pattern": {"type": "string", "description": "The glob pattern to match files against"},
            "path": {
                "type": "string",
                "description": "The directory to search in. If not specified, the current working "
                "directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "
                '"undefined" or "null" - simply omit it for the default behavior. Must be a valid directory path '
                "if provided.",
            },
        },
        ["pattern"],
    ),
    "grep": fn(
        "grep",
        "- Fast content search tool that works with any codebase size\n- Searches file contents using regular "
        'expressions (eg. "log.*Error", "function\\s+\\w+")\n- Filter files with the include parameter (eg. '
        '"*.js", "*.{ts,tsx}")\n- Returns file paths and line numbers with matching lines',
        {
            "pattern": {"type": "string", "description": "The regex pattern to search for in file contents"},
            "path": {
                "type": "string",
                "description": "The directory to search in. Defaults to the current working directory.",
            },
            "include": {
                "type": "string",
                "description": 'File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")',
            },
        },
        ["pattern"],
    ),
}


for _n, _t in OPENCODE_TOOLS.items():  # parameter schemas as recorded from opencode 1.19.2 under Harbor 0.24
    _t["function"]["parameters"] = OPENCODE_PARAMETERS[_n]


def _oc_bash(s, a):
    timeout = a.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout < 0):
        return f"Invalid timeout value: {timeout}. Timeout must be a positive number."
    ms = int(timeout or OC_TIMEOUT_MS)
    wd = s.ws.abspath(a["workdir"]) if a.get("workdir") else None
    r = ops.exec_(s.ws, str(a.get("command", "")), ms / 1000, workdir=wd)
    s.last_exec = r
    t = ctx.tail_lines(r.output, OC_MAX_LINES, OC_MAX_BYTES)
    out = t.text or "(no output)"
    if t.truncated:
        path = _spill(s, r.output, f"{s.ws.scratch}/opencode/tool-output/tool_{_digest(r.output)}")
        out = f"...output truncated...\n\nFull output saved to: {path}\n\n" + out
    if r.timed_out:
        out += (
            "\n\n<shell_metadata>\nshell tool terminated command after exceeding timeout " + str(ms) + " ms. If "
            "this command is expected to take longer and is not waiting for interactive input, retry with a larger "
            "timeout value in milliseconds.\n</shell_metadata>"
        )
    return out


def _oc_miss(s, path: str) -> str:
    parent, base = posixpath.dirname(path), posixpath.basename(path).lower()
    try:
        entries, _ = ops.list_dir(s.ws, parent, 1, hidden=True)
    except OpError:
        entries = []
    near = [
        posixpath.join(parent, e.rstrip("/"))
        for e in entries
        if base and (base in e.rstrip("/").lower() or e.rstrip("/").lower() in base)
    ][:3]
    if near:
        return f"File not found: {path}\n\nDid you mean one of these?\n" + "\n".join(near)
    return f"File not found: {path}"


def _oc_read(s, a):
    path = s.ws.abspath(a.get("filePath", ""))
    offset, limit = int(a.get("offset") or 1), int(a.get("limit") or OC_MAX_LINES)
    k = s.ws.kind(path)
    if k is None:
        return _oc_miss(s, path)
    if k == "dir":
        entries, _ = ops.list_dir(s.ws, path, 1, hidden=True)
        sliced = entries[offset - 1 : offset - 1 + limit]
        more = offset - 1 + len(sliced) < len(entries)
        note = (
            (
                f"\n(Showing {len(sliced)} of {len(entries)} entries. Use 'offset' parameter to read beyond entry "
                f"{offset + len(sliced)})"
            )
            if more
            else f"\n({len(entries)} entries)"
        )
        return "\n".join(
            [f"<path>{path}</path>", "<type>directory</type>", "<entries>", "\n".join(sliced), note, "</entries>"]
        )
    text = ops.read(s.ws, path)
    if "\x00" in text[:8192]:
        return f"Cannot read binary file: {path}"
    lines = ops.lines_of(text)
    count = len(lines)
    if count < offset and not (count == 0 and offset == 1):
        return f"Offset {offset} is out of range for this file ({count} lines)"
    raw, size, cut, more = [], 0, False, False
    for line in lines[offset - 1 :]:
        if len(raw) >= limit:
            more = True
            break
        line = line[:2000] + "... (line truncated to 2000 chars)" if len(line) > 2000 else line
        n = len(line.encode()) + (1 if raw else 0)
        if size + n > OC_MAX_BYTES:
            cut = more = True
            break
        raw.append(line)
        size += n
    out = f"<path>{path}</path>\n<type>file</type>\n<content>\n"
    out += "\n".join(f"{i + offset}: {line}" for i, line in enumerate(raw))
    last = offset + len(raw) - 1
    if cut:
        out += f"\n\n(Output capped at 50 KB. Showing lines {offset}-{last}. Use offset={last + 1} to continue.)"
    elif more:
        out += f"\n\n(Showing lines {offset}-{last} of {count}. Use offset={last + 1} to continue.)"
    else:
        out += f"\n\n(End of file - total {count} lines)"
    return out + "\n</content>"


def _oc_write(s, a):
    path = s.ws.abspath(a.get("filePath", ""))
    try:
        ops.write(s.ws, path, str(a.get("content", "")))
    except OpError as e:
        return (
            f"EISDIR: illegal operation on a directory, open '{path}'"
            if e.kind == "is_dir"
            else f"Failed to write {path}: {e.data.get('detail', e.kind)}"
        )
    return "Wrote file successfully."


OC_EDIT_ERRORS = {
    "identical": "No changes to apply: oldString and newString are identical.",
    "empty_old": "oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use "
    "write for an intentional full-file replacement.",
    "no_match": "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and "
    "line endings.",
    "multiple": "Found multiple matches for oldString. Provide more surrounding context to make the match unique.",
}


def _oc_edit(s, a):
    path = s.ws.abspath(a.get("filePath", ""))
    old, new = str(a.get("oldString", "")), str(a.get("newString", ""))
    if old == new:
        return OC_EDIT_ERRORS["identical"]
    k = s.ws.kind(path)
    if old == "" and k is None:  # an empty oldString creates the file
        ops.write(s.ws, path, new)
        return "Edit applied successfully."
    if k is None:
        return f"File {path} not found"
    if k == "dir":
        return f"Path is a directory, not a file: {path}"
    try:
        ops.replace(s.ws, path, old, new, bool(a.get("replaceAll")))
    except OpError as e:
        return OC_EDIT_ERRORS.get(e.kind, f"Error: {e.kind}")
    return "Edit applied successfully."


def _oc_glob(s, a):
    root = s.ws.abspath(a.get("path") or s.ws.workdir)
    try:
        files = ops.glob(s.ws, str(a.get("pattern", "")), root, limit=100)
    except OpError as e:
        return f"glob path must be a directory: {root}" if e.kind == "not_dir" else "No files found"
    if not files:
        return "No files found"
    out = list(files)
    if len(files) == 100:
        out += [
            "",
            "(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)",
        ]
    return "\n".join(out)


def _oc_grep(s, a):
    try:
        rows = ops.grep(s.ws, str(a.get("pattern", "")), a.get("path") or None, a.get("include") or None, limit=100)
    except OpError:
        rows = []
    if not rows:
        return "No files found"
    more = len(rows) == 100
    out, current = [f"Found {len(rows)} matches{' (more matches available)' if more else ''}"], ""
    for path, line, text in rows:
        if path != current:
            if current:
                out.append("")
            current = path
            out.append(f"{path}:")
        out.append(f"  Line {line}: {text}\n")  # ripgrep's match text keeps its line terminator
    if more:
        out += ["", "(Results truncated. Consider using a more specific path or pattern.)"]
    return "\n".join(out)


def opencode() -> HarnessProfile:
    binds = {
        "bash": _oc_bash,
        "read": _oc_read,
        "write": _oc_write,
        "edit": _oc_edit,
        "glob": _oc_glob,
        "grep": _oc_grep,
    }
    return HarnessProfile(
        name="opencode",
        version="1.19.2",
        system=OPENCODE_SYSTEM,
        tools=[ToolSpec(OPENCODE_TOOLS[n], binds[n]) for n in ("bash", "edit", "glob", "grep", "read", "write")],
        user='"{instruction}\n"',  # Harbor 0.24's opencode adapter passes the instruction quoted (recorded)
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=OC_TIMEOUT_MS / 1000,
        context=Context("compaction", emulated=False, note="automatic summary near the context limit"),
        alias=lambda name, args, names: (name.lower(), args) if name.lower() in names else None,  # repairToolCall
        unknown_tool=lambda name, names, args: (
            "The arguments provided to the tool are invalid: Model tried to call "
            f"unavailable tool '{name}'. Available tools: "
            f"{', '.join(sorted([*names, 'invalid']))}."
        ),
        values={"model": MODEL, "date": DATE},
        emulates="opencode v1.19.2 session/prompt/default.txt, session/system.ts, tool/{shell,read,write,edit,glob,"
        "grep}.ts (task, todowrite, webfetch, question disabled as for single-agent runs)",
    )


# ================================================================== pi 1.3.0

PI_MAX_LINES, PI_MAX_BYTES = 2000, 50 * 1024

PI_SYSTEM = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, "
    "executing commands, editing code, and writing new files.\n\n"
    "<tools>\n- read: Read file contents\n- bash: Execute bash commands (ls, grep, find, etc.)\n"
    "- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n"
    "- write: Create or overwrite files\n\n"
    "In addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>\n\n"
    "<rules>\n- Use bash for file operations like ls, rg, find\n- Use read to examine files instead of cat or sed.\n"
    "- Use edit for precise changes (edits[].oldText must match exactly)\n"
    "- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] "
    "instead of multiple edit calls\n"
    "- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit "
    "overlapping or nested edits. Merge nearby changes into one edit.\n"
    "- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large "
    "unchanged regions.\n- Use write only for new files or complete rewrites.\n- Be concise in your responses\n"
    "- Show file paths clearly when working with files\n</rules>\n\n"
    "<cwd>\n{workdir}\n</cwd>"
)

PI_TOOLS = {
    "read": fn(
        "read",
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as "
        f"attachments. For text files, output is truncated to {PI_MAX_LINES} lines or 50KB (whichever is hit first). "
        "Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
        ["path"],
    ),
    "bash": fn(
        "bash",
        "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to "
        f"last {PI_MAX_LINES} lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp "
        "file. Optionally provide a timeout in seconds.",
        {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
        },
        ["command"],
    ),
    "edit": fn(
        "edit",
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping "
        "region of the original file. If two changes affect the same block or nearby lines, merge them into one edit "
        "instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant "
        "changes.",
        {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text for one targeted replacement. It must be "
                            "unique in the original file and must not overlap with any other edits[].oldText in the "
                            "same call.",
                        },
                        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                    },
                    "required": ["oldText", "newText"],
                },
                "description": "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead.",
            },
        },
        ["path", "edits"],
    ),
    "write": fn(
        "write",
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates "
        "parent directories.",
        {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        ["path", "content"],
    ),
}

PI_BASH_CAP = 600  # pi has no default timeout; an episode still needs one


def _pi_output(s, output: str, empty: str) -> str:
    """pi's formatOutput: the tail within 2000 lines / 50KB, the full output spilled to a temp file when cut."""
    t = ctx.tail_lines(output, PI_MAX_LINES, PI_MAX_BYTES)
    text = t.text or empty
    if t.truncated:
        path = _spill(s, output, f"{s.ws.scratch}/pi-bash-{_digest(output)}.log")
        if t.by == "lines":
            text += f"\n\n[Showing lines {t.first_line}-{t.last_line} of {t.total_lines}. Full output: {path}]"
        else:
            text += (
                f"\n\n[Showing lines {t.first_line}-{t.last_line} of {t.total_lines} "
                f"({ctx.fmt_size(PI_MAX_BYTES)} limit). Full output: {path}]"
            )
    return text


def _pi_status(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status


def _pi_bash(s, a):
    timeout = a.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        return "Invalid timeout: must be a finite number of seconds"
    limit = float(timeout) if timeout else PI_BASH_CAP
    r = ops.exec_(s.ws, str(a.get("command", "")), limit)
    s.last_exec = r
    if r.timed_out:
        return _pi_status(_pi_output(s, r.output, ""), f"Command timed out after {limit:g} seconds")
    text = _pi_output(s, r.output, "(no output)")
    return _pi_status(text, f"Command exited with code {r.exit_code}") if r.exit_code != 0 else text


def _pi_read(s, a):
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown)
    k = s.ws.kind(path)
    if k is None:
        return f"ENOENT: no such file or directory, access '{path}'"
    if k == "dir":
        return "EISDIR: illegal operation on a directory, read"
    lines = ops.read(s.ws, path).split("\n")
    offset, limit = a.get("offset"), a.get("limit")
    start = max(0, int(offset) - 1) if offset else 0
    if start >= len(lines):
        return f"Offset {offset} is beyond end of file ({len(lines)} lines total)"
    end = min(start + int(limit), len(lines)) if limit is not None else len(lines)
    t = ctx.head_lines("\n".join(lines[start:end]), PI_MAX_LINES, PI_MAX_BYTES)
    first = start + 1
    if t.truncated and not t.text and lines[start] and len(lines[start].encode()) > PI_MAX_BYTES:
        size = ctx.fmt_size(len(lines[start].encode()))
        return (
            f"[Line {first} is {size}, exceeds {ctx.fmt_size(PI_MAX_BYTES)} limit. Use bash: sed -n '{first}p' "
            f"{shown} | head -c {PI_MAX_BYTES}]"
        )
    if t.truncated:
        last = first + t.last_line - 1
        if t.by == "lines":
            return f"{t.text}\n\n[Showing lines {first}-{last} of {len(lines)}. Use offset={last + 1} to continue.]"
        return (
            f"{t.text}\n\n[Showing lines {first}-{last} of {len(lines)} ({ctx.fmt_size(PI_MAX_BYTES)} limit). "
            f"Use offset={last + 1} to continue.]"
        )
    if limit is not None and end < len(lines):
        return f"{t.text}\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
    return t.text


def _pi_edit_error(e: OpError, shown: str, single: bool) -> str:
    i = e.data.get("index", 0)
    if e.kind == "no_match":
        return (
            (
                f"Could not find the exact text in {shown}. The old text must match exactly including all whitespace "
                "and newlines."
            )
            if single
            else (
                f"Could not find edits[{i}] in {shown}. The oldText must match "
                "exactly including all whitespace and newlines."
            )
        )
    if e.kind == "multiple":
        n = e.data.get("count")
        return (
            (
                f"Found {n} occurrences of the text in {shown}. The text must be unique. Please provide more context "
                "to make it unique."
            )
            if single
            else (
                f"Found {n} occurrences of edits[{i}] in {shown}. Each oldText "
                "must be unique. Please provide more context to make it unique."
            )
        )
    if e.kind == "empty_old":
        return (
            f"oldText must not be empty in {shown}." if single else f"edits[{i}].oldText must not be empty in {shown}."
        )
    if e.kind == "identical":
        return (
            (
                f"No changes made to {shown}. The replacement produced identical content. This might indicate an "
                "issue with special characters or the text not existing as expected."
            )
            if single
            else f"No changes made to {shown}. The replacements produced identical content."
        )
    if e.kind == "overlap":
        return (
            f"edits[{e.data['first']}] and edits[{e.data['second']}] overlap in {shown}. Merge them into one edit "
            "or target disjoint regions."
        )
    return f"Could not edit file: {shown}. Error code: {'EISDIR' if e.kind == 'is_dir' else e.kind}."


def _pi_edit(s, a):
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown)
    edits = a.get("edits")
    # prepareEditArguments (dist/core/tools/edit.js:43-73) repairs the arguments before validation: edits sent as a
    # JSON string is parsed (an array, or one {oldText, newText} object), a single object becomes a one-element list,
    # and top-level oldText/newText (the legacy shape) is appended to edits
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            edits = parsed
        elif (
            isinstance(parsed, dict)
            and isinstance(parsed.get("oldText"), str)
            and isinstance(parsed.get("newText"), str)
        ):
            edits = [parsed]
    if isinstance(edits, dict):
        edits = [edits]
    if isinstance(a.get("oldText"), str) and isinstance(a.get("newText"), str):
        edits = [*(edits if isinstance(edits, list) else []), {"oldText": a["oldText"], "newText": a["newText"]}]
    if not isinstance(edits, list) or not edits:
        return "Edit tool input is invalid. edits must contain at least one replacement."
    if s.ws.kind(path) is None:
        return f"Could not edit file: {shown}. Error code: ENOENT."
    pairs = [(str((e or {}).get("oldText", "")), str((e or {}).get("newText", ""))) for e in edits]
    try:
        r = ops.multi_replace(s.ws, path, pairs)
    except OpError as e:
        return _pi_edit_error(e, shown, len(pairs) == 1)
    return f"Successfully replaced {r.count} block(s) in {shown}."


def _pi_write(s, a):
    shown = str(a.get("path", ""))
    path = s.ws.abspath(shown)
    try:
        ops.write(s.ws, path, str(a.get("content", "")))
    except OpError as e:
        return (
            f"EISDIR: illegal operation on a directory, open '{path}'"
            if e.kind == "is_dir"
            else f"Failed to write {shown}: {e.data.get('detail', e.kind)}"
        )
    return f"Successfully wrote to {shown}"


def pi() -> HarnessProfile:
    binds = {"read": _pi_read, "bash": _pi_bash, "edit": _pi_edit, "write": _pi_write}
    return HarnessProfile(
        name="pi",
        version="1.3.0",
        system=PI_SYSTEM,
        tools=[ToolSpec(PI_TOOLS[n], binds[n]) for n in ("read", "bash", "edit", "write")],
        stop=NoCall(),
        answer=LastText(),
        exec_timeout=PI_BASH_CAP,
        context=Context("compaction", emulated=False, note="compaction module on overflow"),
        unknown_tool=lambda name, names, args: f"Tool {name} not found",
        emulates="@earendil-works/pi-coding-agent 1.3.0 dist/core/system-prompt.js, dist/core/tools/{read,bash,edit,"
        "edit-diff,write,truncate}.js (the <docs> section about pi's own documentation is left out)",
    )


# ================================================================== OpenHands SDK 1.55.1

OH_MAX_OUTPUT = 30000  # MAX_CMD_OUTPUT_SIZE
OH_EDITOR_MAX = 16000  # MAX_RESPONSE_LEN_CHAR
OH_SOFT_TIMEOUT = 30
OH_ERROR = "[An error occurred during execution.]\n"
OH_TIMEOUT_HELP = (
    "You may wait longer to see additional output by sending empty command '', send other commands to "
    'interact with the current process, send keys ("C-c", "C-z", "C-d") to interrupt/kill the '
    "previous command before sending your new command, or use the timeout parameter in terminal for "
    "future commands."
)
OH_CLIP = (
    "<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>"
)
OH_FILE_CLIP = (
    "<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with `grep -n` in order to find the "
    "line numbers of what you are looking for.</NOTE>"
)

OPENHANDS_SYSTEM = (
    "<SOUL>\nYou are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\n"
    "</SOUL>\n\n"
    "<ROLE>\n* Your primary role is to assist users by executing commands, modifying code, and solving technical "
    "problems effectively. Be thorough, methodical, and prioritize quality over speed.\n* If the user asks a question, "
    'like "why is X happening", don\'t try to fix the problem: answer the question.\n</ROLE>\n\n'
    "<EFFICIENCY>\n* Each action is somewhat expensive: combine actions where you can (several shell commands in one, "
    "sed and grep to edit or view several files at once).\n* Explore with find, grep and git and narrow filters.\n"
    "</EFFICIENCY>\n\n"
    "<FILE_SYSTEM_GUIDELINES>\n* Do not assume a path the user gives is relative to the working directory: explore "
    "first.\n* For large files, use file_editor view with view_range instead of reading the whole file.\n* Edit files "
    "in place rather than creating renamed copies, and remove temporary files you created when you are done.\n"
    "</FILE_SYSTEM_GUIDELINES>\n\n"
    "<CODE_QUALITY>\n* Write clean, minimal code; understand the code before changing it; keep changes focused.\n"
    "</CODE_QUALITY>\n\n"
    "<PROBLEM_SOLVING_WORKFLOW>\n1. EXPLORATION: explore the relevant files and understand the context\n"
    "2. ANALYSIS: consider several approaches and choose the most promising\n3. TESTING: for a bug, reproduce it with a "
    "test before fixing\n4. IMPLEMENTATION: make focused, minimal changes\n5. VERIFICATION: test your implementation "
    "thoroughly\n</PROBLEM_SOLVING_WORKFLOW>\n\n"
    "<SECURITY_RISK_ASSESSMENT>\nSet security_risk on tools that support it: LOW for safe read-only actions, MEDIUM for "
    "project-scoped edits or execution, HIGH for system-level or untrusted operations (system settings, global "
    "installs, sudo, deleting critical files, running untrusted code, sending data out).\n</SECURITY_RISK_ASSESSMENT>\n\n"
    "<ENVIRONMENT_SETUP>\n* If a needed tool or dependency is missing, install it when possible (dependency files "
    "first).\n</ENVIRONMENT_SETUP>\n\n"
    "<TROUBLESHOOTING>\n* After repeated failed attempts, step back, list the likely causes and address the most "
    "likely first.\n</TROUBLESHOOTING>\n\n"
    "<PROCESS_MANAGEMENT>\n* Do not kill processes by broad patterns (pkill -f server); find the exact PID (ps aux) "
    "and kill that.\n</PROCESS_MANAGEMENT>"
)

_OH_SUMMARY = {
    "type": "string",
    "description": "A concise summary (approximately 10 words) describing what this "
    "specific action does. Focus on the key operation and target. Example: 'List all Python files in "
    "current directory'",
}
_OH_RISK = {
    "type": "string",
    "description": "Security risk levels for actions.\n\nBased on OpenHands security risk "
    "levels but adapted for agent-sdk.\nInteger values allow for easy comparison and ordering.",
    "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"],
}

OPENHANDS_TOOLS = {
    "terminal": fn(
        "terminal",
        "Execute a shell command in the terminal within a persistent shell session.\n\n"
        "* One command at a time; chain several with `&&` or `;`.\n"
        "* Environment variables, virtual environments and working directory changes persist across commands.\n"
        f"* Commands pause after {OH_SOFT_TIMEOUT} seconds without new output unless you set a longer `timeout`; a "
        "command that returns exit code `-1` is still running: with `is_input=true` send an empty command for more "
        "output, text to its STDIN, or keys such as `C-c`, `C-d`, `C-z`.\n"
        "* Run long-lived commands in the background with output redirected to a file.\n"
        "* Do not use `set -e` or `set -euo pipefail`. Prefer absolute paths; avoid excessive `cd`.\n"
        "* Large output may be truncated. `reset=true` starts a fresh terminal (clears variables, cwd and processes).",
        {
            "security_risk": _OH_RISK,
            "summary": _OH_SUMMARY,
            "command": {
                "type": "string",
                "description": "The shell command to execute. Can be empty string to view "
                "additional logs when the previous exit code is `-1`. Can be a special key name when `is_input` "
                "is True: `C-c` (Ctrl+C), `C-d` (Ctrl+D/EOF), `C-z` (Ctrl+Z), or any `C-<letter>` for Ctrl "
                "sequences; navigation keys `UP`, `DOWN`, `LEFT`, `RIGHT`, `HOME`, `END`, `PGUP`, `PGDN`; and "
                "`TAB`, `ESC`, `BS` (Backspace), `ENTER`. You can only execute one command at a time. Use the "
                "platform-appropriate shell syntax described in the tool description when chaining commands.",
            },
            "is_input": {
                "type": "boolean",
                "description": "If True, the command is an input to the running process. If "
                "False, the command is executed in the terminal session. Default is False.",
            },
            "timeout": {
                "type": "number",
                "description": "Optional. Sets a maximum time limit (in seconds) for running "
                "the command. If the command takes longer than this limit, you’ll be asked whether to "
                "continue or stop it. If you don’t set a value, the command will instead pause and ask for "
                "confirmation when it produces no new output for 30 seconds. Use a higher value if the command "
                "is expected to take a long time (like installation or testing), or if it has a known fixed "
                "duration (like sleep).",
            },
            "reset": {
                "type": "boolean",
                "description": "If True, reset the terminal by creating a new session. Use this "
                "only when the terminal becomes unresponsive. Note that all previously set environment variables "
                "and session state will be lost after reset. Cannot be used with is_input=True.",
            },
        },
        ["command"],
    ),
    "file_editor": fn(
        "file_editor",
        "Custom editing tool for viewing, creating and editing files in plain-text format\n"
        "* State is persistent across command calls\n"
        "* `view` on a file shows it with `cat -n` line numbers; on a directory it lists non-hidden files and "
        "directories up to 2 levels deep\n"
        "* `create` cannot overwrite an existing file\n"
        "* Long output is truncated and marked with `<response clipped>`\n"
        "* `undo_edit` reverts the last edit made to the file at `path`\n"
        "* Always use absolute paths. `old_str` must match EXACTLY one or more consecutive lines of the file, "
        "whitespace and indentation included, and must be unique (include 3-5 lines of context); `new_str` replaces it "
        "and must differ from it. Prefer several edits to one file in a single message.\n\n"
        "Your current working directory is: {workdir}\n"
        "When exploring project structure, start with this directory instead of the root filesystem.",
        {
            "security_risk": _OH_RISK,
            "summary": _OH_SUMMARY,
            "command": {
                "type": "string",
                "description": "The commands to run. Allowed options are: `view`, `create`, "
                "`str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
            },
            "path": {"type": "string", "description": "Absolute path to file or directory."},
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
                "string (if not given, no string will be added). Required parameter of `insert` command containing "
                "the string to insert.",
            },
            "insert_line": {
                "type": "integer",
                "description": "Required parameter of `insert` command. The `new_str` will "
                "be inserted AFTER the line `insert_line` of `path`.",
            },
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional parameter of `view` "
                "command when `path` points to a file. If none is given, the full file is shown. If provided, "
                "the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 "
                "and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` "
                "to the end of the file.",
            },
        },
        ["command", "path"],
    ),
    "finish": fn(
        "finish",
        "Signals the completion of the current task or conversation: when the task is done, or when you cannot "
        "proceed. The message summarises what you did and the results, the next steps for the user, why you could not "
        "complete the task, or a follow-up question.",
        {"summary": _OH_SUMMARY, "message": {"type": "string", "description": "Final message to send to the user."}},
        ["message"],
    ),
    "think": fn(
        "think",
        "Use the tool to think about something. It will not obtain new information or make any changes to the "
        "repository, but just log the thought. Use it when complex reasoning or brainstorming is needed (possible "
        "fixes for a bug, failing tests, refactoring approaches, design decisions, debugging hypotheses).",
        {"summary": _OH_SUMMARY, "thought": {"type": "string", "description": "The thought to log."}},
        ["thought"],
    ),
}


OH_ALIASES = {
    "bash": "terminal",
    "command": "terminal",
    "execute": "terminal",
    "execute_bash": "terminal",
    "git": "terminal",
    "reset": "terminal",
    "str_replace": "file_editor",
    "str_replace_editor": "file_editor",
}


def openhands_alias(name: str, args: dict, names: list[str]):
    """normalize_tool_call (openhands-sdk agent/utils.py): legacy names, malformed names ("str_replace</parameter"),
    git/reset as command prefixes, bare find/git/ls/pwd as terminal commands."""
    base = (re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", name) or [name])[0]
    if base != name and base in names:
        return base, args
    target = OH_ALIASES.get(base)
    if target in names:
        if target == "terminal" and base in ("git", "reset"):
            cmd = str(args.get("command") or "")
            keep = {k: v for k, v in args.items() if k in ("security_risk", "summary")}
            keep["command"] = base if not cmd else (cmd if cmd.split(maxsplit=1)[:1] == [base] else f"{base} {cmd}")
            return target, keep
        return target, args
    if "terminal" in names and base in ("find", "git", "ls", "pwd") and not args:
        return "terminal", {"command": base}
    return None


def _oh_terminal(s, a):
    cmd = str(a.get("command", ""))
    if a.get("reset"):
        if a.get("is_input"):
            return OH_ERROR + "Cannot use reset=True with is_input=True"
        ops.reset_shell(s.ws)
        if not cmd.strip():
            return (
                "Terminal session has been reset. All previous environment variables and session state have been "
                "cleared."
            )
    if a.get("is_input"):  # no process is ever left running between calls here
        return OH_ERROR + "No previous running command to interact with."
    if not cmd.strip():
        return OH_ERROR + "No previous running command to retrieve logs from."
    timeout = a.get("timeout")
    hard = isinstance(timeout, (int, float)) and timeout > 0
    r = ops.exec_(s.ws, cmd, float(timeout) if hard else OH_SOFT_TIMEOUT, persistent=True)
    s.last_exec = r
    out = r.output
    if r.timed_out:
        suffix = (
            f"\n[The command timed out after {timeout:g} seconds. {OH_TIMEOUT_HELP}]"
            if hard
            else f"\n[The command has no new output after {OH_SOFT_TIMEOUT} seconds. {OH_TIMEOUT_HELP}]"
        )
        text = out + suffix
    else:
        text = out.rstrip("\n") + f"\n[The command completed with exit code {r.exit_code}.]"
        if r.cwd:
            text += f"\n[Current working directory: {r.cwd}]"
        text += f"\n[Command finished with exit code {r.exit_code}]"
    return ctx.middle_clip(text, OH_MAX_OUTPUT, OH_CLIP)


def _oh_cat_n(lines: list[str], start: int, description: str) -> str:
    body = ctx.middle_clip("\n".join(lines), OH_EDITOR_MAX, OH_FILE_CLIP).split("\n")
    out = f"Here's the result of running `cat -n` on {description}:\n"
    if body and body != [""]:
        out += "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(body)) + "\n"
    return out


def _oh_snippet(after: str, line: int, n_new: int) -> tuple[list[str], int]:
    lines = ops.lines_of(after)
    start = max(0, line - 1 - 4)
    end = min(len(lines), line - 1 + n_new + 4)
    return lines[start:end], start + 1


def _oh_param(name: str, value, reason: str) -> str:
    return OH_ERROR + f"Invalid `{name}` parameter: {value}. {reason}"


def _oh_file_editor(s, a):
    command, path = a.get("command"), str(a.get("path", ""))
    if command not in ("view", "create", "str_replace", "insert", "undo_edit"):
        return _oh_param(
            "command", command, "Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`."
        )
    if not path.startswith("/"):
        return _oh_param("path", path, "The path should be an absolute path.")
    k = s.ws.kind(path)
    if command == "create":
        if k == "file":
            return _oh_param(
                "path", path, f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        if k == "dir":
            return _oh_param(
                "path", path, f"The path {path} is a directory and only the `view` command can be used on directories."
            )
        if a.get("file_text") is None:
            return _oh_param("file_text", None, "Parameter `file_text` is required for command: create.")
        ops.write(s.ws, path, str(a["file_text"]))
        return f"File created successfully at: {path}"
    if k is None:
        return _oh_param("path", path, f"The path {path} does not exist. Please provide a valid path.")
    if k == "dir" and command != "view":
        return _oh_param(
            "path", path, f"The path {path} is a directory and only the `view` command can be used on directories."
        )
    if command == "view":
        if k == "dir":
            if a.get("view_range"):
                return _oh_param(
                    "view_range",
                    a["view_range"],
                    "The `view_range` parameter is not allowed when `path` points to a directory.",
                )
            entries, hidden = ops.list_dir(s.ws, path, 2)
            root = path.rstrip("/") or "/"
            listing = [root + "/"] + [posixpath.join(root, e) for e in entries]
            msg = (
                f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
                + "\n".join(listing)
            )
            if hidden:
                msg += (
                    f"\n\n{hidden} hidden files/directories in this directory are excluded. You can use 'ls -la "
                    f"{path}' to see them."
                )
            return ctx.middle_clip(msg, OH_EDITOR_MAX, OH_CLIP)
        lines = ops.lines_of(ops.read(s.ws, path))
        vr = a.get("view_range")
        if not vr:
            return _oh_cat_n(lines, 1, path)
        if not (isinstance(vr, list) and len(vr) == 2 and all(isinstance(x, int) for x in vr)):
            return _oh_param("view_range", vr, "It should be a list of two integers.")
        first, last = vr
        n = len(lines)
        if first < 1 or first > n:
            return _oh_param(
                "view_range",
                vr,
                f"Its first element `{first}` should be within the range of lines of the file: [1, {n}].",
            )
        if last != -1 and last < first:
            return _oh_param(
                "view_range",
                vr,
                f"Its second element `{last}` should be greater than or equal to the first element `{first}`.",
            )
        end = n if last == -1 else min(last, n)
        return _oh_cat_n(lines[first - 1 : end], first, path)
    if command == "str_replace":
        if a.get("old_str") is None:
            return _oh_param("old_str", None, "Parameter `old_str` is required for command: str_replace.")
        old, new = str(a["old_str"]), str(a.get("new_str") or "")
        try:
            e = ops.replace(s.ws, path, old, new)
        except OpError as err:
            if err.kind == "identical":
                return _oh_param(
                    "new_str", new, "No replacement was performed. `new_str` and `old_str` must be different."
                )
            if err.kind == "multiple":
                return (
                    OH_ERROR + f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines "
                    f"{err.data['lines']}. Please ensure it is unique."
                )
            return OH_ERROR + f"No replacement was performed, old_str `{old}` did not appear verbatim in {path}."
        snip, start = _oh_snippet(e.after, e.line, new.count("\n") + 1)
        return (
            f"The file {path} has been edited. "
            + _oh_cat_n(snip, start, f"a snippet of {path}")
            + "Review the changes and make sure they are as expected. Edit the file again if necessary."
        )
    if command == "insert":
        line = a.get("insert_line")
        if line is None:
            return _oh_param("insert_line", None, "Parameter `insert_line` is required for command: insert.")
        if a.get("new_str") is None:
            return _oh_param("new_str", None, "Parameter `new_str` is required for command: insert.")
        try:
            e = ops.insert(s.ws, path, int(line), str(a["new_str"]))
        except (OpError, ValueError, TypeError):
            n = len(ops.lines_of(ops.read(s.ws, path)))
            return _oh_param("insert_line", line, f"It should be within the range of allowed values: [0, {n}]")
        snip, start = _oh_snippet(e.after, e.line, str(a["new_str"]).count("\n") + 1)
        return (
            f"The file {path} has been edited. "
            + _oh_cat_n(snip, start, "a snippet of the edited file")
            + "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, "
            "etc). Edit the file again if necessary."
        )
    try:  # undo_edit
        e = ops.undo_edit(s.ws, path)
    except OpError:
        return OH_ERROR + f"No edit history found for {path}."
    return f"Last edit to {path} undone successfully. " + _oh_cat_n(ops.lines_of(e.after), 1, path)


def openhands_sdk() -> HarnessProfile:
    binds = {
        "terminal": _oh_terminal,
        "file_editor": _oh_file_editor,
        "finish": lambda s, a: str(a.get("message") or ""),
        "think": lambda s, a: "Your thought has been logged.",
    }
    return HarnessProfile(
        name="openhands-sdk",
        version="1.55.1",
        system=OPENHANDS_SYSTEM,
        tools=[ToolSpec(OPENHANDS_TOOLS[n], binds[n]) for n in ("terminal", "file_editor", "finish", "think")],
        stop=Finish("finish", "message", no_call_ends=True),
        answer=StopAnswer(LastText()),
        exec_timeout=OH_SOFT_TIMEOUT,
        context=Context("compaction", emulated=False, note="LLMSummarizingCondenser (when configured)"),
        alias=openhands_alias,
        unknown_tool=lambda name, names, args: f"Tool '{name}' not found. Available: {list(names)}",
        emulates="openhands-sdk / openhands-tools 1.55.1 as Harbor 0.24's openhands_sdk_runner builds the agent "
        "(terminal, file_editor, task_tracker left out, finish, think; security analyzer on); schemas from "
        "Tool.to_openai_tool(add_security_risk_prediction=True)",
    )


for _f, _n, _v in (
    (palingenesis_default, "palingenesis-default", "1"),
    (palingenesis_default_2, "palingenesis-default", "2"),
    (mini_swe_agent, "mini-swe-agent", "2.4.6"),
    (mini_swe_agent_text, "mini-swe-agent-text", "2.4.6"),
    (opencode, "opencode", "1.19.2"),
    (pi, "pi", "1.3.0"),
    (openhands_sdk, "openhands-sdk", "1.55.1"),
):
    register(_n, _v, _f)
register("palingenesis-default", "3", palingenesis_default_3, default=False)  # the bare name stays @2

from palingenesis.rl.envs.harness import profiles_more, terminus  # noqa: E402,F401 — phase 2 profiles
