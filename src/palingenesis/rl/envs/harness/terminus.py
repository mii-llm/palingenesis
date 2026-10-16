"""terminus-2 (Harbor 0.24.0's built-in agent): no tools, no system message; each response is a JSON object
{"analysis", "plan", "commands": [{"keystrokes", "duration"}], "task_complete"} whose keystrokes go verbatim to a
persistent tmux pane; the next user turn is the terminal output (new output, else the visible screen) capped at
10,000 bytes; completion needs "task_complete": true twice in a row (the first gets a confirmation question).

The response parser follows terminus_json_plain_parser.py (Harbor, Apache-2.0), including its tolerance for small-
model JSON mistakes: extra text around the object is a warning, a missing closing brace or an object buried in prose
is auto-corrected with a warning, anything else is a parsing error that executes nothing and is answered with
terminus's own feedback ("Previous response had parsing errors: ..."). The task image needs tmux (harbor3-base:3:
research/tiny-reliable-agent/harness/Dockerfile.tmux; HarborEnv(images={"harbor3-base:1": "harbor3-base:3"})).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from palingenesis.rl.envs.harness import ops
from palingenesis.rl.envs.harness.context import Context
from palingenesis.rl.envs.harness.ops import OpError
from palingenesis.rl.envs.harness.profile import HarnessProfile, ToolSpec, fn_schema, register
from palingenesis.rl.envs.harness.protocols import FormatError, Protocol
from palingenesis.rl.envs.harness.rules import StopAnswer, StopRule

ACTION = "__terminus__"
MAX_BYTES = 10000

TERMINUS_USER = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task "
    "description and the output from previously executed commands. Your goal is to solve the task by providing "
    "batches of shell commands.\n\n"
    "Format your response as JSON with the following structure:\n\n"
    "{\n"
    '  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been '
    'accomplished? What still needs to be done?",\n'
    '  "plan": "Describe your plan for the next steps. What commands will you run and why?",\n'
    '  "commands": [\n'
    '    {\n      "keystrokes": "ls -la\\n",\n      "duration": 0.1\n    },\n'
    '    {\n      "keystrokes": "cd project\\n",\n      "duration": 0.1\n    }\n'
    "  ],\n"
    '  "task_complete": true\n'
    "}\n\n"
    'Required fields: "analysis", "plan", "commands" (an array of command objects). Optional: "task_complete" '
    "(boolean, default false).\n\n"
    'Each command object has "keystrokes" (required: the exact keystrokes sent to the terminal) and "duration" '
    "(seconds to wait before the next command, default 1.0).\n\n"
    'IMPORTANT: "keystrokes" are sent verbatim. End every command with a newline (\\n) or it will not execute. For '
    "special keys use tmux-style sequences: C-c for Ctrl+C, C-d for Ctrl+D.\n\n"
    "Use 0.1 seconds for immediate commands (cd, ls, echo, cat), 1.0 for commands such as gcc or find, and more for "
    "slow ones (make, long scripts). Prefer a short duration: you can always wait again with "
    '{"keystrokes": "", "duration": 10.0}. Never wait longer than 60 seconds.\n\n'
    "Notes: do not add whitespace around keystrokes unless it is part of the command; extra text before or after the "
    "JSON generates warnings but is tolerated; the JSON must be valid (escape quotes and special characters); the "
    "commands array may be empty to wait without acting.\n\n"
    "Task Description:\n{instruction}\n\n"
    "Current terminal state:\n{terminal_state}")


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    commands: list[ParsedCommand] = field(default_factory=list)
    is_task_complete: bool = False
    error: str = ""
    warning: str = ""


def _join(warnings: list[str]) -> str:
    return "- " + "\n- ".join(warnings) if warnings else ""


def _extract_json(response: str) -> tuple[str, list[str]]:
    start = end = -1
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(response):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    end = i + 1
                    break
    if start == -1 or end == -1:
        return "", ["No valid JSON object found"]
    warnings = []
    if response[:start].strip():
        warnings.append("Extra text detected before JSON object")
    if response[end:].strip():
        warnings.append("Extra text detected after JSON object")
    return response[start:end], warnings


def _try_parse(response: str) -> ParseResult:
    content, warnings = _extract_json(response)
    if not content:
        return ParseResult([], False, "No valid JSON found in response", _join(warnings))
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {e}"
        msg += f" | Content: {content!r}" if len(content) < 200 else f" | Content preview: {content[:100]!r}..."
        return ParseResult([], False, msg, _join(warnings))
    if not isinstance(data, dict):
        return ParseResult([], False, "Response must be a JSON object", _join(warnings))
    missing = [f for f in ("analysis", "plan", "commands") if f not in data]
    if missing:
        return ParseResult([], False, f"Missing required fields: {', '.join(missing)}", _join(warnings))
    if not isinstance(data.get("analysis", ""), str):
        warnings.append("Field 'analysis' should be a string")
    if not isinstance(data.get("plan", ""), str):
        warnings.append("Field 'plan' should be a string")
    commands = data.get("commands", [])
    if not isinstance(commands, list):
        return ParseResult([], False, "Field 'commands' must be an array", _join(warnings))
    positions = {f: m.start() for f in ("analysis", "plan", "commands") if (m := re.search(f'"({f})"\\s*:', content))}
    if len(positions) >= 2:
        expected = [f for f in ("analysis", "plan", "commands") if f in positions]
        actual = sorted(positions, key=positions.get)
        if actual != expected:
            warnings.append(f"Fields appear in wrong order. Found: {' → '.join(actual)}, expected: "
                            f"{' → '.join(expected)}")
    tc = data.get("task_complete")
    if tc is not None and not isinstance(tc, (bool, str)):
        warnings.append("Field 'task_complete' should be a boolean or string")
    complete = data.get("task_complete", False)
    if isinstance(complete, str):
        complete = complete.lower() in ("true", "1", "yes")
    parsed, error = [], ""
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            error = f"Command {i + 1} must be an object"
            break
        if "keystrokes" not in cmd:
            error = f"Command {i + 1} missing required 'keystrokes' field"
            break
        keys = cmd["keystrokes"]
        if not isinstance(keys, str):
            error = f"Command {i + 1} 'keystrokes' must be a string"
            break
        if "duration" in cmd:
            duration = cmd["duration"]
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                warnings.append(f"Command {i + 1}: Invalid duration value, using default 1.0")
                duration = 1.0
        else:
            warnings.append(f"Command {i + 1}: Missing duration field, using default 1.0")
            duration = 1.0
        unknown = set(cmd) - {"keystrokes", "duration"}
        if unknown:
            warnings.append(f"Command {i + 1}: Unknown fields: {', '.join(sorted(unknown))}")
        if i < len(commands) - 1 and keys and not keys.endswith("\n"):
            warnings.append(f"Command {i + 1} should end with newline when followed by another command. Otherwise "
                            "the two commands will be concatenated together on the same line.")
        parsed.append(ParsedCommand(keys, float(duration)))
    if error:
        if complete:  # with task_complete, command errors are only warnings
            return ParseResult([], True, "", _join(warnings + [error]))
        return ParseResult([], False, error, _join(warnings))
    return ParseResult(parsed, bool(complete), "", _join(warnings))


def parse_response(response: str) -> ParseResult:
    """terminus-2's JSON parser with its two auto-fixes (a missing closing brace; an object buried in prose)."""
    result = _try_parse(response)
    if not result.error:
        return result
    fixes = []
    if any(k in result.error for k in ("Invalid JSON", "Expecting", "Unterminated", "No valid JSON found")):
        n = response.count("{") - response.count("}")
        if n > 0:
            fixes.append(("Fixed incomplete JSON by adding missing closing brace", response + "}" * n))
    for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL):
        try:
            json.loads(match)
        except json.JSONDecodeError:
            continue
        fixes.append(("Extracted JSON from mixed content", match))
        break
    for name, fixed in fixes:
        corrected = _try_parse(fixed)
        if not corrected.error:
            note = f"AUTO-CORRECTED: {name} - please fix this in future responses"
            corrected.warning = f"- {note}\n{corrected.warning}" if corrected.warning else f"- {note}"
            return corrected
    return result


def limit_output(output: str, max_bytes: int = MAX_BYTES) -> str:
    data = output.encode()
    if len(data) <= max_bytes:
        return output
    half = max_bytes // 2
    first = data[:half].decode(errors="ignore")
    last = data[-half:].decode(errors="ignore")
    omitted = len(data) - len(first.encode()) - len(last.encode())
    return f"{first}\n[... output limited to {max_bytes} bytes; {omitted} interior bytes omitted ...]\n{last}"


class TerminusJSON(Protocol):
    """The whole response is one action batch (ACTION); a parsing error is answered with terminus's feedback."""

    name = "terminus-json"

    def parse(self, content):
        result = parse_response(content or "")
        if result.error:
            feedback = f"ERROR: {result.error}" + (f"\nWARNINGS: {result.warning}" if result.warning else "")
            raise FormatError(f"Previous response had parsing errors:\n{feedback}\n\nPlease fix these issues and "
                              "provide a proper JSON response.")
        return [(ACTION, {"commands": [{"keystrokes": c.keystrokes, "duration": c.duration} for c in result.commands],
                          "task_complete": result.is_task_complete, "warning": result.warning})]

    def render(self, calls):
        name, args = calls[0]
        return json.dumps({"analysis": args.get("analysis", ""), "plan": args.get("plan", ""),
                           "commands": args.get("commands", []), "task_complete": bool(args.get("task_complete"))},
                          indent=2)

    def observation(self, results):
        return "\n\n".join(out for _, out in results)


class TaskComplete(StopRule):
    """task_complete twice in a row (the first is answered with terminus's confirmation question)."""

    requires_action = True  # a response without a JSON object is a parsing error, not the end


def _terminus_batch(s, a):
    tmux: ops.Tmux = s.flags["tmux"]
    for c in a.get("commands") or []:
        ops.keys(s.ws, tmux, str(c.get("keystrokes", "")), min(float(c.get("duration", 1.0)), 60))
    terminal = limit_output(tmux.incremental())
    pending = s.flags.get("pending_completion", False)
    if a.get("task_complete"):
        if pending:
            s.finish(None)
            return terminal
        s.flags["pending_completion"] = True
        return (f"Current terminal state:\n{terminal}\n\nAre you sure you want to mark the task as complete? This will "
                "trigger your solution to be graded and you won't be able to make any further corrections. If so, "
                'include "task_complete": true in your JSON response again.')
    s.flags["pending_completion"] = False
    if a.get("warning"):
        return f"Previous response had warnings:\nWARNINGS: {a['warning']}\n\n{terminal}"
    return terminal


def _terminus_start(s) -> dict[str, str]:
    tmux = ops.Tmux(s.ws, width=160, height=40)
    try:
        tmux.start()
    except OpError as e:
        raise RuntimeError("terminus-2 needs tmux in the task image (build harbor3-base:3 from "
                           "research/tiny-reliable-agent/harness/Dockerfile.tmux and pass HarborEnv(images="
                           "{'harbor3-base:1': 'harbor3-base:3'})): " + str(e.data.get("detail", ""))) from None
    tmux.wait(0.5)
    s.flags["tmux"] = tmux
    return {"terminal_state": limit_output(tmux.incremental())}


def terminus_format_error(reason: str, n: int) -> str:
    return reason  # TerminusJSON words the whole feedback


def terminus_2() -> HarnessProfile:
    return HarnessProfile(
        name="terminus-2", version="harbor-0.24.0", system=None, user=TERMINUS_USER,
        tools=[ToolSpec(fn_schema(ACTION, "a terminus-2 response (internal)", {}, []), _terminus_batch)],
        protocol=TerminusJSON(), stop=TaskComplete(), answer=StopAnswer(None), start=_terminus_start,
        format_error=terminus_format_error,
        context=Context("compaction", emulated=False, note="unwind + 3-subagent summarisation near the limit"),
        emulates="Harbor 0.24.0 agents/terminus_2 (terminus-json-plain.txt, terminus_json_plain_parser.py, "
                 "tmux_session.py; parser json, pane 160x40, 10 KB output cap, task_complete twice); summarisation "
                 "off as for RL capture")


register("terminus-2", "harbor-0.24.0", terminus_2)
