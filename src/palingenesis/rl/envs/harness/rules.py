"""Stop rules (when a harness ends the episode) and answer rules (what it takes as the agent's final message, given
to the verifier as /logs/agent/final_message.md).

Stop rules see every executed action (on_tool) and say whether a turn without an action ends the episode
(requires_action False) or is a format error the harness answers (True: mini-swe-agent).

    NoCall()                  a turn without a tool call ends the episode (opencode, pi, Claude Code, palingenesis)
    Sentinel(text)            a shell command whose output's first line is `text` (exit 0) submits; the rest of the
                              output is the submission (mini-swe-agent: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT)
    Submit(tool, confirm)     a submit tool ends it; confirm=True asks once and ends on the next consecutive submit
                              (SWE-agent's review step)
    Finish(tool, field)       a finish tool ends it with args[field] as the answer (OpenHands finish(message)); with
                              no_call_ends a turn without a tool call ends it too
    Flag(confirm)             a completion flag in a text action (terminus-2 "task_complete": true, twice in a row);
                              text protocols that carry a flag emit the action FLAG

Answer rules:
    LastText()                the last assistant message's text (today's HarborEnv behaviour)
    StopAnswer(fallback)      what the stop rule captured (finish message, sentinel submission), else the fallback
                              (LastText() or None: an empty answer)
"""

from __future__ import annotations

from dataclasses import dataclass

FLAG = "__flag__"  # a text protocol's completion flag, run as an action


class StopRule:
    requires_action = False

    def on_tool(self, session, name: str, args: dict, output: str) -> str:
        return output


@dataclass
class NoCall(StopRule):
    requires_action: bool = False


@dataclass
class Sentinel(StopRule):
    text: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    tool: str = "bash"
    requires_action: bool = True

    def on_tool(self, session, name, args, output):
        r = session.last_exec
        if name == self.tool and r is not None and r.exit_code == 0:
            lines = r.output.lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() == self.text:
                session.finish("".join(lines[1:]))
        return output


@dataclass
class Submit(StopRule):
    tool: str = "submit"
    confirm: bool = False
    confirm_message: str = "Are you sure you want to submit? Call submit again to confirm."
    requires_action: bool = False

    def on_tool(self, session, name, args, output):
        if name != self.tool:
            session.flags.pop("submit_armed", None)
            return output
        if self.confirm and not session.flags.get("submit_armed"):
            session.flags["submit_armed"] = True
            return self.confirm_message
        session.finish(None)
        return output


@dataclass
class Finish(StopRule):
    tool: str = "finish"
    field: str = "message"
    no_call_ends: bool = True

    @property
    def requires_action(self):  # noqa: D401 — a turn without a call: the end (no_call_ends) or a format error
        return not self.no_call_ends

    def on_tool(self, session, name, args, output):
        if name == self.tool:
            session.finish(str(args.get(self.field) or ""))
        return output


@dataclass
class Flag(StopRule):
    confirm: bool = True
    confirm_message: str = ("Are you sure you want to mark the task as complete? If so, set the completion flag "
                            "again in your next response.")
    requires_action: bool = True

    def on_tool(self, session, name, args, output):
        if name != FLAG:
            session.flags.pop("flag_armed", None)
            return output
        if self.confirm and not session.flags.get("flag_armed"):
            session.flags["flag_armed"] = True
            return self.confirm_message
        session.finish(None)
        return output


def last_text(messages: list[dict] | None) -> str | None:
    """The last assistant message's content (None without a conversation) — HarborEnv's final message."""
    return next((m.get("content") or "" for m in reversed(messages or []) if m.get("role") == "assistant"), None)


class AnswerRule:
    def __call__(self, session, messages: list[dict] | None) -> str | None:
        raise NotImplementedError


@dataclass
class LastText(AnswerRule):
    def __call__(self, session, messages):
        return last_text(messages)


@dataclass
class StopAnswer(AnswerRule):
    fallback: AnswerRule | None = None

    def __call__(self, session, messages):
        if session.answer is not None:
            return session.answer
        if self.fallback is not None:
            return self.fallback(session, messages)
        return "" if messages is not None else None
