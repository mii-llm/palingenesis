"""Deliberate learning signals: what may become a note, and what never does.

A note comes only from a deliberate act:

    explicit command   the newest message is a user message that starts with a command pattern
                       ("remember ...", "ricorda che ...", configurable), up to signals.max_note_chars
    correction         an explicit call with the prompt and the corrected answer (a UI button)
    note / document    added or approved by the user (user scope) or an admin (team / org scope)

Never a signal (prompt-injection and poisoning defence): assistant, system and tool messages, tool
results inside a user message, earlier turns, text inside code fences, blockquotes or quotes, anything
after the command's first line (pasted content: an email pasted below a command has no blank line before
it), and traffic in general. A remembered note that
reads like an instruction ("always", "ignore", "you must") is held for the user's confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Command words in several languages, optionally followed by "that" / "che" / "que" / ":".
_LEAD = r"(?:please\s+|per\s+favore\s+|por\s+favor\s+|s'il\s+te\s+plaît\s+|bitte\s+)?"
# The body starts on the command's own line ([^\S\n]: whitespace other than a line break).
_THAT = r"(?:[^\S\n]*[:,-][^\S\n]*|[^\S\n]+)(?:(?:that|che|que|qu'|dass)[^\S\n]+)?"
_INSTRUCTION = re.compile(
    r"\b(?:always|never|ignore|disregard|you must|from now on|override|"
    r"sempre|mai|ignora|devi|d'ora in poi|"
    r"siempre|nunca|debes|"
    r"toujours|jamais|tu dois|"
    r"immer|niemals|ignoriere|du musst)\b",
    re.I,
)
_QUOTE_OPENERS = ('"', "'", "“", "‘", "«", "`", ">")


@dataclass
class Signal:
    kind: str  # remembered | correction
    text: str
    needs_confirmation: bool = False
    prompt: str = ""


def message_text(message: dict) -> str:
    """The user's own text of a message: string content, or its text parts (tool results and other
    non-text parts are dropped)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return ""


def _command_re(patterns: list[str]) -> re.Pattern:
    words = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in sorted(patterns, key=len, reverse=True))
    return re.compile(rf"^{_LEAD}(?:{words})\b{_THAT}(?P<body>.+)", re.I | re.S)


def detect_command(messages: list[dict], patterns: list[str], max_chars: int = 1000) -> Signal | None:
    """The note an explicit command in the newest message asks to remember, or None.

    Only the newest message is examined, and only if it is a user message: a conversation ending with
    an assistant or tool turn carries no new command. The command must open the message; its body ends
    at the first line break or code fence (what follows is pasted content).
    """
    if not messages or messages[-1].get("role") != "user":
        return None
    text = message_text(messages[-1]).strip()
    if not text or text.startswith(_QUOTE_OPENERS):
        return None
    m = _command_re(patterns).match(text)
    if m is None:
        return None
    body = m.group("body")
    body = re.split(r"\n|```", body, maxsplit=1)[0].strip()
    if not body or len(body) > max_chars:
        return None
    return Signal("remembered", body, needs_confirmation=bool(_INSTRUCTION.search(body)))


def correction_signal(prompt: str, correction: str, max_chars: int = 1000) -> Signal:
    """A correction: rendered to the teacher as a plain statement; its prompt becomes a train question."""
    correction = correction.strip()
    if not correction:
        raise ValueError("a correction needs the corrected answer")
    if len(correction) > max_chars:
        raise ValueError(f"a correction is at most {max_chars} characters (signals.max_note_chars)")
    return Signal("correction", correction, prompt=prompt.strip())


def looks_like_instruction(text: str) -> bool:
    return bool(_INSTRUCTION.search(text))
