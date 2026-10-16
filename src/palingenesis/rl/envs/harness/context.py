"""Truncation and context policies: how a harness shortens a long tool output (and words it), and what it does when
the conversation outgrows the context.

Truncation is applied by each profile's tool bindings with the harness's own wording; the trainer's
env.max_tool_output_tokens still caps every observation afterwards (set it above the profiles' own limits so the
harness wording is what the policy sees).

    head_tail_chars  mini-swe-agent: first / last N characters and the elided count
    head_lines       pi / opencode read: the first lines within a line and byte budget
    tail_lines       pi / opencode bash: the last lines within a line and byte budget
    middle_clip      OpenHands: head and tail characters around a notice

Context: palingenesis rollouts are token-exact, so history is never rewritten; a harness that compacts (opencode,
pi, OpenHands condenser, terminus summarisation) is emulated up to its compaction point only. Context records the
harness's policy so evaluation can say which episodes would have been compacted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Truncated:
    text: str
    truncated: bool = False
    total_lines: int = 0
    first_line: int = 1  # 1-based line numbers of the kept span
    last_line: int = 0
    by: str = ""  # "lines" or "bytes"


def _lines(text: str) -> list[str]:
    return text.split("\n")


def head_lines(text: str, max_lines: int = 2000, max_bytes: int = 50 * 1024) -> Truncated:
    """The first lines of `text` within max_lines and max_bytes (whole lines only)."""
    lines = _lines(text)
    kept, size, by = [], 0, ""
    for line in lines:
        if len(kept) >= max_lines:
            by = "lines"
            break
        n = len(line.encode()) + (1 if kept else 0)
        if size + n > max_bytes:
            by = "bytes"
            break
        kept.append(line)
        size += n
    return Truncated("\n".join(kept), bool(by), len(lines), 1, len(kept), by)


def tail_lines(text: str, max_lines: int = 2000, max_bytes: int = 50 * 1024) -> Truncated:
    """The last lines of `text` within max_lines and max_bytes (whole lines only)."""
    lines = _lines(text)
    kept, size, by = [], 0, ""
    for line in reversed(lines):
        if len(kept) >= max_lines:
            by = "lines"
            break
        n = len(line.encode()) + (1 if kept else 0)
        if size + n > max_bytes:
            by = "bytes"
            break
        kept.append(line)
        size += n
    kept.reverse()
    return Truncated("\n".join(kept), bool(by), len(lines), len(lines) - len(kept) + 1, len(lines), by)


def head_tail_chars(text: str, limit: int = 10000, head: int = 5000, tail: int = 5000) -> tuple[str, str, int] | None:
    """(head, tail, elided characters) when len(text) >= limit, else None."""
    if len(text) < limit:
        return None
    return text[:head], text[-tail:] if tail else "", len(text) - limit


def middle_clip(text: str, limit: int, notice: str) -> str:
    """Head and tail of `text` around `notice`, the whole within `limit` characters (OpenHands maybe_truncate)."""
    if not limit or len(text) <= limit:
        return text
    if len(notice) >= limit:
        return notice[:limit]
    available = limit - len(notice)
    head = available // 2 + available % 2
    tail = available - head
    return text[:head] + notice + (text[-tail:] if tail else "")


def fmt_size(n: int) -> str:
    """pi's formatSize: bytes, KB, MB."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


@dataclass
class Context:
    """A harness's context policy. strategy "none" (mini-swe-agent: the history only grows) or "compaction" (a
    summary replaces the history near the limit: opencode, pi, OpenHands' condenser, terminus-2); emulated=False
    means palingenesis does not reproduce it (rollouts are token-exact) and the episode ends at the context limit."""

    strategy: str = "none"
    limit_tokens: int | None = None
    emulated: bool = True
    note: str = ""
