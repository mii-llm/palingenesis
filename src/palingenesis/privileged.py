"""Privileged context: the teacher's copy of a conversation with text the student does not see (a hint, a fact card,
a restated constraint, the question read twice, a rewording), for on-policy self-distillation with privileged
information (OPSD; `pgs distill`, sources.<name>.privileged_*) and context distillation (`pgs train` with distill
teachers' privileged_field). Standard library only: the configs validate templates with it.
"""

import string
from typing import Any

PRIVILEGED_PLACEHOLDERS = ("content", "privileged", "question", "system")


def privileged_template_error(template: Any, with_field: bool) -> str | None:
    """Why `template` is not a valid privileged-context template, or None.

    Placeholders: {content} (the turn being rewritten), {privileged} (the row's privileged text), {question} (the
    last user turn's text) and {system} (the system turn's text, "" without one); no other name, index, conversion
    or format spec ({{ and }} are literal braces). {privileged} must appear exactly when the view reads a row field
    (`with_field`), and the template must change the turn (not just "{content}")."""
    if not isinstance(template, str) or not template:
        return f"must be a non-empty string, got {template!r}"
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        return f"is not a valid format string ({e}): {template!r}"
    names = set()
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if name not in PRIVILEGED_PLACEHOLDERS or spec or conversion:
            allowed = ", ".join("{" + p + "}" for p in PRIVILEGED_PLACEHOLDERS)
            return (
                f"may only use the placeholders {allowed} (no format spec or conversion; write {{{{ and }}}} for "
                f"literal braces), got a field {name!r} in {template!r}"
            )
        names.add(name)
    if with_field and "privileged" not in names:
        return f"reads a row field, so it must place it with {{privileged}}: {template!r}"
    if not with_field and "privileged" in names:
        return f"uses {{privileged}} but names no field to fill it from: {template!r}"
    if template.strip() == "{content}":
        return "leaves the turn unchanged (the teacher would see what the student sees)"
    return None


def _text(content: Any) -> str:
    """A message's text: a string as is, the text parts of a content list joined, None as ""."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return str(content)


def with_privileged(messages: list[dict], privileged: str, template: str, role: str = "user") -> list[dict]:
    """The teacher's copy of `messages` with privileged context added by `template`.

    Placeholders (privileged_template_error): {content} is the turn the template rewrites, {privileged} the row's
    privileged text ("" for a view without a field), {question} the last user turn's text, {system} the system turn's
    text ("" without one). E.g. "{content}\\n\\n{privileged}" (a hint after the question), "{content}\\n\\nRead the
    question again: {question}" (RE2), "{privileged}" (the turn replaced by a rewording), "{system}\\n\\n{content}"
    (the system facts repeated next to the question).

    role "user": the last user turn is rewritten. The last user turn, not the last message: in a mid-episode
    state the conversation ends with a tool result, which must stay what the tool returned.
    role "system": the system turn is rewritten ({content} = its text, "" without one, then a system turn is
    inserted first). A note there reads as an instruction to the assistant, not as something the user said."""
    last = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
    has_system = bool(messages) and messages[0]["role"] == "system"
    values = {
        "privileged": privileged,
        "question": _text(messages[last]["content"]) if last is not None else "",
        "system": _text(messages[0]["content"]) if has_system else "",
    }
    if role == "system":
        out = list(messages)
        if has_system:
            out[0] = {**out[0], "content": template.format(content=_text(out[0]["content"]), **values)}
        else:
            out.insert(0, {"role": "system", "content": template.format(content="", **values).strip()})
        return out
    if last is None:
        raise ValueError("privileged context needs a user turn to attach to")
    out = list(messages)
    out[last] = {**messages[last], "content": template.format(content=_text(messages[last]["content"]), **values)}
    return out
