"""Teacher views: context distillation, where the teacher reads a conversation the student does not.

The student trains on the plain rendering of a row; the teacher scores the same answer given more
context: a system prompt, documents or a hint (`privileged_field`), few-shot examples
(`prefix_messages`), its own template options, its own copy of the conversation, or anything a
`view_fn` builds. Both renderings are masked by the same rules, so their trained tokens are the same
answer tokens; `align` checks that they are (token for token, up to the shared-vocabulary mapping)
and pairs each student position with the teacher position that predicts the same token. A row whose
answers differ in the two renderings is dropped and counted (`kd_view_mismatch`).

Context distillation: Askell et al., arXiv:2112.00861; Snell et al., arXiv:2209.15189.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from palingenesis.distill.config import TeacherSpec

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def input_field(name: str) -> str:
    return f"teacher_input_ids.{name}"


def positions_field(name: str) -> str:
    return f"teacher_positions.{name}"


def _load_messages(spec) -> list[dict]:
    if isinstance(spec, list):
        return [dict(m) for m in spec]
    path = Path(spec)
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        out = []
        for r in rows:
            out.extend(r["messages"] if isinstance(r, dict) and "messages" in r else [r])
        return out
    data = json.loads(text)
    return data["messages"] if isinstance(data, dict) else data


class TeacherView:
    """Builds the teacher's messages for a row, then aligns its rendering with the student's."""

    def __init__(self, spec: TeacherSpec, bridge=None):
        self.spec = spec
        self.name = spec.name
        self.prefix = _load_messages(spec.prefix_messages) if spec.prefix_messages else []
        self.view_fn = None
        if spec.view_fn:
            from palingenesis.hooks import load_object

            self.view_fn = load_object(spec.view_fn)
        # student id -> teacher id for the trained tokens (shared vocabulary with an end-of-turn swap)
        self.swap = dict(getattr(bridge, "swap", {}) or {})

    def describe(self) -> dict:
        """What defines the view (recorded in a cache's metadata)."""
        s = self.spec
        return {
            "system_prompt": s.system_prompt,
            "prefix_messages": self.prefix,
            "privileged_field": s.privileged_field,
            "privileged_template": s.privileged_template if s.privileged_field else "",
            "privileged_role": s.privileged_role if s.privileged_field else "",
            "messages_field": s.messages_field,
            "chat_template_kwargs": s.chat_template_kwargs,
            "view_fn": s.view_fn,
        }

    def messages(self, row: dict, messages: list[dict]) -> list[dict] | None:
        """The teacher's conversation for `row` (whose student conversation is `messages`)."""
        s = self.spec
        if s.messages_field:
            own = row.get(s.messages_field)
            if isinstance(own, str):
                own = json.loads(own)
            out = [dict(m) for m in own] if own else [dict(m) for m in messages]
        else:
            out = [dict(m) for m in messages]
        if s.privileged_field and row.get(s.privileged_field) not in (None, ""):
            from palingenesis.opd.formatting import with_privileged

            out = with_privileged(out, str(row[s.privileged_field]), s.privileged_template, s.privileged_role)
        if s.system_prompt:
            if out and out[0].get("role") == "system":
                out[0] = {**out[0], "content": s.system_prompt}
            else:
                out.insert(0, {"role": "system", "content": s.system_prompt})
        if self.prefix:  # few-shot examples are context: their answers are never trained
            at = 1 if out and out[0].get("role") == "system" else 0
            out[at:at] = [{**m, "loss": False} if m.get("role") == "assistant" else dict(m) for m in self.prefix]
        if self.view_fn is not None:
            out = self.view_fn(row, out)
        return out

    def align(self, student: dict, teacher: dict) -> torch.Tensor | None:
        """teacher_positions [T_student]: for each student position t whose next token is trained, the
        teacher position predicting the same token (-1 elsewhere); None if the answers differ.

        The teacher may train more tokens than the student (the student's copy was truncated to fit
        its length budget): the student's trained tokens must then be the teacher's first ones."""
        s_lab, t_lab = student["labels"], teacher["labels"]
        s_pos = torch.nonzero(s_lab != IGNORE_INDEX).flatten()
        t_pos = torch.nonzero(t_lab != IGNORE_INDEX).flatten()
        if s_pos.numel() == 0 or s_pos.numel() > t_pos.numel():
            return None
        t_pos = t_pos[: s_pos.numel()]
        s_tok = student["input_ids"][s_pos]
        if self.swap:
            s_tok = torch.tensor([self.swap.get(int(t), int(t)) for t in s_tok], dtype=torch.long)
        if not torch.equal(s_tok, teacher["input_ids"][t_pos]):
            return None
        keep = (s_pos > 0) & (t_pos > 0)
        out = torch.full_like(s_lab, -1)
        out[s_pos[keep] - 1] = t_pos[keep] - 1
        return out
