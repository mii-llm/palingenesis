"""Privileged context for on-policy self-distillation (OPSD): templates and their placeholders, several kinds of
privileged context in one source (privileged_views), per-row templates, and validation."""

import json
import random
from collections import Counter

import pytest

from palingenesis.privileged import privileged_template_error, with_privileged

CHAT = [
    {"role": "system", "content": "Today is Monday. The store closes at 18:00."},
    {"role": "user", "content": "Can I still buy milk at 17:30?"},
]


def test_placeholders_re2_rewording_and_system_facts():
    re2 = with_privileged(CHAT, "", "{content}\n\nRead the question again: {question}")
    assert (
        re2[1]["content"] == "Can I still buy milk at 17:30?\n\nRead the question again: Can I still buy milk at 17:30?"
    )
    assert re2[0] == CHAT[0] and CHAT[1]["content"] == "Can I still buy milk at 17:30?"  # the student's copy
    reworded = with_privileged(CHAT, "Is the store open at 17:30 for milk?", "{privileged}")
    assert reworded[1]["content"] == "Is the store open at 17:30 for milk?"
    facts = with_privileged(CHAT, "", "{system}\n\n{content}")
    assert facts[1]["content"].startswith("Today is Monday.") and facts[1]["content"].endswith("17:30?")
    card = with_privileged(CHAT, "closing: 18:00", "{content}\n\nFact card:\n{privileged}", role="system")
    assert card[0]["content"] == "Today is Monday. The store closes at 18:00.\n\nFact card:\nclosing: 18:00"
    # no system turn: {system} is empty; a system view creates the turn
    plain = [CHAT[1]]
    assert with_privileged(plain, "", "{system}{content}")[0]["content"] == CHAT[1]["content"]
    assert with_privileged(plain, "", "Question: {question}", role="system")[0] == {
        "role": "system",
        "content": "Question: Can I still buy milk at 17:30?",
    }
    # list content (text parts) reads as its text; literal braces stay
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert with_privileged(parts, "", "{{x}} {content}")[0]["content"] == "{x} ab"


@pytest.mark.parametrize(
    "template, with_field, problem",
    [
        ("{content}\n\n{privileged}", True, None),
        ("{privileged}", True, None),
        ("{content}\n\nAgain: {question}", False, None),
        ("{{json}} {content} {system}", False, None),
        ("{content}", False, "unchanged"),
        ("{content} {privileged}", False, "names no field"),
        ("{content} again", True, "must place it"),
        ("{content} {answer}", True, "placeholders"),
        ("{content!r} {privileged}", True, "placeholders"),
        ("{content:>10} {privileged}", True, "placeholders"),
        ("{0} {privileged}", True, "placeholders"),
        ("{content", True, "format string"),
        ("", True, "non-empty"),
    ],
)
def test_template_validation(template, with_field, problem):
    error = privileged_template_error(template, with_field)
    assert (error is None) if problem is None else (problem in error)


def _rows(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _source(tmp_path, rows, **options):
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import ChatMessagesSource

    config = SourceConfig(path=_rows(tmp_path, rows), dev_size=0, max_new_tokens=16, **options)
    return ChatMessagesSource(config, eval_samples=2, seed=0, rng=random.Random(0))


def test_views_draw_by_weight_apply_by_field_and_fix_dev_rows(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "q0"}], "constraint": "answer in one word"},
        {"messages": [{"role": "user", "content": "q1"}], "constraint": "", "answer": 0},
        {"messages": [{"role": "user", "content": "q2"}]},
    ]
    views = [
        {"name": "re2", "template": "{content}\nAgain: {question}", "weight": 1},
        {"name": "constraint", "template": "{content}\nRemember: {privileged}", "field": "constraint", "weight": 3},
    ]
    source = _source(tmp_path, rows, privileged_views=views)
    rng = random.Random(1)
    drawn = Counter(source.meta(rows[0], rng)["_view"] for _ in range(2000))
    assert set(drawn) == {"re2", "constraint"} and 0.70 < drawn["constraint"] / 2000 < 0.80
    meta = source.meta(rows[0], random.Random(5))
    expected = {"re2": "q0\nAgain: q0", "constraint": "q0\nRemember: answer in one word"}[meta["_view"]]
    assert meta["teacher_messages"] == [{"role": "user", "content": expected}]
    # an empty field: only the field-less view applies
    assert {source.meta(rows[1], rng)["_view"] for _ in range(50)} == {"re2"}
    # without an rng (held-out rows): one fixed view per row
    assert len({source.meta(rows[0])["_view"] for _ in range(20)}) == 1
    # batch stats: the share of each view
    stats = source.batch_stats([(source.meta(r, rng), "") for r in rows])
    assert sum(stats.values()) == pytest.approx(1.0) and all(k.startswith("privileged/") for k in stats)


def test_field_value_zero_counts_and_rows_without_views_are_plain(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "2-2?"}], "answer": 0},
        {"messages": [{"role": "user", "content": "free"}], "answer": " "},
    ]
    source = _source(tmp_path, rows, privileged_field="answer", privileged_template="{content}\nAnswer: {privileged}")
    assert source.meta(rows[0])["teacher_messages"][0]["content"] == "2-2?\nAnswer: 0"
    assert "teacher_messages" not in source.meta(rows[1])  # whitespace only is empty
    assert source.batch_stats([(source.meta(r), "") for r in rows]) == {
        "privileged/none": 0.5,
        "privileged/privileged": 0.5,
    }


def test_per_row_templates(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "q"}], "hint": "h", "tpl": "Hint first: {privileged}\n{content}"},
        {"messages": [{"role": "user", "content": "r"}], "hint": "k"},
    ]
    source = _source(tmp_path, rows, privileged_field="hint", privileged_template_field="tpl")
    assert source.meta(rows[0])["teacher_messages"][0]["content"] == "Hint first: h\nq"
    assert source.meta(rows[1])["teacher_messages"][0]["content"] == "r\n\nk"  # the source's template
    bad = rows + [{"messages": [{"role": "user", "content": "s"}], "hint": "x", "tpl": "{content} {oops}"}]
    with pytest.raises(ValueError, match="row 2: its tpl"):
        _source(tmp_path, bad, privileged_field="hint", privileged_template_field="tpl")


def test_view_options_are_validated():
    from palingenesis.opd.config import OPDConfig, OPDConfigError

    def errors(**source):
        config = OPDConfig()
        config.set("model.student", "s", "test")
        config.set("teachers.t.model", "t", "test")
        config.set("sources.a.path", "a.jsonl", "test")
        for key, value in source.items():
            config.set(f"sources.a.{key}", value, "test")
        try:
            config.validate()
        except OPDConfigError as exc:
            return str(exc)
        return ""

    re2 = {"name": "re2", "template": "{content}\n\n{question}"}
    assert errors(privileged_views=[re2]) == ""
    assert errors(privileged_views=[re2, {"name": "fact", "template": "{privileged}", "field": "f", "weight": 2}]) == ""
    assert "replaces privileged_field" in errors(privileged_views=[re2], privileged_field="x")
    assert "names must be unique" in errors(privileged_views=[re2, re2])
    assert "keys" in errors(privileged_views=[{**re2, "kind": "x"}])
    assert "weight must be a number > 0" in errors(privileged_views=[{**re2, "weight": 0}])
    assert "short identifier" in errors(privileged_views=[{**re2, "name": "a b"}])
    assert "privileged_views[0].role" in errors(privileged_views=[{**re2, "role": "assistant"}])
    assert "privileged_views[0].template" in errors(privileged_views=[{**re2, "template": "{content} {privileged}"}])
    assert "format 'messages' only" in errors(privileged_views=[re2], format="mcqa")
    assert "needs privileged_field or privileged_views" in errors(privileged_template_field="tpl")
    assert "" == errors(privileged_field="hint", privileged_template="{privileged}")  # a rewording: no {content}
