"""Privileged views that change the tool list the teacher sees (context ablation of a trigger): tools: drop and
tools_field, on-policy (pgs distill) and off-policy (teacher views in pgs train)."""

import json
import random

import pytest

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
STOCKS = {
    "type": "function",
    "function": {"name": "get_stock_price", "description": "A stock's price", "parameters": {"type": "object"}},
}


def _source(tmp_path, rows, views):
    from palingenesis.opd.config import SourceConfig
    from palingenesis.opd.sources import ChatMessagesSource

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = SourceConfig(path=str(path), dev_size=0, max_new_tokens=16, privileged_views=views)
    return ChatMessagesSource(config, eval_samples=2, seed=0, rng=random.Random(0))


def _errors(**source):
    from palingenesis.opd.config import OPDConfig, OPDConfigError

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


def test_teacher_prompt_has_no_tool_schemas_while_the_student_has_them(tmp_path):
    from palingenesis.opd.orchestrator import Pipeline
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    rows = [
        {"messages": [{"role": "user", "content": "Write a haiku about autumn."}], "tools": [WEATHER]},
        {"messages": [{"role": "user", "content": "Write a limerick."}]},  # no tools: nothing to drop
    ]
    source = _source(tmp_path, rows, [{"name": "no_tools", "tools": "drop"}])
    meta = source.meta(rows[0])
    assert meta["teacher_tools"] is None and meta["_view"] == "no_tools" and "teacher_messages" not in meta
    assert source.meta(rows[1]) == {}  # the view does not apply

    class Route:
        tokenizer = tok

    pipe = Pipeline.__new__(Pipeline)
    pipe.student_tok, pipe.routes, pipe.chat_template_kwargs = tok, {"t": Route()}, {"enable_thinking": False}
    request = pipe.request(rows[0]["messages"], 16, meta, "s", "t")
    student, teacher = tok.decode(request.prompt_ids), tok.decode(request.teacher_prompt_ids)
    assert "get_weather" in student and "get_weather" not in teacher
    assert student.endswith(teacher[-60:])  # the same request, only the tool block differs
    # without the view, the teacher renders the student's tools (unchanged behaviour)
    plain = pipe.request(rows[0]["messages"], 16, {"tools": [WEATHER]}, "s", "t")
    assert plain.prompt_ids == plain.teacher_prompt_ids


def test_tools_field_gives_the_teacher_its_own_list_and_follows_the_field_rules(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "a"}], "tools": [WEATHER, STOCKS], "teacher_tools": [WEATHER]},
        {"messages": [{"role": "user", "content": "b"}], "tools": [WEATHER], "teacher_tools": json.dumps([])},
        {"messages": [{"role": "user", "content": "c"}], "tools": [WEATHER]},  # field missing: plain row
        {"messages": [{"role": "user", "content": "d"}], "tools": [WEATHER], "teacher_tools": ""},  # empty: plain
    ]
    view = {"name": "subset", "tools_field": "teacher_tools"}
    source = _source(tmp_path, rows, [view])
    assert source.meta(rows[0])["teacher_tools"] == [WEATHER]
    assert source.meta(rows[1])["teacher_tools"] is None  # [] = the teacher sees no tools
    assert "teacher_tools" not in source.meta(rows[2]) and "teacher_tools" not in source.meta(rows[3])
    # a view with a template and a tool change does both
    both = _source(tmp_path, rows, [{**view, "name": "both", "template": "{content}\nAct, do not ask."}])
    meta = both.meta(rows[0])
    assert meta["teacher_tools"] == [WEATHER] and meta["teacher_messages"][0]["content"] == "a\nAct, do not ask."
    bad = rows + [{"messages": [{"role": "user", "content": "e"}], "teacher_tools": "{not json"}]
    with pytest.raises(ValueError, match="row 4: teacher_tools"):
        _source(tmp_path, bad, [view])
    with pytest.raises(ValueError, match="list of tool schemas"):
        _source(tmp_path, [{"messages": [{"role": "user", "content": "f"}], "teacher_tools": {"x": 1}}], [view])


def test_view_validation_tools():
    assert _errors(privileged_views=[{"name": "no_tools", "tools": "drop"}]) == ""
    assert _errors(privileged_views=[{"name": "subset", "tools_field": "teacher_tools"}]) == ""
    assert _errors(privileged_views=[{"name": "x", "tools": "drop", "template": "{content}\nAct."}]) == ""
    assert "changes nothing the teacher sees" in _errors(privileged_views=[{"name": "noop"}])
    assert "changes nothing the teacher sees" in _errors(privileged_views=[{"name": "noop", "tools": "keep"}])
    assert "unchanged" in _errors(privileged_views=[{"name": "noop", "template": "{content}"}])
    assert "needs a template" in _errors(privileged_views=[{"name": "x", "tools": "drop", "field": "hint"}])
    assert "contradict" in _errors(privileged_views=[{"name": "x", "tools": "drop", "tools_field": "t"}])
    assert "tools must be one of" in _errors(privileged_views=[{"name": "x", "tools": "none"}])


def test_off_policy_teacher_view_drops_or_replaces_the_tools(tmp_path):
    from palingenesis.config import Config, ConfigError
    from palingenesis.data import _load_dataset_source, source_stage
    from palingenesis.distill.config import parse_teachers
    from palingenesis.distill.views import TeacherView
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    row = {
        "messages": [
            {"role": "user", "content": "Write a haiku about autumn."},
            {"role": "assistant", "content": "Leaves fall."},
        ],
        "tools": [WEATHER, STOCKS],
        "teacher_tools": [STOCKS],
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    c = Config()
    c.data.max_seq_length = 1024
    c.data.chat_template_kwargs = {"enable_thinking": False}

    def render(**options):
        spec = parse_teachers({"t": {"model": "x", **options}})[0]
        stage = source_stage(
            _load_dataset_source(str(path), "train", False), tok, c.data, None, 0, 1,
            teacher_views=[(TeacherView(spec), tok, 2048)],
        )  # fmt: skip
        sample = next(iter(stage))
        return tok.decode(sample["input_ids"]), tok.decode(sample["teacher_input_ids.t"]), sample

    student, teacher, sample = render(tools="drop")
    assert "get_weather" in student and "get_weather" not in teacher and "get_stock_price" not in teacher
    trained = (sample["labels"][1:] != -100).nonzero().flatten()
    assert (sample["teacher_positions.t"][trained] >= 0).all()  # the answer tokens still align
    _, teacher, _ = render(tools_field="teacher_tools")
    assert "get_stock_price" in teacher and "get_weather" not in teacher
    _, teacher, _ = render(system_prompt="Be brief.")  # tools kept by default
    assert "get_weather" in teacher
    spec = parse_teachers({"t": {"model": "x", "tools": "drop"}})[0]
    assert spec.has_view and TeacherView(spec).describe()["tools"] == "drop"
    assert "tools" not in TeacherView(parse_teachers({"t": {"model": "x", "system_prompt": "s"}})[0]).describe()
    bad = Config()
    bad.model.name_or_path = "m"
    bad.distill.enabled = True
    bad.distill.teachers = {"t": {"model": "x", "tools": "none"}}
    with pytest.raises(ConfigError, match="tools must be 'keep' or 'drop'"):
        bad.validate()
