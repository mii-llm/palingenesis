"""pgs eval: task configs, field mapping, the built-in scorers, summaries and the `pgs compare` format, with a
scripted engine through the RL pipeline (no GPU)."""

import json

import pytest

pytest.importorskip("torch")

from palingenesis.eval import ifeval  # noqa: E402
from palingenesis.eval.config import (  # noqa: E402
    MODES,
    EvalConfig,
    EvalConfigError,
    EvalTasksConfig,
    SamplingConfig,
    builtin_tasks,
    load_task,
    merge_sampling,
    resolve_modes,
    task_errors,
)
from palingenesis.eval.scorers import (  # noqa: E402
    exact_match,
    extract_choice,
    extract_json,
    extract_number,
    function_call,
    json_schema,
    mcq,
    numeric,
    regex_match,
    resolve_scorers,
    schema_errors,
    validate_judge,
)
from palingenesis.eval.scorers import ifeval as ifeval_scorer  # noqa: E402
from palingenesis.eval.tasks import build_items, map_field, render, tool_schemas  # noqa: E402

# ------------------------------------------------------------------- configs


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_builtin_tasks_load_and_validate():
    tasks = builtin_tasks()
    assert {"gsm8k", "math500", "mmlu_pro", "ifeval", "bfcl_simple"} <= set(tasks)
    for name in tasks:
        task = load_task(name)
        assert task.name == name and task.dataset and task.description
        assert task_errors(task, set(MODES)) == []
        assert resolve_scorers(task)[0].fn is not None  # every built-in task names a scorer that exists


def test_task_errors_name_the_option(tmp_path):
    with pytest.raises(EvalConfigError, match="unknown task 'gsm8'.*did you mean gsm8k"):
        load_task("gsm8")
    with pytest.raises(EvalConfigError, match=r"unknown option task.scorrer \(did you mean scorer\?\)"):
        load_task({"name": "t", "dataset": "d.jsonl", "scorrer": "mcq"})
    with pytest.raises(EvalConfigError, match="task.limit must be a int"):
        load_task({"name": "t", "dataset": "d.jsonl", "limit": [1]})
    with pytest.raises(EvalConfigError, match="`name` must be letters"):
        load_task({"name": "a/b", "dataset": "d.jsonl"})
    with pytest.raises(EvalConfigError, match="no score"):
        task_errors(load_task({"name": "t", "dataset": "d.jsonl"}), set(MODES))
    with pytest.raises(EvalConfigError, match="give one of prompt, messages and prompt_field"):
        task_errors(
            load_task({"name": "t", "dataset": "d", "scorer": "mcq", "prompt": "x", "prompt_field": "q"}), set()
        )
    bad = {
        "name": "t",
        "dataset": "d",
        "scorer": "mcq",
        "fields": {"a": {"from": "x", "regx": "y"}, "b": {"regex": "("}},
    }
    with pytest.raises(EvalConfigError) as e:
        task_errors(load_task(bad), set(MODES))
    assert "unknown key 'regx' (did you mean regex?)" in str(e.value)
    assert "fields.b: give exactly one of from, template, value" in str(e.value) and "does not compile" in str(e.value)
    with pytest.raises(EvalConfigError, match="is not a scorer.*did you mean numeric"):
        resolve_scorers(load_task({"name": "t", "dataset": "d", "scorer": "numerik"}))
    with pytest.raises(EvalConfigError, match="sampling: top_p"):
        load_task({"name": "t", "dataset": "d", "scorer": "mcq", "sampling": {"top_p": "high"}}) and task_errors(
            load_task({"name": "t", "dataset": "d", "scorer": "mcq", "sampling": {"top_p": "high"}}), set()
        )
    path = tmp_path / "mine.yaml"
    path.write_text("dataset: d.jsonl\nscorer: exact_match\nk: 4\nsampling: {temperature: 0.7}\n")
    task = load_task(str(path))
    assert (task.name, task.k) == ("mine", 4)
    override = load_task({"task": "gsm8k", "limit": 50, "name": "gsm8k_small"})
    assert (override.name, override.limit, override.dataset) == ("gsm8k_small", 50, "openai/gsm8k")


def test_eval_config_from_yaml_and_validation(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text(
        "model: {path: some/model}\n"
        "engine: {backend: hf, max_model_len: 4096}\n"
        "run: {k: 4, modes: [plain, careful], output_dir: out}\n"
        "sampling: {temperature: 0.7, top_p: 0.8}\n"
        "modes: {careful: {chat_template_kwargs: {enable_thinking: true}, sampling: {max_tokens: 8192}}}\n"
        "tasks: [gsm8k, {task: math500, limit: 10, modes: [think]}]\n"
    )
    config = EvalConfig.from_yaml(path)
    assert config.validate() == []
    assert [t.name for t in config.tasks] == ["gsm8k", "math500"] and config.tasks[1].limit == 10
    modes = resolve_modes(config.run.modes, config.modes, "run.modes")
    assert modes["plain"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert modes["careful"]["sampling"].max_tokens == 8192
    config.set("run.k", "8")
    config.set("sampling.temperature", "none")
    config.set("engine.vllm_args", "{max_num_batched_tokens: 4096}")
    assert config.run.k == 8 and config.sampling.temperature is None
    assert config.engine["vllm_args"] == {"max_num_batched_tokens": 4096}
    with pytest.raises(EvalConfigError, match="use model.<option>"):
        config.set("tasks.0", "x")
    # errors are collected and name what to change
    config.model.path, config.run.modes, config.engine["temperature"] = "", ["plain", "fast"], 0.5
    config.sampling.top_p = 1.5
    with pytest.raises(EvalConfigError) as e:
        config.validate()
    text = str(e.value)
    assert "model.path is empty" in text and "unknown mode 'fast'" in text
    assert "engine.temperature: set sampling.temperature" in text and "sampling.top_p must be in (0, 1]" in text
    path.write_text("model: {path: m}\nrunn: {k: 1}\n")
    with pytest.raises(EvalConfigError, match=r"unknown section `runn` \(did you mean run\?\)"):
        EvalConfig.from_yaml(path)
    path.write_text("model: {path: m}\nrun: {k: four}\n")
    with pytest.raises(EvalConfigError, match="run.k='four' is not a valid int"):
        EvalConfig.from_yaml(path)
    # greedy with several samples is flagged
    config = EvalConfig()
    config.model.path, config.run.k, config.sampling.temperature = "m", 4, 0.0
    config.tasks = [load_task("gsm8k")]
    assert "the same greedy sample 4 times" in config.validate()[0]


def test_sampling_layers_and_defaults(monkeypatch):
    from palingenesis.eval import runner

    merged = merge_sampling(SamplingConfig(temperature=0.7, top_k=20), SamplingConfig(max_tokens=512, top_k=None))
    assert (merged.temperature, merged.top_k, merged.max_tokens) == (0.7, 20, 512)
    monkeypatch.setattr(runner, "model_sampling", lambda model: SamplingConfig())
    s, source = runner.resolve_sampling("m", 1)
    assert (s.temperature, s.top_p, s.top_k, s.max_tokens, source) == (0.0, 1.0, 0, 2048, "default")
    assert runner.resolve_sampling("m", 4)[0].temperature == 1.0  # k identical greedy samples carry nothing
    monkeypatch.setattr(runner, "model_sampling", lambda model: SamplingConfig(temperature=0.6, top_p=0.95, top_k=20))
    s, source = runner.resolve_sampling("m", 1, SamplingConfig(top_k=40))
    assert (s.temperature, s.top_p, s.top_k, source) == (0.6, 0.95, 40, "generation_config")
    s, source = runner.resolve_sampling("m", 1, SamplingConfig(temperature=0.0))
    assert (s.temperature, s.top_p, source) == (0.0, 1.0, "config")  # an explicit temperature keeps its own top_p
    assert runner.engine_sampling(s) == 0.0
    sampling = runner.engine_sampling(runner.resolve_sampling("m", 1, SamplingConfig(presence_penalty=1.5))[0])
    assert (sampling.temperature, sampling.top_k, sampling.presence_penalty) == (0.6, 20, 1.5)


# ------------------------------------------------------------- field mapping


def test_field_mapping_transforms():
    row = {
        "answer": "She has 16 - 3 = 13.\n#### 1,300",
        "options": ["red", " green ", "blue"],
        "answer_index": 2,
        "nested": {"turns": [[{"role": "user", "content": "hi"}]]},
        "meta": '{"level": 3}',
        "label": "yes",
        "empty": None,
    }
    assert map_field("nested.turns.0.0.content", row, "w") == "hi"
    assert map_field({"from": "answer", "regex": r"####\s*(.+)$", "replace": {",": ""}}, row, "w") == "1300"
    assert map_field({"from": "options", "choices": True}, row, "w") == "A. red\nB. green\nC. blue"
    assert map_field({"from": "answer_index", "letter": True}, row, "w") == "C"
    assert map_field({"from": "meta", "json": True}, row, "w") == {"level": 3}
    assert map_field({"from": "label", "map": {"yes": "A", "no": "B"}}, row, "w") == "A"
    assert map_field({"value": [1, 2]}, row, "w") == [1, 2]
    assert map_field({"template": "{{label}} / {{nested.turns.0.0.content}}"}, row, "w") == "yes / hi"
    assert map_field({"from": "empty", "default": "n/a"}, row, "w") == "n/a"
    assert map_field({"from": "answer", "strip": False, "regex": r"(#### )", "group": 1}, row, "w") == "#### "
    for spec, message in [
        ("missing", "column 'missing' is not in the dataset"),
        ({"from": "empty"}, "column 'empty' is null"),
        ({"from": "label", "regex": r"\d+"}, "does not match"),
        ({"from": "label", "map": {"no": "B"}}, "map has no entry for 'yes'"),
        ({"from": "label", "choices": True}, "choices needs a list"),
        ({"from": "label", "letter": True}, "letter needs an option index"),
    ]:
        with pytest.raises(EvalConfigError, match=message):
            map_field(spec, row, "task t, row 0: fields.x")


def test_prompt_rendering_leaves_latex_alone():
    row = {"problem": "Compute $\\frac{1}{2}+\\frac{1}{2}$.", "answer": "1"}
    text = render("{{problem}}\n\nPut the final answer in \\boxed{}. Use {answer} literally. {{ answer }}", row, "w")
    assert (
        text == "Compute $\\frac{1}{2}+\\frac{1}{2}$.\n\nPut the final answer in \\boxed{}. Use {answer} literally. 1"
    )
    with pytest.raises(EvalConfigError, match=r"placeholder \{\{questoin\}\} is not a field"):
        render("{{questoin}}", row, "task t, row 0: prompt")


def test_tool_schemas_normalise_types():
    spec = {
        "name": "math.hypot",
        "description": "norm",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": "float"}, "type": {"type": "tuple", "items": {"type": "String"}}},
            "required": ["x"],
        },
    }
    (tool,) = tool_schemas([spec])
    parameters = tool["function"]["parameters"]
    assert tool["type"] == "function" and tool["function"]["name"] == "math.hypot"
    assert parameters["type"] == "object" and parameters["properties"]["x"] == {"type": "number"}
    assert parameters["properties"]["type"] == {"type": "array", "items": {"type": "string"}}  # a property named type
    assert tool_schemas(json.dumps([tool])) == [tool]  # already wrapped, as a JSON string
    with pytest.raises(EvalConfigError, match="needs a name"):
        tool_schemas([{"description": "x"}])


def test_build_items_jsonl_parquet_limit_ids(tmp_path):
    rows = [
        {"qid": f"q{i}", "question": f"What is {i} + {i}?", "answer": str(2 * i), "topic": i % 3} for i in range(20)
    ]
    data = _write(tmp_path / "data.jsonl", rows)
    spec = {
        "name": "sums",
        "dataset": str(data),
        "id_field": "qid",
        "cluster_field": "topic",
        "fields": {"gold": "answer"},
        "prompt": "{{question}} Reply with \\boxed{}.",
        "scorer": "numeric",
        "scorer_args": {"field": "gold"},
    }
    task = load_task(spec)
    items = build_items(task)
    assert len(items) == 20 and items[3].id == "sums/q3" and items[3].cluster == "0"
    assert items[3].row["messages"] == [{"role": "user", "content": "What is 3 + 3? Reply with \\boxed{}."}]
    assert items[3].row["gold"] == "6" and items[3].row["question"] == "What is 3 + 3?"  # columns travel along
    sample = [i.id for i in build_items(load_task({**spec, "limit": 5}))]
    assert len(sample) == 5 and sample == [i.id for i in build_items(task, limit=5)]  # fixed, the run's or the task's
    assert sample == sorted(sample, key=lambda s: int(s.split("q")[1])) and sample != [f"sums/q{i}" for i in range(5)]
    assert sample != [i.id for i in build_items(load_task({**spec, "limit": 5, "sample_seed": 1}))]
    chat = load_task({**spec, "prompt": "", "messages": [{"role": "system", "content": "Topic {{topic}}."},
                                                          {"role": "user", "content": "{{question}}"}]})  # fmt: skip
    assert build_items(chat)[1].row["messages"][0] == {"role": "system", "content": "Topic 1."}
    with pytest.raises(EvalConfigError, match="appears twice"):
        build_items(load_task({**spec, "id_field": "topic"}))
    with pytest.raises(EvalConfigError, match="row 0: fields.gold: column 'answr' is not in the dataset"):
        build_items(load_task({**spec, "fields": {"gold": "answr"}}))
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"question": ["Pick."], "options": [["x", "y"]], "answer_index": [1]})
    frame.to_parquet(tmp_path / "mc.parquet")
    mc = load_task({
        "name": "mc", "dataset": str(tmp_path / "mc.parquet"), "scorer": "mcq", "prompt": "{{question}}\n{{choices}}",
        "fields": {"choices": {"from": "options", "choices": True}, "answer": {"from": "answer_index", "letter": True}},
    })  # fmt: skip
    (item,) = build_items(mc)
    assert item.row["messages"][0]["content"] == "Pick.\nA. x\nB. y" and item.row["answer"] == "B"
    assert item.row["options"] == ["x", "y"] and item.id == "mc/0"  # arrays become lists


def test_join_and_json_lines_with_json_extension(tmp_path):
    (tmp_path / "possible_answer").mkdir()
    _write(tmp_path / "q.json", [{"id": "s_0", "question": [[{"role": "user", "content": "Area of a 3 by 4 box?"}]],
                                  "function": [{"name": "area", "parameters": {"type": "dict", "properties": {}}}]}])  # fmt: skip
    _write(tmp_path / "possible_answer" / "q.json", [{"id": "s_0", "ground_truth": [{"area": {"w": [3], "h": [4]}}]}])
    task = load_task({"task": "bfcl_simple", "dataset": str(tmp_path), "data_files": "q.json",
                      "join": {"data_files": "possible_answer/q.json", "key": "id"}})  # fmt: skip
    (item,) = build_items(task)
    assert item.id == "bfcl_simple/s_0" and item.row["messages"][0]["content"] == "Area of a 3 by 4 box?"
    assert item.row["tools"][0]["function"]["parameters"]["type"] == "object"
    assert item.row["ground_truth"] == [{"area": {"w": [3], "h": [4]}}]
    _write(tmp_path / "possible_answer" / "q.json", [{"id": "other", "ground_truth": []}])
    with pytest.raises(EvalConfigError, match="join: no row with id = 's_0'"):
        build_items(task)


def test_offline_dataset_from_the_hub_cache(tmp_path, monkeypatch):
    """Offline with an empty datasets cache: the split's file in the Hub cache snapshot is read directly."""
    datasets = pytest.importorskip("datasets")
    hub = pytest.importorskip("huggingface_hub")

    def offline(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(datasets, "load_dataset", offline)
    (tmp_path / "main").mkdir()
    _write(tmp_path / "main" / "test.jsonl", [{"question": "2 + 2?", "answer": "#### 4"}])
    _write(tmp_path / "main" / "train.jsonl", [{"question": "1 + 1?", "answer": "#### 2"}] * 3)
    monkeypatch.setattr(hub, "snapshot_download", lambda repo, **kwargs: str(tmp_path))
    (item,) = build_items(load_task("gsm8k"))
    assert item.row["answer"] == "4" and item.row["messages"][0]["content"].startswith("2 + 2?")
    with pytest.raises(EvalConfigError, match="cannot load openai/gsm8k .*split validation.*ConnectionError: offline"):
        build_items(load_task({"task": "gsm8k", "split": "validation"}))


# -------------------------------------------------------------------- scorers


def test_exact_match():
    assert exact_match("  Paris ", answer="paris") == 1.0
    assert exact_match("Paris", answer="paris", ignore_case=False) == 0.0
    assert exact_match("It is Paris.", answer="Paris") == 0.0
    assert exact_match("Reasoning...\nFinal answer: **Paris**.", answer="Paris", extract="answer") == 1.0
    assert exact_match("thinking\nParis", answer=["Rome", "Paris"], extract="last_line") == 1.0
    assert exact_match("so \\boxed{x^2 + 1}", gold="x^2 + 1", field="gold", extract="boxed") == 1.0
    assert exact_match("no box", answer="1", extract="boxed") == 0.0
    assert exact_match("CODE: ab-12", answer="AB 12", extract="regex", pattern=r"CODE: (\S+)") == 0.0
    assert exact_match("the u.s.a!", answer="The USA", ignore_punctuation=True) == 1.0
    with pytest.raises(KeyError, match="no 'answer' column"):
        exact_match("x")


def test_numeric_and_number_extraction():
    assert numeric("16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\nAnswer: 18", answer="18") == 1.0
    assert numeric("She makes $1,300.50 a day.", answer="1300.5") == 1.0  # the last number
    assert numeric("The total is \\boxed{18} dollars, not 20.", answer="18") == 1.0  # the box over later numbers
    assert numeric("The total is \\boxed{18} dollars, not 20.", answer="18", extract="last_number") == 0.0
    assert numeric("Answer: 18", answer="18", extract="boxed") == 0.0  # boxed only
    assert numeric("Answer: 3/4", answer="0.75") == 1.0
    assert numeric("It is -2.5e3.", answer="-2500") == 1.0
    assert numeric("about 3.14", answer="3.14159", rel_tol=1e-2) == 1.0
    assert numeric("about 3.14", answer="3.14159") == 0.0
    assert numeric("I do not know.", answer="18") == 0.0
    assert numeric("Answer: 72%", answer=["0.72", "72"]) == 1.0
    assert extract_number("first 5 then \\boxed{\\text{seven}} and 9") == 9.0  # a box without a number falls through
    with pytest.raises(ValueError, match="holds no number"):
        numeric("1", answer="n/a")


def test_mcq_letter_extraction():
    assert mcq("Options A and B are wrong. The answer is (C).", answer="C") == 1.0
    assert mcq("I considered B. Answer: D", answer="D") == 1.0
    assert mcq("The correct option is **J**", answer="J") == 1.0
    assert mcq("\\boxed{\\text{B}}", answer="B") == 1.0
    assert mcq("b", answer="B") == 1.0  # a bare letter
    assert mcq("Between A and C, C fits best", answer="C") == 1.0  # the last standalone letter
    assert mcq("The answer is (C).", answer=2) == 1.0 and mcq("The answer is (C).", answer="(c)") == 1.0
    assert mcq("The answer is (C).", answer="B") == 0.0
    assert mcq("I cannot decide.", answer="A") == 0.0
    assert extract_choice("The answer is K", letters="ABCD") is None
    assert extract_choice("Answer: A. But on reflection the answer is B") == "B"  # the last statement wins


def test_regex_match():
    assert regex_match("order #A-17 confirmed", pattern=r"#([A-Z]-\d+)", field="code", code="a-17") == 1.0
    assert regex_match("order #A-17 confirmed", pattern=r"#([A-Z]-\d+)", field="code", code="A-18") == 0.0
    assert regex_match("order confirmed", pattern=r"#([A-Z]-\d+)", field="code", code="A-17") == 0.0
    assert regex_match("2026-11-25", pattern=r"^\d{4}-\d{2}-\d{2}$") == 1.0  # no field: a match is enough


def test_json_schema_scorer():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0},
                       "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }  # fmt: skip
    good = 'Here it is:\n```json\n{"name": "Ada", "age": 36, "tags": ["a"]}\n```'
    assert json_schema(good, schema=schema) == 1.0
    assert json_schema('{"name": "Ada", "age": 36}', schema_field="rule", rule=json.dumps(schema)) == 1.0
    assert json_schema('The object {"name": "Ada", "age": 36} is valid.', schema=schema) == 1.0
    assert json_schema('{"name": "Ada"}', schema=schema) == 0.0  # required
    assert json_schema('{"name": "Ada", "age": "36"}', schema=schema) == 0.0  # type
    assert json_schema('{"name": "Ada", "age": 36, "x": 1}', schema=schema) == 0.0  # additionalProperties
    assert json_schema('{"name": "Ada", "age": 36, "tags": ["c"]}', schema=schema) == 0.0  # enum
    assert json_schema("not json at all", schema=schema) == 0.0
    assert json_schema('{"a": [1, 2]}', expected_field="want", want='{"a": [1, 2]}') == 1.0
    assert json_schema('{"a": [2, 1]}', expected_field="want", want={"a": [1, 2]}) == 0.0
    with pytest.raises(ValueError, match="give schema"):
        json_schema("{}")
    assert extract_json("```\n[1, 2]\n```") == [1, 2]


def test_builtin_schema_validator_matches_keywords():
    from palingenesis.eval.scorers import _schema_errors

    schema = {
        "type": "object",
        "properties": {
            "kind": {"const": "point"},
            "xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "label": {"anyOf": [{"type": "null"}, {"type": "string", "pattern": "^[a-z]+$"}]},
            "n": {"oneOf": [{"type": "integer", "maximum": 3}, {"type": "integer", "minimum": 10}]},
        },
        "required": ["kind", "xy"],
    }
    assert _schema_errors({"kind": "point", "xy": [1, 2.5], "label": None, "n": 12}, schema, "$") == []
    errors = _schema_errors({"kind": "line", "xy": [1, True, 3], "label": "Ab", "n": 5}, schema, "$")
    assert len(errors) == 5 and any("anyOf" in e for e in errors) and any("oneOf" in e for e in errors)
    assert _schema_errors([], schema, "$") == ["$: expected object, got list"]
    assert _schema_errors({"xy": [1, 2]}, schema, "$") == ["$: missing 'kind'"]
    assert schema_errors({"kind": "point", "xy": [1, 2]}, schema) == []  # jsonschema when installed, else the same


def _calls(*calls):
    tool_calls = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]
    return [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": tool_calls}]


def test_function_call_scorer():
    expected = [{"calculate_area": {"base": [10], "height": [5, 5.0], "unit": ["units", ""]}}]
    score = lambda *calls, **kw: function_call(_calls(*calls), expected_calls=kw.pop("expected", expected), **kw)  # noqa: E731
    assert score(("calculate_area", {"base": 10, "height": 5})) == 1.0  # the optional argument left out
    assert score(("calculate_area", '{"base": 10.0, "height": 5, "unit": "Units"}')) == 1.0  # JSON string, 10.0, case
    assert score(("calculate_area", {"base": 10})) == 0.0  # a required argument is missing
    assert score(("calculate_area", {"base": 10, "height": 6})) == 0.0  # a value outside the accepted ones
    assert score(("calculate_area", {"base": 10, "height": 5, "color": "red"})) == 0.0  # an argument not listed
    assert score(("area", {"base": 10, "height": 5})) == 0.0  # another function
    assert score() == 0.0 and score(("calculate_area", {"base": 10, "height": 5}), ("calculate_area", {})) == 0.0
    assert score(("calculate_area", "{broken")) == 0.0
    two = [{"get_weather": {"city": ["New York", "NYC"]}}, {"get_time": {"zone": ["UTC"]}}]
    assert score(("get_time", {"zone": "utc"}), ("get_weather", {"city": "new-york"}), expected=two) == 1.0
    assert score(("get_time", {"zone": "utc"}), ("get_weather", {"city": "NYC"}), expected=two, ordered=True) == 0.0
    assert score(expected=[]) == 1.0 and score(("get_time", {}), expected=[]) == 0.0  # irrelevance: no call expected
    nested = [{"book": {"guest": [{"name": ["Ada"], "age": [36, ""]}], "nights": [[1, 2]]}}]
    assert score(("book", {"guest": {"name": "ada"}, "nights": [1, 2]}), expected=nested) == 1.0
    assert score(("book", {"guest": {"name": "Bob"}, "nights": [1, 2]}), expected=nested) == 0.0
    assert function_call(_calls(("f", {})), field="ground_truth", ground_truth='[{"f": {}}]') == 1.0


def test_ifeval_checks():
    f = lambda text, ids, kwargs, **kw: ifeval.follows(text, ids, kwargs, checker="builtin", **kw)  # noqa: E731
    essay = "Dear team, here is the plan.\n\n***\n\nWe ship on Monday.\n\n***\n\nP.S. Bring coffee"
    cases = [
        ("punctuation:no_comma", {}, "No commas here.", "One, two."),
        ("length_constraints:number_words", {"relation": "at least", "num_words": 3}, "one two three", "one two"),
        ("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "one two", "one two three"),
        (
            "length_constraints:number_sentences",
            {"relation": "at least", "num_sentences": 2},
            "Hi there. Bye now.",
            "Hi.",
        ),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 3}, essay, "one\n\n***\n\ntwo"),
        (
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Then"},
            "First we plan.\n\nThen, we build.",
            "First we plan.\n\nWe build.",
        ),
        ("keywords:existence", {"keywords": ["plan", "ship"]}, essay, "No such words."),
        ("keywords:frequency", {"keyword": "ha", "frequency": 2, "relation": "at least"}, "ha ha", "ha"),
        ("keywords:forbidden_words", {"forbidden_words": ["cat"]}, "A concatenation.", "The cat sat."),
        ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "less than"}, "zebra", "Zzz"),
        ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [address]", "[name] only"),
        ("detectable_content:postscript", {"postscript_marker": "P.S."}, essay, "No postscript."),
        ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n* two", "* one\n* two\n- three"),
        ("detectable_format:constrained_response", {}, "My answer is maybe.", "Maybe."),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* and **b**", "*a* only"),
        (
            "detectable_format:multiple_sections",
            {"section_spliter": "SECTION", "num_sections": 2},
            "SECTION 1\nintro\nSECTION 2\nbody",
            "SECTION 1\nintro",
        ),
        ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "a: 1"),
        ("detectable_format:title", {}, "<<A Title>>\nbody", "A Title\nbody"),
        ("combination:two_responses", {}, "first ****** second", "same ****** same"),
        ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi"}, "say hi\nHi!", "Hi!"),
        ("startend:end_checker", {"end_phrase": "Is there anything else?"}, "Done. is there anything else?", "Done."),
        ("startend:quotation", {}, '"All of it."', 'Only "part".'),
        (
            "change_case:capital_word_frequency",
            {"capital_frequency": 2, "capital_relation": "at least"},
            "NO WAY",
            "NO",
        ),
        ("change_case:english_capital", {}, "ALL CAPS HERE.", "Not All Caps."),
        ("change_case:english_lowercase", {}, "all lower here.", "Not lower."),
    ]
    for instruction, kwargs, good, bad in cases:
        nulls = {**kwargs, "unused": None}  # the dataset stores every key, null where unused
        assert f(good, [instruction], [nulls]) == [True], (instruction, good)
        assert f(bad, [instruction], [nulls]) == [False], (instruction, bad)
    assert set(ifeval.CHECKS) - {c[0] for c in cases} == {"language:response_language"}
    both = ["punctuation:no_comma", "length_constraints:number_words"]
    assert f("one, two", both, [{}, {"relation": "at least", "num_words": 2}]) == [False, True]
    assert f("", ["punctuation:no_comma"], [{}]) == [False]  # an empty response follows nothing
    # loose: the first / last line and asterisks may be ignored
    wrapped = 'Sure, here it is:\n{"a": 1}\nHope that helps'
    assert f(wrapped, ["detectable_format:json_format"], [{}]) == [False]
    assert f(wrapped, ["detectable_format:json_format"], [{}], loose=True) == [True]
    with pytest.raises(ValueError, match="unknown instruction"):
        f("x", ["made:up"], [{}])
    # the scorer: prompt level (all), instruction level (share), and items that cannot be checked
    row = {"instruction_id_list": both, "kwargs": [{}, {"relation": "at least", "num_words": 2}]}
    assert ifeval_scorer("one, two", checker="builtin", **row) == 0.0
    assert ifeval_scorer("one, two", checker="builtin", level="instruction", **row) == 0.5
    assert ifeval_scorer("one two", checker="builtin", **row) == 1.0
    language = {"instruction_id_list": ["language:response_language"], "kwargs": [{"language": "it"}]}
    if not ifeval.has_langdetect():
        assert ifeval_scorer("Ciao a tutti.", checker="builtin", **language) is None
    if ifeval.official_registry() is None:
        with pytest.raises(RuntimeError, match="needs the reference implementation"):
            ifeval.follows("x", ["punctuation:no_comma"], [{}], checker="official")


def test_sentence_splitter():
    text = "Dr. Smith paid $3.50 for it. Was it worth it? Yes! See example.com for more."
    assert ifeval.split_into_sentences(text) == [
        "Dr. Smith paid $3.50 for it.", "Was it worth it?", "Yes!", "See example.com for more.",
    ]  # fmt: skip


def test_judge_validation(tmp_path, monkeypatch):
    from palingenesis.rl import rewards

    task = load_task({"name": "essay", "dataset": "d.jsonl", "scorer": "judge",
                      "scorer_args": {"url": "http://judge", "model": "j"}})  # fmt: skip
    (reward,) = resolve_scorers(task)
    with pytest.raises(EvalConfigError, match="must be checked against human labels"):
        validate_judge(reward, {}, "essay")
    assert validate_judge(reward, {"skip": True}, "essay") == {"validated": False, "skipped": True}
    labelled = [{"question": f"q{i}", "completion": "good" if i % 2 else "bad", "label": i % 2} for i in range(20)]
    path = _write(tmp_path / "labels.jsonl", labelled)

    async def agreeing(url, model, content, api_key, timeout, max_tokens=1024):
        return "Score: 9" if "[Answer]\ngood" in content else "Score: 2"

    monkeypatch.setattr(rewards, "_chat", agreeing)
    report = validate_judge(reward, {"file": str(path), "min_agreement": 0.9}, "essay")
    assert report["validated"] and report["agreement"] == 1.0 and report["kappa"] == 1.0 and report["examples"] == 20

    async def lenient(url, model, content, api_key, timeout, max_tokens=1024):
        return "Score: 9"

    monkeypatch.setattr(rewards, "_chat", lenient)
    with pytest.raises(EvalConfigError, match="agrees with the labels on 50.0% of 20 examples"):
        validate_judge(reward, {"file": str(path)}, "essay")
    with pytest.raises(EvalConfigError, match="fewer than min_examples"):
        validate_judge(reward, {"file": str(_write(tmp_path / "few.jsonl", labelled[:3]))}, "essay")
    with pytest.raises(EvalConfigError, match="unknown keys"):
        validate_judge(reward, {"fle": "x"}, "essay")


# ----------------------------------------------------- summaries and compare


def test_summary_pass_at_k_and_compare_format(tmp_path, capsys):
    from palingenesis.agent_tooling.compare import load_results
    from palingenesis.agent_tooling.compare import main as compare_main
    from palingenesis.eval.runner import summarize
    from palingenesis.evalstats import pass_at_k, wilson_interval

    patterns = [[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]]
    rows = [
        {"id": f"t/{i}", "scores": [float(s) for s in scores], "finish": ["stop", "stop", "length", "stop"],
         "tokens": [10, 20, 30, 40], "loop": [False, False, i == 0, False]}
        for i, scores in enumerate(patterns)
    ]  # fmt: skip
    rows.append({"id": "t/5", "scores": [], "finish": ["error"] * 4, "tokens": [0] * 4, "loop": [False] * 4})
    s = summarize(rows, resamples=200)
    assert s["items"] == 5 and s["unscored_items"] == 1 and s["samples"] == 24 and s["unscored_samples"] == 4
    assert s["mean_score"]["estimate"] == pytest.approx(9 / 20)
    for k in (1, 2, 4):
        expected = sum(pass_at_k(4, sum(p), k) for p in patterns) / 5
        assert s[f"pass@{k}"]["estimate"] == pytest.approx(expected)
        assert s[f"pass@{k}"]["lo"] <= expected <= s[f"pass@{k}"]["hi"]
    assert s["pass@4"]["estimate"] == pytest.approx(4 / 5) and s["pass@1"]["estimate"] == s["mean_score"]["estimate"]
    lo, hi = wilson_interval(3, 5)
    assert s["wilson"] == {"successes": 3, "n": 5, "estimate": 0.6, "lo": lo, "hi": hi}  # the first sample per item
    assert s["truncated_share"] == pytest.approx(5 / 24) and s["error_share"] == pytest.approx(4 / 24)
    assert s["loop_share"] == pytest.approx(1 / 24) and s["finish"]["stop"] == pytest.approx(15 / 24)
    assert s["completion_tokens"]["mean"] == pytest.approx(500 / 24) and s["completion_tokens"]["max"] == 40
    # the scored rows are a `pgs compare` file as they are
    a = _write(tmp_path / "a.jsonl", rows[:5])
    b = _write(tmp_path / "b.jsonl", [{**r, "scores": [1.0] * 4} for r in rows[:5]])
    assert load_results(str(a))["t/1"] == {"scores": [1.0, 0.0, 0.0, 0.0], "cluster": None}
    assert compare_main([str(a), str(b), "--require", "better", "--resamples", "500", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["a"]["mean_score"]["estimate"] == pytest.approx(9 / 20)
    assert report["comparison"]["mean_score"]["diff"] == pytest.approx(11 / 20)
    assert compare_main([str(b), str(a), "--require", "not_worse", "--resamples", "500"]) == 1


# ------------------------------------------------- the pipeline, end to end


class ScriptedEngine:
    """Answers each prompt from a script: {question substring: [answer per sample, ...]}, cycling. In think mode
    (the prompt does not end with a closed think block) the answer follows a reasoning block and comes from the
    `think` script when there is one."""

    def __init__(self, tok, eot, plain, think=None, truncate=()):
        self.tok, self.eot, self.plain, self.think, self.truncate = tok, eot, plain, think or plain, truncate
        self.seen, self.version, self.samplings, self.budgets = {}, 0, [], []

    def generate(self, prompts, max_new_tokens, temperature):
        from palingenesis.opd.rollout import Rollout

        out = []
        self.samplings.append(temperature)
        self.budgets += list(max_new_tokens)
        for prompt in prompts:
            text = self.tok.decode(prompt)
            thinking = not text.rstrip().endswith("</think>")
            script = self.think if thinking else self.plain
            key = next(q for q in script if q in text)
            n = self.seen.get((key, thinking), 0)
            self.seen[(key, thinking)] = n + 1
            answer = script[key][n % len(script[key])]
            if thinking:
                answer = f"<think>\nLet me work it out.\n</think>\n\n{answer}"
            ids = self.tok.encode(answer, add_special_tokens=False)
            if key in self.truncate:
                out.append(Rollout(ids, [-0.5] * len(ids), "length", self.version))
            else:
                out.append(Rollout(ids + [self.eot], [-0.5] * (len(ids) + 1), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


@pytest.fixture
def qwen(monkeypatch):
    from palingenesis.opd.teachers import end_of_turn_id
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    eot = end_of_turn_id(tok, {"enable_thinking": False})
    monkeypatch.setattr("palingenesis.opd.trainer.student_stop_ids", lambda *a, **k: (eot,))
    monkeypatch.setattr("palingenesis.eval.runner.model_sampling", lambda model: SamplingConfig())
    return tok, eot


def _sums(tmp_path, n=4):
    names = ["alpha", "beta", "gamma", "delta", "epsilon"][:n]
    rows = [
        {"qid": name, "question": f"Question {name}: what is 3 + 4?", "solution": "3 + 4 = 7\n#### 7"} for name in names
    ]
    return {
        "name": "sums",
        "dataset": str(_write(tmp_path / "sums.jsonl", rows)),
        "id_field": "qid",
        "fields": {"answer": {"from": "solution", "regex": r"####\s*(.+)$"}},
        "prompt": '{{question}}\nEnd with "Answer: <number>".',
        "scorer": "numeric",
        "sampling": {"max_tokens": 64},
    }


def test_evaluator_both_modes_and_compare(tmp_path, qwen, capsys):
    from palingenesis.agent_tooling.compare import main as compare_main
    from palingenesis.eval.runner import Evaluator

    tok, eot = qwen
    right, wrong = "3 + 4 = 7.\nAnswer: 7", "Answer: 8"
    plain = {"alpha": [right], "beta": [right, wrong], "gamma": [wrong, wrong, wrong, right], "delta": [wrong]}
    think = {"alpha": [right], "beta": [right], "gamma": [right, wrong], "delta": [wrong, right, right, right]}
    engine = ScriptedEngine(tok, eot, plain, think, truncate=("delta",))
    task = load_task(_sums(tmp_path))
    evaluator = Evaluator(tok, engine, "Qwen/Qwen3-0.6B", engine_options={"max_model_len": 1024}, resamples=300)
    out = tmp_path / "out"
    s_plain = evaluator.run(task, "plain", k=4, out_dir=out, sampling=SamplingConfig(temperature=0.7, top_k=20))
    s_think = evaluator.run(task, "think", k=4, out_dir=out, sampling=SamplingConfig(temperature=0.7, top_k=20))
    # scores: the item mean, then over items; unbiased pass@k
    assert s_plain["items"] == 4 and s_plain["samples"] == 16 and s_plain["k"] == 4
    assert s_plain["mean_score"]["estimate"] == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)
    assert s_plain["pass@4"]["estimate"] == pytest.approx(3 / 4)
    assert s_plain["pass@2"]["estimate"] == pytest.approx((1 + 5 / 6 + 1 / 2 + 0) / 4)
    assert s_think["mean_score"]["estimate"] == pytest.approx((1 + 1 + 0.5 + 0.75) / 4)
    assert s_plain["chat_template_kwargs"] == {"enable_thinking": False} and s_think["mode"] == "think"
    assert s_plain["truncated_share"] == pytest.approx(4 / 16) and s_plain["finish"] == {"length": 0.25, "stop": 0.75}
    assert s_plain["sampling"]["temperature"] == 0.7 and s_plain["sampling"]["max_tokens"] == 64
    assert s_plain["sampling"]["temperature_source"] == "config" and s_plain["unscored_samples"] == 0
    assert engine.samplings[0].top_k == 20 and engine.samplings[0].temperature == 0.7 and max(engine.budgets) == 64
    # the files: pgs compare rows, samples with texts, a summary
    rows = [json.loads(line) for line in (out / "sums.plain.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["sums/alpha", "sums/beta", "sums/gamma", "sums/delta"]
    assert rows[1]["scores"] == [1.0, 0.0, 1.0, 0.0] and rows[3]["finish"] == ["length"] * 4
    assert set(rows[0]) == {"id", "scores", "task", "mode", "finish", "tokens", "loop", "key"}
    samples = [json.loads(line) for line in (out / "sums.think.samples.jsonl").read_text().splitlines()]
    assert len(samples) == 16 and samples[0]["reasoning"] == "Let me work it out."
    assert (
        samples[0]["completion"] == right and samples[0]["score"] == 1.0 and samples[0]["rewards"] == {"numeric": 1.0}
    )
    assert samples[0]["prompt"][0]["content"].startswith("Question alpha") and "prompt" not in samples[1]
    assert json.loads((out / "sums.think.summary.json").read_text())["mean_score"] == s_think["mean_score"]
    # the two modes are paired by item id
    code = compare_main([str(out / "sums.plain.jsonl"), str(out / "sums.think.jsonl"), "--json", "--resamples", "300",
                         "--require", "not_worse"])  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["comparison"]["paired_items"] == 4
    assert report["comparison"]["mean_score"]["diff"] == pytest.approx(0.8125 - 0.4375)
    assert report["a"]["pass@4"]["estimate"] == pytest.approx(s_plain["pass@4"]["estimate"])
    # resume keeps the finished items; without it the files start over
    calls = len(engine.samplings)
    same = SamplingConfig(temperature=0.7, top_k=20)
    again = evaluator.run(task, "plain", k=4, out_dir=out, resume=True, sampling=same)
    assert len(engine.samplings) == calls and again["mean_score"] == s_plain["mean_score"]
    # ... of the same request: another sampling is another result, generated again
    evaluator.run(task, "plain", k=4, out_dir=out, resume=True, sampling=SamplingConfig(temperature=0.5))
    assert len(engine.samplings) == calls + 1
    evaluator.close()


def test_function_calling_task_through_the_tool_parser(tmp_path, qwen):
    from palingenesis.eval.runner import Evaluator

    tok, eot = qwen
    weather = {"name": "get_weather", "description": "Weather of a city.",
               "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}, "required": ["city"]}}  # fmt: skip
    rows = [
        {"id": "w1", "question": [[{"role": "user", "content": "Weather in Rome, case one?"}]], "function": [weather],
         "ground_truth": [{"get_weather": {"city": ["Rome", "Roma"]}}]},
        {"id": "w2", "question": [[{"role": "user", "content": "Weather in Oslo, case two?"}]], "function": [weather],
         "ground_truth": [{"get_weather": {"city": ["Oslo"]}}]},
    ]  # fmt: skip
    call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "%s"}}\n</tool_call>'
    engine = ScriptedEngine(tok, eot, {"case one": [call % "roma"], "case two": ["It is sunny in Oslo."]})
    task = load_task({"task": "bfcl_simple", "name": "calls", "dataset": str(_write(tmp_path / "calls.jsonl", rows)),
                      "data_files": "", "join": {}})  # fmt: skip
    summary = Evaluator(tok, engine, "Qwen/Qwen3-0.6B", resamples=100).run(task, "plain", out_dir=tmp_path / "out")
    assert summary["mean_score"]["estimate"] == 0.5 and summary["wilson"]["successes"] == 1
    samples = [json.loads(line) for line in (tmp_path / "out" / "calls.plain.samples.jsonl").read_text().splitlines()]
    assert [s["score"] for s in samples] == [1.0, 0.0] and samples[0]["rewards"] == {"function_call": 1.0}
    assert "get_weather" in tok.decode(
        tok.apply_chat_template(samples[0]["prompt"], tools=build_items(task)[0].row["tools"])
    )


def test_cli_end_to_end(tmp_path, qwen, monkeypatch, capsys):
    from palingenesis.eval import cli

    tok, eot = qwen
    task_file = tmp_path / "sums.yaml"
    task_file.write_text(json.dumps(_sums(tmp_path)))  # JSON is YAML
    engines = []

    def fake_engine(config, model, tokenizer, stop_ids):
        assert config.rollout.backend == "hf" and config.rollout.max_model_len == 2048 and stop_ids == (eot,)
        engines.append(ScriptedEngine(tok, eot, {"alpha": ["Answer: 7"], "beta": ["Answer: 7"],
                                                  "gamma": ["Answer: 1"], "delta": ["no idea"]}))  # fmt: skip
        return engines[-1]

    monkeypatch.setattr("palingenesis.eval.runner.build_engine", fake_engine)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tok)
    out = tmp_path / "out"
    argv = ["--model", "Qwen/Qwen3-0.6B", "--tasks", str(task_file), "--modes", "plain,think", "--out", str(out),
            "--backend", "hf", "--max_model_len", "2048", "--limit", "3", "--json", "--run.resamples", "200",
            "--no_cache"]  # fmt: skip
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert [(r["task"], r["mode"], r["items"]) for r in report["results"]] == [
        ("sums", "plain", 3),
        ("sums", "think", 3),
    ]
    assert len(engines) == 1  # one engine for every task and mode
    assert report["results"][0]["sampling"]["temperature"] == 0.0  # no generation config: greedy at k = 1
    assert json.loads((out / "summary.json").read_text())["model"] == "Qwen/Qwen3-0.6B"
    assert sorted(p.name for p in out.iterdir()) == [
        "report.md", "summary.json", "sums.plain.jsonl", "sums.plain.samples.jsonl", "sums.plain.summary.json",
        "sums.think.jsonl", "sums.think.samples.jsonl", "sums.think.summary.json",
    ]  # fmt: skip
    # config errors: exit 1 with the reason, before any model is loaded
    assert cli.main(["--tasks", "gsm8k", "--out", str(out)]) == 1
    assert "model.path is empty" in capsys.readouterr().err
    assert cli.main(["--model", "m", "--tasks", "gsm9k"]) == 1
    assert "unknown task 'gsm9k'" in capsys.readouterr().err
    assert cli.main(["--model", "m", "--tasks", str(task_file), "--modes", "fast"]) == 1
    assert "unknown mode 'fast'" in capsys.readouterr().err
    assert cli.main(["--list"]) == 0 and "gsm8k" in capsys.readouterr().out


def test_unscored_items_stay_out_of_the_compare_file(tmp_path, qwen):
    from palingenesis.agent_tooling.compare import load_results
    from palingenesis.eval.runner import Evaluator

    tok, eot = qwen
    rows = [
        {"key": 1, "prompt": "Reply without commas, case one.", "instruction_id_list": ["punctuation:no_comma"],
         "kwargs": [{"num_words": None}]},
        {"key": 2, "prompt": "Reply in Italian, case two.", "instruction_id_list": ["language:response_language"],
         "kwargs": [{"language": "it"}]},
    ]  # fmt: skip
    task = load_task({"task": "ifeval", "dataset": str(_write(tmp_path / "if.jsonl", rows)),
                      "scorer_args": {"checker": "builtin"}})  # fmt: skip
    engine = ScriptedEngine(tok, eot, {"case one": ["No commas at all."], "case two": ["Ciao a tutti."]})
    summary = Evaluator(tok, engine, "Qwen/Qwen3-0.6B", resamples=100).run(task, "plain", out_dir=tmp_path / "out")
    results = load_results(str(tmp_path / "out" / "ifeval.plain.jsonl"))
    if ifeval.has_langdetect():
        assert summary["items"] == 2 and summary["unscored_items"] == 0
    else:  # the language item cannot be checked here: counted, and left out of the paired file
        assert summary["items"] == 1 and summary["unscored_items"] == 1 and summary["unscored_samples"] == 1
        assert list(results) == ["ifeval/1"] and summary["mean_score"]["estimate"] == 1.0


# --------------------------------------------------- evaluation while training


def test_eval_tasks_section_in_training_configs(tmp_path):
    from palingenesis.config import Config, ConfigError
    from palingenesis.opd.config import OPDConfig
    from palingenesis.rl.config import RLConfig

    for cls in (Config, RLConfig, OPDConfig):
        path = tmp_path / "c.yaml"
        path.write_text("eval_tasks: {tasks: [gsm8k], every: 50, limit: 100, modes: [plain, think]}\n")
        config = cls.from_yaml(path)
        assert config.eval_tasks.enabled and config.eval_tasks.every == 50 and config.eval_tasks.errors() == []
        assert not cls().eval_tasks.enabled
        path.write_text("eval_tasks: {task: [gsm8k]}\n")
        with pytest.raises(ConfigError, match="unknown option eval_tasks.task"):
            cls.from_yaml(path)
    spec = EvalTasksConfig(tasks=["gsm9k"], modes=["fast"], k=0)
    errors = " ".join(spec.errors())
    assert "k >= 1" in errors and "unknown mode 'fast'" in errors
    assert "unknown task 'gsm9k'" in " ".join(EvalTasksConfig(tasks=["gsm9k"]).errors())
    config = Config()
    config.eval_tasks.tasks = ["gsm9k"]
    with pytest.raises(ConfigError, match="unknown task 'gsm9k'"):
        config.validate()


def test_training_eval_hook(tmp_path, qwen):
    from palingenesis.agent_tooling.compare import main as compare_main
    from palingenesis.eval.training import TrainingEval

    tok, eot = qwen
    spec = EvalTasksConfig(tasks=[_sums(tmp_path)], every=10, limit=3, modes=["plain", "think"])
    hook = TrainingEval(
        spec, "Qwen/Qwen3-0.6B", tok, tmp_path / "run", every=50, engine_options={"max_model_len": 1024}
    )
    assert [hook.due(s, 25) for s in (0, 5, 10, 20, 25)] == [True, False, True, True, True]
    before = ScriptedEngine(tok, eot, {"alpha": ["Answer: 1"], "beta": ["Answer: 7"], "delta": ["Answer: 2"],
                                       "gamma": ["Answer: 3"]})  # fmt: skip
    after = ScriptedEngine(tok, eot, {q: ["Answer: 7"] for q in ("alpha", "beta", "gamma", "delta")})
    metrics = hook.run(before, 0)
    assert not hook.due(0, 25)  # once per step
    assert set(metrics) == {f"tasks/sums/{mode}/{name}" for mode in ("plain", "think")
                            for name in ("score", "truncated", "loops", "tokens")}  # fmt: skip
    assert metrics["tasks/sums/plain/score"] == pytest.approx(1 / 3)
    assert hook.run(after, 10)["tasks/sums/think/score"] == 1.0
    files = sorted(p.name for p in (tmp_path / "run" / "eval_tasks" / "step_10").iterdir())
    assert files == ["sums.plain.jsonl", "sums.plain.summary.json", "sums.think.jsonl", "sums.think.summary.json"]
    base = tmp_path / "run" / "eval_tasks"
    code = compare_main([str(base / "step_0" / "sums.plain.jsonl"), str(base / "step_10" / "sums.plain.jsonl"),
                         "--require", "not_worse", "--resamples", "200"])  # fmt: skip
    assert code == 0  # two steps of one run compare directly: the same items
    # a failing evaluation switches itself off instead of stopping the run
    broken = TrainingEval(
        EvalTasksConfig(tasks=[{**_sums(tmp_path), "fields": {"answer": "nope"}}]), "m", tok, tmp_path
    )
    assert broken.run(after, 0) == {} and broken.disabled and not broken.due(10, 25)
    with pytest.raises(EvalConfigError, match="unknown task"):
        TrainingEval(EvalTasksConfig(tasks=["gsm9k"]), "m", tok, tmp_path)


def test_shared_client_is_not_closed_by_the_evaluation(tmp_path, qwen):
    from palingenesis.eval.runner import Evaluator, rl_config
    from palingenesis.eval.tasks import build_items as items_of

    tok, eot = qwen
    engine = ScriptedEngine(tok, eot, {"alpha": ["Answer: 7"], "beta": ["Answer: 7"]})
    task = load_task(_sums(tmp_path, n=2))
    items = items_of(task)
    evaluator = Evaluator(tok, engine, "Qwen/Qwen3-0.6B")
    sampling = SamplingConfig(temperature=0.0, max_tokens=32)
    config = rl_config("Qwen/Qwen3-0.6B", task, {"enable_thinking": False}, sampling)
    owner = evaluator._pipeline(config, items, resolve_scorers(task), {"enable_thinking": False})
    try:
        evaluator.client = owner.client
        guest = evaluator._pipeline(config, items, resolve_scorers(task), {"enable_thinking": False})
        assert guest.client is owner.client and not guest.owns_client and owner.owns_client
        guest.close()
        groups = owner._await(owner._eval([i.row for i in items], 0.0, 1))  # the owner's client still works
        assert [g[0].reward for g in groups] == [1.0, 1.0]
    finally:
        owner.close()


def test_rl_config_takes_task_env_and_engine_options():
    from palingenesis.eval.runner import rl_config

    task = load_task({"name": "agent", "dataset": "d.jsonl", "system_prompt": "Be brief.", "threshold": 0.9,
                      "env": {"type": "tools", "tools": ["math:sqrt"], "max_turns": 6}, "rewards": {"ok": "exact"}})  # fmt: skip
    sampling = SamplingConfig(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, max_tokens=512)
    config = rl_config("m", task, {"enable_thinking": True}, sampling, {"max_model_len": 32768, "backend": "sglang"})
    assert (config.env.type, config.env.max_turns, config.rollout.max_model_len) == ("tools", 6, 32768)
    assert config.rollout.max_new_tokens == 512 and config.completion_budget == 512 * 6  # the whole episode
    assert config.data.system_prompt == "Be brief." and config.data.success_threshold == 0.9
    assert config.model.chat_template_kwargs == {"enable_thinking": True} and config.rollout.backend == "sglang"
    assert [r.name for r in resolve_scorers(task)] == ["ok"]
    with pytest.raises(EvalConfigError, match=r"unknown option env.max_turn \(did you mean env.max_turns\?\)"):
        rl_config("m", load_task({"name": "a", "dataset": "d", "scorer": "mcq", "env": {"max_turn": 3}}), {}, sampling)
    with pytest.raises(EvalConfigError, match="unknown option engine.max_modellen"):
        rl_config("m", load_task("gsm8k"), {}, sampling, {"max_modellen": 1})
