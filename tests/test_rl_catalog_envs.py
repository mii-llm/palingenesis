"""The built-in catalogue environments: every audit (oracle paid, degenerate policies not, reuse clean), their
verifiers on the cases that matter, the dataset converters on rows of the published shape, and CPU throughput."""

import asyncio
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")  # tool schemas come from transformers.utils.get_json_schema

from palingenesis.rl.catalog import audit, get, load_rows, names  # noqa: E402
from palingenesis.rl.catalog.audit import render  # noqa: E402
from palingenesis.rl.catalog.runner import make_pool, play, play_many  # noqa: E402

GENERATED = [name for name in names() if name != "harbor"]
# oracle episodes per second a laptop CPU must reach (measured values are 10-100x above; code runs a process per
# submission, math pays the verifier pool's start once)
FLOORS = {"code-unit-tests": 5, "code-stdio": 5, "math-boxed": 2}
THROUGHPUT: dict[str, dict] = {}


def run(coroutine):
    return asyncio.run(coroutine)


def one(name: str, row: dict, policy, max_turns: int | None = None):
    spec = get(name)
    return run(play(make_pool(spec), dict(row), policy, max_turns or spec.max_turns))


def reply(text):
    return lambda row, messages: text


# ---------------------------------------------------------------- the audits


@pytest.mark.parametrize("name", GENERATED)
def test_builtin_audit_oracle_passes_degenerate_policies_do_not(name):
    spec = get(name)
    report = audit(spec, n=28, seed=3)
    assert report["ok"], render(report)
    assert report["oracle"]["passed"] == report["rows"] == 28 and report["oracle"]["errors"] == 0
    assert {"empty", "refusal", "echo_prompt", "tool_spam"} <= set(report["degenerate"])
    assert set(spec.degenerate) <= set(report["degenerate"]) and len(spec.degenerate) >= 2
    for policy, summary in report["degenerate"].items():
        assert summary["errors"] == 0, (policy, summary["first_error"])
        assert summary["pass_rate"] <= spec.chance, (policy, summary)
    assert report["reuse"]["passed"] == report["reuse"]["episodes"]
    assert not report["holes"]


@pytest.mark.parametrize("name", GENERATED)
def test_cpu_throughput_of_the_scripted_oracle(name):
    """Steps per second of each environment on CPU, with a second pass so one-time costs (the verifier pool's
    workers, the first import of sympy, a corpus index) are not in the number."""
    import time

    spec = get(name)
    rows = load_rows(spec, n=48, seed=11)

    async def measure():
        pool = make_pool(spec)
        await play_many(pool, rows[:8], spec.oracle, spec.max_turns)  # warm-up
        start = time.perf_counter()
        episodes = await play_many(pool, rows, spec.oracle, spec.max_turns, concurrency=32)
        seconds = time.perf_counter() - start
        await pool.shutdown()
        return episodes, seconds

    episodes, seconds = run(measure())
    assert all(e.reward is not None and e.reward >= spec.success_threshold for e in episodes)
    steps = sum(e.steps for e in episodes)
    THROUGHPUT[name] = {"episodes_per_s": len(episodes) / seconds, "steps_per_s": steps / seconds}
    print(f"\n{name}: {len(episodes) / seconds:.0f} episodes/s, {steps / seconds:.0f} steps/s (oracle, CPU)")
    assert len(episodes) / seconds > FLOORS.get(name, 100), THROUGHPUT[name]


def test_generators_are_deterministic_and_rows_are_json():
    for name in GENERATED:
        spec = get(name)
        a, b, c = load_rows(spec, n=6, seed=5), load_rows(spec, n=6, seed=5), load_rows(spec, n=6, seed=6)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), name
        assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True), name
        assert len({row["id"] for row in a}) == 6 and all(row.get("prompt") for row in a), name


def test_builtin_entries_are_permissive_and_declare_what_they_need():
    for name in names():
        spec = get(name)
        assert spec.permissive, name
        assert spec.license == "Apache-2.0" and spec.source == "builtin"
        for dataset in spec.datasets.values():
            assert dataset.kind in ("generator", "hub", "file")
            if dataset.kind == "hub":
                assert dataset.license in ("MIT", "Apache-2.0", "CC-BY-4.0") and dataset.citation and dataset.url
    assert get("code-unit-tests").sandbox == get("code-stdio").sandbox == "process"
    assert get("harbor").sandbox == "docker" and get("math-boxed").sandbox == "none"
    assert get("instruction-following").eval_only and get("text-to-sql").trainable


# ----------------------------------------------------------------------- math


def test_math_forms_close_the_unevaluated_question_hole():
    from palingenesis.rl.catalog.envs import math_boxed as m

    equal = lambda p, g, form="number": run(m.answers_equal(p, g, form))  # noqa: E731
    assert equal("\\frac{3}{4}", "3/4") and equal("0.75", "3/4") and equal("-\\dfrac{6}{8}", "-3/4")
    assert equal("1,234", "1234") and equal("x = 7", "7") and equal("42.0", "42")
    assert not equal("17 \\times 23 + 5", "396")  # the question, unevaluated
    assert not equal("3/4 + 0", "3/4") and not equal("", "3") and not equal(None, "3")
    assert equal("\\left(2,\\ -3\\right)", "(2, -3)", "tuple") and not equal("(-3, 2)", "(2, -3)", "tuple")
    assert equal("2 + 7x + 6x^{2}", "6x^2 + 7x + 2", "expanded")  # sympy, in the verifier pool
    assert not equal("(2x + 1)(3x + 2)", "6x^2 + 7x + 2", "expanded")  # the factored question
    assert equal("(2x + 1)(3x + 2)", "6x^2 + 7x + 2", "expression")
    assert not equal("6x^2 + 7x + 3", "6x^2 + 7x + 2", "expanded")


def test_math_never_evaluates_code_or_runaway_powers():
    from palingenesis.rl.catalog.envs import math_boxed as m

    for text in ("__import__('os').system('true')", "x.__class__", "open(1)", "9**9**9**9", "2**100000", "lambda: 1",
                 "1" * 40, "exec"):  # fmt: skip
        assert not m.safe_expression(m.normalize(text)), text
        assert not m.symbolic_equal(m.normalize(text), "1")
    assert m.safe_expression("6*x**2 + 7*x + sqrt(2)*pi")


def test_math_only_the_last_box_counts():
    row = load_rows(get("math-boxed"), n=1)[0]
    gold = row["answer"]
    assert one("math-boxed", row, reply(f"\\boxed{{99999}} then \\boxed{{{gold}}}")).reward == 1.0
    assert one("math-boxed", row, reply(f"\\boxed{{{gold}}} or \\boxed{{99999}}")).reward == 0.0
    assert one("math-boxed", row, reply(f"The answer is {gold}.")).rewards == {"correct": 0.0, "metric/boxed": 0.0}


def test_gsm8k_rows_convert():
    from palingenesis.rl.catalog.envs.math_boxed import from_gsm8k

    row = from_gsm8k({"question": "Ann has 1,000 beads and buys 280 more. How many?", "answer": "1000 + 280 = <<1000+280=1280>>1280\n#### 1,280"})
    assert row["answer"] == "1280" and row["form"] == "number" and row["prompt"].endswith("\\boxed{}.")
    assert one("math-boxed", row, reply("\\boxed{1,280}")).reward == 1.0
    assert from_gsm8k({"question": "q", "answer": "no marker"}) is None


# ----------------------------------------------------------------------- code


def test_stdio_runs_every_input_in_one_job_with_real_file_descriptors():
    from palingenesis.rl.catalog.envs.code import grade_stdio, process_sandbox

    class Counting:
        def __init__(self, inner):
            self.inner, self.jobs = inner, 0

        async def run(self, jobs):
            self.jobs += len(jobs)
            return await self.inner.run(jobs)

    sandbox = Counting(process_sandbox())
    inputs, expected = ["1 2\n", "10 20\n", "5 5\n"], ["3", "30", "10"]
    programs = {
        "input()": "a, b = map(int, input().split())\nprint(a + b)",
        "sys.stdin.buffer": "import sys\nprint(sum(map(int, sys.stdin.buffer.read().split())))",
        "open(0)": "print(sum(map(int, open(0).read().split())))",
        "os.read": "import os\nprint(sum(map(int, os.read(0, 100).split())))",
        "exit()": "import sys\nprint(sum(map(int, input().split())))\nsys.exit(0)",
        "main guard": "def main():\n    print(sum(map(int, input().split())))\nif __name__ == '__main__':\n    main()",
    }
    for label, program in programs.items():
        assert run(grade_stdio(sandbox, program, inputs, expected)) == (1.0, 3, "passed"), label
    assert sandbox.jobs == len(programs)  # one process per submission, not per test
    assert run(grade_stdio(sandbox, "print(3)", inputs, expected))[:2] == (0.0, 1)
    assert run(grade_stdio(sandbox, "import os\nprint(3)\nos._exit(0)", inputs, expected))[0] == 0.0
    assert run(grade_stdio(sandbox, "raise SystemExit(3)", inputs, expected))[0] == 0.0
    assert run(grade_stdio(sandbox, None, inputs, expected)) == (0.0, 0, "no_code")


def test_code_limits_end_runaway_programs_as_failures_not_skips():
    from palingenesis.rl.catalog.envs.code import grade_calls, grade_stdio, process_sandbox

    sandbox = process_sandbox()
    reward, _, status = run(grade_stdio(sandbox, "while True:\n    pass", ["1\n"], ["1"], timeout=1.0))
    assert reward == 0.0 and status == "timeout"
    reward, _, status = run(grade_calls(sandbox, "def f(n):\n    while True:\n        pass", "f", [[1]], [1], timeout=1.0))
    assert reward == 0.0 and status == "timeout"
    flood = "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)"
    assert run(grade_stdio(sandbox, flood, ["1\n"], ["1"], timeout=2.0))[0] == 0.0


def test_unit_tests_compare_call_results_outside_the_sandbox():
    from palingenesis.rl.catalog.envs import code

    assert code.functional_tests(["assert f(1, [2]) == 3", "assert f('a', {'k': 1}) == [1, 2]"]) == (
        "f",
        [[1, [2]], ["a", {"k": 1}]],
        [3, [1, 2]],
    )
    assert code.functional_tests(["assert f((1, 2)) == (3, 4)"]) is None  # a tuple argument is not JSON-plain
    assert code.functional_tests(["assert f(1) == 1", "assert g(1) == 1"]) is None
    assert code.functional_tests(["assert abs(f(1) - 0.5) < 1e-6"]) is None
    assert code.functional_tests("def check(candidate):\n    assert candidate(1) == 1\n") is None
    mbpp = code.from_mbpp(
        {"prompt": "Write a function to add two numbers.", "code": "def add(a, b):\n    return a + b\n",
         "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"], "test_imports": []}  # fmt: skip
    )
    assert one("code-unit-tests", mbpp, code.oracle).reward == 1.0
    for attack in (code.rewrite_harness, code.always_equal, code.exit_zero, code.forge_report):
        assert one("code-unit-tests", {**mbpp, "entry_point": "add"}, attack).reward == 0.0, attack.__name__
    wrong = reply("```python\ndef add(a, b):\n    return 3\n```")
    assert one("code-unit-tests", mbpp, wrong).reward == 0.0


def test_humaneval_rows_run_through_the_assert_harness():
    from palingenesis.rl.catalog.envs import code

    row = code.from_humaneval(
        {"prompt": "def double(x):\n    \"\"\"Twice x.\"\"\"\n", "canonical_solution": "    return 2 * x\n",
         "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(-1) == -2\n", "entry_point": "double"}  # fmt: skip
    )
    assert one("code-unit-tests", row, code.oracle).reward == 1.0
    assert one("code-unit-tests", row, code.exit_zero).reward == 0.0
    assert one("code-unit-tests", row, code.always_equal).reward == 0.0
    assert one("code-unit-tests", row, reply("```python\ndef double(x):\n    return x\n```")).reward == 0.0


# --------------------------------------------------------------- tool calling


def test_function_call_matching_is_exact_on_calls_and_lenient_on_spelling():
    from palingenesis.rl.catalog.envs.function_call import calls_match, from_bfcl

    expected = [{"name": "get_weather", "arguments": {"city": ["Lisbon"], "days": [1, ""], "unit": ["celsius", ""]}}]
    assert calls_match([("get_weather", {"city": "lisbon"})], expected)
    assert calls_match([("get_weather", {"city": " Lisbon ", "days": 1.0})], expected)
    assert not calls_match([("get_weather", {"city": "Lisbon", "days": 2})], expected)
    assert not calls_match([("get_weather", {"city": "Lisbon", "country": "PT"})], expected)  # an argument not listed
    assert not calls_match([("get_weather", {})], expected) and not calls_match([], expected)
    assert not calls_match([("get_weather", {"city": "Lisbon"})] * 2, expected)
    assert not calls_match([("get_weather", {"city": "Lisbon", "days": True})], expected)  # a bool is not 1
    assert not calls_match([("get_weather", "Lisbon")], expected)
    two = expected + [{"name": "create_timer", "arguments": {"minutes": [5]}}]
    assert calls_match([("create_timer", {"minutes": 5}), ("get_weather", {"city": "Lisbon"})], two)
    row = from_bfcl(
        {"id": "simple_0", "question": [[{"role": "user", "content": "Area of a 3 by 4 rectangle?"}]],
         "function": [{"name": "area", "description": "Area.", "parameters": {"type": "dict", "properties": {"w": {"type": "float"}, "h": {"type": "integer"}}, "required": ["w", "h"]}}],
         "ground_truth": [{"area": {"w": [3, 3.0], "h": [4]}}]}  # fmt: skip
    )
    assert row["tools"][0]["function"]["parameters"]["type"] == "object"
    assert row["tools"][0]["function"]["parameters"]["properties"]["w"]["type"] == "number"
    assert one("tools-function-call", row, lambda r, m: [("area", {"w": 3, "h": 4})]).reward == 1.0
    assert one("tools-function-call", row, lambda r, m: [("area", {"w": 4, "h": 3})]).reward == 0.0


def test_function_call_irrelevance_rows_pay_only_abstention():
    rows = load_rows(get("tools-function-call"), n=14)
    irrelevant = [row for row in rows if row["category"] == "irrelevance"]
    assert len(irrelevant) == 2 and all(row["answer"] == [] for row in irrelevant)
    row = irrelevant[0]
    assert one("tools-function-call", row, reply("None of these tools can do that.")).reward == 1.0
    name = row["tools"][0]["function"]["name"]
    episode = one("tools-function-call", row, lambda r, m: [(name, {})])
    assert episode.reward == 0.0 and episode.finish == "env_done" and episode.turns == 1


@pytest.mark.parametrize("name", ["tools-filesystem", "tools-calendar", "tools-shop"])
def test_stateful_rewards_are_on_the_outcome_not_the_path(name):
    from palingenesis.rl.catalog.runner import turn_index

    spec = get(name)
    rows = load_rows(spec, n=40, seed=2)
    assert len({row["task"] for row in rows}) >= 4
    for row in rows[:12]:
        calls = [tuple(call) for call in row["solution"]]

        def all_at_once(r, messages, calls=calls):  # another path to the same world: every call in one turn
            if turn_index(messages) == 0:
                return calls
            return r["solution_answer"]

        acting = [c for c in calls if not c[0].startswith(("list_", "read_", "search_", "get_"))]
        episode = one(name, row, all_at_once)
        if name != "tools-filesystem" or len(acting) <= 1:  # parallel moves need their directory first
            assert episode.reward == 1.0, (row["task"], episode.messages[-1])
        if acting:  # one action short of the goal

            def partial(r, messages, calls=calls, drop=acting[-1]):
                if turn_index(messages) == 0:
                    return [c for c in calls if c != drop]
                return r["solution_answer"]

            assert one(name, row, partial).reward == 0.0, row["task"]
        elif row["expected_answer"]:
            wrong = reply(f"ANSWER: {row['expected_answer']}9")
            assert one(name, row, wrong).rewards["metric/state"] == 1.0
            assert one(name, row, wrong).reward == 0.0
            assert one(name, row, reply(f"Probably {row['expected_answer']}.")).reward == 0.0  # no ANSWER line


def test_stateful_tools_answer_bad_calls_with_errors():
    row = load_rows(get("tools-shop"), n=1)[0]
    shipped = next(oid for oid, o in row["state"]["orders"].items() if o["status"] == "shipped")
    calls = [("cancel_order", {"order_id": shipped}), ("cancel_order", {"order_id": "o-999"}),
             ("place_order", {"customer": "c-ana", "product_id": "p-1", "quantity": 10_000, "address": "x"}),
             ("get_product", {}), ("refund_everything", {})]  # fmt: skip
    episode = one("tools-shop", row, lambda r, m: calls if len(m) == 1 else "Done.")
    assert episode.tool_calls == 5 and episode.tool_errors == 5 and episode.finish == "stop"
    errors = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert "can no longer be cancelled" in errors[0] and "no order" in errors[1] and "in stock" in errors[2]
    assert "unknown tool" in errors[4]


# ---------------------------------------------------------------- instructions


def test_instruction_checks():
    from palingenesis.rl.catalog.envs.instructions import CHECKS, check, describe

    text = "<<A Title>>\nFirst paragraph with a lantern. It has two sentences.\n\n* one\n* two\n\nP.S. bye"
    yes = [
        {"id": "word_count", "relation": "at most", "n": 30}, {"id": "sentence_count", "relation": "at least", "n": 2},
        {"id": "paragraph_count", "n": 3}, {"id": "include_keywords", "keywords": ["Lantern", "title"]},
        {"id": "forbidden_words", "forbidden": ["very"]}, {"id": "keyword_frequency", "keyword": "two", "relation": "exactly", "n": 2},
        {"id": "bullet_points", "n": 2}, {"id": "title"}, {"id": "postscript"}, {"id": "end_phrase", "phrase": "p.s. BYE"},
        {"id": "letter_frequency", "letter": "a", "relation": "at least", "n": 5}, {"id": "no_commas"},
    ]  # fmt: skip
    no = [
        {"id": "word_count", "relation": "at least", "n": 30}, {"id": "paragraph_count", "n": 2},
        {"id": "include_keywords", "keywords": ["lantern", "anchor"]}, {"id": "forbidden_words", "forbidden": ["paragraph"]},
        {"id": "bullet_points", "n": 3}, {"id": "lowercase"}, {"id": "uppercase"}, {"id": "quotation"},
        {"id": "sections", "n": 1}, {"id": "highlighted", "n": 1}, {"id": "placeholders", "n": 1},
        {"id": "capital_words", "relation": "at least", "n": 2}, {"id": "unknown_check"}, {"id": "word_count"},
    ]  # fmt: skip
    assert all(check(text, c) for c in yes), [c for c in yes if not check(text, c)]
    assert not any(check(text, c) for c in no), [c for c in no if check(text, c)]
    assert check("SECTION 1\nA *b* [c] D.\nSECTION 2\nE *f* [g] HH II.", {"id": "sections", "n": 2})
    assert check("ALL CAPS HERE", {"id": "uppercase"}) and check("all lower. here", {"id": "lowercase"})
    assert not check("", {"id": "no_commas"}) and not check("", {"id": "forbidden_words", "forbidden": ["x"]})
    for name in CHECKS:  # every check has a description
        arguments = {"relation": "at least", "n": 2, "keywords": ["a"], "forbidden": ["a"], "keyword": "a", "letter": "a", "phrase": "a"}
        assert describe({"id": name, **arguments})


def test_ifeval_rows_convert_or_drop():
    from palingenesis.rl.catalog.envs.instructions import from_ifeval

    row = from_ifeval(
        {"key": 1000, "prompt": "Write 300+ words. No commas.",
         "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words", "detectable_format:multiple_sections"],
         "kwargs": [{}, {"relation": "at least", "num_words": 300}, {"section_spliter": "Part", "num_sections": 2}]}  # fmt: skip
    )
    assert row["constraints"] == [
        {"id": "no_commas"},
        {"id": "word_count", "relation": "at least", "n": 300},
        {"id": "sections", "n": 2, "marker": "Part"},
    ]
    text = "Part 1\n" + "word " * 200 + "\nPart 2\n" + "word " * 120
    assert one("instruction-following", row, reply(text)).reward == 1.0
    assert one("instruction-following", row, reply(text + ", and more")).rewards["metric/constraints"] == pytest.approx(2 / 3)
    unsupported = {"key": 1, "prompt": "p", "instruction_id_list": ["language:response_language"], "kwargs": [{"language": "it"}]}
    assert from_ifeval(unsupported) is None
    assert from_ifeval({"key": 2, "prompt": "p", "instruction_id_list": ["keywords:existence"], "kwargs": [{}]}) is None


# ------------------------------------------------------------------------ sql


def test_sql_is_read_only_bounded_and_graded_on_the_hidden_twin():
    from palingenesis.rl.catalog.envs import sql

    row = load_rows(get("text-to-sql"), n=1, seed=4)[0]
    table = sql._first_table(row)
    connection = sql.open_database(row["schema"])
    before = sql.run_query(connection, f"SELECT COUNT(*) FROM {table}")[1]
    for statement in (f"DELETE FROM {table}", f"DROP TABLE {table}", f"UPDATE {table} SET id = 0", "PRAGMA writable_schema = 1",
                      "ATTACH DATABASE '/tmp/pgs_should_not_exist.db' AS other", "CREATE TABLE t (x)",
                      "SELECT load_extension('x')", "SELECT 1; SELECT 2", ""):  # fmt: skip
        with pytest.raises(sql.SqlError):
            sql.run_query(connection, statement)
    with pytest.raises(sql.SqlError, match="too long"):
        sql.run_query(connection, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")
    assert sql.run_query(connection, f"SELECT COUNT(*) FROM {table}")[1] == before
    connection.close()
    visible_only = one("text-to-sql", row, sql.hardcode_visible)
    assert visible_only.rewards == {"correct": 0.0, "metric/valid": 1.0, "metric/visible": 1.0}
    assert one("text-to-sql", row, reply(row["gold_sql"])).reward == 1.0  # a bare query is a reply too
    assert sql.results_equal([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=False)
    assert not sql.results_equal([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=True)
    assert sql.results_equal([(0.1 + 0.2,)], [(0.3,)], True) and not sql.results_equal([(1,)], [(1,), (1,)], False)
    assert sql.is_ordered("SELECT a FROM t ORDER BY a") and not sql.is_ordered("SELECT (SELECT a FROM t ORDER BY a LIMIT 1)")


def test_sql_tool_shows_rows_and_parallel_calls_do_not_deadlock():
    row = load_rows(get("text-to-sql"), n=1, seed=4)[0]
    from palingenesis.rl.catalog.envs.sql import _first_table

    queries = [("run_sql", {"query": f"SELECT * FROM {_first_table(row)}"})] * 6 + [("run_sql", {"query": "SELEC"})]
    episode = one("text-to-sql", row, lambda r, m: queries if len(m) == 1 else f"```sql\n{r['gold_sql']}\n```")
    outputs = [m["content"] for m in episode.messages if m["role"] == "tool"]
    assert episode.reward == 1.0 and episode.tool_errors == 1 and " | " in outputs[0] and outputs[-1].startswith("Error:")


def test_gretel_rows_convert_when_they_run_in_sqlite():
    from palingenesis.rl.catalog.envs.sql import from_gretel

    context = "CREATE TABLE sales (id INT, region TEXT, amount INT); INSERT INTO sales VALUES (1, 'north', 5), (2, 'south', 7);"
    row = from_gretel({"sql_prompt": "Total amount in the north?", "sql_context": context, "sql": "SELECT SUM(amount) FROM sales WHERE region = 'north';"})
    assert row["gold_sql"].endswith("'north'") and "CREATE TABLE sales" in row["prompt"] and "INSERT" not in row["prompt"]
    assert one("text-to-sql", row, reply("```sql\nSELECT SUM(amount) FROM sales WHERE region='north'\n```")).reward == 1.0
    assert from_gretel({"sql_prompt": "q", "sql_context": context, "sql": "DELETE FROM sales"}) is None
    assert from_gretel({"sql_prompt": "q", "sql_context": "CREATE TABLE t (x JSONB[]) PARTITION BY x;", "sql": "SELECT 1"}) is None
    assert from_gretel({"sql_prompt": "q", "sql_context": context, "sql": "SELECT id FROM sales WHERE amount > 99"}) is None


# ----------------------------------------------------------------- extraction


def test_extraction_schema_validation_and_field_comparison():
    from palingenesis.rl.catalog.envs.extraction import compare, validate

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "n": {"type": "integer", "minimum": 0, "maximum": 9},
            "price": {"type": ["number", "null"]},
            "kind": {"type": "string", "enum": ["a", "b"]},
            "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "x-unordered": True},
        },
        "required": ["name", "n"],
        "additionalProperties": False,
    }
    good = {"name": "x", "n": 3, "price": None, "kind": "a", "date": "2031-01-02", "tags": ["p", "q"]}
    assert validate(good, schema) == []
    for bad in ({**good, "n": "3"}, {**good, "n": True}, {**good, "n": 12}, {**good, "kind": "c"}, {**good, "date": "2 Jan"},
                {**good, "tags": []}, {**good, "extra": 1}, {"n": 1}, {**good, "price": "free"}, {**good, "name": ""}, [good]):  # fmt: skip
        assert validate(bad, schema), bad
    assert compare({**good, "name": " X ", "tags": ["q", "P"]}, good, schema) == (7, 7)
    assert compare({**good, "price": 0}, good, schema)[0] == 6 and compare({**good, "n": 3.0}, good, schema)[0] == 7
    assert compare({**good, "tags": ["p"]}, good, schema)[0] == 5 and compare("nope", good, schema)[0] == 0
    row = load_rows(get("json-extraction"), n=1)[0]
    raw = one("json-extraction", row, reply(json.dumps(row["answer"])))  # bare JSON, no fence
    assert raw.reward == 1.0 and raw.rewards["metric/fields"] == 1.0
    prose = one("json-extraction", row, reply("The answer: " + json.dumps(row["answer"])))
    assert prose.rewards == {"correct": 0.0, "metric/parsed": 0.0, "metric/valid": 0.0, "metric/fields": 0.0}


# ---------------------------------------------------------------------- games


def test_wordle_feedback_handles_repeated_letters_and_a_solver_wins_without_the_gold():
    from palingenesis.rl.catalog.envs import games

    assert len(games.WORDS) == len(set(games.WORDS)) > 400 and all(len(w) == 5 and w.isalpha() for w in games.WORDS)
    assert games.feedback("crane", "crane") == "GGGGG" and games.feedback("zzzzz", "crane") == "_____"
    assert games.feedback("speed", "erase") == "Y_YY_"
    assert games.feedback("geese", "those") == "___GG" and games.feedback("allow", "label") == "YYY__"
    spec = get("wordle")
    rows = load_rows(spec, n=40, seed=9)

    async def play_all():
        return await play_many(make_pool(spec), rows, games.wordle_solver, 8)

    episodes = run(play_all())
    assert sum(e.reward == 1.0 for e in episodes) >= 34  # played through <guess> tags, from the marks alone
    assert all(m["role"] != "tool" for e in episodes for m in e.messages)  # the text protocol answers in user turns
    row = rows[0]
    invalid = one("wordle", row, lambda r, m: [("guess", {"word": "toolongword"})])
    assert invalid.tool_errors == 8 - 1 and invalid.rewards["metric/guesses"] == 0.0 and invalid.finish == "turns"
    lost = one("wordle", {**row, "answer": "crane"}, lambda r, m: [("guess", {"word": "slate"})])
    assert lost.finish == "env_done" and lost.rewards == {"solved": 0.0, "metric/guesses": 6.0, "metric/greens": 0.4}


def test_number_guess_budget_needs_bisection():
    from palingenesis.rl.catalog.envs import games

    spec = get("number-guess")
    for limit in (100, 1000, 100000):
        rows = load_rows(spec, n=20, seed=1, limit=limit)

        async def play_all():
            return await play_many(make_pool(spec), rows, games.bisect, 30)

        assert all(e.reward == 1.0 for e in run(play_all())), limit
    row = load_rows(spec, n=1)[0]
    told = one("number-guess", row, reply(f"It is {row['secret']}."))
    assert told.reward == 0.0
    tagged = one("number-guess", row, reply(f"<guess>{row['secret']}</guess>"))
    assert tagged.reward == 1.0 and tagged.finish == "env_done"


# --------------------------------------------------------------------- search


def test_corpus_search_finds_documents_and_answers_need_exact_match(tmp_path):
    from palingenesis.rl.catalog.envs import search

    world = search.build_world(7, 150)
    index = search.Index(world["documents"])
    assert len(world["documents"]) == 150 and len({d["title"] for d in world["documents"]}) == 150
    hits = sum(index.search(d["title"], 1)[0][0] == d["id"] for d in world["documents"])
    assert hits == 150  # an entity's own document is the first hit for its name
    assert index.search("zzz qqq", 3) == [] and search.corpus_index({"seed": 7, "size": 150}) is search.corpus_index({"seed": 7, "size": 150})
    rows = load_rows(get("corpus-qa"), n=12, seed=7)
    assert {row["hops"] for row in rows} == {1, 2}
    row = rows[0]
    assert one("corpus-qa", row, reply(f"ANSWER: The {row['answer'].upper()}.")).reward == 1.0
    assert one("corpus-qa", row, reply(f"ANSWER: {row['answer']} or something else")).reward == 0.0
    assert one("corpus-qa", row, reply(f"It is {row['answer']}.")).reward == 0.0
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text(
        json.dumps({"id": "a", "title": "Kettle", "text": "The kettle was invented in Brindmoor."}) + "\n"
        + json.dumps({"id": "b", "title": "Lamp", "text": "The lamp was invented in Oskvale."}) + "\n"
    )  # fmt: skip
    own = {"prompt": "Where was the lamp invented?", "answer": ["Oskvale", "Oskvale town"], "corpus": str(corpus)}

    def policy(r, messages):
        if len(messages) == 1:
            return [("search", {"query": "lamp invented"})]
        assert messages[-1]["content"].startswith("[b] Lamp")
        return [("read", {"doc_id": "[b]"})] if len(messages) == 3 else "ANSWER: oskvale"

    episode = one("corpus-qa", own, policy)
    assert episode.rewards == {"correct": 1.0, "metric/searched": 1.0} and "Oskvale" in episode.messages[-2]["content"]


def test_throughput_table_is_complete():
    """Printed with -s: the table the guide quotes."""
    if len(THROUGHPUT) < len(GENERATED):
        pytest.skip("run the whole file for the table")
    for name, numbers in sorted(THROUGHPUT.items()):
        print(f"{name:<24}{numbers['episodes_per_s']:>10.0f} episodes/s{numbers['steps_per_s']:>10.0f} steps/s")
