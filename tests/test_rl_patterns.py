"""Patterns (palingenesis.rl.scenarios.library): multi-role scenarios over an ordinary single-agent RL dataset,
scored by the run's `rewards:` functions. Each pattern end to end with scripted partners; the balance of the
checker's candidates; prompts built from the row's fields, in two languages, and the errors of a row that lacks
one; rewards taken from the config; the audit on a dataset; trained sessions through the rollout loop; the
Harbor-backed plan_execute on a stand-in environment; `pgs scenario patterns` and `pgs scenario check`."""

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_rl import tiny_policy, tokenizer  # noqa: E402
from test_rl_scenarios import _pipeline  # noqa: E402

from palingenesis.rl.config import RLConfig, RLConfigError  # noqa: E402
from palingenesis.rl.rewards import resolve_rewards  # noqa: E402
from palingenesis.rl.scenarios import build_scenario, play  # noqa: E402
from palingenesis.rl.scenarios.base import PATTERNS, tagged_text  # noqa: E402
from palingenesis.rl.scenarios.library.solve_check import SolveCheck, numeric_slips  # noqa: E402
from palingenesis.rl.scenarios.runtime import ScenarioError  # noqa: E402


def configured_rewards(**functions):
    """Rewards as a config's `rewards:` section gives them: {name: built-in or module:function}."""
    config = RLConfig()
    for name, fn in functions.items():
        spec = fn if isinstance(fn, dict) else {"fn": fn}
        for key, value in spec.items():
            config.set(f"rewards.{name}.{key}", value)
    return resolve_rewards(config.rewards)


def pattern(name, args=None, rows=None, **functions):
    scenario = build_scenario(name, args)
    scenario.bind(configured_rewards(**(functions or {"correct": "math"})))
    scenario.prepare(rows or [])
    return scenario


def math_rows(n=6):
    return [
        {
            "id": f"m{i}",
            "question": f"What is {i + 2} times {i + 13}?" if i % 2 == 0 else f"Quanto fa {i + 2} per {i + 13}?",
            "answer": str((i + 2) * (i + 13)),
            "language": "en" if i % 2 == 0 else "it",
        }
        for i in range(n)
    ]


def reading_rows(n=4):
    cities = ["Lima", "Oslo", "Cairo", "Hanoi"]
    return [
        {
            "id": f"r{i}",
            "context": f"Report {i}. "
            + "The harbour was busy all year. " * 30
            + f"The office moved to {cities[i]} in 1987.",
            "question": "Where did the office move to?",
            "answer": cities[i],
        }
        for i in range(n)
    ]


def oracles(scenario):
    return {role: table["oracle"] for role, table in scenario.partners().items()}


def by_role(episode, role):
    return [s for s in episode.sessions if s.role == role]


# ------------------------------------------------------------------ the runtime's two additions


def test_a_partner_can_read_the_row_and_a_tool_can_carry_its_schema():
    from palingenesis.rl.scenarios import Scenario
    from palingenesis.rl.scenarios.base import user
    from palingenesis.rl.scenarios.runtime import SessionPort

    class Ask(Scenario):
        name, roles = "ask", ("a",)

        async def run(self, episode, row):
            (await episode.run("a", [user("?")])).reward("x", 1.0)

    seen = []
    play(Ask(), [{"id": 1, "answer": "42"}], {"a": lambda messages, row: seen.append(row["answer"]) or "ok"})
    assert seen == ["42"]

    async def bash(**arguments):
        return "ran"

    bash.schema = {"type": "function", "function": {"name": "bash", "description": "Run.", "parameters": {}}}
    assert SessionPort({"bash": bash}, False).schemas == [bash.schema]
    assert tagged_text("so\n**ANSWER:** 3/4 of it\n", "ANSWER") == "3/4 of it"
    assert tagged_text("ANSWER: 1\nANSWER: 2", "ANSWER") is None and tagged_text("ANSWER:", "ANSWER") is None


# ------------------------------------------------------------------ solve_check


def test_solve_check_runs_on_a_plain_math_dataset_with_the_configured_rewards():
    rows = math_rows()
    scenario = pattern("solve_check", rows=rows)
    for row, (episode,) in zip(rows, play(scenario, rows, oracles(scenario))):
        solvers, checkers = by_role(episode, "solver"), by_role(episode, "checker")
        assert len(solvers) == 2 and len(checkers) == 4  # two attempts; two candidates x two checker samples
        assert all(s.rewards["correct"] == 1.0 and s.rewards["metric/answered"] == 1.0 for s in solvers)
        assert solvers[0].rewards["metric/checked"] == 1.0 and solvers[0].rewards["metric/team"] == 1.0
        assert solvers[0].rewards["metric/twin_slip"] == 1.0 and solvers[0].rewards["metric/twin_gold"] == 0.0
        assert all(c.rewards == {"verdict": 1.0} for c in checkers)
        assert {c.key[1] for c in checkers} == {"own", "twin"}  # verdicts on different candidates are not compared
        shown = [entry["delivered"] for entry in episode.log]
        assert len(shown) == 2 and row["answer"] in shown and all(text.lstrip("-").isdigit() for text in shown)
        # the prompts are the row's own text in the row's language, and the candidate is the final answer alone
        assert solvers[0].messages[0]["content"].startswith(row["question"])
        assert ("Termina la risposta" if row["language"] == "it" else "End your reply") in solvers[0].messages[0][
            "content"
        ]
        assert row["question"] in checkers[0].messages[0]["content"]
        assert ("La sua risposta finale" if row["language"] == "it" else "Their final answer") in checkers[0].messages[
            0
        ]["content"]


def wrong_solver(messages, row):
    return f"ANSWER: {int(row['answer']) + 7}"


def half_right_solver():
    turn = itertools.count()
    return lambda messages, row: f"ANSWER: {int(row['answer']) + 7 * (next(turn) % 2)}"


@pytest.mark.parametrize("solver", ["oracle", "wrong", "silent", "half"])
def test_no_constant_verdict_earns_anything_whatever_the_solver_does(solver):
    rows = math_rows()
    scenario = pattern("solve_check", rows=rows)
    table = scenario.partners()
    solvers = {
        "oracle": table["solver"]["oracle"],
        "wrong": wrong_solver,
        "silent": table["solver"]["silent"],
        "half": half_right_solver(),
    }
    for name, expected in (("always_accept", 0.0), ("always_reject", -0.5), ("silent", -2.0), ("echo", -0.5)):
        partners = {"solver": solvers[solver], "checker": table["checker"][name]}
        for (episode,) in play(scenario, rows, partners):
            per_candidate = {}
            for checker in by_role(episode, "checker"):
                per_candidate.setdefault(checker.key[1], checker.rewards["verdict"])
            assert len(per_candidate) == 2 and sum(per_candidate.values()) == expected, (solver, name)
    # the oracle checker earns in full in every case, and the pair ends right
    for (episode,) in play(scenario, rows, {"solver": solvers[solver], "checker": table["checker"]["oracle"]}):
        assert all(c.rewards["verdict"] == 1.0 for c in by_role(episode, "checker"))
        first = by_role(episode, "solver")[0]
        assert first.rewards["metric/team"] == (0.0 if solver == "silent" else 1.0) or solver == "silent"


def test_the_twin_is_another_attempt_when_one_is_wrong_and_the_gold_when_the_solver_is():
    rows = math_rows(4)
    scenario = pattern("solve_check", rows=rows)
    checker = scenario.partners()["checker"]["oracle"]
    for row, (episode,) in zip(rows, play(scenario, rows, {"solver": half_right_solver(), "checker": checker})):
        first, second = by_role(episode, "solver")
        assert (first.rewards["correct"], second.rewards["correct"]) == (1.0, 0.0)
        assert first.rewards["metric/twin_sibling"] == 1.0
        assert sorted(e["delivered"] for e in episode.log) == sorted([row["answer"], str(int(row["answer"]) + 7)])
    for row, (episode,) in zip(rows, play(scenario, rows, {"solver": wrong_solver, "checker": checker})):
        first = by_role(episode, "solver")[0]
        assert first.rewards["correct"] == 0.0 and first.rewards["metric/twin_gold"] == 1.0
        assert first.rewards["metric/team"] == 1.0  # the checker's correction is the pair's answer


def test_wrong_twins_fall_back_to_distractors_and_other_rows_and_are_checked_by_the_rewards():
    assert numeric_slips("x+1", None) == [] and numeric_slips("3.5", None) == []
    import random

    slips = numeric_slips("120", random.Random(0))
    assert set(slips) == {"119", "121", "110", "130", "21"} and "120" not in slips

    words = [
        {"id": "w0", "prompt": "Capital of Peru?", "answer": "Lima", "distractors": ["lima", "Quito"]},
        {"id": "w1", "prompt": "Capital of Norway?", "answer": "Oslo"},
        {"id": "w2", "prompt": "Capital of Egypt?", "answer": "Cairo"},
    ]
    scenario = pattern("solve_check", rows=words, correct="exact")
    episodes = [e for (e,) in play(scenario, words, oracles(scenario))]
    first = by_role(episodes[0], "solver")[0]
    # "lima" passes the (case-insensitive) reward, so it is not a wrong answer: the next distractor is used
    assert first.rewards["metric/twin_distractor"] == 1.0 and "Quito" in [e["delivered"] for e in episodes[0].log]
    for episode, row in zip(episodes[1:], words[1:]):
        assert by_role(episode, "solver")[0].rewards["metric/twin_other_row"] == 1.0
        other = next(e["delivered"] for e in episode.log if e["delivered"] != row["answer"])
        assert other in {w["answer"] for w in words} - {row["answer"]}

    # no wrong candidate at all (one row, a text answer): no checker session, and the episode says so
    alone = pattern("solve_check", rows=words[1:2], correct="exact")
    ((episode,),) = play(alone, words[1:2], oracles(alone))
    assert by_role(episode, "checker") == [] and by_role(episode, "solver")[0].rewards["metric/checked"] == 0.0


def test_a_candidate_cut_by_the_channel_is_scored_again():
    rows = [
        {"id": "long", "prompt": "Name it.", "answer": "a very long reference answer"},
        {"id": "b", "prompt": "?", "answer": "b"},
    ]
    scenario = pattern("solve_check", {"max_candidate_chars": 6}, rows=rows, correct="exact")
    ((episode,),) = play(scenario, rows[:1], oracles(scenario))
    # the cut answer is no longer right: the pair would be two wrong answers, so the checker is not asked
    assert by_role(episode, "checker") == [] and any(e["cut"] for e in episode.log)


# ------------------------------------------------------------------ the audit


class RejectionsAlwaysPay(SolveCheck):
    """A broken verifier: it pays every rejection, checked or not."""

    async def verdict_reward(self, text, candidate_right, row):
        reward, verdict, corrected = await super().verdict_reward(text, candidate_right, row)
        return (1.0 if verdict is False else reward), verdict, corrected


def test_the_audit_runs_on_your_rows_and_catches_a_degenerate_checker():
    rows = math_rows(8)
    report = pattern("solve_check").audit(rows=rows)
    assert report["failures"] == [] and report["oracle"] == {"solver": 1.0, "checker": 1.0}
    assert (
        report["degenerate"]["checker/always_accept"] == 0.0 and report["degenerate"]["checker/always_reject"] == -0.25
    )
    assert all(value <= 0 for value in report["degenerate"].values())

    broken = RejectionsAlwaysPay()
    broken.bind(configured_rewards(correct="math"))
    report = broken.audit(rows=rows)
    assert report["degenerate"]["checker/always_reject"] == 1.0
    assert any("degenerate checker/always_reject" in failure for failure in report["failures"])

    # rewards that accept every answer leave no wrong candidate: nothing pays the checker, and the audit says so
    lenient = pattern("solve_check", correct={"fn": "regex", "args": {"pattern": "."}})
    report = lenient.audit(rows=rows)
    assert any(failure.startswith("oracle checker") for failure in report["failures"])

    # a gold column that holds a worked solution is not what the math reward compares with: the oracle is unpaid
    worked = [{**row, "answer": f"We multiply. The result is {row['answer']}."} for row in rows]
    report = pattern("solve_check").audit(rows=worked)
    assert any(failure.startswith("oracle solver") for failure in report["failures"])

    with pytest.raises(NotImplementedError, match="it has no generator"):
        pattern("solve_check").audit(n=5)


# ------------------------------------------------------------------ prompts and row fields


def test_prompts_are_read_like_single_agent_rows_and_a_missing_field_fails_clearly():
    scenario = pattern("solve_check")
    chat = {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}],
        "answer": "4",
    }
    assert [m["role"] for m in scenario.solver_messages(chat)] == ["system", "user"]
    assert scenario.solver_messages(chat)[1]["content"].startswith("2+2?\n\nEnd your reply with exactly one line")
    for field in ("prompt", "question", "problem"):
        assert scenario.task({field: " 2+2? ", "answer": "4"}) == "2+2?"
    assert scenario.task({"prompt": json.dumps(chat["messages"]), "answer": "4"}) == "2+2?"  # a chat stored as JSON
    assert scenario.language({"language": "IT"}) == "it" and scenario.language({"language": "fr"}) == "en"
    assert pattern("solve_check", {"language": "it"}).language({}) == "it"
    assert scenario.gold({"answer": ["Lima", "lima"]}) == "Lima" and scenario.gold({"answer": ""}) is None

    # data.prompt_field and data.system_prompt of the run apply, as in single-agent RL
    config = RLConfig()
    config.set("data.prompt_field", "task")
    config.set("data.system_prompt", "You are careful.")
    config.set("data.success_threshold", 0.8)
    bound = build_scenario("solve_check")
    bound.bind(configured_rewards(correct="math"), None, config)
    messages = bound.solver_messages({"task": "2+2?", "question": "ignored", "answer": "4"})
    assert messages[0] == {"role": "system", "content": "You are careful."} and messages[1]["content"].startswith(
        "2+2?"
    )
    assert bound.pass_mark == 0.8 and pattern("solve_check", {"threshold": 1.0}).pass_mark == 1.0

    # no protocol: the row's prompt is untouched and the whole reply is the answer
    free = pattern("solve_check", {"answer_tag": ""})
    assert free.solver_messages({"prompt": "2+2?"}) == [{"role": "user", "content": "2+2?"}]
    assert (
        free.final_answer(" \\boxed{4} ") == "\\boxed{4}"
        and free.correction("VERDICT: REJECT\n\\boxed{5}") == "\\boxed{5}"
    )
    assert "followed in that case by the right answer" in free.checker_prompt({"prompt": "2+2?"}, "4")

    (problem,) = scenario.check_row({"id": "x1", "text": "2+2?", "answer": "4"})
    assert "solve_check: row 'x1'" in problem and "no prompt" in problem and "['answer', 'id', 'text']" in problem
    assert "scenario.args.prompt_field" in problem
    assert (
        scenario.check_row({"text": "2+2?"})
        and pattern("solve_check", {"prompt_field": "text"}).check_row({"text": "2+2?"}) == []
    )
    (problem,) = scenario.check_row(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    )
    assert "must end with a user message" in problem
    with pytest.raises(ScenarioError, match="no prompt"):
        play(scenario, [{"id": "x2", "answer": "4"}], oracles(scenario))

    handoff = pattern("handoff", correct="qa_match")
    (problem,) = handoff.check_row({"id": "h1", "question": "Where?", "answer": "Lima"})
    assert "no text in the field 'context'" in problem and "scenario.args.material_field" in problem
    (problem,) = handoff.check_row({"id": "h2", "context": "text", "answer": "Lima"})
    assert "scenario.args.question_field" in problem
    assert (
        pattern("handoff", {"material_field": "passage", "question_field": "q"}).check_row({"passage": "p", "q": "?"})
        == []
    )
    with pytest.raises(ValueError, match="question_field"):
        build_scenario("handoff", {"prompt_field": "q"})
    with pytest.raises(ValueError, match="language must be one of"):
        build_scenario("solve_check", {"language": "fr"})
    with pytest.raises(ScenarioError, match="none are bound"):
        play(build_scenario("solve_check"), math_rows(1), oracles(build_scenario("solve_check")))


def call_count(completion, answer, calls=[], **row):  # noqa: B006 — the list is the test's counter
    calls.append(completion)
    return float(completion.strip() == str(answer))


def test_answers_are_scored_by_the_rewards_of_the_config_and_by_nothing_else():
    rows = [{"id": "d", "prompt": "Half of nine?", "answer": "4.5", "target": "4,5"}]
    replies = {
        "solver": lambda messages: "Half is four and a half.\nANSWER: 4.50",
        "checker": lambda messages: "VERDICT: ACCEPT",
    }
    outcomes = {}
    for label, functions in {
        "math": {"correct": "math"},  # 4.50 equals 4.5
        "exact": {"correct": "exact"},  # the strings differ
        "exact_other_column": {"correct": {"fn": "exact", "args": {"field": "target"}}},
        "weighted": {
            "correct": {"fn": "math", "weight": 0.5},
            "format": {"fn": "regex", "weight": 0.25, "args": {"pattern": r"^\d"}},
        },
        "custom": {"mine": f"{__name__}:call_count"},
    }.items():
        scenario = pattern("solve_check", rows=rows, **functions)
        ((episode,),) = play(scenario, rows, replies)
        outcomes[label] = by_role(episode, "solver")[0].rewards
    assert outcomes["math"]["correct"] == 1.0 and outcomes["exact"]["correct"] == 0.0
    assert outcomes["exact_other_column"]["correct"] == 0.0
    assert outcomes["weighted"]["correct"] == 0.75  # the weighted sum, as in single-agent RL; each reward is logged
    assert outcomes["weighted"]["metric/correct"] == 1.0 and outcomes["weighted"]["metric/format"] == 1.0
    assert outcomes["custom"]["correct"] == 0.0 and "4.50" in call_count.__defaults__[0]  # the final answer alone

    # a reward that cannot score (SkipSample) leaves the session out of the loss instead of scoring it 0
    def outage(completion, **row):
        from palingenesis.rl.rewards import SkipSample

        raise SkipSample("judge down")

    scenario = build_scenario("lead_workers")
    scenario.bind(resolve_rewards({}, {"judge": outage}))
    ((episode,),) = play(scenario, rows, oracles(scenario))
    assert all(scenario.reward(s.role, s.rewards) is None for s in episode.sessions)


# ------------------------------------------------------------------ handoff


def test_handoff_pays_the_writer_for_what_its_note_adds():
    rows = reading_rows()
    scenario = pattern("handoff", {"max_note_chars": 200}, correct="qa_match")
    for row, (episode,) in zip(rows, play(scenario, rows, oracles(scenario))):
        (writer,), readers = by_role(episode, "writer"), by_role(episode, "reader")
        assert (
            row["context"][:50] in writer.messages[0]["content"]
            and row["question"] not in writer.messages[0]["content"]
        )
        assert len(readers) == 3 and [r.key[1] for r in readers].count("control") == 1
        for reader in readers:
            text = reader.messages[0]["content"]
            assert row["question"] in text and row["context"][:50] not in text  # never the material
            if reader.key[1] == "control":
                assert "(no notes)" in text and reader.rewards == {"metric/control": 0.0}
                assert scenario.reward("reader", reader.rewards) is None  # logged, not trained
            else:
                assert row["answer"] in text and reader.rewards["correct"] == 1.0
        assert writer.rewards["readers"] == 1.0 and writer.rewards["lift"] == 1.0
        assert scenario.reward("writer", writer.rewards) == 1.0

    table = scenario.partners()

    def knows_anyway(messages, row):
        return f"ANSWER: {row['answer']}"

    # a reader that needs no note is paid for its answers, and the writer for nothing
    for (episode,) in play(scenario, rows, {"writer": table["writer"]["filler"], "reader": knows_anyway}):
        (writer,) = by_role(episode, "writer")
        assert writer.rewards["readers"] == 1.0 and writer.rewards["lift"] == 0.0
        assert scenario.reward("writer", writer.rewards) == 0.0
    assert pattern("handoff", correct="qa_match").audit(rows=rows)["failures"] == []
    report = scenario.audit(rows=rows)
    assert report["failures"] == [] and report["degenerate"]["writer/silent"] == 0.0

    # without the control the same writer is paid in full: the option's documented cost
    plain = pattern("handoff", {"control": False}, correct="qa_match")
    ((episode,),) = play(plain, rows[:1], {"writer": table["writer"]["filler"], "reader": knows_anyway})
    (writer,) = by_role(episode, "writer")
    assert len(by_role(episode, "reader")) == 2 and plain.reward("writer", writer.rewards) == 1.0
    assert plain.audit(rows=rows)["failures"] == []

    # a note that is cut loses what was cut, and the check warns when the material fits in the note
    tight = pattern("handoff", {"max_note_chars": 3}, correct="qa_match")
    ((episode,),) = play(tight, rows[2:3], oracles(tight))
    assert by_role(episode, "writer")[0].rewards["metric/note_cut"] == 1.0
    assert by_role(episode, "writer")[0].rewards["readers"] == 0.0
    assert (
        scenario.advice(rows) == []
        and "best note is a copy" in pattern("handoff", {"max_note_chars": 5000}).advice(rows)[0]
    )
    assert "no 'answer' column" in scenario.advice([{k: v for k, v in rows[0].items() if k != "answer"}])[0]


# ------------------------------------------------------------------ lead_workers


def test_lead_workers_delegates_merges_and_credits_each_stage_in_its_own_group():
    rows = math_rows(4)
    scenario = pattern("lead_workers", rows=rows)
    for row, (episode,) in zip(rows, play(scenario, rows, oracles(scenario))):
        leads, workers = by_role(episode, "lead"), by_role(episode, "worker")
        slots = [s.key[1] for s in leads]
        assert slots.count("plan") == 1 and slots.count("merge") == 4 and slots.count("control") == 1
        assert len(workers) == 6 and sorted(w.key[1] for w in workers) == ["0", "0", "1", "1", "2", "2"]
        plan = next(s for s in leads if s.key[1] == "plan")
        assert plan.messages[0]["content"].startswith(row["question"]) and "SUBTASK:" in plan.messages[0]["content"]
        assert plan.rewards["total"] == 1.0 and plan.rewards["lift"] == 1.0 and plan.rewards["metric/subtasks"] == 3.0
        assert scenario.reward("lead", plan.rewards) == 1.0  # lift is logged with weight 0
        merges = [s for s in leads if s.key[1] == "merge"]
        assert all(m.rewards["correct"] == 1.0 for m in merges)
        # a merge continues the lead's own conversation: its prompt, its plan, then the workers' replies
        assert [m["role"] for m in merges[0].messages] == ["user", "assistant", "user", "assistant"]
        assert (
            merges[0].messages[1]["content"] == plan.text and f": {row['answer']}" in merges[0].messages[2]["content"]
        )
        assert len({m.key for m in merges}) == 2  # merges are compared within their team only
        assert len({w.key for w in workers}) == 3  # and workers with those that got the same subtask
        assert all(w.rewards == {"team": 1.0} and row["question"] in w.messages[0]["content"] for w in workers)
        control = next(s for s in leads if s.key[1] == "control")
        assert scenario.reward("lead", control.rewards) is None and control.rewards["metric/control"] == 0.0

    # a lead that solves alone is paid for the answer, its plan gets no lift, and no worker runs
    def solo(messages, row):
        return "I will do it myself." if "SUBTASK:" in messages[-1]["content"] else f"ANSWER: {row['answer']}"

    ((episode,),) = play(scenario, rows[:1], {"lead": solo, "worker": lambda messages: ""})
    plan = next(s for s in by_role(episode, "lead") if s.key[1] == "plan")
    assert by_role(episode, "worker") == [] and plan.rewards["total"] == 1.0 and plan.rewards["lift"] == 0.0
    merge = next(s for s in by_role(episode, "lead") if s.key[1] == "merge")
    assert "(no subtask was given" in merge.messages[2]["content"]

    report = scenario.audit(rows=rows)
    assert report["failures"] == [] and report["degenerate"]["worker/silent"] == 0.0
    assert report["degenerate"]["lead/no_merge"] == 0.0
    assert scenario.subtasks("1. SUBTASK: a\n- **SUBTASK:** b\nSUBTASK:\nnot one") == ["a", "b"]
    only_two = pattern("lead_workers", {"max_workers": 2, "team_samples": 1, "merge_samples": 1, "control": False})
    ((episode,),) = play(only_two, rows[:1], oracles(only_two))
    assert len(by_role(episode, "worker")) == 2 and len(by_role(episode, "lead")) == 2


# ------------------------------------------------------------------ plan_execute on a stand-in environment


class StandInHarbor:
    """The interface plan_execute uses of HarborEnv: reset(task_dir, **row), tool_schemas, call_tool, done,
    get_reward(messages), close. The 'tests' pass when the command `make fix` was run."""

    instances: list = []
    tool_deadline = None

    def __init__(self, reward_key="reward"):
        self.done, self.closed, self.commands = False, False, []
        StandInHarbor.instances.append(self)

    def reset(self, task_dir, **row):
        self.task_dir = task_dir
        return None

    def tool_schemas(self):
        def fn(name, properties):
            parameters = {"type": "object", "properties": properties, "required": list(properties)}
            return {"type": "function", "function": {"name": name, "description": f"{name}.", "parameters": parameters}}

        return [fn("bash", {"command": {"type": "string"}}), fn("submit", {})]

    def call_tool(self, name, arguments):
        if name == "submit":
            self.done = True
            return "Submitted."
        self.commands.append(arguments["command"])
        return "[exit code 0]"

    def get_reward(self, messages=None):
        return {
            "reward": float(any("make fix" in c for c in self.commands)),
            "metric/commands": float(len(self.commands)),
        }

    def close(self):
        self.closed = True


class ProfiledHarbor(StandInHarbor):
    def reset(self, task_dir, **row):
        return [{"role": "user", "content": "a harness's own conversation"}]


def harbor_rows(tmp_path, n=2):
    rows = []
    for i in range(n):
        task = tmp_path / f"task{i}"
        (task / "solution").mkdir(parents=True)
        (task / "solution" / "solve.sh").write_text("#!/bin/bash\nmake fix\n")
        messages = [
            {"role": "system", "content": "You work in a container."},
            {"role": "user", "content": f"Fix build {i}."},
        ]
        rows.append({"task_dir": str(task), "messages": messages, "language": "it" if i else "en"})
    return rows


def test_plan_execute_runs_the_environment_as_it_is_and_the_tests_pay(tmp_path):
    rows = harbor_rows(tmp_path)
    args = {"env": f"{__name__}:StandInHarbor", "control": True}
    scenario = build_scenario("plan_execute", args)
    StandInHarbor.instances.clear()
    for row, (episode,) in zip(rows, play(scenario, rows, oracles(scenario))):
        (planner,), executors = by_role(episode, "planner"), by_role(episode, "executor")
        assert (
            "Fix build" in planner.messages[0]["content"]
            and "You cannot run anything" in planner.messages[0]["content"]
            or row["language"] == "it"
        )
        assert len(executors) == 3 and [e.key[1] for e in executors].count("control") == 1
        for executor in executors:
            assert executor.messages[0] == row["messages"][0]  # the row's own system prompt
            if executor.key[1] == "control":
                assert executor.messages[1] == row["messages"][1] and executor.rewards["metric/control"] == 0.0
                assert scenario.reward("executor", executor.rewards) is None
            else:
                marker = "Il piano di un collega" if row["language"] == "it" else "A colleague's plan"
                assert marker in executor.messages[1]["content"] and "make fix" in executor.messages[1]["content"]
                assert executor.rewards == {"metric/commands": 1.0, "tests": 1.0}
        assert planner.rewards["executed"] == 1.0 and planner.rewards["lift"] == 1.0
    assert len(StandInHarbor.instances) == 6 and all(env.closed and env.task_dir for env in StandInHarbor.instances)

    report = scenario.audit(rows=rows)
    assert report["failures"] == [] and report["oracle"] == {"planner": 1.0, "executor": 1.0}
    assert report["degenerate"] == {
        "planner/silent": 0.0,
        "planner/restate": 0.0,
        "executor/idle": 0.0,
        "executor/silent": 0.0,
    }
    assert all(env.closed for env in StandInHarbor.instances)

    (problem,) = scenario.check_row({"messages": rows[0]["messages"]})
    assert "no 'task_dir'" in problem
    (problem,) = scenario.check_row({"task_dir": "t"})
    assert "pgs harbor rows" in problem
    with pytest.raises(ValueError, match="harness profiles"):
        build_scenario("plan_execute", {"env_args": {"profiles": {"opencode": 1.0}}})
    refused = build_scenario("plan_execute", {"env": f"{__name__}:ProfiledHarbor"})
    with pytest.raises(ScenarioError, match="a conversation of its own"):
        play(refused, rows[:1], oracles(refused))
    assert StandInHarbor.instances[-1].closed  # closed on the failure path too


def test_a_trained_executor_calls_the_environments_tools_with_their_schemas(tmp_path):
    rows = harbor_rows(tmp_path, 1)

    def policy(prompt, call):
        if "<tool_response>" in prompt:
            return '<tool_call>\n{"name": "submit", "arguments": {}}\n</tool_call>'
        command = "make fix" if call % 2 else "ls"
        return f"<tool_call>\n{json.dumps({'name': 'bash', 'arguments': {'command': command}})}\n</tool_call>"

    StandInHarbor.instances.clear()
    pipeline, _, _ = _pipeline(
        "plan_execute",
        rows,
        policy,
        {"planner": {"partner": "scripted:restate"}},
        **{
            "scenario.args": {"env": f"{__name__}:StandInHarbor", "executor_samples": 2},
            "rollout.max_new_tokens": 64,
            "rollout.max_completion_tokens": 256,
            "rollout.max_model_len": 4096,
        },
    )
    try:
        ((group,),) = [pipeline._await(pipeline._eval(rows, 1.0, 1))]
    finally:
        pipeline.close()
    assert [t.role for t in group] == ["executor", "executor"]  # the planner was a partner
    assert len({t.comparison for t in group}) == 1  # both started from the same plan
    for t in group:
        assert [s["function"]["name"] for s in t.info["tools"]] == ["bash", "submit"]
        assert t.info["tools"] == StandInHarbor.instances[0].tool_schemas()  # the environment's own schemas
        assert t.tool_calls == 2 and t.tool_errors == 0 and t.finish in ("stop", "env_done") and t.scored
    assert sorted(t.rewards["tests"] for t in group) == [0.0, 1.0]
    assert all(env.closed for env in StandInHarbor.instances) and len(StandInHarbor.instances) == 2


# ------------------------------------------------------------------ trained sessions, the trainer, the exports


def test_trained_solvers_and_checkers_are_scored_by_the_rewards_through_the_rollout_loop():
    rows = [{"id": f"q{i}", "question": f"What is {i + 2} times 11?", "answer": str((i + 2) * 11)} for i in range(2)]

    def policy(prompt, call):
        task = prompt.split("What is ")[1].split(" times")[0]
        if "VERDICT" in prompt:  # a checker that accepts whatever ends in the right digit
            shown = prompt.split("Their final answer:\n")[1].split("\n")[0]
            return "VERDICT: ACCEPT" if shown == str(int(task) * 11) else f"VERDICT: REJECT\nANSWER: {int(task) * 11}"
        return f"I multiply.\nANSWER: {int(task) * 11 + (call % 2)}"  # every other attempt is off by one

    pipeline, _, _ = _pipeline("solve_check", rows, policy)
    config = RLConfig()
    config.set("rewards.correct.fn", "math")
    pipeline.scenario.bind(resolve_rewards(config.rewards), None, config)
    pipeline.scenario.prepare(rows)
    try:
        groups = pipeline._await(pipeline._eval(rows, 1.0, 2))
        stats = pipeline._group_stats([t for g in groups for t in g])
    finally:
        pipeline.close()
    for group in groups:
        solvers = [t for t in group if t.role == "solver"]
        checkers = [t for t in group if t.role == "checker"]
        assert len(solvers) == 4 and len(checkers) == 8 and all(t.scored and t.finish == "stop" for t in group)
        assert len({t.comparison for t in solvers}) == 1  # every attempt of the row answers the same prompt
        assert len({t.comparison for t in checkers}) == 4  # per episode and candidate
        assert sorted(t.rewards["correct"] for t in solvers) == [0.0, 0.0, 1.0, 1.0]
        assert all(t.rewards["verdict"] == 1.0 for t in checkers)
    assert stats["role/solver/reward"] == 0.5 and stats["role/checker/reward"] == 1.0
    assert stats["reward"] == stats["role/solver/reward"]


def reply_length(completion, **row):
    """A dense reward for a random policy: longer answers score higher."""
    return min(len(completion), 40) / 40.0


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_the_trainer_binds_its_rewards_to_the_pattern_trains_and_exports(tmp_path):
    from palingenesis.rl.scenarios.export import dpo_rows, sft_rows
    from palingenesis.rl.trainer import RLTrainer
    from palingenesis.rollouts import read_records

    rows = [{"question": f"Say something about the number {i}.", "answer": str(i)} for i in range(16)]
    config = RLConfig()
    # a random model never samples its end-of-turn token; a session cut by the token limit has not answered. One
    # token in five ends a reply here, so that sessions end by themselves, after different lengths.
    tok = tokenizer("Qwen/Qwen3-0.6B")
    stops = [
        t for t in tok.convert_ids_to_tokens(list(range(1000, 100000, 5))) if tok.convert_tokens_to_ids(t) is not None
    ]
    settings = {
        "model.policy": tiny_policy(tmp_path),
        "model.stop_tokens": stops,
        "model.use_liger_kernel": False,
        "model.chat_template_kwargs": {"enable_thinking": False},
        "data.dataset": _write(tmp_path / "rows.jsonl", rows),
        "data.eval_size": 2,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 2,
        "rollout.group_size": 2,
        "rollout.max_new_tokens": 8,
        "scenario.type": "solve_check",
        "scenario.args": {"answer_tag": "", "checker_samples": 1},
        "rewards.length.fn": f"{__name__}:reply_length",
        "loss.advantage_std": "role",
        "loss.advantage_std_floor": 0.05,
        "logging.dump_trajectories": 1,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
    }
    for key, value in settings.items():
        config.set(key, value)
    trainer = RLTrainer(config)
    assert [r.name for r in trainer.scenario.rewards] == ["length"] and trainer.scenario.rewards[0].fn is reply_length
    assert trainer.scenario.pass_mark == config.data.success_threshold and len(trainer.scenario.other_golds) == 14
    trainer.logged = []
    trainer._log = lambda kind, metrics, step: trainer.logged.append((kind, dict(metrics)))
    trainer.train()
    steps = [m for kind, m in trainer.logged if kind == "step"]
    assert len(steps) == 2 and all(m["episode/failed"] == 0 and m["role/solver/sessions"] > 0 for m in steps)
    assert any(m["trained_sequences"] > 0 and "role/solver/rewards/correct" in m for m in steps)

    # rollouts become datasets again: single-agent conversations per role, and preference pairs
    records = list(read_records(tmp_path / "run" / "rollouts"))
    assert "solver" in {r["role"] for r in records}
    sft = list(sft_rows(records, roles=["solver"], min_reward=None))
    assert sft and all(row["role"] == "solver" and row["messages"][0]["role"] == "user" for row in sft)
    assert all(row["messages"][0]["content"].startswith("Say something about") for row in sft)
    assert all(p["reward_chosen"] > p["reward_rejected"] for p in dpo_rows(records, roles=["solver"], margin=0.01))

    # rows that lack the prompt stop the run before the first rollout, naming the field
    config.set("data.dataset", _write(tmp_path / "bad.jsonl", [{"text": "no prompt here", "answer": "1"}] * 6))
    config.set("data.eval_size", 0)
    with pytest.raises(RLConfigError, match="do not fit scenario solve_check.*scenario.args.prompt_field"):
        RLTrainer(config)


# ------------------------------------------------------------------ configuration and the command line


def test_a_pattern_needs_rewards_and_a_generated_scenario_refuses_them():
    def validated(**settings):
        config = RLConfig()
        for key, value in {"model.policy": "p", "data.dataset": "d.jsonl", "rollout.backend": "hf", **settings}.items():
            config.set(key, value)
        return config.validate

    validated(**{"scenario.type": "solve_check", "rewards.correct.fn": "math"})()
    validated(**{"scenario.type": "handoff", "rewards.f1.fn": "qa_match", "roles.reader.partner": "scripted:oracle"})()
    validated(**{"scenario.type": "plan_execute", "roles.planner.train": False})()
    validated(**{"scenario.type": "lead_workers"})(python_rewards=True)
    with pytest.raises(RLConfigError, match="is a pattern over your own dataset.*Add a rewards: section"):
        validated(**{"scenario.type": "solve_check"})()
    with pytest.raises(RLConfigError, match="the scenario's verifier sets the rewards"):
        validated(**{"scenario.type": "plan_execute", "rewards.correct.fn": "math"})()
    with pytest.raises(RLConfigError, match="has no such scripted partner"):
        validated(
            **{"scenario.type": "solve_check", "rewards.c.fn": "math", "roles.checker.partner": "scripted:nope"}
        )()
    with pytest.raises(RLConfigError, match="solver_samples and checker_samples must be >= 1"):
        validated(**{"scenario.type": "solve_check", "rewards.c.fn": "math", "scenario.args": {"solver_samples": 0}})()


def test_cli_lists_the_patterns_and_checks_a_dataset_against_one(tmp_path, capsys):
    from palingenesis.rl.scenarios.cli import describe_patterns, main

    assert main(["patterns"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in PATTERNS) and "scenario.args.material_field" in out and "checker: oracle" in out
    described = {p["pattern"]: p for p in describe_patterns()}
    assert (
        described["solve_check"]["roles"] == ["solver", "checker"]
        and described["solve_check"]["options"]["answer_tag"] == "ANSWER"
    )
    assert "prompt_field" not in described["handoff"]["options"] and "question_field" in described["handoff"]["options"]
    assert (
        described["plan_execute"]["scored_by"] == "the environment's tests"
        and "task_dir" in described["plan_execute"]["fields"]
    )
    assert main(["list"]) == 0 and "solve_check (pattern over your dataset)" in capsys.readouterr().out
    assert main(["audit", "solve_check"]) == 1 and "pgs scenario check --config" in capsys.readouterr().out

    def config(name, dataset, scenario="{type: solve_check}", rewards="{correct: {fn: math}}"):
        path = tmp_path / name
        path.write_text(
            f"model: {{policy: p}}\ndata: {{dataset: {dataset}}}\nrollout: {{backend: hf}}\n"
            f"rewards: {rewards}\nscenario: {scenario}\n"
        )
        return str(path)

    good = _write(tmp_path / "good.jsonl", math_rows(12))
    assert main(["check", "--config", config("good.yaml", good), "--n", "8"]) == 0
    out = capsys.readouterr().out
    assert "8 checked, 8 fit the scenario" in out and "rewards     correct (math, x1)" in out
    assert "--- prompt of solver (en) ---" in out and "--- prompt of checker (it) ---" in out
    assert "Controlla tu la risposta" in out and "degenerate  checker/always_accept    +0.000" in out
    assert "scenario check: passed" in out
    assert main(["check", "--config", config("good.yaml", good), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] == [] and report["audit"]["oracle"]["checker"] == 1.0 and len(report["prompts"]) == 4

    # a column under another name: the check names it and the option that fixes it
    renamed = _write(tmp_path / "renamed.jsonl", [{"task": r["question"], "answer": r["answer"]} for r in math_rows(6)])
    assert main(["check", "--config", config("renamed.yaml", renamed)]) == 1
    out = capsys.readouterr().out
    assert (
        "FAIL  6 of 6 rows do not fit" in out
        and "scenario.args.prompt_field" in out
        and "scenario check: FAILED" in out
    )
    fixed = config("fixed.yaml", renamed, scenario="{type: solve_check, args: {prompt_field: task}}")
    assert main(["check", "--config", fixed]) == 0
    capsys.readouterr()

    # a gold column with worked solutions is not what the math reward compares with: the oracle is unpaid, with a hint
    worked = _write(
        tmp_path / "worked.jsonl", [{**r, "answer": f"It is {r['answer']}, of course"} for r in math_rows(6)]
    )
    assert main(["check", "--config", config("worked.yaml", worked)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  oracle solver" in out and "the gold text does not pass your rewards" in out

    # a reward that reads a column the rows do not have
    assert main(["check", "--config", config("col.yaml", good, rewards="{c: {fn: math, args: {field: target}}}")]) == 1
    assert "could not be played: KeyError" in capsys.readouterr().out
    assert main(["check", "--config", config("none.yaml", good, rewards="{}")]) == 1
    assert "Add a rewards: section" in capsys.readouterr().out
    assert main(["check", "--config", config("unknown.yaml", good, rewards="{c: {fn: 'nowhere.module:fn'}}")]) == 1
    assert "FAIL  rewards: ModuleNotFoundError" in capsys.readouterr().out

    reading = _write(tmp_path / "reading.jsonl", reading_rows())
    handoff = config(
        "handoff.yaml",
        reading,
        scenario="{type: handoff, args: {max_note_chars: 5000}}",
        rewards="{em: {fn: qa_match}}",
    )
    assert main(["check", "--config", handoff]) == 0
    out = capsys.readouterr().out
    assert "WARN  handoff: 4 of 4 materials fit" in out and "--- prompt of writer (en) ---" in out
