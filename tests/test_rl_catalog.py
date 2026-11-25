"""The environment catalogue: spec validation, the registry and entry points, the audit's findings, the verifier
pool, the adapters, the Harbor entry (with a fake Docker runtime) and the `pgs env` commands."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from palingenesis.rl.catalog import DatasetSpec, EnvSpec, SpecError, audit, get, registry, scripted  # noqa: E402
from palingenesis.rl.catalog.cli import main  # noqa: E402


class EchoEnv:
    """Pays a reply that contains the answer anywhere: the kind of verifier the audit exists to catch."""

    tools = ()

    def reset(self, answer="", **row):
        self.answer = answer

    def get_reward(self, messages):
        return float(self.answer in (messages[-1].get("content") or ""))


def toy_rows(n, seed=0, leak=True):
    return [
        {"id": f"toy/{i}", "prompt": f"Repeat the code {'K' + str(i) if leak else ''}.", "answer": f"K{i}"}
        for i in range(n)
    ]


def toy(name="toy", env=EchoEnv, **changes):
    fields = dict(
        name=name,
        version="0.1.0",
        family="other",
        summary="A toy.",
        env=env,
        license="MIT",
        verifier="The answer appears in the reply.",
        datasets={"synthetic": DatasetSpec(generator=toy_rows, license="MIT")},
        oracle=lambda row, messages: row["answer"],
    )
    return EnvSpec(**{**fields, **changes})


@pytest.fixture
def clean_registry():
    registry.reload()
    yield registry
    registry.reload()


# ----------------------------------------------------------------------- spec


def test_spec_validation_lists_every_problem():
    with pytest.raises(SpecError) as error:
        toy(name="Bad Name", version="1", family="puzzles", license="", verifier=" ", sandbox="vm", datasets={},
            max_turns=0, chance=1.0, oracle=None, reference={"m": {"pass@1": 1.5, "pass@4": "high"}})  # fmt: skip
    message = str(error.value)
    for part in ("name must be", "MAJOR.MINOR.PATCH", "family 'puzzles'", "license is empty", "verifier is empty",
                 "sandbox 'vm'", "datasets is empty", "max_turns", "chance", "no oracle policy", "pass@1", "not a number"):  # fmt: skip
        assert part in message, part
    with pytest.raises(SpecError, match="one of generator, hub_id, path"):
        toy(datasets={"d": DatasetSpec(generator=toy_rows, hub_id="org/data", license="MIT")})
    with pytest.raises(SpecError, match="needs a license"):
        toy(datasets={"d": DatasetSpec(hub_id="org/data")})
    with pytest.raises(SpecError, match="degenerate"):
        toy(degenerate={"x": "not callable"})
    with pytest.raises(Exception):  # frozen: an entry does not change after registration
        toy().version = "9.9.9"


def test_spec_describes_itself():
    spec = toy(
        name="acme/toy",
        datasets={
            "synthetic": DatasetSpec(generator=toy_rows, license="MIT"),
            "hub": DatasetSpec(
                hub_id="org/data", config="main", split="test", license="CC-BY-SA-4.0", citation="X 2024"
            ),
        },
        reference={"org/model": {"pass@1": 0.31, "pass@4": 0.55}},
        trainable=False,
        requires=("definitely_not_installed_module",),
    )
    entry = spec.describe()
    assert json.loads(json.dumps(entry)) == entry
    assert entry["env"].endswith(":EchoEnv") and entry["eval_only"] and not entry["permissive"]  # the share-alike data
    assert entry["datasets"]["hub"] == {
        "kind": "hub",
        "license": "CC-BY-SA-4.0",
        "hub_id": "org/data",
        "config": "main",
        "split": "test",
        "citation": "X 2024",
    }
    assert spec.default_dataset == "synthetic" and spec.missing_requirements() == ["definitely_not_installed_module"]
    assert (
        spec.env_class() is EchoEnv
        and toy(env="palingenesis.rl.catalog.envs.sql:SqlEnv").env_class().__name__ == "SqlEnv"
    )
    assert [t["function"]["name"] for t in get("tools-shop").tool_schemas()][:2] == ["search_products", "get_product"]


# ------------------------------------------------------------------- registry


def test_registry_holds_the_builtin_catalogue(clean_registry):
    names = registry.names()
    assert names == sorted(registry.BUILTIN) and len(names) == 14 and not registry.load_errors()
    families = {spec.family for spec in registry.specs()}
    assert {
        "math",
        "code",
        "tool-calling",
        "instruction-following",
        "sql",
        "extraction",
        "games",
        "search",
        "shell",
    } <= families
    assert [s.name for s in registry.specs(family="games")] == ["number-guess", "wordle"]
    assert [s.name for s in registry.specs(trainable=False)] == ["instruction-following"]
    assert len(registry.specs(permissive=True)) == 14
    with pytest.raises(registry.UnknownEnvironment, match="Did you mean wordle"):
        registry.get("wordel")
    with pytest.raises(SpecError, match="already registered"):
        registry.register(toy(name="wordle"))
    registry.register(toy())
    assert registry.get("toy").summary == "A toy." and "toy" in registry.names()
    registry.register(toy(summary="Another."), replace=True)
    assert registry.get("toy").summary == "Another."
    registry.unregister("toy")
    assert "toy" not in registry.names()


def test_entry_points_register_third_party_environments(clean_registry, monkeypatch):
    def entry(name, value, dist="acme-envs"):
        return SimpleNamespace(name=name, value=f"acme:{name}", dist=SimpleNamespace(name=dist), load=value)

    def broken():
        raise ImportError("No module named 'acme_missing'")

    entries = [
        entry("one", lambda: toy(name="acme/one")),  # an EnvSpec
        entry("many", lambda: (lambda: [toy(name="acme/two"), toy(name="acme/three")])),  # a function returning specs
        entry("broken", broken),
        entry("shadow", lambda: toy(name="wordle")),  # may not replace a built-in
        entry("wrong", lambda: {"name": "not a spec"}),
        entry("builtin", broken, dist="palingenesis"),  # palingenesis' own entry points are the built-ins
    ]
    monkeypatch.setattr(registry, "_entry_points", lambda: entries)
    names = registry.names()
    assert {"acme/one", "acme/two", "acme/three"} <= set(names) and len(names) == 17
    assert registry.get("acme/one").source == "acme-envs" and registry.get("wordle").source == "builtin"
    errors = registry.load_errors()
    assert len(errors) == 3
    assert "acme_missing" in errors[0] and "built-in environment" in errors[1] and "expected an EnvSpec" in errors[2]
    assert registry.ENTRY_POINT_GROUP == "palingenesis.environments"


def test_a_spec_under_development_loads_from_a_file(tmp_path, clean_registry):
    path = tmp_path / "mine.py"
    path.write_text(
        "from palingenesis.rl.catalog import DatasetSpec, EnvSpec\n"
        "class Env:\n    tools = ()\n    def reset(self, answer='', **row):\n        self.answer = answer\n"
        "    def get_reward(self, messages):\n        return float(messages[-1]['content'].strip() == self.answer)\n"
        "def rows(n, seed=0):\n    return [{'prompt': f'Say only the number {i + 10}.', 'answer': str(i + 10)} for i in range(n)]\n"
        "SPEC = EnvSpec(name='mine', version='0.0.1', family='other', summary='s', env=Env, license='MIT', verifier='exact',\n"
        "               datasets={'d': DatasetSpec(generator=rows, license='MIT')}, oracle=lambda row, messages: row['answer'])\n"
    )
    spec = registry.get(f"{path}:SPEC")
    assert spec.name == "mine" and "mine" not in registry.names()
    report = audit(spec, n=8)
    assert report["ok"] and report["oracle"]["passed"] == 8


def test_listing_the_catalogue_imports_nothing_heavy():
    code = (
        "import sys\nimport palingenesis.rl\nbefore = set(sys.modules)\n"
        "from palingenesis.rl.catalog import registry\nassert len(registry.specs()) == 14, registry.load_errors()\n"
        "added = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(sorted(added & {'transformers', 'sympy', 'datasets', 'pandas', 'vllm', 'jsonschema', 'docker'}))\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": ""}
    )
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stdout + out.stderr


# ---------------------------------------------------------------------- audit


def test_audit_reports_a_reward_hacking_hole_with_the_policy_and_a_row():
    report = audit(toy(), n=10)
    assert not report["ok"] and report["oracle"]["passed"] == 10
    assert [hole["policy"] for hole in report["holes"]] == ["echo_prompt"]
    hole = report["holes"][0]
    assert hole["passed"] == 10 and hole["row"] == "toy/0" and "K0" in hole["reply"]
    assert (
        "degenerate echo_prompt: passes 10 of 10 rows" in report["failures"][0]
        and "reward-hacking hole" in report["failures"][0]
    )
    fixed = toy(datasets={"synthetic": DatasetSpec(generator=toy_rows, license="MIT", options={"leak": False})})
    assert audit(fixed, n=10)["ok"]


def test_audit_allows_chance_and_warns_about_partial_credit():
    class Half:
        tools = ()

        def reset(self, answer="", **row):
            self.answer, self.free = answer, row["id"].endswith(("0", "1"))

        def get_reward(self, messages):
            reply = messages[-1].get("content") or ""
            return 1.0 if reply == self.answer or self.free else 0.3

    report = audit(toy(env=Half, chance=0.25), n=10)  # rows 0 and 1 pay anything: 20% < chance
    assert report["ok"] and report["degenerate"]["empty"]["passed"] == 2
    assert any("partial credit" in w and "mean reward 0.44" in w for w in report["warnings"])
    strict = audit(toy(env=Half), n=10)
    assert not strict["ok"] and len(strict["holes"]) == 4


def test_audit_catches_unsolvable_rows_state_leaks_and_crashes():
    def rows_with_a_bad_gold(n, seed=0):
        rows = toy_rows(n, leak=False)
        rows[3]["answer"] = "K3"
        return rows

    def wrong_oracle(row, messages):
        return "nope" if row["id"] == "toy/3" else row["answer"]

    report = audit(
        toy(oracle=wrong_oracle, datasets={"synthetic": DatasetSpec(generator=rows_with_a_bad_gold, license="MIT")}),
        n=6,
    )
    assert not report["ok"] and "oracle: 1 of 6 rows are not solved; row toy/3: reward 0.0" in report["failures"][0]

    class Sticky:  # remembers that it was ever told to unlock: the next episode on the instance starts unlocked
        def __init__(self):
            self.unlocked = False

        def reset(self, **row):
            self.done = False

        def unlock(self, code: str) -> str:
            """Unlock with a code.

            Args:
                code: The code.
            """
            self.unlocked = self.unlocked or code == "open"
            return "ok"

        def get_reward(self):
            return float(self.unlocked)

    leaky = toy(
        env=Sticky,
        max_turns=3,
        datasets={
            "synthetic": DatasetSpec(
                generator=lambda n, seed=0: [
                    {"id": f"s/{i}", "prompt": "Unlock.", "solution": [["unlock", {"code": "open"}]]} for i in range(n)
                ],
                license="MIT",
            )
        },
        oracle=scripted(),
    )
    report = audit(leaky, n=4, concurrency=1)
    assert not report["ok"] and report["holes"], report  # every later policy inherits the unlocked instance

    class Crashes:
        tools = ()

        def reset(self, answer="", **row):
            self.answer = answer

        def get_reward(self, messages):
            return float(messages[-1]["content"].split()[0] == self.answer)  # IndexError on an empty reply

    report = audit(
        toy(
            env=Crashes, datasets={"synthetic": DatasetSpec(generator=toy_rows, license="MIT", options={"leak": False})}
        ),
        n=4,
    )
    assert not report["ok"] and any(
        "degenerate empty: the environment raised on 4 of 4" in f for f in report["failures"]
    )
    missing = audit(toy(requires=("definitely_not_installed_module",)))
    assert not missing["ok"] and "missing Python modules" in missing["failures"][0]


# ---------------------------------------------------------------- verifier pool


def _square(x):
    return x * x


def _slow(x):
    while True:
        x += 1


def _fails(x):
    raise ValueError(f"bad {x}")


def test_verifier_pool_batches_times_out_and_reports_errors():
    from palingenesis.rl.catalog.verify import VerifierPool, VerifyTimeout

    pool = VerifierPool(workers=2)

    async def work():
        squares = await asyncio.gather(*(pool.verify(_square, i) for i in range(200)))
        mixed = await asyncio.gather(
            pool.verify(_square, 3), pool.verify(_slow, 1, timeout=0.5), pool.verify(_fails, 7), pool.verify(_square, 4),
            return_exceptions=True,
        )  # fmt: skip
        return squares, mixed

    try:
        squares, mixed = asyncio.run(work())
    finally:
        pool.shutdown()
    assert squares == [i * i for i in range(200)]
    assert pool.items == 204 and pool.batches <= 8  # 200 checks travelled as one batch per worker
    assert mixed[0] == 9 and mixed[3] == 16  # a hung or failing check costs only itself
    assert isinstance(mixed[1], VerifyTimeout) and isinstance(mixed[2], RuntimeError) and "bad 7" in str(mixed[2])
    inline = VerifierPool(workers=0)
    assert asyncio.run(inline.verify(_square, 5)) == 25


# -------------------------------------------------------------------- adapters


class FakeParser:
    def parse_answer(self, completion):
        return completion[-1]["content"].strip() if completion else ""


def _fake_verifiers(tools=False):
    parser = FakeParser()

    def exact(parser, completion, answer, **kwargs):
        return 1.0 if parser.parse_answer(completion) == answer else 0.0

    async def used_tool(completion, state):
        return float(any(m["role"] == "tool" for m in completion)) if state["turn"] else 0.0

    def reverse(text: str) -> str:
        """Reverse a string.

        Args:
            text: The string.
        """
        return text[::-1]

    kind = type("ToolEnv" if tools else "SingleTurnEnv", (), {})
    env = kind()
    env.dataset = [{"question": f"Reverse abc{i}", "answer": f"{i}cba", "info": {"k": i}} for i in range(6)]
    env.eval_dataset = None
    env.system_prompt = "Answer with the reversed string only."
    env.parser = parser
    env.rubric = SimpleNamespace(funcs=[exact, used_tool], weights=[1.0, 0.0], parser=parser)
    env.max_turns = 4 if tools else 1
    if tools:
        env.tools = [reverse]
    return env


def test_verifiers_environments_import_with_their_rubric_and_tools():
    from palingenesis.rl.catalog.adapters import verifiers_rows, verifiers_spec
    from palingenesis.rl.catalog.runner import load_rows, make_pool, play

    env = _fake_verifiers(tools=True)
    with pytest.raises(SpecError, match="license is required"):
        verifiers_spec("acme/reverse", env, license="", summary="Reverse.")
    with pytest.raises(SpecError, match="not a permissive"):
        verifiers_spec("acme/reverse", env, license="CC-BY-NC-4.0", summary="Reverse.")
    assert not verifiers_spec(
        "acme/reverse", env, license="CC-BY-NC-4.0", summary="R.", allow_non_permissive=True
    ).permissive
    rows = verifiers_rows(env)
    assert rows[2]["messages"] == [
        {"role": "system", "content": env.system_prompt},
        {"role": "user", "content": "Reverse abc2"},
    ]
    assert rows[2]["answer"] == "2cba" and rows[2]["info"] == {"k": 2}

    def solve(row, messages):
        if len(messages) == 2:
            return [("reverse", {"text": row["messages"][-1]["content"].split()[-1]})]
        return messages[-1]["content"]

    spec = verifiers_spec("acme/reverse", env, license="MIT", summary="Reverse a string.", oracle=solve)
    assert spec.max_turns == 4 and [t["function"]["name"] for t in spec.tool_schemas(rows[0])] == ["reverse"]
    episode = asyncio.run(play(make_pool(spec), load_rows(spec, n=1)[0], solve, 4))
    assert episode.rewards == {"reward": 1.0, "metric/exact": 1.0, "metric/used_tool": 1.0} and episode.tool_calls == 1
    report = audit(spec, n=6)
    assert report["ok"] and report["oracle"]["passed"] == 6 and report["degenerate"]["tool_spam"]["passed"] == 0
    unsolved = verifiers_spec("acme/reverse-single", _fake_verifiers(), license="MIT", summary="Reverse a string.")
    report = audit(unsolved, n=6)  # no solver: the degenerate policies only, and the report says so
    assert report["ok"] and report["oracle"] is None and "no oracle policy" in report["warnings"][0]
    assert set(report["degenerate"]) == {"empty", "refusal", "echo_prompt", "tool_spam"}
    multi_turn = type("MultiTurnEnv", (), {"env_response": lambda self: None, "rubric": None})()
    with pytest.raises(NotImplementedError, match="SingleTurnEnv and ToolEnv"):
        asyncio.run(
            play(make_pool(verifiers_spec("acme/chat", multi_turn, license="MIT", summary="s")), rows[0], solve, 2)
        )


def test_openenv_servers_become_catalogue_entries():
    from palingenesis.rl.catalog.adapters import openenv_spec, seed_rows

    spec = openenv_spec(
        "acme/echo", base_url="http://localhost:8001", license="BSD-3-Clause", summary="Echo.", family="games"
    )
    assert spec.env_ref() == "palingenesis.rl.envs.openenv:OpenEnvAdapter" and spec.requires == ("openenv",)
    assert spec.env_args == {"reset_fields": ["seed"], "prompt": "observation", "base_url": "http://localhost:8001"}
    assert spec.audit is not None and spec.oracle is None and spec.permissive
    assert seed_rows(3, seed=2) == [{"prompt": "", "seed": 200006 + i} for i in range(3)]
    local = openenv_spec(
        "acme/local", env_class="pkg.server:Env", license="MIT", summary="s", rows="tasks.jsonl", oracle=lambda r, m: ""
    )
    assert local.audit is None and local.datasets["rows"].path == "tasks.jsonl" and local.requires == ()
    with pytest.raises(SpecError, match="not a permissive"):
        openenv_spec("acme/gpl", base_url="http://x", license="GPL-3.0", summary="s")


# ----------------------------------------------------------------------- harbor


class FakeRuntime:
    """DockerRuntime stand-in: the tests pass when solve.sh ran; a task named *freebie* passes regardless."""

    def __init__(self, task, workdir=None):
        self.task, self.solved = task, False

    def start(self):
        pass

    def stop(self):
        pass

    def put_dir(self, source, destination):
        pass

    def exec(self, command, timeout=None):
        self.solved = self.solved or "solve.sh" in command
        return SimpleNamespace(exit_code=0, output="")

    def verify(self, final_message=None):
        return {"reward": float(self.solved or "freebie" in self.task.name)}


def _task(root: Path, name: str, solution: bool = True) -> None:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(f'[task]\nname = "{name}"\n')
    (task / "instruction.md").write_text(f"Create the file for {name}.")
    (task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    if solution:
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text("touch done\n")


def test_harbor_tasks_are_audited_through_the_catalogue(tmp_path, monkeypatch, capsys):
    from palingenesis.rl.catalog.runner import load_rows
    from palingenesis.rl.envs import harbor

    monkeypatch.setattr(harbor, "DockerRuntime", FakeRuntime)
    good = tmp_path / "good"
    _task(good, "write-a-file")
    _task(good, "fix-a-bug")
    spec = get("harbor")
    rows = load_rows(spec, str(good))
    assert len(rows) == 2 and all("task_dir" in row and row["messages"][-1]["role"] == "user" for row in rows)
    report = audit(spec, dataset=str(good))
    assert (
        report["ok"]
        and report["rows"] == 2
        and report["oracle"]["passed"] == 2
        and report["degenerate"]["nop"]["passed"] == 0
    )
    assert main(["test", "harbor", "--dataset", str(good)]) == 0
    assert "audit: passed" in capsys.readouterr().out
    bad = tmp_path / "bad"
    _task(bad, "freebie-task")
    _task(bad, "no-solution", solution=False)
    report = audit(spec, dataset=str(bad))
    assert not report["ok"] and len(report["failures"]) == 2 and len(report["holes"]) == 1
    assert "freebie-task pays doing nothing" in report["holes"][0] and report["degenerate"]["nop"]["passed"] == 1
    assert any("no-solution" in f and "no oracle solution" in f for f in report["failures"])
    without = audit(spec)
    assert not without["ok"] and "--dataset" in without["failures"][0]
    assert spec.env_class() is harbor.HarborEnv and spec.sandbox == "docker"


# -------------------------------------------------------------------------- cli


def test_cli_list_and_info(capsys):
    assert main(["list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(listing["environments"]) == 14 and listing["errors"] == []
    assert main(["list", "--family", "tool-calling"]) == 0
    out = capsys.readouterr().out
    assert "tools-shop" in out and "wordle" not in out and "4 environments" in out
    assert main(["list", "--eval-only"]) == 0 and "[eval-only]" in capsys.readouterr().out
    assert main(["info", "tools-calendar", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert [t["function"]["name"] for t in info["tools"]] == [
        "list_events",
        "create_event",
        "reschedule_event",
        "cancel_event",
    ]
    assert (
        info["example_row"]["prompt"] and "type: palingenesis.rl.catalog.envs.stateful:CalendarEnv" in info["rl_config"]
    )
    assert info["policies"] == {
        "oracle": True,
        "degenerate": ["answer_listing", "cancel_everything", "claim_done", "look_then_claim"],
    }
    for name in registry.names():
        assert main(["info", name]) == 0
        out = capsys.readouterr().out
        assert "verifier" in out and "licence" in out and "datasets" in out, name
    with pytest.raises(SystemExit, match="Did you mean"):
        main(["info", "text-to-sq"])


def test_cli_test_exit_codes_and_json(tmp_path, capsys):
    assert main(["test", "wordle", "number-guess", "--n", "12"]) == 0
    out = capsys.readouterr().out
    assert out.count("audit: passed") == 2 and "2 of 2 audits passed" in out and "steps/s" in out
    assert main(["test", "json-extraction", "--n", "6", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["oracle"]["steps_per_s"] > 0 and "schema_valid_blanks" in report["degenerate"]
    weak = tmp_path / "weak.py"
    weak.write_text(
        "from palingenesis.rl.catalog import DatasetSpec, EnvSpec\n"
        "class Env:\n    tools = ()\n    def reset(self, answer='', **row):\n        self.answer = answer\n"
        "    def get_reward(self, messages):\n        return float(self.answer in messages[-1]['content'])\n"
        "def rows(n, seed=0):\n    return [{'id': f'w/{i}', 'prompt': f'Repeat K{i}.', 'answer': f'K{i}'} for i in range(n)]\n"
        "SPEC = EnvSpec(name='weak', version='0.0.1', family='other', summary='s', env=Env, license='MIT', verifier='substring',\n"
        "               datasets={'d': DatasetSpec(generator=rows, license='MIT')}, oracle=lambda row, messages: row['answer'])\n"
    )
    assert main(["test", f"{weak}:SPEC", "--n", "5"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  degenerate echo_prompt: passes 5 of 5 rows" in out and "audit: FAILED" in out
    with pytest.raises(SystemExit):
        main(["test"])


def test_cli_rows_writes_training_rows_and_refuses_eval_only(tmp_path, capsys):
    out = tmp_path / "rows.jsonl"
    assert main(["rows", "number-guess", "--n", "7", "--seed", "3", "--option", "limit=100", "-o", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 7 and all(row["limit"] == 100 and 1 <= row["secret"] <= 100 for row in rows)
    printed = capsys.readouterr().out
    assert "type: palingenesis.rl.catalog.envs.games:NumberGuessEnv" in printed and f"dataset: {out}" in printed
    from palingenesis.rl.config import RLConfig

    config = tmp_path / "rl.yaml"
    config.write_text("model: {policy: m}\n" + printed.split("\n\n", 1)[1])
    loaded = RLConfig.from_yaml(str(config))  # the printed lines are a valid RL config section
    assert (
        loaded.env.type.endswith(":NumberGuessEnv") and loaded.env.max_turns == 13 and loaded.data.dataset == str(out)
    )
    with pytest.raises(SystemExit, match="evaluation-only"):
        main(["rows", "instruction-following", "-o", str(out)])
    assert main(["rows", "instruction-following", "-o", str(out), "--n", "3", "--allow-eval-only"]) == 0


def test_cli_new_scaffolds_an_environment_whose_test_passes(tmp_path, capsys, monkeypatch):
    assert main(["new", "acme/price-check", "--dir", str(tmp_path), "--family", "tool-calling"]) == 0
    root = tmp_path / "price-check"
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) == [
        "price_check.py",
        "pyproject.toml",
        "test_price_check.py",
    ]
    import tomllib

    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["entry-points"]["palingenesis.environments"] == {"price_check": "price_check:SPEC"}
    assert project["project"]["name"] == "pgs-env-acme-price-check"
    spec = registry.get(f"{root / 'price_check.py'}:SPEC")
    assert (
        spec.name == "acme/price-check"
        and spec.family == "tool-calling"
        and spec.env_class().__name__ == "PriceCheckEnv"
    )
    assert main(["test", f"{root / 'price_check.py'}:SPEC"]) == 0
    assert "audit: passed" in capsys.readouterr().out
    monkeypatch.syspath_prepend(str(root))  # the generated test, as pytest would run it in the package
    namespace: dict = {}
    exec(compile((root / "tests" / "test_price_check.py").read_text(), "test_price_check.py", "exec"), namespace)
    namespace["test_oracle_is_paid_and_degenerate_policies_are_not"]()
    with pytest.raises(SystemExit, match="not empty"):
        main(["new", "acme/price-check", "--dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="lower case"):
        main(["new", "Bad Name", "--dir", str(tmp_path)])


class GoldOnceEngine:
    """Answers \\boxed{gold} to the first sample of each prompt and \\boxed{0} to the others."""

    def __init__(self, tok, eot, rows):
        self.tok, self.eot, self.rows, self.seen, self.version = tok, eot, rows, {}, 0

    def generate(self, prompts, max_new_tokens, temperature):
        from palingenesis.opd.rollout import Rollout

        out = []
        for prompt in prompts:
            text = self.tok.decode(prompt)
            row = next(r for r in self.rows if r["prompt"].split("\n")[0] in text)
            n = self.seen[row["id"]] = self.seen.get(row["id"], 0) + 1
            answer = row["solution_answer"] if n == 1 or row["id"].endswith("/0") else "0"
            ids = self.tok.encode(f"So \\boxed{{{answer}}}.", add_special_tokens=False) + [self.eot]
            out.append(Rollout(ids, [-0.5] * len(ids), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def test_cli_run_reports_pass_at_k_through_the_rollout_loop(tmp_path, monkeypatch, capsys):
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.catalog.runner import load_rows
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    rows = load_rows(get("math-boxed"), n=4, seed=0, difficulty=1)
    engine = GoldOnceEngine(tok, end_of_turn_id(tok, {"enable_thinking": False}), rows)
    monkeypatch.setattr("palingenesis.rl.screen._engine", lambda config, model, stop_ids, pad_id: engine)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(
        "palingenesis.opd.trainer.student_stop_ids", lambda *a, **k: (tok.convert_tokens_to_ids("<|im_end|>"),)
    )
    out = tmp_path / "results.jsonl"
    arguments = ["run", "math-boxed", "--model", "Qwen/Qwen3-0.6B", "--k", "4", "--n", "4", "--option", "difficulty=1",
                 "--thinking", "off", "--backend", "hf", "--max_model_len", "2048", "--output", str(out), "--json"]  # fmt: skip
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (
        summary["env"] == "math-boxed"
        and summary["rows"] == 4
        and summary["episodes"] == 16
        and summary["unscored"] == 0
    )
    assert summary["pass@1"] == pytest.approx((1.0 + 3 * 0.25) / 4) and summary["pass@4"] == 1.0
    assert (
        summary["finishes"] == {"stop": 1.0} and summary["episodes_per_s"] > 0 and summary["model"] == "Qwen/Qwen3-0.6B"
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [sum(r["scores"]) for r in records] == [4, 1, 1, 1] and records[0]["id"] == rows[0]["id"]
    from palingenesis.agent_tooling import compare

    monkeypatch.setattr(sys, "argv", ["pgs compare", str(out), "--json"])  # the results file is what pgs compare reads
    assert compare.main() in (0, None)
    assert '"items": 4' in capsys.readouterr().out


def test_cli_run_refuses_unsandboxed_code_by_default():
    from palingenesis.rl.catalog import cli

    made = []

    def fake_make_sandbox(config):
        made.append(config.backend)
        return "docker-sandbox"

    import palingenesis.rl.sandbox as sandbox_module

    original, sandbox_module.make_sandbox = sandbox_module.make_sandbox, fake_make_sandbox
    try:
        assert cli._sandbox(get("code-stdio"), "") == "docker-sandbox" and made == ["docker"]
        assert cli._sandbox(get("wordle"), "") is None
        assert type(cli._sandbox(get("code-stdio"), "process")).__name__ == "SubprocessSandbox"
    finally:
        sandbox_module.make_sandbox = original
