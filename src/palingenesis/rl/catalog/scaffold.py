"""pgs env new: a package skeleton for an environment of your own.

    pgs env new acme/lookup            # ./lookup/: pyproject.toml, lookup.py, tests/test_lookup.py

The module holds a small working environment (one tool, a final answer), its generator, an oracle, a degenerate
policy and the EnvSpec; the pyproject registers the spec under the "palingenesis.environments" entry-point group,
so `pip install -e .` makes it appear in `pgs env list`; the test runs the audit. Replace the task, keep the shape.
"""

from __future__ import annotations

import re
from pathlib import Path

MODULE = '''"""{name}: describe the task in one line.

Rows: {{"prompt", "records", "answer", "solution", "solution_answer"}}.
"""

import random

from palingenesis.rl.catalog import DatasetSpec, EnvSpec, scripted, turn_index
from palingenesis.rl.env import ToolFailed


class {cls}:
    """One instance is reused for many episodes: everything an episode needs is set in reset()."""

    tools = ("lookup",)  # the methods the policy may call

    def reset(self, records=None, answer="", **row):
        self.records, self.answer = dict(records or {{}}), str(answer)

    def lookup(self, key: str) -> str:
        """Look up the value stored under a key.

        Args:
            key: The key to look up.
        """
        if key not in self.records:
            raise ToolFailed(f"no such key: {{key}}")  # the policy sees "Error: no such key: ..."
        return self.records[key]

    def get_reward(self, messages):
        reply = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "assistant"), "")
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        # the last line, exactly: an answer buried in a list of guesses does not count
        return {{"correct": float(bool(lines) and lines[-1] == f"ANSWER: {{self.answer}}")}}


def generate(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        keys = [f"key-{{rng.randrange(10_000)}}" for _ in range(6)]
        records = {{key: f"{{rng.randrange(16**6):06x}}" for key in keys}}
        key = rng.choice(list(records))
        rows.append(
            {{
                "id": f"{name}/synthetic/{{seed}}/{{i}}",
                "prompt": f"Use the lookup tool to find the value stored under {{key}}. "
                'End your reply with the line "ANSWER: <value>".',
                "records": records,
                "answer": records[key],
                "solution": [["lookup", {{"key": key}}]],  # the oracle's calls, one per turn
                "solution_answer": f"ANSWER: {{records[key]}}",
            }}
        )
    return rows


def guess(row, messages):
    """A policy that must not be paid: answers without looking anything up."""
    return "ANSWER: 000000"


def lookup_everything(row, messages):
    """Another: looks every key up and offers all the values."""
    if turn_index(messages) == 0:
        return [("lookup", {{"key": key}}) for key in row["records"]]
    return "ANSWER: " + " ".join(row["records"].values())


SPEC = EnvSpec(
    name="{name}",
    version="0.1.0",
    family="{family}",
    summary="Look a value up with a tool and report it.",
    env={cls},
    license="Apache-2.0",
    verifier='The last line of the final message is "ANSWER: <the value stored under the key>".',
    datasets={{"synthetic": DatasetSpec(generator=generate, license="Apache-2.0")}},
    max_turns=4,
    oracle=scripted(),  # plays row["solution"], then replies row["solution_answer"]
    degenerate={{"guess": guess, "lookup_everything": lookup_everything}},
)
'''

TEST = '''from palingenesis.rl.catalog import audit

from {module} import SPEC


def test_oracle_is_paid_and_degenerate_policies_are_not():
    report = audit(SPEC, n=32)
    assert report["ok"], report["failures"]
    assert report["oracle"]["pass_rate"] == 1.0
    assert all(policy["passed"] == 0 for policy in report["degenerate"].values())
'''

PYPROJECT = """[project]
name = "{distribution}"
version = "0.1.0"
description = "A palingenesis environment: {name}"
requires-python = ">=3.11"
dependencies = ["palingenesis"]

[project.entry-points."palingenesis.environments"]
{entry} = "{module}:SPEC"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
include = ["{module}.py"]

[tool.pytest.ini_options]
pythonpath = ["."]
"""


def scaffold(name: str, directory: str = ".", family: str = "other") -> Path:
    """Write the skeleton of environment `name` under `directory`; returns the package directory."""
    from palingenesis.rl.catalog.spec import FAMILIES, _NAME

    if not _NAME.fullmatch(name):
        raise ValueError(f"{name!r}: lower case letters, digits, '.', '_' or '-', optionally '<publisher>/<name>'")
    if family not in FAMILIES:
        raise ValueError(f"family {family!r} is not one of {', '.join(FAMILIES)}")
    short = name.rsplit("/", 1)[-1]
    module = re.sub(r"[^a-z0-9]+", "_", short).strip("_")
    if not module or module[0].isdigit():
        module = f"env_{module}"
    cls = "".join(part.capitalize() for part in module.split("_")) + "Env"
    root = Path(directory) / short
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} exists and is not empty")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    fields = {
        "name": name,
        "cls": cls,
        "family": family,
        "module": module,
        "entry": module,
        "distribution": "pgs-env-" + re.sub(r"[^a-z0-9]+", "-", name).strip("-"),
    }
    (root / f"{module}.py").write_text(MODULE.format(**fields))
    (root / "tests" / f"test_{module}.py").write_text(TEST.format(**fields))
    (root / "pyproject.toml").write_text(PYPROJECT.format(**fields))
    return root
