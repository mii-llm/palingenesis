"""Scenarios: what a multi-role task is, as data and as code.

A scenario is a class with

  name        its registry name (`scenario.type` in the config, the "scenario" field of its rows)
  roles       the role names it uses; the config maps each to a model (`roles.<name>`)
  outcome     the role whose reward says whether the task was done (the prompt sampler's retire / retry signal
              and the outcome/* metrics)
  weights     {role: {component: weight}}: how each role's reward components add up (`scenario.args.weights`
              overrides entries). A component with weight 0 is logged and does not train.
  generate    rows from a seed: `generate(n, seed) -> list[dict]`. A row is the task alone (the numbers, the
              texts, the gold answers): prompts are built by `run`, so one row serves every role, language and
              prompt revision. Every row has "scenario", "id" and "language".
  run         `async def run(episode, row)`: the interaction (palingenesis.rl.scenarios.runtime) and the verifier,
              which calls `session.reward(component, value)`.
  partners    {role: {name: partner}}: scripted players. "oracle" plays the role correctly; the others are the
              degenerate policies the verifier must not pay (`audit`). Roles can also be given to them in a run
              (`roles.<name>.partner: scripted:oracle`): a fixed opponent, a perfect worker, a baseline.

Rows are plain JSONL, made by `pgs scenario generate` and read through `data.dataset` like any RL data.

A pattern (palingenesis.rl.scenarios.library.patterns) is a scenario without a generator: its rows are an ordinary
single-agent RL dataset and its verifier is the run's `rewards:` section (`uses_rewards`). Four more hooks serve
them, each a no-op by default:

  fields      {row field: what it holds}: what `pgs scenario patterns` lists
  bind        the run's rewards, sandbox and config, given once before the first episode
  check_row   the problems of one row (a missing field), as text: the trainer and `pgs scenario check` stop on them
  prepare     the rows the run will use, before the first episode (a pool of answers to draw wrong candidates from)
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from palingenesis.rl.scenarios.runtime import Episode, Partner, play

LANGUAGES = ("en", "it")


class Scenario:
    name = ""
    roles: tuple[str, ...] = ()
    outcome = ""
    weights: dict[str, dict[str, float]] = {}
    # audit: the least mean reward the oracle partners must earn together, per role (default: 0.9 x the sum of the
    # role's positive weights; a zero-sum game sets its own, since its oracles cannot both win)
    oracle_floor: dict[str, float] = {}
    uses_rewards = False  # True: sessions are scored by the run's `rewards:` functions (patterns), given by `bind`
    fields: dict[str, str] = {}  # the row fields it reads: {name: what it holds}
    check_rows = 200  # the rows `pgs scenario check` samples by default

    def __init__(self, weights: dict[str, dict[str, float]] | None = None):
        merged = {role: dict(components) for role, components in type(self).weights.items()}
        for role, components in (weights or {}).items():
            if role not in self.roles:
                raise ValueError(f"scenario {self.name}: weights name the role {role!r}; its roles are {self.roles}")
            merged.setdefault(role, {}).update({k: float(v) for k, v in components.items()})
        self.weights = merged

    @classmethod
    def generate(cls, n: int, seed: int = 0, **params) -> list[dict]:
        raise NotImplementedError

    async def run(self, episode: Episode, row: dict) -> None:
        raise NotImplementedError

    def partners(self) -> dict[str, dict[str, Partner]]:
        return {}

    def bind(self, rewards: list, sandbox: Any = None, config: Any = None) -> None:
        """The run's reward functions (palingenesis.rl.rewards.Reward), its code sandbox and its RL config."""

    def check_row(self, row: dict) -> list[str]:
        """What is wrong with a row for this scenario (a missing or empty field), one message each."""
        return []

    def prepare(self, rows: list[dict]) -> None:
        """Called once with the rows the episodes will be drawn from."""

    def advice(self, rows: list[dict]) -> list[str]:
        """Warnings about rows that are valid but make the task degenerate (`pgs scenario check` prints them)."""
        return []

    def reward(self, role: str, components: dict[str, float | None]) -> float | None:
        """A session's reward from its components: the weighted sum of those that apply (not None, not
        `metric/...`); None when none applies (the session cannot be judged and stays out of the loss)."""
        weights = self.weights.get(role, {})
        applicable = [
            (weights.get(name, 1.0), value)
            for name, value in components.items()
            if value is not None and not name.startswith("metric/")
        ]
        return sum(w * v for w, v in applicable) if applicable else None

    # ------------------------------------------------------------------ audit

    def audit(self, n: int = 200, seed: int = 0, rows: list[dict] | None = None, loop: Any = None) -> dict:
        """Play the scripted partners through the verifier on `n` generated rows (or on `rows`: a dataset of
        your own): the oracles together, then each degenerate partner in its role with oracles in the others.
        Returns {"oracle": {role: mean reward}, "degenerate": {"role/name": mean reward}, "failures": [...]}:
        an oracle must earn more than 0.9 of its role's best reward, a degenerate partner at most 0 on average.
        `loop`: an event loop to play on (see `play`)."""
        rows = self.generate(n, seed) if rows is None else list(rows)
        self.prepare(rows)
        table = self.partners()
        missing = [role for role in self.roles if "oracle" not in table.get(role, {})]
        if missing:
            raise ValueError(f"scenario {self.name}: no oracle partner for {missing}")
        oracles = {role: table[role]["oracle"] for role in self.roles}

        def means(partners: dict[str, Partner]) -> dict[str, float]:
            totals: dict[str, list[float]] = {role: [] for role in self.roles}
            for episodes in play(self, rows, partners, seed, loop=loop):
                for episode in episodes:
                    for session in episode.sessions:
                        value = self.reward(session.role, session.rewards)
                        if value is not None:
                            totals[session.role].append(value)
            return {role: sum(v) / len(v) if v else float("nan") for role, v in totals.items()}

        report: dict[str, Any] = {"oracle": means(oracles), "degenerate": {}, "failures": []}
        for role, value in report["oracle"].items():
            best = sum(abs(w) for w in self.weights.get(role, {}).values() if w > 0) or 1.0
            floor = self.oracle_floor.get(role, 0.9 * best)
            if not value >= floor:
                report["failures"].append(f"oracle {role}: mean reward {value:.3f} (expected >= {floor:.2f})")
        for role in self.roles:
            for name, partner in table.get(role, {}).items():
                if name == "oracle":
                    continue
                value = means({**oracles, role: partner})[role]
                report["degenerate"][f"{role}/{name}"] = value
                if value > 1e-9:
                    report["failures"].append(f"degenerate {role}/{name}: mean reward {value:.3f} (must be <= 0)")
        return report


# ---------------------------------------------------------------- registry

BUILTIN = {
    "take_away": "palingenesis.rl.scenarios.library.take_away:TakeAway",
    "ledger_sum": "palingenesis.rl.scenarios.library.ledger_sum:LedgerSum",
    "checked_answer": "palingenesis.rl.scenarios.library.checked_answer:CheckedAnswer",
    "solve_check": "palingenesis.rl.scenarios.library.solve_check:SolveCheck",
    "handoff": "palingenesis.rl.scenarios.library.handoff:Handoff",
    "lead_workers": "palingenesis.rl.scenarios.library.lead_workers:LeadWorkers",
    "plan_execute": "palingenesis.rl.scenarios.library.plan_execute:PlanExecute",
}
# the built-ins that have a generator of their own, and those that run on a dataset you bring (patterns)
GENERATED = ("take_away", "ledger_sum", "checked_answer")
PATTERNS = ("solve_check", "handoff", "lead_workers", "plan_execute")


def scenario_class(spec: str) -> type[Scenario]:
    """A built-in scenario's class by name, or "module:Class" / "path/to/file.py:Class"."""
    target = BUILTIN.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown scenario {spec!r}: built-ins are {sorted(BUILTIN)}; yours is 'module:Class'")
    if target.split(":")[0].endswith(".py"):
        from palingenesis.rl.rewards import load_object

        cls = load_object(target)
    else:
        module, _, name = target.partition(":")
        cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Scenario)):
        raise ValueError(f"{spec!r} is not a palingenesis.rl.scenarios.Scenario subclass")
    return cls


def build_scenario(spec: str, args: dict | None = None) -> Scenario:
    return scenario_class(spec)(**(args or {}))


# ------------------------------------------------------------------ parsing


def tagged_integer(text: str, tag: str) -> int | None:
    """The integer of the one line `TAG: n` in `text` (case-insensitive, the colon optional). None when no such
    line or several: a reply with two answers has not answered."""
    found = re.findall(rf"(?im)^[ \t>*_`]*{re.escape(tag)}\b[ \t]*:?[ \t]*(-?\d[\d.,]*)[ \t*_`.]*$", text or "")
    if len(found) != 1:
        return None
    digits = re.sub(r"[.,](?=\d{3}\b)", "", found[0]).rstrip(".,")
    return int(digits) if re.fullmatch(r"-?\d+", digits) else None


def tagged_word(text: str, tag: str, words: tuple[str, ...]) -> str | None:
    """The word of the one line `TAG: word` in `text`, lower-cased, if it is one of `words`; else None."""
    found = re.findall(rf"(?im)^[ \t>*_`]*{re.escape(tag)}\b[ \t]*:?[ \t]*([A-Za-zÀ-ÿ]+)[ \t*_`.]*$", text or "")
    if len(found) != 1 or found[0].lower() not in words:
        return None
    return found[0].lower()


def tagged_text(text: str, tag: str) -> str | None:
    """The text of the one line `TAG: ...` in `text` (case-insensitive tag), stripped of surrounding emphasis
    marks. None when there is no such line, several, or nothing after the tag."""
    found = re.findall(rf"(?im)^[ \t>*_`]*{re.escape(tag)}\b[ \t*_`]*:[ \t]*(.*?)[ \t]*$", text or "")
    if len(found) != 1:
        return None
    value = found[0].strip().strip("*_`").strip()
    return value or None


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def system(content: str) -> dict:
    return {"role": "system", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}
