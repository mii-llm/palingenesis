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

    def audit(self, n: int = 200, seed: int = 0) -> dict:
        """Play the scripted partners through the verifier on `n` generated rows: the oracles together, then
        each degenerate partner in its role with oracles in the others. Returns {"oracle": {role: mean reward},
        "degenerate": {"role/name": mean reward}, "failures": [...]}: an oracle must earn more than 0.9 of its
        role's best reward, a degenerate partner at most 0 on average."""
        rows = self.generate(n, seed)
        table = self.partners()
        missing = [role for role in self.roles if "oracle" not in table.get(role, {})]
        if missing:
            raise ValueError(f"scenario {self.name}: no oracle partner for {missing}")
        oracles = {role: table[role]["oracle"] for role in self.roles}

        def means(partners: dict[str, Partner]) -> dict[str, float]:
            totals: dict[str, list[float]] = {role: [] for role in self.roles}
            for episodes in play(self, rows, partners, seed):
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
}


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


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def system(content: str) -> dict:
    return {"role": "system", "content": content}
