"""Publishing gates over an evaluation's metrics (thresholds in the `gates:` config section).

    gate                 passes when                                        kind
    acquisition          closed-book share of the probes answered in context >= acquisition_min   absolute
    update               same, on facts that superseded one                 >= update_min       absolute
    stale                answers stating a superseded value                 <= stale_max        absolute
    retention            net loss vs live on earlier facts (McNemar)        <= retention_max_net_loss   paired
    ancestor_retention   net loss vs live on the ancestors' facts           <= ancestor_retention_max_net_loss
    general_qa           net loss vs live on general QA                     <= general_qa_max_net_loss  paired
    leak                 changed, wrong answers naming a new entity         <= leak_max_net     paired by construction
    confabulation        net new answers to questions the notes do not answer  <= confab_max_net   paired
    meta_reference       closed-book answers mentioning notes / context     <= meta_reference_max   absolute
    drift_increment      KL(target || candidate) - KL(target || live)       <= drift_increment_max
    drift_cap            KL(target || candidate)                            <= drift_cap

A gate with nothing to measure (no superseded fact, no earlier fact, ...) is skipped and passes. The
defaults are uncalibrated (`gates.calibrated: false`) until a calibration on null sessions sets each
threshold to a per-gate false-fail rate; every report says which. A failed gate blocks making the version
live unless forced, and the manifest records the forced publish.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

REMEDIES = {
    "acquisition": "more passes or a higher learning rate; read the per-fact diagnostics for notes the model "
    "misreads, and rewrite them",
    "update": "the old value still wins: check that the teacher adopts the new note (preflight), raise "
    "replay.visits_per_fact, or rebuild the scope",
    "stale": "superseded values come back: make sure no active note still states the old value, raise replay, "
    "or rebuild the scope",
    "retention": "earlier facts were lost: raise replay.visits_per_fact, lower recipe.learning_rate or "
    "recipe.passes, or rebuild the scope",
    "ancestor_retention": "the scope's adapter overrides its ancestors' facts: lower recipe.learning_rate or "
    "recipe.passes, raise the anchor share",
    "general_qa": "general answers got worse: raise recipe.mix.anchor, lower recipe.learning_rate",
    "leak": "the scope's entities appear in unrelated answers: raise recipe.mix.nearmiss and recipe.mix.anchor",
    "confabulation": "the model now answers questions the notes do not answer: raise recipe.mix.anchor, make the "
    "notes specific, lower recipe.learning_rate",
    "meta_reference": "answers talk about notes or context: check the teacher's samples and the system prompt; "
    "the trainer drops teacher samples that do",
    "drift_increment": "the outputs moved too far in one session: lower recipe.learning_rate or recipe.passes, raise "
    "recipe.mix.anchor",
    "drift_cap": "the scope's adapter has drifted too far overall: rebuild the scope from its active notes",
}


@dataclass
class Gate:
    name: str
    value: float | int | None
    threshold: float | int
    op: str  # >= | <=
    passed: bool
    skipped: bool = False
    n: int = 0
    remedy: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def line(self) -> str:
        if self.skipped:
            return f"{self.name}: skipped (nothing to measure)"
        value = "n/a" if self.value is None else f"{self.value:g}"
        verdict = "pass" if self.passed else "FAIL"
        return f"{self.name}: {value} (needs {self.op} {self.threshold:g}, n={self.n}) {verdict}"


def _check(name: str, value, threshold, op: str, n: int, skip: bool = False) -> Gate:
    if skip:
        return Gate(name, None, threshold, op, True, skipped=True, n=n)
    ok = value is not None and (value >= threshold if op == ">=" else value <= threshold)
    return Gate(name, value, threshold, op, ok, n=n, remedy="" if ok else REMEDIES[name])


def evaluate(metrics: dict, config) -> list[Gate]:
    """The gates of `metrics` (evaluate.compute_metrics + drift) under the `gates:` config section."""
    gates = []
    for name, threshold in (("acquisition", config.acquisition_min), ("update", config.update_min)):
        m = metrics[name]
        n = m["candidate"]["n"]
        # No probe answered with the notes in context: the closed-book share is undefined, which fails.
        gates.append(_check(name, m["relative"]["rate"], threshold, ">=", m["relative"]["n"], skip=n == 0))
    stale = metrics["stale"]["candidate"]
    gates.append(_check("stale", stale["rate"], config.stale_max, "<=", stale["n"], skip=stale["n"] == 0))
    for name, threshold in (
        ("retention", config.retention_max_net_loss),
        ("ancestor_retention", config.ancestor_retention_max_net_loss),
        ("general_qa", config.general_qa_max_net_loss),
        ("confabulation", config.confab_max_net),
    ):
        paired = metrics[name]["paired"]
        gates.append(_check(name, paired["net_loss"], threshold, "<=", paired["n"], skip=paired["n"] == 0))
    leak = metrics["leak"]
    gates.append(_check("leak", leak["k"], config.leak_max_net, "<=", leak["n"], skip=leak["n"] == 0))
    meta = metrics["meta_reference"]["candidate"]
    gates.append(
        _check("meta_reference", meta["rate"], config.meta_reference_max, "<=", meta["n"], skip=meta["n"] == 0)
    )
    drift = metrics.get("drift") or {}
    tokens = drift.get("tokens", 0)
    gates.append(
        _check("drift_increment", drift.get("increment"), config.drift_increment_max, "<=", tokens, not tokens)
    )
    gates.append(_check("drift_cap", drift.get("candidate"), config.drift_cap, "<=", tokens, not tokens))
    return gates


def passed(gates: list[Gate]) -> bool:
    return all(g.passed for g in gates)
