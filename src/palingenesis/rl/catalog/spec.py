"""The catalogue's model: what an environment declares about itself.

An EnvSpec is the catalogue entry of one environment: the class that implements the environment protocol
(palingenesis.rl.env), where its tasks come from (a generator, a Hugging Face dataset pulled at run time, a file),
how an episode is verified, what it needs to run (no sandbox, a process sandbox, Docker), and the scripted policies
that audit it: an oracle that must be paid, and degenerate policies that must not (see catalog.audit).

    SPEC = EnvSpec(
        name="acme/lookup", version="0.1.0", family="tool-calling",
        summary="Look a record up with one tool call.",
        env=LookupEnv,
        datasets={"synthetic": DatasetSpec(generator=generate, license="Apache-2.0")},
        verifier="the final state of the store equals the goal state",
        license="Apache-2.0",
        oracle=oracle, degenerate={"claim_done": claim_done},
    )

Specs are validated when they are built (SpecError lists every problem), so a registry only ever holds valid entries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

SANDBOXES = ("none", "process", "docker")
FAMILIES = (
    "math",
    "code",
    "tool-calling",
    "instruction-following",
    "sql",
    "extraction",
    "games",
    "search",
    "shell",
    "other",
)
# Licences that allow use, modification and redistribution with at most attribution (no share-alike, no
# non-commercial clause). The built-in catalogue only uses these, for code and for data.
PERMISSIVE = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
        "Unlicense",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "PDDL-1.0",
        "CDLA-Permissive-2.0",
    }
)

_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*(/[a-z0-9][a-z0-9_.-]*)?")
_VERSION = re.compile(r"\d+\.\d+\.\d+([.-]?(a|b|rc|dev|post)\d*)?")

# A scripted policy: called before every assistant turn with the row and the conversation so far. It returns the
# turn: text (a final reply), a list of (tool name, arguments) calls, or {"content": text, "calls": [...]}.
Policy = Callable[[dict, list[dict]], Any]


class SpecError(ValueError):
    """An environment spec that does not validate; the message lists every problem."""


def is_permissive(license: str) -> bool:
    return license in PERMISSIVE


@dataclass(frozen=True)
class DatasetSpec:
    """Where an environment's rows come from.

    generator   (n, seed, **options) -> rows, or "module:function": procedural tasks, nothing downloaded
    hub_id      a Hugging Face dataset, loaded at run time (never vendored); `convert` maps one of its rows to an
                environment row, or returns None to drop it (a task the environment cannot verify)
    path        a local file (JSONL / JSON / parquet) or, for Harbor, a directory of tasks; `convert` applies too
    """

    generator: Callable[..., list[dict]] | str | None = None
    hub_id: str = ""
    config: str = ""
    split: str = "test"
    revision: str = ""
    path: str = ""
    convert: Callable[[dict], dict | None] | str | None = None
    license: str = ""
    citation: str = ""
    url: str = ""
    description: str = ""
    options: dict[str, Any] = field(default_factory=dict)  # the generator's defaults (difficulty, ...)

    @property
    def kind(self) -> str:
        return "generator" if self.generator is not None else "hub" if self.hub_id else "file"

    def problems(self, where: str) -> list[str]:
        out = []
        sources = sum((self.generator is not None, bool(self.hub_id), bool(self.path)))
        if sources > 1:
            out.append(f"{where}: set one of generator, hub_id, path")
        if not self.license and self.kind != "file":
            out.append(f"{where}: a dataset needs a license (an SPDX identifier)")
        if self.generator is not None and not (callable(self.generator) or _is_ref(self.generator)):
            out.append(f"{where}: generator must be a callable or 'module:function'")
        if self.convert is not None and not (callable(self.convert) or _is_ref(self.convert)):
            out.append(f"{where}: convert must be a callable or 'module:function'")
        return out

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "license": self.license}
        for key in ("hub_id", "config", "split", "revision", "path", "citation", "url", "description"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.options:
            out["options"] = dict(self.options)
        return out


def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and ":" in value and not value.startswith(":") and not value.endswith(":")


@dataclass(frozen=True)
class EnvSpec:
    name: str  # "wordle", or "<publisher>/<name>" for third-party packages
    version: str  # of the environment: bump it when rows, prompts or the verifier change (results stop comparing)
    family: str  # one of FAMILIES
    summary: str
    env: Any  # the environment class (palingenesis.rl.env protocol), or "module:Class"
    license: str  # of the environment's code
    verifier: str  # what is checked, in one or two sentences
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)  # the first one is the default
    env_args: dict[str, Any] = field(default_factory=dict)
    sandbox: str = "none"  # none | process | docker
    difficulty: dict[str, Any] = field(default_factory=dict)  # e.g. {"option": "difficulty", "levels": [1, 2, 3]}
    citation: str = ""
    trainable: bool = True  # False: evaluation only (the verifier is not safe to optimize against)
    reference: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> {"pass@1": ..., ...}, measured
    max_turns: int = 1
    max_new_tokens: int = 1024
    success_threshold: float = 1.0  # a reward at or above it is a pass
    chance: float = 0.0  # the pass rate a degenerate policy may reach honestly (guessing, abstention rows)
    oracle: Policy | None = None
    degenerate: dict[str, Policy] = field(default_factory=dict)
    audit: Callable[..., dict] | None = None  # replaces the scripted audit (Harbor: the tasks' own oracle / nop)
    requires: tuple[str, ...] = ()  # importable modules the environment needs beyond palingenesis' own
    tags: tuple[str, ...] = ()
    notes: str = ""
    source: str = ""  # who registered it: "builtin", or the distribution of an entry point

    def __post_init__(self):
        problems = self.problems()
        if problems:
            raise SpecError(f"environment spec {self.name!r}: " + "; ".join(problems))

    def problems(self) -> list[str]:
        out = []
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            out.append("name must be lower case letters, digits, '.', '_' or '-', optionally '<publisher>/<name>'")
        if not isinstance(self.version, str) or not _VERSION.fullmatch(self.version):
            out.append(f"version {self.version!r} is not MAJOR.MINOR.PATCH")
        if self.family not in FAMILIES:
            out.append(f"family {self.family!r} is not one of {', '.join(FAMILIES)}")
        if not self.summary or not self.summary.strip():
            out.append("summary is empty")
        if not (isinstance(self.env, type) or callable(self.env) or _is_ref(self.env)):
            out.append("env must be a class or 'module:Class'")
        if not self.license:
            out.append("license is empty (an SPDX identifier, e.g. Apache-2.0)")
        if not self.verifier or not self.verifier.strip():
            out.append("verifier is empty: say what the reward checks")
        if self.sandbox not in SANDBOXES:
            out.append(f"sandbox {self.sandbox!r} is not one of {', '.join(SANDBOXES)}")
        if not isinstance(self.datasets, dict) or not self.datasets:
            out.append("datasets is empty: give at least one DatasetSpec")
        else:
            for key, dataset in self.datasets.items():
                if not isinstance(dataset, DatasetSpec):
                    out.append(f"datasets[{key!r}] is not a DatasetSpec")
                else:
                    out.extend(dataset.problems(f"datasets[{key!r}]"))
        if not isinstance(self.max_turns, int) or self.max_turns < 1:
            out.append("max_turns must be a positive integer")
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            out.append("max_new_tokens must be a positive integer")
        if not 0.0 <= self.chance < 1.0:
            out.append("chance must be in [0, 1)")
        if self.oracle is not None and not callable(self.oracle):
            out.append("oracle must be a callable policy(row, messages)")
        for key, policy in (self.degenerate or {}).items():
            if not callable(policy):
                out.append(f"degenerate[{key!r}] must be a callable policy(row, messages)")
        if self.audit is None and self.oracle is None:
            out.append("no oracle policy: an environment nobody can show to be solvable cannot be audited")
        for model, metrics in (self.reference or {}).items():
            if not isinstance(metrics, dict) or not metrics:
                out.append(f"reference[{model!r}] must map metric names to numbers")
                continue
            for metric, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    out.append(f"reference[{model!r}][{metric!r}] is not a number")
                elif metric.startswith("pass@") and not 0.0 <= value <= 1.0:
                    out.append(f"reference[{model!r}][{metric!r}] = {value} is not a rate in [0, 1]")
        return out

    # ------------------------------------------------------------------ derived

    @property
    def eval_only(self) -> bool:
        return not self.trainable

    @property
    def default_dataset(self) -> str:
        return next(iter(self.datasets))

    @property
    def permissive(self) -> bool:
        """Whether the environment's code and every dataset it declares carry a permissive licence."""
        return is_permissive(self.license) and all(
            is_permissive(d.license) for d in self.datasets.values() if d.kind != "file"
        )

    def env_class(self) -> Any:
        if isinstance(self.env, str):
            from palingenesis.rl.rewards import load_object

            return load_object(self.env)
        return self.env

    def env_ref(self) -> str:
        """`module:Class`, as an RL config's env.type takes it."""
        if isinstance(self.env, str):
            return self.env
        return f"{self.env.__module__}:{self.env.__qualname__}"

    def missing_requirements(self) -> list[str]:
        import importlib.util

        return [module for module in self.requires if importlib.util.find_spec(module) is None]

    def tool_schemas(self, row: dict | None = None) -> list[dict]:
        """The tools the policy gets (for a row, when they depend on it)."""
        import asyncio

        from palingenesis.rl.catalog.runner import episode_tools

        return asyncio.run(episode_tools(self, row))

    def describe(self) -> dict[str, Any]:
        """The catalogue entry as plain data (pgs env info --json)."""
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "summary": self.summary,
            "env": self.env_ref(),
            "env_args": dict(self.env_args),
            "datasets": {key: d.describe() for key, d in self.datasets.items()},
            "verifier": self.verifier,
            "sandbox": self.sandbox,
            "difficulty": dict(self.difficulty),
            "license": self.license,
            "permissive": self.permissive,
            "citation": self.citation,
            "trainable": self.trainable,
            "eval_only": self.eval_only,
            "reference": {m: dict(v) for m, v in self.reference.items()},
            "max_turns": self.max_turns,
            "max_new_tokens": self.max_new_tokens,
            "success_threshold": self.success_threshold,
            "chance": self.chance,
            "policies": {"oracle": self.oracle is not None, "degenerate": sorted(self.degenerate)},
            "requires": list(self.requires),
            "tags": list(self.tags),
            "notes": self.notes,
            "source": self.source,
        }
