"""The environment registry: built-in environments plus those other packages register through entry points.

    from palingenesis.rl.catalog import get, specs, register
    spec = get("wordle")
    for spec in specs(family="tool-calling"): ...

A third-party package registers environments in its pyproject.toml:

    [project.entry-points."palingenesis.environments"]
    lookup = "acme_envs.lookup:SPEC"          # an EnvSpec, a list of them, or a function returning either

Entry points are loaded on first use. One that fails to import or does not validate is reported
(registry.load_errors(), shown by `pgs env list`) and skipped: a broken third-party package never hides the rest of
the catalogue. Third-party names cannot replace built-in ones; publish them as "<publisher>/<name>".
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from typing import Any, Iterable

from palingenesis.rl.catalog.spec import EnvSpec, SpecError

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "palingenesis.environments"

# name -> "module:ATTRIBUTE" (imported on first use; the modules are stdlib-only at import)
BUILTIN: dict[str, str] = {
    "math-boxed": "palingenesis.rl.catalog.envs.math_boxed:SPEC",
    "code-unit-tests": "palingenesis.rl.catalog.envs.code:UNIT_TESTS",
    "code-stdio": "palingenesis.rl.catalog.envs.code:STDIO",
    "tools-function-call": "palingenesis.rl.catalog.envs.function_call:SPEC",
    "tools-filesystem": "palingenesis.rl.catalog.envs.stateful:FILESYSTEM",
    "tools-calendar": "palingenesis.rl.catalog.envs.stateful:CALENDAR",
    "tools-shop": "palingenesis.rl.catalog.envs.stateful:SHOP",
    "instruction-following": "palingenesis.rl.catalog.envs.instructions:SPEC",
    "text-to-sql": "palingenesis.rl.catalog.envs.sql:SPEC",
    "json-extraction": "palingenesis.rl.catalog.envs.extraction:SPEC",
    "wordle": "palingenesis.rl.catalog.envs.games:WORDLE",
    "number-guess": "palingenesis.rl.catalog.envs.games:NUMBER_GUESS",
    "corpus-qa": "palingenesis.rl.catalog.envs.search:SPEC",
    "harbor": "palingenesis.rl.catalog.envs.harbor:SPEC",
}

_registry: dict[str, EnvSpec] = {}
_errors: list[str] = []
_loaded = {"builtin": False, "entry_points": False}


class UnknownEnvironment(KeyError):
    def __str__(self) -> str:
        return str(self.args[0])


def register(spec: EnvSpec, replace: bool = False, source: str = "") -> EnvSpec:
    """Add a spec to the registry (a name registered twice is an error unless `replace`)."""
    if not isinstance(spec, EnvSpec):
        raise SpecError(f"register() takes an EnvSpec, got {type(spec).__name__}")
    if source and not spec.source:
        spec = dataclasses.replace(spec, source=source)
    existing = _registry.get(spec.name)
    if existing is not None and not replace:
        raise SpecError(
            f"environment {spec.name!r} is already registered (by {existing.source or 'this process'}); "
            "choose another name, e.g. '<publisher>/<name>'"
        )
    _registry[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    _registry.pop(name, None)


def _as_specs(value: Any) -> list[EnvSpec]:
    if callable(value) and not isinstance(value, EnvSpec):
        value = value()
    if isinstance(value, EnvSpec):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, EnvSpec) for v in value):
        return list(value)
    raise SpecError(f"expected an EnvSpec, a list of them or a function returning either, got {type(value).__name__}")


def _load_builtin() -> None:
    if _loaded["builtin"]:
        return
    _loaded["builtin"] = True
    for name, ref in BUILTIN.items():
        module, _, attr = ref.partition(":")
        try:
            spec = getattr(importlib.import_module(module), attr)
            if spec.name != name:
                raise SpecError(f"{ref} is named {spec.name!r}, not {name!r}")
            if name not in _registry:
                register(spec, source="builtin")
        except Exception as error:  # noqa: BLE001 — one broken built-in must not hide the others
            _errors.append(f"builtin {name}: {type(error).__name__}: {error}")


def _entry_points() -> Iterable[Any]:
    from importlib.metadata import entry_points

    return entry_points(group=ENTRY_POINT_GROUP)


def _load_entry_points() -> None:
    if _loaded["entry_points"]:
        return
    _loaded["entry_points"] = True
    for entry in _entry_points():
        distribution = getattr(getattr(entry, "dist", None), "name", None) or "entry point"
        if distribution == "palingenesis":
            continue  # the built-ins, already loaded (an installed palingenesis lists them as entry points too)
        try:
            for spec in _as_specs(entry.load()):
                if spec.name in BUILTIN:
                    raise SpecError(f"{spec.name!r} is a built-in environment; publish it as '<publisher>/<name>'")
                register(spec, source=distribution)
        except Exception as error:  # noqa: BLE001 — reported, not raised: the rest of the catalogue still loads
            _errors.append(f"{distribution} ({entry.name} = {entry.value}): {type(error).__name__}: {error}")
            logger.warning("environment entry point %s failed to load: %s", entry.name, error)


def load(entry_points: bool = True) -> None:
    _load_builtin()
    if entry_points:
        _load_entry_points()


def reload() -> None:
    """Forget what was loaded and read the built-ins and entry points again (tests; after installing a package)."""
    _registry.clear()
    _errors.clear()
    _loaded.update(builtin=False, entry_points=False)


def load_errors() -> list[str]:
    load()
    return list(_errors)


def get(name: str) -> EnvSpec:
    """The spec registered as `name`; "module:ATTRIBUTE" or "path/to/file.py:ATTRIBUTE" loads one that is not
    registered (an environment under development)."""
    load()
    if name in _registry:
        return _registry[name]
    if ":" in name:
        from palingenesis.rl.rewards import load_object

        found = _as_specs(load_object(name))
        if len(found) != 1:
            raise SpecError(f"{name} holds {len(found)} specs; name one")
        return found[0]
    import difflib

    close = difflib.get_close_matches(name, list(_registry), n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(close)}?" if close else " `pgs env list` shows the catalogue."
    raise UnknownEnvironment(f"no environment named {name!r}.{hint}")


def names() -> list[str]:
    load()
    return sorted(_registry)


def specs(family: str = "", trainable: bool | None = None, permissive: bool | None = None) -> list[EnvSpec]:
    """The registered specs, by family then name, optionally filtered."""
    load()
    out = [
        spec
        for spec in _registry.values()
        if (not family or spec.family == family)
        and (trainable is None or spec.trainable == trainable)
        and (permissive is None or spec.permissive == permissive)
    ]
    return sorted(out, key=lambda s: (s.family, s.name))
