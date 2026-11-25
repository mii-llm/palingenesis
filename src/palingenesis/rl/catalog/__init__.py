"""The environment catalogue: where RL and evaluation environments are found, audited and run.

    pgs env list [--family tool-calling]         what is registered (built-in and from installed packages)
    pgs env info text-to-sql                     the catalogue entry: data, tools, verifier, sandbox, licence
    pgs env test text-to-sql                     the audit: an oracle must be paid, degenerate policies must not
    pgs env run text-to-sql --model M --k 4      k episodes per row with a model: pass@1, pass@k, episodes/s
    pgs env rows text-to-sql --n 2000 -o rows.jsonl   rows for `pgs rl` (data.dataset) with the env.type to use
    pgs env new my-env                           a package skeleton: environment, spec, entry point, test

    from palingenesis.rl.catalog import audit, get, load_rows, specs
    spec = get("wordle")
    report = audit(spec, n=64)                   # report["ok"], report["holes"]
    rows = load_rows(spec, n=1000, seed=1)       # rows for RLTrainer / run_episodes with spec.env_class()

Modules: spec (EnvSpec, DatasetSpec: the typed entry), registry (built-ins and the "palingenesis.environments"
entry-point group), runner (rows; episodes played by scripted policies), audit (oracle / degenerate / reuse),
verify (batched verification in a process pool), adapters (OpenEnv servers and verifiers environments as specs),
scaffold (pgs env new), envs (the built-in environments). See the Environments guide.
"""

from palingenesis.rl.catalog.audit import audit, audit_async
from palingenesis.rl.catalog.registry import ENTRY_POINT_GROUP, get, load_errors, names, register, specs, unregister
from palingenesis.rl.catalog.runner import Episode, load_rows, play, play_many, scripted, turn_index
from palingenesis.rl.catalog.spec import FAMILIES, PERMISSIVE, SANDBOXES, DatasetSpec, EnvSpec, SpecError

__all__ = [
    "ENTRY_POINT_GROUP",
    "FAMILIES",
    "PERMISSIVE",
    "SANDBOXES",
    "DatasetSpec",
    "EnvSpec",
    "Episode",
    "SpecError",
    "audit",
    "audit_async",
    "get",
    "load_errors",
    "load_rows",
    "names",
    "play",
    "play_many",
    "register",
    "scripted",
    "specs",
    "turn_index",
    "unregister",
]
