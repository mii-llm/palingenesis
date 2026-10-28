"""vLLM plugin: a LoRA resolver over learn stores, so a server loads published scope adapters on demand.

    pip install 'palingenesis[learn]'          # in the vLLM environment
    VLLM_PLUGINS=palingenesis_learn VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PGS_LEARN_STORE=/srv/stores \\
      vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64

Registered under vLLM's `vllm.general_plugins` entry point (palingenesis_learn =
palingenesis.learn.vllm_plugin:register), which adds the resolver to vLLM's LoRAResolverRegistry. vLLM asks
its resolvers only for a model name it has not loaded, and only with VLLM_ALLOW_RUNTIME_LORA_UPDATING set:

    <scope>@v<n>   the n-th published version of the scope (user:acme/support/ana@v3), from published.json
                   (a version being published resolves too: the publisher warms it and runs the canary on it)
    <scope>        the scope's live published version at the first request. vLLM then keeps that name loaded:
                   it follows a later publish only after the name is unloaded (the publisher unloads it on the
                   servers it publishes to). The adapter is loaded under its versioned name either way, and vLLM
                   keys its prefix cache by that name, so no cached block crosses versions

PGS_LEARN_STORE is a store, a directory of stores (one per org), or several of them separated by ':'. A name
resolves only to a version that was published (not failed), whose adapters were not forgotten, and for the
base model the server runs (PGS_LEARN_CHECK_BASE=0 skips the base check). A missing composite is stacked
again from the chain's adapters (an exact concatenation of small tensors).
"""

from __future__ import annotations

import logging
import os
import zlib
from pathlib import Path

import yaml

from palingenesis.learn.store import STATE_FILE, Store, StoreError

logger = logging.getLogger(__name__)
RESOLVER_NAME = "palingenesis_learn"


def lora_id(name: str) -> int:
    """A stable positive LoRA int id for a served name (vLLM assigns its own when it resolves)."""
    return zlib.crc32(name.encode()) % (2**31 - 1) + 1


def store_roots(value: str | None = None) -> list[Path]:
    value = os.environ.get("PGS_LEARN_STORE", "") if value is None else value
    return [Path(p) for p in value.split(os.pathsep) if p]


def _org(path: Path) -> str | None:
    try:
        return (yaml.safe_load((path / STATE_FILE).read_text()) or {}).get("org")
    except (OSError, yaml.YAMLError):
        return None


def find_store(roots: list[Path], org: str) -> Path | None:
    """The store holding `org` under `roots`: a root that is that org's store, root/<org>, or a sub-directory
    whose learn_state.yaml names the org."""
    for root in roots:
        if (root / STATE_FILE).exists():
            if _org(root) == org:
                return root
            continue
        direct = root / org
        if (direct / STATE_FILE).exists() and _org(direct) == org:
            return direct
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (child / STATE_FILE).exists() and _org(child) == org:
                    return child
    return None


def resolve(roots: list[Path], base_model_name: str, lora_name: str, check_base: bool = True) -> tuple[str, str] | None:
    """(served name, adapter path) for a requested model name, or None when it is not a published scope
    version of a store under `roots`."""
    from palingenesis.learn.serving import _gone, parse_name, same_model

    try:
        scope, n = parse_name(lora_name)
    except StoreError:
        return None
    path = find_store(roots, scope.org)
    if path is None:
        return None
    store = Store(path)
    rec = store.published()["scopes"].get(str(scope))
    if not rec:
        return None
    n = rec.get("live") if n is None else n
    entry = rec["versions"].get(str(n)) if n is not None else None
    if entry is None or entry["state"] in ("failed", "pruned") or entry.get("engine", "vllm") != "vllm":
        return None
    if any(_gone(store, s, v) for s, v in entry["chain"]):
        return None  # a forgotten adapter is never served
    if check_base and not same_model(store.state["base"], base_model_name):
        logger.warning(
            "palingenesis_learn: %s is published for base %s, but this server runs %s; not resolved",
            lora_name,
            store.state["base"],
            base_model_name,
        )
        return None
    composite = store.root / entry["composite"]
    if not composite.resolve().is_relative_to(store.root.resolve()):
        logger.warning("palingenesis_learn: %s points outside its store; not resolved", lora_name)
        return None
    if not (composite / "manifest.json").exists():  # written last: a composite without it is incomplete
        _restack(store, entry, composite)
    return entry["name"], str(composite)


def _restack(store: Store, entry: dict, composite: Path) -> None:
    from palingenesis.learn import adapter as adapters

    paths = [store.scope(s).version_dir(v) for s, v in entry["chain"]]
    ranks = [store.scope(s).manifest(v)["rank"] for s, v in entry["chain"]]
    manifest = {"chain": entry["chain"], "rank": entry["rank"], "ranks": ranks, "id": entry["composite_id"]}
    adapters.stack_composite(paths, composite, store.state["base"], entry["rank"], manifest)


def register() -> None:
    """vLLM general-plugin entry point: register the store resolver (in every vLLM process; idempotent)."""
    import asyncio

    from vllm.lora.request import LoRARequest
    from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

    roots = store_roots()
    if not roots:
        logger.warning("palingenesis_learn: PGS_LEARN_STORE is not set; the LoRA resolver is not registered")
        return
    check_base = os.environ.get("PGS_LEARN_CHECK_BASE", "1").strip().lower() not in ("0", "false", "no")

    class StoreResolver(LoRAResolver):
        """Resolves <scope>@v<n> and <scope> to published composites in the learn stores."""

        async def resolve_lora(self, base_model_name: str, lora_name: str):
            try:
                found = await asyncio.to_thread(resolve, roots, base_model_name, lora_name, check_base)
            except Exception:  # noqa: BLE001 - a broken store must not fail requests for other names
                logger.exception("palingenesis_learn: resolving %s failed", lora_name)
                return None
            if found is None:
                return None
            name, path = found
            return LoRARequest(lora_name=name, lora_int_id=lora_id(name), lora_path=path)

    if RESOLVER_NAME not in LoRAResolverRegistry.get_supported_resolvers():
        LoRAResolverRegistry.register_resolver(RESOLVER_NAME, StoreResolver())
