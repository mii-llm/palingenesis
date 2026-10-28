"""The pgs learn vLLM plugin without vLLM: the resolver's name parsing and store lookup, refusals (failed,
forgotten, another base, unknown names), recomposition of a missing composite, and registration through
vLLM's LoRAResolverRegistry with the vLLM 0.26 interfaces stubbed."""

import abc
import asyncio
import sys
import types
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from learn_fakes import CharReference, FakeServer, gold_answers, make_version  # noqa: E402

from palingenesis.learn import (  # noqa: E402
    LearnConfig,
    Learner,
    vllm_plugin,
)
from palingenesis.learn.store import remove_tree  # noqa: E402
from palingenesis.learn.vllm_plugin import find_store, lora_id, resolve  # noqa: E402

ORG, USER = "org:acme", "user:acme/support/ana"


@pytest.fixture
def published(tmp_path):
    """A store of org acme under tmp_path/stores/acme-store with org:acme@v1 and @v2 published (v2 live)."""
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 4, "team": 2, "user": 2}
    learner = Learner.init(tmp_path / "stores" / "acme-store", "Qwen/Qwen3.5-4B", "acme", config)
    server = FakeServer(base="Qwen/Qwen3.5-4B")
    try:
        for note in ("The price of Orion Pro is 25 euros.", "The price of Orion Max is 40 euros."):
            learner.add_notes(ORG, [note], approve=True)
            learner.set_live(ORG, make_version(learner, ORG))
            server.answers = gold_answers(learner, [ORG])
            pub = learner.publisher(servers=[server.url], reference=CharReference(server.answers))
            pub.publish(ORG)
    finally:
        server.close()
    return learner


def test_resolves_versions_and_the_live_alias(published):
    roots = [published.store.root.parent]
    name, path = resolve(roots, "Qwen/Qwen3.5-4B", "org:acme@v1")
    assert name == "org:acme@v1" and Path(path, "adapter_model.safetensors").exists()
    # the bare scope is the live version, under its versioned name (vLLM keys its prefix cache by it)
    assert resolve(roots, "Qwen/Qwen3.5-4B", "org:acme")[0] == "org:acme@v2"
    # a local copy of the same base resolves too
    assert resolve(roots, "/models/Qwen3.5-4B", "org:acme@v2") is not None


@pytest.mark.parametrize(
    "name",
    ["org:acme@v9", "org:other@v1", "user:acme/support/ana@v1", "Qwen/Qwen3.5-4B", "org:acme@1", "", "a@b@c"],
)
def test_unknown_names_return_none(published, name):
    assert resolve([published.store.root.parent], "Qwen/Qwen3.5-4B", name) is None


def test_refuses_another_base_failed_and_forgotten_versions(published):
    roots = [published.store.root.parent]
    assert resolve(roots, "Qwen/Qwen3-1.7B", "org:acme@v2") is None
    assert resolve(roots, "Qwen/Qwen3-1.7B", "org:acme@v2", check_base=False) is not None
    data = published.store.published()
    data["scopes"][ORG]["versions"]["1"]["state"] = "failed"
    published.store.write_published(data)
    assert resolve(roots, "Qwen/Qwen3.5-4B", "org:acme@v1") is None
    # forgetting a note prunes the versions that held it: nothing of them resolves, the alias included
    note = published.store.scope(ORG).active_notes()[0].id
    published.forget(ORG, [note], unload=False)
    assert resolve(roots, "Qwen/Qwen3.5-4B", "org:acme@v2") is None
    assert resolve(roots, "Qwen/Qwen3.5-4B", "org:acme") is None


def test_recomposes_a_missing_composite(published):
    roots = [published.store.root]
    name, path = resolve(roots, "Qwen/Qwen3.5-4B", "org:acme@v2")
    before = Path(path, "adapter_model.safetensors").read_bytes()
    remove_tree(Path(path))
    assert resolve(roots, "Qwen/Qwen3.5-4B", "org:acme@v2") == (name, path)
    assert Path(path, "adapter_model.safetensors").read_bytes() == before  # stacking is deterministic


def test_store_discovery(published, tmp_path):
    root = published.store.root
    assert find_store([root], "acme") == root  # the store itself
    assert find_store([root.parent], "acme") == root  # a directory of stores
    assert find_store([root], "other") is None
    assert find_store([tmp_path / "missing"], "acme") is None
    assert vllm_plugin.store_roots(f"/a{':'}/b") == [Path("/a"), Path("/b")]
    assert 1 <= lora_id("org:acme@v1") < 2**31 and lora_id("org:acme@v1") == lora_id("org:acme@v1")


@pytest.fixture
def fake_vllm(monkeypatch):
    """vllm.lora.request.LoRARequest and vllm.lora.resolver as in vLLM 0.26 (the parts the plugin uses)."""

    class LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path="", base_model_name=None):
            assert lora_int_id >= 1 and lora_path
            self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path

    class LoRAResolver(abc.ABC):
        @abc.abstractmethod
        async def resolve_lora(self, base_model_name, lora_name): ...

    class Registry:
        def __init__(self):
            self.resolvers = {}

        def get_supported_resolvers(self):
            return self.resolvers.keys()

        def register_resolver(self, name, resolver):
            self.resolvers[name] = resolver

    registry = Registry()
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.lora": types.ModuleType("vllm.lora"),
        "vllm.lora.request": types.ModuleType("vllm.lora.request"),
        "vllm.lora.resolver": types.ModuleType("vllm.lora.resolver"),
    }
    modules["vllm.lora.request"].LoRARequest = LoRARequest
    modules["vllm.lora.resolver"].LoRAResolver = LoRAResolver
    modules["vllm.lora.resolver"].LoRAResolverRegistry = registry
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return registry


def test_register_and_resolve_through_vllm_interfaces(published, fake_vllm, monkeypatch):
    monkeypatch.setenv("PGS_LEARN_STORE", str(published.store.root.parent))
    vllm_plugin.register()
    vllm_plugin.register()  # vLLM loads general plugins in every process, possibly more than once
    resolver = fake_vllm.resolvers[vllm_plugin.RESOLVER_NAME]
    request = asyncio.run(resolver.resolve_lora("Qwen/Qwen3.5-4B", "org:acme"))
    assert request.lora_name == "org:acme@v2" and Path(request.lora_path).is_dir()
    assert asyncio.run(resolver.resolve_lora("Qwen/Qwen3.5-4B", "org:acme@v7")) is None
    assert asyncio.run(resolver.resolve_lora("Qwen/Qwen3.5-4B", "some-other-lora")) is None


def test_register_without_a_store_is_a_no_op(fake_vllm, monkeypatch):
    monkeypatch.delenv("PGS_LEARN_STORE", raising=False)
    vllm_plugin.register()
    assert not fake_vllm.resolvers


def test_entry_point_is_declared():
    import tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text())
    plugins = data["project"]["entry-points"]["vllm.general_plugins"]
    assert plugins == {"palingenesis_learn": "palingenesis.learn.vllm_plugin:register"}
