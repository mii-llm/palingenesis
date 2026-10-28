"""pgs learn serving on a real vLLM server (GPU integration; skipped without CUDA, vLLM and PEFT).

Per model (PGS_LEARN_GPU_MODELS, default one hybrid and one dense: Qwen/Qwen3.5-0.8B, Qwen/Qwen3-1.7B), a
store with an org and a user scope whose live adapters are random MLP LoRAs on the model's real modules, and a
`vllm serve` launched with runtime LoRA updates and the palingenesis_learn resolver plugin:

  G1  publish the stacked (org + user) adapter through /v1/load_lora_adapter: the HF + PEFT parity canary
      passes; a fault-injected export (wrong module prefix, or a dropped module) fails it
  G1b publish through the resolver plugin (no load route call): the warm-up request loads it from the store
  G2  versioned hot swap under load: streaming generations while v1 -> v2 is published; no errors, and every
      stream is answered by one version
  G3  prefix cache: the same long system prefix through the bare alias before and after a publish matches
      the new version's own name (no block crosses versions); an in-place reload is checked and reported
  G7  the stacked rank (org 32 + user 32 = 64) loads under --max-lora-rank 64

Run: pytest tests/test_learn_gpu.py -v (one model: PGS_LEARN_GPU_MODELS=Qwen/Qwen3.5-0.8B).
"""

import importlib.util
import json
import os
import sys
import threading
import urllib.request
import warnings

import pytest

sys.path.insert(0, "src")
warnings.simplefilter("ignore")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
    pytest.skip("needs CUDA and vLLM", allow_module_level=True)
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from palingenesis.learn import (  # noqa: E402
    LearnConfig,
    Learner,
    serving,
)
from palingenesis.learn import adapter as adapters  # noqa: E402
from palingenesis.learn.config import MLP_TARGET_REGEX  # noqa: E402
from palingenesis.learn.evaluate import chain_versions  # noqa: E402
from palingenesis.learn.store import sha256_file  # noqa: E402
from palingenesis.opd.rollout import VLLMServer  # noqa: E402

MODELS = os.environ.get("PGS_LEARN_GPU_MODELS", "Qwen/Qwen3.5-0.8B,Qwen/Qwen3-1.7B").split(",")
ORG, USER = "org:gpu", "user:gpu/ana"
LONG_SYSTEM = "You are the support assistant of a company that sells telescopes. " * 60


def random_version(learner, scope, shapes, seed, scale=0.05):
    """A live version of `scope` with a random LoRA on every MLP projection (passing report, as eval leaves it)."""
    from palingenesis.learn import gates as gating
    from palingenesis.learn import report as reporting
    from palingenesis.learn.evaluate import Answers, Probe, compute_metrics, kl_increment
    from palingenesis.learn.generate import StubGenerator
    from palingenesis.learn.recipe import plan_session
    from palingenesis.learn.store import ScopeId
    from palingenesis.learn.trainer import write_session_files

    scope = ScopeId.parse(scope)
    ss = learner.store.scope(scope).ensure()
    rank = learner.config.scopes.ranks[scope.level]
    plan = plan_session(ss, StubGenerator(), learner.config)
    number = ss.next_number()
    session, version = f"s{number:04d}", f"v{number:04d}"
    write_session_files(ss.dir / "sessions" / session, plan, learner.config)
    gen = torch.Generator().manual_seed(seed)
    tensors = {
        name: (torch.randn(rank, i, generator=gen) / i**0.5, torch.randn(o, rank, generator=gen) * scale / rank**0.5)
        for name, (i, o) in shapes.items()
    }
    out = ss.version_dir(version)
    adapters.write_adapter(tensors, out, learner.store.state["base"])
    manifest = {
        "version": version,
        "session": session,
        "rank": rank,
        "parent_versions": {s: v for s, v in chain_versions(learner.store, scope)[:-1] if v},
        "note_ids": [n.id for n in ss.active_notes()],
        "fact_ids": [f.id for f in plan.new_facts],
        "sha256": sha256_file(out / adapters.ADAPTER_WEIGHTS),
    }
    ss.write_manifest(version, manifest)
    probes = [Probe("acquisition", "a?", ["f"], ["v"])]
    metrics, _ = compute_metrics(probes, Answers(["v"], ["no"], ["v"]), [])
    metrics["drift"] = kl_increment(None, None, 0, 0)
    gates = gating.evaluate(metrics, learner.config.gates)
    report = {
        "scope": str(scope),
        "version": version,
        "session": session,
        "created": "2026-01-01T00:00:00Z",
        "live_chain": chain_versions(learner.store, scope),
        "candidate_chain": [],
        "calibrated": False,
        "metrics": metrics,
        "gates": [g.to_dict() for g in gates],
        "passed": True,
        "warnings": [],
    }
    reporting.write(ss.dir / "sessions" / session, report, [])
    learner.set_live(scope, version)
    return version


@pytest.fixture(scope="module", params=MODELS)
def setup(request, tmp_path_factory):
    model_id = request.param
    config = LearnConfig()
    config.generate.generator = "stub"
    config.scopes.ranks = {"org": 32, "team": 16, "user": 32}  # 32 + 32 = 64
    config.serving.drain_seconds = 0
    config.serving.canary_probes = 8
    root = tmp_path_factory.mktemp("gpu_store") / "gpu"
    learner = Learner.init(root, model_id, "gpu", config)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    modules = dict(hf.named_modules())
    shapes = {
        n: (modules[n].in_features, modules[n].out_features) for n in adapters.target_modules(hf, MLP_TARGET_REGEX)
    }
    del hf
    learner.add_notes(ORG, ["The Orion 8 telescope weighs 12 kg."], approve=True)
    random_version(learner, ORG, shapes, 1)
    learner.remember(USER, "my observatory is called Kestrel Ridge")
    random_version(learner, USER, shapes, 2)
    env = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_PLUGINS": "palingenesis_learn",
        "PGS_LEARN_STORE": str(root),
    }
    args = ("--enable-lora", "--max-lora-rank", "64", "--max-loras", "4", "--gpu-memory-utilization", "0.4")
    log = str(tmp_path_factory.mktemp("logs") / "vllm.log")
    server = VLLMServer(model_id, args=args, log_path=log, env=env, dev_mode=False)
    yield learner, server, shapes
    server.close()


def test_g1_publish_with_the_parity_canary(setup):
    learner, server, _ = setup
    result = learner.publish(USER, servers=[server.url])
    assert result.canary and all(c.passed for c in result.canary), [c.line() for c in result.canary]
    assert json.loads((learner.store.root / result.composite / "manifest.json").read_text())["rank"] == 64  # G7


@pytest.mark.parametrize("fault", ["prefix", "dropped"])
def test_g1_fault_injected_export_fails_the_canary(setup, monkeypatch, tmp_path, fault):
    learner, server, _ = setup
    real = serving.engine_adapter

    def broken(composite, engine):
        out = tmp_path / f"broken-{fault}"
        if fault == "prefix":
            adapters.rename_modules(composite, out, {"model.": "model.wrong."})
        else:
            from safetensors.torch import load_file, save_file

            out.mkdir(exist_ok=True)
            weights = load_file(str(real(composite, engine) / adapters.ADAPTER_WEIGHTS))
            save_file({k: v for k, v in weights.items() if "down_proj" not in k}, str(out / adapters.ADAPTER_WEIGHTS))
            (out / adapters.ADAPTER_CONFIG).write_text((composite / adapters.ADAPTER_CONFIG).read_text())
        return out

    monkeypatch.setattr(serving, "engine_adapter", broken)
    live = learner.store.published()["scopes"].get(USER, {}).get("live")
    random_version(learner, USER, setup[2], 10 + len(fault))
    with pytest.raises(serving.PublishError):
        learner.publish(USER, servers=[server.url])
    assert learner.store.published()["scopes"].get(USER, {}).get("live") == live


def test_g1b_resolver_transport(setup):
    learner, server, shapes = setup
    random_version(learner, USER, shapes, 3)
    before = len(_models(server))
    result = learner.publish(USER, servers=[server.url], transport="resolver")
    assert all(c.passed for c in result.canary)
    assert result.name in _models(server) and len(_models(server)) >= before


def _models(server):
    with urllib.request.urlopen(f"{server.url}/v1/models") as r:
        return [c["id"] for c in json.loads(r.read())["data"]]


def _stream(server, model, out, i):
    body = {"model": model, "prompt": "Count from one to twenty:", "max_tokens": 64, "stream": True}
    req = urllib.request.Request(
        f"{server.url}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    names = set()
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data: {"):
                names.add(json.loads(line[6:])["model"])
    out[i] = names


def test_g2_versioned_hot_swap_under_load(setup):
    learner, server, shapes = setup
    learner.publish(USER, servers=[server.url])
    name = learner.store.published()["scopes"][USER]
    first = name["versions"][str(name["live"])]["name"]
    out = [None] * 64
    threads = [threading.Thread(target=_stream, args=(server, first, out, i)) for i in range(64)]
    for t in threads:
        t.start()
    random_version(learner, USER, shapes, 4)
    learner.config.serving.drain_seconds = 120  # in-flight streams keep the old name
    learner.publish(USER, servers=[server.url])
    learner.config.serving.drain_seconds = 0
    for t in threads:
        t.join()
    assert all(names == {first} for names in out), out


def test_g3_prefix_cache_never_crosses_versions(setup):
    learner, server, shapes = setup
    client = serving.make_client("vllm", server.url, learner.store.state["base"])
    prompt = LONG_SYSTEM + "\nQuestion: which telescope weighs 12 kg?"
    learner.publish(USER, servers=[server.url])
    alias_before = client.generate(USER, prompt, 32)
    random_version(learner, USER, shapes, 5)
    learner.publish(USER, servers=[server.url])
    rec = learner.store.published()["scopes"][USER]
    current = rec["versions"][str(rec["live"])]["name"]
    alias_after = client.generate(USER, prompt, 32)
    named = client.generate(current, prompt, 32)
    # The alias now answers as the new version's own name; the old answer came from other blocks.
    assert alias_after == named
    assert alias_before is not None


def test_g3_in_place_reload_is_reported(setup, tmp_path):
    """Documents vLLM's in-place reload (load_inplace) with a warm prefix: the library never uses it."""
    learner, server, shapes = setup
    client = serving.make_client("vllm", server.url, learner.store.state["base"])
    rec = learner.store.published()["scopes"][USER]
    entry = rec["versions"][str(rec["live"])]
    composite = learner.store.root / entry["composite"]
    other = learner.store.root / rec["versions"][str(max(int(n) for n in rec["versions"]) - 1)]["composite"]
    client.load("inplace-test", str(other))
    ids = list(range(1000, 1600))
    client.prompt_logprobs("inplace-test", ids, 1)
    client.request(
        "POST",
        "/v1/load_lora_adapter",
        {"lora_name": "inplace-test", "lora_path": str(composite), "load_inplace": True},
    )
    after = client.prompt_logprobs("inplace-test", ids, 1)
    reference = client.prompt_logprobs(entry["name"], ids, 1)
    gap = max(abs(a - b) for a, b in zip(after, reference))
    client.unload("inplace-test")
    if gap > 1e-3:
        pytest.xfail(f"in-place reload serves cached state of the old adapter (max logprob gap {gap:.3g})")
