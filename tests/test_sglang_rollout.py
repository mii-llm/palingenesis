"""SGLang rollout engines (palingenesis.opd.sglang_rollout) on CPU: request building, output and log-prob
alignment, engine arguments, weight bucketing, cache flushes after weight updates, sleep / wake, streaming,
routing replay decoding, the server engine's weight sync, and the configuration checks. A scripted engine stands in
for sglang.Engine; the GPU checks are scripts/gpu_checks/sglang.sh."""

import asyncio
import base64
import os
import pickle
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")

from palingenesis.opd.rollout import Sampling  # noqa: E402
from palingenesis.opd.sglang_rollout import (  # noqa: E402
    OWNED_OPTIONS,
    InProcessEngine,
    SGLangRollout,
    SGLangServer,
    SGLangServerRollout,
    _ok,
    build_request,
    candidates_from_output,
    check_options,
    clamp_budget,
    decode_routed_experts,
    engine_env,
    engine_options,
    finish_type,
    launcher_env_removed,
    moe_layer_indices,
    output_tokens_and_logprobs,
    sampling_params,
    serialize_named_tensors,
    server_flags,
    weight_buckets,
)

STOP = (7, 9)


def _output(prompt, n_tokens=3, stop=True, logprob=-0.5, trim_stop=False, **meta):
    """An SGLang (skip_tokenizer_init) output for `prompt`: tokens prompt[-1]+1, +2, ..., ending with stop id 7."""
    tokens = [prompt[-1] + 1 + k for k in range(n_tokens - 1)] + ([7] if stop else [prompt[-1] + n_tokens])
    entries = [(logprob - 0.01 * k, t, None) for k, t in enumerate(tokens)]
    info = {
        "finish_reason": {"type": "stop", "matched": 7} if stop else {"type": "length", "length": n_tokens},
        "prompt_tokens": len(prompt),
        "cached_tokens": len(prompt) - 1,
        "output_token_logprobs": entries,
        **meta,
    }
    return {"output_ids": tokens[:-1] if trim_stop else tokens, "meta_info": info}


class ScriptedEngine:
    """The InProcessEngine calls, scripted. Like SGLang, a weight update waits for the requests in flight
    (its model-update lock), and it runs on the engine's event loop."""

    memory_tags = ["kv_cache", "weights", "cuda_graph"]

    def __init__(self, respond=None, delay=0.0):
        self.loop = asyncio.new_event_loop()
        self.respond = respond or (lambda request: _output(request["input_ids"]))
        self.delay = delay
        self.requests: list[dict] = []
        self.calls: list[tuple] = []
        self.inflight = 0
        self.gate: asyncio.Event | None = None

    def run(self, coroutine):
        return self.loop.run_until_complete(coroutine)

    async def agenerate(self, request):
        self.requests.append(request)
        self.inflight += 1
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
            return self.respond(request)
        finally:
            self.inflight -= 1

    def update_weights_from_tensor(self, named, flush_cache):
        async def drain():
            while self.inflight:
                await asyncio.sleep(0.001)

        self.run(drain())
        self.calls.append(("update", [n for n, _ in named], [tuple(t.shape) for _, t in named], flush_cache))
        return True, "ok"

    def release(self, tags):
        self.calls.append(("release", list(tags)))

    def resume(self, tags):
        self.calls.append(("resume", list(tags)))

    def flush_cache(self):
        self.calls.append(("flush",))
        return True

    def load_lora_adapter(self, name, path):
        self.calls.append(("load_lora", name, path))
        return True, ""

    def unload_lora_adapter(self, name):
        self.calls.append(("unload_lora", name))
        return True, ""

    def close(self):
        self.calls.append(("close",))


def _engine(**kwargs):
    scripted = kwargs.pop("engine", None) or ScriptedEngine()
    defaults = dict(max_model_len=64, bucket_mb=1, engine=scripted)
    return SGLangRollout("model", STOP, **{**defaults, **kwargs}), scripted


# ------------------------------------------------------------------ requests


def test_sampling_params_map_to_sglang():
    params = sampling_params(Sampling(0.7, 0.9, 0, 1.5), 32, [9, 7])
    assert params == {
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,  # SGLang's "off"; 0 is refused there
        "presence_penalty": 1.5,
        "stop_token_ids": [7, 9],
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "n": 1,
    }
    assert sampling_params(Sampling(1.0, 1.0, 20), 8, STOP)["top_k"] == 20


def test_requests_ask_for_logprobs_masks_and_routing_only_when_needed():
    greedy = build_request([1, 2], 8, Sampling(0.0), STOP, 64)
    assert greedy["return_logprob"] is False and "return_sampling_mask" not in greedy
    plain = build_request([1, 2], 8, Sampling(1.0), STOP, 64)
    assert plain["return_logprob"] is True and "return_sampling_mask" not in plain
    truncated = build_request([1, 2], 8, Sampling(1.0, 0.95, 20), STOP, 64, routing=True, lora="a-v3")
    assert truncated["return_sampling_mask"] is True and truncated["return_routed_experts"] is True
    assert truncated["lora_path"] == "a-v3"
    # SGLang aborts a request for the mask of a top_p-only sampling (no bound on its support): not asked for
    assert "return_sampling_mask" not in build_request([1, 2], 8, Sampling(1.0, 0.9, 0), STOP, 64)
    # Sampling.candidates: that many top log-probs per sampled token (none for greedy)
    assert "top_logprobs_num" not in truncated
    assert build_request([1], 8, Sampling(1.0, 0.95, 20, candidates=28), STOP, 64)["top_logprobs_num"] == 28
    assert "top_logprobs_num" not in build_request([1], 8, Sampling(0.0, candidates=8), STOP, 64)


def test_budget_is_clamped_to_the_context_like_vllm():
    assert clamp_budget(10, 100, 64) == 54  # SGLang refuses prompt + max_new_tokens > context; vLLM stops there
    assert clamp_budget(10, 20, 64) == 20
    assert build_request(list(range(60)), 100, Sampling(1.0), STOP, 64)["sampling_params"]["max_new_tokens"] == 4
    with pytest.raises(ValueError, match="no room"):
        clamp_budget(64, 8, 64)


def test_finish_reasons():
    assert finish_type({"finish_reason": {"type": "stop", "matched": 7}}) == "stop"
    assert finish_type({"finish_reason": {"type": "length", "length": 8}}) == "length"
    assert finish_type({"finish_reason": "stop"}) == "stop"
    with pytest.raises(RuntimeError, match="aborted"):
        finish_type({"finish_reason": {"type": "abort", "message": "out of memory"}})


# ------------------------------------------------------------------ log-prob alignment


def test_logprobs_align_with_the_sampled_tokens():
    out = _output([1, 2, 3])
    tokens, logprobs = output_tokens_and_logprobs(out, True)
    assert tokens == [4, 5, 7] and logprobs == pytest.approx([-0.5, -0.51, -0.52])
    # output_ids without the matched stop token: the log-probs' ids are the completion
    trimmed = _output([1, 2, 3], trim_stop=True)
    assert output_tokens_and_logprobs(trimmed, True)[0] == [4, 5, 7]
    # greedy: no log-probs asked for, none returned
    assert output_tokens_and_logprobs({"output_ids": [4, 7], "meta_info": {}}, False) == ([4, 7], [])


def test_misaligned_logprobs_are_an_error():
    out = _output([1, 2, 3])
    out["output_ids"] = [4, 6, 7]
    with pytest.raises(RuntimeError, match="cannot be aligned"):
        output_tokens_and_logprobs(out, True)
    with pytest.raises(RuntimeError, match="return_logprob"):
        output_tokens_and_logprobs({"output_ids": [4], "meta_info": {}}, True)


def test_truncated_samplings_use_the_renormalised_logprobs(caplog):
    out = _output([1, 2, 3], output_token_sampling_logprobs=[-0.1, -0.2, -0.3])
    assert output_tokens_and_logprobs(out, True, truncated=True)[1] == [-0.1, -0.2, -0.3]
    assert output_tokens_and_logprobs(out, True, truncated=False)[1] == pytest.approx([-0.5, -0.51, -0.52])
    with caplog.at_level("WARNING"):  # an SGLang without return_sampling_mask: the untruncated ones, said once
        assert output_tokens_and_logprobs(_output([1, 2]), True, truncated=True)[1] == pytest.approx(
            [-0.5, -0.51, -0.52]
        )


# ------------------------------------------------------------------ the sampler's candidates


def _sglang_step(logits, temperature, top_k, top_p, n, sampled=None, seed=0):
    """One sampled position as SGLang 0.5.17 reports it (sglang/srt/layers/sampler.py): p = softmax(logits / T);
    the support is the first top_k tokens of the descending sort whose preceding cumulative p is <= top_p (p not
    renormalised after top-k); the token is drawn from p over it. Returns (token, log p(token), the n top
    (log p, id, None) entries, the mask ids, log(p(token) / p(support)))."""
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / temperature, -1)
    order = torch.argsort(p, descending=True)
    sorted_p = p[order]
    before = torch.cumsum(sorted_p, 0) - sorted_p
    keep = (torch.arange(len(p)) < (top_k if top_k > 0 else len(p))) & (before <= top_p)
    support = order[keep].tolist()
    if sampled is None:
        generator = torch.Generator().manual_seed(seed)
        sampled = support[int(torch.multinomial(sorted_p[keep], 1, generator=generator))]
    if sampled not in support:  # "a backend/numeric edge": SGLang adds the sampled token to the mask
        support.append(sampled)
    mass = p[support].sum()
    top = [(float(p[i].log()), int(i), None) for i in order[:n]]
    return sampled, float(p[sampled].log()), top, support, float((p[sampled] / mass).log())


def _sglang_output(steps, mask=True):
    meta = {
        "finish_reason": {"type": "length", "length": len(steps)},
        "output_token_logprobs": [(lp, token, None) for token, lp, _, _, _ in steps],
        "output_top_logprobs": [top for _, _, top, _, _ in steps],
    }
    if mask:
        meta["output_token_sampling_mask"] = [support for _, _, _, support, _ in steps]
        meta["output_token_sampling_logprobs"] = [renormalised for _, _, _, _, renormalised in steps]
    return {"output_ids": [token for token, *_ in steps], "meta_info": meta}


def test_candidates_of_a_plain_sampling_are_the_top_logprobs():
    steps = [_sglang_step([2.0, 1.0, 0.0, -1.0, -30.0], 0.7, 0, 1.0, n=3, sampled=1)]
    out = _sglang_output(steps, mask=False)
    c = candidates_from_output(out, [1], 3, truncated=False)
    assert c.counts.tolist() == [3] and c.ids.tolist() == [0, 1, 2]
    assert c.logprobs.tolist() == pytest.approx([entry[0] for entry in steps[0][2]])
    # a sampled token below the top k is not a candidate (as with vLLM); -inf (None through JSON) is dropped
    steps = [_sglang_step([2.0, 1.0, 0.0, -1.0, -30.0], 0.7, 0, 1.0, n=2, sampled=3)]
    out = _sglang_output(steps, mask=False)
    out["meta_info"]["output_top_logprobs"][0].append((None, 4, None))
    assert candidates_from_output(out, [3], 2, truncated=False).ids.tolist() == [0, 1]
    with pytest.raises(RuntimeError, match="top log-probs for 0 of 1"):
        candidates_from_output(
            {"output_ids": [1], "meta_info": {"output_token_logprobs": [(-0.1, 1, None)]}}, [1], 3, False
        )


def test_candidates_of_a_truncated_sampling_are_sglangs_support_renormalised():
    """The candidates are the ids of SGLang's sampling mask with log p(v) - log p(S): they sum to 1 over the
    support and the sampled token's is its behaviour log-prob. SGLang's nucleus is cut on p as it is, not on the
    top-k distribution: here it keeps 3 tokens where vLLM's order (top-k, renormalise, top-p) would keep 2."""
    logits, temperature, top_k, top_p = [2.0, 1.6, 1.2, 0.8, 0.4, 0.0, -0.4, -0.8], 1.0, 3, 0.7
    p = torch.softmax(torch.tensor(logits, dtype=torch.float64), -1)
    assert p[:2].sum() <= top_p  # SGLang: the mass before the third token is within top_p
    assert p[:2].sum() / p[:3].sum() > top_p  # vLLM: renormalised over the top 3 it is not
    steps = [_sglang_step(logits, temperature, top_k, top_p, n=top_k + 8, seed=s) for s in range(4)]
    out = _sglang_output(steps)
    tokens = [token for token, *_ in steps]
    c = candidates_from_output(out, tokens, top_k + 8, truncated=True)
    assert c.counts.tolist() == [3] * 4 and c.ids.reshape(4, 3).tolist() == [[0, 1, 2]] * 4
    rows = torch.from_numpy(c.logprobs.reshape(4, 3)).double()
    torch.testing.assert_close(rows.exp().sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-6, rtol=0)
    for row, (token, _, _, _, renormalised) in zip(rows, steps):
        assert float(row[token]) == pytest.approx(renormalised, abs=1e-6)


def test_candidates_keep_a_token_sampled_outside_the_reconstructed_support():
    steps = [_sglang_step([2.0, 1.0, 0.0, -1.0, -2.0], 1.0, 2, 1.0, n=10, sampled=3)]
    c = candidates_from_output(_sglang_output(steps), [3], 10, truncated=True)
    assert c.ids.tolist() == [0, 1, 3]  # SGLang's mask holds it: the trainer renormalises over all three
    assert float(torch.from_numpy(c.logprobs).double().exp().sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(c.logprobs[2]) == pytest.approx(steps[0][4], abs=1e-6)


def test_candidates_of_a_support_that_cannot_be_scored_whole():
    logits = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5]
    # more support than candidates: the k most likely, k of them (the trainer reads k of k as possibly incomplete)
    steps = [_sglang_step(logits, 1.0, 5, 1.0, n=3, sampled=1)]
    c = candidates_from_output(_sglang_output(steps), [1], 3, truncated=True)
    assert c.counts.tolist() == [3] and c.ids.tolist() == [0, 1, 2]
    # a support token without a returned log-prob although the support fits: the sampled token is left out of the
    # candidates, so the trainer leaves it out of the loss instead of renormalising over part of its support
    steps = [_sglang_step(logits, 1.0, 3, 1.0, n=8, sampled=1)]
    out = _sglang_output(steps)
    out["meta_info"]["output_top_logprobs"][0] = [e for e in steps[0][2] if e[1] != 2]
    c = candidates_from_output(out, [1], 8, truncated=True)
    assert c.ids.tolist() == [0]
    # no mask (an older SGLang, or top_p without top_k): an error, not a rollout scored over another support
    with pytest.raises(RuntimeError, match="did not return the support"):
        candidates_from_output(_sglang_output(steps, mask=False), [1], 8, truncated=True)
    out = _sglang_output(steps)
    out["meta_info"]["output_token_sampling_logprobs"] = [None]
    with pytest.raises(RuntimeError, match="no sampling mask"):
        candidates_from_output(out, [1], 8, truncated=True)


def test_sglang_support_gives_ratio_one_in_the_trainer():
    """End to end with the trainer's renormalisation (rl.losses.sampled_logprobs): the same weights scored over
    the candidates of SGLang's support give back SGLang's behaviour log-prob (ratio 1 on-policy), and every
    token's support counts as seen whole (fewer candidates than the top_k + 8 asked for)."""
    from palingenesis.rl.config import CANDIDATE_SLACK
    from palingenesis.rl.losses import SamplerCandidates, sampled_logprobs, target_logprobs

    torch.manual_seed(5)
    vocabulary, hidden_size, temperature, top_k, top_p = 31, 8, 0.7, 6, 0.9
    weight = torch.randn(vocabulary, hidden_size, dtype=torch.float64)
    hidden = torch.randn(5, hidden_size, dtype=torch.float64)
    asked = top_k + CANDIDATE_SLACK
    steps = [_sglang_step(row @ weight.T, temperature, top_k, top_p, n=asked, seed=i) for i, row in enumerate(hidden)]
    tokens = [token for token, *_ in steps]
    behaviour = torch.tensor([renormalised for *_, renormalised in steps], dtype=torch.float64)
    candidates = candidates_from_output(_sglang_output(steps), tokens, asked, truncated=True)
    assert all(1 <= count <= top_k for count in candidates.counts.tolist())
    sampler = SamplerCandidates.build([candidates], asked, True, "cpu")
    head = torch.nn.Linear(hidden_size, vocabulary, bias=False, dtype=torch.float64)
    head.weight.data.copy_(weight)
    out = sampled_logprobs(hidden, head, torch.tensor(tokens), sampler, scale=1 / temperature)
    assert out.valid.all()
    torch.testing.assert_close(out.lp.double(), behaviour, atol=1e-5, rtol=1e-5)
    plain, _ = target_logprobs(hidden, head, torch.tensor(tokens), scale=1 / temperature)
    assert (plain.double() < behaviour - 1e-4).all()  # the bias a full-vocabulary score would carry


def test_engines_return_the_candidates_with_the_rollout():
    steps = [_sglang_step([2.0, 1.0, 0.5, 0.0, -1.0], 1.0, 3, 0.95, n=11, seed=s) for s in range(2)]
    sampling = Sampling(1.0, 0.95, 3, candidates=11)
    engine, scripted = _engine()
    scripted.respond = lambda request: _sglang_output(steps)
    (rollout,) = engine.generate([[1, 2]], [2], sampling)
    assert scripted.requests[0]["top_logprobs_num"] == 11 and scripted.requests[0]["return_sampling_mask"] is True
    assert len(rollout.candidates) == len(rollout.completion_ids) == 2
    assert rollout.logprobs == pytest.approx([step[4] for step in steps])
    for t, token in enumerate(rollout.completion_ids):  # the sampled token is in its support
        assert token in rollout.candidates.select([t]).ids.tolist()
    (plain,) = engine.generate([[1, 2]], [2], 1.0)
    assert plain.candidates is None
    engine.stream_add("a", [1, 2], 2, sampling, n=2)
    while engine.stream_pending():
        finished = engine.stream_step()
    assert [len(r.candidates) for r in finished[0][1]] == [2, 2]
    engine.close()

    class Server(ScriptedServer):
        def post(self, path, payload=None, timeout=3600.0):
            self.posts.append((path, payload))
            return [_sglang_output(steps) for _ in payload["input_ids"]]

    server = Server()
    remote = SGLangServerRollout(server, STOP, model="m", max_model_len=64, parallel=1)
    rollouts = remote.generate([[1], [2]], [2, 2], sampling)
    assert server.posts[-1][1]["top_logprobs_num"] == [11, 11]
    assert all(len(r.candidates) == 2 for r in rollouts)


# ------------------------------------------------------------------ engine arguments


def test_engine_options_follow_the_rollout_options():
    options = engine_options(
        "Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.3,
        max_model_len=4096,
        enforce_eager=True,
        seed=3,
        sleep_mode=True,
        prefix_caching=False,
        max_num_seqs=128,
        fp32_head=True,
        ssm_state_dtype="float32",
        kv_cache_dtype="fp8",
        routing_replay=True,
        extra={"attention_backend": "fa3"},
    )
    assert options == {
        "model_path": "Qwen/Qwen3-0.6B",
        "dtype": "bfloat16",
        "skip_tokenizer_init": True,
        "mem_fraction_static": 0.3,
        "context_length": 4096,
        "random_seed": 3,
        "disable_radix_cache": True,
        "enable_memory_saver": True,
        "disable_cuda_graph": True,
        "max_running_requests": 128,
        "enable_fp32_lm_head": True,
        "mamba_ssm_dtype": "float32",
        "kv_cache_dtype": "fp8_e4m3",
        "enable_return_routed_experts": True,
        "attention_backend": "fa3",
    }
    minimal = engine_options("m", gpu_memory_utilization=0.5, max_model_len=128, ssm_state_dtype="auto")
    assert "mamba_ssm_dtype" not in minimal and "max_running_requests" not in minimal
    assert minimal["disable_radix_cache"] is False


def test_owned_engine_options_are_refused():
    with pytest.raises(ValueError, match=r"rollout.sglang_args.mem_fraction_static is set by rollout.gpu_memory"):
        engine_options("m", gpu_memory_utilization=0.5, max_model_len=128, extra={"mem_fraction_static": 0.9})
    assert "enable_return_routed_experts" in OWNED_OPTIONS


def test_unknown_engine_arguments_name_the_closest():
    check_options({"mem_fraction_static": 0.3}, ["mem_fraction_static", "model_path"])
    with pytest.raises(ValueError, match=r"enable_fp32_lm_heads \(did you mean enable_fp32_lm_head\?\)"):
        check_options({"enable_fp32_lm_heads": True}, ["enable_fp32_lm_head"])


def test_server_flags():
    flags = server_flags(
        {
            "model_path": "m",
            "skip_tokenizer_init": True,
            "disable_radix_cache": False,
            "context_length": 128,
            "json_model_override_args": {"a": 1},
            "random_seed": None,
        }
    )
    assert flags == ["--skip-tokenizer-init", "--context-length", "128", "--json-model-override-args", '{"a": 1}']


def test_launcher_variables_are_hidden_while_engines_spawn(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with launcher_env_removed():
        assert "TORCHELASTIC_USE_AGENT_STORE" not in os.environ and "WORLD_SIZE" not in os.environ
    assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "True" and os.environ["WORLD_SIZE"] == "2"


def test_engines_start_without_variables_that_change_the_logprobs(monkeypatch, caplog):
    """SGLANG_RETURN_ORIGINAL_LOGPROB makes SGLang report the log-probs of the unscaled logits: an engine the
    trainer starts never sees it (the in-process engine's children copy os.environ, a server gets engine_env)."""
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("KEEP_ME", "x")
    with caplog.at_level("WARNING"):
        env = engine_env()
    assert "SGLANG_RETURN_ORIGINAL_LOGPROB" not in env and "MASTER_PORT" not in env and env["KEEP_ME"] == "x"
    assert "SGLANG_RETURN_ORIGINAL_LOGPROB" in caplog.text
    with launcher_env_removed():
        assert "SGLANG_RETURN_ORIGINAL_LOGPROB" not in os.environ and os.environ["KEEP_ME"] == "x"
    assert os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "1" and os.environ["MASTER_PORT"] == "29500"


def test_sleep_mode_engines_start_with_torchs_cuda_runtime_first(monkeypatch, tmp_path):
    """torch_memory_saver's preloaded hook needs libcudart and takes the first the loader finds: a system
    toolkit's would then be torch's too, and torch 2.11 + cu129 does not import with an older one. The wheel's
    directory goes first while the engine's processes are spawned, and the variable is put back."""
    import importlib.util

    from palingenesis.opd import sglang_rollout

    lib = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcudart.so.12").write_bytes(b"")
    spec = SimpleNamespace(submodule_search_locations=[str(lib.parent)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec if name == "nvidia.cuda_runtime" else None)
    assert sglang_rollout.cuda_runtime_library_dirs() == [str(lib)]
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/usr/local/cuda/lib64{os.pathsep}{lib}")
    with sglang_rollout.memory_saver_library_path():
        assert os.environ["LD_LIBRARY_PATH"] == f"{lib}{os.pathsep}/usr/local/cuda/lib64"
    assert os.environ["LD_LIBRARY_PATH"] == f"/usr/local/cuda/lib64{os.pathsep}{lib}"
    monkeypatch.delenv("LD_LIBRARY_PATH")
    with sglang_rollout.memory_saver_library_path():
        assert os.environ["LD_LIBRARY_PATH"] == str(lib)
    assert "LD_LIBRARY_PATH" not in os.environ
    with sglang_rollout.memory_saver_library_path(False):  # no sleep mode: nothing is preloaded
        assert "LD_LIBRARY_PATH" not in os.environ
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # a torch on the system CUDA
    with sglang_rollout.memory_saver_library_path():
        assert "LD_LIBRARY_PATH" not in os.environ


def test_results_of_engine_calls_carry_their_message():
    """Weight updates return (success, message); the LoRA calls an object whose text is error_message."""
    assert _ok((False, "shape mismatch")) == (False, "shape mismatch")
    assert _ok({"success": False, "message": "no such group"}) == (False, "no such group")
    lora = SimpleNamespace(success=False, error_message="LoRA is not enabled", loaded_adapters=None)
    assert _ok(lora) == (False, "LoRA is not enabled")
    assert _ok(SimpleNamespace(success=True, error_message=None)) == (True, "")


# ------------------------------------------------------------------ weights


def test_weight_buckets_bound_size_cast_and_share_one_buffer():
    torch.manual_seed(0)
    named = [(f"w{i}", torch.randn(256, 256)) for i in range(5)]  # 128 KiB each in bf16
    named.insert(2, ("big", torch.randn(1024, 1024)))  # 2 MiB: a bucket of its own
    named.append(("steps", torch.arange(4)))  # not floating point: kept as it is
    buckets = list(weight_buckets(iter(named), bucket_bytes=300 * 1024))
    assert [[n for n, _ in b] for b in buckets] == [["w0", "w1"], ["big"], ["w2", "w3"], ["w4", "steps"]]
    for bucket in buckets[:3]:
        assert all(t.dtype == torch.bfloat16 for _, t in bucket)
        storages = {t.untyped_storage().data_ptr() for _, t in bucket}
        assert len(storages) == 1  # views of one flat buffer: one CUDA IPC handle per bucket
    assert buckets[3][1][1].dtype == torch.int64  # mixed dtypes: not packed, not cast
    sent = dict(kv for b in buckets for kv in b)
    for name, tensor in named:
        expected = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        assert torch.equal(sent[name], expected) and sent[name].shape == tensor.shape
    assert list(weight_buckets(iter(()), 1024)) == []


def test_serialized_tensors_unpickle_to_the_same_weights():
    named = [("a", torch.arange(6, dtype=torch.bfloat16).view(2, 3)), ("b", torch.ones(4))]
    payload = serialize_named_tensors(named)
    restored = pickle.loads(base64.b64decode(payload))
    assert [n for n, _ in restored] == ["a", "b"]
    assert all(torch.equal(x, y) for (_, x), (_, y) in zip(restored, named))


def test_weight_update_flushes_the_cache_with_its_last_bucket():
    engine, scripted = _engine()
    named = [(f"layer{i}.weight", torch.randn(512, 512)) for i in range(5)]  # 512 KiB each in bf16; 1 MiB buckets
    engine.update_weights(iter(named), version=3)
    updates = [c for c in scripted.calls if c[0] == "update"]
    assert [c[1] for c in updates] == [
        ["layer0.weight", "layer1.weight"],
        ["layer2.weight", "layer3.weight"],
        ["layer4.weight"],
    ]
    assert [c[3] for c in updates] == [False, False, True]  # the radix cache is flushed once, after the last
    assert engine.version == 3


def test_a_refused_weight_update_raises_and_keeps_the_version():
    engine, scripted = _engine()
    scripted.update_weights_from_tensor = lambda named, flush_cache: (False, "shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        engine.update_weights([("w", torch.ones(2))], version=1)
    assert engine.version == 0


def test_sleep_releases_everything_and_wakes_after_an_update():
    engine, scripted = _engine(sleep_mode=True)
    engine.sleep()
    assert scripted.calls == [("release", ["kv_cache", "weights", "cuda_graph"])]
    assert engine.asleep and engine.version == -1
    with pytest.raises(RuntimeError, match="update_weights before wake"):
        engine.wake()
    engine.update_weights([("w", torch.ones(2))], version=1)  # the weights' memory comes back before the load
    engine.wake()
    assert scripted.calls[1:] == [
        ("resume", ["weights"]),
        ("update", ["w"], [(2,)], True),
        ("resume", ["kv_cache", "cuda_graph"]),
    ]
    assert not engine.asleep and engine.version == 1
    engine.sleep()
    engine.sleep()  # once
    assert scripted.calls[-1] == ("release", ["kv_cache", "weights", "cuda_graph"])
    assert sum(c[0] == "release" for c in scripted.calls) == 2


class IdleScripted(ScriptedEngine):
    """A scripted engine with SGLang's idle rule: the scheduler still holds a batch for `settle` waits after the
    last answer, and a release (or the cache flush of an update) before that is a failed assertion there."""

    def __init__(self, settle=2, **kwargs):
        super().__init__(**kwargs)
        self.settle = settle
        self.busy = 0
        self.waits = 0

    async def agenerate(self, request):
        try:
            return await super().agenerate(request)
        finally:
            self.busy = self.settle  # answered, its batch not retired yet

    def wait_idle(self, timeout):
        async def drain():
            while self.inflight:
                await asyncio.sleep(0.001)

        self.run(drain())
        self.waits += 1
        if self.settle < 0:
            return False, "Timed out waiting for idle state."
        self.busy = 0
        return True, ""

    def release(self, tags):
        assert (
            not self.inflight and not self.busy
        ), "release_memory_occupation should be called only when server is idle."
        super().release(tags)

    def update_weights_from_tensor(self, named, flush_cache):
        assert not self.inflight and not (flush_cache and self.busy), "Cache flush failed after updating weights"
        self.calls.append(("update", flush_cache))
        return True, "ok"


def test_sleep_waits_for_requests_in_flight_and_for_an_idle_scheduler():
    """pgs distill slept the engine right after its batch came back, while the scheduler still held that batch:
    SGLang asserts idleness in release_memory_occupation. sleep now waits for the requests in flight, then for the
    scheduler to be idle, and only then releases."""
    scripted = IdleScripted(delay=0.01)
    engine, _ = _engine(engine=scripted, sleep_mode=True)
    engine.generate([[1], [2]], [8, 8], 1.0)
    assert scripted.busy  # every request answered, the scheduler not idle yet
    engine.sleep()
    assert scripted.waits == 1 and scripted.calls == [("release", ["kv_cache", "weights", "cuda_graph"])]

    scripted = IdleScripted()
    engine, _ = _engine(engine=scripted, sleep_mode=True)
    scripted.gate = asyncio.Event()
    engine.stream_add("a", [1], 8, 1.0, n=2)
    assert engine.stream_step() == [] and scripted.inflight == 2  # in flight (held at the gate) when sleep is called
    scripted.loop.call_later(0.02, scripted.gate.set)
    engine.sleep()  # waits for them, then for idleness, then releases
    assert scripted.inflight == 0 and scripted.calls == [("release", ["kv_cache", "weights", "cuda_graph"])]
    assert engine.asleep
    (finished,) = engine.stream_step()  # their completions were kept
    assert finished[0] == "a" and len(finished[1]) == 2 and finished[1][0].policy_version == 0


def test_a_weight_update_waits_for_an_idle_scheduler_and_a_busy_one_is_an_error():
    scripted = IdleScripted()
    engine, _ = _engine(engine=scripted)
    engine.generate([[1]], [8], 1.0)
    engine.update_weights([("w", torch.ones(2))], version=1)  # its last bucket flushes the cache: idle first
    assert scripted.waits == 1 and scripted.calls == [("update", True)] and engine.version == 1

    stuck = IdleScripted(settle=-1)
    engine, _ = _engine(engine=stuck, sleep_mode=True)
    with pytest.raises(RuntimeError, match=r"did not become idle within 120s.*Timed out waiting for idle"):
        engine.sleep()
    assert not engine.asleep and stuck.calls == []  # nothing was released
    with pytest.raises(RuntimeError, match="the weight update was not attempted"):
        engine.update_weights([("w", torch.ones(2))], version=1)
    assert engine.version == 0

    asleep = IdleScripted()
    engine, _ = _engine(engine=asleep, sleep_mode=True)
    engine.sleep()
    engine.update_weights([("w", torch.ones(2))], version=1)  # asleep: nothing ran since the release, no wait
    assert asleep.waits == 1 and asleep.calls[1:] == [("resume", ["weights"]), ("update", True)]


def test_control_calls_give_up_when_the_scheduler_died():
    """A scheduler that failed an assertion answers nothing: release / resume / the idle wait stop waiting once the
    engine is marked dead, instead of holding their thread (and the run's shutdown) forever."""
    engine = object.__new__(InProcessEngine)
    engine.failed, engine._helper, engine.POLL_SECONDS = "", None, 0.01
    engine.loop = asyncio.new_event_loop()
    engine.engine = SimpleNamespace()

    async def never():
        await asyncio.Event().wait()

    async def answer():
        return "done"

    assert engine.run(engine._guarded(answer())) == "done"
    engine.loop.call_later(0.03, lambda: setattr(engine, "failed", "scheduler process died"))
    with pytest.raises(RuntimeError, match="scheduler process died"):
        engine.run(engine._guarded(never()))
    engine.loop.close()


def test_without_sleep_mode_the_engine_keeps_its_memory():
    engine, scripted = _engine(sleep_mode=False)
    engine.sleep()
    engine.wake()
    assert scripted.calls == [] and engine.version == 0


# ------------------------------------------------------------------ generation


def test_generate_returns_aligned_rollouts_with_versions_and_prefill_counts():
    engine, scripted = _engine()
    engine.version = 4
    prompts = [[1, 2], [3, 4, 5], [1, 2]]
    rollouts = engine.generate(prompts, [8, 8, 8], [Sampling(1.0), Sampling(1.0), Sampling(1.0)])
    assert [r.completion_ids for r in rollouts] == [[3, 4, 7], [6, 7, 7], [3, 4, 7]]
    assert all(len(r.logprobs) == len(r.completion_ids) for r in rollouts)
    assert all(r.finish_reason == "stop" and r.policy_version == 4 and r.first_version is None for r in rollouts)
    assert len(scripted.requests) == 3  # sampled duplicates are requests of their own (the radix cache shares them)
    assert engine.prompt_tokens == 7 and engine.cached_tokens == 4
    assert {tuple(r["sampling_params"]["stop_token_ids"]) for r in scripted.requests} == {(7, 9)}


def test_greedy_duplicates_are_generated_once():
    engine, scripted = _engine()
    rollouts = engine.generate([[1, 2], [1, 2], [5]], [8, 8, 8], 0.0)
    assert len(scripted.requests) == 2 and rollouts[0] is rollouts[1]
    assert rollouts[0].logprobs == [] and rollouts[2].completion_ids == [6, 7, 7]


def test_length_finish_and_images_refused():
    engine, _ = _engine(engine=ScriptedEngine(lambda r: _output(r["input_ids"], stop=False)))
    (rollout,) = engine.generate([[1]], [3], 1.0)
    assert rollout.finish_reason == "length" and rollout.completion_ids == [2, 3, 4]
    with pytest.raises(ValueError, match="token ids only"):
        engine.generate([[1]], [3], 1.0, images=[["picture.png"]])


def test_streaming_requests_join_and_return_as_they_finish():
    engine, scripted = _engine(engine=ScriptedEngine(delay=0.01))
    engine.stream_add("a", [1, 2], 8, 1.0, n=3)
    engine.stream_add("b", [5], 8, 0.0, n=2)  # greedy: one request, its rollout shared by the n waiters
    finished = {}
    for _ in range(200):
        for request_id, rollouts in engine.stream_step():
            finished[request_id] = rollouts
        if not engine.stream_pending() and len(finished) == 2:
            break
    assert len(finished["a"]) == 3 and len(finished["b"]) == 2
    assert finished["b"][0] is finished["b"][1] and finished["b"][0].logprobs == []
    assert len(scripted.requests) == 4
    assert engine.streams == {}


def test_a_weight_update_waits_for_requests_in_flight():
    """SGLang loads new weights once the requests in flight have finished: they are tagged with the version that
    sampled them all, and requests added afterwards with the new one."""
    scripted = ScriptedEngine()
    engine, _ = _engine(engine=scripted)
    scripted.gate = asyncio.Event()
    engine.stream_add("old", [1], 8, 1.0, n=1)
    assert engine.stream_step() == []  # held at the gate
    scripted.loop.call_later(0.01, scripted.gate.set)
    engine.update_weights([("w", torch.ones(2))], version=1)  # drains: "old" finishes first, under version 0
    engine.stream_add("new", [1], 8, 1.0, n=1)
    finished = {}
    while len(finished) < 2:
        finished.update(dict(engine.stream_step()))
    assert finished["old"][0].policy_version == 0 and finished["old"][0].first_version is None
    assert finished["new"][0].policy_version == 1


def test_no_request_starts_between_the_buckets_of_an_update():
    """An update is one engine call per bucket, and SGLang's model-update lock covers one call: the engine finishes
    the requests in flight before the first bucket, so none can start on weights of two versions."""

    class LockPerCall(ScriptedEngine):  # as SGLang: a call does not wait for requests that have not started
        def update_weights_from_tensor(self, named, flush_cache):
            self.calls.append(("update", self.inflight, flush_cache))
            return True, "ok"

    scripted = LockPerCall(delay=0.01)
    engine, _ = _engine(engine=scripted)
    engine.stream_add("a", [1], 8, 1.0, n=2)
    engine.stream_add("b", [2], 8, 1.0, n=1)
    named = [(f"w{i}", torch.randn(512, 512)) for i in range(3)]  # two buckets of 1 MiB
    engine.update_weights(named, version=1)
    assert scripted.calls == [("update", 0, False), ("update", 0, True)]  # nothing in flight at either bucket
    finished = dict(engine.stream_step())
    assert {r.policy_version for rollouts in finished.values() for r in rollouts} == {0}
    assert len(finished["a"]) == 2 and not engine.stream_pending()


@pytest.mark.parametrize("copies", [1, 2])
def test_a_weight_update_waits_for_every_request_when_one_of_them_failed(copies):
    """A request in flight that fails must not end the drain: the others still decode, and a bucket pushed then
    would give them weights of two versions. The failure surfaces at the next stream_step, as before."""

    class OneFails(ScriptedEngine):  # the first request for [9] fails at once; every other one takes a while
        failed = False

        async def agenerate(self, request):
            self.inflight += 1
            try:
                if request["input_ids"] == [9] and not self.failed:
                    self.failed = True
                    raise ValueError("refused")
                await asyncio.sleep(0.05)
                return self.respond(request)
            finally:
                self.inflight -= 1

        def update_weights_from_tensor(self, named, flush_cache):  # as SGLang: no wait for what has not started
            self.calls.append(("update", self.inflight, flush_cache))
            return True, "ok"

    scripted = OneFails()
    engine, _ = _engine(engine=scripted)
    if copies == 1:
        engine.stream_add("bad", [9], 8, 1.0, n=1)
        engine.stream_add("slow", [1], 8, 1.0, n=1)
    else:  # one request, two samples: the first fails while its sibling still decodes
        engine.stream_add("bad", [9], 8, 1.0, n=2)
    named = [(f"w{i}", torch.randn(512, 512)) for i in range(3)]  # two buckets of 1 MiB
    engine.update_weights(named, version=1)
    assert scripted.calls == [("update", 0, False), ("update", 0, True)]  # nothing in flight at either bucket
    assert engine.version == 1
    with pytest.raises(ValueError, match="refused"):
        engine.stream_step()
    assert engine.streams == {}


def test_generate_reports_the_first_failure_and_a_dead_engine(monkeypatch):
    def respond(request):
        if request["input_ids"] == [9]:
            raise ValueError("The input (9 tokens) is longer than the model's context length")
        return _output(request["input_ids"])

    engine, scripted = _engine(engine=ScriptedEngine(respond))
    with pytest.raises(ValueError, match="context length"):
        engine.generate([[1], [9], [2]], [8, 8, 8], 1.0)
    assert scripted.inflight == 0  # the others were cancelled, not left on the loop

    # the scheduler process died: a request would wait for it forever; the waiting call notices and raises
    monkeypatch.setattr(SGLangRollout, "POLL_SECONDS", 0.01)
    hung, scripted = _engine(engine=ScriptedEngine())
    scripted.gate = asyncio.Event()  # never set: no answer comes
    scripted.failed = ""

    def check():
        if scripted.failed:
            raise RuntimeError(scripted.failed)

    scripted.check = check
    scripted.loop.call_later(0.03, lambda: setattr(scripted, "failed", "scheduler process died"))
    with pytest.raises(RuntimeError, match="scheduler process died"):
        hung.generate([[1]], [8], 1.0)
    with pytest.raises(RuntimeError, match="scheduler process died"):
        hung.stream_step()


def test_a_dead_scheduler_ends_the_run_with_an_error_not_the_process_tree():
    """SGLang's watchdog sends SIGQUIT when the scheduler dies. The engine wrapper's handler stops the engine's own
    processes and raises in the trainer (SGLang's would kill the trainer and every child it has)."""
    stopped = []
    inner = SimpleNamespace(get_all_child_pids=lambda: [], shutdown=lambda: stopped.append("shutdown"))
    engine = object.__new__(InProcessEngine)
    engine.engine, engine.failed, engine._helper = inner, "", None
    engine.loop = asyncio.new_event_loop()
    engine.check()
    with pytest.raises(RuntimeError, match="scheduler process died"):
        engine._scheduler_died()
    assert stopped == ["shutdown"] and engine.engine is None
    assert engine._scheduler_died() is None  # a second signal (the scheduler's own, then the watchdog's)
    with pytest.raises(RuntimeError, match="scheduler process died"):
        engine.check()

    async def nothing():
        return 1

    with pytest.raises(RuntimeError, match="scheduler process died"):
        engine.run(nothing())
    with pytest.raises(RuntimeError, match="scheduler process died"):
        engine.update_weights_from_tensor([], True)
    engine.loop.close()


def test_a_failed_request_fails_the_step_and_clears_the_streams():
    def respond(request):
        if request["input_ids"] == [9]:
            return {"output_ids": [], "meta_info": {"finish_reason": {"type": "abort", "message": "too long"}}}
        return _output(request["input_ids"])

    engine, _ = _engine(engine=ScriptedEngine(respond, delay=0.0))
    engine.stream_add("ok", [1], 8, 1.0, n=1)
    engine.stream_add("bad", [9], 8, 1.0, n=1)
    with pytest.raises(RuntimeError, match="too long"):
        for _ in range(50):
            engine.stream_step()
    assert engine.streams == {} and not engine.stream_pending()


def test_generation_client_streams_through_the_sglang_engine():
    from palingenesis.rl.generation import GenerationClient

    engine, _ = _engine(engine=ScriptedEngine(delay=0.005))
    client = GenerationClient(engine, merge=True)
    assert client.streaming

    async def main():
        return await asyncio.gather(*(client.generate([1, k + 2], 8, 1.0) for k in range(4)))

    try:
        generations = asyncio.run(main())
    finally:
        client.close()
    assert [g.ids for g in generations] == [[3, 4, 7], [4, 5, 7], [5, 6, 7], [6, 7, 7]]
    assert all(len(g.logprobs) == 3 and g.finish == "stop" for g in generations)


def test_adapters_load_by_version_and_requests_name_them():
    engine, scripted = _engine()
    engine.load_adapter("policy", "/adapters/policy/v1", 1)
    engine.load_adapter("policy", "/adapters/policy/v2", 2)
    assert scripted.calls == [
        ("load_lora", "policy-v1", "/adapters/policy/v1"),
        ("load_lora", "policy-v2", "/adapters/policy/v2"),
        ("unload_lora", "policy-v1"),
    ]
    engine.generate([[1]], [4], 1.0, adapter="policy")
    assert scripted.requests[-1]["lora_path"] == "policy-v2"
    engine.generate([[1]], [4], 1.0, adapter="base")
    assert "lora_path" not in scripted.requests[-1]
    with pytest.raises(KeyError, match="not loaded"):
        engine.generate([[1]], [4], 1.0, adapter="critic")


# ------------------------------------------------------------------ routing replay


def test_routed_experts_decode_to_the_moe_layers():
    import numpy as np

    positions, layers, top_k = 4, 3, 2
    array = np.arange(positions * layers * top_k, dtype=np.int32).reshape(positions, layers, top_k)
    encoded = base64.b64encode(array.tobytes()).decode()
    assert np.array_equal(decode_routed_experts(encoded, positions, top_k), array)
    assert np.array_equal(decode_routed_experts(encoded, positions, top_k, [1, 2]), array[:, [1, 2]])
    assert np.array_equal(decode_routed_experts(array.tolist(), positions, top_k, [0, 1, 2]), array)
    assert decode_routed_experts(encoded, positions + 1, top_k) is None  # not that layout: no replay
    assert decode_routed_experts(None, positions, top_k) is None


def test_moe_layer_indices():
    qwen = SimpleNamespace(num_hidden_layers=4, num_experts=8, mlp_only_layers=[0], decoder_sparse_step=1)
    assert moe_layer_indices(qwen) == [1, 2, 3]
    deepseek = SimpleNamespace(num_hidden_layers=5, n_routed_experts=8, first_k_dense_replace=2, moe_layer_freq=1)
    assert moe_layer_indices(deepseek) == [2, 3, 4]
    assert moe_layer_indices(SimpleNamespace(num_hidden_layers=4)) is None
    wrapped = SimpleNamespace(text_config=qwen)
    assert moe_layer_indices(wrapped) == [1, 2, 3]


def test_rollouts_carry_the_routing_of_every_forwarded_position():
    import numpy as np

    def respond(request):
        out = _output(request["input_ids"])
        positions = len(request["input_ids"]) + len(out["output_ids"]) - 1
        routed = np.full((positions, 2, 2), 5, dtype=np.int32)
        out["meta_info"]["routed_experts"] = base64.b64encode(routed.tobytes()).decode()
        return out

    engine, scripted = _engine(engine=ScriptedEngine(respond))
    engine.routing = (2, [0, 1])
    (rollout,) = engine.generate([[1, 2]], [8], 1.0)
    assert rollout.routed_experts.shape == (2 + 3 - 1, 2, 2)
    assert scripted.requests[0]["return_routed_experts"] is True


# ------------------------------------------------------------------ server engine


class ScriptedServer:
    def __init__(self, launched=True, fail=""):
        self.process = object() if launched else None
        self.url = "http://127.0.0.1:30000"
        self.posts: list[tuple[str, dict]] = []
        self.fail = fail

    def post(self, path, payload=None, timeout=0):
        self.posts.append((path, payload))
        if path == self.fail:
            return {"success": False, "message": "refused"}
        if path == "/generate":
            return [_output(ids) for ids in payload["input_ids"]]
        return {"success": True, "message": ""}

    def info(self):
        return {"tp_size": self.ranks}

    ranks = 1


def test_server_generates_in_parallel_batches_in_order():
    server = ScriptedServer()
    engine = SGLangServerRollout(server, STOP, model="m", max_model_len=64, parallel=2)
    rollouts = engine.generate([[1], [2], [3]], [8, 8, 8], Sampling(1.0, 0.9, 20))
    assert [r.completion_ids for r in rollouts] == [[2, 3, 7], [3, 4, 7], [4, 5, 7]]
    sizes = sorted(len(p["input_ids"]) for path, p in server.posts if path == "/generate")
    assert sizes == [1, 2]
    assert all(p["return_sampling_mask"] == [True] * len(p["input_ids"]) for _, p in server.posts)
    with pytest.raises(ValueError, match="one policy"):
        engine.generate([[1]], [8], 1.0, adapter="critic")


def test_server_tensor_sync_sends_buckets_and_flushes_last():
    server = ScriptedServer()
    engine = SGLangServerRollout(server, STOP, model="m", max_model_len=64, bucket_mb=1)
    assert engine.weight_sync == "tensor"  # a launched server shares the trainer's GPU
    named = [(f"w{i}", torch.randn(512, 512)) for i in range(3)]
    engine.update_weights(named, version=2)
    posts = [p for path, p in server.posts if path == "/update_weights_from_tensor"]
    assert [p["flush_cache"] for p in posts] == [False, True]
    restored = pickle.loads(base64.b64decode(posts[0]["serialized_named_tensors"][0]))
    assert [n for n, _ in restored] == ["w0", "w1"] and restored[0][1].dtype == torch.bfloat16
    assert engine.version == 2
    failing = SGLangServerRollout(ScriptedServer(fail="/update_weights_from_tensor"), STOP, model="m", max_model_len=64)
    with pytest.raises(RuntimeError, match="refused"):
        failing.update_weights(named, version=3)
    assert failing.version == 0


def test_server_tensor_sync_sends_one_payload_per_tensor_parallel_rank():
    """SGLang's ranks each read serialized_named_tensors[their rank]."""
    server = ScriptedServer()
    server.ranks = 2
    engine = SGLangServerRollout(server, STOP, model="m", max_model_len=64)
    engine.update_weights([("w", torch.ones(2))], version=1)
    (post,) = [p for path, p in server.posts if path == "/update_weights_from_tensor"]
    assert len(post["serialized_named_tensors"]) == 2


def test_server_updates_wait_for_the_generate_call_in_flight():
    """An update is several requests (one per bucket): a completion must not be sampled between two of them."""
    server = ScriptedServer()
    started, release = threading.Event(), threading.Event()
    post = server.post

    def slow_post(path, payload=None, timeout=0):
        if path == "/generate":
            started.set()
            assert release.wait(5)
        return post(path, payload, timeout)

    server.post = slow_post
    engine = SGLangServerRollout(server, STOP, model="m", max_model_len=64, parallel=1)
    generating = threading.Thread(target=engine.generate, args=([[1]], [8], 1.0))
    generating.start()
    assert started.wait(5)
    updating = threading.Thread(target=engine.update_weights, args=([("w", torch.ones(2))], 1))
    updating.start()
    updating.join(0.05)
    assert updating.is_alive() and engine.version == 0
    assert not [path for path, _ in server.posts if path.startswith("/update")]
    release.set()
    generating.join(5)
    updating.join(5)
    assert engine.version == 1 and [path for path, _ in server.posts][-1] == "/update_weights_from_tensor"


def test_a_server_that_exited_is_named_with_its_log():
    server = object.__new__(SGLangServer)
    server.model, server.url, server.log_path = "m", "http://127.0.0.1:9", "/out/sglang.log"
    server.process = SimpleNamespace(poll=lambda: -9, returncode=-9)
    with pytest.raises(RuntimeError, match=r"exited with code -9 during /generate; see /out/sglang.log"):
        server.request("/generate", {}, timeout=2)
    server.process = None  # a server at rollout.url: the connection error itself
    with pytest.raises(RuntimeError, match="SGLang server /generate at http://127.0.0.1:9"):
        server.request("/generate", {}, timeout=2)


def test_server_auto_sync_is_distributed_for_a_running_server():
    engine = SGLangServerRollout(ScriptedServer(launched=False), STOP, model="m", max_model_len=64)
    assert engine.weight_sync == "distributed"
    with pytest.raises(ValueError, match="weight_sync_dir"):
        SGLangServerRollout(ScriptedServer(), STOP, model="m", max_model_len=64, weight_sync="disk")


def test_server_disk_sync_writes_a_checkpoint_and_drops_the_previous(tmp_path):
    from safetensors.torch import load_file

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "toy"}')
    server = ScriptedServer()
    engine = SGLangServerRollout(
        server, STOP, model=str(model), max_model_len=64, weight_sync="disk", sync_dir=str(tmp_path / "sync")
    )
    engine.update_weights([("w", torch.ones(2, 2))], version=1)
    engine.update_weights([("w", torch.full((2, 2), 2.0))], version=2)
    assert [p["model_path"] for path, p in server.posts] == [str(tmp_path / "sync/v1"), str(tmp_path / "sync/v2")]
    assert not (tmp_path / "sync/v1").exists()
    weights = load_file(str(tmp_path / "sync/v2/model.safetensors"))
    assert torch.equal(weights["w"], torch.full((2, 2), 2.0, dtype=torch.bfloat16))
    assert (tmp_path / "sync/v2/config.json").read_text() == '{"model_type": "toy"}'
    engine.close()  # the run's last checkpoint goes too
    assert not (tmp_path / "sync/v2").exists() and engine.disk_versions == []

    refusing = SGLangServerRollout(
        ScriptedServer(fail="/update_weights_from_disk"),
        STOP,
        model=str(model),
        max_model_len=64,
        weight_sync="disk",
        sync_dir=str(tmp_path / "sync"),
    )
    with pytest.raises(RuntimeError, match="refused"):
        refusing.update_weights([("w", torch.ones(2, 2))], version=3)
    assert refusing.version == 0 and (tmp_path / "sync/v3").exists()
    refusing.close()  # a refused checkpoint is not left behind either
    assert not (tmp_path / "sync/v3").exists()


# ------------------------------------------------------------------ trainers and configuration


def test_fsdp_weight_push_pauses_the_sglang_engine():
    from palingenesis.rl.trainer import RLTrainer

    engine_lock, held = threading.Lock(), []
    trainer = object.__new__(RLTrainer)
    trainer.engine = object.__new__(SGLangRollout)
    trainer.engine.update_weights = lambda named, version: held.append((engine_lock.locked(), version))
    trainer.model = torch.nn.Linear(1, 1)
    trainer.parallel = SimpleNamespace(full_parameters=lambda model: iter(()))
    trainer.pipeline = SimpleNamespace(lock=threading.RLock(), client=SimpleNamespace(paused=lambda: engine_lock))
    trainer._push_weights(3)
    assert held == [(True, 3)]


def _rl(**options):
    from palingenesis.rl.config import RLConfig

    config = RLConfig()
    for key, value in {"model.policy": "m", "data.dataset": "d.jsonl", "rewards.r.fn": "math"}.items():
        config.set(key, value)
    for key, value in options.items():
        config.set(key, value)
    return config


@pytest.mark.parametrize(
    "options, message",
    [
        ({"rollout.backend": "sglang", "rollout.sglang_args": {"mem_fraction_static": 0.5}}, "gpu_memory_utilization"),
        ({"rollout.backend": "sglang", "rollout.vllm_args": {"max_num_batched_tokens": 512}}, "sglang_args"),
        ({"rollout.backend": "vllm", "rollout.sglang_args": {"attention_backend": "fa3"}}, "need rollout.backend"),
        ({"rollout.backend": "sglang", "rollout.weight_sync": "disk"}, "sglang_server only"),
        ({"rollout.backend": "sglang_server", "rollout.weight_sync": "nfs"}, "weight_sync must be one of"),
        ({"rollout.backend": "sglang_server", "rollout.weight_sync_dir": "/x"}, "weight_sync disk only"),
        ({"rollout.backend": "sglang_server", "rollout.weight_sync": "distributed"}, "needs a server on another GPU"),
        ({"rollout.backend": "sglang", "rollout.weight_bucket_mb": 0}, "weight_bucket_mb"),
        ({"rollout.backend": "sglang", "model.multimodal": True}, "token ids only"),
        ({"rollout.backend": "vllm", "rollout.url": "http://x:1"}, "rollout.url"),
    ],
)
def test_rl_config_refuses_wrong_sglang_settings(options, message):
    from palingenesis.rl.config import RLConfigError

    with pytest.raises(RLConfigError, match=message):
        _rl(**options).validate()


def test_rl_config_accepts_the_sglang_backends(monkeypatch):
    from palingenesis.rl.config import RLConfigError

    _rl(**{"rollout.backend": "sglang", "rollout.sglang_args": {"attention_backend": "fa3"}}).validate()
    _rl(
        **{"rollout.backend": "sglang_server", "rollout.weight_sync": "disk", "rollout.weight_sync_dir": "/s"}
    ).validate()
    _rl(**{"rollout.backend": "sglang", "rollout.max_staleness": 1}).validate()
    _rl(
        **{"rollout.backend": "sglang_server", "rollout.weight_sync": "distributed", "rollout.url": "http://h:1"}
    ).validate()
    warnings = _rl(**{"rollout.backend": "sglang_server", "rollout.url": "http://h:1", "rollout.fp32_head": True})
    assert any("enable-fp32-lm-head" in w for w in warnings.validate())
    # truncated sampling and score centering: the engines return the sampler's candidates (SGLang's support)
    from palingenesis.rl.config import training_sampling

    truncated = _rl(**{"rollout.backend": "sglang", "rollout.top_p": 0.95, "rollout.top_k": 20})
    assert not truncated.validate()
    assert training_sampling(truncated).candidates == 28
    centered = _rl(**{"rollout.backend": "sglang_server", "loss.score_centering": "topk"})
    assert not centered.validate() and training_sampling(centered).candidates == 8
    running = _rl(
        **{"rollout.backend": "sglang_server", "rollout.url": "http://h:1", "rollout.top_k": 20, "rollout.top_p": 0.9}
    )
    assert any("SGLang >= 0.5.12" in w for w in running.validate())
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RLConfigError, match="sglang_server is single-process"):
        _rl(**{"rollout.backend": "sglang_server"}).validate()


def test_opd_config_takes_the_sglang_backends():
    from palingenesis.opd.config import OPDConfig, OPDConfigError

    def config(**options):
        c = OPDConfig()
        base = {"model.student": "s", "teachers.t.model": "t", "sources.d.path": "d.jsonl", "sources.d.teacher": "t"}
        for key, value in {**base, **options}.items():
            c.set(key, value, "test")
        return c

    config(**{"rollout.backend": "sglang"}).validate()
    config(**{"rollout.backend": "sglang_server", "rollout.url": "http://h:1"}).validate()
    with pytest.raises(OPDConfigError, match="rollout.url applies"):
        config(**{"rollout.backend": "sglang", "rollout.url": "http://h:1"}).validate()
    with pytest.raises(OPDConfigError, match="context_length is set by rollout.max_model_len"):
        config(**{"rollout.backend": "sglang", "rollout.sglang_args": {"context_length": 99}}).validate()


def _recording(monkeypatch, module, name="SGLangRollout"):
    """Replace `module.<name>` with a recorder whose calls must bind the real constructor's signature."""
    import inspect

    from palingenesis.opd import sglang_rollout

    real = getattr(sglang_rollout, name)
    calls = []

    def build(*args, **kwargs):
        inspect.signature(real).bind(*args, **kwargs)  # a wrong keyword fails here, as it would on a GPU
        calls.append(kwargs)
        return SimpleNamespace(kind=name, kwargs=kwargs)

    monkeypatch.setattr(module, name, build)
    return calls


def test_trainers_and_screen_build_the_sglang_engines_with_their_options(monkeypatch, tmp_path):
    from palingenesis.opd import sglang_rollout
    from palingenesis.rl import screen
    from palingenesis.rl import trainer as rl_trainer
    from palingenesis.rl.trainer import RLTrainer

    config = _rl(
        **{
            "rollout.backend": "sglang",
            "rollout.sglang_args": {"attention_backend": "fa3"},
            "rollout.weight_bucket_mb": 64,
            "rollout.kv_cache_dtype": "fp8",
            "train.output_dir": str(tmp_path),
        }
    )
    calls = _recording(monkeypatch, rl_trainer)
    trainer = object.__new__(RLTrainer)
    trainer.config, trainer.stop_ids, trainer.routing_replay = config, (7,), False
    trainer._sglang_engine(rank=1)
    (kwargs,) = calls
    assert kwargs["sleep_mode"] is True and kwargs["bucket_mb"] == 64 and kwargs["seed"] == config.train.seed + 1
    assert kwargs["engine_kwargs"] == {"attention_backend": "fa3"} and kwargs["kv_cache_dtype"] == "fp8"

    server_calls = _recording(monkeypatch, rl_trainer, "SGLangServer")
    monkeypatch.setattr(rl_trainer, "server_option_errors", lambda options: [])
    config.set("rollout.backend", "sglang_server")
    trainer._sglang_server()
    (server,) = server_calls
    assert server["options"]["context_length"] == config.rollout.max_model_len
    assert server["options"]["attention_backend"] == "fa3" and server["log_path"].endswith("sglang.log")

    screen_calls = _recording(monkeypatch, sglang_rollout)
    config.set("rollout.backend", "sglang")
    screen._engine(config, "m", (7,), 0)
    assert screen_calls and screen_calls[0]["engine_kwargs"] == {"attention_backend": "fa3"}


# ------------------------------------------------------------------ the GPU check's install plan


def _checks():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts/gpu_checks/sglang_checks.py"
    spec = importlib.util.spec_from_file_location("sglang_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Requires-Dist of sglang 0.5.17 (the CUDA-specific lines and a few others)
SGLANG_REQUIRES = [
    "aiohttp",
    "cuda-python>=13.0",
    "flashinfer_python[cu13]==0.6.15.post1",
    "humming-kernels[cu13]==0.1.10",
    "nvidia-cutlass-dsl[cu13]==4.6.0",
    "sgl-deep-gemm==0.1.5.post1",
    "sglang-kernel==0.4.5",
    "torch==2.11.0",
    "torchaudio==2.11.0",
    'torchcodec==0.11.1; sys_platform != "linux" or platform_machine != "aarch64"',
    "torchvision",
    "transformers==5.12.1",
    'pytest; extra == "test"',
    'torch==9.9.9; extra == "dev"',
]


def test_gpu_check_installs_one_cuda_build_of_everything():
    """SGLang's wheel declares the CUDA 13 builds: on a CUDA 12.9 driver every one of them is replaced in the
    resolution itself (installing first and swapping afterwards pulled the CUDA 13 libraries back)."""
    checks = _checks()
    assert checks.install_overrides(SGLANG_REQUIRES, "cu129") == [
        "cuda-python>=12,<13",
        "flashinfer-python[cu12]==0.6.15.post1",
        "humming-kernels[cu12]==0.1.10",
        "nvidia-cutlass-dsl==4.6.0",  # no cu12 extra: the plain package is the CUDA 12 one
        "sgl-deep-gemm==0.1.5.post1+cu129",
        "sglang-kernel==0.4.5+cu129",
        "torch==2.11.0+cu129",
        "torchaudio==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
    ]
    # the wheel's own CUDA: only torch is held to its index
    assert checks.install_overrides(SGLANG_REQUIRES, "cu130") == [
        "torch==2.11.0+cu130",
        "torchaudio==2.11.0+cu130",
        "torchvision==0.26.0+cu130",
    ]
    assert checks.cuda_version("cu129") == "12.9" and checks.cuda_version("cu130") == "13.0"
    # uv takes pre-releases only for requirements given to it directly: those of SGLang's are named in the plan
    requires = [*SGLANG_REQUIRES, "flash-attn-4>=4.0.0b18", 'x==1.0rc1; extra == "dev"', "blobfile==3.0.0"]
    assert checks.prerelease_requirements(requires) == ["flash-attn-4>=4.0.0b18"]
    with pytest.raises(ValueError, match="CUDA_INDEX"):
        checks.cuda_version("12.9")
    with pytest.raises(ValueError, match="does not pin torch"):
        checks.install_overrides(["torch>=2.11"], "cu129")


def test_gpu_check_refuses_a_resolution_of_another_cuda():
    checks = _checks()
    good = [
        "sglang==0.5.17",
        "torch==2.11.0+cu129",
        "torchaudio==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
        "sglang-kernel==0.4.5+cu129",
        "sgl-deep-gemm==0.1.5.post1+cu129",
        "nvidia-cudnn-cu12==9.17.1.4",
        "nvidia-cuda-nvdisasm==13.3.1  # unsuffixed tools are fine",
    ]
    assert checks.lock_problems(good, "cu129") == []
    bad = ["torch==2.11.0", "nvidia-cudnn-cu13==9.19.0.56", "nvidia_nccl_cu13==2.28.9", "sglang-kernel==0.4.5"]
    problems = checks.lock_problems(bad, "cu129")
    assert "nvidia-cudnn-cu13 is a CUDA 13 package" in problems and "nvidia-nccl-cu13 is a CUDA 13 package" in problems
    assert (
        "torch==2.11.0 is not the cu129 build" in problems and "sglang-kernel==0.4.5 is not the cu129 build" in problems
    )
    assert "sglang is not in the resolution" in problems
    assert checks.lock_problems(["sglang==0.5.17", "torch==2.11.0", "nvidia-cudnn-cu13==9.19.0.56"], "cu130") == []


def test_gpu_check_install_verify_names_what_is_wrong(tmp_path, capsys, monkeypatch):
    """The check fails and says why: the wrong torch build, no visible GPU, SGLang missing. What it probes (torch's
    CUDA build, the GPU, the sglang import) is set here, so the messages are the same on any machine."""
    torch = pytest.importorskip("torch")
    checks = _checks()
    lock = tmp_path / "lock.txt"
    lock.write_text("sglang==0.5.17\ntorch==2.11.0\n")
    args = SimpleNamespace(lock=str(lock), lock_only=True, cuda="cu129", sglang_version="0.5.17")
    with pytest.raises(SystemExit) as stopped:
        checks.install_verify(args)
    assert stopped.value.code == 1 and "torch==2.11.0 is not the cu129 build" in capsys.readouterr().err

    args.lock_only, args.lock = False, ""
    monkeypatch.setitem(sys.modules, "sglang", None)  # `import sglang` fails, installed or not
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)  # a CPU-only build
    with pytest.raises(SystemExit) as stopped:
        checks.install_verify(args)
    err = capsys.readouterr().err
    assert stopped.value.code == 1 and "import sglang: " in err and "None in sys.modules" in err
    assert f"torch {torch.__version__} is built for CUDA None, not 12.9" in err

    monkeypatch.setattr(torch.version, "cuda", "12.9")  # the build asked for, on a machine without a visible GPU
    with pytest.raises(SystemExit) as stopped:
        checks.install_verify(args)
    err = capsys.readouterr().err
    assert stopped.value.code == 1 and "import sglang: " in err and "None in sys.modules" in err
    assert f"torch {torch.__version__} does not see a GPU (driver too old for cu129?)" in err


def _report(checks, tmp_path, capsys, check, record, **options):
    import json

    path = tmp_path / f"{check}.json"
    path.write_text(json.dumps(record))
    args = SimpleNamespace(check=check, inputs=[str(path)], max_delta=0.08, min_ratio=0.7, exit_code=0, **options)
    checks.report(args)
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_gpu_check_weight_update_is_judged_by_logprobs_not_by_identical_greedy_text(tmp_path, capsys):
    """Greedy completions do not repeat exactly with the same weights (batches change with timing, bf16 logits
    tie): on the first GPU run 62% did. A restored engine is one that samples the original log-probs again, and a
    perturbation counts only when the trainer's own log-probs moved by it."""
    checks = _checks()
    record = {
        "model": "m",
        "mode": "colocated",
        "changed_share": 1.0,
        "repeat_share": 0.625,
        "restored_share": 0.625,
        "perturbation_dlogprob": 0.45,
        "mean_abs_dlogprob_new_weights": 0.02,
        "mean_abs_dlogprob_old_weights": 0.42,
        "mean_abs_dlogprob_restored": 0.011,
    }
    assert _report(checks, tmp_path, capsys, "update", record)["status"] == "PASS"
    for change in (
        {"perturbation_dlogprob": 0.03},  # a perturbation the log-probs cannot tell from the bf16 mismatch
        {"mean_abs_dlogprob_new_weights": 0.3},  # the engine does not sample the new weights
        {"mean_abs_dlogprob_restored": 0.3},  # nor the old ones after they are pushed back
        {"restored_share": 0.2},  # far below what the same weights repeat
    ):
        assert _report(checks, tmp_path, capsys, "update", {**record, **change})["status"] == "FAIL", change


def test_gpu_check_sleep_lines(tmp_path, capsys):
    checks = _checks()
    saver = {"model": "m", "tokens": 16, "released_gib": 20.0, "resumed_gib": 20.0}
    assert _report(checks, tmp_path, capsys, "saver", saver) == {"check": "0_memory_saver", "status": "PASS", **saver}
    assert _report(checks, tmp_path, capsys, "saver", {**saver, "released_gib": 0.0})["status"] == "FAIL"
    sleep = {
        "model": "m",
        "repeat_share": 0.625,
        "released_share": 0.95,
        "same_greedy_share": 0.5,
        "released_again_share": 0.93,
        "same_greedy_again_share": 0.625,
        "mean_abs_dlogprob_after_wake": 0.012,
    }
    assert _report(checks, tmp_path, capsys, "sleep", sleep)["status"] == "PASS"
    assert (
        _report(checks, tmp_path, capsys, "sleep", {**sleep, "mean_abs_dlogprob_after_wake": 0.5})["status"] == "FAIL"
    )
    assert _report(checks, tmp_path, capsys, "sleep", {**sleep, "released_again_share": 0.3})["status"] == "FAIL"


def test_gpu_check_smoke_configs_load_and_leave_rows_to_train_on(tmp_path):
    """The 10-step smokes' configs: valid for pgs rl / pgs distill, and the distillation source keeps training rows
    after its dev split (the default dev_size of 200 held out the whole 64-row file)."""
    import random

    from palingenesis.opd.config import OPDConfig
    from palingenesis.opd.sources import ChatMessagesSource
    from palingenesis.rl.config import RLConfig

    checks = _checks()
    for kind in ("rl", "distill"):
        args = SimpleNamespace(
            kind=kind,
            model="m",
            teacher="t",
            backend="sglang",
            workdir=str(tmp_path / kind),
            out=str(tmp_path / f"{kind}.yaml"),
        )
        checks.smoke_config(args)
    rl = RLConfig.from_yaml(tmp_path / "rl.yaml")
    rl.validate()
    assert rl.rollout.backend == "sglang" and rl.train.steps == 10
    distill = OPDConfig.from_yaml(tmp_path / "distill.yaml")
    distill.validate()
    source = distill.sources["arith"]
    built = ChatMessagesSource(source, distill.train.eval_samples, distill.train.seed, random.Random(0))
    assert len(built.dev_rows) == source.dev_size == 8
    assert len(built.train_rows) >= 4 * distill.rollout.batch_prompts


def test_gpu_check_smoke_report_reads_the_metrics_line_by_line(tmp_path, capsys):
    """metrics.jsonl has one JSON object per step (reading it as one document failed: 'no result'); a last line cut
    off by a killed run is left out."""
    import json

    checks = _checks()
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": step, "train/loss": 1.0 / (step + 1), "rollout_tok_s": 900.0} for step in range(10)]
    rows[0]["policy/mismatch_k3_step0"] = 0.001
    path.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"step": 10, "train/lo')
    args = SimpleNamespace(check="smoke_rl", inputs=[str(path)], exit_code=0, max_delta=0.08, min_ratio=0.7)
    checks.report(args)
    line = json.loads(capsys.readouterr().out.strip())
    assert line["check"] == "4_smoke_rl" and line["status"] == "PASS" and line["steps_logged"] == 10
    assert line["mismatch_k3_step0"] == 0.001 and line["last"]["train/loss"] == 0.1
    args.exit_code = 1  # the run failed: its metrics do not make it pass
    checks.report(args)
    assert json.loads(capsys.readouterr().out.strip())["status"] == "FAIL"
    args.exit_code, args.inputs = 0, [str(tmp_path / "missing.jsonl")]
    checks.report(args)
    line = json.loads(capsys.readouterr().out.strip())
    assert line["status"] == "FAIL" and line["steps_logged"] == 0
