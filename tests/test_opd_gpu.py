"""OPD on a GPU with vLLM: a rollout engine holds exactly the trainer's weights after update_weights.

Skipped without CUDA and vLLM (the server case also needs ray, which vLLM 0.26's
CUDA IPC weight transfer imports). Downloads Qwen3.5-0.8B and Qwen3-0.6B.

Log-probabilities of fixed probe sequences from the trainer's model (fp32 master
weights, bf16 autocast) and from vLLM (bf16) are compared three times: with the
initial weights (bf16-level difference), after perturbing every trainer parameter
(large difference: vLLM still has the old weights), and after update_weights
(back to the initial level). A parameter the update fails to load, e.g. under a
name the engine does not map, keeps the difference large.
"""

import importlib.util
import sys

import pytest

sys.path.insert(0, "src")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None:
    pytest.skip("needs a CUDA GPU and vLLM", allow_module_level=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from palingenesis.logits import final_hidden_states, output_head  # noqa: E402
from palingenesis.opd.rollout import (  # noqa: E402
    VLLMColocateRollout,
    VLLMServer,
    VLLMServerRollout,
    checkpoint_named_parameters,
)

CONVERSATIONS = [
    ("What is 17 * 23?", "17 * 23 = 391. Answer: 391"),
    ("Write a haiku about autumn.", "Crimson leaves drifting,\nquiet rivers carry them\ntoward the winter sea."),
    (
        "Explain what a hash map is in one sentence.",
        "A hash map stores key-value pairs and finds a value by hashing its key.",
    ),
]


def probes(tok):
    return [
        tok.encode(
            tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + a,
            add_special_tokens=False,
        )
        for q, a in CONVERSATIONS
    ]


@torch.no_grad()
def trainer_logprobs(model, seqs):
    head, out = output_head(model), []
    for s in seqs:
        ids = torch.tensor([s], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(final_hidden_states(model, ids)[0, :-1]).float()
        out.append(torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None]).squeeze(1).cpu())
    return torch.cat(out)


def check_sync(model, engine, engine_logprobs, seqs, sleep: bool):
    def gap():
        return (trainer_logprobs(model, seqs) - engine_logprobs()).abs().mean().item()

    base = gap()
    g = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g, device="cuda") * p.std() * 0.05)
    before = gap()
    if sleep:
        engine.sleep()
    engine.update_weights(checkpoint_named_parameters(model), 1)
    engine.wake()
    after = gap()
    assert after < 1.5 * base, (base, before, after)
    assert before > 10 * after, (base, before, after)


def test_colocated_engine_receives_the_weights():
    """Qwen3.5: transformers loads it under other parameter names than its checkpoint's.
    (One sleep-mode vLLM engine per process: vLLM's CuMem allocator allows no second.)"""
    from vllm import SamplingParams

    name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(name)
    seqs = probes(tok)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda()
    engine = VLLMColocateRollout(
        name,
        (tok.eos_token_id,),
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        enforce_eager=False,
        seed=0,
        sleep_mode=True,
    )

    def engine_logprobs():
        outs = engine.llm.generate(
            [{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False
        )
        return torch.tensor([o.prompt_logprobs[i][s[i]].logprob for o, s in zip(outs, seqs) for i in range(1, len(s))])

    check_sync(model, engine, engine_logprobs, seqs, sleep=True)


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="vLLM 0.26's CUDA IPC weight transfer imports ray")
def test_server_receives_the_weights_over_cuda_ipc(tmp_path):
    name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(name)
    seqs = probes(tok)
    server = VLLMServer(
        name,
        args=(
            "--gpu-memory-utilization",
            "0.3",
            "--max-model-len",
            "1024",
            "--weight-transfer-config",
            '{"backend": "ipc"}',
        ),
        log_path=str(tmp_path / "server.log"),
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda()
        engine = VLLMServerRollout(server, (tok.eos_token_id,))

        def engine_logprobs():
            choices = server.complete(seqs, parallel=1, max_tokens=1, temperature=1.0, prompt_logprobs=0)
            return torch.tensor(
                [c["prompt_logprobs"][i][str(s[i])]["logprob"] for c, s in zip(choices, seqs) for i in range(1, len(s))]
            )

        check_sync(model, engine, engine_logprobs, seqs, sleep=False)
    finally:
        server.close()
