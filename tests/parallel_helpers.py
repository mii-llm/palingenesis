"""Tiny random-weight models of every architecture the tensor / pipeline plans split, and a runner for
`pgs train`'s loop over several gloo processes on the CPU (tests/test_tensor_parallel.py,
tests/test_pipeline_parallel.py)."""

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

COMMON = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, head_dim=8)


def _llama(vocab):
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(LlamaConfig(vocab_size=vocab, intermediate_size=64, max_position_embeddings=512, **COMMON))


def _mistral(vocab):
    from transformers import MistralConfig, MistralForCausalLM

    return MistralForCausalLM(
        MistralConfig(vocab_size=vocab, intermediate_size=64, max_position_embeddings=512, **COMMON)
    )


def _qwen2(vocab):  # q / k / v biases
    from transformers import Qwen2Config, Qwen2ForCausalLM

    c = dict(COMMON)
    c.pop("head_dim")
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, intermediate_size=64, max_position_embeddings=512, **c))


def _qwen3(vocab, tie=True):  # q_norm / k_norm, tied embeddings (as Qwen3-0.6B)
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab, intermediate_size=64, max_position_embeddings=512, tie_word_embeddings=tie, **COMMON
        )
    )


def _gemma(vocab):  # embeddings scaled in the backbone's forward
    from transformers import GemmaConfig, GemmaForCausalLM

    return GemmaForCausalLM(GemmaConfig(vocab_size=vocab, intermediate_size=64, max_position_embeddings=512, **COMMON))


def _gemma2(vocab):  # attention and final logit soft-capping, sliding windows, four norms per layer
    from transformers import Gemma2Config, Gemma2ForCausalLM

    return Gemma2ForCausalLM(
        Gemma2Config(
            vocab_size=vocab,
            intermediate_size=64,
            max_position_embeddings=512,
            sliding_window=16,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0,
            **COMMON,
        )
    )


def _gemma3(vocab):  # four norms per layer, scaled embeddings, sliding-window layers, q_norm / k_norm
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=vocab,
            intermediate_size=64,
            max_position_embeddings=512,
            sliding_window=16,
            layer_types=["sliding_attention", "full_attention"] * 2,
            **COMMON,
        )
    )


def _qwen3_5(vocab):  # Gated DeltaNet + gated full attention (Qwen3.5), tied embeddings
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=vocab,
            intermediate_size=64,
            layer_types=["linear_attention", "full_attention"] * 2,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            tie_word_embeddings=True,
            **COMMON,
        )
    )
    for m in model.modules():  # slow decay: the state carries across tokens, A_log gets real gradients
        if hasattr(m, "A_log"):
            with torch.no_grad():
                m.A_log.copy_(torch.empty_like(m.A_log).uniform_(0.01, 0.1).log())
    return model


def _qwen3_moe(vocab):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=vocab,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            norm_topk_prob=True,
            **COMMON,
        )
    )


ARCHITECTURES = {
    "llama": _llama,
    "mistral": _mistral,
    "qwen2": _qwen2,
    "qwen3": _qwen3,
    "gemma": _gemma,
    "gemma2": _gemma2,
    "gemma3": _gemma3,
    "qwen3_5": _qwen3_5,
    "qwen3_moe": _qwen3_moe,
}


def save_architecture(path, name: str, tok, seed: int = 0, untie: bool = False) -> str:
    torch.manual_seed(seed)
    model = ARCHITECTURES[name](len(tok))
    model.config.use_cache = False
    if untie:  # an output head of its own (pipeline stages hold the embeddings and the head apart)
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def summary(result, output_dir) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(Path(output_dir) / "final")
    return {
        "loss": result.metrics["train/loss"],
        "eval": result.metrics.get("eval/loss"),
        "grad_norm": result.metrics.get("train/grad_norm"),
        "steps": result.global_step,
        "weights": {k: v.flatten().tolist() for k, v in model.state_dict().items()},
    }


def _worker(rank, world, config, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from palingenesis.train import _train_on

        result = _train_on(config, {"tracker": None}, None, rank, world, torch.device("cpu"))
        if rank == 0:
            Path(out).write_text(json.dumps(summary(result, config.train.output_dir)))
    finally:
        dist.destroy_process_group()


def run(config, world: int) -> dict:
    """`pgs train`'s loop on `world` CPU processes; rank 0's metrics and the exported (final) weights."""
    if world == 1:
        return summary(H.run_cpu(config), config.train.output_dir)
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(world, config, free_port(), out), nprocs=world, join=True)
        return json.loads(Path(out).read_text())


def assert_same(got: dict, ref: dict, rtol: float = 1e-4, atol: float = 1e-5):
    import pytest

    assert got["steps"] == ref["steps"]
    assert got["loss"] == pytest.approx(ref["loss"], rel=rtol)
    if ref["eval"] is not None:
        assert got["eval"] == pytest.approx(ref["eval"], rel=rtol)
    if ref["grad_norm"] is not None:
        assert got["grad_norm"] == pytest.approx(ref["grad_norm"], rel=10 * rtol)
    assert got["weights"].keys() == ref["weights"].keys()
    for name, w in ref["weights"].items():
        torch.testing.assert_close(torch.tensor(got["weights"][name]), torch.tensor(w), rtol=rtol, atol=atol, msg=name)
