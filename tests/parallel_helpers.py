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


def same_rank_stages_run_on_gloo() -> bool:
    """Whether a rank holding two neighbouring pipeline stages (zbv, dualpipev; any schedule on one rank) can run
    on gloo: torch 2.11's schedules warm the connection up with a send from the rank to itself, which NCCL
    accepts and gloo crashes on; later versions pass same-rank stages' data in memory."""
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti

    return hasattr(PipelineScheduleMulti, "_warmup_p2p")


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


# ── gradients, parameter by parameter (float64) ────────────────────────────────


def plain_tensor_conv(x, weight, bias=None, activation=None, seq_idx=None):
    """A stand-in for causal-conv1d's `causal_conv1d_fn` (a compiled custom operator, Linux + CUDA only): the
    same result as the layer's own convolution, and, like the operator, no distributed tensors accepted."""
    from torch.distributed.tensor import DTensor

    for name, value in (("x", x), ("weight", weight), ("bias", bias)):
        if isinstance(value, DTensor):
            raise RuntimeError(f"causal_conv1d_fn got a DTensor {name}: the kernel takes plain tensors")
    out = torch.nn.functional.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    out = out[..., : x.shape[-1]]
    return torch.nn.functional.silu(out) if activation in ("silu", "swish") else out


def _loss(model, ids, labels, denominator, tp=None):
    from palingenesis.logits import final_hidden_states, output_head
    from palingenesis.loss import chunked_cross_entropy_loss

    hidden = final_hidden_states(model, tp.pad(ids) if tp is not None else ids)
    if tp is not None:
        labels = tp.shard(labels, -100)
    return chunked_cross_entropy_loss(hidden, labels, output_head(model), num_chunks=2, global_valid_tokens=denominator)


def _gradient_worker(rank, world, port, arch, fsdp, conv_kernel, double, out):
    """One forward / backward of the tensor-parallel model in float64 on `world` ranks, as the trainer builds it
    (`fsdp`: FSDP2 over the data dimension on top), against the plain model in the same process. Every rank
    writes, per parameter, the relative error of the gradient the optimizer would use on it."""
    import copy
    from types import SimpleNamespace

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from palingenesis.distributed import apply_fsdp
        from palingenesis.loss import shift_labels
        from palingenesis.tensor_parallel import apply_tensor_parallel

        torch.manual_seed(0)
        ref = ARCHITECTURES[arch](64)
        ref = ref.double() if double else ref
        ref.config.use_cache = False
        if conv_kernel:
            for m in ref.modules():
                if "causal_conv1d_fn" in vars(m):
                    m.causal_conv1d_fn = plain_tensor_conv
        model = copy.deepcopy(ref)
        mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("dp", "tp"))
        tp = apply_tensor_parallel(model, mesh["tp"])
        if fsdp:
            apply_fsdp(model, mesh, SimpleNamespace(reshard_after_forward=True, cpu_offload=False), bf16=False)
        ids = torch.randint(0, 64, (2, 23), generator=torch.Generator().manual_seed(1))  # 23: padded to 24
        labels = shift_labels(ids.clone())
        n = int((labels != -100).sum())
        _loss(ref, ids, labels, n).backward()
        _loss(model, ids, labels, n, tp).backward()
        reference = {k: p.grad for k, p in ref.named_parameters()}
        errors, partial, missing = {}, [], []
        for name, p in model.named_parameters():
            grad, want = p.grad, reference[name]
            if grad is None:
                missing.append(name)
                continue
            if not isinstance(grad, DTensor):  # whole on every rank, its own tokens: summed by FSDP in the trainer
                full = grad.clone()
                dist.all_reduce(full)
            else:
                # Without a partial placement, full_tensor() is built from the values the ranks hold: for a
                # replicated parameter (the norms) this rank's own copy, the one its optimizer steps with. A
                # partial one is each rank's share, still to be summed over the group (FSDP2 does: see `partial`).
                if any(pl.is_partial() for pl in grad.placements):
                    partial.append(name)
                full = grad.full_tensor()
            if name in tp.sections:
                full = tp.sections[name].canonical(full, world)
            errors[name] = float((full - want).norm() / want.norm().clamp_min(1e-300))
        Path(f"{out}.{rank}").write_text(json.dumps({"errors": errors, "partial": partial, "missing": missing}))
    finally:
        dist.destroy_process_group()


def gradient_errors(
    arch: str, world: int = 2, fsdp: bool = True, conv_kernel: bool = False, double: bool = True
) -> list[dict]:
    """Per rank: {"errors": parameter -> relative gradient error against one process, "partial": parameters whose
    gradient is still an unreduced partial sum, "missing": parameters without a gradient}."""
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/gradients.json"
        args = (world, free_port(), arch, fsdp, conv_kernel, double, out)
        mp.spawn(_gradient_worker, args=args, nprocs=world, join=True)
        return [json.loads(Path(f"{out}.{r}").read_text()) for r in range(world)]


def _pipeline_gradient_worker(rank, port, arch, schedule, tp_on, microbatches, out):
    """One schedule step of a two-stage pipeline on ONE rank (zbv / dualpipev keep both stages of the V on it),
    FSDP2 inside each stage, in float64, against the plain model: the layout scripts/gpu_checks/tp_pp.sh runs on
    one GPU."""
    import copy
    from types import SimpleNamespace

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=1)
    try:
        from palingenesis.distributed import apply_fsdp
        from palingenesis.loss import shift_labels
        from palingenesis.pipeline_parallel import PipelineParallel
        from palingenesis.tensor_parallel import apply_tensor_parallel

        torch.manual_seed(0)
        ref = ARCHITECTURES[arch](64).double()
        ref.config.use_cache = False
        ref.config.tie_word_embeddings = False
        ref.lm_head.weight = torch.nn.Parameter(ref.lm_head.weight.detach().clone())
        model = copy.deepcopy(ref)
        names = ("pp", "dp", "tp") if tp_on else ("pp", "dp")
        mesh = init_device_mesh("cpu", (1,) * len(names), mesh_dim_names=names)
        rows, seq = 4, 24
        parallel = SimpleNamespace(
            pp_schedule=schedule,
            pp_stages_per_rank=0,
            pp_microbatches=microbatches,
            pp_first_stage_less_layers=1,
            pp_last_stage_less_layers=1,
            reshard_after_forward=True,
            cpu_offload=False,
        )
        pipe = PipelineParallel(model, mesh, parallel, seq_len=seq, flatten_packed=False)
        if tp_on:
            pipe.tp = apply_tensor_parallel(pipe.model, mesh["tp"], pipeline=True)
        pipe.loss_kind, pipe.num_chunks = "chunked", (lambda n: 2)
        pipe.build(torch.device("cpu"), lambda m: apply_fsdp(m, mesh, parallel, bf16=False, reshard_layers=False))
        pipe.train(True)
        ids = torch.randint(0, 64, (rows, seq), generator=torch.Generator().manual_seed(1))
        labels = shift_labels(ids.clone())
        labels[:, :5] = -100
        n = int((labels != -100).sum())
        loss_ref = _loss(ref, ids, labels, n)
        loss_ref.backward()
        loss = pipe.step(ids, labels, None, None, n)
        reference = {k: p.grad for k, p in ref.named_parameters()}
        errors, missing = {}, sorted(set(reference) - {k for k, _ in pipe.model.named_parameters()})
        for name, p in pipe.model.named_parameters():
            if p.grad is None:
                missing.append(name)
                continue
            full = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            want = reference[name]
            errors[name] = float((full - want).norm() / want.norm().clamp_min(1e-300))
        result = {"loss": loss, "loss_ref": float(loss_ref), "errors": errors, "missing": missing}
        Path(out).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def pipeline_gradient_errors(arch: str, schedule: str, tp: bool = False, microbatches: int = 2) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/gradients.json"
        args = (free_port(), arch, schedule, tp, microbatches, out)
        mp.spawn(_pipeline_gradient_worker, args=args, nprocs=1, join=True)
        return json.loads(Path(out).read_text())
