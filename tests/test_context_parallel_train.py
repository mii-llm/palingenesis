"""Context parallelism through the trainer, end to end on the CPU: `pgs train`'s loop in several
gloo processes (FSDP2 over the flattened data x context mesh) trains the same weights and logs
the same loss as without context parallelism: one process for a context-parallel group of 2, and
data-parallel FSDP over 2 processes for a 2 x 2 mesh. Covers the data shard per data-parallel
rank, padding to the layout's multiple, label shift, position ids, loss normalisation, gradient
reduction over both mesh dimensions, sharded evaluation and online-teacher distillation; and a
hybrid student (Qwen3.5-style Gated DeltaNet and full attention), with activation checkpointing
recomputing the split linear attention in the backward."""

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cp")
    tok = H.tokenizer()
    H.save_model(tmp / "student", tok, seed=0)
    H.save_model(tmp / "teacher", tok, seed=1, hidden=48)
    _save_hybrid(tmp / "hybrid", tok)
    data = H.write_rows(tmp / "rows.jsonl")
    return tmp, data


def _save_hybrid(path, tok):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
    )
    _slow_decay(model)
    model.config.use_cache = False
    model.save_pretrained(path)
    tok.save_pretrained(path)


def _slow_decay(model):
    """Decay rates of 0.01-0.1 per token instead of the initialisation's up to 16: the state then carries
    across chunk boundaries (and the decay parameters get gradients well above rounding)."""
    for m in model.modules():
        if hasattr(m, "A_log"):
            with torch.no_grad():
                m.A_log.copy_(torch.empty_like(m.A_log).uniform_(0.01, 0.1).log())


def _config(setup, name: str, cp: bool, fsdp: bool, distill: bool = False, student: str = "student"):
    tmp, data = setup
    c = H.cpu_config(tmp / name, tmp / student, data, steps=3)
    c.train.per_device_batch_size = 2
    c.train.gradient_accumulation_steps = 2
    c.data.eval_dataset = data
    c.data.eval_every = 3
    c.parallel.fsdp = fsdp
    c.parallel.context_parallel = cp
    if distill:
        c.distill.enabled = True
        c.distill.ce_weight = 0.5
        c.distill.teachers = {"t": {"model": str(tmp / "teacher"), "dtype": "float32"}}
    return c


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _summary(result, output_dir) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(Path(output_dir) / "final")
    return {
        "loss": result.metrics["train/loss"],
        "eval": result.metrics.get("eval/loss"),
        "weights": {k: v.flatten().tolist() for k, v in model.state_dict().items()},
    }


def _worker(rank, world, config, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from palingenesis.train import _train_on

        run = {"tracker": None}
        result = _train_on(config, run, None, rank, world, torch.device("cpu"))
        if rank == 0:
            Path(out).write_text(json.dumps(_summary(result, config.train.output_dir)))
    finally:
        dist.destroy_process_group()


def _run(config, world: int) -> dict:
    if world == 1:
        return _summary(H.run_cpu(config), config.train.output_dir)
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(world, config, _free_port(), out), nprocs=world, join=True)
        return json.loads(Path(out).read_text())


def _assert_same(got: dict, ref: dict):
    assert got["loss"] == pytest.approx(ref["loss"], rel=1e-4)
    assert got["eval"] == pytest.approx(ref["eval"], rel=1e-4)
    assert got["weights"].keys() == ref["weights"].keys()
    for name, w in ref["weights"].items():
        torch.testing.assert_close(torch.tensor(got["weights"][name]), torch.tensor(w), rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize("distill", [False, True])
def test_two_context_parallel_ranks_train_like_one_process(setup, distill):
    ref = _run(_config(setup, f"one_{distill}", cp=False, fsdp=False, distill=distill), world=1)
    got = _run(_config(setup, f"cp2_{distill}", cp=True, fsdp=True, distill=distill), world=2)
    _assert_same(got, ref)


def test_data_by_context_mesh_trains_like_data_parallel(setup):
    ref = _run(_config(setup, "dp2", cp=False, fsdp=True), world=2)
    c = _config(setup, "dp2_cp2", cp=True, fsdp=True)
    c.parallel.cp_degree = 2
    got = _run(c, world=4)
    _assert_same(got, ref)


@pytest.mark.parametrize("checkpointing", ["none", "full"])
def test_hybrid_linear_attention_model_trains_like_one_process(setup, checkpointing):
    ref = _run(_config(setup, f"hybrid_one_{checkpointing}", cp=False, fsdp=False, student="hybrid"), world=1)
    c = _config(setup, f"hybrid_cp2_{checkpointing}", cp=True, fsdp=True, student="hybrid")
    c.train.gradient_checkpointing = checkpointing
    _assert_same(_run(c, world=2), ref)
