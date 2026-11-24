"""parallel.cpu_offload on one process: build_mesh gives a mesh of one rank, init_single_process_group a one-rank
group, and FSDP2 with CPU offload trains to the same weights as the plain module (AdamW and adamw_bf16), with the
optimizer stepping the offloaded shards. Config validation of the combinations it does not support. CPU (gloo), in a
spawned process so that the test session keeps no process group."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from palingenesis.config import Config, ConfigError, ParallelConfig
from palingenesis.distributed import build_mesh

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _tiny(dtype):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).to(dtype)
    model.config.use_cache = False
    return model


def _steps(model, optimizer, n=3):
    g = torch.Generator().manual_seed(1)
    for _ in range(n):
        ids = torch.randint(0, 96, (2, 16), generator=g)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _weights(model) -> dict[str, torch.Tensor]:
    return {
        n: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().float().clone()
        for n, p in model.named_parameters()
    }


def _worker(rank, optimizer_name, dtype, out):
    os.environ.pop("MASTER_PORT", None)
    from palingenesis.distributed import apply_fsdp, init_single_process_group
    from palingenesis.optim import build_optimizer

    init_single_process_group()
    try:
        config = ParallelConfig(cpu_offload=True)
        mesh = build_mesh(1, config)
        assert mesh is not None and mesh.size() == 1
        plain = _tiny(dtype)
        _steps(plain, build_optimizer(plain, 1e-3, 0.1, optimizer_name=optimizer_name))
        sharded = apply_fsdp(_tiny(dtype), mesh, config, bf16=False)
        assert all(p.device.type == "cpu" for p in sharded.parameters())
        _steps(sharded, build_optimizer(sharded, 1e-3, 0.1, optimizer_name=optimizer_name))
        a, b = _weights(plain), _weights(sharded)
        diff = max(float((a[n] - b[n]).abs().max()) for n in a)
        torch.save({"diff": diff, "moved": max(float((a[n] - _weights(_tiny(dtype))[n]).abs().max()) for n in a)}, out)
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_name, dtype", [("adamw", torch.float32), ("adamw_bf16", torch.bfloat16)])
def test_offloaded_fsdp_on_one_rank_matches_plain_training(optimizer_name, dtype):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.pt")
        mp.spawn(_worker, args=(optimizer_name, dtype, out), nprocs=1, join=True)
        result = torch.load(out)
    assert result["moved"] > 1e-4  # the steps did train
    # one rank: the same math and, for adamw_bf16, the same noise keys (names, offsets in a one-rank shard)
    assert result["diff"] <= 1e-6, result


def test_build_mesh_without_offload_is_none_on_one_process():
    assert build_mesh(1, ParallelConfig()) is None
    assert build_mesh(1, ParallelConfig(fsdp=False)) is None


def test_validation():
    cfg = Config()
    cfg.parallel.cpu_offload = True
    assert cfg.validate() == []
    cfg.parallel.fsdp = False  # one GPU: cpu_offload implies FSDP2 (shipped configs set fsdp: false with it)
    assert cfg.validate() == []
    for section, key, value, match in [
        ("memory", "seco", True, "memory.seco"),
        ("memory", "gradient_release", True, "gradient_release"),
    ]:
        cfg = Config()
        cfg.parallel.cpu_offload = True
        setattr(getattr(cfg, section), key, value)
        if key == "gradient_release":
            cfg.train.gradient_accumulation_steps = 1
        with pytest.raises(ConfigError, match=match):
            cfg.validate()
