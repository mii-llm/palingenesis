"""Muon under torch.distributed (two CPU processes, gloo): the orthogonalization work split across ranks gives the
single-process result, and a step on an FSDP2-sharded model equals the step on the unsharded model."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from palingenesis.muon import MuonAdamW, build_muon, orthogonalize


def _updates():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(64, 256, generator=g) for _ in range(5)]


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128, bias=False), nn.ReLU(), nn.Linear(128, 64, bias=True))


def _step(model):
    torch.manual_seed(1)
    x = torch.randn(8, 64)
    opt = build_muon(model, lr=1e-2, weight_decay=0.1, min_dim=16, compile=False)
    model(x).pow(2).mean().backward()
    opt.step()


def _full_params(model):
    return {n: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().flatten().tolist()
            for n, p in model.named_parameters()}


def _worker(rank, world, mode, out, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        if mode == "split":
            opt = MuonAdamW([{"params": [nn.Parameter(torch.zeros(2, 2))], "use_muon": True}], compile=False)
            res = opt._orthogonalize_batch(_updates(), "gram")
            if rank == 0:
                Path(out).write_text(json.dumps(res.flatten().tolist()))
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            model = _net()
            mesh = init_device_mesh("cpu", (world,))
            for layer in (model[0], model[2]):
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            _step(model)
            params = _full_params(model)
            if rank == 0:
                Path(out).write_text(json.dumps(params))
    finally:
        dist.destroy_process_group()


def test_work_split_across_ranks_matches_one_process():
    ref = orthogonalize(torch.stack(_updates()), "gram")
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/r.json"
        mp.spawn(_worker, args=(2, "split", out, 29541), nprocs=2, join=True)
        got = torch.tensor(json.loads(Path(out).read_text())).view_as(ref)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def test_fsdp2_step_matches_unsharded():
    pytest.importorskip("torch.distributed.fsdp")
    model = _net()
    _step(model)
    ref = _full_params(model)
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/p.json"
        mp.spawn(_worker, args=(2, "fsdp", out, 29542), nprocs=2, join=True)
        got = json.loads(Path(out).read_text())
    assert got.keys() == ref.keys()
    for n in ref:
        torch.testing.assert_close(torch.tensor(got[n]), torch.tensor(ref[n]), rtol=1e-4, atol=1e-6, msg=n)
