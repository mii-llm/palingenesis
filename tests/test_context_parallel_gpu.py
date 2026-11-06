"""Context parallelism on GPUs (NCCL, bf16, flash kernels): loss and gradients of a context-parallel
group equal one GPU's on the whole sequence, for full-causal and sliding-window attention.
Needs at least 2 CUDA devices; run with `pytest tests/test_context_parallel_gpu.py`."""

import json
import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs at least 2 CUDA devices"
)

IGNORE = -100
VOCAB = 512


def _model(kind: str):
    torch.manual_seed(0)
    common = dict(
        vocab_size=VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=8192,
        attn_implementation="sdpa",
    )
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(LlamaConfig(**common))
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        model = Qwen3ForCausalLM(
            Qwen3Config(
                **common,
                use_sliding_window=True,
                sliding_window=300,
                layer_types=["sliding_attention", "full_attention"],
            )
        )
    model.config.use_cache = False
    return model


def _batch(seq_len: int):
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (2, seq_len))
    labels = ids.clone()
    labels[:, :17] = IGNORE
    labels[1, seq_len - 100 :] = IGNORE
    return ids, labels


def _loss_sum(model, ids, labels, position_ids=None):
    from palingenesis.loss import chunked_cross_entropy_loss

    kwargs = {} if position_ids is None else {"position_ids": position_ids}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=ids, **kwargs).last_hidden_state
        return chunked_cross_entropy_loss(hidden, labels, model.lm_head, num_chunks=4, global_valid_tokens=1.0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, kind, seq_len, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from palingenesis import context_parallel as C
        from palingenesis.loss import shift_labels

        model = _model(kind).cuda()
        C.route_attention(model, list(model.model.layers))
        cp = C.ContextParallel(dist.group.WORLD, world, rank)
        ids, labels = (t.cuda() for t in _batch(seq_len))
        labels = shift_labels(labels)
        cp.activate(seq_len, ids.device)
        try:
            loss = _loss_sum(model, cp.shard(ids), cp.shard(labels, IGNORE), cp.position_ids(seq_len, 2, ids.device))
            loss.backward()
        finally:
            cp.deactivate()
        total = loss.detach().float().clone()
        dist.all_reduce(total)
        grads = {}
        for name, p in model.named_parameters():
            dist.all_reduce(p.grad)
            grads[name] = p.grad.float().cpu().flatten().tolist()
        if rank == 0:
            Path(out).write_text(json.dumps({"loss": total.item(), "grads": grads}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["llama", "qwen3_sliding"])
def test_context_parallel_matches_one_gpu(kind):
    from palingenesis.loss import shift_labels

    seq_len = 4000  # not a multiple of 2 x world: padded
    model = _model(kind).cuda()
    ids, labels = (t.cuda() for t in _batch(seq_len))
    ref = _loss_sum(model, ids, shift_labels(labels))
    ref.backward()
    ref_grads = {n: p.grad.float().cpu().flatten() for n, p in model.named_parameters()}

    world = min(torch.cuda.device_count(), 4)
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(world, kind, seq_len, _free_port(), out), nprocs=world, join=True)
        got = json.loads(Path(out).read_text())

    # bf16 kernels with a different summation order: compare at bf16 precision
    assert got["loss"] == pytest.approx(ref.item(), rel=2e-3)
    for name, g in ref_grads.items():
        cos = torch.nn.functional.cosine_similarity(torch.tensor(got["grads"][name]), g, dim=0)
        assert cos > 0.999, f"{name}: cosine {cos:.5f}"
        torch.testing.assert_close(torch.tensor(got["grads"][name]).norm(), g.norm(), rtol=1e-2, atol=1e-6)
