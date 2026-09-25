"""The trainer's loss path under FSDP2: apply_fsdp, then the backbone and the output
head called separately (chunked CE never runs the model's own forward).

Two CPU processes (gloo) each take half of a batch; the gradients they reduce must
equal single-process gradients on the whole batch, for tied and untied embeddings.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

IGNORE_INDEX = -100


def _tiny(tie: bool):
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
            tie_word_embeddings=tie,
        )
    )
    model.config.use_cache = False
    return model


def _batch():
    torch.manual_seed(1)
    ids = torch.randint(0, 96, (4, 24))
    labels = ids.clone()
    labels[:, :5] = IGNORE_INDEX
    labels[1, :15] = IGNORE_INDEX  # unequal valid-token counts across the two ranks
    return ids, labels


def _loss(model, ids, labels, denom):
    from palingenesis.logits import output_head
    from palingenesis.loss import chunked_cross_entropy_loss, shift_labels
    from palingenesis.train import _get_hidden_states

    hidden = _get_hidden_states(model, ids, torch.ones_like(ids))
    return chunked_cross_entropy_loss(
        hidden, shift_labels(labels), output_head(model), num_chunks=3, global_valid_tokens=denom
    )


def _full_grads(model) -> dict[str, list]:
    from palingenesis.checkpoint import hf_state_dict

    grads = {}
    for name, p in model.named_parameters():
        g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
        grads[name] = g.flatten().tolist()
    return hf_state_dict(grads)


def _worker(rank, world_size, tie, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from palingenesis.config import ParallelConfig
        from palingenesis.distributed import apply_fsdp

        model = apply_fsdp(
            _tiny(tie), init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",)), ParallelConfig(), bf16=False
        )
        ids, labels = _batch()
        from palingenesis.loss import shift_labels

        denom = int((shift_labels(labels) != IGNORE_INDEX).sum())  # global valid tokens, as the trainer
        rows = slice(rank * 2, rank * 2 + 2)
        loss = _loss(model, ids[rows], labels[rows], denom)
        loss.backward()
        total = loss.detach().clone()
        dist.all_reduce(total)
        grads = _full_grads(model)
        if rank == 0:
            Path(out).write_text(json.dumps({"loss": total.item(), "grads": grads}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tie", [True, False])
def test_fsdp_gradients_match_single_process(tie):
    from palingenesis.loss import shift_labels

    model = _tiny(tie)
    ids, labels = _batch()
    ref_loss = _loss(model, ids, labels, int((shift_labels(labels) != IGNORE_INDEX).sum()))
    ref_loss.backward()
    ref = _full_grads(model)

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(2, tie, out), nprocs=2, join=True)
        got = json.loads(Path(out).read_text())

    assert got["loss"] == pytest.approx(ref_loss.item(), rel=1e-5)
    assert got["grads"].keys() == ref.keys()
    for name in ref:
        torch.testing.assert_close(
            torch.tensor(got["grads"][name]), torch.tensor(ref[name]), rtol=1e-4, atol=1e-6, msg=name
        )
