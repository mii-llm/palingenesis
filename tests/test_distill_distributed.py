"""Off-policy distillation under FSDP2: two CPU processes (gloo) each take half a batch, with the
student sharded and the online teacher sharded too (placement "shard"); the ternary QAT plugin's
grid regulariser and clamp act on the sharded latents. Losses and gradients must equal a single
process on the whole batch."""

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
sys.path.insert(0, str(Path(__file__).parent))

IGNORE = -100
VOCAB = 96


def _models(group_size):
    import distill_helpers as H

    from palingenesis.hooks import PluginContext
    from palingenesis.qat import TernaryQAT

    student = H.tiny_llama(VOCAB, hidden=32, seed=0, tie=True)
    plugin = TernaryQAT(group_size=group_size, grid_weight=0.3, clamp=1.1, quantize_embedding=True, quantize_head=True)
    student = plugin.transform_model(student, PluginContext(config=None, total_steps=10))
    teacher = H.tiny_llama(VOCAB, hidden=48, seed=1)
    return student, teacher, plugin


def _batch():
    torch.manual_seed(5)
    ids = torch.randint(0, VOCAB, (4, 20))
    labels = ids.clone()
    labels[:, :6] = IGNORE
    labels[2, :14] = IGNORE  # unequal trained-token counts across ranks
    return ids, labels


class _Tok:
    """The vocabulary interface the distiller needs (identical tokenizers for student and teacher)."""

    def __len__(self):
        return VOCAB


def _distiller(student, teacher, mesh=None):
    """As the trainer does it: teachers placed, feature hooks and projections attached to the student
    before the student is wrapped by FSDP."""
    from palingenesis.distill.config import DistillConfig, TeacherSpec
    from palingenesis.distill.distiller import Distiller
    from palingenesis.distill.teachers import OnlineTeacher
    from palingenesis.hooks import FeatureSpec

    tok = _Tok()
    spec = TeacherSpec("t", model="in-memory", dtype="float32", placement="shard" if mesh is not None else "replicate")
    online = OnlineTeacher(spec, tok, torch.device("cpu"), "sdpa", model=teacher, tokenizer=tok, mesh=mesh, bf16=False)
    cfg = DistillConfig(enabled=True, divergence="jsd", beta=0.3, ce_weight=0.2, temperature=2.0)
    distiller = Distiller(cfg, [spec], {"t": online}, {}, features=[FeatureSpec(student=1, teacher=1, weight=0.5)])
    distiller.attach_features(student, torch.device("cpu"))
    return distiller


def _step(distiller, student, plugin, ids, labels, denom):
    from palingenesis.hooks import PluginContext
    from palingenesis.loss import shift_labels
    from palingenesis.train import _get_hidden_states, _get_lm_head

    shifted = shift_labels(labels)
    targets = distiller.targets({}, ids, shifted, {"attention_mask": None}, compute_dtype=torch.float32)
    with distiller.capturing():
        hidden = _get_hidden_states(student, ids)
    loss, _ = distiller.loss(hidden, targets, _get_lm_head(student), denom, distiller.student_feature_rows(targets))
    loss.backward()
    distiller.release()
    plugin.on_before_optimizer_step(student, None, PluginContext(config=None, total_steps=10, step=5))
    return loss.detach()


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _snapshot(model) -> dict:
    from palingenesis.checkpoint import hf_state_dict

    out = {}
    for name, p in model.named_parameters():
        out[f"grad:{name}"] = _full(p.grad).flatten().tolist() if p.grad is not None else None
    return hf_state_dict(out)


def _weights(model) -> dict:
    from palingenesis.checkpoint import hf_state_dict

    return hf_state_dict({n: _full(p.detach()).flatten().tolist() for n, p in model.named_parameters()})


def _worker(rank, world, group_size, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29540 + group_size))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from palingenesis.config import ParallelConfig
        from palingenesis.distributed import apply_fsdp
        from palingenesis.hooks import PluginContext
        from palingenesis.loss import shift_labels

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        student, teacher, plugin = _models(group_size)
        distiller = _distiller(student, teacher, mesh=mesh)
        student = apply_fsdp(student, mesh, ParallelConfig(), bf16=False)
        ids, labels = _batch()
        denom = float((shift_labels(labels) != IGNORE).sum())
        rows = slice(rank * 2, rank * 2 + 2)
        loss = _step(distiller, student, plugin, ids[rows], labels[rows], denom)
        dist.all_reduce(loss)
        grads = _snapshot(student)
        plugin.on_after_optimizer_step(student, None, PluginContext(config=None))
        weights = _weights(student)
        if rank == 0:
            Path(out).write_text(json.dumps({"loss": loss.item(), "grads": grads, "weights": weights}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("group_size", [0, 16])
def test_sharded_student_and_teacher_match_one_process(group_size):
    from palingenesis.hooks import PluginContext
    from palingenesis.loss import shift_labels

    student, teacher, plugin = _models(group_size)
    ids, labels = _batch()
    denom = float((shift_labels(labels) != IGNORE).sum())
    ref_loss = _step(_distiller(student, teacher), student, plugin, ids, labels, denom)
    ref_grads = _snapshot(student)
    plugin.on_after_optimizer_step(student, None, PluginContext(config=None))
    ref_weights = _weights(student)

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/result.json"
        mp.spawn(_worker, args=(2, group_size, out), nprocs=2, join=True)
        got = json.loads(Path(out).read_text())

    assert got["loss"] == pytest.approx(ref_loss.item(), rel=1e-5)
    assert got["grads"].keys() == ref_grads.keys()
    for name, ref in ref_grads.items():
        if ref is None:
            assert got["grads"][name] is None, name
            continue
        torch.testing.assert_close(torch.tensor(got["grads"][name]), torch.tensor(ref), rtol=1e-4, atol=1e-6, msg=name)
    for name, ref in ref_weights.items():
        torch.testing.assert_close(
            torch.tensor(got["weights"][name]), torch.tensor(ref), rtol=1e-5, atol=1e-6, msg=name
        )
