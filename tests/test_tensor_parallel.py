"""Tensor parallelism through the trainer, end to end on the CPU: `pgs train`'s loop over gloo processes
(palingenesis.tensor_parallel, FSDP2 over the other dimensions) trains the same weights, logs the same loss,
gradient norm and evaluation loss as one process, for every architecture with a plan; and composes with data
parallelism (dp x tp mesh), packing, activation checkpointing, chunked / plain losses, checkpoint save and
resume."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402
import parallel_helpers as P  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tp")
    tok = H.tokenizer()
    models = {name: P.save_architecture(tmp / name, name, tok) for name in P.ARCHITECTURES}
    data = H.write_rows(tmp / "rows.jsonl")
    return tmp, models, data


def _config(setup, name: str, arch: str = "llama", tp: int = 1, fsdp: bool = True, steps: int = 3):
    tmp, models, data = setup
    c = H.cpu_config(tmp / name, models[arch], data, steps=steps)
    c.train.per_device_batch_size = 2
    c.train.gradient_accumulation_steps = 2
    c.train.max_grad_norm = 1.0
    c.data.eval_dataset = data
    c.data.eval_every = steps
    c.parallel.fsdp = fsdp
    c.parallel.tp_degree = tp
    c.moe.balance = False
    return c


@pytest.mark.parametrize("arch", ["llama", "qwen3", "qwen3_5"])
def test_two_tensor_parallel_ranks_train_like_one_process(setup, arch):
    ref = P.run(_config(setup, f"one_{arch}", arch, fsdp=False), world=1)
    got = P.run(_config(setup, f"tp2_{arch}", arch, tp=2), world=2)
    P.assert_same(got, ref)


@pytest.mark.parametrize("arch", ["mistral", "qwen2", "gemma", "gemma2", "gemma3", "qwen3_moe"])
def test_other_architectures_train_like_one_process(setup, arch):
    ref = P.run(_config(setup, f"one_{arch}", arch, fsdp=False), world=1)
    got = P.run(_config(setup, f"tp2_{arch}", arch, tp=2), world=2)
    P.assert_same(got, ref)


@pytest.mark.parametrize("arch", ["llama", "qwen3_5"])
def test_four_tensor_parallel_ranks(setup, arch):
    ref = P.run(_config(setup, f"one4_{arch}", arch, fsdp=False), world=1)
    got = P.run(_config(setup, f"tp4_{arch}", arch, tp=4), world=4)
    P.assert_same(got, ref)


def test_data_by_tensor_mesh_trains_like_data_parallel(setup):
    # 2 x 2: FSDP over the data ranks, tensor parallel inside each pair; the same rows per data rank as FSDP alone
    ref = P.run(_config(setup, "dp2", "qwen3_5"), world=2)
    got = P.run(_config(setup, "dp2_tp2", "qwen3_5", tp=2), world=4)
    P.assert_same(got, ref, rtol=1e-3, atol=1e-4)  # Adam amplifies reduction-order noise on tiny gradients


@pytest.mark.parametrize(
    "variant", ["packing", "checkpointing_full", "checkpointing_selective", "plain_loss", "observations", "dft"]
)
def test_training_options_compose(setup, variant):
    def configure(c):
        if variant == "packing":
            c.data.packing = True
            c.data.max_seq_length = 125  # rows padded to a multiple of tp with a padding document
        elif variant.startswith("checkpointing"):
            c.train.gradient_checkpointing = variant.split("_")[1]
        elif variant == "plain_loss":
            c.memory.chunked_loss = False
        elif variant == "observations":
            c.data.include_observations = True
            c.data.observation_weight = 0.5
        elif variant == "dft":
            c.plugins.dft = True
        return c

    ref = P.run(configure(_config(setup, f"one_{variant}", fsdp=False)), world=1)
    got = P.run(configure(_config(setup, f"tp2_{variant}", tp=2)), world=2)
    P.assert_same(got, ref)


def test_moe_router_balancing_matches_one_process(setup):
    # every tensor-parallel rank routes the whole sequence: the balancer counts each token once
    def configure(c):
        c.moe.balance = True
        c.moe.balance_loss_coef = 0.05
        c.moe.z_loss_coef = 0.01
        return c

    ref = P.run(configure(_config(setup, "one_moe_bal", "qwen3_moe", fsdp=False)), world=1)
    got = P.run(configure(_config(setup, "tp2_moe_bal", "qwen3_moe", tp=2)), world=2)
    P.assert_same(got, ref)


def test_resume_from_a_sharded_checkpoint_continues_the_run(setup):
    # DCP checkpoints of DTensor shards: a run resumed at step 2 ends where the uninterrupted one does
    whole = _config(setup, "tp2_whole", "qwen3_5", tp=2, steps=4)
    whole.train.save_steps = 2
    ref = P.run(whole, world=2)
    resumed = _config(setup, "tp2_resumed", "qwen3_5", tp=2, steps=4)
    resumed.train.resume_from = str(Path(whole.train.output_dir) / "step-2")
    got = P.run(resumed, world=2)
    P.assert_same(got, ref, rtol=1e-3, atol=1e-4)
    meta = (Path(whole.train.output_dir) / "step-2" / "training_meta.json").read_text()
    assert '"tp_degree": 2' in meta and "in_proj_qkv" in meta

    # the linear-attention q / k / v are stored rank by rank: another tp_degree must not load them
    other = _config(setup, "dp2_from_tp2", "qwen3_5", tp=1, steps=4)
    other.train.resume_from = resumed.train.resume_from
    with pytest.raises(Exception, match="saved with parallel.tp_degree=2"):
        P.run(other, world=2)


def test_checkpoint_without_rank_by_rank_parameters_resumes_at_another_degree(setup):
    # Llama has no parameter stored rank by rank: DCP reshards a tp=2 checkpoint into plain FSDP (dp=2) and the
    # run continues from its step (the load itself is what is checked: other rows per step, so no reference).
    whole = _config(setup, "tp2_llama_whole", "llama", tp=2, steps=4)
    whole.train.save_steps = 2
    ref = P.run(whole, world=2)
    meta = (Path(whole.train.output_dir) / "step-2" / "training_meta.json").read_text()
    assert '"rank_by_rank": []' in meta
    resumed = _config(setup, "dp2_from_tp2_llama", "llama", tp=1, steps=4)
    resumed.train.resume_from = str(Path(whole.train.output_dir) / "step-2")
    got = P.run(resumed, world=2)
    assert got["steps"] == ref["steps"] == 4 and got["loss"] > 0


@pytest.mark.parametrize("arch", ["qwen3", "gemma3", "qwen3_5", "qwen3_moe"])
def test_every_gradient_matches_one_process_in_float64(arch):
    # One backward on two ranks as the trainer builds the model (tensor parallel, FSDP2 on top), parameter by
    # parameter: a replicated weight used on each rank's tokens or heads (layer norms, q_norm / k_norm, the gated
    # norm) has a different partial gradient on every rank, and each rank's optimizer must step with their sum.
    # FSDP2 reduces in float32: 1e-6 (the grouped expert matmul has no float64 kernel: float32 and 1e-3 there).
    # A missing reduction is an error of order 1.
    double = arch != "qwen3_moe"
    bound = 1e-6 if double else 1e-3
    ranks = P.gradient_errors(arch, world=2, fsdp=True, double=double)
    for rank, result in enumerate(ranks):
        assert not result["missing"], (rank, result["missing"])
        assert not result["partial"], (rank, result["partial"])
        worst = max(result["errors"], key=result["errors"].get)
        assert result["errors"][worst] < bound, (rank, worst, result["errors"][worst])
    norms = [n for n in ranks[0]["errors"] if "norm" in n]
    assert any("q_norm" in n or "linear_attn.norm" in n or "layernorm" in n for n in norms)


def test_replicated_norm_gradients_are_exact_without_fsdp():
    # The plan alone (the layout scripts/gpu_checks/tp_pp.sh checks): DTensor leaves the norms' gradients as
    # partial sums, and gathering them reduces over the group. Exact to float64 rounding.
    for result in P.gradient_errors("qwen3", world=2, fsdp=False):
        assert not result["missing"]
        assert any("norm" in n for n in result["partial"])
        worst = max(result["errors"], key=result["errors"].get)
        assert result["errors"][worst] < 1e-12, (worst, result["errors"][worst])


def test_gated_delta_net_convolution_kernel_gets_plain_tensors():
    # With causal-conv1d installed, transformers' forward hands conv1d.weight / bias to the kernel itself (a
    # custom operator: "got mixed torch.Tensor and DTensor") instead of calling the convolution module.
    for rank, result in enumerate(P.gradient_errors("qwen3_5", world=2, fsdp=True, conv_kernel=True)):
        assert not result["missing"] and not result["partial"]
        assert any("conv1d.weight" in n for n in result["errors"])
        worst = max(result["errors"], key=result["errors"].get)
        assert result["errors"][worst] < 1e-6, (rank, worst, result["errors"][worst])


def test_local_parameters_come_back_when_the_forward_raises():
    from torch import nn

    from palingenesis.tensor_parallel import _local_params

    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(4, 4, 2, groups=4)

        def forward(self, x):
            assert not hasattr(self.conv1d.weight, "to_local")  # a plain tensor inside the forward
            raise ValueError("boom")

    import torch.distributed as dist

    from palingenesis.distributed import init_single_process_group

    created = not dist.is_initialized()
    if created:
        init_single_process_group()
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        owner = Owner()
        _local_params(
            owner, init_device_mesh("cpu", (1,), mesh_dim_names=("tp",)), {"weight": (Shard(0), False)}, "conv1d"
        )
        with pytest.raises(ValueError, match="boom"):
            owner(torch.zeros(1, 4, 3))
        assert isinstance(owner.conv1d.weight, DTensor) and isinstance(owner.conv1d.weight, nn.Parameter)
    finally:
        if created:
            dist.destroy_process_group()


def test_clip_grad_norm_scales_dtensor_gradients_in_place_and_refuses_partial_ones():
    """Gradients are clipped where they are stored. A Partial gradient (a sum over ranks not reduced yet) has no
    stored value to scale: redistributing it gives a copy, and scaling that would leave p.grad unclipped."""
    from types import SimpleNamespace

    import torch.distributed as dist

    from palingenesis.distributed import clip_grad_norm_, init_single_process_group

    created = not dist.is_initialized()
    if created:
        init_single_process_group()
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        sharded = SimpleNamespace(grad=DTensor.from_local(torch.full((2, 2), 3.0), mesh, [Shard(0)], run_check=False))
        whole = SimpleNamespace(grad=DTensor.from_local(torch.full((4,), 4.0), mesh, [Replicate()], run_check=False))
        plain = SimpleNamespace(grad=torch.zeros(3))
        norm = clip_grad_norm_([sharded, whole, plain, SimpleNamespace(grad=None)], max_norm=1.0)
        assert float(norm) == pytest.approx(10.0)  # sqrt(4 x 9 + 4 x 16)
        torch.testing.assert_close(sharded.grad.to_local(), torch.full((2, 2), 0.3), rtol=1e-5, atol=0)
        torch.testing.assert_close(whole.grad.to_local(), torch.full((4,), 0.4), rtol=1e-5, atol=0)
        partial = SimpleNamespace(grad=DTensor.from_local(torch.ones(2, 2), mesh, [Partial()], run_check=False))
        with pytest.raises(RuntimeError, match="Partial"):
            clip_grad_norm_([sharded, partial], max_norm=1.0)
        torch.testing.assert_close(partial.grad.to_local(), torch.ones(2, 2))  # nothing was scaled
        torch.testing.assert_close(sharded.grad.to_local(), torch.full((2, 2), 0.3), rtol=1e-5, atol=0)
    finally:
        if created:
            dist.destroy_process_group()


def test_layout_check():
    from palingenesis.checkpoint import check_layout

    check_layout({}, {}, "x")
    check_layout({"tp_degree": 2, "rank_by_rank": []}, {"tp_degree": 4, "rank_by_rank": []}, "x")  # plain reshard
    check_layout({"tp_degree": 2, "rank_by_rank": ["a"]}, {"tp_degree": 2, "rank_by_rank": ["a"]}, "x")
    with pytest.raises(RuntimeError, match="tp_degree=2"):
        check_layout({"tp_degree": 2, "rank_by_rank": ["a"]}, {}, "x")
    with pytest.raises(RuntimeError):
        check_layout({}, {"tp_degree": 2, "rank_by_rank": ["a"]}, "x")


def test_sections_round_trip():
    from palingenesis.tensor_parallel import Sections

    s = Sections(0, (4, 4, 8))
    t = torch.arange(16.0)
    by_rank = s.to_ranks(t, 2)
    assert by_rank.tolist() == [0, 1, 4, 5, 8, 9, 10, 11, 2, 3, 6, 7, 12, 13, 14, 15]
    assert torch.equal(s.canonical(by_rank, 2), t)
    m = Sections(1, (3, 3))
    w = torch.randn(2, 6, 5)
    assert torch.equal(m.canonical(m.to_ranks(w, 3), 3), w)


def test_support_errors_name_what_does_not_split():
    from types import SimpleNamespace

    from palingenesis.tensor_parallel import support_errors

    qwen3 = SimpleNamespace(model_type="qwen3", num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072)
    assert support_errors(qwen3, 8) == []
    assert "num_key_value_heads (8)" in support_errors(qwen3, 16)[0]
    assert "has no tensor-parallel plan" in support_errors(SimpleNamespace(model_type="phi3"), 2)[0]
    hybrid = SimpleNamespace(
        model_type="qwen3_5_text",
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=3584,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=16,
    )
    assert support_errors(hybrid, 2) == []
    assert any("num_key_value_heads" in e for e in support_errors(hybrid, 4))
    shared = SimpleNamespace(
        model_type="qwen3_moe", num_attention_heads=8, num_experts=8, shared_expert_intermediate_size=64
    )
    assert any("shared expert" in e for e in support_errors(shared, 2))


def test_a_layer_outside_the_plan_is_refused():
    # a plugin that wraps a projection (LoRA, QAT) leaves the plan nothing it knows how to split
    from torch import nn

    from palingenesis.tensor_parallel import TensorParallel, TensorParallelError, _parallelize_layer

    model = P.ARCHITECTURES["llama"](64)
    layer = model.model.layers[0]
    layer.self_attn.q_proj = nn.Sequential(layer.self_attn.q_proj)
    tp = TensorParallel(mesh=None, size=2, rank=0)
    with pytest.raises(TensorParallelError, match="q_proj"):
        _parallelize_layer(layer, "model.layers.0", tp)


def test_per_rank_memory_estimate():
    from palingenesis.agent_tooling.profile_memory import _per_rank
    from palingenesis.config import Config

    model = P.ARCHITECTURES["llama"](64)
    config = Config()
    config.parallel.tp_degree = 2
    kwargs = dict(weights_gb=16.0, activations_gb=8.0, logits_gb=2.0, layers=4)
    one = _per_rank(config, model, 2, **kwargs)
    two = _per_rank(config, model, 4, **kwargs)  # dp=2 on top: weights sharded again, activations unchanged
    assert one["dims"] == "pp=1 x dp=1 x cp=1 x tp=2" and one["activations_gb"] == pytest.approx(4.0)
    assert one["weights_optimizer_grads_gb"] == pytest.approx(8.0)
    assert two["weights_optimizer_grads_gb"] == pytest.approx(4.0) and two["activations_gb"] == one["activations_gb"]
    assert "cannot split" in _per_rank(config, model, 3, **kwargs)["error"]
    assert _per_rank(Config(), model, 1, **kwargs) is None
