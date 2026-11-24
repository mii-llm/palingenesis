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


@pytest.mark.parametrize("arch", ["mistral", "qwen2", "gemma3", "qwen3_moe"])
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
    "variant", ["packing", "checkpointing_full", "checkpointing_selective", "plain_loss", "observations"]
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
