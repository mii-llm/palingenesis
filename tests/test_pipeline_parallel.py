"""Pipeline parallelism through the trainer, end to end on the CPU: `pgs train`'s loop over gloo processes
(palingenesis.pipeline_parallel: torch.distributed.pipelining stages and schedules, FSDP2 inside each stage)
trains the same weights, logs the same loss, gradient norm and evaluation loss as one process; with every
schedule, with data and tensor parallelism next to it (pp x dp, pp x tp), packing, and checkpoints that resume
and export in the model's own layout."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402
import parallel_helpers as P  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pp")
    tok = H.tokenizer()
    models = {}
    for name in ("llama", "qwen3_moe", "gemma3", "qwen3_5", "qwen3"):
        models[name] = P.save_architecture(tmp / name, name, tok, untie=True)
    data = H.write_rows(tmp / "rows.jsonl")
    return tmp, models, data


def _config(setup, name, arch="llama", pp=1, tp=1, schedule="1f1b", fsdp=True, steps=3, microbatches=0):
    tmp, models, data = setup
    c = H.cpu_config(tmp / name, models[arch], data, steps=steps)
    c.train.per_device_batch_size = 4
    c.train.gradient_accumulation_steps = 2
    c.train.max_grad_norm = 1.0
    c.data.max_seq_length = 64
    c.data.eval_dataset = data
    c.data.eval_every = steps
    c.parallel.fsdp = fsdp
    c.parallel.pp_degree = pp
    c.parallel.tp_degree = tp
    c.parallel.pp_schedule = schedule
    c.parallel.pp_microbatches = microbatches
    c.moe.balance = False
    return c


@pytest.mark.parametrize(
    "schedule", ["gpipe", "1f1b", "interleaved_1f1b", "looped_bfs", "zbv", "interleaved_zero_bubble"]
)
def test_schedules_train_like_one_process(setup, schedule):
    ref = P.run(_config(setup, f"one_{schedule}", fsdp=False), world=1)
    got = P.run(_config(setup, f"pp2_{schedule}", pp=2, schedule=schedule), world=2)
    P.assert_same(got, ref)


@pytest.mark.parametrize("arch", ["qwen3_moe", "gemma3", "qwen3", "qwen3_5"])
def test_architectures(setup, arch):
    ref = P.run(_config(setup, f"one_{arch}", arch, fsdp=False), world=1)
    got = P.run(_config(setup, f"pp2_{arch}", arch, pp=2, microbatches=2), world=2)
    P.assert_same(got, ref)


def test_pipeline_by_data_parallel(setup):
    ref = P.run(_config(setup, "dp2_ref"), world=2)
    got = P.run(_config(setup, "pp2_dp2", pp=2), world=4)
    P.assert_same(got, ref, rtol=1e-3, atol=1e-4)


def test_pipeline_by_tensor_parallel(setup):
    ref = P.run(_config(setup, "one_pptp", fsdp=False), world=1)
    got = P.run(_config(setup, "pp2_tp2", pp=2, tp=2), world=4)
    P.assert_same(got, ref, rtol=1e-3, atol=1e-4)


def test_packing(setup):
    def packed(c):
        c.data.packing = True
        return c

    ref = P.run(packed(_config(setup, "one_packed", fsdp=False)), world=1)
    got = P.run(packed(_config(setup, "pp2_packed", pp=2)), world=2)
    P.assert_same(got, ref)


def test_resume_continues_the_run(setup):
    # DCP checkpoint of every stage's shards: resumed at step 1, the run ends where the uninterrupted one does
    whole = _config(setup, "pp_whole", pp=2)
    whole.train.save_steps = 1
    ref = P.run(whole, world=2)
    resumed = _config(setup, "pp_resumed", pp=2)
    resumed.train.resume_from = str(Path(whole.train.output_dir) / "step-1")
    P.assert_same(P.run(resumed, world=2), ref, rtol=1e-3, atol=1e-4)


def test_checkpoint_resumes_at_another_pipeline_degree(setup):
    # parameters keep the model's own names on every stage: a pp=2 checkpoint loads into plain FSDP (and back)
    first = _config(setup, "pp_first", pp=2)
    first.train.save_steps = 1
    first.train.gradient_accumulation_steps = 1
    P.run(first, world=2)
    on_fsdp = _config(setup, "fsdp_from_pp")
    on_fsdp.train.gradient_accumulation_steps = 1
    on_fsdp.train.resume_from = str(Path(first.train.output_dir) / "step-1")
    on_fsdp.train.save_steps = 2
    got = P.run(on_fsdp, world=2)
    assert got["steps"] == 3 and got["loss"] > 0
    back = _config(setup, "pp_from_fsdp", pp=2)
    back.train.gradient_accumulation_steps = 1
    back.train.resume_from = str(Path(on_fsdp.train.output_dir) / "step-2")
    assert P.run(back, world=2)["steps"] == 3
