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


def _needs_same_rank_stages(schedule):
    # zbv / dualpipev put two neighbouring stages on a rank: on gloo that needs torch's in-memory hand-over
    # (torch 2.11 sends from the rank to itself, which only NCCL accepts: scripts/gpu_checks/tp_pp.sh covers it)
    if schedule in ("zbv", "dualpipev") and not P.same_rank_stages_run_on_gloo():
        pytest.skip("this torch's V-shaped pipeline schedules need NCCL (a rank sends to itself)")


@pytest.mark.parametrize(
    "schedule", ["gpipe", "1f1b", "interleaved_1f1b", "looped_bfs", "zbv", "dualpipev", "interleaved_zero_bubble"]
)
def test_schedules_train_like_one_process(setup, schedule):
    _needs_same_rank_stages(schedule)
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


@pytest.mark.parametrize(
    "arch,schedule,tp",
    [("qwen3", "zbv", False), ("qwen3", "dualpipev", False), ("qwen3_5", "zbv", False), ("qwen3_5", "zbv", True)],
)
def test_every_gradient_of_a_v_pipeline_matches_the_plain_model_in_float64(arch, schedule, tp):
    _needs_same_rank_stages(schedule)
    # Both stages of the V on one rank, FSDP2 inside (the one-GPU layout of scripts/gpu_checks/tp_pp.sh): the loss
    # and every parameter's gradient, summed over the pipeline micro-batches, against one plain backward. In
    # float64 the micro-batches' summation order is invisible, so what is left is the pipeline's arithmetic.
    result = P.pipeline_gradient_errors(arch, schedule, tp=tp)
    assert not result["missing"], result["missing"]
    assert result["loss"] == pytest.approx(result["loss_ref"], rel=1e-9)
    worst = max(result["errors"], key=result["errors"].get)
    assert result["errors"][worst] < 1e-6, (worst, result["errors"][worst])  # FSDP2 reduces in float32


def test_gpu_check_script_runs_on_cpu(setup, tmp_path):
    # scripts/gpu_checks/tp_pp.sh end to end on tiny models (the CPU smoke mode, both dtype passes): every check
    # reports, none FAILs, and each gradient check names the parameters that differ most.
    import json
    import os
    import shutil
    import subprocess
    import sys

    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    tmp, models, _ = setup
    root = Path(__file__).parent.parent
    env = dict(
        os.environ,
        PGS_PY=sys.executable,
        PGS_SRC=str(root / "src"),
        PGS_DEVICE="cpu",
        PGS_DTYPES="fp32,bf16",
        PGS_SEQ="32",
        PGS_QWEN3=models["qwen3"],
        PGS_QWEN35=models["qwen3_5"],
        CUDA_VISIBLE_DEVICES="",
    )
    run = subprocess.run(
        ["bash", str(root / "scripts/gpu_checks/tp_pp.sh")], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    by_name = {line["check"]: line for line in lines}
    failed = {n: line for n, line in by_name.items() if line["status"] == "FAIL"}
    assert not failed and run.returncode == 0, (failed, run.stderr[-2000:])
    names = ["tp2_cpu_qwen3", "tp2_cpu_qwen35", "tp1_qwen35_fp32"]
    pipelines = ["pp_dualpipev_qwen3_fp32", "pp_tp_zbv_qwen35_bf16"]
    if P.same_rank_stages_run_on_gloo():
        names += pipelines
    else:  # reported as skipped, with the reason
        assert all(by_name[name]["status"] == "SKIP" and "gloo" in by_name[name]["reason"] for name in pipelines)
    for name in names:
        assert by_name[name]["status"] == "PASS", by_name[name]
        assert by_name[name]["worst"], by_name[name]
    assert "bf16_noise" in by_name["tp1_qwen3_bf16"]["worst"][0]
    assert by_name["summary"]["status"] == "PASS"
