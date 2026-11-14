"""What several ranks must agree on, with 3 CPU processes (gloo); 2 would hide a rank that decides alone.

Each rank reaches a filesystem check at its own time and computes its own floating-point results. A decision
taken per rank (is the pre-tokenized cache valid? which checkpoint is the latest? is this evaluation loss a
new best?) can differ between ranks, and the ranks then run different collectives: the run hangs or pairs up
unrelated collectives. These tests make the ranks disagree on purpose and check that rank 0's view decides,
that a path only rank 0 can read raises on every rank, and that the trainer's checkpoints resume on the same
and on another number of ranks."""

import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

import distill_helpers as H  # noqa: E402

WORLD = 3


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank, world, port, out, fn, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        result = {"ok": fn(rank, world, *args)}
    except Exception as e:  # noqa: BLE001 — reported to the test
        result = {"error": type(e).__name__, "message": str(e)}
    finally:
        dist.destroy_process_group()
    Path(out, f"rank{rank}.json").write_text(json.dumps(result))


def _run(fn, tmp_path, *args, world: int = WORLD, timeout: float = 180.0) -> list[dict]:
    """fn(rank, world, *args) on `world` gloo ranks; every rank's result. A hang fails the test
    (a rank waiting in a collective the others never run) instead of blocking it."""
    out = tmp_path / f"results-{time.monotonic_ns()}"
    out.mkdir()
    ctx = mp.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_entry, args=(r, world, port, str(out), fn, args)) for r in range(world)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + timeout
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
    hung = [r for r, p in enumerate(procs) if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join()
    assert not hung, f"rank(s) {hung} still waiting after {timeout}s: the ranks ran different collectives"
    return [json.loads((out / f"rank{r}.json").read_text()) for r in range(world)]


def _aligned() -> int:
    """A collective after the code under test: every rank must still be at the same one."""
    t = torch.tensor([dist.get_rank()])
    dist.all_reduce(t)
    return int(t)


# ── distributed.main_value / rank_values ───────────────────────────────────────


def _values(rank, world):
    from palingenesis.distributed import main_value, rank_values

    return [main_value({"path": f"ckpt-of-rank-{rank}"}), main_value(None if rank == 0 else "x"), rank_values(rank * 2)]


def test_main_value_is_rank_0s_and_rank_values_lists_every_rank(tmp_path):
    for got in _run(_values, tmp_path):
        assert got["ok"] == [{"path": "ckpt-of-rank-0"}, None, [0, 2, 4]]


def test_one_process_needs_no_process_group():
    from palingenesis.distributed import main_value, rank_values

    assert main_value("a") == "a" and rank_values(True) == [True]


# ── the pre-tokenized cache ────────────────────────────────────────────────────


def _write_cache(cache_dir, fingerprint):
    from palingenesis.data import PRETOK_DATA, PRETOK_META

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / PRETOK_DATA).write_bytes(b"")
    (Path(cache_dir) / PRETOK_META).write_text(json.dumps({"fingerprint": fingerprint}))


def _pretokenize(rank, world, cache_dirs, late_rank):
    from palingenesis.train import _ensure_pretokenized

    built = []

    def materialize(reason):
        built.append(reason)
        _write_cache(cache_dirs[rank], "fp")

    if rank == late_rank:  # reaches the check after rank 0 has written the cache (a slower model load)
        time.sleep(3.0)
    _ensure_pretokenized(cache_dirs[rank], "fp", materialize, wait_s=0.0)
    return [built, _aligned()]


def test_rank_0_decides_whether_the_cache_is_built(tmp_path):
    """A rank arriving after rank 0 built the cache sees a valid one; it must still take part in the barrier."""
    shared = [str(tmp_path / "cache")] * WORLD
    got = _run(_pretokenize, tmp_path, shared, 2)
    assert [g["ok"][0] for g in got] == [["no cache found"], [], []]
    assert all(g["ok"][1] == sum(range(WORLD)) for g in got)

    got = _run(_pretokenize, tmp_path, shared, 2)  # valid now: nobody builds
    assert [g["ok"][0] for g in got] == [[], [], []]


def test_a_cache_that_is_not_shared_raises_on_every_rank(tmp_path):
    per_rank = [str(tmp_path / f"node{r}" / "cache") for r in range(WORLD)]  # each rank its own filesystem
    got = _run(_pretokenize, tmp_path, per_rank, -1)
    assert [g.get("error") for g in got] == ["ConfigError"] * WORLD
    assert "rank(s) [1, 2]" in got[0]["message"] and "shared" in got[0]["message"]


# ── the checkpoint to resume ───────────────────────────────────────────────────


def _checkpoint(output_dir, step):
    from palingenesis.checkpoint import TRAINING_META_FILE

    d = Path(output_dir) / f"step-{step}"
    d.mkdir(parents=True)
    (d / TRAINING_META_FILE).write_text(json.dumps({"step": step}))
    return str(d)


def _resume(rank, world, roots, resume_from):
    from palingenesis.config import Config
    from palingenesis.train import _resolve_resume

    os.chdir(roots[rank])  # the same path on every rank, on this rank's filesystem
    c = Config()
    c.train.output_dir = "out"
    c.train.resume_from = resume_from
    return [_resolve_resume(c), _aligned()]


def _roots(tmp_path, shared: bool) -> list[str]:
    roots = [tmp_path / ("shared" if shared else f"node{r}") for r in range(WORLD)]
    for root in roots:
        root.mkdir(exist_ok=True)
    return [str(r) for r in roots]


def test_every_rank_resumes_the_checkpoint_rank_0_found(tmp_path):
    roots = _roots(tmp_path, shared=True)
    for g in _run(_resume, tmp_path, roots, "auto"):  # nothing to resume: a fresh start
        assert g["ok"] == [None, sum(range(WORLD))]
    _checkpoint(Path(roots[0]) / "out", 2)
    _checkpoint(Path(roots[0]) / "out", 4)
    for g in _run(_resume, tmp_path, roots, "auto"):
        assert g["ok"] == [str(Path("out") / "step-4"), sum(range(WORLD))]


def test_an_output_dir_that_is_not_shared_raises_on_every_rank(tmp_path):
    """Before: rank 0 resumed step 4 while the ranks of the other nodes started from step 0."""
    roots = _roots(tmp_path, shared=False)
    _checkpoint(Path(roots[0]) / "out", 4)
    got = _run(_resume, tmp_path, roots, "auto")
    assert [g.get("error") for g in got] == ["ConfigError"] * WORLD
    assert "rank(s) [1, 2]" in got[1]["message"]


def test_resume_position_follows_the_number_of_data_ranks():
    from palingenesis.train import _rescale_resume_position

    assert _rescale_resume_position(9, None, 4) == 9  # a checkpoint without the record
    assert _rescale_resume_position(9, 4, 4) == 9
    assert _rescale_resume_position(9, 4, 2) == 18  # half the ranks: each reads twice the share
    assert _rescale_resume_position(9, 2, 8) == 2


# ── the best checkpoint ────────────────────────────────────────────────────────


def _best(rank, world, output_dir):
    from torch.distributed.device_mesh import init_device_mesh

    from palingenesis.checkpoint import BestModelTracker
    from palingenesis.config import ParallelConfig
    from palingenesis.distributed import apply_fsdp

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
    model = apply_fsdp(H.tiny_llama(64), mesh, ParallelConfig(), bf16=False)
    tracker = BestModelTracker(output_dir)
    # each rank's own evaluation of the same model: equal up to the last bits, then a rank that
    # alone sees an improvement, then one that alone does not
    losses = [(2.0, 2.0, 2.0), (2.0, 2.0 - 1e-7, 2.0), (1.5, 1.5, 2.5)]
    saved = [tracker.update(per_rank[rank], step, model, None, True) for step, per_rank in enumerate(losses, 1)]
    return [saved, tracker.best_loss, tracker.best_step, _aligned()]


def test_best_checkpoint_follows_rank_0s_evaluation(tmp_path):
    out = str(tmp_path / "out")
    for g in _run(_best, tmp_path, out):
        assert g["ok"] == [[True, False, True], 1.5, 3, sum(range(WORLD))]
    assert json.loads((Path(out) / "best" / "best_meta.json").read_text()) == {"step": 3, "eval_loss": 1.5}


# ── the trainer: checkpoints, resume, another number of ranks ──────────────────


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ranks")
    tok = H.tokenizer()
    H.save_model(tmp / "student", tok, seed=0)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"What is {i} plus {i + 1}?"},
                {"role": "assistant", "content": f"The answer is {2 * i + 1}, since {i} + {i + 1} = {2 * i + 1}."},
            ]
        }
        for i in range(96)
    ]
    return tmp, H.write_rows(tmp / "rows.jsonl", rows)


def _config(setup, name: str, fsdp: bool = True):
    tmp, data = setup
    c = H.cpu_config(tmp / name, tmp / "student", data, steps=6)
    c.train.save_steps = 2
    c.train.keep_checkpoints = 0
    c.parallel.fsdp = fsdp
    return c


def _train(rank, world, config):
    from transformers import AutoModelForCausalLM

    from palingenesis.train import _train_on

    result = _train_on(config, {"tracker": None}, None, rank, world, torch.device("cpu"))
    if rank != 0:
        return None
    model = AutoModelForCausalLM.from_pretrained(Path(config.train.output_dir) / "final")
    return {
        "steps": result.global_step,
        "loss": result.metrics["train/loss"],
        "weights": {k: v.flatten().tolist() for k, v in model.state_dict().items()},
    }


def _keep_only(output_dir: str, step: int) -> None:
    """The output directory as a run stopped after the save at `step` leaves it."""
    for d in Path(output_dir).iterdir():
        if d.name != f"step-{step}":
            shutil.rmtree(d) if d.is_dir() else d.unlink()


@pytest.fixture(scope="module")
def three_rank_run(setup, tmp_path_factory):
    config = _config(setup, "full")
    got = _run(_train, tmp_path_factory.mktemp("full"), config, timeout=600.0)
    assert all("ok" in g for g in got), got
    return config, got[0]["ok"]


def test_fsdp_checkpoint_holds_every_ranks_state(three_rank_run):
    config, _ = three_rank_run
    step = Path(config.train.output_dir) / "step-4"
    meta = json.loads((step / "training_meta.json").read_text())
    assert (meta["step"], meta["data_ranks"]) == (4, WORLD)
    assert sorted(p.name for p in step.glob("rng_state*")) == [
        "rng_state.rank1.safetensors",
        "rng_state.rank2.safetensors",
        "rng_state.safetensors",
    ]
    assert not list(step.glob("*.tmp"))


def test_resumed_run_ends_with_the_uninterrupted_runs_weights(setup, three_rank_run, tmp_path):
    full_config, full = three_rank_run
    config = _config(setup, "resumed")
    shutil.copytree(full_config.train.output_dir, config.train.output_dir)
    _keep_only(config.train.output_dir, 4)
    config.train.resume_from = "auto"
    got = _run(_train, tmp_path, config, timeout=600.0)
    assert all("ok" in g for g in got), got
    resumed = got[0]["ok"]
    assert resumed["steps"] == full["steps"] == 6
    assert resumed["loss"] == pytest.approx(full["loss"], rel=1e-5)
    for name, w in full["weights"].items():
        torch.testing.assert_close(torch.tensor(resumed["weights"][name]), torch.tensor(w), rtol=1e-5, atol=1e-6)


def test_checkpoint_resumes_on_fewer_ranks(setup, three_rank_run, tmp_path):
    """DCP re-shards the model and optimizer state; the position in the epoch follows the rank count."""
    full_config, _ = three_rank_run
    config = _config(setup, "fewer")
    shutil.copytree(full_config.train.output_dir, config.train.output_dir)
    _keep_only(config.train.output_dir, 4)
    config.train.resume_from = "auto"
    got = _run(_train, tmp_path, config, world=2, timeout=600.0)
    assert all("ok" in g for g in got), got
    assert got[0]["ok"]["steps"] == 6
    assert got[0]["ok"]["loss"] == got[0]["ok"]["loss"]  # finite: the weights were loaded, not reinitialised
    meta = json.loads((Path(config.train.output_dir) / "step-6" / "training_meta.json").read_text())
    assert meta["data_ranks"] == 2


def test_sharded_checkpoint_refuses_one_process(setup, three_rank_run):
    """Before: the weights were skipped silently and the run continued step 4 on the pretrained model."""
    full_config, _ = three_rank_run
    config = _config(setup, "single", fsdp=False)
    shutil.copytree(full_config.train.output_dir, config.train.output_dir)
    _keep_only(config.train.output_dir, 4)
    config.train.resume_from = "auto"
    with pytest.raises(RuntimeError, match="sharded \\(DCP\\) state"):
        H.run_cpu(config)


def test_single_process_checkpoint_refuses_several_ranks(setup, tmp_path):
    config = _config(setup, "from-single", fsdp=False)
    config.train.max_steps = 2
    H.run_cpu(config)
    config.parallel.fsdp = True
    config.train.max_steps = 4
    config.train.resume_from = "auto"
    got = _run(_train, tmp_path, config, world=2, timeout=600.0)
    assert [g.get("error") for g in got] == ["RuntimeError"] * 2
    assert "single-process run" in got[0]["message"]


# ── devices, launchers, data order ─────────────────────────────────────────────


def test_local_rank_must_be_a_visible_device():
    from palingenesis.distributed import local_device_index

    assert local_device_index(3, 8) == 3
    with pytest.raises(RuntimeError, match="LOCAL_RANK=3 but this process sees 1 GPU"):
        local_device_index(3, 1)  # one GPU bound per task while LOCAL_RANK still counts them


def test_slurm_tasks_started_without_torchrun_are_warned(monkeypatch, caplog):
    from palingenesis.distributed import setup_distributed

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING", logger="palingenesis.distributed"):
        assert setup_distributed() == (0, 0, 1)
    assert "1 of 16 tasks" in caplog.text and "torchrun" in caplog.text

    caplog.clear()
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")
    with caplog.at_level("WARNING", logger="palingenesis.distributed"):
        setup_distributed()
    assert not caplog.text


def test_every_rank_draws_its_own_order_of_sources():
    from palingenesis.data import MixedDataset

    a, b = [{"s": "a"}] * 64, [{"s": "b"}] * 64

    def order(**kw):
        return "".join(x["s"] for x in MixedDataset([a, b], [0.5, 0.5], seed=7, epoch_draws=48, **kw))

    assert order(rank=0) == order()  # one process: unchanged
    assert order(rank=1) == order(rank=1)  # the ranks of a context-parallel group read the same rows
    assert len({order(rank=r) for r in range(8)}) == 8
