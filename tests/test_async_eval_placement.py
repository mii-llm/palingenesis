"""eval_tasks placement: the configuration of an asynchronous evaluator and the checks that keep it off the GPUs
that train (no GPU, no model: nvidia-smi listings are given as text)."""

import pytest

from palingenesis.eval.config import EvalTasksConfig
from palingenesis.eval.placement import (
    child_environment,
    host_memory_error,
    overlap_error,
    parse_listing,
    rank_gpu,
    uuids,
)

TASK = {"name": "plus", "dataset": "rows.jsonl", "prompt": "{{question}}", "scorer": "numeric"}

LISTING = """GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaaa)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-bbbb)
GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-cccc)
  MIG 4g.40gb     Device  0: (UUID: MIG-c4c4)
  MIG 3g.40gb     Device  1: (UUID: MIG-c3c3)
"""


def section(**options):
    return EvalTasksConfig(tasks=[TASK], **options)


def test_defaults_are_inline_and_valid():
    spec = section()
    assert spec.placement == "inline" and not spec.is_async
    assert spec.errors() == []
    assert spec.trainer_errors(save_steps=0, eval_every=100) == []
    assert EvalTasksConfig().errors() == []  # no tasks: nothing checked


@pytest.mark.parametrize(
    "options, message",
    [
        ({"placement": "sideways"}, "placement must be inline or async"),
        ({"devices": [6]}, "only used with eval_tasks.placement async"),
        ({"transport": "checkpoint"}, "only used with eval_tasks.placement async"),
        ({"placement": "async"}, "exactly one of eval_tasks.devices"),
        ({"placement": "async", "devices": [6], "service": "auto"}, "exactly one of"),
        ({"placement": "async", "devices": [6, 6]}, "distinct GPU indices"),
        ({"placement": "async", "service": "nohost"}, "auto or host:port"),
        ({"placement": "async", "service": "auto", "transport": "disk"}, "push or checkpoint"),
        ({"placement": "async", "service": "h:1", "transport": "checkpoint"}, "not host:port"),
        ({"placement": "async", "service": "auto", "transport": "checkpoint", "push_gbps": 5}, "only used with"),
        ({"placement": "async", "service": "auto", "backlog": "all"}, "backlog all needs"),
        ({"placement": "async", "service": "auto", "select": "plus/plain/score"}, "select needs"),
        ({"placement": "async", "service": "auto", "transport": "checkpoint", "select": "plus"}, "<task>/<mode>"),
        ({"placement": "async", "service": "auto", "engine": {"gpu_mem": 0.5}}, "unknown option(s) gpu_mem"),
        ({"placement": "async", "service": "auto", "final_timeout_s": -1}, ">= 0"),
    ],
)
def test_placement_errors(options, message):
    problems = section(**options).errors()
    assert any(message in p for p in problems), problems


def test_valid_async_sections():
    assert section(placement="async", devices=[6, 7]).errors() == []
    assert section(placement="async", devices="6,7").device_list() == [6, 7]
    assert section(placement="async", service="node3:7100", push_gbps=20).errors() == []
    spec = section(
        placement="async", service="auto", transport="checkpoint", backlog="all", select="plus/plain/score",
        engine={"backend": "vllm", "gpu_memory_utilization": 0.8, "replicas": 2},
    )  # fmt: skip
    assert spec.errors() == []


def test_trainer_errors():
    spec = section(placement="async", service="auto", transport="checkpoint", every=200)
    assert "set train.save_steps" in spec.trainer_errors(save_steps=0, eval_every=0)[0]
    assert "multiple of" in spec.trainer_errors(save_steps=150, eval_every=0)[0]
    assert spec.trainer_errors(save_steps=100, eval_every=0) == []
    assert "tp_degree" in spec.trainer_errors(save_steps=100, eval_every=0, tp_degree=2, trainer="train")[0]
    assert (
        "pp_degree"
        in section(placement="async", devices=[1]).trainer_errors(save_steps=0, eval_every=5, pp_degree=2)[0]
    )
    every_from_trainer = section(placement="async", service="auto", transport="checkpoint")
    assert every_from_trainer.trainer_errors(save_steps=50, eval_every=100) == []


def test_listing_and_uuids():
    devices = parse_listing(LISTING)
    assert [d["uuid"] for d in devices] == ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]
    assert devices[2]["mig"] == ["MIG-c4c4", "MIG-c3c3"]
    assert uuids([0, 1], {}, devices) == {0: "GPU-aaaa", 1: "GPU-bbbb"}
    # remapped by CUDA_VISIBLE_DEVICES: index 0 of this view is physical GPU 1
    assert uuids([0, 1], {"CUDA_VISIBLE_DEVICES": "1,0"}, devices) == {0: "GPU-bbbb", 1: "GPU-aaaa"}
    assert uuids([0], {"CUDA_VISIBLE_DEVICES": "MIG-c3c3"}, devices) == {0: "MIG-c3c3"}
    assert uuids([3], {"CUDA_VISIBLE_DEVICES": "0"}, devices) == {}  # not visible
    assert rank_gpu(1, {"CUDA_VISIBLE_DEVICES": "2,0"}, devices)["uuid"] == "GPU-aaaa"


def test_overlap_is_refused_and_mig_is_not():
    training = [{"uuid": "GPU-aaaa", "label": "GPU 0"}, {"uuid": "GPU-bbbb", "label": "GPU 1"}]
    assert overlap_error(training, [{"uuid": "GPU-cccc", "label": "GPU 2"}]) == ""
    error = overlap_error(training, [{"uuid": "GPU-bbbb", "label": "GPU 1"}])
    assert "GPU 1 (GPU-bbbb) trains" in error and "eval_tasks.devices: [6, 7]" in error
    # the same physical GPU split by MIG: two devices
    assert overlap_error([{"uuid": "MIG-c4c4", "label": "GPU 0"}], [{"uuid": "MIG-c3c3", "label": "GPU 0"}]) == ""
    engines = [{"uuid": "GPU-cccc", "label": "engine 10.0.0.2:7000", "what": "holds a rollout engine"}]
    assert "holds a rollout engine" in overlap_error(engines, [{"uuid": "gpu-CCCC", "label": "GPU 2"}])


def test_single_gpu_message():
    error = overlap_error([{"uuid": "GPU-aaaa", "label": "GPU 0"}], [{"uuid": "GPU-aaaa", "label": "GPU 0"}], True)
    assert "GPU 0 (GPU-aaaa) trains" in error
    for alternative in ("placement: inline", "MIG", "pgs eval watch"):
        assert alternative in error


def test_child_environment():
    parent = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "6", "MASTER_ADDR": "x", "CUDA_VISIBLE_DEVICES": "0,1,4,5,6,7",
              "HF_HOME": "/hf", "TORCHELASTIC_RUN_ID": "r"}  # fmt: skip
    env = child_environment([4, 5], parent)
    assert env["CUDA_VISIBLE_DEVICES"] == "6,7"  # indices in the parent's view
    assert env["HF_HOME"] == "/hf" and env["OMP_NUM_THREADS"] == "4"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "TORCHELASTIC_RUN_ID"} & set(env)
    assert child_environment([6], {})["CUDA_VISIBLE_DEVICES"] == "6"
    with pytest.raises(ValueError, match="not visible"):
        child_environment([3], {"CUDA_VISIBLE_DEVICES": "0,1"})


def test_host_memory_check_points_to_checkpoint_transport():
    assert host_memory_error(10 * 2**30, available=100 * 2**30) == ""
    error = host_memory_error(140 * 2**30, available=200 * 2**30)
    assert "140.0 GiB" in error and "transport checkpoint" in error
    assert host_memory_error(1, available=None) in ("", host_memory_error(1))
