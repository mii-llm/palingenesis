"""NVIDIA vs AMD decisions (palingenesis.accelerator) and the call sites that use them.

No GPU needed: torch's device probes are replaced by a fake GPU. On ROCm, torch reports
a CUDA-style capability derived from the gfx target (gfx90a -> (9, 0), gfx942 -> (9, 4)),
which is what made raw capability checks wrong on AMD.
"""

import logging
import sys
import types

import pytest
import torch
from torch import nn

from palingenesis import accelerator
from palingenesis.config import Config, ConfigError

GPUS = {
    # name: (torch.version.hip, torch.version.cuda, capability, gcnArchName, device name)
    "mi250": ("7.2.53211", None, (9, 0), "gfx90a:sramecc+:xnack-", "AMD Instinct MI250X"),
    "mi300": ("7.2.53211", None, (9, 4), "gfx942:sramecc+:xnack-", "AMD Instinct MI300X"),
    "mi355": ("7.2.53211", None, (9, 5), "gfx950:sramecc+:xnack-", "AMD Instinct MI355X"),
    "a100": (None, "12.9", (8, 0), None, "NVIDIA A100-SXM4-80GB"),
    "h100": (None, "12.9", (9, 0), None, "NVIDIA H100 80GB HBM3"),
    # DGX Spark: an integrated GPU (CUDA's `integrated` property) that allocates from the host's RAM
    "gb10": (None, "13.0", (12, 1), None, "NVIDIA GB10"),
}
UNIFIED_MEMORY = {"gb10"}


@pytest.fixture
def gpu(monkeypatch):
    """gpu("mi300") makes torch look like that machine; gpu(None) like a ROCm/CUDA build with no GPU."""

    def use(name, *, hip=None, cuda="12.9"):
        if name is not None:
            hip, cuda, capability, arch, device = GPUS[name]
            props = types.SimpleNamespace(name=device, major=capability[0], minor=capability[1], total_memory=64 << 30)
            props.is_integrated = int(name in UNIFIED_MEMORY)
            if arch:
                props.gcnArchName = arch
            monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
            monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: name is not None)
        monkeypatch.setattr(torch.version, "hip", hip, raising=False)
        monkeypatch.setattr(torch.version, "cuda", cuda, raising=False)

    return use


@pytest.fixture
def train_mod(monkeypatch):
    """palingenesis.train with distributed setup, logging and global torch state kept local to the test."""
    import palingenesis.train as train_mod

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "")
    monkeypatch.setattr(train_mod, "setup_distributed", lambda: (0, 0, 1))
    monkeypatch.setattr(train_mod, "setup_logging", lambda rank: None)
    monkeypatch.setattr(train_mod.torch.cuda, "manual_seed_all", lambda seed: None)
    precision = torch.get_float32_matmul_precision()
    yield train_mod
    torch.set_float32_matmul_precision(precision)


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize(
    "name,rocm,arch",
    [("mi250", True, "gfx90a"), ("mi300", True, "gfx942"), ("h100", False, "sm_90"), ("gb10", False, "sm_121")],
)
def test_platform_and_arch(gpu, name, rocm, arch):
    gpu(name)
    assert accelerator.is_rocm() is rocm
    assert accelerator.gpu_arch() == arch  # gfx feature flags stripped
    line = accelerator.describe()
    assert GPUS[name][4] in line and arch in line and ("ROCm" if rocm else "CUDA") in line
    assert ("shared with the host" in line) is (name in UNIFIED_MEMORY)


# ------------------------------------------------------------- unified memory

GIB = 1 << 30


@pytest.fixture
def memory(monkeypatch):
    """memory(cuda_free, host_available): what CUDA and the kernel report, in GiB, on the fake 64 GiB device."""

    def use(cuda_free, host_available):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (cuda_free * GIB, 64 * GIB))
        monkeypatch.setattr(
            accelerator, "host_available_bytes", lambda: None if host_available is None else host_available * GIB
        )

    return use


def test_host_available_reads_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       127582240 kB\nMemFree:  1000 kB\nMemAvailable:   121634816 kB\nCached: 5 kB\n")
    assert accelerator.host_available_bytes(str(meminfo)) == 121634816 * 1024
    meminfo.write_text("MemTotal:       127582240 kB\n")
    assert accelerator.host_available_bytes(str(meminfo)) is None
    assert accelerator.host_available_bytes(str(tmp_path / "missing")) is None


@pytest.mark.parametrize(
    "name,cuda_free,host,free",
    [
        ("gb10", 30, 58, 58),  # page cache: CUDA says 30 GiB free, the kernel would hand out 58
        ("gb10", 30, 100, 64),  # never more than the device
        ("gb10", 30, 12, 30),  # CUDA's own figure when it is the larger one
        ("gb10", 30, None, 30),  # MemAvailable unreadable
        ("h100", 30, 500, 30),  # discrete GPU: host RAM is another pool
    ],
)
def test_memory_info_counts_reclaimable_host_memory_on_unified_memory(gpu, memory, name, cuda_free, host, free):
    gpu(name)
    memory(cuda_free, host)
    assert accelerator.is_unified_memory() is (name in UNIFIED_MEMORY)
    assert accelerator.memory_info() == (free * GIB, 64 * GIB)


@pytest.mark.parametrize(
    "name,host,budget", [("gb10", 40, 40), ("gb10", 100, 64), ("gb10", None, 64), ("h100", 40, 64), ("a100", 500, 64)]
)
def test_memory_budget_is_what_the_host_leaves_on_unified_memory(gpu, memory, name, host, budget):
    gpu(name)
    memory(1, host)
    assert accelerator.memory_budget_bytes() == budget * GIB


def test_unified_memory_needs_a_gpu(gpu):
    gpu(None)
    assert accelerator.is_unified_memory() is False


@pytest.mark.parametrize("name,mode", [("gb10", "device"), ("h100", "cpu")])
def test_surgery_gram_mode_sees_reclaimable_memory(gpu, memory, name, mode):
    from palingenesis import surgery

    gpu(name)
    memory(10, 60)  # 10 GiB free by CUDA's count, 60 GiB available on the host
    # 2 * K * P_edit = 16 GiB of stored gradients, 9.6 GiB + 8 GiB of headroom
    chosen, info = surgery.choose_gram_mode("auto", 8, GIB, "cuda:0")
    assert chosen == mode
    assert info["device_free_bytes"] == (60 if name == "gb10" else 10) * GIB


@pytest.mark.parametrize("name,warned", [("gb10", True), ("h100", False)])
def test_rl_cpu_offload_warns_on_unified_memory(gpu, memory, caplog, name, warned):
    from palingenesis.rl import trainer

    gpu(name)
    memory(60, 60)
    with caplog.at_level(logging.WARNING, logger="palingenesis.rl.trainer"):
        trainer._check_host_memory(nn.Linear(4, 4), world=1)
    assert ("saves no memory" in caplog.text) is warned


def test_no_gpu(gpu):
    gpu(None, hip="7.2.0", cuda=None)
    assert accelerator.is_rocm() and accelerator.gpu_arch() is None
    assert accelerator.describe() == "not available (torch built for ROCm 7.2.0)"


# ------------------------------------------------------------------ messages


def test_no_gpu_message_is_platform_specific(gpu, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    gpu(None, hip="7.2.0", cuda=None)
    rocm = accelerator.no_gpu_message()
    assert "AMD" in rocm and "amd-smi" in rocm and "nvidia" not in rocm.lower()
    assert "HIP_VISIBLE_DEVICES=''" in rocm  # an empty value hides every GPU: show it
    gpu(None, hip=None, cuda="12.9")
    cuda = accelerator.no_gpu_message()
    assert "nvidia-smi" in cuda and "CUDA 12.9" in cuda and "ROCm" not in cuda
    gpu(None, hip=None, cuda=None)
    assert "CPU-only" in accelerator.no_gpu_message()


def test_pgs_version_names_the_platform(gpu, monkeypatch, capsys):
    from palingenesis import cli

    gpu("mi300")
    monkeypatch.setattr(sys, "argv", ["pgs", "version"])
    cli.main()
    out = capsys.readouterr().out
    assert "CUDA:" not in out
    assert "GPU: 8x AMD Instinct MI300X (64 GB), ROCm 7.2.53211, arch gfx942" in out


def test_train_reports_missing_gpu_for_the_installed_build(gpu, train_mod):
    gpu(None, hip="7.2.0", cuda=None)
    with pytest.raises(RuntimeError, match="No usable AMD GPU"):
        train_mod.train(Config())


# ------------------------------------------------------------ visible devices


@pytest.mark.parametrize(
    "env,ok",
    [
        ({"HIP_VISIBLE_DEVICES": "0,1", "CUDA_VISIBLE_DEVICES": "0,1"}, True),
        ({"HIP_VISIBLE_DEVICES": "2"}, True),
        ({"CUDA_VISIBLE_DEVICES": "3"}, True),
        ({"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "1"}, True),  # empty = unset, as in vLLM
        ({"ROCR_VISIBLE_DEVICES": "4", "HIP_VISIBLE_DEVICES": "0"}, True),  # HIP indexes into ROCR's list
        ({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"}, False),
    ],
)
def test_visible_devices_on_rocm(gpu, env, ok):
    gpu("mi300")
    if ok:
        accelerator.check_visible_devices(env)
    else:
        with pytest.raises(RuntimeError, match="select different GPUs"):
            accelerator.check_visible_devices(env)


def test_visible_devices_ignored_on_cuda(gpu):
    gpu("h100")  # HIP_VISIBLE_DEVICES means nothing to CUDA
    accelerator.check_visible_devices({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"})


def test_train_checks_visible_devices_before_loading(gpu, train_mod, monkeypatch):
    gpu("mi250")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="select different GPUs"):
        train_mod.train(Config())


def test_train_refuses_several_ranks_without_fsdp(gpu, train_mod, monkeypatch):
    """There is no DDP path: with parallel.fsdp false every rank would train its own model."""
    gpu("mi250")
    monkeypatch.setattr(train_mod, "setup_distributed", lambda: (1, 1, 2))
    monkeypatch.setattr(train_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("loaded"))
    config = Config()
    config.parallel.fsdp = False
    with pytest.raises(ConfigError, match="2 separate models"):
        train_mod.train(config)


# ------------------------------------------------------------------ attention


@pytest.fixture
def attn_env(monkeypatch):
    """attn_env(flash_attn=..., kernels=...) fakes which attention packages are installed."""
    import transformers.utils as tu

    def use(*, flash_attn: bool, kernels: bool):
        monkeypatch.setattr(tu, "is_flash_attn_2_available", lambda: flash_attn)
        monkeypatch.setattr(tu, "is_kernels_available", lambda *a, **k: kernels)

    return use


@pytest.mark.parametrize("name,suggests_aiter", [("mi250", False), ("mi300", True), ("mi355", True)])
def test_flash_attention_2_missing_on_rocm_is_a_config_error(gpu, attn_env, name, suggests_aiter):
    gpu(name)
    # kernels installed does not help on ROCm: transformers would fetch kernels-community/flash-attn2 (CUDA only)
    attn_env(flash_attn=False, kernels=True)
    with pytest.raises(ConfigError, match="sdpa") as e:
        accelerator.check_attn_implementation("flash_attention_2")
    assert ("attn_implementation: kernels-community/aiter-flash-attn" in str(e.value)) is suggests_aiter


@pytest.mark.parametrize("name,ok", [("mi250", False), ("mi300", True), ("mi355", True)])
def test_aiter_only_on_the_gpus_it_ships_configs_for(gpu, attn_env, name, ok):
    """aiter reads configs/<arch>-MHA-DEFAULT.json at its first call; there is none for gfx90a."""
    gpu(name)
    attn_env(flash_attn=True, kernels=True)
    for spec in ("kernels-community/aiter-flash-attn@v2", "kernels-community/aiter-flash-attn"):
        if ok:
            accelerator.check_attn_implementation(spec)
        else:
            with pytest.raises(ConfigError, match="gfx90a") as e:
                accelerator.check_attn_implementation(spec)
            assert "sdpa" in str(e.value) and "ROCm flash-attn" in str(e.value)


def test_flash_attention_2_on_nvidia_unchanged(gpu, attn_env):
    gpu("h100")
    attn_env(flash_attn=True, kernels=False)
    accelerator.check_attn_implementation("flash_attention_2")
    attn_env(flash_attn=False, kernels=True)  # transformers' own Hub fallback (kernels-community/flash-attn2)
    accelerator.check_attn_implementation("flash_attention_2")
    attn_env(flash_attn=False, kernels=False)
    with pytest.raises(ConfigError, match="model.attn_implementation: sdpa"):
        accelerator.check_attn_implementation("flash_attention_2")


def test_rocm_flash_attn_build_is_accepted(gpu, attn_env):
    gpu("mi300")
    attn_env(flash_attn=True, kernels=False)
    accelerator.check_attn_implementation("flash_attention_2")


def test_hub_kernel_needs_the_kernels_package(gpu, attn_env):
    gpu("mi300")
    attn_env(flash_attn=False, kernels=False)
    with pytest.raises(ConfigError, match="pip install kernels"):
        accelerator.check_attn_implementation("kernels-community/aiter-flash-attn@v2")
    attn_env(flash_attn=False, kernels=True)
    accelerator.check_attn_implementation("kernels-community/aiter-flash-attn@v2")
    for name in ("sdpa", "eager", "flex_attention"):  # left to transformers
        accelerator.check_attn_implementation(name)


def test_hub_kernel_names():
    assert accelerator.is_hub_kernel("kernels-community/aiter-flash-attn@v2")
    assert accelerator.is_hub_kernel("kernels-community/flash-attn2:flash_attn_varlen_func")
    assert not accelerator.is_hub_kernel("flash_attention_2") and not accelerator.is_hub_kernel("a/b/c")
    assert accelerator.is_flash_varlen("flash_attention_2")
    assert accelerator.is_flash_varlen("kernels-community/aiter-flash-attn@v2")
    assert not accelerator.is_flash_varlen("sdpa") and not accelerator.is_flash_varlen("kernels-community/sage")


def test_unavailable_flash_attention_fails_before_any_model_load(gpu, attn_env, train_mod, monkeypatch):
    """No silent switch to sdpa and no second load: the run stops with the config error."""
    gpu("mi250")
    attn_env(flash_attn=False, kernels=False)
    tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=0)
    monkeypatch.setattr(train_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    loads = []
    monkeypatch.setattr(train_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: loads.append(k))
    config = Config()
    config.model.attn_implementation = "flash_attention_2"
    config.model.use_liger_kernel = False
    with pytest.raises(ConfigError, match="flash-attn is not installed for this ROCm torch"):
        train_mod.train(config)
    assert loads == []
    assert config.model.attn_implementation == "flash_attention_2"


@pytest.mark.parametrize(
    "attn,dtype,autocast",
    [
        ("flash_attention_2", torch.float32, True),  # DPO's fp32 weights: flash kernels take fp16 / bf16 only
        ("kernels-community/aiter-flash-attn@v2", torch.float32, True),
        ("sdpa", torch.float32, False),
        ("flash_attention_2", torch.bfloat16, False),
    ],
)
def test_output_head_check_runs_flash_attention_in_bf16(attn, dtype, autocast):
    """The pre-training head check calls the model's forward; with fp32 weights and flash
    attention it must run under bf16 autocast like training, or the kernel rejects fp32."""
    from palingenesis.logits import verify_output_head

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(_attn_implementation=attn)
            self.embed, self.lm_head = nn.Embedding(64, 16), nn.Linear(16, 64, bias=False)
            self.autocast = []

        def forward(self, input_ids):
            self.autocast.append(torch.is_autocast_enabled("cpu"))
            return types.SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))

    model = Tiny().to(dtype)
    verify_output_head(model, model.lm_head, backbone=lambda input_ids: model.embed(input_ids))
    assert model.autocast == [autocast]


def test_packing_accepts_hub_flash_kernels():
    from palingenesis.packing import check_packing_support

    model = types.SimpleNamespace(config=types.SimpleNamespace(layer_types=["full_attention"]))
    check_packing_support(model, "kernels-community/aiter-flash-attn@v2")
    with pytest.raises(ConfigError, match="not supported"):
        check_packing_support(model, "kernels-community/sage-attention")


# ---------------------------------------------------------- cut cross-entropy


@pytest.mark.parametrize(
    "name,precision,ok",
    [
        ("mi250", "high", False),  # tf32 dots: Triton's AMD backend allows them on gfx942 only
        ("mi355", "high", False),
        ("mi300", "high", True),
        ("mi250", "highest", True),  # CCE then asks for ieee dots
        ("mi250", "medium", True),
        ("h100", "high", True),
        ("a100", "high", True),
    ],
)
def test_cce_gate_by_arch(gpu, name, precision, ok):
    gpu(name)
    allowed, reason = accelerator.cce_support(precision)
    assert allowed is ok
    if not ok:
        assert GPUS[name][3].split(":")[0] in reason and "chunked CE" in reason


def test_cce_gate_logs_in_train(gpu, caplog):
    from palingenesis.train import _cce_supported

    gpu("mi250")
    with caplog.at_level(logging.WARNING, logger="palingenesis.train"):
        assert _cce_supported("high") is False
    assert "gfx90a" in caplog.text
    gpu("h100")
    assert _cce_supported("high") is True


# ------------------------------------------------------------------------ FP8


@pytest.mark.parametrize(
    "name,status",
    [("mi250", "unsupported"), ("mi300", "unvalidated"), ("mi355", "unvalidated"), ("a100", "unsupported")]
    + [("h100", "supported"), ("gb10", "unvalidated")],
)
def test_fp8_support_by_arch(gpu, name, status):
    gpu(name)
    assert accelerator.fp8_support()[0] == status


@pytest.fixture
def fake_torchao(monkeypatch):
    """A stand-in torchao.float8 that records conversions."""
    calls = []

    class Float8LinearConfig:
        @staticmethod
        def from_recipe_name(name):
            return name

    float8 = types.ModuleType("torchao.float8")
    float8.Float8LinearConfig = Float8LinearConfig
    float8.convert_to_float8_training = lambda model, config, module_filter_fn: calls.append(config)
    torchao = types.ModuleType("torchao")
    torchao.float8 = float8
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", float8)
    return calls


@pytest.mark.parametrize(
    "name,converted",
    [("mi250", False), ("a100", False), ("mi300", True), ("mi355", True), ("h100", True), ("gb10", True)],
)
def test_float8_training_gate(gpu, fake_torchao, caplog, name, converted):
    from palingenesis.float8 import apply_float8_training

    gpu(name)
    with caplog.at_level(logging.WARNING, logger="palingenesis.float8"):
        assert apply_float8_training(nn.Sequential(nn.Linear(32, 32))) is converted
    assert bool(fake_torchao) is converted
    if name == "mi250":
        assert "gfx90a has no FP8" in caplog.text
    if name in ("mi300", "gb10"):
        assert "not validated" in caplog.text


@pytest.mark.parametrize("name,auto", [("mi250", False), ("mi300", False), ("h100", True), ("gb10", False)])
def test_autopilot_turns_on_fp8_only_where_validated(gpu, name, auto):
    from palingenesis.autopilot.profile import profile_hardware

    gpu(name)
    assert profile_hardware()["supports_float8"] is auto


# ----------------------------------------------------------------------- Muon


@pytest.mark.parametrize(
    "name,tried", [("mi250", False), ("mi300", False), ("a100", False), ("h100", True), ("gb10", True)]
)
def test_quack_muon_backend_only_on_nvidia_hopper(gpu, monkeypatch, name, tried):
    from palingenesis import muon

    gpu(name)
    attempts = []

    def quack():
        attempts.append(name)
        raise ImportError("quack not installed")

    monkeypatch.setattr(muon, "_QuackOps", quack)
    assert muon._quack_ops() is None
    assert bool(attempts) is tried


# --------------------------------------------------------------- vLLM server


def test_vllm_server_rollout_warns_on_rocm_but_runs(gpu, caplog):
    from palingenesis.opd.rollout import VLLMServerRollout

    posts = []
    server = types.SimpleNamespace(post=lambda path, payload=None: posts.append(path))
    gpu("mi300")
    with caplog.at_level(logging.WARNING, logger="palingenesis.opd.rollout"):
        VLLMServerRollout(server, (0,))
    assert "untested" in caplog.text and posts == ["/init_weight_transfer_engine"]
    caplog.clear()
    gpu("h100")
    with caplog.at_level(logging.WARNING, logger="palingenesis.opd.rollout"):
        VLLMServerRollout(server, (0,))
    assert "untested" not in caplog.text


def test_config_rejects_unknown_attn_implementation():
    import pytest

    from palingenesis.config import Config, ConfigError

    c = Config()
    c.model.attn_implementation = "sdap"
    with pytest.raises(ConfigError, match="attn_implementation"):
        c.validate()
    for ok in ("sdpa", "eager", "kernels-community/aiter-flash-attn@v2"):
        c.model.attn_implementation = ok
        try:
            c.validate()
        except ConfigError as e:
            assert "attn_implementation" not in str(e)


@pytest.fixture
def fake_devices(monkeypatch):
    """torch.cuda's current device, per thread as on a real GPU (a new thread starts on 0)."""
    import threading

    local = threading.local()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: getattr(local, "index", 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: setattr(local, "index", torch.device("cuda", i).index))
    return local


def test_thread_targets_keep_the_rank_device(fake_devices):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    torch.cuda.set_device(3)  # rank 3's main thread
    seen = {}

    def worker(name):
        seen[name] = torch.cuda.current_device()

    for name, target in (("bare", worker), ("wrapped", accelerator.on_current_device(worker))):
        t = threading.Thread(target=target, args=(name,))
        t.start()
        t.join()
    with ThreadPoolExecutor(1, initializer=accelerator.on_current_device(lambda: None)) as pool:
        seen["executor"] = pool.submit(torch.cuda.current_device).result()
    assert seen == {"bare": 0, "wrapped": 3, "executor": 3}


def test_device_scope(monkeypatch):
    entered = []

    class Guard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            entered.append(str(self.device))

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", Guard)
    for device in ("cuda:1", torch.device("cuda", 2), "cuda", "cpu"):
        with accelerator.device_scope(device):
            pass
    assert entered == ["cuda:1", "cuda:2"]  # a bare "cuda" or CPU keeps the current device
