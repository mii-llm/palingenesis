"""Where an asynchronous evaluator may run: never on a GPU that trains.

An evaluator next to the trainer on one GPU would take memory the trainer counts on and compete for its SMs, so
the run would slow down or run out of memory; a reserved memory budget removes neither. GPUs are told apart by
their UUIDs (nvidia-smi), so CUDA_VISIBLE_DEVICES remapping, Slurm cgroups and torchrun's local ranks cannot hide
an overlap. MIG instances have UUIDs of their own and are separate GPUs here (their memory and SMs are partitioned
in hardware), which is the one way to share a physical GPU.

Also here: the environment of an evaluator the trainer starts (torchrun's variables removed, its own
CUDA_VISIBLE_DEVICES) and the host-memory check of the push staging area.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess

# what torchrun / torch.distributed put in a worker's environment: an evaluator started from a rank must not join
# (or be confused by) the trainer's process group
TORCHRUN_VARIABLES = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK",
    "ROLE_WORLD_SIZE", "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "OMP_NUM_THREADS",
)  # fmt: skip
_LINE = re.compile(r"^\s*(GPU|MIG)\b.*?(\d+)?.*\(UUID:\s*([^)]+)\)")
HOST_MEMORY_SHARE = 0.5  # the push staging area may take at most this share of the available host memory


def _strip(uuid: str) -> str:
    return uuid.strip().lower().removeprefix("gpu-").removeprefix("mig-")


def nvidia_devices() -> list[dict]:
    """The devices nvidia-smi lists, in its order: [{index, uuid, mig: [uuid, ...]}] (empty without nvidia-smi)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_listing(out)


def parse_listing(text: str) -> list[dict]:
    """nvidia-smi -L output as [{index, uuid, mig: [uuid, ...]}]."""
    devices: list[dict] = []
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        kind, uuid = match.group(1), match.group(3).strip()
        if kind == "GPU":
            devices.append({"index": len(devices), "uuid": uuid, "mig": []})
        elif devices:
            devices[-1]["mig"].append(uuid)
    return devices


def visible_entries(environ: dict | None = None) -> list[str] | None:
    """The CUDA_VISIBLE_DEVICES entries of an environment (None: unset, every GPU)."""
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [d.strip() for d in visible.split(",") if d.strip()]


def uuid_of(entry: str, devices: list[dict]) -> str:
    """The UUID of a CUDA_VISIBLE_DEVICES entry (an nvidia-smi index or a GPU- / MIG- UUID). A GPU split into MIG
    instances is its first instance, as CUDA enumerates it. "" when unknown."""
    if entry.upper().startswith(("GPU-", "MIG-")):
        return entry
    if entry.isdigit():
        index = int(entry)
        if index < len(devices):
            device = devices[index]
            return device["mig"][0] if device["mig"] else device["uuid"]
    return ""


def uuids(indices: list[int], environ: dict | None = None, devices: list[dict] | None = None) -> dict[int, str]:
    """{index: UUID} of GPU indices in the view of an environment (its CUDA_VISIBLE_DEVICES, else nvidia-smi's
    order). Indices with no UUID found are left out."""
    devices = nvidia_devices() if devices is None else devices
    entries = visible_entries(environ)
    out = {}
    for index in indices:
        if entries is None:
            entry = str(index)
        elif index < len(entries):
            entry = entries[index]
        else:
            continue
        uuid = uuid_of(entry, devices)
        if uuid:
            out[index] = uuid
    return out


def physical_entry(index: int, environ: dict | None = None) -> str:
    """The CUDA_VISIBLE_DEVICES entry of the index-th GPU of an environment's view."""
    entries = visible_entries(environ)
    if entries is None:
        return str(index)
    if index >= len(entries):
        raise ValueError(f"GPU {index} is not visible (CUDA_VISIBLE_DEVICES={','.join(entries)})")
    return entries[index]


def rank_gpu(local_rank: int, environ: dict | None = None, devices: list[dict] | None = None) -> dict:
    """What one trainer rank uses: {host, uuid, label} (uuid "" without nvidia-smi: a CPU run)."""
    found = uuids([local_rank], environ, devices)
    entries = visible_entries(environ)
    label = entries[local_rank] if entries is not None and local_rank < len(entries) else str(local_rank)
    return {"host": socket.gethostname(), "uuid": found.get(local_rank, ""), "label": f"GPU {label}"}


def overlap_error(training: list[dict], evaluator: list[dict], single_gpu: bool = False) -> str:
    """ "" when no evaluator GPU is one that trains (or holds a rollout engine), else the error to show.
    training / evaluator: [{uuid, label}] (label: how the user named the GPU)."""
    used = {_strip(t["uuid"]): t for t in training if t.get("uuid")}
    shared = [(e, used[_strip(e["uuid"])]) for e in evaluator if e.get("uuid") and _strip(e["uuid"]) in used]
    if not shared:
        return ""
    evaluator_gpu, trainer_gpu = shared[0]
    what = trainer_gpu.get("what", "trains")
    head = (
        f"eval_tasks.placement async needs a GPU the trainer does not use, and {evaluator_gpu['label']} "
        f"({evaluator_gpu['uuid']}) {what}"
        + (
            f" ({trainer_gpu['label']})"
            if trainer_gpu.get("label") and trainer_gpu["label"] != evaluator_gpu["label"]
            else ""
        )
        + "."
    )
    if single_gpu:
        return (
            head + "\nOn one GPU: placement: inline (training pauses for the evaluation), split the GPU with MIG and "
            "give the evaluator the second instance, or evaluate the saved checkpoints afterwards with "
            "`pgs eval watch <output_dir>` (or `pgs eval --model <checkpoint>`)."
        )
    return (
        head + "\nGive the evaluator GPUs no trainer rank or rollout engine uses (e.g. torchrun --nproc_per_node 6 and "
        "eval_tasks.devices: [6, 7] on an 8-GPU node), run `pgs eval serve` / `pgs eval watch` on another node with "
        "eval_tasks.service: auto, or use placement: inline."
    )


def child_environment(devices: list[int] | None, environ: dict | None = None) -> dict:
    """The environment of an evaluator process started by a trainer rank: torchrun's variables removed, the
    evaluator's GPUs (indices in this environment's view) as its CUDA_VISIBLE_DEVICES."""
    environ = dict(os.environ if environ is None else environ)
    visible = [physical_entry(d, environ) for d in devices or []]
    for name in TORCHRUN_VARIABLES:
        environ.pop(name, None)
    if devices is not None:
        environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    environ["OMP_NUM_THREADS"] = "4"  # its scorers and tokenizers leave the cores to the trainer's data loaders
    environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environ


def available_host_memory() -> int | None:
    """MemAvailable in bytes (Linux), None where it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def host_memory_error(nbytes: int, available: int | None = None) -> str:
    """ "" when a pinned staging area of `nbytes` fits in HOST_MEMORY_SHARE of the available host memory."""
    available = available_host_memory() if available is None else available
    if available is None or nbytes <= HOST_MEMORY_SHARE * available:
        return ""
    return (
        f"eval_tasks.transport push stages the weights in pinned host memory on rank 0: {nbytes / 2**30:.1f} GiB, "
        f"more than {HOST_MEMORY_SHARE:.0%} of the {available / 2**30:.1f} GiB available. Use eval_tasks.transport "
        "checkpoint (the evaluator reads the checkpoints train.save_steps writes; no host staging)."
    )
