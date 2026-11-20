"""A cluster description turned into a layout for an RL run: what goes on each GPU, with what share of its
memory, and the command that starts it.

    plan(Cluster(nodes=1, gpus_per_node=1, gpu_memory_gb=80), Workload(params_b=0.8, lora_rank=16, engines=2))

The layout the trainer implements is colocated data parallelism: every GPU holds a trainer rank, the rollout
engine(s) of that rank and its share of the prompts, and episodes never leave their rank, so sessions talk through
in-process queues and no collective runs while rollouts are generated. The planner sizes that layout; it does not
invent others. Its memory figures are estimates from parameter counts (bytes per parameter below), to be checked
against `memory/peak_gb` and the engines' own logs on the first run.

  full weights   bf16 parameters 2 + fp32 master 4 + fp32 gradient 4 + AdamW moments 8 = 18 bytes per parameter on
                 one GPU; under FSDP the 16 bytes of master, gradient and moments are divided by the ranks.
  adapters       the frozen base in bf16, 2 bytes per parameter (on one GPU; fp32 shards under FSDP: 4 bytes
                 divided by the ranks, plus the 2 gathered), and the adapters, megabytes.
  an engine      its own bf16 copy, 2 bytes per parameter, plus its KV cache.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class Cluster:
    nodes: int = 1
    gpus_per_node: int = 1
    gpu_memory_gb: float = 80.0


@dataclass
class Workload:
    params_b: float = 0.8  # the policy's parameters, in billions
    lora_rank: int = 0  # 0 = full weights
    engines: int = 1  # rollout engines per rank: 1, plus one per role adapter and one for a frozen base
    activations_gb: float = 8.0  # the trainer's activations and logits for one micro-batch (train.micro_tokens)
    kv_gb_per_engine: float = 8.0  # the least KV cache an engine needs to be worth running
    batch_prompts: int = 32
    backend: str = "vllm"


@dataclass
class Layout:
    ranks: int
    mode: str  # "single" | "fsdp"
    trainer_gb: float
    engine_weights_gb: float
    gpu_memory_utilization: float  # per engine
    kv_gb_per_engine: float
    free_gb: float
    overrides: dict = field(default_factory=dict)
    command: str = ""
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def plan(cluster: Cluster, workload: Workload, config_path: str = "config.yaml") -> Layout:
    ranks = cluster.nodes * cluster.gpus_per_node
    gb = workload.params_b  # 1e9 parameters x 1 byte = 1 GB
    notes, errors = [], []
    if ranks < 1 or workload.engines < 1:
        raise ValueError("a cluster needs at least one GPU and a workload at least one engine")
    mode = "single" if ranks == 1 else "fsdp"
    if workload.lora_rank:
        trainer = 2 * gb if ranks == 1 else 4 * gb / ranks + 2 * gb
        trainer += 0.2  # adapters, their gradients and moments
    else:
        trainer = 18 * gb if ranks == 1 else 16 * gb / ranks + 2 * gb
    trainer += workload.activations_gb
    weights = 2 * gb
    room = cluster.gpu_memory_gb - trainer
    per_engine = room / workload.engines
    kv = per_engine - weights
    if kv < workload.kv_gb_per_engine:
        errors.append(
            f"each engine would get {per_engine:.1f} GB for {weights:.1f} GB of weights and {max(kv, 0):.1f} GB of "
            f"KV cache (wanted: {workload.kv_gb_per_engine:.1f} GB or more): "
            + (
                "train adapters instead of full weights (lora.rank), "
                if not workload.lora_rank
                else "use fewer engines (roles sharing weights), "
            )
            + "a GPU with more memory, or train.cpu_offload."
        )
    # an in-process engine sleeps while the trainer steps (rollout.sleep, max_staleness 0): it may use the
    # activations' share too; a server keeps its memory, so it gets only what is left
    sleeping = workload.backend == "vllm" and workload.engines == 1
    usable = per_engine + (workload.activations_gb if sleeping else 0.0)
    utilization = max(0.05, min(0.9, usable / cluster.gpu_memory_gb))
    overrides: dict = {"rollout.gpu_memory_utilization": round(utilization, 2)}
    if workload.batch_prompts % ranks:
        overrides["rollout.batch_prompts"] = math.ceil(workload.batch_prompts / ranks) * ranks
        notes.append(f"rollout.batch_prompts raised to a multiple of the {ranks} ranks.")
    if workload.engines > 1:
        overrides["rollout.backend"] = "vllm_server"
        notes.append("several engines per GPU: one vLLM server each (rollout.backend vllm_server).")
        if ranks > 1:
            errors.append(
                "roles with their own adapter (several engines) run on one GPU for now: with several GPUs share "
                "the weights between the roles (prompts tell them apart) or give each adapter its own run."
            )
    if ranks > 1:
        overrides["rollout.backend"] = overrides.get("rollout.backend", "vllm")
        notes.append(
            "one trainer rank and one engine per GPU; the policy is sharded with FSDP2 and each rank loads the "
            "gathered weights into its own engine after every step."
        )
        if not workload.lora_rank:
            notes.append(f"{16 * gb / ranks:.1f} GB of master weights, gradients and AdamW moments per GPU.")
    if cluster.nodes > 1:
        command = (
            f"torchrun --nnodes {cluster.nodes} --nproc_per_node {cluster.gpus_per_node} --rdzv_backend c10d "
            f"--rdzv_endpoint $HEAD_NODE:29400 -m palingenesis.rl.trainer --config {config_path}"
        )
    elif ranks > 1:
        command = f"torchrun --nproc_per_node {ranks} -m palingenesis.rl.trainer --config {config_path}"
    else:
        command = f"pgs rl --config {config_path}"
    command += "".join(f" --{key} {value}" for key, value in overrides.items())
    return Layout(
        ranks=ranks,
        mode=mode,
        trainer_gb=round(trainer, 1),
        engine_weights_gb=round(weights, 1),
        gpu_memory_utilization=overrides["rollout.gpu_memory_utilization"],
        kv_gb_per_engine=round(max(kv, 0.0), 1),
        free_gb=round(cluster.gpu_memory_gb - trainer - workload.engines * per_engine, 1),
        overrides=overrides,
        command=command,
        notes=notes,
        errors=errors,
    )


def workload_of(config, params_b: float) -> Workload:
    """The workload of an RLConfig whose policy has `params_b` billion parameters."""
    from palingenesis.rl.scenarios.config import BASE, adapter_names

    extra = len(adapter_names(config)) + any(role.partner == BASE for role in config.roles.values())
    return Workload(
        params_b=params_b,
        lora_rank=config.lora.rank,
        engines=1 + extra,
        batch_prompts=config.rollout.batch_prompts,
        backend=config.rollout.backend,
    )
