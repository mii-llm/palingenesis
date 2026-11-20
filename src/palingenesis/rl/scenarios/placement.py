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
    replicate_base: bool = False  # train.replicate_base: with adapters, every rank holds the whole frozen base


@dataclass
class Layout:
    ranks: int
    mode: str  # "single" | "fsdp" | "replicated"
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
    replicated = workload.replicate_base and bool(workload.lora_rank) and ranks > 1
    mode = "single" if ranks == 1 else "replicated" if replicated else "fsdp"
    if workload.lora_rank:
        trainer = 2 * gb if ranks == 1 or replicated else 4 * gb / ranks + 2 * gb
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
            "one trainer rank and one engine per GPU; every rank holds the whole frozen base and folds its own "
            "engine's weights; the adapters' gradients are summed over the ranks once per step."
            if replicated
            else "one trainer rank and one engine per GPU; the policy is sharded with FSDP2 and each rank loads "
            "the gathered weights into its own engine after every step."
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
        replicate_base=config.train.replicate_base,
    )


# ------------------------------------------------------------ engines apart


def worker_flags(config) -> str:
    """The flags of `python -m palingenesis.rl.pool.worker` that must agree with an RLConfig: a trainer refuses a
    worker whose stop tokens, chat-template arguments, context length or head precision differ from its own."""
    import json
    import shlex

    m, r = config.model, config.rollout
    flags = f"--max-model-len {r.max_model_len} --ssm-state-dtype {r.ssm_state_dtype}"
    if r.fp32_head:
        flags += " --fp32-head"
    if m.stop_tokens:
        flags += f" --stop-tokens {shlex.quote(json.dumps(list(m.stop_tokens)))}"
    if m.chat_template_kwargs:
        flags += f" --chat-template-kwargs {shlex.quote(json.dumps(dict(m.chat_template_kwargs)))}"
    if r.max_num_seqs:
        flags += f" --max-num-seqs {r.max_num_seqs}"
    return flags


@dataclass
class PoolLayout:
    """Trainers and rollout engines on separate GPUs (rollout.backend remote, palingenesis.rl.pool)."""

    trainers: list[str]  # "host:gpu" of every trainer rank, in rank order
    engines: list[dict]  # {"address": "host:port", "gpus": ["host:gpu", ...], "command": the worker's launch line}
    mode: str  # the trainers' layout: "single" | "fsdp" | "replicated"
    push: str  # what a weight update sends: "full" | "fold" | "native"
    push_mb: float  # megabytes per update per engine (an estimate from the parameter count)
    push_seconds: float  # ... and the time until the last engine has it (the engines of other nodes share a link)
    trainer_gb: float
    engine_gb: float  # weights per engine GPU; the rest is its KV cache
    overrides: dict = field(default_factory=dict)
    command: str = ""
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


LOOPBACK_GBPS = 16.0  # a TCP stream between two processes of one host: about 2 GB/s


def plan_pool(
    cluster: Cluster,
    workload: Workload,
    engine_gpus: int,
    config_path: str = "config.yaml",
    hosts: list[str] | None = None,
    model: str = "<model>",
    engine_tp: int = 1,
    link_gbps: float = 25.0,
    base_port: int = 7000,
    adapters: int = 0,
    flags: str = "",
    base_partner: bool = False,
) -> PoolLayout:
    """A layout with `engine_gpus` of the cluster's GPUs running engine workers and the rest trainer ranks.

    GPU sets. When `engine_gpus` divides evenly over the nodes, every node gives its last GPUs to engines: each
    trainer rank then has engines on its own host, and weights cross a network only from rank 0's node to the
    others. Otherwise whole nodes become engine nodes (`engine_gpus` a multiple of a node's GPUs). `engine_tp`
    GPUs per engine (vLLM tensor parallelism) for a policy too large for one; engines are data-parallel replicas.

    Sync. Rank 0 stages each update in host memory and sends it to every engine while the trainers step; an
    engine applies it between two of its steps. `push_mb` is 2 bytes per parameter for full weights; with
    adapters, their factors: about 1% of the parameters at rank 16, in float32 (measure it: lora/trainable_params
    x 4 bytes, or pool/push_mb). `push_seconds` is the time until the last engine has an update: the engines
    of other nodes are all fed through rank 0's link (`link_gbps`), so their pushes share it; an engine on rank 0's
    node has a loopback stream of its own. The first update also carries the frozen base (as large as "full").

    `flags`: worker flags every launch line carries (`worker_flags(config)`: the settings a trainer checks when it
    connects). `adapters`: roles' own adapters; `base_partner`: a role plays the frozen base; either makes the
    engines serve named adapters ("native")."""
    hosts = hosts or [f"node{i}" for i in range(cluster.nodes)]
    total = cluster.nodes * cluster.gpus_per_node
    notes, errors = [], []
    if len(hosts) != cluster.nodes:
        raise ValueError(f"{len(hosts)} host names for {cluster.nodes} nodes")
    if not 0 < engine_gpus < total or engine_gpus % engine_tp:
        raise ValueError("engine_gpus must leave at least one trainer GPU and be a multiple of engine_tp")
    per_node = engine_gpus // cluster.nodes if engine_gpus % cluster.nodes == 0 else 0
    if per_node and per_node % engine_tp == 0:
        split = {host: cluster.gpus_per_node - per_node for host in hosts}  # trainer GPUs of each node
    elif engine_gpus % cluster.gpus_per_node == 0:
        engine_nodes = engine_gpus // cluster.gpus_per_node
        split = {
            host: (cluster.gpus_per_node if i < cluster.nodes - engine_nodes else 0) for i, host in enumerate(hosts)
        }
    else:
        raise ValueError(
            f"{engine_gpus} engine GPUs fit neither the same number on each of the {cluster.nodes} nodes nor whole "
            f"nodes of {cluster.gpus_per_node}"
        )
    trainers = [f"{host}:{gpu}" for host in hosts for gpu in range(split[host])]
    ranks = len(trainers)
    trainer_hosts = [host for host in hosts if split[host]]
    if len({split[host] for host in trainer_hosts}) > 1:
        errors.append("the trainer nodes have different numbers of trainer GPUs: torchrun needs the same on each.")
    replicated = workload.replicate_base and bool(workload.lora_rank) and ranks > 1
    mode = "single" if ranks == 1 else "replicated" if replicated else "fsdp"
    gb = workload.params_b
    if workload.lora_rank:
        trainer_gb = (2 * gb if ranks == 1 or replicated else 4 * gb / ranks + 2 * gb) + 0.2
    else:
        trainer_gb = 18 * gb if ranks == 1 else 16 * gb / ranks + 2 * gb
    trainer_gb += workload.activations_gb
    engine_gb = 2 * gb / engine_tp
    if trainer_gb > cluster.gpu_memory_gb:
        errors.append(f"a trainer rank needs about {trainer_gb:.0f} GB, a GPU has {cluster.gpu_memory_gb:.0f}.")
    if engine_gb + workload.kv_gb_per_engine > cluster.gpu_memory_gb:
        errors.append(f"an engine GPU holds {engine_gb:.0f} GB of weights: raise engine_tp or use larger GPUs.")
    if workload.lora_rank and mode != "fsdp":
        push = "native" if adapters or base_partner else "fold"
        push_mb = 40.0 * gb * workload.lora_rank / 16 * max(adapters, 1)
    else:
        push, push_mb = "full", 2 * gb * 1024
        if workload.lora_rank:
            notes.append(
                "adapters under FSDP are sent folded into the full weights: train.replicate_base sends factors."
            )
    engines = []
    for host in hosts:
        gpus = list(range(split[host], cluster.gpus_per_node))
        for k in range(0, len(gpus), engine_tp):
            group = gpus[k : k + engine_tp]
            port = base_port + len([e for e in engines if e["address"].startswith(host + ":")])
            line = f"--model {model} --host 0.0.0.0 --port {port}" + (f" {flags}" if flags else "")
            if engine_tp > 1:
                line += f""" --vllm-args '{{"tensor_parallel_size": {engine_tp}}}'"""
            if push == "native":
                line += f" --lora-rank {workload.lora_rank} --max-loras {2 * (max(adapters, 1) + 1)}"
            engines.append(
                {
                    "address": f"{host}:{port}",
                    "gpus": [f"{host}:{gpu}" for gpu in group],
                    "command": f"CUDA_VISIBLE_DEVICES={','.join(map(str, group))} python -m palingenesis.rl.pool.worker {line}",
                }
            )
    if len(engines) < ranks:
        errors.append(f"{len(engines)} engines for {ranks} trainer ranks: every rank needs at least one engine.")
    remote = sum(engine["address"].split(":")[0] != hosts[0] for engine in engines)  # fed through rank 0's link
    push_seconds = max(push_mb / (LOOPBACK_GBPS * 125.0), remote * push_mb / (link_gbps * 125.0))
    batch = workload.batch_prompts
    overrides: dict = {"rollout.backend": "remote", "rollout.engines": [e["address"] for e in engines]}
    if batch % ranks:
        overrides["rollout.batch_prompts"] = math.ceil(batch / ranks) * ranks
    notes.append(
        f"rank r generates with engines[r::{ranks}]; rank 0 sends every update to all {len(engines)} engines "
        f"({push}: about {push_mb:.0f} MB each; {push_seconds:.2f} s until the last has it, {remote} of them over "
        f"rank 0's {link_gbps:g} Gbit link), without waiting for them."
    )
    notes.append(
        "rollout.max_staleness 1 or more: with 0 every update waits for the engines. Keep the update's travel time "
        "under a training step, or the engines fall behind (pool/lag) and rows are dropped as stale."
    )
    visible = ",".join(map(str, range(split[trainer_hosts[0]])))
    prefix = f"CUDA_VISIBLE_DEVICES={visible} "
    if len(trainer_hosts) > 1:
        command = (
            f"{prefix}torchrun --nnodes {len(trainer_hosts)} --nproc_per_node {split[trainer_hosts[0]]} "
            f"--rdzv_backend c10d --rdzv_endpoint {trainer_hosts[0]}:29400 -m palingenesis.rl.trainer --config {config_path}"
        )
    elif ranks > 1:
        command = f"{prefix}torchrun --nproc_per_node {ranks} -m palingenesis.rl.trainer --config {config_path}"
    else:
        command = f"{prefix}pgs rl --config {config_path}"
    return PoolLayout(
        trainers=trainers,
        engines=engines,
        mode=mode,
        push=push,
        push_mb=round(push_mb, 1),
        push_seconds=round(push_seconds, 3),
        trainer_gb=round(trainer_gb, 1),
        engine_gb=round(engine_gb, 1),
        overrides=overrides,
        command=command,
        notes=notes,
        errors=errors,
    )
