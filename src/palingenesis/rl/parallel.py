"""The RL run's parallel layout: one process, or data-parallel ranks with FSDP2.

    torchrun --nproc_per_node 8 -m palingenesis.rl.trainer --config configs/rl_math.yaml

Every rank owns one GPU, a vLLM engine on it (colocated: it sleeps while the trainer
steps), and its shard of the prompts; the policy is sharded across ranks with FSDP2
(fp32 master weights, bf16 compute). A step is exact data parallelism:

  - rollouts, rewards and group baselines are rank-local (a group never spans ranks)
  - the loss normalizers (prompts, tokens, sequences) and the batch advantage std are
    global (all-reduced), so the gradient is the one a single process would compute
  - every rank runs the same number of micro-batches (short ranks add zero-weight ones):
    FSDP's all-gathers are collectives
  - after the optimizer step, full weights are gathered one parameter at a time and loaded
    into each rank's engine, on the trainer thread (collectives never run concurrently)

train.cpu_offload keeps parameters, gradients and optimizer state in CPU memory (FSDP2's
offload, also on a single GPU): the GPU holds one layer's weights at a time.
"""

import logging
import os
import socket
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Parallel:
    def __init__(self, fsdp: bool, cpu_offload: bool = False, reshard_after_forward: bool = True):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.fsdp = fsdp or cpu_offload or self.world > 1
        self.cpu_offload = cpu_offload
        self.reshard_after_forward = reshard_after_forward
        cuda = torch.cuda.is_available()
        self.device_type = "cuda" if cuda else "cpu"
        if cuda:
            torch.cuda.set_device(self.local_rank)
        if self.fsdp and not dist.is_initialized():
            if self.world == 1:  # FSDP2 on one GPU (CPU offload) still needs a process group
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", str(_free_port()))
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="nccl" if cuda else "gloo")
        self.mesh = None
        if self.fsdp:
            from torch.distributed.device_mesh import init_device_mesh

            self.mesh = init_device_mesh(self.device_type, (self.world,), mesh_dim_names=("dp",))

    @property
    def main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if self.device_type == "cuda" else "cpu"

    # ------------------------------------------------------------------ model

    def shard(self, model: nn.Module) -> nn.Module:
        """FSDP2 over the transformer layers, then the root (which keeps the embeddings,
        final norm and head gathered from forward to backward, so the loss can apply the head
        outside the model's forward)."""
        from palingenesis.config import ParallelConfig
        from palingenesis.distributed import apply_fsdp

        config = ParallelConfig(cpu_offload=self.cpu_offload, reshard_after_forward=self.reshard_after_forward)
        model = apply_fsdp(model, self.mesh, config, bf16=True)
        logger.info(
            "FSDP2: policy sharded over %d rank(s)%s", self.world, " with CPU offload" if self.cpu_offload else ""
        )
        return model

    def reshard(self, model: nn.Module) -> None:
        """Return every FSDP module to its sharded parameters. The root keeps its parameters
        gathered after a forward until the backward, so a forward without one (generation,
        evaluation) leaves them gathered; saving or gathering the weights needs the shards."""
        if not self.fsdp:
            return
        from torch.distributed.fsdp import FSDPModule

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def gradient_sync(self, model: nn.Module, enabled: bool) -> None:
        """Reduce-scatter gradients only on a step's last micro-batch."""
        if self.fsdp and hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(enabled)

    def full_parameters(self, model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
        """(name, full bf16 tensor) per parameter, gathered one at a time (every rank must
        iterate, in the same order: each gather is a collective)."""
        self.reshard(model)
        for name, param in model.named_parameters():
            tensor = param.detach()
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            yield name, tensor.to(device=self.device, dtype=torch.bfloat16)

    # ------------------------------------------------------------ collectives

    def sum(self, values: list[float]) -> list[float]:
        """Element-wise sum over ranks, in one collective."""
        if self.world == 1:
            return list(values)
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return t.tolist()

    def max(self, value: int) -> int:
        if self.world == 1:
            return value
        t = torch.tensor([value], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def gather(self, obj) -> list:
        """Every rank's `obj` (small Python objects: per-step statistics)."""
        if self.world == 1:
            return [obj]
        out: list = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def mean_stats(per_rank: list[dict[str, float]], weights: list[float] | None = None) -> dict[str, float]:
    """Statistics averaged over ranks (weighted, e.g. by trajectories); keys any rank has."""
    weights = weights or [1.0] * len(per_rank)
    keys = sorted({k for stats in per_rank for k in stats})
    out = {}
    for k in keys:
        pairs = [(s[k], w) for s, w in zip(per_rank, weights) if k in s]
        total = sum(w for _, w in pairs)
        out[k] = sum(v * w for v, w in pairs) / total if total else 0.0
    return out
