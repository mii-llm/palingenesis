"""How the processes of a run split into parallel dimensions, and the rules that make a split valid.

Pure Python (no torch import): configuration validation and the memory estimator use it.

A run of N processes is a grid pp x dp x cp x tp, outermost first:

  * tp (tensor parallel): the innermost dimension, so a group is a run of consecutive ranks, the GPUs of
    one node (its all-gathers / reduce-scatters run every layer and need NVLink bandwidth);
  * cp (context parallel): splits each sequence over its group (K / V all-gather per attention layer);
  * dp (data parallel): different rows; FSDP2 shards parameters over dp x cp;
  * pp (pipeline parallel): the outermost dimension, its point-to-point activations cross nodes best.

This is Megatron-Core's and torchtitan's order (parallel_state.py: tp-cp-ep-dp-pp innermost first; torchtitan's
ParallelDims: pp, dp_replicate, dp_shard, cp, tp).
"""

from dataclasses import dataclass

# Pipeline schedules: name -> (torch.distributed.pipelining class name, stages per rank: default, minimum, maximum).
PP_SCHEDULES: dict[str, tuple[str, int, int, int | None]] = {
    "gpipe": ("ScheduleGPipe", 1, 1, 1),
    "1f1b": ("Schedule1F1B", 1, 1, 1),
    "interleaved_1f1b": ("ScheduleInterleaved1F1B", 2, 2, None),
    "looped_bfs": ("ScheduleLoopedBFS", 2, 2, None),
    "interleaved_zero_bubble": ("ScheduleInterleavedZeroBubble", 2, 2, None),
    "zbv": ("ScheduleZBVZeroBubble", 2, 2, 2),
}


def schedule_errors(schedule: str, stages_per_rank: int) -> list[str]:
    """parallel.pp_stages_per_rank against the schedule (0: its default)."""
    _, _, low, high = PP_SCHEDULES[schedule]
    if stages_per_rank == 0 or low <= stages_per_rank and (high is None or stages_per_rank <= high):
        return []
    allowed = str(low) if high == low else f">= {low}" if high is None else f"{low}..{high}"
    return [
        f"parallel.pp_schedule={schedule} runs {allowed} stage(s) per rank; parallel.pp_stages_per_rank is "
        f"{stages_per_rank}."
    ]


def stages_per_rank(schedule: str, requested: int) -> int:
    return requested or PP_SCHEDULES[schedule][1]


@dataclass(frozen=True)
class ParallelDims:
    """The sizes of the grid's dimensions for `world_size` processes."""

    world_size: int
    tp: int = 1
    pp: int = 1
    cp: int = 1

    @property
    def dp(self) -> int:
        return self.world_size // (self.tp * self.pp * self.cp)

    def errors(self) -> list[str]:
        product = self.tp * self.pp * self.cp
        if self.world_size % product:
            return [
                f"{self.world_size} processes cannot split into tp_degree={self.tp} x pp_degree={self.pp} x "
                f"cp_degree={self.cp} (= {product}) groups: the product must divide the number of processes (the "
                "rest is data parallel)."
            ]
        return []

    def describe(self) -> str:
        parts = [
            f"{name}={size}" for name, size in (("pp", self.pp), ("dp", self.dp), ("cp", self.cp), ("tp", self.tp))
        ]
        return " x ".join(parts)


def head_errors(tp: int, heads: dict[str, int]) -> list[str]:
    """Tensor parallelism splits heads: every count in `heads` (name -> number) must be a multiple of tp."""
    return [
        f"parallel.tp_degree={tp} must divide the model's {name} ({count}): each rank holds a whole number of them."
        for name, count in heads.items()
        if count and count % tp
    ]


def split_layers(num_layers: int, num_stages: int, first_less: int = 1, last_less: int = 1) -> list[list[int]]:
    """The layers of each pipeline stage: as even as possible when the embeddings count as `first_less` layers
    on the first stage and the final norm + output head as `last_less` on the last (torchtitan's
    generate_llm_fqn_per_model_part weighting). Every stage gets at least one layer."""
    if num_stages < 1:
        raise ValueError("a pipeline needs at least one stage")
    if num_stages > num_layers:
        raise ValueError(
            f"{num_stages} pipeline stages for {num_layers} layers: every stage needs at least one layer "
            "(lower parallel.pp_degree or parallel.pp_stages_per_rank)."
        )
    total = num_layers + first_less + last_less
    # greedy cut points over the cumulative weight, offset by the embeddings' weight
    bounds = [0]
    for stage in range(1, num_stages):
        target = total * stage / num_stages - first_less
        cut = max(bounds[-1] + 1, min(num_layers - (num_stages - stage), round(target)))
        bounds.append(cut)
    bounds.append(num_layers)
    return [list(range(bounds[i], bounds[i + 1])) for i in range(num_stages)]


def stage_indices(pp_rank: int, pp_size: int, per_rank: int, v_shape: bool) -> list[int]:
    """The stages pipeline rank `pp_rank` runs: looped (rank, rank + pp, ...) or V-shaped (rank and its mirror)."""
    if v_shape:
        return [pp_rank, 2 * pp_size - 1 - pp_rank]
    return [pp_rank + s * pp_size for s in range(per_rank)]
