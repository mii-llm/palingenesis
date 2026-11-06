"""Off-policy knowledge distillation inside `pgs train`.

The student trains on the data pipeline's fixed sequences (gold data, or text a teacher wrote)
against the distributions of one or more teachers at every trained token; everything else (data
mixing, masking, packing, FSDP, activation checkpointing, SeCO, checkpointing, plugins) is the SFT
trainer's. On-policy distillation, where the student samples, is `pgs distill` (palingenesis.opd).

Modules:
  config      DistillConfig (the `distill:` section), TeacherSpec, validation
  losses      divergences (forward / reverse KL, generalized JSD, skew KL, TVD) over dense or top-k supports
  chunked     the loss over the output head a slice of rows at a time (no [tokens, vocab] logits)
  teachers    online teachers (frozen, replicated or FSDP-sharded) and the shared-vocabulary check
  cache       precomputed top-k targets: content-addressed, sharded, memory-mapped, resumable
  build       `pgs distill-cache`: build a cache with an HF or vLLM teacher
  views       teacher views: context distillation (privileged context, few-shot, system prompt)
  features    capturing module outputs for hidden-state matching
  distiller   per micro-batch targets and loss, used by the trainer

    pgs train --config distill_offpolicy.yaml
    pgs distill-cache --config distill_offpolicy.yaml --teacher big --out caches/big
"""

from palingenesis.distill.config import DIVERGENCES, DistillConfig, TeacherSpec, parse_teachers

__all__ = ["DIVERGENCES", "DistillConfig", "Distiller", "TeacherSpec", "TeacherTargets", "parse_teachers"]


def __getattr__(name):
    # the distiller pulls in torch and transformers; the config stays importable without them
    if name in ("Distiller", "TeacherTargets", "TeacherRows", "build_distiller"):
        from palingenesis.distill import distiller

        return getattr(distiller, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
