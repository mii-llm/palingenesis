"""On-policy distillation (OPD) — the student samples, the teacher scores.

Unlike offline distillation (teacher generates a dataset, student does SFT on
it), OPD trains the student on its *own* completions: every optimizer step the
student samples with its current weights and the loss is the reverse KL to the
teacher's distribution over exactly those tokens. The student is corrected
where *it* goes, not where the teacher would have gone — no train/inference
mismatch, no importance sampling.

Works across a student/teacher pair with *different chat templates* as long as
they share a base vocabulary: prompts are rendered per-model with each model's
own template, and only completion tokens are aligned (see `token_bridge`).

Current scope: the engine (token_bridge, on-policy sampling, reverse-KL loss)
is task-agnostic; the data layer (pool, formatting, dev metric) targets
multiple-choice QA pools. Generic prompt sources are the planned next step.

Entry points:
    pgs distill       --config configs/distill_opd.yaml   # train
    pgs distill-score --config configs/distill_opd.yaml --out scored.jsonl  # annotate pool with teacher answers
    python -m palingenesis.opd.trainer ... / python -m palingenesis.opd.score_pool ...
"""

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible

__all__ = [
    "OPDConfig",
    "OPDTrainer",
    "TokenBridge",
    "TokenBridgeError",
    "check_compatible",
]


def __getattr__(name):
    # OPDTrainer pulls in torch/transformers; keep pool/config/bridge importable
    # in torch-free contexts (data prep, tests, tooling).
    if name == "OPDTrainer":
        from palingenesis.opd.trainer import OPDTrainer

        return OPDTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
