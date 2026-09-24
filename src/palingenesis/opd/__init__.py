"""On-policy distillation (OPD) — the student samples, the teacher scores.

Unlike offline distillation (teacher generates a dataset, student does SFT on
it), OPD trains the student on its *own* completions: every step the student
samples with its current weights and the loss pulls its distribution toward the
teacher's on exactly those tokens. The student is corrected where *it* goes,
not where the teacher would have gone.

Modules:
  config        OPDConfig: model, teachers, sources, rollout, loss, train, logging
  sources       prompt sources (chat "messages", "mcqa" pools) and their dev metrics
  rollout       student samplers: HF generate, in-process vLLM, vLLM server
  teachers      teacher scorers: HF (full distribution) and vLLM (top-k, prefill-only)
  align         student <-> teacher token alignment: shared vocabulary, or byte chunks
                across different tokenizers
  losses        full_rkl, topk_kl, sampled_rkl, xtok — sliced over the vocabulary
  fused_rkl     full_rkl for plain linear heads: fused Triton passes, analytic gradient
  orchestrator  the rollout pipeline and its background thread (bounded staleness)
  trainer       the training loop
  token_bridge  the shared-vocabulary bridge (end-of-turn remapping, compatibility check)

Entry points:
    pgs distill       --config configs/distill_math.yaml   # train
    pgs distill-score --config configs/distill_opd.yaml --out scored.jsonl  # annotate an mcqa pool
"""

from palingenesis.opd.config import OPDConfig, OPDConfigError
from palingenesis.opd.sources import ChatMessagesSource, McqaPoolSource, MixedSource, PromptSource
from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible

__all__ = [
    "ChatMessagesSource",
    "McqaPoolSource",
    "MixedSource",
    "OPDConfig",
    "OPDConfigError",
    "OPDTrainer",
    "PromptSource",
    "TokenBridge",
    "TokenBridgeError",
    "check_compatible",
]


def __getattr__(name):
    # OPDTrainer pulls in torch/transformers; keep config/sources/bridge importable
    # in torch-free contexts (data prep, tests, tooling).
    if name == "OPDTrainer":
        from palingenesis.opd.trainer import OPDTrainer

        return OPDTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
