"""Multi-source evaluation: weighted scoring across multiple eval datasets.

Unifies best-model selection and MSFT source weight scheduling around
a single multi-dimensional evaluation infrastructure.

When eval_sources is configured, the system:
1. Evaluates on each source independently every eval_every steps
2. Computes a weighted composite score for best-model tracking
3. Provides per-source loss signals for MSFT weight adjustment
4. Warns when any source exceeds its regression floor (catastrophic forgetting)
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from palingenesis.loss import shift_labels

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass(slots=True)
class EvalSourceResult:
    """Result from evaluating one source."""

    name: str
    loss: float
    tokens: int
    regressed: bool = False  # True if loss > regression_floor


@dataclass(slots=True)
class MultiEvalResult:
    """Combined result from evaluating all sources."""

    score: float  # Weighted composite (lower = better)
    per_source: dict[str, float] = field(default_factory=dict)  # name → loss
    regressions: list[str] = field(default_factory=list)  # sources above floor
    tokens_total: int = 0


class MultiEvaluator:
    """Evaluates model on multiple datasets with weighted scoring.

    Each eval source has:
    - name: human-readable identifier
    - batches: pre-collected evaluation batches
    - weight: importance for composite score (0-1, sum to 1)
    - regression_floor: alarm threshold (optional)

    Usage:
        evaluator = MultiEvaluator(eval_sources_config, tokenizer, max_seq_len, device)
        result = evaluator.evaluate(model, dtype=torch.bfloat16)
        # result.score → for BestModelTracker
        # result.per_source → for MSFT weight adjustment
        # result.regressions → for warnings
    """

    def __init__(
        self,
        eval_sources: list[dict],
        tokenizer,
        max_seq_length: int,
        device: torch.device,
    ):
        self.device = device
        self.sources: list[dict] = []

        # Normalize weights to sum to 1
        total_weight = sum(s.get("weight", 1.0) for s in eval_sources)

        for src in eval_sources:
            name = src.get("name", src.get("dataset", "unknown"))
            weight = src.get("weight", 1.0) / total_weight
            samples = src.get("samples", 100)
            regression_floor = src.get("regression_floor", None)

            # Load and pre-collect eval batches
            batches = self._load_source(src, tokenizer, max_seq_length, samples)

            self.sources.append(
                {
                    "name": name,
                    "weight": weight,
                    "batches": batches,
                    "regression_floor": regression_floor,
                }
            )

        source_names = [s["name"] for s in self.sources]
        source_weights = [f"{s['weight']:.2f}" for s in self.sources]
        logger.info(f"MultiEvaluator: {len(self.sources)} sources: {list(zip(source_names, source_weights))}")

    def _load_source(self, src_config: dict, tokenizer, max_seq_length: int, max_samples: int) -> list[dict]:
        """Load and pre-collect eval batches for one source."""
        from palingenesis.data import ChatDataset, _collate_fn

        dataset_path = src_config.get("dataset", "")
        split = src_config.get("split", "test")

        try:
            from datasets import load_dataset

            raw = load_dataset(dataset_path, split=split, streaming=True)
        except Exception:
            # Try as local JSONL
            import json
            from pathlib import Path

            path = Path(dataset_path)
            if path.exists():

                class LocalDataset:
                    def __iter__(self_inner):
                        with path.open() as f:
                            for line in f:
                                if line.strip():
                                    yield json.loads(line)

                raw = LocalDataset()
            else:
                logger.warning(f"MultiEval: cannot load source '{dataset_path}', skipping")
                return []

        # mode mirrors the training `sources`: "sft" = chat-templated messages with
        # assistant-only loss; "pretrain" = raw text with all-token loss (no template).
        # Use "pretrain" for language-modeling eval (e.g. held-out Italian docs) so the
        # measured CE/ppl is true next-token LM, not ppl conditioned on a chat wrapper.
        mode = src_config.get("mode", "sft")
        if mode == "pretrain":
            from palingenesis.data import PretrainDataset

            ds = PretrainDataset(
                raw,
                tokenizer,
                max_seq_length,
                text_field=src_config.get("text_field", "text"),
                rank=0,
                world_size=1,
            )
        elif mode == "sft":
            ds = ChatDataset(
                raw,
                tokenizer,
                max_seq_length,
                src_config.get("messages_field", "messages"),
                rank=0,
                world_size=1,
                last_turn_only=src_config.get("last_turn_only", False),
            )
        else:
            logger.warning(f"MultiEval: unknown mode '{mode}' for source '{src_config.get('name', '?')}', skipping")
            return []

        batches = []
        count = 0
        try:
            for sample in ds:
                batches.append(sample)
                count += 1
                if count >= max_samples:
                    break
        except Exception as e:
            logger.warning(f"MultiEval: error loading source '{src_config.get('name', '?')}': {e}")
            return []

        if not batches:
            return []

        # Collate into mini-batches of 4
        pad_id = tokenizer.pad_token_id or 0
        collated = []
        for i in range(0, len(batches), 4):
            chunk = batches[i : i + 4]
            collated.append(_collate_fn(chunk, pad_id))

        return collated

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> MultiEvalResult:
        """Evaluate model on all sources, return weighted composite score.

        Args:
            model: The model to evaluate
            dtype: Compute dtype for autocast

        Returns:
            MultiEvalResult with composite score, per-source losses, and regressions
        """
        model.eval()

        per_source: dict[str, float] = {}
        regressions: list[str] = []
        tokens_total = 0
        weighted_sum = 0.0

        for src in self.sources:
            if not src["batches"]:
                continue

            source_loss = 0.0
            source_tokens = 0

            for batch in src["batches"]:
                batch_device = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                input_ids = batch_device["input_ids"]
                attention_mask = batch_device["attention_mask"]
                # Shift for next-token prediction: logits[t] predicts input_ids[t+1]
                labels = shift_labels(batch_device["labels"])

                valid = (labels != IGNORE_INDEX).sum().item()
                if valid == 0:
                    continue

                with torch.amp.autocast("cuda", dtype=dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                source_loss += loss.item()
                source_tokens += valid

            if source_tokens > 0:
                avg_loss = source_loss / source_tokens
                per_source[src["name"]] = avg_loss
                weighted_sum += avg_loss * src["weight"]
                tokens_total += source_tokens

                # Check regression floor
                if src["regression_floor"] is not None and avg_loss > src["regression_floor"]:
                    regressions.append(src["name"])
                    logger.warning(
                        f"⚠️  REGRESSION: eval source '{src['name']}' loss={avg_loss:.4f} "
                        f"exceeds floor={src['regression_floor']}"
                    )

        model.train()

        if regressions:
            logger.warning(f"Regression detected on: {regressions}. Consider reducing source weights or stopping.")

        return MultiEvalResult(
            score=weighted_sum,
            per_source=per_source,
            regressions=regressions,
            tokens_total=tokens_total,
        )
