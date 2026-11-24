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

from palingenesis.config import ConfigError
from palingenesis.logits import scored_ce_sum
from palingenesis.loss import shift_labels
from palingenesis.multimodal import forward_kwargs

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class _JsonlRows:
    """Rows of a local JSONL file, read again on every iteration."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        import json

        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


class _ParquetRows:
    """Rows of a local parquet file, read batch by batch (pyarrow, imported when read)."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(self.path).iter_batches(batch_size=256):
            yield from batch.to_pylist()


def open_eval_rows(dataset: str, split: str = "test", name: str = ""):
    """The rows of an eval source: a local .jsonl, .json (a list of rows) or .parquet file, a
    directory written by `pgs prepare`, or a Hub dataset (its `split`). Raises ConfigError
    when there is nothing to read there."""
    from pathlib import Path

    label = f"source '{name or dataset}'"
    if not dataset:
        raise ConfigError(f"data.eval_sources: {label} has no `dataset`.")
    path = Path(dataset)
    if path.is_file():
        if path.suffix == ".parquet":
            return _ParquetRows(path)
        if path.suffix == ".jsonl":
            return _JsonlRows(path)
        if path.suffix == ".json":
            import json

            rows = json.loads(path.read_text())
            if not isinstance(rows, list):
                raise ConfigError(f"data.eval_sources: {label}: {path} must hold a list of rows.")
            return rows
        raise ConfigError(
            f"data.eval_sources: {label}: {path} is not a .jsonl, .json or .parquet file (got {path.suffix!r})."
        )
    try:
        from palingenesis.data import _load_dataset_source

        return _load_dataset_source(dataset, split, streaming=True)
    except Exception as e:
        raise ConfigError(
            f"data.eval_sources: {label}: cannot load {dataset!r} (split {split!r}) as a local file or a Hub "
            f"dataset: {type(e).__name__}: {e}"
        ) from e


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
        """Load and pre-collect eval batches for one source.

        A source that cannot be read, has an unknown mode or gives no usable sample is a
        ConfigError: an evaluation set that was configured and silently left out would let a
        run report a composite that is not the one asked for."""
        from palingenesis.data import ChatDataset, _collate_fn

        name = src_config.get("name", src_config.get("dataset", "?"))
        raw = open_eval_rows(src_config.get("dataset", ""), src_config.get("split", "test"), name)

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
                append_eos=src_config.get("append_eos", True),
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
                tools_field=src_config.get("tools_field", "tools"),
            )
        else:
            raise ConfigError(f"data.eval_sources: source '{name}' has mode {mode!r}; use sft or pretrain.")

        batches = []
        try:
            for sample in ds:
                batches.append(sample)
                if len(batches) >= max_samples:
                    break
        except Exception as e:
            raise ConfigError(
                f"data.eval_sources: source '{name}' ({src_config.get('dataset', '')}) could not be read: "
                f"{type(e).__name__}: {e}"
            ) from e

        if not batches:
            field_name = "text_field" if mode == "pretrain" else "messages_field"
            raise ConfigError(
                f"data.eval_sources: source '{name}' ({src_config.get('dataset', '')}) gives no usable sample "
                f"in mode {mode}. Check its {field_name}, mode and split."
            )

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

                # Logits only at scored positions, in slices: the full [B, S, V]
                # logits of model(...) dominated peak memory (palingenesis.logits).
                loss, valid = scored_ce_sum(
                    model, input_ids, attention_mask, labels, dtype, forward_kwargs=forward_kwargs(batch_device, {})
                )
                source_loss += loss
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
