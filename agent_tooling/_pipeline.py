"""The sample stream the trainer builds from a config, for diagnostics.

Diagnostics must look at exactly what training consumes, so this goes through
the trainer's own functions (`_load_dataset_source`, `build_dataset`,
`PreferenceDataset`) with the same arguments `palingenesis.train` passes:
local files, multi-source mixes, prepared datasets, masking options and
packing all behave as in training.
"""

from __future__ import annotations

from collections.abc import Iterator

import agent_tooling._path_setup  # noqa: F401
from palingenesis.config import Config


def load_tokenizer(config: Config):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, trust_remote_code=config.model.trust_remote_code, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def dataset_id(config: Config) -> str:
    """The data the trainer reads: the prepared output when preprocess is enabled."""
    if config.preprocess.enabled:
        from palingenesis.prepare import find_prepared_dataset

        prepared = find_prepared_dataset(config.preprocess.output_dir)
        if prepared is None:
            raise FileNotFoundError(
                f"preprocess.enabled=true but no prepared dataset in '{config.preprocess.output_dir}'. "
                "Run: pgs prepare --config <this config>"
            )
        return str(prepared)
    return config.data.dataset


def training_samples(config: Config, tokenizer) -> Iterator[dict]:
    """Per-sequence dicts (input_ids, attention_mask, labels[, position_ids]) as
    the trainer's DataLoader receives them. With DPO enabled, each preference
    pair yields its chosen and rejected sequences, marked with `side`."""
    from palingenesis.data import _load_dataset_source, build_dataset

    batch_size = config.train.per_device_batch_size
    if config.dpo.enabled:
        from palingenesis.dpo import PreferenceDataset

        d = config.dpo
        pairs = PreferenceDataset(
            _load_dataset_source(dataset_id(config), config.data.dataset_split, config.data.streaming),
            tokenizer,
            config.data.max_seq_length,
            prompt_field=d.prompt_field,
            chosen_field=d.chosen_field,
            rejected_field=d.rejected_field,
            last_turn_only=config.data.last_turn_only,
            train_on_reasoning=config.data.train_on_reasoning,
            truncate_rejected=d.truncate_rejected,
        )
        for pair in pairs:
            yield {**pair["chosen"], "side": "chosen"}
            yield {**pair["rejected"], "side": "rejected"}
        return
    if config.data.sources:
        yield from build_dataset(config.data, tokenizer, config.data, 0, 1, batch_size)
        return
    raw = _load_dataset_source(dataset_id(config), config.data.dataset_split, config.data.streaming)
    yield from build_dataset(raw, tokenizer, config.data, 0, 1, batch_size)
