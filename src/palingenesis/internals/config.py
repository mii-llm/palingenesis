"""Configuration of `pgs internals run`: capture, per-layer probe sweeps and checkpoint comparisons in one job.

    models:                                  # name -> checkpoint (HF id or directory)
      base: Qwen/Qwen3.5-0.8B
      sft: runs/sft/final
    train: data/known_en.jsonl               # rows: messages, label (0/1 or a probability), group
    tests:                                   # other datasets / languages: transfer of the train-split probe
      popqa: data/known_popqa.jsonl
      it: data/known_it.jsonl
    compare: [[base, sft]]                   # CKA + probe transfer per layer, A -> B
    position: last_prompt
    layers: all
    output_dir: runs/internals/known

The file may nest everything under `internals:`; `--set key=value` overrides (values parsed as YAML).
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from palingenesis.internals.capture import DTYPES
from palingenesis.internals.positions import POSITIONS
from palingenesis.internals.probes import KINDS

MODEL_DTYPES = ("bfloat16", "float16", "float32")


class InternalsConfigError(ValueError):
    pass


@dataclass
class InternalsConfig:
    models: dict = field(default_factory=dict)  # name -> path
    train: str = ""  # JSONL of labelled rows: the probe's train / val / test splits (grouped)
    tests: dict = field(default_factory=dict)  # name -> JSONL: extra test sets (transfer)
    output_dir: str = ""
    compare: list = field(default_factory=list)  # [[a, b], ...] names in `models`
    position: str = "last_prompt"
    layers: str | list | int = "all"
    probe_kind: str = "logistic"  # logistic | diffmean | lda
    val: float = 0.1  # fraction of the train file's groups for validation (layer and strength choice, calibration)
    test: float = 0.2  # fraction held out for the in-distribution test
    bootstrap: int = 1000  # cluster-bootstrap resamples for AUROC intervals (0: none)
    label_threshold: float = 0.5  # labels >= this are positive (soft labels stay soft for training)
    control: bool = True  # group-level control task per layer
    chat_template_kwargs: dict = field(default_factory=dict)
    store_dtype: str = "bfloat16"  # activations on disk
    model_dtype: str | None = None  # default bf16 on CUDA, fp32 on CPU
    batch_size: int = 16
    max_batch_tokens: int = 32768
    max_tokens: int | None = None  # rows longer than this are dropped (counted)
    limit: int | None = None  # use the first N rows of every file
    device: str | None = None
    seed: int = 0
    overwrite: bool = False  # recapture stores that already exist (otherwise a matching store is reused)

    @classmethod
    def from_dict(cls, raw: dict, where: str = "config") -> InternalsConfig:
        if not isinstance(raw, dict):
            raise InternalsConfigError(f"{where}: expected a mapping")
        if set(raw) == {"internals"}:
            raw = raw["internals"] or {}
        config = cls()
        for key, value in raw.items():
            config.set(key, value, where)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Sequence[str] = ()) -> InternalsConfig:
        import yaml

        with Path(path).open() as f:
            config = cls.from_dict(yaml.safe_load(f) or {}, str(path))
        config.apply_overrides(overrides)
        return config

    def apply_overrides(self, overrides: Sequence[str]) -> None:
        import yaml

        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise InternalsConfigError(f"--set {item!r}: expected KEY=VALUE")
            self.set(key.strip(), yaml.safe_load(value), "command line")

    def set(self, key: str, value, where: str = "config") -> None:
        parts = key.split(".")
        if parts[0] in ("models", "tests") and len(parts) == 2:
            getattr(self, parts[0])[parts[1]] = value
            return
        if len(parts) != 1:
            raise InternalsConfigError(f"{where}: unknown key {key!r}")
        names = [f.name for f in dataclasses.fields(self)]
        if key not in names:
            hint = difflib.get_close_matches(key, names, n=1)
            raise InternalsConfigError(
                f"{where}: unknown option {key!r}" + (f" (did you mean {hint[0]!r}?)" if hint else "")
            )
        if isinstance(getattr(self, key), float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        setattr(self, key, value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def validate(self) -> list[str]:
        """Errors (empty list = valid)."""
        e: list[str] = []
        if not self.models or not isinstance(self.models, dict):
            e.append("models: at least one name -> checkpoint")
        if not self.train:
            e.append("train is required (a JSONL of labelled rows)")
        if not self.output_dir:
            e.append("output_dir is required")
        if not isinstance(self.tests, dict):
            e.append("tests must be a mapping name -> JSONL")
        if self.position not in POSITIONS:
            e.append(f"position must be one of {', '.join(POSITIONS)}, not {self.position!r}")
        elif self.position.endswith("_all"):
            e.append("position *_all gives one row per token: probes need one row per example (use *_mean or *_last)")
        if self.probe_kind not in KINDS:
            e.append(f"probe_kind must be one of {KINDS}, not {self.probe_kind!r}")
        if not (0 < self.val < 1 and 0 < self.test < 1 and self.val + self.test < 1):
            e.append("val and test must be fractions in (0, 1) with val + test < 1")
        if not isinstance(self.bootstrap, int) or self.bootstrap < 0:
            e.append("bootstrap must be an integer >= 0")
        if not (isinstance(self.label_threshold, (int, float)) and math.isfinite(self.label_threshold)):
            e.append("label_threshold must be a number")
        if self.store_dtype not in DTYPES:
            e.append(f"store_dtype must be one of {sorted(DTYPES)}, not {self.store_dtype!r}")
        if self.model_dtype is not None and self.model_dtype not in MODEL_DTYPES:
            e.append(f"model_dtype must be one of {MODEL_DTYPES}, not {self.model_dtype!r}")
        for k in ("batch_size", "max_batch_tokens"):
            v = getattr(self, k)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                e.append(f"{k} must be an integer >= 1")
        for k in ("max_tokens", "limit"):
            v = getattr(self, k)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                e.append(f"{k} must be an integer >= 1 (or null)")
        if not isinstance(self.compare, list):
            e.append("compare must be a list of [a, b] pairs")
        else:
            for pair in self.compare:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    e.append(f"compare: {pair!r} is not an [a, b] pair")
                elif any(name not in (self.models or {}) for name in pair):
                    e.append(f"compare: {pair!r} names a model not in models")
        names = list(self.tests or {}) + ["train"]
        if len(set(names)) != len(names):
            e.append("tests: a test set cannot be called 'train'")
        return e
