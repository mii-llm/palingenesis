"""Weight-space merging of checkpoints that share an architecture: linear combinations (interpolation, partial
interpolation back, soups), task arithmetic, and materializing partial checkpoints.

    linear(models, weights, out)                 w = sum_i a_i W_i                 (weights sum to 1 unless unnormalized)
    interpolate(start, end, lam, out)            w = start + lam (end - start)     (lam .3-.5: pull an overshooting
                                                                                    SFT / RL endpoint back toward its start)
    soup(models, out)                            w = mean_i W_i
    task_arithmetic(base, models, weights, out)  w = base + sum_i l_i (W_i - base) (add what each model learned on top of a
                                                                                    shared base; [1, l] = A + l (B - base))
    materialize(checkpoint, template, out)       a partial checkpoint (e.g. a text-only training save) completed with the
                                                 template's other tensors and files, so it loads like the template

Tensors are matched by name across differently sharded checkpoints, never by shard. Math runs in float32, one tensor at a
time (memory: one output shard plus one tensor per source), and the result is saved in the template's dtype (or `dtype`).
The output mirrors the template: its tensor set, shard layout and non-weight files (config, tokenizer, chat template,
processor configs). Shapes must match; non-finite results are an error. A tensor that some source lacks is an error by
default; missing="template" copies the template's tensor unchanged (and reports how many). `merge_config.json` in the
output records the method, sources, coefficients and counts. Sources are local directories or Hugging Face model ids.

    pgs merge interpolate <start> <end> --lam 0.3 -o out/
    pgs merge linear A:0.7 B:0.3 -o out/
    pgs merge soup A B C -o out/
    pgs merge task-arithmetic --base Qwen/Qwen3.5-0.8B agent:1 expert:0.3 -o out/
    pgs merge materialize runs/x/step-200/model --template runs/x/final -o out/
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
# non-weight files copied from the template (training-only files are left out)
COPY_SUFFIXES = {".json", ".jinja", ".txt", ".model", ".tiktoken", ".py"}
SKIP_FILES = {INDEX, "merge_config.json", "training_meta.json", "rl_config.json", "tracker_run_id.json"}
DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
          "fp32": torch.float32, "float32": torch.float32}


class MergeError(ValueError):
    pass


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path), allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model",
                                                             "*.tiktoken"]))


class Checkpoint:
    """A safetensors checkpoint: tensor name -> shard, read lazily one tensor at a time."""

    def __init__(self, path: str | Path):
        self.source = str(path)
        self.dir = _resolve(path)
        if (self.dir / INDEX).exists():
            self.weight_map: dict[str, str] = json.loads((self.dir / INDEX).read_text())["weight_map"]
        elif (self.dir / SINGLE).exists():
            with safe_open(self.dir / SINGLE, "pt") as f:
                self.weight_map = {k: SINGLE for k in f.keys()}
        else:
            shards = sorted(self.dir.glob("*.safetensors"))
            if not shards:
                raise MergeError(f"{path}: no safetensors weights found")
            self.weight_map = {}
            for s in shards:
                with safe_open(s, "pt") as f:
                    self.weight_map.update({k: s.name for k in f.keys()})
        self._open: dict[str, object] = {}

    def keys(self) -> list[str]:
        return list(self.weight_map)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def _handle(self, shard: str):
        if shard not in self._open:
            self._open[shard] = safe_open(self.dir / shard, "pt")
        return self._open[shard]

    def shape(self, name: str) -> list[int]:
        return list(self._handle(self.weight_map[name]).get_slice(name).get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)

    def shards(self) -> dict[str, list[str]]:
        """Shard file -> tensor names, in a stable order."""
        out: dict[str, list[str]] = {}
        for k, s in self.weight_map.items():
            out.setdefault(s, []).append(k)
        return out


@dataclass
class MergeReport:
    out: str
    method: str
    sources: list[dict]
    base: str | None
    template: str
    tensors: int
    from_template: int = 0
    dtype: str | None = None
    missing: str = "error"
    extra_dropped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        extra = f", {self.from_template} copied from the template" if self.from_template else ""
        return f"{self.method}: {self.tensors} tensors{extra} -> {self.out}"


def _parse_dtype(dtype) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if str(dtype) not in DTYPES:
        raise MergeError(f"unknown dtype {dtype!r}; one of {sorted(set(DTYPES))}")
    return DTYPES[str(dtype)]


def _prepare_out(out: str | Path, overwrite: bool) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise MergeError(f"{out} exists and is not empty (overwrite=True to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _copy_files(template: Checkpoint, out: Path) -> None:
    for f in template.dir.iterdir():
        if f.is_file() and f.suffix in COPY_SUFFIXES and f.name not in SKIP_FILES:
            shutil.copy(f, out / f.name)


def _combine(method: str, terms: list[tuple[Checkpoint, float]], out: str | Path, *, base: Checkpoint | None,
             template: Checkpoint, missing: str, dtype, overwrite: bool, extra: str) -> MergeReport:
    """out = base + sum_i c_i (W_i - base) when base is given, else sum_i c_i W_i; tensor set and layout = template's."""
    if missing not in ("error", "template"):
        raise MergeError(f"missing must be 'error' or 'template', not {missing!r}")
    if extra not in ("error", "drop"):
        raise MergeError(f"extra must be 'error' or 'drop', not {extra!r}")
    out_dtype = _parse_dtype(dtype)
    names = set(template.keys())
    others = set().union(*(set(c.keys()) for c, _ in terms)) | (set(base.keys()) if base is not None else set())
    extra_names = sorted(others - names)
    if extra_names and extra == "error":
        raise MergeError(f"{len(extra_names)} tensors are not in the template (e.g. {extra_names[:3]}); "
                         "extra='drop' to leave them out")
    out = _prepare_out(out, overwrite)
    from_template, written = 0, 0
    total_size = 0
    index: dict[str, str] = {}
    for shard, shard_names in template.shards().items():
        tensors: dict[str, torch.Tensor] = {}
        for name in shard_names:
            ref_shape = template.shape(name)
            holders = [c for c, _ in terms if name in c] + ([base] if base is not None and name in base else [])
            complete = all(name in c for c, _ in terms) and (base is None or name in base)
            if not complete:
                if missing == "error":
                    lacking = [c.source for c, _ in terms if name not in c] + (
                        [base.source] if base is not None and name not in base else [])
                    raise MergeError(f"tensor {name!r} is missing from {lacking} (missing='template' copies the template's)")
                t = template.tensor(name)
                from_template += 1
            else:
                for c in holders:
                    if c.shape(name) != ref_shape:
                        raise MergeError(f"tensor {name!r}: shape {c.shape(name)} in {c.source} vs {ref_shape} in the template")
                ref = template.tensor(name)
                if not ref.is_floating_point():  # integer buffers are copied, and must agree across the sources
                    for c in holders:
                        if not torch.equal(c.tensor(name), ref):
                            raise MergeError(f"non-float tensor {name!r} differs between {c.source} and the template")
                    tensors[name] = ref.contiguous()
                    index[name] = shard
                    total_size += ref.numel() * ref.element_size()
                    written += 1
                    continue
                acc = base.tensor(name).float() if base is not None else torch.zeros(ref.shape, dtype=torch.float32)
                b = acc.clone() if base is not None else None
                for c, coef in terms:
                    w = c.tensor(name).float()
                    acc += coef * (w - b) if b is not None else coef * w
                if not torch.isfinite(acc).all():
                    raise MergeError(f"tensor {name!r}: non-finite values after merging")
                t = acc.to(out_dtype or ref.dtype)
            if out_dtype is not None and t.dtype != out_dtype and t.is_floating_point():  # template copies too
                t = t.to(out_dtype)
            tensors[name] = t.contiguous()
            index[name] = shard
            total_size += t.numel() * t.element_size()
            written += 1
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        del tensors
    if (template.dir / INDEX).exists() or len(template.shards()) > 1:
        (out / INDEX).write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": index}, indent=2))
    _copy_files(template, out)
    report = MergeReport(str(out), method, [{"model": c.source, "coef": coef} for c, coef in terms],
                         base.source if base is not None else None, template.source, written, from_template,
                         str(out_dtype).replace("torch.", "") if out_dtype is not None else None, missing, extra_names)
    try:
        from palingenesis import __version__
    except ImportError:  # pragma: no cover
        __version__ = "unknown"
    (out / "merge_config.json").write_text(json.dumps({**asdict(report), "palingenesis": __version__}, indent=2))
    return report


def _checkpoints(models: Iterable[str | Path]) -> list[Checkpoint]:
    return [m if isinstance(m, Checkpoint) else Checkpoint(m) for m in models]


def linear(models: list[str | Path], weights: list[float], out: str | Path, *, template: str | Path | None = None,
           missing: str = "error", dtype=None, unnormalized: bool = False, overwrite: bool = False,
           extra: str = "error") -> MergeReport:
    """w = sum_i weights[i] * models[i]. Weights must sum to 1 (a convex or affine combination) unless unnormalized=True.
    template: whose tensor set, shard layout, dtype and files the output takes (default: the first model)."""
    if len(models) != len(weights) or not models:
        raise MergeError("linear needs one weight per model")
    if not unnormalized and not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise MergeError(f"weights sum to {sum(weights):g}, not 1 (unnormalized=True to allow it)")
    cks = _checkpoints(models)
    tpl = Checkpoint(template) if template is not None else cks[0]
    return _combine("linear", list(zip(cks, map(float, weights))), out, base=None, template=tpl, missing=missing,
                    dtype=dtype, overwrite=overwrite, extra=extra)


def interpolate(start: str | Path, end: str | Path, lam: float, out: str | Path, **kwargs) -> MergeReport:
    """w = start + lam * (end - start) = (1 - lam) start + lam end; lam outside [0, 1] extrapolates.
    Template: the start (pass template= to change)."""
    if kwargs.get("template") is None:
        kwargs["template"] = start
    report = linear([start, end], [1.0 - lam, lam], out, **kwargs)
    return _relabel(report, "interpolate", {"lam": lam})


def soup(models: list[str | Path], out: str | Path, **kwargs) -> MergeReport:
    """w = mean of the models (a uniform soup of same-origin checkpoints)."""
    if not models:
        raise MergeError("soup needs at least one model")
    report = linear(models, [1.0 / len(models)] * len(models), out, **kwargs)
    return _relabel(report, "soup")


def task_arithmetic(base: str | Path, models: list[str | Path], weights: list[float], out: str | Path, *,
                    template: str | Path | None = None, missing: str = "error", dtype=None, overwrite: bool = False,
                    extra: str = "error") -> MergeReport:
    """w = base + sum_i weights[i] * (models[i] - base): each model's change from the shared base, scaled and added.
    With weights [1, lam], A keeps all of its own training and gains lam of B's (a linear merge would also pull A back
    toward the base). Template: the first model (pass template= to change)."""
    if len(models) != len(weights) or not models:
        raise MergeError("task_arithmetic needs one weight per model")
    cks = _checkpoints(models)
    b = base if isinstance(base, Checkpoint) else Checkpoint(base)
    tpl = Checkpoint(template) if template is not None else cks[0]
    return _combine("task_arithmetic", list(zip(cks, map(float, weights))), out, base=b, template=tpl, missing=missing,
                    dtype=dtype, overwrite=overwrite, extra=extra)


def materialize(checkpoint: str | Path, template: str | Path, out: str | Path, *, dtype=None,
                overwrite: bool = False) -> MergeReport:
    """A partial checkpoint (e.g. a text-only training save of a multimodal model) completed into a loadable model: its
    tensors, cast to the template's dtype, plus the template's other tensors and files."""
    report = linear([checkpoint], [1.0], out, template=template, missing="template", dtype=dtype, overwrite=overwrite)
    return _relabel(report, "materialize")


def _relabel(report: MergeReport, method: str, extra: dict | None = None) -> MergeReport:
    report.method = method
    path = Path(report.out) / "merge_config.json"
    data = json.loads(path.read_text())
    data["method"] = method
    data.update(extra or {})
    path.write_text(json.dumps(data, indent=2))
    return report


# ------------------------------------------------------------------ CLI


def _model_weights(specs: list[str]) -> tuple[list[str], list[float]]:
    models, weights = [], []
    for s in specs:
        path, sep, w = s.rpartition(":")
        try:
            if not sep:
                raise ValueError
            weights.append(float(w))
            models.append(path)
        except ValueError:
            raise MergeError(f"expected MODEL:WEIGHT, got {s!r}") from None
    return models, weights


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="pgs merge", description="weight-space merging of checkpoints (palingenesis.merge)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, template=True):
        p.add_argument("-o", "--out", required=True, help="output directory")
        if template:
            p.add_argument("--template", default=None, help="model whose tensor set, layout, dtype and files the output takes")
            p.add_argument("--missing", choices=("error", "template"), default="error",
                           help="a tensor some source lacks: error, or copy the template's")
        p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=None, help="output dtype (default: the template's)")
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("interpolate", help="start + lam (end - start)")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--lam", type=float, required=True)
    common(p)
    p = sub.add_parser("linear", help="sum_i a_i W_i (MODEL:WEIGHT ..., weights sum to 1)")
    p.add_argument("models", nargs="+", metavar="MODEL:WEIGHT")
    p.add_argument("--unnormalized", action="store_true", help="allow weights that do not sum to 1")
    common(p)
    p = sub.add_parser("soup", help="mean of the models")
    p.add_argument("models", nargs="+")
    common(p)
    p = sub.add_parser("task-arithmetic", help="base + sum_i l_i (W_i - base) (MODEL:WEIGHT ...)")
    p.add_argument("--base", required=True)
    p.add_argument("models", nargs="+", metavar="MODEL:WEIGHT")
    common(p)
    p = sub.add_parser("materialize", help="complete a partial checkpoint from a full export")
    p.add_argument("checkpoint")
    p.add_argument("--template", required=True)
    common(p, template=False)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    kw = {"dtype": a.dtype, "overwrite": a.overwrite}
    if a.cmd != "materialize":
        kw.update(template=a.template, missing=a.missing)
    try:
        if a.cmd == "interpolate":
            if a.template is None:
                kw.pop("template")
            report = interpolate(a.start, a.end, a.lam, a.out, **kw)
        elif a.cmd == "linear":
            models, weights = _model_weights(a.models)
            report = linear(models, weights, a.out, unnormalized=a.unnormalized, **kw)
        elif a.cmd == "soup":
            report = soup(a.models, a.out, **kw)
        elif a.cmd == "task-arithmetic":
            models, weights = _model_weights(a.models)
            report = task_arithmetic(a.base, models, weights, a.out, **kw)
        else:
            report = materialize(a.checkpoint, a.template, a.out, **kw)
    except MergeError as e:
        print(f"pgs merge: {e}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
