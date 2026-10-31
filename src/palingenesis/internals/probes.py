"""Linear probes on captured residuals: grouped splits, fits, controls, calibration, per-layer sweeps, artifacts.

Fits (all on train-split statistics only):

    logistic   L2-regularised logistic regression on standardised features (LBFGS, full batch); the strength is
               chosen on the validation split from a grid. The default classifier: best recall at low FPR (Goldowsky-
               Dill et al. 2025) and the standard probe of Kadavath 2022 / Kossen 2024 / Orgad 2024.
    diffmean   difference of class means (mass-mean probe, Marks & Tegmark 2023): score = (x - midpoint) . u with u the
               unit vector of mu+ - mu-. No hyperparameters; transfers as well or better out of distribution (Paulo et
               al. 2024) and is the direction that is causally used (Marks & Tegmark; ITI; Im & Li 2025).
    lda        the covariance-adjusted mass-mean probe (Sigma^-1 (mu+ - mu-), pooled within-class covariance of the
               standardised features with a shrinkage chosen on validation).

Whatever the classifier, every artifact also carries the raw diff-of-means vector `mean_diff` (mu+ - mu-, residual
units): steering and ablation use it, never the classifier weights (classify with logistic, intervene with the mean
difference).

Labels are floats in [0, 1] (hard 0/1, or soft P(correct) from k samples as in Kadavath 2022): training uses them as
soft targets, evaluation binarises at `label_threshold`.

Splits are by group (question, entity, conversation; translations of one question share a group): no group is in
two splits. The layer is chosen on validation, never on test. Calibration (Platt) is fitted on validation and is
valid for that distribution only (Kadavath 2022: discrimination transfers, calibration does not): recalibrate per
domain or language.

Controls reported for every layer:
    control    the same fit on group-level random labels with the train base rate (Hewitt & Liang 2019's control
               task, with the group as the "type"): its test AUROC must be ~.5; above that, groups leak across splits.
               selectivity = AUROC - control AUROC.
    length     AUROC of the prompt length alone (a surface confound, Belinkov 2022).
    layer 0    the sweep includes the embeddings (r = 0): a high AUROC there means the label is lexical.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from palingenesis.internals import stats

KINDS = ("logistic", "diffmean", "lda")
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SHRINK_GRID = (1e-3, 1e-2, 1e-1, 0.5)
FPRS = (0.01, 0.05, 0.1)
PROBE_VERSION = 1


# ------------------------------------------------------------------------------------------------------- splits


def grouped_split(groups: list, val: float = 0.1, test: float = 0.2, seed: int = 0) -> dict[str, torch.Tensor]:
    """Indices of train / val / test: groups are shuffled (seeded) and assigned whole, by cumulative item count."""
    if not (0 <= val < 1 and 0 <= test < 1 and val + test < 1):
        raise ValueError("grouped_split: val and test must be fractions with val + test < 1")
    gid = stats.group_ids(groups, len(groups))
    n_groups = int(gid.max()) + 1
    order = torch.randperm(n_groups, generator=torch.Generator().manual_seed(seed))
    sizes = torch.bincount(gid, minlength=n_groups)
    cum = torch.cumsum(sizes[order], 0).double() / len(groups)
    split_of = torch.empty(n_groups, dtype=torch.long)
    split_of[order] = torch.where(cum <= test, 2, torch.where(cum <= test + val, 1, 0))
    if test > 0 and not (split_of == 2).any():
        split_of[order[0]] = 2
    if val > 0 and not (split_of == 1).any():
        split_of[order[-1]] = 1
    s = split_of[gid]
    out = {name: torch.nonzero(s == k).flatten() for k, name in enumerate(("train", "val", "test"))}
    if len(out["train"]) == 0:
        raise ValueError(f"grouped_split: {n_groups} group(s) are too few for train / val / test splits")
    return out


def control_labels(y: torch.Tensor, groups: list, seed: int = 0) -> torch.Tensor:
    """Group-level random labels with the base rate of `y` (every item of a group gets its group's label)."""
    gid = stats.group_ids(groups, len(groups))
    g = torch.Generator().manual_seed(seed + 7919)
    per_group = (torch.rand(int(gid.max()) + 1, generator=g) < float(y.float().mean())).float()
    return per_group[gid]


# ------------------------------------------------------------------------------------------------------- fits


def _standardise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-6) if len(x) > 1 else torch.ones_like(mean)
    return mean, scale


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int = 200) -> tuple[torch.Tensor, float]:
    """Raw-space (weight, bias) of L2 logistic regression on standardised features: loss = mean BCE + l2 * |w|^2."""
    mean, scale = _standardise(x)
    xs = (x - mean) / scale
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe", tolerance_grad=1e-9)
    target = y.to(x.dtype)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xs @ w + b, target) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w_raw = w.detach() / scale
    return w_raw, float(b.detach() - mean @ w_raw)


def class_means(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted class means (soft labels weight each row by y and 1 - y)."""
    yy = y.to(x.dtype)[:, None]
    pos = (yy * x).sum(0) / yy.sum().clamp_min(1e-12)
    neg = ((1 - yy) * x).sum(0) / (1 - yy).sum().clamp_min(1e-12)
    return pos, neg


def fit_diffmean(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
    pos, neg = class_means(x, y)
    u = (pos - neg) / (pos - neg).norm().clamp_min(1e-12)
    return u, float(-((pos + neg) / 2) @ u)


def fit_lda(x: torch.Tensor, y: torch.Tensor, shrink: float) -> tuple[torch.Tensor, float]:
    """Sigma^-1 (mu+ - mu-) on standardised features, Sigma = (1 - s) * pooled within-class covariance + s * I."""
    mean, scale = _standardise(x)
    xs = (x - mean) / scale
    pos, neg = class_means(xs, y)
    hard = y >= 0.5
    centred = torch.where(hard[:, None], xs - pos, xs - neg)
    cov = centred.T @ centred / max(len(xs) - 2, 1)
    cov = (1 - shrink) * cov + shrink * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    w = torch.linalg.solve(cov, pos - neg)
    w_raw = w / scale
    return w_raw, float(-(((pos + neg) / 2) @ w) - mean @ w_raw)


# ------------------------------------------------------------------------------------------------------- artifact


@dataclass
class Probe:
    """A fitted linear probe on residual `layer` at `position`.

    logit(x) = x . weight + bias on the RAW residual (standardisation folded in), p = sigmoid(a * logit + b) with the
    Platt calibration (a, b). `mean_diff` = mu+ - mu- in residual units (steering / ablation direction; alpha = 1 moves
    a negative-class mean onto the positive-class mean); `resid_norm` the mean residual norm at the layer (to read a
    steering vector's size). `thresholds` are logit thresholds chosen on validation ("fpr_0.05": flags 5% of
    validation negatives). `meta` records what is needed to apply it again: the model fingerprint, chat template
    kwargs, label semantics, the fit, metrics."""

    kind: str
    layer: int
    position: str
    weight: torch.Tensor
    bias: float
    mean_diff: torch.Tensor
    calibration: tuple[float, float] = (1.0, 0.0)
    resid_norm: float = float("nan")
    thresholds: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.weight.to(x.device).float() + self.bias

    def proba(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.calibration
        return torch.sigmoid(a * self.logit(x) + b)

    def flag(self, x: torch.Tensor, at: str = "fpr_0.05") -> torch.Tensor:
        if at not in self.thresholds:
            raise KeyError(f"probe has no threshold {at!r} (have {sorted(self.thresholds)})")
        return self.logit(x) > self.thresholds[at]

    @property
    def direction(self) -> torch.Tensor:
        """Unit diff-of-means direction (the intervention direction)."""
        return self.mean_diff / self.mean_diff.norm().clamp_min(1e-12)

    def save(self, path: str | Path) -> Path:
        """One .safetensors file: tensors weight / mean_diff, everything else as JSON in the header metadata."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "version": PROBE_VERSION,
            "kind": self.kind,
            "layer": self.layer,
            "position": self.position,
            "bias": self.bias,
            "calibration": list(self.calibration),
            "resid_norm": self.resid_norm,
            "thresholds": self.thresholds,
            "meta": self.meta,
        }
        tensors = {
            "weight": self.weight.float().cpu().contiguous(),
            "mean_diff": self.mean_diff.float().cpu().contiguous(),
        }
        save_file(tensors, str(path), metadata={"palingenesis_probe": json.dumps(header)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            raw = (f.metadata() or {}).get("palingenesis_probe")
            if raw is None:
                raise ValueError(f"{path}: not a palingenesis probe (no palingenesis_probe metadata)")
            header = json.loads(raw)
            weight, mean_diff = f.get_tensor("weight"), f.get_tensor("mean_diff")
        if header.get("version", 0) > PROBE_VERSION:
            raise ValueError(f"{path}: probe format v{header['version']} is newer than this library's")
        return cls(
            kind=header["kind"],
            layer=int(header["layer"]),
            position=header["position"],
            weight=weight,
            bias=float(header["bias"]),
            mean_diff=mean_diff,
            calibration=tuple(header["calibration"]),
            resid_norm=float(header["resid_norm"]),
            thresholds=header.get("thresholds", {}),
            meta=header.get("meta", {}),
        )


# ------------------------------------------------------------------------------------------------------- fit + eval


def _fit(kind: str, x, y, strength: float) -> tuple[torch.Tensor, float]:
    if kind == "logistic":
        return fit_logistic(x, y, strength)
    if kind == "lda":
        return fit_lda(x, y, strength)
    return fit_diffmean(x, y)


def _grid(kind: str, grid) -> tuple:
    if kind == "diffmean":
        return (0.0,)
    return tuple(grid) if grid else (L2_GRID if kind == "logistic" else SHRINK_GRID)


def train_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    split: dict[str, torch.Tensor],
    *,
    kind: str = "logistic",
    grid: tuple | None = None,
    label_threshold: float = 0.5,
    layer: int = -1,
    position: str = "",
    meta: dict | None = None,
) -> Probe:
    """Fit on split["train"] (the strength chosen by validation AUROC), calibrate and set thresholds on split["val"]."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    tr, va = split["train"], split["val"]
    yb = (y >= label_threshold).float()
    if yb[tr].min() == yb[tr].max():
        raise ValueError("train_probe: the train split has a single class")
    best = None
    for strength in _grid(kind, grid):
        w, b = _fit(kind, x[tr], y[tr], strength)
        score = stats.auroc(x[va] @ w + b, yb[va]) if len(va) else float("nan")
        if best is None or (not math.isnan(score) and (math.isnan(best[0]) or score > best[0])):
            best = (score, strength, w, b)
    val_auroc, strength, w, b = best
    pos, neg = class_means(x[tr], y[tr])
    probe = Probe(
        kind=kind,
        layer=layer,
        position=position,
        weight=w,
        bias=b,
        mean_diff=pos - neg,
        resid_norm=float(x[tr].norm(dim=1).mean()),
        meta={
            **(meta or {}),
            "strength": strength,
            "n_train": len(tr),
            "n_val": len(va),
            "label_threshold": label_threshold,
        },
    )
    if len(va) and yb[va].min() != yb[va].max():
        z = probe.logit(x[va])
        probe.calibration = stats.fit_platt(z, yb[va])
        probe.thresholds = {f"fpr_{f:g}": stats.threshold_at_fpr(z, yb[va], f) for f in FPRS}
    probe.meta["val_auroc"] = val_auroc
    return probe


def evaluate(
    probe: Probe, x: torch.Tensor, y: torch.Tensor, groups: list | None = None, *, resamples: int = 1000, seed: int = 0
) -> dict:
    """AUROC with a cluster-bootstrap CI, Brier, log loss, ECE of the calibrated probability, TPR at the validation
    thresholds (and the FPR they give here)."""
    yb = (y >= probe.meta.get("label_threshold", 0.5)).float()
    z = probe.logit(x)
    p = probe.proba(x)
    out = stats.bootstrap_auroc(z, yb, groups, resamples=resamples, seed=seed)
    out.update(brier=stats.brier(p, yb), log_loss=stats.log_loss(p, yb), ece=stats.ece(p, yb))
    out["at"] = {name: stats.tpr_at(z, yb, t) for name, t in probe.thresholds.items()}
    return out


# ------------------------------------------------------------------------------------------------------- sweep


def _rounded(d):
    if isinstance(d, float):
        return round(d, 4) if math.isfinite(d) else None
    if isinstance(d, dict):
        return {k: _rounded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_rounded(v) for v in d]
    return d


def sweep(
    store,
    *,
    layers: list[int] | None = None,
    kind: str = "logistic",
    tests: dict | None = None,
    val: float = 0.1,
    test: float = 0.2,
    seed: int = 0,
    resamples: int = 1000,
    grid: tuple | None = None,
    label_threshold: float = 0.5,
    control: bool = True,
    save_dir: str | Path | None = None,
    label_name: str = "label",
) -> dict:
    """Probe every residual of `store` (an ActivationStore): fit on its grouped train split, select the layer on
    validation AUROC, report test AUROC [CI] on the held-out groups and on every extra store in `tests` (other
    datasets, other languages: transfer), the diff-of-means AUROC next to the chosen kind, the control task, the length
    baseline. With save_dir, writes layer_<r>.safetensors for every layer and best.safetensors (the validation pick).
    """
    tests = tests or {}
    y_all = store.labels()
    keep = torch.nonzero(~torch.isnan(y_all)).flatten()
    if len(keep) < len(y_all):
        raise ValueError(f"{store.path}: {len(y_all) - len(keep)} rows have no label")
    for name, t in tests.items():
        if t.position != store.position:
            raise ValueError(f"test set {name}: position {t.position} differs from the train store's {store.position}")
        if t.meta["model"].get("weights_sha") != store.meta["model"].get("weights_sha"):
            raise ValueError(f"test set {name} was captured from another model than the train store")
    groups = store.groups()
    split = grouped_split(groups, val, test, seed)
    yb = (y_all >= label_threshold).float()
    ctrl = control_labels(yb, groups, seed) if control else None  # one random label per group, the train base rate
    test_groups = [groups[i] for i in split["test"].tolist()]
    layers = store.layers if layers is None else [r for r in layers if r in store.layers]
    meta = {
        "model": store.meta["model"],
        "position": store.position,
        "chat_template_kwargs": store.meta.get("chat_template_kwargs", {}),
        "label": label_name,
        "store": str(store.path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    from palingenesis import __version__

    meta["palingenesis"] = __version__
    report = {
        "train_store": str(store.path),
        "kind": kind,
        "position": store.position,
        "n": len(y_all),
        "positive_rate": float(yb.mean()),
        "split": {k: len(v) for k, v in split.items()},
        "length_auroc": {"test": stats.auroc(store.n_tokens()[split["test"]], yb[split["test"]])},
        "layers": {},
    }
    for name, t in tests.items():
        ty = (t.labels() >= label_threshold).float()
        report["length_auroc"][name] = stats.auroc(t.n_tokens(), ty)
    probes = {}
    for r in layers:
        x = store.layer(r)
        mixer = store.meta["mixers"][store.layers.index(r)] if store.meta.get("mixers") else None
        probe = train_probe(
            x,
            y_all,
            split,
            kind=kind,
            grid=grid,
            label_threshold=label_threshold,
            layer=r,
            position=store.position,
            meta={**meta, "mixer": mixer},
        )
        entry = {"mixer": mixer, "val_auroc": probe.meta["val_auroc"], "strength": probe.meta["strength"]}
        entry["test"] = evaluate(
            probe, x[split["test"]], y_all[split["test"]], test_groups, resamples=resamples, seed=seed
        )
        if kind != "diffmean":
            dm = train_probe(x, y_all, split, kind="diffmean", label_threshold=label_threshold, layer=r)
            entry["diffmean_test_auroc"] = stats.auroc(dm.logit(x[split["test"]]), yb[split["test"]])
        if ctrl is not None:
            same = (probe.meta["strength"],) if kind != "diffmean" else None
            try:
                cprobe = train_probe(x, ctrl, split, kind=kind, grid=same, layer=r)
                entry["control_test_auroc"] = stats.auroc(cprobe.logit(x[split["test"]]), ctrl[split["test"]])
            except ValueError:  # the control draw gave the train split one class (tiny data)
                entry["control_test_auroc"] = float("nan")
            entry["selectivity"] = entry["test"]["auroc"] - entry["control_test_auroc"]
        for name, t in tests.items():
            entry[name] = evaluate(probe, t.layer(r), t.labels(), t.groups(), resamples=resamples, seed=seed)
        probe.meta["metrics"] = _rounded({k: v for k, v in entry.items() if k not in ("mixer",)})
        report["layers"][r] = _rounded(entry)
        probes[r] = probe
    best = max(probes, key=lambda r: (probes[r].meta["val_auroc"], -r))
    report["best_layer"] = best
    report["best_layer_note"] = "chosen on validation AUROC; test AUROCs of the other layers are not selection-free"
    if save_dir is not None:
        save_dir = Path(save_dir)
        for r, probe in probes.items():
            probe.save(save_dir / f"layer_{r}.safetensors")
        probes[best].save(save_dir / "best.safetensors")
        (save_dir / "sweep.json").write_text(json.dumps(report, indent=2))
    report["probes"] = probes
    return report
