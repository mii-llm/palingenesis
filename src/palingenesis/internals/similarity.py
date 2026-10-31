"""What did a training stage change, layer by layer: representation similarity and probe transfer between checkpoints.

Linear CKA (Kornblith et al. 2019) between the residuals of two models on the SAME inputs and positions:

    CKA(X, Y) = HSIC(XXᵀ, YYᵀ) / sqrt(HSIC(XXᵀ, XXᵀ) HSIC(YYᵀ, YYᵀ))

invariant to orthogonal transforms and isotropic scaling of either side, not to arbitrary linear maps. The default is
the DEBIASED estimator (the unbiased HSIC of Song et al. 2012, as in Nguyen et al. 2021's minibatch CKA): the biased
one is pushed toward 1 when the number of rows is not much larger than the hidden size (Murphy et al. 2024), which is
the usual case here (a few thousand prompts, d = 1024-4096). Both are computed from d x d feature products, never an
n x n Gram matrix, in float64.

CKA is dominated by the top principal components and can be moved without changing what the model does (Davari et al.
2022; Ding et al. 2021), so `compare_stores` always reports it next to measures a probe would notice:

    cosine        mean cosine between a row's residual in A and in B (same basis: fine-tuning keeps coordinates)
    norm_ratio    mean |B| / |A|
    transfer      a probe fitted on A, evaluated on B's held-out rows (and B's own probe, and the angle between
                  their diff-of-means directions): high own-AUROC with poor transfer = the information is still there
                  but moved (re-read); both high with a behaviour change = the readout changed, not the representation
"""

from __future__ import annotations

import math

import torch

from palingenesis.internals import probes as P


def _hsic_terms(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased HSIC (Song et al. 2012) of the linear kernels of x and y, times n(n-3) (cancels in CKA)."""
    n = x.shape[0]
    kd = (x * x).sum(1)  # diag of K = X Xᵀ
    ld = (y * y).sum(1)
    tr = float((x.T @ y).pow(2).sum() - (kd * ld).sum())  # tr(K~ L~), K~ = K with a zero diagonal
    ksum = float(x.sum(0).pow(2).sum() - kd.sum())  # 1ᵀ K~ 1
    lsum = float(y.sum(0).pow(2).sum() - ld.sum())
    k1 = x @ x.sum(0) - kd  # K~ 1
    l1 = y @ y.sum(0) - ld
    return tr + ksum * lsum / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * float(k1 @ l1)


def linear_cka(x: torch.Tensor, y: torch.Tensor, debiased: bool = True) -> float:
    """Linear CKA of x [n, d1] and y [n, d2] (rows paired). debiased needs n >= 4."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"linear_cka: {x.shape[0]} vs {y.shape[0]} rows")
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    if debiased:
        if x.shape[0] < 4:
            raise ValueError("linear_cka: the debiased estimator needs at least 4 rows")
        xy, xx, yy = _hsic_terms(x, y), _hsic_terms(x, x), _hsic_terms(y, y)
    else:
        xy = float((x.T @ y).pow(2).sum())
        xx = float((x.T @ x).pow(2).sum())
        yy = float((y.T @ y).pow(2).sum())
    if xx <= 0 or yy <= 0:
        return float("nan")
    return xy / math.sqrt(xx * yy)


def _same_rows(a, b) -> None:
    key = lambda s: [(r["row"], r.get("token")) for r in s.rows]  # noqa: E731
    if a.position != b.position or key(a) != key(b):
        raise ValueError(
            f"compare: {a.path} and {b.path} do not hold the same rows at the same positions "
            "(capture both models from the same data, position and max_tokens)"
        )


def compare_stores(
    a,
    b,
    *,
    layers: list[int] | None = None,
    probe_kind: str = "logistic",
    val: float = 0.1,
    test: float = 0.2,
    seed: int = 0,
    resamples: int = 1000,
    label_threshold: float = 0.5,
) -> dict:
    """Per-layer comparison of two stores captured from two checkpoints on the same rows (ActivationStore a, b)."""
    _same_rows(a, b)
    layers = [r for r in (layers if layers is not None else a.layers) if r in a.layers and r in b.layers]
    y = a.labels()
    labelled = bool(len(y)) and not torch.isnan(y).any()
    groups = a.groups()
    split = P.grouped_split(groups, val, test, seed) if labelled else None
    report = {"a": str(a.path), "b": str(b.path), "position": a.position, "n": len(a), "layers": {}}
    for r in layers:
        xa, xb = a.layer(r), b.layer(r)
        entry = {
            "cka": linear_cka(xa, xb, debiased=len(xa) >= 4),
            "cka_biased": linear_cka(xa, xb, debiased=False),
            "cosine": float(torch.nn.functional.cosine_similarity(xa.double(), xb.double(), dim=1).mean()),
            "norm_ratio": float((xb.norm(dim=1) / xa.norm(dim=1).clamp_min(1e-12)).mean()),
        }
        if labelled:
            yb = (y >= label_threshold).float()
            if yb[split["train"]].min() != yb[split["train"]].max():
                te = split["test"]
                test_groups = [groups[i] for i in te.tolist()]
                pa = P.train_probe(xa, y, split, kind=probe_kind, label_threshold=label_threshold, layer=r)
                pb = P.train_probe(xb, y, split, kind=probe_kind, label_threshold=label_threshold, layer=r)
                ev = lambda p, x: P.evaluate(p, x[te], y[te], test_groups, resamples=resamples, seed=seed)  # noqa: E731
                entry["transfer"] = {
                    "a_on_a": ev(pa, xa),
                    "a_on_b": ev(pa, xb),
                    "b_on_b": ev(pb, xb),
                    "direction_cosine": float(torch.nn.functional.cosine_similarity(pa.mean_diff, pb.mean_diff, dim=0)),
                }
        report["layers"][r] = P._rounded(entry)
    return report


def summarise_transfer(entry: dict) -> str:
    """One line per layer for the CLI: CKA, cosine, and A->A / A->B / B->B AUROC."""
    t = entry.get("transfer")
    base = f"cka {entry['cka']:.3f}  cos {entry['cosine']:.3f}  norm x{entry['norm_ratio']:.2f}"
    if not t:
        return base
    f = lambda d: f"{d['auroc']:.3f}" if d.get("auroc") is not None else "  nan"  # noqa: E731
    return (
        base
        + f"  auroc A->A {f(t['a_on_a'])}  A->B {f(t['a_on_b'])}  B->B {f(t['b_on_b'])}  dir cos {t['direction_cosine']:.3f}"
    )
