"""Statistics for probe evaluation, in torch (no scikit-learn): AUROC with a cluster bootstrap, calibration (Platt
scaling, Brier, equal-mass ECE), operating points (TPR at a fixed FPR).

AUROC is the Mann-Whitney statistic with ties counted as one half. Its confidence interval resamples GROUPS with
replacement (a cluster bootstrap): several rows of one question, entity or conversation (paraphrases, translations,
the tokens of one answer) are not independent, and an item-level bootstrap or DeLong's interval would be too narrow.
All resamples are computed at once as integer weights on the sorted scores, O(resamples x n) without re-sorting.
"""

from __future__ import annotations

import math

import torch


def _as(x, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(x).detach().to("cpu", dtype).reshape(-1)


def weighted_auroc(scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """AUROC for each row of `weights` ([R, n] non-negative; one row = one resample). Ties count one half."""
    s, y = _as(scores), _as(labels).bool()
    w = weights.to(torch.float64)
    uniq, inv = torch.unique(s, return_inverse=True)  # sorted ascending
    pos = torch.zeros(w.shape[0], len(uniq), dtype=torch.float64).index_add_(1, inv[y], w[:, y])
    neg = torch.zeros(w.shape[0], len(uniq), dtype=torch.float64).index_add_(1, inv[~y], w[:, ~y])
    below = torch.cumsum(neg, 1) - neg
    num = (pos * (below + 0.5 * neg)).sum(1)
    den = pos.sum(1) * neg.sum(1)
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(num, float("nan")))


def auroc(scores, labels) -> float:
    """AUROC (nan when one class is absent)."""
    s = _as(scores)
    return float(weighted_auroc(s, _as(labels), torch.ones(1, len(s), dtype=torch.float64))[0])


def group_ids(groups: list | None, n: int) -> torch.Tensor:
    """Integer group of each item (every item its own group when groups is None)."""
    if groups is None:
        return torch.arange(n)
    index: dict = {}
    return torch.tensor([index.setdefault(g, len(index)) for g in groups])


def bootstrap_auroc(
    scores,
    labels,
    groups: list | None = None,
    *,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 250,
) -> dict:
    """AUROC with a percentile cluster-bootstrap interval: {"auroc", "lo", "hi", "n", "positives", "groups"}."""
    s, y = _as(scores), _as(labels)
    gid = group_ids(groups, len(s))
    n_groups = int(gid.max()) + 1 if len(gid) else 0
    out = {"auroc": auroc(s, y), "lo": float("nan"), "hi": float("nan"), "n": len(s), "positives": int(y.sum())}
    out["groups"] = n_groups
    if resamples <= 0 or n_groups < 2 or math.isnan(out["auroc"]):
        return out
    g = torch.Generator().manual_seed(seed)
    values = []
    for start in range(0, resamples, chunk):
        r = min(chunk, resamples - start)
        draws = torch.randint(0, n_groups, (r, n_groups), generator=g)
        counts = torch.zeros(r, n_groups, dtype=torch.float64).scatter_add_(
            1, draws, torch.ones(r, n_groups, dtype=torch.float64)
        )
        values.append(weighted_auroc(s, y, counts[:, gid]))
    v = torch.cat(values)
    v = v[~torch.isnan(v)]
    if len(v):
        out["lo"] = float(torch.quantile(v, alpha / 2))
        out["hi"] = float(torch.quantile(v, 1 - alpha / 2))
    return out


def fit_platt(logits, labels, steps: int = 100) -> tuple[float, float]:
    """Platt scaling: (a, b) minimising the log loss of sigmoid(a * logit + b) (Platt 1999)."""
    z, y = _as(logits), _as(labels)
    params = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(params[0] * z + params[1], y)
        loss.backward()
        return loss

    opt.step(closure)
    a, b = params.detach().tolist()
    return float(a), float(b)


def brier(probs, labels) -> float:
    p, y = _as(probs), _as(labels)
    return float(((p - y) ** 2).mean())


def log_loss(probs, labels, eps: float = 1e-12) -> float:
    p, y = _as(probs).clamp(eps, 1 - eps), _as(labels)
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def ece(probs, labels, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (Nixon et al. 2019): equal-width bins are mostly empty for a
    rare positive class."""
    p, y = _as(probs), _as(labels)
    if len(p) == 0:
        return float("nan")
    order = torch.argsort(p)
    total = 0.0
    for idx in torch.tensor_split(order, min(bins, len(p))):
        if len(idx):
            total += len(idx) * abs(float(p[idx].mean() - y[idx].mean()))
    return total / len(p)


def threshold_at_fpr(scores, labels, fpr: float) -> float:
    """The smallest threshold t with FPR(score > t) <= fpr on these items (scores above t are flagged positive)."""
    s, y = _as(scores), _as(labels).bool()
    neg = torch.sort(s[~y], descending=True).values
    if len(neg) == 0:
        return float(s.min()) - 1.0
    k = int(math.floor(fpr * len(neg)))  # negatives allowed above the threshold
    return float(neg[k]) if k < len(neg) else float(neg[-1]) - 1e-9


def tpr_at(scores, labels, threshold: float) -> dict:
    s, y = _as(scores), _as(labels).bool()
    flagged = s > threshold
    return {
        "threshold": threshold,
        "tpr": float(flagged[y].double().mean()) if y.any() else float("nan"),
        "fpr": float(flagged[~y].double().mean()) if (~y).any() else float("nan"),
    }
