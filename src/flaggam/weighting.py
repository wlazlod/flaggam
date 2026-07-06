"""Feature-weight statistics and the compact score representation (spec §6.6-6.7).

Compact scores are a classification-only ablation/option; benchmark default is
the full Z(X). A flag contributes to the score of its enriched class.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from .bases import Basis

_FLAG_KINDS = {"threshold_low", "threshold_high", "category", "missing_indicator"}


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation between numerical x and binary y."""
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def correlation_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation ratio eta of numerical x across the groups defined by y."""
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = sum(
        len(x[y == c]) * (x[y == c].mean() - grand) ** 2 for c in np.unique(y)
    )
    return float(np.sqrt(ss_between / ss_total))


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Cramer's V between categorical x and target labels y."""
    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy(dtype=float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(1), table.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def feature_weights(
    X: pd.DataFrame, y: np.ndarray, task: str, numerical: list[str]
) -> dict[str, float]:
    """Per-feature association weight on non-missing training rows."""
    weights: dict[str, float] = {}
    for col in X.columns:
        obs = X[col].notna().to_numpy()
        if obs.sum() < 2:
            weights[col] = 0.0
            continue
        xv, yv = X[col].to_numpy()[obs], y[obs]
        if col in numerical:
            xv = xv.astype(float)
            weights[col] = (
                point_biserial(xv, yv) if task == "binary" else correlation_ratio(xv, yv)
            )
        else:
            weights[col] = cramers_v(xv, yv)
    return weights


def compact_scores(
    Z: sparse.spmatrix,
    bases: list[Basis],
    classes: np.ndarray,
    weights: dict[str, float] | None,
) -> np.ndarray:
    """Per-class (optionally feature-weighted) sums of triggered flags: (n, K)."""
    Zd = np.asarray(Z.todense(), dtype=float)
    out = np.zeros((Zd.shape[0], len(classes)))
    class_index = {c: k for k, c in enumerate(classes)}
    for j, b in enumerate(bases):
        if b.kind not in _FLAG_KINDS or b.enriched_class not in class_index:
            continue
        w = 1.0 if weights is None else weights.get(b.feature, 0.0)
        out[:, class_index[b.enriched_class]] += w * Zd[:, j]
    return out
