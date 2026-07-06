"""Missing-indicator discovery. 'no_evidence' semantics live in bases.transform."""

import logging

import numpy as np
import pandas as pd

from .bases import MissingIndicatorBasis
from .screening import bh_adjust, chi_square_test, two_proportion_test

logger = logging.getLogger(__name__)


def discover_missing_indicators(
    X: pd.DataFrame, y: np.ndarray, task: str, min_support: int, fdr_alpha: float
) -> list[MissingIndicatorBasis]:
    """One candidate per feature; BH across features (one test each)."""
    candidates: list[tuple[str, int, float]] = []
    for col in X.columns:
        miss = X[col].isna().to_numpy()
        n_miss = int(miss.sum())
        if n_miss < min_support or (len(miss) - n_miss) < min_support:
            continue
        if task == "binary":
            k1, k0 = int(y[miss].sum()), int(y[~miss].sum())
            p = two_proportion_test(k1, n_miss, k0, len(miss) - n_miss)
        else:
            p = chi_square_test(y[miss], y[~miss])
        candidates.append((col, n_miss, p))
    if not candidates:
        return []
    p_adj = bh_adjust(np.array([c[2] for c in candidates]))
    out = []
    for (col, n_miss, p), pa in zip(candidates, p_adj, strict=False):
        if pa <= fdr_alpha:
            out.append(
                MissingIndicatorBasis(
                    feature=col,
                    support=n_miss,
                    effect_size=float("nan"),
                    p_value=p,
                    p_adj=float(pa),
                    enriched_class=None,
                )
            )
    return out
