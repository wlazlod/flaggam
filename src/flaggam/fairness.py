"""Fairness diagnostics and rule-level proxy audit for FlagGAM.

This module is an ORIGINAL ADDITION and is not part of Zhao & Welsch
(arXiv:2605.31189); it operationalizes the paper's own Impact-Statement
warning that selected rules may encode bias or proxies for protected
attributes. Thresholds and binarization notes in docs/DECISIONS.md entry 21.
"""

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from flaggam.calibration import expected_calibration_error
from flaggam.heads import AdditiveHead

logger = logging.getLogger(__name__)


def group_metrics(
    y_true: Any, y_prob: Any, A: Any, threshold: float = 0.5, n_bins: int = 10
) -> dict[str, Any]:
    """Per-group PD metrics and max-minus-min gaps for protected attribute A."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    a = np.asarray(A).ravel()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("group_metrics supports binary y_true only")
    rows = {}
    for g in pd.unique(a):
        m = a == g
        yg, pg = y[m], p[m]
        sel = (pg >= threshold).astype(int)
        pos = yg == 1
        single_class = len(np.unique(yg)) < 2
        rows[g] = {
            "n": int(m.sum()),
            "base_rate": float(yg.mean()),
            "mean_predicted": float(pg.mean()),
            "selection_rate": float(sel.mean()),
            "tpr": float(sel[pos].mean()) if pos.any() else np.nan,
            "auroc": np.nan if single_class else float(roc_auc_score(yg, pg)),
            "ece": expected_calibration_error(yg, pg, n_bins=n_bins),
        }
    by_group = pd.DataFrame.from_dict(rows, orient="index")

    def _gap(col: str) -> float:
        vals = by_group[col].dropna()
        return float(vals.max() - vals.min()) if len(vals) > 1 else np.nan

    gaps = {
        "demographic_parity_diff": _gap("selection_rate"),
        "equal_opportunity_diff": _gap("tpr"),
        "auroc_gap": _gap("auroc"),
    }
    return {"by_group": by_group, "gaps": gaps}


def _cramers_v(z: np.ndarray, a: np.ndarray) -> float:
    table = pd.crosstab(z, a).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0


class ProxyAudit:
    """Rank fitted rule bases by association with a protected attribute.

    Association is computed on the BINARIZED basis indicator `z > 0`: exact
    for threshold/category/missing_indicator flag bases, and a documented
    approximation (fires vs. does-not-fire) for the continuous hinge/trend
    bases. See docs/DECISIONS.md entry 21.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def _indicators(self, X: Any) -> tuple[np.ndarray, list]:
        est = self.estimator
        df = est._to_frame(X, reset=False)
        Z = est.core_.transform(df).toarray()
        return (Z > 0).astype(int), est.core_.bases_

    def report(self, X: Any, A: Any, threshold: float = 0.2) -> pd.DataFrame:
        """One row per basis: feature, rule, kind, association, method, flagged."""
        Zb, bases = self._indicators(X)
        a = np.asarray(A).ravel()
        numeric_a = np.issubdtype(a.dtype, np.number)
        rows = []
        for j, basis in enumerate(bases):
            z = Zb[:, j]
            if numeric_a:
                assoc = 0.0 if len(np.unique(z)) < 2 else abs(stats.pointbiserialr(z, a)[0])
                method = "point_biserial"
            else:
                assoc = _cramers_v(z, a)
                method = "cramers_v"
            rows.append(
                {
                    "feature": basis.feature,
                    "rule": basis.name,
                    "kind": basis.kind,
                    "association": assoc,
                    "method": method,
                    "flagged": assoc > threshold,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("association", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    def drop_proxies(
        self, X: Any, y: Any, A: Any, threshold: float = 0.2
    ) -> tuple[Any, pd.DataFrame]:
        """Refit the head without flagged bases; return (new_estimator, trade-off row).

        Supports only fitted binary classifiers with `representation="full"`
        and `head="additive"`: compact-score columns don't map 1:1 to bases
        and flexible heads can't be refit column-wise; multiclass is out of
        PD scope (see docs/DECISIONS.md entry 21).
        """
        est = self.estimator
        n_classes = len(getattr(est, "classes_", []))
        if not (n_classes == 2 and est.representation == "full" and est.head == "additive"):
            raise ValueError(
                "drop_proxies requires a fitted binary classifier with "
                "representation='full' and head='additive'"
            )
        if est.monotonic_constraints is not None:
            raise ValueError(
                "drop_proxies does not support monotonic-constrained estimators "
                "(the head refit would discard the constraints)"
            )
        report = self.report(X, A, threshold=threshold)
        flagged_rules = set(report.loc[report.flagged, "rule"])
        y_arr = np.asarray(y).ravel()
        p_before = est.predict_proba(X)[:, 1]

        new_est = copy.deepcopy(est)
        new_est.core_.bases_ = [b for b in est.core_.bases_ if b.name not in flagged_rules]
        df = new_est._to_frame(X, reset=False)
        Z = new_est.core_.transform(df)
        y_enc = new_est.label_encoder_.transform(y_arr)
        new_est.head_ = AdditiveHead(
            "binary", C=new_est.C, random_state=new_est.random_state
        ).fit(Z, y_enc)
        p_after = new_est.predict_proba(X)[:, 1]

        trade = pd.DataFrame(
            [
                {
                    "n_dropped": int(len(est.core_.bases_) - len(new_est.core_.bases_)),
                    "auroc_before": float(roc_auc_score(y_arr, p_before)),
                    "auroc_after": float(roc_auc_score(y_arr, p_after)),
                    "dp_diff_before": group_metrics(y_arr, p_before, A)["gaps"][
                        "demographic_parity_diff"
                    ],
                    "dp_diff_after": group_metrics(y_arr, p_after, A)["gaps"][
                        "demographic_parity_diff"
                    ],
                }
            ]
        )
        if not flagged_rules:
            logger.info("drop_proxies: nothing flagged at threshold %.3f", threshold)
        return new_est, trade
