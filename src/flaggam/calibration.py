"""PD calibration diagnostics and recalibration for FlagGAM.

This module is an ORIGINAL ADDITION and is not part of Zhao & Welsch
(arXiv:2605.31189): the paper evaluates only ranking metrics and never
assesses probability calibration. Design notes in docs/DECISIONS.md entry 19.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _validate_binary(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary 0/1")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")
    return y_true.astype(float), y_prob


def _bin_edges(y_prob: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"unknown strategy {strategy!r}")


def reliability_curve(
    y_true: Any, y_prob: Any, n_bins: int = 10, strategy: str = "uniform"
) -> pd.DataFrame:
    """Binned reliability table (empty bins dropped)."""
    y, p = _validate_binary(y_true, y_prob)
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "mean_predicted": float(p[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def brier_score(y_true: Any, y_prob: Any) -> float:
    y, p = _validate_binary(y_true, y_prob)
    return float(brier_score_loss(y, p))


def expected_calibration_error(
    y_true: Any, y_prob: Any, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    table = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    w = table["count"] / table["count"].sum()
    return float((w * (table["mean_predicted"] - table["fraction_positive"]).abs()).sum())


def calibration_in_the_large(y_true: Any, y_prob: Any) -> dict[str, float]:
    y, p = _validate_binary(y_true, y_prob)
    mean_pred, obs = float(p.mean()), float(y.mean())
    return {
        "mean_predicted": mean_pred,
        "observed_rate": obs,
        "difference": mean_pred - obs,
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _take(X: Any, idx: np.ndarray) -> Any:
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


class CalibratedFlagGAM:
    """Recalibration wrapper: platt / isotonic / base_rate over a FlagGAM classifier.

    cv=k cross-fits k clones to obtain leak-free out-of-fold predictions for
    the single pooled calibrator, then refits the estimator on all data;
    cv="prefit" treats `estimator` as already fitted and (X, y) in fit() as
    pure calibration data. Binary classification only.
    """

    def __init__(
        self,
        estimator: Any,
        method: str = "platt",
        cv: int | str = 5,
        target_rate: float | None = None,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.target_rate = target_rate

    def _oof_probs(self, X: Any, y: np.ndarray) -> np.ndarray:
        if self.cv == "prefit":
            return self.estimator.predict_proba(X)[:, 1]
        oof = np.empty(len(y), dtype=float)
        skf = StratifiedKFold(n_splits=int(self.cv), shuffle=True, random_state=0)
        Xf = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        for tr, te in skf.split(np.zeros(len(y)), y):
            m = clone(self.estimator).fit(_take(Xf, tr), y[tr])
            oof[te] = m.predict_proba(_take(Xf, te))[:, 1]
        return oof

    def fit(self, X: Any, y: Any) -> "CalibratedFlagGAM":
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CalibratedFlagGAM supports binary targets only")
        # oof is P(class-1) from the estimator's LabelEncoder, whose sort order
        # matches np.unique; encode y the same way so calibrators see numeric 0/1.
        y01 = (y == classes[1]).astype(float)
        if self.method == "base_rate" and self.target_rate is None:
            raise ValueError("method='base_rate' requires target_rate")
        if self.method not in ("platt", "isotonic", "base_rate"):
            raise ValueError(f"unknown method {self.method!r}")
        oof = self._oof_probs(X, y)
        if self.method == "platt":
            self.calibrator_: Any = LogisticRegression(C=1e6, solver="lbfgs").fit(
                _logit(oof).reshape(-1, 1), y01
            )
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            ).fit(oof, y01)
        else:  # base_rate
            logits = _logit(oof)
            self.calibrator_ = float(
                brentq(lambda d: expit(logits + d).mean() - self.target_rate, -30, 30)
            )
        self.estimator_ = (
            self.estimator if self.cv == "prefit" else clone(self.estimator).fit(X, y)
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.estimator_.predict_proba(X)[:, 1]
        if self.method == "platt":
            p1 = self.calibrator_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            p1 = self.calibrator_.predict(raw)
        else:
            p1 = expit(_logit(raw) + self.calibrator_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
