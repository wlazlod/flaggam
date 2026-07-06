"""Flag Core Module: rule discovery and Z(X) construction (training data only)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .bases import Basis, CategoryBasis, HingeBasis, ThresholdBasis, TrendBasis
from .missing import discover_missing_indicators
from .screening import (
    bh_adjust,
    chi_square_test,
    compute_min_support,
    log_odds_ratio,
    risk_difference,
    standardized_mean_difference,
    two_proportion_test,
    welch_t_test,
)

logger = logging.getLogger(__name__)


class FlagCoreModule:
    """Discovers per-feature flag bases and assembles the sparse Z(X) matrix."""

    def __init__(
        self,
        task: str,
        quantile_low: tuple[float, float] = (0.05, 0.45),
        quantile_high: tuple[float, float] = (0.55, 0.95),
        quantile_step: float = 0.05,
        min_support: int | str = "auto",
        fdr_alpha: float = 0.05,
        effect_size: str = "risk_difference",
        missing: str = "no_evidence",
    ) -> None:
        self.task = task
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.quantile_step = quantile_step
        self.min_support = min_support
        self.fdr_alpha = fdr_alpha
        self.effect_size = effect_size
        self.missing = missing

    # ---- fit -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FlagCoreModule":
        n = len(X)
        self.min_support_ = (
            compute_min_support(n) if self.min_support == "auto" else int(self.min_support)
        )
        self.numerical_features_ = [c for c in X.columns if _is_numeric(X[c])]
        self.categorical_features_ = [c for c in X.columns if c not in self.numerical_features_]
        bases: list[Basis] = []
        for col in self.numerical_features_:
            bases.extend(self._discover_numerical(col, X[col].to_numpy(dtype=float), y))
        for col in self.categorical_features_:
            bases.extend(self._discover_categorical(col, X[col].to_numpy(dtype=object), y))
        if self.missing == "indicator":
            bases.extend(
                discover_missing_indicators(X, y, self.task, self.min_support_, self.fdr_alpha)
            )
        self.bases_ = bases
        logger.info("FlagCoreModule: %d bases from %d features", len(bases), X.shape[1])
        return self

    # ---- numerical, classification -------------------------------------

    def _candidate_cutoffs(self, x_obs: np.ndarray) -> list[tuple[float, str]]:
        qs_low = np.arange(self.quantile_low[0], self.quantile_low[1] + 1e-9, self.quantile_step)
        qs_high = np.arange(
            self.quantile_high[0], self.quantile_high[1] + 1e-9, self.quantile_step
        )
        cands = [(float(np.quantile(x_obs, q)), "low") for q in qs_low]
        cands += [(float(np.quantile(x_obs, q)), "high") for q in qs_high]
        seen: set[tuple[float, str]] = set()
        return [c for c in cands if not (c in seen or seen.add(c))]

    def _discover_numerical(self, col: str, x: np.ndarray, y: np.ndarray) -> list[Basis]:
        obs = ~np.isnan(x)
        x_obs, y_obs = x[obs], y[obs]
        if len(x_obs) < 2 * self.min_support_:
            return []
        if self.task == "regression":
            return self._discover_numerical_regression(col, x_obs, y_obs)  # Task 5
        rows = []  # (cutoff, side, support, p, effect, enriched_class)
        for cutoff, side in self._candidate_cutoffs(x_obs):
            tail = x_obs <= cutoff if side == "low" else x_obs >= cutoff
            n_tail, n_base = int(tail.sum()), int((~tail).sum())
            if n_tail < self.min_support_ or n_base < self.min_support_:
                continue
            p, eff, cls = self._screen_tail_cls(y_obs, tail)
            rows.append((cutoff, side, n_tail, p, eff, cls))
        if not rows:
            return []
        p_adj = bh_adjust(np.array([r[3] for r in rows]))
        out: list[Basis] = []
        for side in ("low", "high"):
            sig = [
                (r, pa) for r, pa in zip(rows, p_adj) if r[1] == side and pa <= self.fdr_alpha
            ]
            if not sig:
                continue
            # Best: largest effect, then smallest p, then lowest cutoff.
            (cutoff, _, supp, p, eff, cls), pa = max(
                sig, key=lambda t: (t[0][4], -t[0][3], -t[0][0])
            )
            out.append(
                ThresholdBasis(
                    feature=col,
                    cutoff=cutoff,
                    side=side,
                    support=supp,
                    effect_size=eff,
                    p_value=p,
                    p_adj=float(pa),
                    enriched_class=cls,
                )
            )
        return out

    def _screen_tail_cls(self, y: np.ndarray, tail: np.ndarray) -> tuple[float, float, Any]:
        y_tail, y_base = y[tail], y[~tail]
        if self.task == "binary":
            k1, k0 = int(y_tail.sum()), int(y_base.sum())
            n1, n0 = len(y_tail), len(y_base)
            p = two_proportion_test(k1, n1, k0, n0)
            eff = (
                risk_difference(k1, n1, k0, n0)
                if self.effect_size == "risk_difference"
                else log_odds_ratio(k1, n1, k0, n0)
            )
            cls = 1 if k1 / n1 >= k0 / n0 else 0
            return p, eff, cls
        p = chi_square_test(y_tail, y_base)
        classes = np.unique(y)
        rate_tail = np.array([(y_tail == c).mean() for c in classes])
        rate_base = np.array([(y_base == c).mean() + 1e-12 for c in classes])
        eff = _cramers_v_2xk(y_tail, y_base)
        return p, eff, classes[int(np.argmax(rate_tail / rate_base))]

    # ---- categorical, classification -----------------------------------

    def _discover_categorical(self, col: str, x: np.ndarray, y: np.ndarray) -> list[Basis]:
        obs = np.array([v is not None and v == v for v in x])
        x_obs, y_obs = x[obs], y[obs]
        if self.task == "regression":
            return self._discover_categorical_regression(col, x_obs, y_obs)  # Task 5
        rows = []
        for level in pd.unique(x_obs):
            in_level = x_obs == level
            n_in, n_out = int(in_level.sum()), int((~in_level).sum())
            if n_in < self.min_support_ or n_out < self.min_support_:
                continue
            p, eff, cls = self._screen_tail_cls(y_obs, in_level)
            rows.append((level, n_in, p, eff, cls))
        if not rows:
            return []
        p_adj = bh_adjust(np.array([r[2] for r in rows]))
        return [
            CategoryBasis(
                feature=col,
                level=level,
                support=n_in,
                effect_size=eff,
                p_value=p,
                p_adj=float(pa),
                enriched_class=cls,
            )
            for (level, n_in, p, eff, cls), pa in zip(rows, p_adj)
            if pa <= self.fdr_alpha
        ]

    # ---- numerical, regression -------------------------------------------

    def _discover_numerical_regression(
        self, col: str, x_obs: np.ndarray, y_obs: np.ndarray
    ) -> list[Basis]:
        mean = float(np.mean(x_obs))
        out: list[Basis] = [
            TrendBasis(
                feature=col, mean=mean, support=len(x_obs), effect_size=float("nan"),
                p_value=float("nan"), p_adj=float("nan"), enriched_class=None,
            )
        ]
        rows = []  # (cutoff, side, support, p, smd)
        for cutoff, side in self._candidate_cutoffs(x_obs):
            tail = x_obs <= cutoff if side == "low" else x_obs >= cutoff
            n_tail, n_base = int(tail.sum()), int((~tail).sum())
            if n_tail < self.min_support_ or n_base < self.min_support_:
                continue
            p = welch_t_test(y_obs[tail], y_obs[~tail])
            smd = standardized_mean_difference(y_obs[tail], y_obs[~tail])
            rows.append((cutoff, side, n_tail, p, smd))
        if not rows:
            return out
        p_adj = bh_adjust(np.array([r[3] for r in rows]))
        for side in ("low", "high"):
            sig = [(r, pa) for r, pa in zip(rows, p_adj) if r[1] == side and pa <= self.fdr_alpha]
            if not sig:
                continue
            (cutoff, _, supp, p, smd), pa = max(sig, key=lambda t: (t[0][4], -t[0][3], -t[0][0]))
            out.append(
                HingeBasis(
                    feature=col, cutoff=cutoff, side=side, support=supp,
                    effect_size=smd, p_value=p, p_adj=float(pa), enriched_class=None,
                )
            )
        return out

    # ---- categorical, regression -----------------------------------------

    def _discover_categorical_regression(
        self, col: str, x_obs: np.ndarray, y_obs: np.ndarray
    ) -> list[Basis]:
        rows = []
        for level in pd.unique(x_obs):
            in_level = x_obs == level
            n_in, n_out = int(in_level.sum()), int((~in_level).sum())
            if n_in < self.min_support_ or n_out < self.min_support_:
                continue
            p = welch_t_test(y_obs[in_level], y_obs[~in_level])
            smd = standardized_mean_difference(y_obs[in_level], y_obs[~in_level])
            rows.append((level, n_in, p, smd))
        if not rows:
            return []
        p_adj = bh_adjust(np.array([r[2] for r in rows]))
        return [
            CategoryBasis(
                feature=col, level=level, support=n_in, effect_size=smd,
                p_value=p, p_adj=float(pa), enriched_class=None,
            )
            for (level, n_in, p, smd), pa in zip(rows, p_adj)
            if pa <= self.fdr_alpha
        ]

    # ---- transform / metadata ------------------------------------------

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        n = len(X)
        if not self.bases_:
            return sparse.csr_matrix((n, 0))
        cols = [
            sparse.csr_matrix(b.transform(X[b.feature].to_numpy()).reshape(-1, 1))
            for b in self.bases_
        ]
        return sparse.hstack(cols, format="csr")

    def metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": [b.feature for b in self.bases_],
                "kind": [b.kind for b in self.bases_],
                "rule": [b.name for b in self.bases_],
                "cutoff": [getattr(b, "cutoff", np.nan) for b in self.bases_],
                "level": [getattr(b, "level", None) for b in self.bases_],
                "support": [b.support for b in self.bases_],
                "effect_size": [b.effect_size for b in self.bases_],
                "p_value": [b.p_value for b in self.bases_],
                "p_adj": [b.p_adj for b in self.bases_],
                "enriched_class": [b.enriched_class for b in self.bases_],
            }
        )


# ---- module-level helpers ----------------------------------------------


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)


def _cramers_v_2xk(y_tail: np.ndarray, y_base: np.ndarray) -> float:
    classes = np.union1d(np.unique(y_tail), np.unique(y_base))
    table = np.array(
        [[np.sum(y_tail == c) for c in classes], [np.sum(y_base == c) for c in classes]],
        dtype=float,
    )
    n = table.sum()
    expected = np.outer(table.sum(1), table.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
