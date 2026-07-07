"""Benchmark protocol: splits, tuning carve, corruption, imputation, tidy results.

Implements spec PLAN.md §10. All randomness flows through explicit seeds /
numpy Generators so that every method within a repeat sees identical splits
and corruption masks.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RESULT_COLUMNS = ["dataset", "method", "seed", "condition", "metric", "value"]


def make_split(y: pd.Series, seed: int, task: str) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    strat = y if task == "binary" else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(te)


def train_val_split(
    X: pd.DataFrame, y: pd.Series, seed: int, task: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    strat = y if task == "binary" else None
    result = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    return cast(tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], result)


@dataclass(frozen=True)
class ImputeStats:
    medians: dict[str, float]
    modes: dict[str, Any]


def _is_cat(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object


def impute_stats(X_train: pd.DataFrame) -> ImputeStats:
    medians = {c: float(X_train[c].median()) for c in X_train.columns if not _is_cat(X_train[c])}
    modes = {
        c: (X_train[c].mode(dropna=True).iloc[0] if X_train[c].notna().any() else "missing")
        for c in X_train.columns
        if _is_cat(X_train[c])
    }
    return ImputeStats(medians=medians, modes=modes)


def apply_impute(X: pd.DataFrame, stats: ImputeStats) -> pd.DataFrame:
    out = X.copy()
    for c, m in stats.medians.items():
        out[c] = out[c].fillna(m)
    for c, m in stats.modes.items():
        if isinstance(out[c].dtype, pd.CategoricalDtype) and m not in out[c].cat.categories:
            out[c] = out[c].cat.add_categories([m])
        out[c] = out[c].fillna(m)
    return out


def corrupt_missing(X: pd.DataFrame, rho: float, rng: np.random.Generator) -> pd.DataFrame:
    out = X.copy()
    mask = rng.uniform(size=X.shape) < rho
    for j, c in enumerate(out.columns):
        col_mask = mask[:, j]
        if col_mask.any():
            out.loc[out.index[col_mask], c] = np.nan
    return out


def corrupt_noise(
    X: pd.DataFrame, rho: float, train_sd: dict[str, float], rng: np.random.Generator
) -> pd.DataFrame:
    out = X.copy()
    for c, sd in train_sd.items():
        col_mask = rng.uniform(size=len(out)) < rho
        noise = rng.normal(0.0, 0.5 * sd, size=int(col_mask.sum()))
        out.loc[out.index[col_mask], c] = out.loc[out.index[col_mask], c] + noise
    return out


def score_binary(y_true: Any, scores: Any) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    return {"auroc": float(roc_auc_score(y_true, scores))}


def score_regression(y_true: Any, pred: Any) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def result_row(
    dataset: str, method: str, seed: int, condition: str, metric: str, value: float
) -> dict[str, Any]:
    return dict(
        dataset=dataset, method=method, seed=int(seed),
        condition=condition, metric=metric, value=float(value),
    )


def write_rows(rows: list[dict[str, Any]], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
