"""Shared benchmark runner: composes protocol.py + methods.py into per-split rows.

Per dataset, per seed: split -> impute stats from train -> corrupt the test frame
per condition (rng seeded once per (seed, condition), shared across methods) ->
per method: fit on the (optionally imputed) clean train, predict on every
condition's (optionally imputed) test frame, score, and append tidy rows. A
single method's failure on a single split is caught, logged with a traceback,
and yields no rows for that split — never a silently zero-filled row.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.methods import Method, get_methods
from benchmarks.protocol import (
    apply_impute,
    corrupt_missing,
    corrupt_noise,
    impute_stats,
    make_split,
    result_row,
    score_binary,
    score_regression,
    write_rows,
)
from flaggam.datasets import CLASSIFICATION, REGRESSION, DatasetSpec

logger = logging.getLogger(__name__)

_MISS_RHO = {"miss25": 0.25, "miss50": 0.50}
_NOISE_RHO = {"noise25": 0.25, "noise50": 0.50}
_CONDITIONS = ["clean", "miss25", "miss50", "noise25", "noise50"]
# Stable per-condition salt for rng seeding (see _corrupted_frame). Never derive
# this from hash(condition): Python's built-in hash() on strings is
# process-randomized (PYTHONHASHSEED), which would break cross-run reproducibility.
_CONDITION_SALT = {"clean": 0, "miss25": 1, "miss50": 2, "noise25": 3, "noise50": 4}


@dataclass(frozen=True)
class RunConfig:
    datasets: list[str]
    methods: list[str] | None
    seeds: range
    conditions: list[str]
    out: Path


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Register --datasets/--methods/--n-splits/--seed-start/--out/--conditions."""
    p.add_argument("--datasets", nargs="*", default=None, help="Registry keys (default: all).")
    p.add_argument(
        "--methods", nargs="*", default=None, help="Method names (default: all available)."
    )
    p.add_argument("--n-splits", type=int, default=1000, help="Number of seeds/repeats.")
    p.add_argument(
        "--seed-start", type=int, default=0, help="First seed (seeds = start..start+n_splits)."
    )
    p.add_argument("--out", type=str, default=None, help="Output CSV path.")
    p.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        choices=_CONDITIONS,
        help="Corruption conditions to evaluate (default: clean only).",
    )


def _corrupted_frame(
    X_test: pd.DataFrame, condition: str, seed: int, train_sd: dict[str, float]
) -> pd.DataFrame:
    if condition == "clean":
        return X_test
    rng = np.random.default_rng((seed, _CONDITION_SALT[condition]))
    if condition in _MISS_RHO:
        return corrupt_missing(X_test, _MISS_RHO[condition], rng)
    if condition in _NOISE_RHO:
        return corrupt_noise(X_test, _NOISE_RHO[condition], train_sd, rng)
    raise ValueError(f"unknown condition: {condition!r}")


def run_benchmark(
    cfg: RunConfig,
    task: str,
    registry: dict[str, DatasetSpec] | None = None,
    factories: dict[str, Callable[[], Method]] | None = None,
) -> None:
    """Run cfg over registry (or CLASSIFICATION/REGRESSION if None), appending rows to cfg.out.

    `factories`, when given, is used directly instead of calling get_methods(task) —
    an orchestration hook for run_ablation.py/run_sensitivity.py, which need method
    sets get_methods() cannot produce (ablation variants, labeled sensitivity
    overrides). cfg.methods filtering still applies; nothing lands in `skipped`.
    """
    reg = registry if registry is not None else (CLASSIFICATION if task == "binary" else REGRESSION)
    if factories is None:
        factories, skipped = get_methods(task)
        for name, reason in skipped.items():
            logger.info("Skipping method %r: %s", name, reason)
    else:
        skipped = {}

    requested = cfg.methods if cfg.methods is not None else list(factories)
    unavailable = [n for n in requested if n not in factories]
    if unavailable:
        logger.warning("Requested methods unavailable, skipping: %s", unavailable)
    method_names = [n for n in requested if n in factories]

    for ds_name in cfg.datasets:
        X, y = reg[ds_name].loader()
        for seed in cfg.seeds:
            if seed % 25 == 0:
                logger.info("dataset=%s seed=%d", ds_name, seed)
            tr_idx, te_idx = make_split(y, seed, task)
            X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
            y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
            stats = impute_stats(X_train)
            train_sd = {c: float(X_train[c].std()) for c in stats.medians}
            condition_frames = {
                cond: _corrupted_frame(X_test, cond, seed, train_sd) for cond in cfg.conditions
            }

            rows: list[dict[str, Any]] = []
            for name in method_names:
                # Buffer this method's rows locally; only merge into `rows` if the
                # full condition loop completes without exception, so a failure on
                # one condition never leaves partial rows from earlier conditions.
                method_rows: list[dict[str, Any]] = []
                try:
                    method = factories[name]()
                    fit_X = apply_impute(X_train, stats) if method.needs_imputation else X_train
                    method.fit(fit_X, y_train, seed=seed)
                    for cond, frame in condition_frames.items():
                        test_X = apply_impute(frame, stats) if method.needs_imputation else frame
                        scores = method.predict_scores(test_X)
                        metrics = (
                            score_binary(y_test, scores)
                            if task == "binary"
                            else score_regression(y_test, scores)
                        )
                        for metric, value in metrics.items():
                            method_rows.append(
                                result_row(ds_name, name, seed, cond, metric, value)
                            )
                except Exception:
                    logger.exception(
                        "method %r failed on dataset=%r seed=%d", name, ds_name, seed
                    )
                else:
                    rows.extend(method_rows)
            write_rows(rows, cfg.out)
