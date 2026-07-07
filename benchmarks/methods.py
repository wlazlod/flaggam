"""Baseline method registry for the FlagGAM benchmark protocol (spec §10, Appendix A grids).

Every Method.fit(X, y, seed): (1) carve train_val_split(X, y, seed, task);
(2) pick hyperparameters on the validation carve; (3) refit on the FULL
training data with the chosen hyperparameters. Test data never enters here.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmarks.protocol import score_binary, score_regression, train_val_split
from flaggam import FlagGAMClassifier, FlagGAMRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cat(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object


def _num_cat(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    nums = [c for c in X.columns if not _is_cat(X[c])]
    cats = [c for c in X.columns if _is_cat(X[c])]
    return nums, cats


def _linear_ct(nums: list[str], cats: list[str]) -> ColumnTransformer:
    """ColumnTransformer for linear models: StandardScaler numerics, OHE categoricals."""
    transformers: list[Any] = []
    if nums:
        transformers.append(("num", StandardScaler(), nums))
    if cats:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)
        )
    return ColumnTransformer(transformers)


def _tree_ct(cats: list[str]) -> ColumnTransformer:
    """ColumnTransformer for tree methods: OHE categoricals, passthrough numerics."""
    if cats:
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)],
            remainder="passthrough",
        )
    return ColumnTransformer([], remainder="passthrough")


def _select(
    grid: list[dict],
    build: Callable,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
) -> dict:
    """Fit each candidate on the tuning-train carve, score on val, return best params."""
    best, best_score = None, -np.inf
    for params in grid:
        m = build(params).fit(X_tr, y_tr)
        s = m.predict_proba(X_val)[:, 1] if task == "binary" else m.predict(X_val)
        score = (
            score_binary(y_val, s)["auroc"]
            if task == "binary"
            else -score_regression(y_val, s)["rmse"]
        )
        if score > best_score:
            best, best_score = params, score
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Method:
    """Base class for benchmark methods.

    Subclasses must implement fit() and predict_scores(). The fit() contract:
    carve a train/val split internally, tune hyperparameters on the val carve,
    then refit on the full X_train before returning self.
    """

    name: str
    needs_imputation: bool
    _task: str  # "binary" or "regression"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        raise NotImplementedError

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(y=1) for binary tasks, predictions for regression."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Logistic regression (binary)
# ---------------------------------------------------------------------------


class LogisticMethod(Method):
    """Logistic regression; per-candidate Pipeline = StandardScaler + OHE + LogisticRegression.

    needs_imputation=True: runner fills NaN before calling fit; pipeline receives NaN-free input.
    Tuning grid: C ∈ {0.01, 0.1, 1, 10}.
    """

    name = "logistic"
    needs_imputation = True
    _task = "binary"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        nums, cats = _num_cat(X_train)
        X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed, self._task)

        def build(params: dict) -> Pipeline:
            return Pipeline(
                [
                    ("prep", _linear_ct(nums, cats)),
                    ("model", LogisticRegression(C=params["C"], max_iter=1000, random_state=seed)),
                ]
            )

        grid = [{"C": c} for c in [0.01, 0.1, 1, 10]]
        best = _select(grid, build, X_tr, y_tr, X_val, y_val, self._task)
        self._model = build(best).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Ridge regression (regression)
# ---------------------------------------------------------------------------


class RidgeMethod(Method):
    """Ridge regression; per-candidate Pipeline = StandardScaler + OHE + Ridge.

    needs_imputation=True: runner fills NaN before calling fit.
    Tuning grid: alpha ∈ {0.01, 0.1, 1, 10}.
    """

    name = "ridge"
    needs_imputation = True
    _task = "regression"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        nums, cats = _num_cat(X_train)
        X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed, self._task)

        def build(params: dict) -> Pipeline:
            return Pipeline(
                [
                    ("prep", _linear_ct(nums, cats)),
                    ("model", Ridge(alpha=params["alpha"])),
                ]
            )

        grid = [{"alpha": a} for a in [1e-3, 1e-2, 0.1, 1.0, 10.0]]
        best = _select(grid, build, X_tr, y_tr, X_val, y_val, self._task)
        self._model = build(best).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# FlagGAM
# ---------------------------------------------------------------------------


class FlagGAMMethod(Method):
    """FlagGAM classifier or regressor, tuned over C (binary) or alpha (regression).

    Note: FlagGAMClassifier accepts a list C for internal cross-validation. We do NOT
    use that path here — the benchmark protocol's own val carve is the tuner. Passing
    a single float C avoids double-tuning (internal CV would tune again on the same
    training fold, biasing hyperparameter selection).
    """

    name = "flaggam"
    needs_imputation = False

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed, self._task)

        if self._task == "binary":
            grid = [{"C": c} for c in [0.01, 0.1, 1, 10]]

            def build(params: dict) -> Any:
                return FlagGAMClassifier(C=params["C"], random_state=seed)

            best = _select(grid, build, X_tr, y_tr, X_val, y_val, self._task)
            self._model = FlagGAMClassifier(C=best["C"], random_state=seed).fit(X_train, y_train)
        else:
            grid = [{"alpha": a} for a in [1e-3, 1e-2, 0.1, 1.0, 10.0]]

            def build(params: dict) -> Any:  # type: ignore[misc]
                return FlagGAMRegressor(alpha=params["alpha"], random_state=seed)

            best = _select(grid, build, X_tr, y_tr, X_val, y_val, self._task)
            self._model = FlagGAMRegressor(
                alpha=best["alpha"], random_state=seed
            ).fit(X_train, y_train)

        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# FlagGAM-RF (flexible RandomForest head)
# ---------------------------------------------------------------------------


class FlagGAMRFMethod(Method):
    """FlagGAM with a flexible RandomForest head; no hyperparameter tuning."""

    name = "flaggam_rf"
    needs_imputation = False

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        if self._task == "binary":
            est = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
            self._model = FlagGAMClassifier(
                head="flexible", flexible_estimator=est, random_state=seed
            ).fit(X_train, y_train)
        else:
            est = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
            self._model = FlagGAMRegressor(
                head="flexible", flexible_estimator=est, random_state=seed
            ).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RFMethod(Method):
    """Random Forest (500 trees, bootstrap, unrestricted depth); no tuning.

    needs_imputation=True: sklearn RF does not handle NaN or pandas Categorical
    natively. OHE for categoricals is applied inside a Pipeline to avoid leakage.
    """

    name = "rf"
    needs_imputation = True

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "Method":
        _, cats = _num_cat(X_train)
        if self._task == "binary":
            model: Any = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        self._model = Pipeline([("prep", _tree_ct(cats)), ("model", model)]).fit(
            X_train, y_train
        )
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_methods(
    task: str,
) -> tuple[dict[str, Callable[[], Method]], dict[str, str]]:
    """Return (available factories, skipped {name: reason}).

    A missing optional import never raises; it lands in `skipped` with the
    ImportError message. Guarded imports for optional deps (imodels, aix360)
    occur at _method_impls load time; this function assembles the dicts based
    on which imports succeeded.
    """
    # Deferred import: _method_impls is only loaded here, after methods.py is
    # fully initialised, so the back-reference from _method_impls to this module
    # (for Method, _select, _num_cat, _tree_ct) resolves without a circular import.
    from benchmarks._method_impls import (
        EBMMethod,
        GLRMMethod,
        RuleFitMethod,
        XGBoostMethod,
        _glrm_available,
        _glrm_skip,
        _rulefit_available,
        _rulefit_skip,
    )

    factories: dict[str, Callable[[], Method]] = {}
    skipped: dict[str, str] = {}

    # Task-specific linear baseline
    if task == "binary":
        factories["logistic"] = LogisticMethod
    else:
        factories["ridge"] = RidgeMethod

    # Shared methods (task-parametrised via lambda)
    factories["flaggam"] = lambda: FlagGAMMethod(task)
    factories["flaggam_rf"] = lambda: FlagGAMRFMethod(task)
    factories["rf"] = lambda: RFMethod(task)
    factories["xgboost"] = lambda: XGBoostMethod(task)
    factories["ebm"] = lambda: EBMMethod(task)

    # RuleFit — optional (imodels)
    if _rulefit_available:
        factories["rulefit"] = lambda: RuleFitMethod(task)
    else:
        skipped["rulefit"] = f"imodels not installed: {_rulefit_skip}"

    # GLRM — optional (aix360 + cvxpy)
    if _glrm_available:
        factories["glrm"] = lambda: GLRMMethod(task)
    else:
        skipped["glrm"] = f"aix360 not installed: {_glrm_skip}"

    return factories, skipped
