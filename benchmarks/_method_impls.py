"""Concrete Method subclasses that depend on optional or heavy libraries.

Imported lazily inside benchmarks.methods.get_methods() to avoid circular
imports and to keep optional-dependency failures local.  Do not import this
module at the top level of any other module.
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

# Safe: _method_impls is only loaded inside get_methods(), at which point
# benchmarks.methods is already fully initialised in sys.modules.
from benchmarks.methods import Method, _num_cat, _select, _tree_ct
from benchmarks.protocol import score_binary, score_regression, train_val_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded optional imports
# ---------------------------------------------------------------------------

_rulefit_skip: str | None = None
try:
    from imodels import RuleFitClassifier, RuleFitRegressor  # type: ignore[import]

    _rulefit_available = True
except Exception as _e_rf:
    _rulefit_skip = str(_e_rf)
    _rulefit_available = False
    RuleFitClassifier = None  # type: ignore[assignment,misc]
    RuleFitRegressor = None  # type: ignore[assignment,misc]

_glrm_skip: str | None = None
try:
    from aix360.algorithms.rbm import (  # type: ignore[import]
        LinearRuleRegression,
        LogisticRuleRegression,
    )

    _glrm_available = True
except Exception as _e_glrm:
    _glrm_skip = str(_e_glrm)
    _glrm_available = False
    LogisticRuleRegression = None  # type: ignore[assignment,misc]
    LinearRuleRegression = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------


def _xgb_tune(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
    seed: int,
) -> tuple[dict, int]:
    """Grid-search max_depth × learning_rate using XGBoost's native early stopping.

    Returns (best_params, best_iteration) where best_iteration is the 0-based
    iteration index from the winning tuning candidate.
    """
    import xgboost as xgb  # type: ignore[import]

    grid = [
        {"max_depth": d, "learning_rate": lr}
        for d in [3, 4, 6]
        for lr in [0.03, 0.1]
    ]
    best_params: dict | None = None
    best_score = -np.inf
    best_iter = 0

    Cls = xgb.XGBClassifier if task == "binary" else xgb.XGBRegressor
    for params in grid:
        m = Cls(
            n_estimators=2000,
            early_stopping_rounds=50,
            enable_categorical=True,
            tree_method="hist",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=seed,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        if task == "binary":
            s = m.predict_proba(X_val)[:, 1]
            score = score_binary(y_val, s)["auroc"]
        else:
            s = m.predict(X_val)
            score = -score_regression(y_val, s)["rmse"]
        if score > best_score:
            best_score = score
            best_params = params
            best_iter = m.best_iteration

    return best_params, best_iter  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


class XGBoostMethod(Method):
    """XGBoost with native NaN and pandas Categorical support (enable_categorical + hist).

    Tuning grid: max_depth ∈ {3,4,6} × learning_rate ∈ {0.03,0.1}, using XGBoost's
    native early stopping (n_estimators=2000, early_stopping_rounds=50) on the val carve.
    Refit on full train uses n_estimators=best_iteration+1 without early stopping.
    """

    name = "xgboost"
    needs_imputation = False

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "XGBoostMethod":
        import xgboost as xgb  # type: ignore[import]

        X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed, self._task)
        best_params, best_iter = _xgb_tune(X_tr, y_tr, X_val, y_val, self._task, seed)

        Cls = xgb.XGBClassifier if self._task == "binary" else xgb.XGBRegressor
        self._model = Cls(
            n_estimators=best_iter + 1,
            enable_categorical=True,
            tree_method="hist",
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            random_state=seed,
        ).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# EBM (interpret)
# ---------------------------------------------------------------------------


class EBMMethod(Method):
    """Explainable Boosting Machine (interpret); no tuning, interactions=0.

    EBM handles NaN natively (needs_imputation=False). It emits a UserWarning
    about missing values in visualisations — suppressed here because it is a
    cosmetic limitation of the explain() interface, not a modelling error.
    """

    name = "ebm"
    needs_imputation = False

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "EBMMethod":
        from interpret.glassbox import (  # type: ignore[import]
            ExplainableBoostingClassifier,
            ExplainableBoostingRegressor,
        )

        Cls = (
            ExplainableBoostingClassifier
            if self._task == "binary"
            else ExplainableBoostingRegressor
        )
        with warnings.catch_warnings():
            # EBM warns "Missing values detected. Our visualizations do not currently
            # display missing values." This is a visualisation limitation, not an error.
            warnings.filterwarnings(
                "ignore",
                message="Missing values detected",
                category=UserWarning,
            )
            self._model = Cls(interactions=0, random_state=seed).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# RuleFit (imodels)
# ---------------------------------------------------------------------------


class RuleFitMethod(Method):
    """imodels RuleFitClassifier/Regressor; tuned over tree_size × max_rules.

    Dense numeric input required: categorical columns are OHE-encoded inside a
    per-candidate Pipeline so preprocessing is refit without leakage.
    Tuning grid: tree_size ∈ {2,3,4} × max_rules ∈ {100,200,500}.

    imodels RuleFit internally uses LogisticRegression with a deprecated `penalty`
    parameter (sklearn ≥1.8 FutureWarning). Suppressed here because the warning
    originates in imodels' own call, not in our code, and has no fix on our side.
    """

    name = "rulefit"
    needs_imputation = True

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "RuleFitMethod":
        from sklearn.pipeline import Pipeline

        _, cats = _num_cat(X_train)
        X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train, seed, self._task)

        def build(params: dict) -> Pipeline:
            if self._task == "binary":
                model: Any = RuleFitClassifier(
                    tree_size=params["tree_size"],
                    max_rules=params["max_rules"],
                    random_state=seed,
                )
            else:
                model = RuleFitRegressor(
                    tree_size=params["tree_size"],
                    max_rules=params["max_rules"],
                    random_state=seed,
                )
            return Pipeline([("prep", _tree_ct(cats)), ("model", model)])

        grid = [
            {"tree_size": ts, "max_rules": mr}
            for ts in [2, 3, 4]
            for mr in [100, 200, 500]
        ]
        with warnings.catch_warnings():
            # imodels RuleFit uses sklearn LogisticRegression with a `penalty` kwarg
            # that was deprecated in sklearn 1.8. No fix available on our side.
            warnings.filterwarnings("ignore", message=".*penalty.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*penalty.*", category=UserWarning)
            warnings.filterwarnings(
                "ignore", message="Inconsistent values.*", category=UserWarning
            )
            best = _select(grid, build, X_tr, y_tr, X_val, y_val, self._task)
            self._model = build(best).fit(X_train, y_train)
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# GLRM (aix360) — optional
# ---------------------------------------------------------------------------


class GLRMMethod(Method):
    """Generalised Linear Rule Model (aix360 rbm). Optional; skipped when aix360
    or its dependency cvxpy is not installed.
    """

    name = "glrm"
    needs_imputation = True

    def __init__(self, task: str) -> None:
        self._task = task

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> "GLRMMethod":
        _, cats = _num_cat(X_train)
        ct = _tree_ct(cats)
        X_enc = ct.fit_transform(X_train)
        if self._task == "binary":
            self._glrm = LogisticRuleRegression(lambda0=1e-3, lambda1=1e-3, useOrd=True)
        else:
            self._glrm = LinearRuleRegression(lambda0=1e-3, lambda1=1e-3)
        self._glrm.fit(X_enc, y_train.values)
        self._ct = ct
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._ct.transform(X)
        if self._task == "binary":
            return self._glrm.predict_proba(X_enc)[:, 1]
        return self._glrm.predict(X_enc)
