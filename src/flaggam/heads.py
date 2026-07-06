"""Prediction heads fit on Z(X) only. Z is used unstandardized (see DECISIONS.md)."""

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV


class AdditiveHead:
    """L2 logistic/softmax (classification) or ridge (regression) on Z(X)."""

    def __init__(
        self,
        task: str,
        C: float | list[float] = 1.0,
        alpha: float | list[float] = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.C = C
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, Z: sparse.spmatrix, y: np.ndarray) -> "AdditiveHead":
        self.n_features_ = Z.shape[1]
        if Z.shape[1] == 0:
            if self.task == "regression":
                self.model_ = DummyRegressor(strategy="mean").fit(np.zeros((len(y), 1)), y)
                self.coef_ = np.zeros(0)
                self.intercept_ = float(np.mean(y))
            else:
                self.model_ = DummyClassifier(strategy="prior").fit(np.zeros((len(y), 1)), y)
                self.coef_ = np.zeros((1, 0))
                self.intercept_ = np.zeros(1)
            return self
        if self.task == "regression":
            if isinstance(self.alpha, (list, tuple)):
                self.model_ = RidgeCV(alphas=list(self.alpha)).fit(Z, y)
            else:
                self.model_ = Ridge(alpha=self.alpha, random_state=self.random_state).fit(Z, y)
        else:
            binary = len(np.unique(y)) <= 2
            if isinstance(self.C, (list, tuple)):
                self.model_ = LogisticRegressionCV(
                    Cs=list(self.C), cv=5, penalty="l2", solver="lbfgs", max_iter=2000,
                    scoring="roc_auc" if binary else "neg_log_loss",
                    random_state=self.random_state,
                ).fit(Z, y)
            else:
                self.model_ = LogisticRegression(
                    C=self.C, penalty="l2", solver="lbfgs", max_iter=2000,
                    random_state=self.random_state,
                ).fit(Z, y)
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    def _guard(self, Z: sparse.spmatrix) -> sparse.spmatrix | np.ndarray:
        return np.zeros((Z.shape[0], 1)) if self.n_features_ == 0 else Z

    def predict(self, Z: sparse.spmatrix) -> np.ndarray:
        return self.model_.predict(self._guard(Z))

    def predict_proba(self, Z: sparse.spmatrix) -> np.ndarray:
        return self.model_.predict_proba(self._guard(Z))


class FlexibleHead:
    """User-supplied tree-ensemble estimator fit on Z(X) only (no raw features)."""

    def __init__(self, estimator, task: str, random_state: int | None = None) -> None:
        self.estimator = estimator
        self.task = task
        self.random_state = random_state

    def fit(self, Z: sparse.spmatrix, y: np.ndarray) -> "FlexibleHead":
        self.n_features_ = Z.shape[1]
        if Z.shape[1] == 0:
            fallback = (
                DummyRegressor(strategy="mean")
                if self.task == "regression"
                else DummyClassifier(strategy="prior")
            )
            self.model_ = fallback.fit(np.zeros((len(y), 1)), y)
            return self
        self.model_ = clone(self.estimator)
        if self.random_state is not None and "random_state" in self.model_.get_params():
            self.model_.set_params(random_state=self.random_state)
        self.model_.fit(Z, y)
        return self

    def _guard(self, Z: sparse.spmatrix) -> sparse.spmatrix | np.ndarray:
        return np.zeros((Z.shape[0], 1)) if self.n_features_ == 0 else Z

    def predict(self, Z: sparse.spmatrix) -> np.ndarray:
        return self.model_.predict(self._guard(Z))

    def predict_proba(self, Z: sparse.spmatrix) -> np.ndarray:
        return self.model_.predict_proba(self._guard(Z))
