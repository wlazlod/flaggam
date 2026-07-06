"""FlagGAMClassifier: sklearn-compatible estimator for rule-basis GAMs."""

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .core import FlagCoreModule
from .heads import AdditiveHead, FlexibleHead
from .weighting import compact_scores, feature_weights


class _BaseFlagGAM(TransformerMixin, BaseEstimator):
    """Shared X conversion, core/head assembly, transform."""

    def _to_frame(self, X, reset: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            # Reject sparse upfront with an explicit message.
            if issparse(X):
                raise ValueError(
                    "A sparse matrix was passed, but dense data is required. "
                    "Use X.toarray() to convert to a dense numpy array."
                )
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(
                    "Expected 2D array, got 1D array instead. "
                    "Reshape your data either using array.reshape(-1, 1) "
                    "if your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
            # Reject complex dtypes with sklearn's standard message.
            if arr.dtype.kind == "c":
                raise ValueError(f"Complex data not supported\n{arr}\n")
            if reset and arr.shape[0] == 0:
                raise ValueError(
                    "0 sample(s) (shape=(0, {})) were passed while a minimum of 1 is "
                    "required.".format(arr.shape[1])
                )
            if reset and arr.shape[1] == 0:
                raise ValueError(
                    "0 feature(s) (shape=({}, 0)) while a minimum of 1 is "
                    "required.".format(arr.shape[0])
                )
            if not reset and arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {arr.shape[1]} features, but {type(self).__name__} is expecting "
                    f"{self.n_features_in_} features as input"
                )
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
            cat = self.categorical_features or []
            cat_col_names = [
                df.columns[i] if isinstance(i, (int, np.integer)) else i for i in cat
            ]
            for col in cat_col_names:
                df[col] = pd.Categorical(df[col])
            for col in df.columns:
                if col not in cat_col_names:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="raise")
                    except (ValueError, TypeError):
                        # Walk the column so float() produces the sklearn-expected
                        # message: "float() argument must be a string or a real number"
                        for v in df[col]:
                            float(v)
                        raise  # pragma: no cover
        if reset:
            self.feature_names_in_ = np.asarray(df.columns, dtype=object)
            self.n_features_in_ = df.shape[1]
        else:
            if df.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {df.shape[1]} features, but {type(self).__name__} is expecting "
                    f"{self.n_features_in_} features as input"
                )
            missing = set(self.feature_names_in_) - set(df.columns)
            if missing:
                raise ValueError(f"X is missing fitted columns: {sorted(missing)}")
            df = df[list(self.feature_names_in_)]
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] > 0 and np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("X contains infinity")
        return df

    def _build_core(self, task: str) -> FlagCoreModule:
        return FlagCoreModule(
            task=task,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            quantile_step=self.quantile_step,
            min_support=self.min_support,
            fdr_alpha=self.fdr_alpha,
            effect_size=getattr(self, "effect_size", "risk_difference"),
            missing=self.missing,
        )

    def transform(self, X):
        """Return Z(X) sparse matrix (one column per discovered basis)."""
        check_is_fitted(self, "core_")
        return self.core_.transform(self._to_frame(X, reset=False))

    def export_rules(self) -> pd.DataFrame:
        check_is_fitted(self, "head_")
        from .inspection import export_rules as _export
        return _export(self)

    def explain(self, X) -> pd.DataFrame:
        check_is_fitted(self, "head_")
        from .inspection import explain as _explain
        return _explain(self, X)


class FlagGAMClassifier(ClassifierMixin, _BaseFlagGAM):
    """Rule-basis generalized additive model classifier (sklearn contract)."""

    def __init__(
        self,
        task="auto",
        quantile_low=(0.05, 0.45),
        quantile_high=(0.55, 0.95),
        quantile_step=0.05,
        min_support="auto",
        fdr_alpha=0.05,
        effect_size="risk_difference",
        representation="full",
        feature_weighting=None,
        head="additive",
        flexible_estimator=None,
        C=1.0,
        missing="no_evidence",
        monotonic_constraints=None,
        categorical_features=None,
        random_state=None,
    ) -> None:
        self.task = task
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.quantile_step = quantile_step
        self.min_support = min_support
        self.fdr_alpha = fdr_alpha
        self.effect_size = effect_size
        self.representation = representation
        self.feature_weighting = feature_weighting
        self.head = head
        self.flexible_estimator = flexible_estimator
        self.C = C
        self.missing = missing
        self.monotonic_constraints = monotonic_constraints
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(self, X, y):
        """Discover flag bases, fit head on Z(X)."""
        df = self._to_frame(X, reset=True)
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        y = np.asarray(y)
        if len(df) != len(y):
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"[{len(df)}, {len(y)}]"
            )
        check_classification_targets(y)
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        y_enc = self.label_encoder_.transform(y)

        task = self.task
        if task == "auto":
            task = "binary" if len(self.classes_) == 2 else "multiclass"

        if self.monotonic_constraints is not None:
            raise NotImplementedError("monotonic constraints ship in the extensions plan")
        if self.head == "flexible" and self.flexible_estimator is None:
            raise ValueError("head='flexible' requires flexible_estimator")

        self.core_ = self._build_core(task).fit(df, y_enc)
        Z = self.core_.transform(df)

        self.feature_weights_ = None
        H = Z
        if self.representation == "compact":
            if self.feature_weighting == "auto":
                self.feature_weights_ = feature_weights(
                    df, y_enc, task, self.core_.numerical_features_
                )
            H = compact_scores(
                Z,
                self.core_.bases_,
                np.arange(len(self.classes_)),
                self.feature_weights_,
            )

        if self.head == "additive":
            self.head_ = AdditiveHead(task, C=self.C, random_state=self.random_state)
        else:
            self.head_ = FlexibleHead(
                self.flexible_estimator, task, random_state=self.random_state
            )
        self.head_.fit(H, y_enc)
        return self

    def _head_input(self, X):
        """Convert X to DataFrame, transform to Z, optionally compact."""
        Z = self.core_.transform(self._to_frame(X, reset=False))
        if self.representation == "compact":
            return compact_scores(
                Z,
                self.core_.bases_,
                np.arange(len(self.classes_)),
                self.feature_weights_,
            )
        return Z

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "head_")
        return self.head_.predict_proba(self._head_input(X))

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags


class FlagGAMRegressor(RegressorMixin, _BaseFlagGAM):
    """Rule-basis generalized additive model regressor (sklearn contract)."""

    def __init__(
        self,
        quantile_low=(0.05, 0.45),
        quantile_high=(0.55, 0.95),
        quantile_step=0.05,
        min_support="auto",
        fdr_alpha=0.05,
        head="additive",
        flexible_estimator=None,
        alpha=1.0,
        missing="no_evidence",
        monotonic_constraints=None,
        categorical_features=None,
        random_state=None,
    ) -> None:
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.quantile_step = quantile_step
        self.min_support = min_support
        self.fdr_alpha = fdr_alpha
        self.head = head
        self.flexible_estimator = flexible_estimator
        self.alpha = alpha
        self.missing = missing
        self.monotonic_constraints = monotonic_constraints
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(self, X, y):
        """Discover flag bases, fit regression head on Z(X)."""
        df = self._to_frame(X, reset=True)
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        y = np.asarray(y)
        if y.dtype.kind == "c":
            raise ValueError(f"Complex data not supported\n{y}\n")
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        y = y.astype(float).ravel()
        if len(df) != len(y):
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"[{len(df)}, {len(y)}]"
            )
        if self.monotonic_constraints is not None:
            raise NotImplementedError("monotonic constraints ship in the extensions plan")
        if self.head == "flexible" and self.flexible_estimator is None:
            raise ValueError("head='flexible' requires flexible_estimator")
        self.core_ = self._build_core("regression").fit(df, y)
        Z = self.core_.transform(df)
        if self.head == "additive":
            self.head_ = AdditiveHead(
                "regression", alpha=self.alpha, random_state=self.random_state
            )
        else:
            self.head_ = FlexibleHead(
                self.flexible_estimator, "regression", random_state=self.random_state
            )
        self.head_.fit(Z, y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "head_")
        return np.asarray(self.head_.predict(self.transform(X)), dtype=float)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = False
        return tags
