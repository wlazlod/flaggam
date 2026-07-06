"""FlagGAMClassifier: sklearn-compatible estimator for rule-basis GAMs."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .core import FlagCoreModule
from .heads import AdditiveHead, FlexibleHead
from .weighting import compact_scores, feature_weights


class _BaseFlagGAM(BaseEstimator):
    """Shared X conversion, core/head assembly, transform."""

    def _to_frame(self, X, reset: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
            cat = self.categorical_features or []
            cat_col_names = [
                df.columns[i] if isinstance(i, (int, np.integer)) else i for i in cat
            ]
            for col in cat_col_names:
                df[col] = pd.Categorical(df[col])
            for col in df.columns:
                if col not in cat_col_names:
                    df[col] = pd.to_numeric(df[col], errors="raise")
        if reset:
            self.feature_names_in_ = np.asarray(df.columns, dtype=object)
            self.n_features_in_ = df.shape[1]
        else:
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
            effect_size=self.effect_size,
            missing=self.missing,
        )

    def transform(self, X):
        """Return Z(X) sparse matrix (one column per discovered basis)."""
        check_is_fitted(self, "core_")
        return self.core_.transform(self._to_frame(X, reset=False))

    def export_rules(self) -> pd.DataFrame:        # implemented in Task 10
        raise NotImplementedError("implemented in inspection task")

    def explain(self, X) -> pd.DataFrame:          # implemented in Task 10
        raise NotImplementedError("implemented in inspection task")


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
