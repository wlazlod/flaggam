"""Benchmark dataset loaders with local caching and license notes.

Each loader returns (X, y): X a DataFrame whose categorical features are
pd.Categorical dtype, y a Series (binary targets are int 0/1 with the
"event"/positive class = 1). Data is fetched at runtime and cached under
data_dir(); raw files are never committed to the repository. Verify each
dataset's license on its source page before redistributing anything.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str  # "binary" | "regression"
    loader: Callable[[], tuple[pd.DataFrame, pd.Series]]


def data_dir() -> Path:
    """Cache directory: $FLAGGAM_DATA_DIR or ~/.cache/flaggam (created)."""
    d = Path(os.environ.get("FLAGGAM_DATA_DIR", Path.home() / ".cache" / "flaggam"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached(name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Parquet cache under data_dir(); fetch once, reuse thereafter."""
    path = data_dir() / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    logger.info("cached %s: %s rows to %s", name, len(df), path)
    return df


# ---------------------------------------------------------------------------
# Offline loaders (bundled via scikit-learn)
# ---------------------------------------------------------------------------


def load_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Diagnostic Breast Cancer (569 x 30, binary; positive=malignant).

    Source: https://archive.ics.uci.edu/dataset/17 (bundled via scikit-learn).
    License: CC BY 4.0 per UCI page — verify before redistribution.
    """
    from sklearn.datasets import load_breast_cancer as _load

    b = _load(as_frame=True)
    X = b.data.astype(float)
    y = pd.Series((b.target == 0).astype(int), name="malignant")  # sklearn: 0=malignant
    return X, y


def load_california() -> tuple[pd.DataFrame, pd.Series]:
    """California Housing (20640 x 8, regression on median house value).

    Source: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
    License: public domain (US Census derived) — verify before redistribution.
    """
    from sklearn.datasets import fetch_california_housing

    b = fetch_california_housing(as_frame=True, data_home=str(data_dir()))
    return b.data.astype(float), b.target.rename("median_house_value")


# ---------------------------------------------------------------------------
# Network loaders
# ---------------------------------------------------------------------------


def load_pima() -> tuple[pd.DataFrame, pd.Series]:
    """Pima Indians Diabetes (768 x 8, binary; positive=diabetic).

    Source: https://www.openml.org/d/37 (Pima Indians Diabetes, dataset id=37).
    License: CC0 per OpenML page — verify before redistribution.
    Observed variant: OpenML dataset 37 (pima_diabetes), 768 rows, 8 features.
    Short attribute names renamed: plas→glucose, pres→blood_pressure,
    skin→skin_thickness, insu→insulin, mass→bmi, pedi→diabetes_pedigree.

    DECISIONS 17: Physiologically impossible zeros (glucose, blood_pressure,
    skin_thickness, insulin, bmi) are replaced with NaN per clinical convention
    and FlagGAM's native-missing design.
    """

    def _fetch() -> pd.DataFrame:
        import openml

        dataset = openml.datasets.get_dataset(37)
        X_raw, y_raw, _, _ = dataset.get_data(target="class")
        assert isinstance(X_raw, pd.DataFrame)
        _rename = {
            "plas": "glucose",
            "pres": "blood_pressure",
            "skin": "skin_thickness",
            "insu": "insulin",
            "mass": "bmi",
            "pedi": "diabetes_pedigree",
        }
        X_raw = X_raw.rename(columns={k: v for k, v in _rename.items() if k in X_raw.columns})
        X_raw["_target"] = y_raw
        return X_raw

    df = _cached("pima", _fetch)
    y = (df["_target"] == "tested_positive").astype(int)
    y.name = "diabetic"
    X = df.drop("_target", axis=1).copy()

    _zero_cols = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for col in _zero_cols:
        if col in X.columns:
            X[col] = X[col].replace(0.0, float("nan"))

    return X, y


def load_heart() -> tuple[pd.DataFrame, pd.Series]:
    """Heart Disease Cleveland (303 x 13, binary; positive=disease present).

    Source: https://archive.ics.uci.edu/dataset/45 (Heart Disease, Cleveland).
    License: CC BY 4.0 — verify before redistribution. Target: num > 0 → 1.
    Observed variant: UCI id=45, Cleveland 303 rows, 13 features (all numeric).
    ca/thal have missing values (kept as NaN).
    Categorical features (originally coded as integers): cp, restecg, slope,
    thal, sex, fbs, exang.
    """

    def _fetch() -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        X["_target"] = y.iloc[:, 0]
        return X

    df = _cached("heart", _fetch)
    y = (df["_target"] > 0).astype(int)
    y.name = "heart_disease"
    X = df.drop("_target", axis=1).copy()

    _cat_cols = ["cp", "restecg", "slope", "thal", "sex", "fbs", "exang"]
    for col in _cat_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col])

    return X, y


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Statlog German Credit (1000 x 20, binary; positive=bad credit).

    Source: https://archive.ics.uci.edu/dataset/144 (Statlog German Credit).
    License: CC BY 4.0 — verify before redistribution. Target: bad credit = 1.
    Observed variant: UCI id=144, 1000 rows, 20 features (Attribute* columns).
    Target: class==2 → bad credit=1 (per UCI docs, 1=good, 2=bad).
    """

    def _fetch() -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=144)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        X["_target"] = y.iloc[:, 0]
        return X

    df = _cached("german_credit", _fetch)
    y = (df["_target"] == 2).astype(int)
    y.name = "bad_credit"
    X = df.drop("_target", axis=1).copy()

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = pd.Categorical(X[col])

    return X, y


def load_adult() -> tuple[pd.DataFrame, pd.Series]:
    """Adult / Census Income (~48842 rows, binary; positive=income >50K).

    Source: https://archive.ics.uci.edu/dataset/2 (Adult / Census Income).
    License: CC BY 4.0 — verify before redistribution. Target: income >50K = 1.
    Observed variant: UCI id=2, 48842 rows, 14 features.
    Whitespace stripped; '?' replaced with NaN; string cols → pd.Categorical.
    """

    def _fetch() -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=2)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = X[col].str.strip()
        y_col = y.iloc[:, 0]
        X["_target"] = y_col.str.strip() if hasattr(y_col, "str") else y_col
        return X

    df = _cached("adult", _fetch)
    df = df.dropna(subset=["_target"])
    y_raw = df["_target"].str.strip()
    X = df.drop("_target", axis=1).copy()

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].replace("?", float("nan"))
        X[col] = pd.Categorical(X[col])

    y = y_raw.str.contains(">50K", na=False).astype(int)
    y.name = "high_income"

    return X, y


def load_bank_marketing() -> tuple[pd.DataFrame, pd.Series]:
    """Bank Marketing (45211 rows, binary; positive=subscribed term deposit).

    Source: https://archive.ics.uci.edu/dataset/222 (Bank Marketing).
    License: CC BY 4.0 — verify before redistribution.
    Observed variant: UCI id=222, 45211 rows, 16 original features.
    The post-call 'duration' column is always dropped (UCI recommendation, spec §9).
    'unknown' is kept as a regular category level (FlagGAM/UFA design: treat
    as explicit category, not NaN, to preserve structure of missingness).
    """

    def _fetch() -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=222)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        X.drop(columns=["duration"], errors="ignore", inplace=True)
        X["_target"] = y.iloc[:, 0]
        return X

    df = _cached("bank_marketing", _fetch)
    y = (df["_target"] == "yes").astype(int)
    y.name = "subscribed"
    X = df.drop("_target", axis=1).copy()

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = pd.Categorical(X[col])

    return X, y


def load_ames() -> tuple[pd.DataFrame, pd.Series]:
    """Ames Housing (2930 rows, regression; target = log(Sale_Price)).

    Source: https://www.openml.org/search?type=data&q=ames+housing (Ames Housing).
    License: public domain (De Cock, 2011) — verify before redistribution.
    Observed variant: OpenML 'ames_housing' version 1 (id=43926), 2930 rows,
    80 features. OpenML returns features already as ordered CategoricalDtype.
    Target is log(Sale_Price); RMSE is reported on the log scale.
    """

    def _fetch() -> pd.DataFrame:
        import openml

        try:
            dataset = openml.datasets.get_dataset("ames_housing", version=1)
        except Exception:
            dl = openml.datasets.list_datasets(data_name="ames_housing", output_format="dataframe")
            did = int(dl.index[0]) if hasattr(dl, "index") else next(iter(dl))
            dataset = openml.datasets.get_dataset(did)

        target_col = dataset.default_target_attribute or "Sale_Price"
        X_raw, y_raw, _, _ = dataset.get_data(target=target_col)
        assert isinstance(X_raw, pd.DataFrame)
        X_raw["_target"] = y_raw
        return X_raw

    df = _cached("ames", _fetch)
    y = np.log(df["_target"].astype(float)).rename("log_sale_price")
    X = df.drop("_target", axis=1).copy()

    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = pd.Categorical(X[col])

    return X, y


def load_wine_white() -> tuple[pd.DataFrame, pd.Series]:
    """Wine Quality white subset (4898 x 11, regression on quality score).

    Source: https://archive.ics.uci.edu/dataset/186 (Wine Quality, white subset).
    License: CC BY 4.0 — verify before redistribution.
    Observed variant: UCI id=186 (combined red+white, 6497 rows); 'color' column
    found in ds.data.original — filtered to white wines → 4898 rows, 11 features.
    All features are float; no categorical columns.
    """

    def _fetch() -> pd.DataFrame:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=186)
        orig = dataset.data.original.copy()
        if "color" in orig.columns:
            orig = orig[orig["color"] == "white"].copy()
            orig.drop(columns=["color"], inplace=True)
        # orig now has 11 feature cols + quality (target)
        return orig

    df = _cached("wine_white", _fetch)
    y = df["quality"].astype(float)
    y.name = "quality"
    X = df.drop("quality", axis=1).copy()

    return X, y


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

CLASSIFICATION: dict[str, DatasetSpec] = {
    "pima": DatasetSpec("pima", "binary", load_pima),
    "breast_cancer": DatasetSpec("breast_cancer", "binary", load_breast_cancer),
    "heart": DatasetSpec("heart", "binary", load_heart),
    "german_credit": DatasetSpec("german_credit", "binary", load_german_credit),
    "adult": DatasetSpec("adult", "binary", load_adult),
    "bank_marketing": DatasetSpec("bank_marketing", "binary", load_bank_marketing),
}

REGRESSION: dict[str, DatasetSpec] = {
    "ames": DatasetSpec("ames", "regression", load_ames),
    "california": DatasetSpec("california", "regression", load_california),
    "wine_white": DatasetSpec("wine_white", "regression", load_wine_white),
}
