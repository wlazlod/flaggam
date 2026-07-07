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
# Network loaders (bodies implemented in Task 2)
# ---------------------------------------------------------------------------


def load_pima() -> tuple[pd.DataFrame, pd.Series]:
    """Pima Indians Diabetes (768 x 8, binary; positive=diabetic).

    Source: https://www.openml.org/d/37 (Pima Indians Diabetes).
    License: CC0 per OpenML page — verify before redistribution.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_heart() -> tuple[pd.DataFrame, pd.Series]:
    """Heart Disease Cleveland (303 x 13, binary; positive=disease present).

    Source: https://archive.ics.uci.edu/dataset/45 (Heart Disease, Cleveland).
    License: CC BY 4.0 — verify before redistribution. Target: num > 0 -> 1.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Statlog German Credit (1000 x 20, binary; positive=bad credit).

    Source: https://archive.ics.uci.edu/dataset/144 (Statlog German Credit).
    License: CC BY 4.0 — verify before redistribution. Target: bad credit = 1.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_adult() -> tuple[pd.DataFrame, pd.Series]:
    """Adult / Census Income (48842 rows, binary; positive=income >50K).

    Source: https://archive.ics.uci.edu/dataset/2 (Adult / Census Income).
    License: CC BY 4.0 — verify before redistribution. Target: income >50K = 1.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_bank_marketing() -> tuple[pd.DataFrame, pd.Series]:
    """Bank Marketing (45211 rows, binary; positive=subscribed term deposit).

    Source: https://archive.ics.uci.edu/dataset/222 (Bank Marketing).
    License: CC BY 4.0 — verify before redistribution.
    The post-call 'duration' column is always dropped (spec §9). Target: y == "yes" = 1.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_ames() -> tuple[pd.DataFrame, pd.Series]:
    """Ames Housing (2930 rows, regression; target = log(SalePrice)).

    Source: https://www.openml.org/search?type=data&q=ames+housing (Ames Housing).
    License: public domain (De Cock, 2011) — verify before redistribution.
    Target is log(SalePrice); RMSE is reported on the log scale.
    """
    raise RuntimeError("network fetch implemented in Task 2")


def load_wine_white() -> tuple[pd.DataFrame, pd.Series]:
    """Wine Quality white subset (4898 x 11, regression on quality score).

    Source: https://archive.ics.uci.edu/dataset/186 (Wine Quality, white subset).
    License: CC BY 4.0 — verify before redistribution.
    """
    raise RuntimeError("network fetch implemented in Task 2")


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
