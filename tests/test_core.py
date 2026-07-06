import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flaggam.core import FlagCoreModule


@pytest.fixture()
def clf_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(40, 10, n)
    noise = rng.normal(0, 1, n)
    purpose = rng.choice(["car", "tv", "edu"], n, p=[0.4, 0.4, 0.2])
    # Signal: low age and purpose=='edu' raise the positive rate.
    logit = -1.5 + 2.0 * (age <= np.quantile(age, 0.2)) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "noise": noise, "purpose": pd.Categorical(purpose)})
    return X, y


def test_discovers_expected_rules(clf_data) -> None:
    X, y = clf_data
    core = FlagCoreModule(task="binary").fit(X, y)
    meta = core.metadata()
    kinds_age = set(meta.loc[meta.feature == "age", "kind"])
    assert "threshold_low" in kinds_age                       # low-age tail found
    assert (meta.feature == "purpose").sum() >= 1             # edu level found
    levels = set(meta.loc[meta.feature == "purpose", "level"])
    assert "edu" in levels
    # At most one cutoff per side per numerical feature.
    for side in ("threshold_low", "threshold_high"):
        assert ((meta.kind == side) & (meta.feature == "age")).sum() <= 1


def test_pure_noise_feature_mostly_silent(clf_data) -> None:
    X, y = clf_data
    core = FlagCoreModule(task="binary").fit(X, y)
    meta = core.metadata()
    assert (meta.feature == "noise").sum() <= 1  # FDR keeps false flags rare


def test_transform_shape_sparse_and_missing(clf_data) -> None:
    X, y = clf_data
    core = FlagCoreModule(task="binary").fit(X, y)
    Z = core.transform(X)
    assert sparse.issparse(Z) and Z.shape == (len(X), len(core.bases_))
    Xm = X.copy()
    Xm.loc[:, "age"] = np.nan
    Zm = core.transform(Xm)
    age_cols = [i for i, b in enumerate(core.bases_) if b.feature == "age"]
    assert Zm[:, age_cols].toarray().sum() == 0.0  # no evidence from missing


def test_min_support_auto(clf_data) -> None:
    X, y = clf_data
    core = FlagCoreModule(task="binary").fit(X, y)
    assert core.min_support_ == 40  # ceil(0.02 * 2000)


def test_multiclass_runs(clf_data) -> None:
    X, y = clf_data
    rng = np.random.default_rng(1)
    y3 = np.where(rng.uniform(size=len(y)) < 0.2, 2, y)
    core = FlagCoreModule(task="multiclass").fit(X, y3)
    assert isinstance(core.bases_, list)
