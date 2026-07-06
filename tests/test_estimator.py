import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from flaggam import FlagGAMClassifier


@pytest.fixture()
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(40, 10, n)
    income = rng.normal(50, 15, n)
    purpose = rng.choice(["car", "tv", "edu"], n)
    logit = -1.5 + 2.0 * (age <= 30) - 1.0 * (income >= 70) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(
        {"age": age, "income": income, "purpose": pd.Categorical(purpose)}
    )
    return X, y


def test_fit_predict_beats_chance(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # fixture Bayes AUC ~= 0.748; 0.72 pins "beats chance" without over-fitting the grid
    assert roc_auc_score(y, proba[:, 1]) > 0.72
    assert set(clf.predict(X)) <= {0, 1}


def test_transform_returns_z(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(random_state=0).fit(X, y)
    Z = clf.transform(X)
    assert Z.shape == (len(X), len(clf.core_.bases_))


def test_ndarray_with_categorical_mask(data) -> None:
    X, y = data
    Xn = np.column_stack([X["age"], X["income"], X["purpose"].astype(str)])
    clf = FlagGAMClassifier(categorical_features=[2], random_state=0)
    clf.fit(Xn.astype(object), y)
    assert clf.n_features_in_ == 3
    assert clf.predict_proba(Xn.astype(object)).shape == (len(y), 2)


def test_compact_representation(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(
        representation="compact", feature_weighting="auto", random_state=0
    ).fit(X, y)
    assert roc_auc_score(y, clf.predict_proba(X)[:, 1]) > 0.65


def test_flexible_head(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(
        head="flexible",
        flexible_estimator=RandomForestClassifier(n_estimators=50),
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X).shape == (len(X), 2)
    with pytest.raises(ValueError):
        FlagGAMClassifier(head="flexible").fit(X, y)


def test_clone_and_determinism(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(random_state=0)
    p1 = clf.fit(X, y).predict_proba(X)
    p2 = clone(clf).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_missing_rows_no_evidence(data) -> None:
    X, y = data
    clf = FlagGAMClassifier(random_state=0).fit(X, y)
    X_missing = X.copy()
    X_missing.iloc[0, :] = [np.nan, np.nan, None]
    Z = clf.transform(X_missing)
    assert Z[0].toarray().sum() == 0.0          # spec §14: zero contribution
    assert clf.predict_proba(X_missing).shape == (len(X), 2)


def test_string_labels(data) -> None:
    X, y = data
    ys = np.where(y == 1, "bad", "good")
    clf = FlagGAMClassifier(random_state=0).fit(X, ys)
    assert set(clf.predict(X)) <= {"bad", "good"}
    assert list(clf.classes_) == ["bad", "good"]
