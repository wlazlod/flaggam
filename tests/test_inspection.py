import numpy as np
import pandas as pd
import pytest

from flaggam import FlagGAMClassifier


@pytest.fixture()
def fitted() -> FlagGAMClassifier:
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(40, 10, n)
    purpose = rng.choice(["car", "tv", "edu"], n)
    logit = -1.5 + 2.0 * (age <= 30) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "purpose": pd.Categorical(purpose)})
    return FlagGAMClassifier(random_state=0).fit(X, y), X


def test_export_rules_columns_and_weights(fitted) -> None:
    clf, _ = fitted
    rules = clf.export_rules()
    for col in [
        "feature",
        "kind",
        "rule",
        "cutoff",
        "level",
        "support",
        "effect_size",
        "p_value",
        "p_adj",
        "weight",
    ]:
        assert col in rules.columns
    assert len(rules) == len(clf.core_.bases_)
    assert rules["weight"].notna().all()


def test_explain_reason_codes(fitted) -> None:
    clf, X = fitted
    codes = clf.explain(X.head(5))
    assert set(codes.columns) >= {"row", "feature", "rule", "value", "contribution"}
    assert set(codes["row"]) <= set(range(5))
    # A young 'edu' row must fire the age rule with positive contribution.
    x_young = pd.DataFrame({"age": [20.0], "purpose": pd.Categorical(["edu"])})
    c = clf.explain(x_young)
    fired = c[c.feature == "age"]
    assert len(fired) == 1 and fired["contribution"].iloc[0] > 0
    # Contributions + intercept reconstruct the decision logit.
    logit = c["contribution"].sum()
    proba = clf.predict_proba(x_young)[0, 1]
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(proba, abs=1e-6)


@pytest.fixture()
def fitted_compact() -> tuple:
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(40, 10, n)
    purpose = rng.choice(["car", "tv", "edu"], n)
    logit = -1.5 + 2.0 * (age <= 30) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "purpose": pd.Categorical(purpose)})
    clf = FlagGAMClassifier(
        representation="compact", feature_weighting="auto", random_state=0
    ).fit(X, y)
    return clf, X


def test_export_rules_rejects_compact(fitted_compact) -> None:
    clf, _ = fitted_compact
    with pytest.raises(ValueError, match="representation='full'"):
        clf.export_rules()


def test_explain_rejects_compact(fitted_compact) -> None:
    clf, X = fitted_compact
    with pytest.raises(ValueError, match="representation='full'"):
        clf.explain(X.head(1))


def test_explain_rejects_flexible_head(fitted) -> None:
    _, X = fitted
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    y = (rng.uniform(size=len(X)) < 0.3).astype(int)
    clf = FlagGAMClassifier(
        head="flexible",
        flexible_estimator=RandomForestClassifier(n_estimators=10),
        random_state=0,
    ).fit(X, y)
    with pytest.raises(ValueError):
        clf.explain(X.head(1))
