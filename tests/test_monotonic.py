"""Tests for flaggam.monotonic (original extension, not in Zhao & Welsch 2026)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from flaggam import FlagGAMClassifier, FlagGAMRegressor
from flaggam.monotonic import bounds_for_bases


def _risky_young(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Risk strictly decreasing in age; second free feature."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    other = rng.normal(0, 1, n)
    y = (rng.uniform(size=n) < expit(2.0 - 0.06 * age + 0.3 * other)).astype(int)
    return pd.DataFrame({"age": age, "other": other}), y


def _grid(feature_vals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"age": feature_vals, "other": np.zeros(len(feature_vals))})


def test_constrained_fit_is_monotone() -> None:
    X, y = _risky_young()
    clf = FlagGAMClassifier(monotonic_constraints={"age": -1}, random_state=0).fit(X, y)
    p = clf.predict_proba(_grid(np.linspace(18, 80, 200)))[:, 1]
    assert (np.diff(p) <= 1e-10).all(), "PD must be non-increasing in age"


def test_constraint_signs_on_coefficients() -> None:
    X, y = _risky_young()
    clf = FlagGAMClassifier(monotonic_constraints={"age": -1}, random_state=0).fit(X, y)
    coefs = np.ravel(clf.head_.coef_)
    for basis, theta in zip(clf.core_.bases_, coefs, strict=False):
        if basis.feature != "age":
            continue
        if basis.kind in ("threshold_low", "hinge_low"):
            assert theta >= -1e-9
        elif basis.kind in ("threshold_high", "hinge_high"):
            assert theta <= 1e-9


def test_unconstrained_matches_default_head() -> None:
    X, y = _risky_young(n=800)
    a = FlagGAMClassifier(random_state=0).fit(X, y)
    b = FlagGAMClassifier(monotonic_constraints={"age": 0}, random_state=0).fit(X, y)
    # all-free bounds: same rule basis, close predictions (different optimizers)
    assert [bb.name for bb in a.core_.bases_] == [bb.name for bb in b.core_.bases_]
    pa = a.predict_proba(X)[:, 1]
    pb = b.predict_proba(X)[:, 1]
    assert np.abs(pa - pb).mean() < 0.02


def test_regression_monotone() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 3000)
    other = rng.normal(size=3000)
    y = 2.0 * (x > 7) - 1.5 * (x < 3) + 0.2 * other + rng.normal(0, 0.3, 3000)
    X = pd.DataFrame({"x": x, "other": other})
    reg = FlagGAMRegressor(monotonic_constraints={"x": 1}, random_state=0).fit(X, y)
    grid = pd.DataFrame({"x": np.linspace(0, 10, 200), "other": np.zeros(200)})
    assert (np.diff(reg.predict(grid)) >= -1e-9).all()


def test_validation_errors() -> None:
    X, y = _risky_young(n=400)
    with pytest.raises(ValueError, match="unknown feature"):
        FlagGAMClassifier(monotonic_constraints={"nope": 1}).fit(X, y)
    with pytest.raises(ValueError, match="values"):
        FlagGAMClassifier(monotonic_constraints={"age": 2}).fit(X, y)
    with pytest.raises(ValueError, match="compact"):
        FlagGAMClassifier(
            representation="compact", monotonic_constraints={"age": -1}
        ).fit(X, y)
    with pytest.raises(ValueError, match="additive"):
        from sklearn.ensemble import RandomForestClassifier

        FlagGAMClassifier(
            head="flexible",
            flexible_estimator=RandomForestClassifier(n_estimators=5),
            monotonic_constraints={"age": -1},
        ).fit(X, y)
    y3 = np.arange(400) % 3
    with pytest.raises(NotImplementedError):
        FlagGAMClassifier(monotonic_constraints={"age": -1}).fit(X, y3)


def test_bounds_for_bases_table() -> None:
    from flaggam.bases import CategoryBasis, ThresholdBasis

    lo = ThresholdBasis(feature="age", support=10, effect_size=0.1, p_value=0.01,
                        p_adj=0.01, cutoff=30.0, side="low")
    hi = ThresholdBasis(feature="age", support=10, effect_size=0.1, p_value=0.01,
                        p_adj=0.01, cutoff=60.0, side="high")
    cat = CategoryBasis(feature="purpose", support=10, effect_size=0.1, p_value=0.01,
                        p_adj=0.01, level="edu")
    assert bounds_for_bases([lo, hi, cat], {"age": 1}) == [
        (None, 0.0), (0.0, None), (None, None),
    ]
    assert bounds_for_bases([lo, hi, cat], {"age": -1}) == [
        (0.0, None), (None, 0.0), (None, None),
    ]


def test_export_rules_and_explain_on_constrained_fit() -> None:
    """Regression: monotonic-constrained heads must yield real weights on export."""
    X, y = _risky_young(n=800)
    clf = FlagGAMClassifier(monotonic_constraints={"age": -1}, random_state=0).fit(X, y)
    rules = clf.export_rules()
    assert len(rules) > 0
    assert rules["weight"].notna().all(), "constrained head must yield real weights"
    exp = clf.explain(X.head(3))
    assert not exp.empty
