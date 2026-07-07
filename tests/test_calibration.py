"""Tests for flaggam.calibration (original extension, not in Zhao & Welsch 2026)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from flaggam import FlagGAMClassifier
from flaggam.calibration import (
    CalibratedFlagGAM,
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    reliability_curve,
)


def _probs(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True PD p, labels y, and an overconfident distortion q of p."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    q = expit(2.5 * np.log(p / (1 - p)))  # overconfident: pushed toward 0/1
    return p, y, q


def _flaggam_data(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    purpose = rng.choice(["car", "tv", "edu"], n)
    logit = -1.5 + 2.0 * (age <= 30) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < expit(logit)).astype(int)
    X = pd.DataFrame({"age": age, "purpose": pd.Categorical(purpose)})
    return X, y


def test_reliability_curve_shape_and_calibrated_case() -> None:
    p, y, _ = _probs()
    df = reliability_curve(y, p, n_bins=10)
    assert list(df.columns) == [
        "bin_lower", "bin_upper", "mean_predicted", "fraction_positive", "count",
    ]
    assert (df["count"] > 0).all()
    # calibrated probabilities: bin means track observed fractions
    assert (df["mean_predicted"] - df["fraction_positive"]).abs().max() < 0.05


def test_ece_orders_calibrated_below_overconfident() -> None:
    p, y, q = _probs()
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, q) > 2 * expected_calibration_error(y, p)
    # quantile strategy also works and is finite
    assert np.isfinite(expected_calibration_error(y, q, strategy="quantile"))


def test_brier_matches_sklearn() -> None:
    from sklearn.metrics import brier_score_loss

    p, y, _ = _probs(n=1000)
    assert brier_score(y, p) == pytest.approx(brier_score_loss(y, p))


def test_calibration_in_the_large() -> None:
    p, y, _ = _probs()
    out = calibration_in_the_large(y, p)
    assert set(out) == {"mean_predicted", "observed_rate", "difference"}
    assert out["difference"] == pytest.approx(
        out["mean_predicted"] - out["observed_rate"]
    )
    assert abs(out["difference"]) < 0.01


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrated_flaggam_cross_fit(method: str) -> None:
    X, y = _flaggam_data()
    cal = CalibratedFlagGAM(FlagGAMClassifier(random_state=0), method=method, cv=3)
    cal.fit(X, y)
    proba = cal.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(np.unique(cal.predict(X))) <= set(cal.classes_)


def test_calibrated_flaggam_prefit() -> None:
    X, y = _flaggam_data()
    fitted = FlagGAMClassifier(random_state=0).fit(X[:1500], y[:1500])
    cal = CalibratedFlagGAM(fitted, method="platt", cv="prefit").fit(X[1500:], y[1500:])
    assert cal.predict_proba(X[1500:]).shape == (500, 2)


def test_base_rate_hits_target() -> None:
    X, y = _flaggam_data()
    cal = CalibratedFlagGAM(
        FlagGAMClassifier(random_state=0), method="base_rate", cv=3, target_rate=0.10
    ).fit(X, y)
    assert cal.predict_proba(X)[:, 1].mean() == pytest.approx(0.10, abs=0.01)


def test_base_rate_requires_target() -> None:
    X, y = _flaggam_data(n=400)
    with pytest.raises(ValueError, match="target_rate"):
        CalibratedFlagGAM(FlagGAMClassifier(), method="base_rate", cv=3).fit(X, y)


def test_calibrated_flaggam_isotonic_string_labels() -> None:
    X, y = _flaggam_data()
    # np.unique sorts "bad" < "good", so class-1 (positive) is "good".
    y_str = np.where(y == 1, "bad", "good")
    cal = CalibratedFlagGAM(
        FlagGAMClassifier(random_state=0), method="isotonic", cv=3
    ).fit(X, y_str)
    proba = cal.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(cal.predict(X.head())) <= {"bad", "good"}


def test_multiclass_rejected() -> None:
    X, _ = _flaggam_data(n=300)
    y3 = np.arange(300) % 3
    with pytest.raises(ValueError, match="binary"):
        CalibratedFlagGAM(FlagGAMClassifier(), cv=3).fit(X, y3)
