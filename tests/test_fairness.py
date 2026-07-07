"""Tests for flaggam.fairness (original extension, not in Zhao & Welsch 2026)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from flaggam import FlagGAMClassifier
from flaggam.fairness import ProxyAudit, group_metrics


def test_group_metrics_hand_computed() -> None:
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.1, 0.3])
    a = np.array(["m", "m", "m", "m", "f", "f", "f", "f"])
    out = group_metrics(y, p, a, threshold=0.5)
    by = out["by_group"]
    assert set(by.index) == {"m", "f"}
    # selection at 0.5: m -> [.9,.8] = 2/4; f -> [.6,.7] = 2/4
    assert out["gaps"]["demographic_parity_diff"] == pytest.approx(0.0)
    # TPR: m positives {.9,.8} both selected = 1.0; f positives {.6,.7} both = 1.0
    assert out["gaps"]["equal_opportunity_diff"] == pytest.approx(0.0)
    assert {"n", "base_rate", "mean_predicted", "selection_rate", "tpr", "auroc", "ece"} <= set(
        by.columns
    )


def test_group_metrics_detects_disparity() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.choice(["g0", "g1"], n)
    p = np.where(a == "g1", 0.7, 0.3) + rng.normal(0, 0.05, n)
    p = np.clip(p, 0.01, 0.99)
    y = (rng.uniform(size=n) < p).astype(int)
    gaps = group_metrics(y, p, a)["gaps"]
    assert gaps["demographic_parity_diff"] > 0.3


def test_group_metrics_binary_only() -> None:
    with pytest.raises(ValueError, match="binary"):
        group_metrics(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]), np.array(["a", "b", "a"]))


def _proxy_data(n: int = 3000, seed: int = 0):
    """Feature `zip3` is a near-perfect proxy for protected `A`; `income` is clean."""
    rng = np.random.default_rng(seed)
    A = rng.choice(["p0", "p1"], n)
    zip3 = np.where(
        rng.uniform(size=n) < 0.95, np.where(A == "p1", "north", "south"),
        np.where(A == "p1", "south", "north"),
    )
    income = rng.normal(50, 15, n)
    logit = -0.5 + 1.2 * (zip3 == "north") - 0.03 * (income - 50)
    y = (rng.uniform(size=n) < expit(logit)).astype(int)
    X = pd.DataFrame(
        {"zip3": pd.Categorical(zip3), "income": income}
    )
    return X, y, A


def test_proxy_audit_flags_proxy_feature() -> None:
    X, y, A = _proxy_data()
    est = FlagGAMClassifier(random_state=0).fit(X, y)
    report = ProxyAudit(est).report(X, A, threshold=0.5)
    assert list(report.columns) == ["feature", "rule", "kind", "association", "method", "flagged"]
    assert (report["association"].values[:-1] >= report["association"].values[1:]).all()
    zip_rows = report[report.feature == "zip3"]
    income_rows = report[report.feature == "income"]
    assert len(zip_rows) and bool(zip_rows["flagged"].all())
    assert not income_rows["flagged"].any()


def test_drop_proxies_removes_and_reports_tradeoff() -> None:
    X, y, A = _proxy_data()
    est = FlagGAMClassifier(random_state=0).fit(X, y)
    new_est, trade = ProxyAudit(est).drop_proxies(X, y, A, threshold=0.5)
    assert all(b.feature != "zip3" for b in new_est.core_.bases_)
    assert est is not new_est and any(b.feature == "zip3" for b in est.core_.bases_)
    row = trade.iloc[0]
    assert row["n_dropped"] >= 1
    assert row["dp_diff_after"] <= row["dp_diff_before"] + 1e-9
    assert 0.5 <= row["auroc_after"] <= row["auroc_before"] + 0.02
    # dropped-proxy model still predicts
    assert new_est.predict_proba(X).shape == (len(X), 2)


def test_point_biserial_for_numeric_A() -> None:
    X, y, A = _proxy_data()
    A_num = (A == "p1").astype(float) + np.random.default_rng(1).normal(0, 0.01, len(A))
    est = FlagGAMClassifier(random_state=0).fit(X, y)
    report = ProxyAudit(est).report(X, A_num)
    assert (report["method"] == "point_biserial").all()
