"""Tests for flaggam.plots (original addition; matplotlib is an optional extra)."""

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import flaggam.plots as plots
from flaggam import FlagGAMClassifier, ProxyAudit, group_metrics


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """README-style synthetic: age + purpose classifier data."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    purpose = rng.choice(["car", "tv", "edu"], n)
    logit = -1.5 + 2.0 * (age <= 30) + 1.5 * (purpose == "edu")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "purpose": pd.Categorical(purpose)})
    return X, y


@pytest.fixture()
def fitted_clf() -> tuple[FlagGAMClassifier, pd.DataFrame, np.ndarray]:
    X, y = _synthetic()
    return FlagGAMClassifier(random_state=0).fit(X, y), X, y


# ---- plot_shape ----------------------------------------------------------


def test_plot_shape_numeric_returns_line(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    ax = plots.plot_shape(clf, "age")
    assert len(ax.lines) >= 1
    assert ax.get_xlabel() == "age"


def test_plot_shape_categorical_returns_bars(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    ax = plots.plot_shape(clf, "purpose")
    assert len(ax.patches) >= 1


def test_plot_shape_unknown_feature_lists_available(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    with pytest.raises(ValueError, match="features with bases"):
        plots.plot_shape(clf, "not_a_feature")


def test_plot_shape_rejects_compact_representation() -> None:
    X, y = _synthetic()
    clf = FlagGAMClassifier(
        representation="compact", feature_weighting="auto", random_state=0
    ).fit(X, y)
    with pytest.raises(ValueError, match="representation='full'"):
        plots.plot_shape(clf, "age")


def test_plot_shape_monotonic_constrained_is_non_increasing() -> None:
    """A -1 constraint on age must yield a non-increasing shape curve."""
    rng = np.random.default_rng(1)
    n = 3000
    age = rng.uniform(18, 80, n)
    other = rng.normal(size=n)
    logit = 2.0 - 0.06 * age + 0.3 * other
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"age": age, "other": other})
    clf = FlagGAMClassifier(monotonic_constraints={"age": -1}, random_state=0).fit(X, y)
    ax = plots.plot_shape(clf, "age")
    contribution = ax.lines[0].get_ydata()
    assert (np.diff(contribution) <= 1e-9).all()


# ---- plot_rule_importance -------------------------------------------------


def test_plot_rule_importance_bar_count(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    n_rules = len(clf.export_rules())
    ax = plots.plot_rule_importance(clf, top_n=2)
    assert len(ax.patches) == min(2, n_rules)
    ax2 = plots.plot_rule_importance(clf, top_n=100)
    assert len(ax2.patches) == min(100, n_rules)


# ---- plot_waterfall --------------------------------------------------------


def test_plot_waterfall_total_matches_explain(fitted_clf) -> None:
    clf, X, _ = fitted_clf
    x_row = X.iloc[[0]]
    expected_total = float(clf.explain(x_row)["contribution"].sum())
    ax = plots.plot_waterfall(clf, x_row)
    total_bar = ax.patches[-1]
    reconstructed = (
        total_bar.get_x() + total_bar.get_width()
        if expected_total >= 0
        else total_bar.get_x()
    )
    assert reconstructed == pytest.approx(expected_total, abs=1e-6)


def test_plot_waterfall_collapses_beyond_max_rules(fitted_clf) -> None:
    clf, X, _ = fitted_clf
    # Row 6 (seed 0) fires two rules (an age rule and a purpose rule); with
    # max_rules=1 the second must collapse into "(other rules)".
    x_row = X.iloc[[6]]
    assert (clf.explain(x_row)["feature"] != "<intercept>").sum() >= 2
    ax = plots.plot_waterfall(clf, x_row, max_rules=1)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "(other rules)" in labels


# ---- plot_reliability -------------------------------------------------------


def test_plot_reliability_has_diagonal_and_data_line(fitted_clf) -> None:
    clf, X, y = fitted_clf
    p = clf.predict_proba(X)[:, 1]
    ax = plots.plot_reliability(y, p, n_bins=10)
    assert len(ax.lines) >= 2


# ---- plot_proxy_association --------------------------------------------------


def test_plot_proxy_association_bar_count(fitted_clf) -> None:
    clf, X, _ = fitted_clf
    A = X["purpose"].astype(str)
    report = ProxyAudit(clf).report(X, A)
    ax = plots.plot_proxy_association(report, top_n=10)
    assert len(ax.patches) == min(10, len(report))


# ---- plot_group_metrics --------------------------------------------------------


def test_plot_group_metrics_bar_count(fitted_clf) -> None:
    clf, X, y = fitted_clf
    A = X["purpose"].astype(str)
    p = clf.predict_proba(X)[:, 1]
    metrics = group_metrics(y, p, A)
    ax = plots.plot_group_metrics(metrics)
    assert len(ax.patches) == len(metrics["by_group"]) * 3


# ---- lazy import discipline --------------------------------------------------


def test_all_six_functions_raise_helpful_import_error(monkeypatch, fitted_clf) -> None:
    clf, X, y = fitted_clf
    A = X["purpose"].astype(str)
    p = clf.predict_proba(X)[:, 1]
    report = ProxyAudit(clf).report(X, A)
    metrics = group_metrics(y, p, A)

    def _boom() -> None:
        raise ImportError(
            "plotting requires matplotlib — install with: pip install flaggam[viz]"
        )

    monkeypatch.setattr(plots, "_plt", _boom)
    calls = [
        lambda: plots.plot_shape(clf, "age"),
        lambda: plots.plot_rule_importance(clf),
        lambda: plots.plot_waterfall(clf, X.iloc[[0]]),
        lambda: plots.plot_reliability(y, p),
        lambda: plots.plot_proxy_association(report),
        lambda: plots.plot_group_metrics(metrics),
    ]
    for call in calls:
        with pytest.raises(ImportError, match=r"flaggam\[viz\]"):
            call()


def test_plots_module_importable_without_matplotlib(monkeypatch) -> None:
    """import flaggam / import flaggam.plots must both work without matplotlib.

    Simulate an environment without matplotlib by making it unimportable, then
    reload the package fresh. The module import itself must succeed; only a
    plot function call should raise ImportError.
    """
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "flaggam.plots", raising=False)
    monkeypatch.delitem(sys.modules, "flaggam", raising=False)

    import flaggam  # noqa: F401
    import flaggam.plots as reloaded

    with pytest.raises(ImportError, match=r"flaggam\[viz\]"):
        reloaded.plot_rule_importance(object())
