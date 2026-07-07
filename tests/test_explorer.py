"""Tests for flaggam.explorer (original addition: self-contained HTML rules explorer)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from flaggam import FlagGAMClassifier, FlagGAMRegressor
from flaggam.explorer import export_rules_html

_DATA_RE = re.compile(
    r'<script type="application/json" id="flaggam-data">(.*?)</script>', re.DOTALL
)


def _synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """README-style synthetic: age + purpose classifier data (mirrors test_plots.py)."""
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


def _extract_payload(html: str) -> dict:
    match = _DATA_RE.search(html)
    assert match is not None
    return json.loads(match.group(1))


def test_html_document_shape(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    html = export_rules_html(clf)
    assert html.startswith("<!DOCTYPE html>")
    assert "<select" in html
    rules = clf.export_rules()
    for feature in rules["feature"].unique():
        assert feature in html


def test_payload_matches_export_rules(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    html = export_rules_html(clf)
    payload = _extract_payload(html)
    rules = clf.export_rules()
    total = sum(len(f["rules"]) for f in payload["features"])
    assert total == len(rules)
    for f in payload["features"]:
        if f["type"] == "numeric":
            assert len(f["curve"]["x"]) == len(f["curve"]["y"])
            assert len(f["curve"]["x"]) > 100
        else:
            assert len(f["bars"]) > 0


def test_curve_matches_plot_shape_math(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    html = export_rules_html(clf)
    payload = _extract_payload(html)
    feat = next(f for f in payload["features"] if f["name"] == "age")
    bases = clf.core_.bases_
    coef = np.ravel(clf.head_.coef_)
    feat_bases = [(j, b) for j, b in enumerate(bases) if b.feature == "age"]
    xs = feat["curve"]["x"]
    for idx in (0, len(xs) // 2, -1):
        x = xs[idx]
        expected = sum(coef[j] * b.transform(np.array([x]))[0] for j, b in feat_bases)
        assert expected == pytest.approx(feat["curve"]["y"][idx], abs=1e-6)


def test_writes_file(fitted_clf, tmp_path) -> None:
    clf, _, _ = fitted_clf
    path = tmp_path / "x.html"
    html = export_rules_html(clf, path=path)
    assert path.read_text(encoding="utf-8") == html


def test_rejects_unfitted_estimator() -> None:
    clf = FlagGAMClassifier(random_state=0)
    with pytest.raises(ValueError):
        export_rules_html(clf)


def test_rejects_compact_representation() -> None:
    X, y = _synthetic()
    clf = FlagGAMClassifier(
        representation="compact", feature_weighting="auto", random_state=0
    ).fit(X, y)
    with pytest.raises(ValueError, match="representation='full'"):
        export_rules_html(clf)


def test_rejects_multiclass() -> None:
    X, _ = _synthetic()
    y3 = np.arange(len(X)) % 3
    clf = FlagGAMClassifier(random_state=0).fit(X, y3)
    with pytest.raises(ValueError, match="binary"):
        export_rules_html(clf)


def test_rejects_flexible_head() -> None:
    from sklearn.ensemble import RandomForestClassifier

    X, y = _synthetic()
    clf = FlagGAMClassifier(
        head="flexible",
        flexible_estimator=RandomForestClassifier(n_estimators=5, random_state=0),
        random_state=0,
    ).fit(X, y)
    with pytest.raises(ValueError, match="additive"):
        export_rules_html(clf)


def test_regression_task() -> None:
    rng = np.random.default_rng(2)
    n = 400
    x1 = rng.normal(size=n)
    y = 2.0 * (x1 > 0) + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"x1": x1})
    reg = FlagGAMRegressor(random_state=0).fit(X, y)
    html = export_rules_html(reg)
    payload = _extract_payload(html)
    assert payload["task"] == "regression"


def test_no_external_resources(fitted_clf) -> None:
    clf, _, _ = fitted_clf
    html = export_rules_html(clf)
    assert "http://" not in html
    assert "https://" not in html
