import numpy as np
import pandas as pd
import pytest
from benchmarks.methods import Method, get_methods


@pytest.fixture()
def clf_xy():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    c1 = pd.Categorical(rng.choice(["a", "b"], n))
    logit = -0.5 + 2.0 * (x1 <= -0.5) + 1.0 * (np.asarray(c1) == "a")
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return pd.DataFrame({"x1": x1, "c1": c1}), y


@pytest.fixture()
def reg_xy():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    y = pd.Series(2.0 * x1 + rng.normal(0, 0.5, n))
    return pd.DataFrame({"x1": x1, "c1": pd.Categorical(rng.choice(["a", "b"], n))}), y


def test_registry_contents() -> None:
    clf, skipped_c = get_methods("binary")
    reg, skipped_r = get_methods("regression")
    core = {"logistic", "flaggam", "flaggam_rf", "rf", "xgboost", "ebm", "rulefit"}
    assert core <= set(clf) | set(skipped_c)
    assert {
        "ridge", "flaggam", "flaggam_rf", "rf", "xgboost", "ebm", "rulefit"
    } <= set(reg) | set(skipped_r)
    # glrm appears somewhere (available or skipped-with-reason)
    assert "glrm" in set(clf) | set(skipped_c)
    for reason in {**skipped_c, **skipped_r}.values():
        assert isinstance(reason, str) and reason


@pytest.mark.parametrize("name", ["logistic", "flaggam", "rf", "xgboost"])
def test_binary_methods_beat_chance(clf_xy, name) -> None:
    X, y = clf_xy
    factories, skipped = get_methods("binary")
    if name not in factories:
        pytest.skip(skipped.get(name, "unavailable"))
    m = factories[name]()
    assert isinstance(m, Method)
    m.fit(X.iloc[:400], y.iloc[:400], seed=0)
    s = m.predict_scores(X.iloc[400:])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y.iloc[400:], s) > 0.6
    # determinism
    m2 = factories[name]().fit(X.iloc[:400], y.iloc[:400], seed=0)
    np.testing.assert_allclose(s, m2.predict_scores(X.iloc[400:]), rtol=1e-6)


@pytest.mark.parametrize("name", ["ridge", "flaggam", "xgboost"])
def test_regression_methods_fit(reg_xy, name) -> None:
    X, y = reg_xy
    factories, skipped = get_methods("regression")
    if name not in factories:
        pytest.skip(skipped.get(name, "unavailable"))
    m = factories[name]().fit(X.iloc[:400], y.iloc[:400], seed=0)
    pred = m.predict_scores(X.iloc[400:])
    ss = ((y.iloc[400:] - pred) ** 2).sum() / ((y.iloc[400:] - y.iloc[400:].mean()) ** 2).sum()
    assert 1 - ss > 0.5


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ebm", "rulefit"])
def test_slow_methods_fit(clf_xy, name) -> None:
    X, y = clf_xy
    factories, skipped = get_methods("binary")
    if name not in factories:
        pytest.skip(skipped.get(name, "unavailable"))
    m = factories[name]().fit(X.iloc[:400], y.iloc[:400], seed=0)
    assert m.predict_scores(X.iloc[400:]).shape == (100,)
