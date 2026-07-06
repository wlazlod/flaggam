import numpy as np
import pandas as pd
import pytest

from flaggam.bases import CategoryBasis, ThresholdBasis
from flaggam.weighting import (
    compact_scores,
    correlation_ratio,
    cramers_v,
    feature_weights,
    point_biserial,
)

META = dict(support=50, effect_size=0.2, p_value=0.001, p_adj=0.004)


def test_point_biserial_known() -> None:
    y = np.array([0, 0, 1, 1])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = np.corrcoef(x, y)[0, 1]
    assert point_biserial(x, y) == pytest.approx(abs(r))


def test_correlation_ratio_bounds() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    x_strong = y + rng.normal(0, 0.1, 300)
    x_null = rng.normal(0, 1, 300)
    assert correlation_ratio(x_strong, y) > 0.9
    assert correlation_ratio(x_null, y) < 0.2


def test_cramers_v_perfect_and_null() -> None:
    y = np.array([0, 1] * 100)
    assert cramers_v(np.where(y == 1, "a", "b"), y) == pytest.approx(1.0, abs=0.05)
    rng = np.random.default_rng(0)
    assert cramers_v(rng.choice(["a", "b"], 200), y) < 0.2


def test_compact_scores_counts_fired_flags() -> None:
    from scipy import sparse

    b0 = ThresholdBasis(feature="age", cutoff=25.0, side="low", enriched_class=1, **META)
    b1 = CategoryBasis(feature="purpose", level="edu", enriched_class=0, **META)
    Z = sparse.csr_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    s_eq = compact_scores(Z, [b0, b1], classes=np.array([0, 1]), weights=None)
    np.testing.assert_array_equal(s_eq, [[1.0, 1.0], [1.0, 0.0]])
    s_w = compact_scores(
        Z, [b0, b1], classes=np.array([0, 1]), weights={"age": 0.5, "purpose": 2.0}
    )
    np.testing.assert_array_equal(s_w, [[2.0, 0.5], [2.0, 0.0]])


def test_feature_weights_types(clf_data=None) -> None:
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": pd.Categorical(rng.choice(["a", "b"], n))})
    y = (rng.uniform(size=n) < 0.3).astype(int)
    w = feature_weights(X, y, task="binary", numerical=["num"])
    assert set(w) == {"num", "cat"} and all(0.0 <= v <= 1.0 for v in w.values())
