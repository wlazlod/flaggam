import numpy as np

from flaggam.bases import (
    CategoryBasis,
    HingeBasis,
    MissingIndicatorBasis,
    ThresholdBasis,
    TrendBasis,
)

META = dict(support=50, effect_size=0.2, p_value=0.001, p_adj=0.004, enriched_class=1)


def test_threshold_low_and_high() -> None:
    x = np.array([1.0, 25.0, 26.0, np.nan])
    low = ThresholdBasis(feature="age", cutoff=25.0, side="low", **META)
    high = ThresholdBasis(feature="age", cutoff=25.0, side="high", **META)
    np.testing.assert_array_equal(low.transform(x), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(high.transform(x), [0.0, 1.0, 1.0, 0.0])
    assert low.kind == "threshold_low" and "<=" in low.name
    assert high.kind == "threshold_high" and ">=" in high.name


def test_category_basis_handles_missing() -> None:
    x = np.array(["car", "tv", None], dtype=object)
    b = CategoryBasis(feature="purpose", level="car", **META)
    np.testing.assert_array_equal(b.transform(x), [1.0, 0.0, 0.0])
    assert b.kind == "category"


def test_hinge_and_trend() -> None:
    x = np.array([1.0, 3.0, np.nan])
    hi = HingeBasis(feature="inc", cutoff=2.0, side="high", **META)
    lo = HingeBasis(feature="inc", cutoff=2.0, side="low", **META)
    np.testing.assert_array_equal(hi.transform(x), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(lo.transform(x), [1.0, 0.0, 0.0])
    tr = TrendBasis(feature="inc", mean=2.0, **META)
    np.testing.assert_array_equal(tr.transform(x), [-1.0, 1.0, 0.0])  # NaN -> 0 (== mean)


def test_missing_indicator() -> None:
    x = np.array([1.0, np.nan])
    b = MissingIndicatorBasis(feature="age", **META)
    np.testing.assert_array_equal(b.transform(x), [0.0, 1.0])


def test_frozen_and_hashable() -> None:
    b = ThresholdBasis(feature="age", cutoff=25.0, side="low", **META)
    assert hash(b)
