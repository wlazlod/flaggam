import numpy as np
import pytest

from flaggam.screening import (
    bh_adjust,
    chi_square_test,
    compute_min_support,
    log_odds_ratio,
    risk_difference,
    standardized_mean_difference,
    two_proportion_test,
    welch_t_test,
)


def test_min_support_formula() -> None:
    assert compute_min_support(100) == 20  # max(20, ceil(2)) = 20
    assert compute_min_support(5000) == 100  # ceil(0.02*5000) = 100
    assert compute_min_support(100_000) == 200  # capped at 200


def test_bh_adjust_matches_known_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = bh_adjust(p)
    # BH: sort p, p*(m/rank), enforce monotonicity from the largest rank down.
    expected = np.array([0.04, 0.05333333, 0.05333333, 0.20])
    np.testing.assert_allclose(adj, expected, rtol=1e-6)
    assert np.all(adj >= p) and np.all(adj <= 1.0)


def test_two_proportion_detects_enrichment() -> None:
    # 60/100 positive in tail vs 30/300 in baseline: clearly significant.
    assert two_proportion_test(60, 100, 30, 300) < 1e-6
    # Same rates: not significant.
    assert two_proportion_test(10, 100, 30, 300) > 0.5


def test_two_proportion_small_counts_uses_fisher() -> None:
    from scipy import stats

    # Expected cell counts < 5 -> must route to Fisher's exact test.
    expected = float(stats.fisher_exact([[3, 3], [1, 7]], alternative="two-sided")[1])
    assert two_proportion_test(3, 6, 1, 8) == pytest.approx(expected, abs=1e-12)


def test_chi_square_multiclass() -> None:
    rng = np.random.default_rng(0)
    y_tail = np.array([0] * 80 + [1] * 10 + [2] * 10)
    y_base = rng.integers(0, 3, size=300)
    assert chi_square_test(y_tail, y_base) < 1e-6


def test_welch_and_smd() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 200)
    b = rng.normal(0.0, 2.0, 300)
    assert welch_t_test(a, b) < 1e-4
    assert standardized_mean_difference(a, b) == pytest.approx(
        0.68987893242611642, rel=1e-9
    )  # pinned for default_rng(0)


def test_effect_sizes() -> None:
    assert risk_difference(60, 100, 30, 300) == pytest.approx(0.5)
    assert log_odds_ratio(60, 100, 30, 300) > 0
    # Continuity correction keeps zero cells finite.
    assert np.isfinite(log_odds_ratio(0, 50, 30, 300))
