"""Screening statistics for rule discovery.

These are screening tools, not confirmatory inference (Zhao & Welsch,
arXiv:2605.31189).
Two-proportion test falls back to Fisher's exact test when any expected
cell count is below 5 (documented decision, spec §6.3).
"""

import math

import numpy as np
from scipy import stats


def compute_min_support(n_train: int) -> int:
    """Minimum tail/level support: min(200, max(20, ceil(0.02 * n_train)))."""
    return min(200, max(20, math.ceil(0.02 * n_train)))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (within one feature's candidates)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m, dtype=float)
    adj[order] = np.clip(ranked, 0.0, 1.0)
    return adj


def _expected_counts(k_tail: int, n_tail: int, k_base: int, n_base: int) -> float:
    table = np.array([[k_tail, n_tail - k_tail], [k_base, n_base - k_base]], dtype=float)
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    return float(expected.min())


def two_proportion_test(k_tail: int, n_tail: int, k_base: int, n_base: int) -> float:
    """Two-sided two-proportion z-test; Fisher's exact if any expected count < 5."""
    if n_tail == 0 or n_base == 0:
        return 1.0
    if _expected_counts(k_tail, n_tail, k_base, n_base) < 5.0:
        table = [[k_tail, n_tail - k_tail], [k_base, n_base - k_base]]
        return float(stats.fisher_exact(table, alternative="two-sided")[1])
    p1, p0 = k_tail / n_tail, k_base / n_base
    pooled = (k_tail + k_base) / (n_tail + n_base)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_tail + 1.0 / n_base))
    if se == 0.0:
        return 1.0
    z = (p1 - p0) / se
    return float(2.0 * stats.norm.sf(abs(z)))


def chi_square_test(y_tail: np.ndarray, y_base: np.ndarray) -> float:
    """Pearson chi-square p-value on the 2 x K (region x class) table."""
    classes = np.union1d(np.unique(y_tail), np.unique(y_base))
    table = np.array(
        [[np.sum(y_tail == c) for c in classes], [np.sum(y_base == c) for c in classes]]
    )
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2 or table.sum(axis=1).min() == 0:
        return 1.0
    return float(stats.chi2_contingency(table, correction=False)[1])


def welch_t_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value; 1.0 when a side is degenerate."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.pvalue) if np.isfinite(res.pvalue) else 1.0


def risk_difference(k_tail: int, n_tail: int, k_base: int, n_base: int) -> float:
    """Absolute difference in positive-class rate, tail vs baseline (spec §6.1)."""
    return abs(k_tail / n_tail - k_base / n_base)


def log_odds_ratio(k_tail: int, n_tail: int, k_base: int, n_base: int) -> float:
    """Absolute log odds ratio with 0.5 continuity correction."""
    a, b = k_tail + 0.5, n_tail - k_tail + 0.5
    c, d = k_base + 0.5, n_base - k_base + 0.5
    return abs(math.log((a * d) / (b * c)))


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute standardized mean difference with pooled (average) variance."""
    var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    if var == 0.0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(b)) / np.sqrt(var))
