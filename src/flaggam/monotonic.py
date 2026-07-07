"""Exact monotonicity constraints for the FlagGAM additive head.

This module is an ORIGINAL ADDITION and is not part of Zhao & Welsch
(arXiv:2605.31189). Because each numerical feature contributes monotone
step/ramp bases (tail flags, hinges, trend), sign constraints on their
coefficients yield EXACT monotonicity of the additive contribution.
Sign table and design notes in docs/DECISIONS.md entry 20.
"""

import logging

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

logger = logging.getLogger(__name__)

_POS = (0.0, None)
_NEG = (None, 0.0)
_FREE = (None, None)

# bound on theta for +1 (risk non-decreasing in x); -1 mirrors, 0/absent = free
_PLUS_ONE = {
    "threshold_low": _NEG,
    "threshold_high": _POS,
    "hinge_low": _NEG,
    "hinge_high": _POS,
    "trend": _POS,
}


def bounds_for_bases(bases: list, constraints: dict) -> list[tuple[float | None, float | None]]:
    """Per-basis-column box bounds implementing spec §8.2 sign constraints."""
    out: list[tuple[float | None, float | None]] = []
    for b in bases:
        direction = constraints.get(b.feature, 0)
        rule = _PLUS_ONE.get(b.kind)
        if direction == 0 or rule is None:  # unconstrained, categorical, or missing
            if direction != 0 and b.kind not in ("category", "missing_indicator"):
                logger.warning(
                    "basis kind %r on constrained feature %r is not in the "
                    "monotonicity sign table; leaving its coefficient unconstrained — "
                    "monotonicity is no longer guaranteed",
                    b.kind,
                    b.feature,
                )
            out.append(_FREE)
        elif direction == 1:
            out.append(rule)
        else:  # -1: mirror the +1 bound by swapping the tuple
            out.append((rule[1], rule[0]))
    return out


def _validate_constraints(constraints: dict, feature_names: list[str]) -> None:
    if not isinstance(constraints, dict):
        raise ValueError("monotonic_constraints must be a dict {feature: +1|-1|0}")
    bad_vals = {v for v in constraints.values() if v not in (1, -1, 0)}
    if bad_vals:
        raise ValueError(f"monotonic_constraints values must be +1, -1 or 0; got {bad_vals}")
    unknown = set(constraints) - set(feature_names)
    if unknown:
        raise ValueError(f"monotonic_constraints reference unknown feature(s): {sorted(unknown)}")


class MonotonicAdditiveHead:
    """L-BFGS-B box-constrained L2 logistic (binary) or ridge (regression) head.

    Drop-in for `AdditiveHead` when `monotonic_constraints` is active. Unlike
    `AdditiveHead`, `C`/`alpha` are single floats only: CV tuning of the
    constrained head is out of scope (spec routes list-valued C/alpha to the
    spec default of 1.0 before construction, see estimator.py).
    """

    coef_: np.ndarray
    intercept_: float | np.ndarray

    def __init__(
        self,
        task: str,
        bounds: list[tuple[float | None, float | None]],
        C: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        self.task = task
        self.bounds = bounds
        self.C = C
        self.alpha = alpha

    def fit(self, Z, y: np.ndarray) -> "MonotonicAdditiveHead":
        self.n_features_ = Z.shape[1]
        y = np.asarray(y, dtype=float)
        Zc = sparse.csr_matrix(Z)
        p = Zc.shape[1]

        if self.task == "regression":

            def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
                theta, b = w[:p], w[p]
                r = y - (Zc @ theta + b)
                f = float(r @ r + self.alpha * theta @ theta)
                g = np.empty(p + 1)
                g[:p] = -2.0 * (Zc.T @ r) + 2.0 * self.alpha * theta
                g[p] = -2.0 * r.sum()
                return f, g

        else:  # binary

            def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
                theta, b = w[:p], w[p]
                m = Zc @ theta + b
                f = float(0.5 * theta @ theta + self.C * (np.logaddexp(0.0, m) - y * m).sum())
                d = expit(m) - y
                g = np.empty(p + 1)
                g[:p] = theta + self.C * (Zc.T @ d)
                g[p] = self.C * d.sum()
                return f, g

        res = minimize(
            obj,
            np.zeros(p + 1),
            jac=True,
            method="L-BFGS-B",
            bounds=list(self.bounds) + [_FREE],
            options={"maxiter": 2000},
        )
        if not res.success:
            logger.warning("constrained head optimizer: %s", res.message)
        theta, b = res.x[:p], float(res.x[p])
        if self.task == "regression":
            self.coef_ = theta
            self.intercept_ = b
        else:
            self.coef_ = theta.reshape(1, -1)
            self.intercept_ = np.array([b])
        return self

    def _margin(self, Z) -> np.ndarray:
        return sparse.csr_matrix(Z) @ np.ravel(self.coef_) + np.ravel(self.intercept_)[0]

    def predict(self, Z) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(self._margin(Z), dtype=float)
        return (expit(self._margin(Z)) >= 0.5).astype(int)

    def predict_proba(self, Z) -> np.ndarray:
        p1 = expit(self._margin(Z))
        return np.column_stack([1 - p1, p1])
