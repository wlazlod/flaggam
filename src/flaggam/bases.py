"""Basis objects: one column of Z(X) each, with screening metadata.

Missing semantics ("no_evidence", spec §7): NaN/None input never triggers a
basis; transform returns 0.0 there. For TrendBasis a missing value maps to
0.0, i.e. the feature mean (documented decision; paper is silent).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _is_missing(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind in "fc":
        return np.isnan(x)
    return np.array([v is None or v != v for v in x], dtype=bool)


@dataclass(frozen=True)
class Basis:
    """One univariate basis function z_ir(x_i) plus its discovery metadata."""

    feature: str
    support: int
    effect_size: float
    p_value: float
    p_adj: float
    enriched_class: Any = None

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ThresholdBasis(Basis):
    """Tail flag 1{x <= c} (side='low') or 1{x >= c} (side='high')."""

    cutoff: float = 0.0
    side: str = "low"

    @property
    def kind(self) -> str:
        return f"threshold_{self.side}"

    @property
    def name(self) -> str:
        op = "<=" if self.side == "low" else ">="
        return f"{self.feature} {op} {self.cutoff:g}"

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        miss = np.isnan(x)
        z = (x <= self.cutoff) if self.side == "low" else (x >= self.cutoff)
        return np.where(miss, 0.0, z.astype(float))


@dataclass(frozen=True)
class CategoryBasis(Basis):
    """Level flag 1{x == v}; also the regression step basis."""

    level: Any = None

    @property
    def kind(self) -> str:
        return "category"

    @property
    def name(self) -> str:
        return f"{self.feature} == {self.level!r}"

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=object)
        miss = _is_missing(x)
        z = np.array([v == self.level for v in x], dtype=float)
        return np.where(miss, 0.0, z)


@dataclass(frozen=True)
class HingeBasis(Basis):
    """Tail-deviation hinge (x - c)_+ (side='high') or (c - x)_+ (side='low')."""

    cutoff: float = 0.0
    side: str = "low"

    @property
    def kind(self) -> str:
        return f"hinge_{self.side}"

    @property
    def name(self) -> str:
        return (
            f"({self.cutoff:g} - {self.feature})_+"
            if self.side == "low"
            else f"({self.feature} - {self.cutoff:g})_+"
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        miss = np.isnan(x)
        d = (self.cutoff - x) if self.side == "low" else (x - self.cutoff)
        return np.where(miss, 0.0, np.maximum(d, 0.0))


@dataclass(frozen=True)
class TrendBasis(Basis):
    """Centered baseline trend x - mean (regression numerical features)."""

    mean: float = 0.0

    @property
    def kind(self) -> str:
        return "trend"

    @property
    def name(self) -> str:
        return f"{self.feature} - mean({self.mean:g})"

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), 0.0, x - self.mean)


@dataclass(frozen=True)
class MissingIndicatorBasis(Basis):
    """Explicit flag 1{x is missing} (missing='indicator' mode only)."""

    @property
    def kind(self) -> str:
        return "missing_indicator"

    @property
    def name(self) -> str:
        return f"{self.feature} is missing"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return _is_missing(np.asarray(x, dtype=object)).astype(float)
