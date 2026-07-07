"""FlagGAM: rule-basis generalized additive models (from-scratch implementation)."""

from .calibration import (
    CalibratedFlagGAM,
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    reliability_curve,
)
from .datasets import CLASSIFICATION, REGRESSION, DatasetSpec
from .estimator import FlagGAMClassifier, FlagGAMRegressor

__version__ = "0.1.0"

__all__ = [
    "FlagGAMClassifier",
    "FlagGAMRegressor",
    "__version__",
    "CLASSIFICATION",
    "REGRESSION",
    "DatasetSpec",
    "CalibratedFlagGAM",
    "reliability_curve",
    "brier_score",
    "expected_calibration_error",
    "calibration_in_the_large",
]
