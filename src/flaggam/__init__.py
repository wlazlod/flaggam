"""FlagGAM: rule-basis generalized additive models (from-scratch implementation)."""

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
]
