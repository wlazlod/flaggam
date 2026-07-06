"""sklearn check_estimator compatibility tests for FlagGAM estimators."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from flaggam import FlagGAMClassifier, FlagGAMRegressor

# Suppress warnings from sklearn/scipy internals that appear during
# check_estimator runs but are unrelated to flaggam's own code.
#
# "Precision loss occurred in moment calculation" — emitted by
#   scipy.stats._axis_nan_policy during check_positive_only_tag_during_fit
#   when scipy processes near-constant synthetic data.
#
# The PytestRemovedIn10Warning (parametrize_with_checks yields a generator)
# is also filtered globally in pyproject.toml for the collection phase.
pytestmark = [
    pytest.mark.filterwarnings("ignore::pytest.PytestRemovedIn10Warning"),
    pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning"),
]

# Checks that cannot apply to a screening-based rule model. Each entry MUST
# carry a reason; keep this list as short as possible.
EXPECTED_FAILED: dict[str, str] = {
    # e.g. "check_classifiers_train": "tiny n -> no basis survives; prior-only predictions"
}


@parametrize_with_checks(
    [FlagGAMClassifier(random_state=0), FlagGAMRegressor(random_state=0)],
    expected_failed_checks=lambda est: EXPECTED_FAILED,
)
def test_sklearn_compat(estimator, check) -> None:
    check(estimator)
