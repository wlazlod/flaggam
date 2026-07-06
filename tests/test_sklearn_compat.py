"""sklearn check_estimator compatibility tests for FlagGAM estimators."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

# Suppress warnings from sklearn/scipy/pytest internals that appear during
# check_estimator runs and are unrelated to our code.
pytestmark = [
    pytest.mark.filterwarnings("ignore::pytest.PytestRemovedIn10Warning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]

from flaggam import FlagGAMClassifier, FlagGAMRegressor

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
