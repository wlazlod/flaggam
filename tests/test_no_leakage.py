"""No-leakage guarantee: test data must not influence rule discovery (spec §14)."""

import numpy as np
import pandas as pd

from flaggam import FlagGAMClassifier


def _make(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2.0 * (x1 <= -1.0) - 1.0)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), y


def test_test_rows_never_influence_discovery() -> None:
    X_train, y_train = _make(2000, seed=0)
    # Test set with a planted, extreme signal on x2 that does NOT exist in train.
    X_test, _ = _make(500, seed=1)
    X_test["x2"] = X_test["x2"] + 100.0

    clf_a = FlagGAMClassifier(random_state=0).fit(X_train, y_train)
    rules_a = clf_a.export_rules()
    _ = clf_a.predict_proba(X_test)  # touching test data after fit ...
    _ = clf_a.transform(X_test)

    clf_b = FlagGAMClassifier(random_state=0).fit(X_train, y_train)
    rules_b = clf_b.export_rules()

    # ... must leave discovered rules and weights bit-identical.
    pd.testing.assert_frame_equal(rules_a, clf_a.export_rules())
    pd.testing.assert_frame_equal(rules_a, rules_b)
    # And no rule may reference the test-only shift region of x2.
    cut_high = rules_a.loc[rules_a.kind == "threshold_high", "cutoff"]
    assert (cut_high < 50).all()
