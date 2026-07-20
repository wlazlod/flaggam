# Calibration

A model can rank well (good AUROC) while still misreporting probabilities — and in credit
settings the predicted probability of default (PD) is the number that prices the loan.
`flaggam.calibration` provides diagnostics (reliability curve, Brier score, expected
calibration error, calibration-in-the-large) and recalibration methods (`platt`,
`isotonic`, `base_rate`) for the predicted probability of the positive class.

This module is an original addition — not part of Zhao & Welsch (2026); the design
rationale is recorded in [DECISIONS](../DECISIONS.md).

## Recalibrating

```python
from flaggam import CalibratedFlagGAM, FlagGAMClassifier

cal = CalibratedFlagGAM(FlagGAMClassifier(random_state=0), method="platt", cv=5)
cal.fit(X, y)
pd_hat = cal.predict_proba(X)[:, 1]
```

Calibration must be fit on data disjoint from head fitting, or the calibrator just
memorizes the head's in-sample optimism. With `cv=k`, `CalibratedFlagGAM` fits the
estimator on each of `k` stratified folds, collects out-of-fold predictions, and fits a
single pooled calibrator on those out-of-fold predictions — keeping the calibrator's
training data disjoint from the head-fitting data for every observation. Pass
`cv="prefit"` to calibrate an already-fitted estimator against held-out data supplied to
`fit()`. Calibration is defined for binary targets only.

## Diagnostics

```python
from flaggam import reliability_curve, brier_score, expected_calibration_error

curve = reliability_curve(y_test, p_hat, n_bins=10)   # per-bin predicted vs. observed
brier_score(y_test, p_hat)
expected_calibration_error(y_test, p_hat)
```

`plot_reliability` draws the reliability diagram with a per-bin count overlay — see
[Visualization](visualization.md). The
[German Credit walkthrough](../notebooks/german_credit.ipynb) shows a raw model's ECE
before and after Platt recalibration.

See the [API reference](../api.md#calibration) for the full API.
