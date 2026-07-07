# Extensions

Three optional modules extend the paper's method. Each is an original addition — not
present in Zhao & Welsch (2026) — and lives in its own module, documented in the
[API Reference](../api/index.md).

## PD Calibration

`flaggam.calibration` provides diagnostics (reliability curve, Brier score, expected
calibration error, calibration-in-the-large) and recalibration methods (`platt`,
`isotonic`, `base_rate`) for the predicted probability of the positive class. Because a
poorly calibrated model can rank well (good AUROC) while still misreporting probabilities,
calibration is fit on data disjoint from head fitting.

```python
from flaggam import CalibratedFlagGAM, FlagGAMClassifier

cal = CalibratedFlagGAM(FlagGAMClassifier(random_state=0), method="platt", cv=5)
cal.fit(X, y)
pd_hat = cal.predict_proba(X)[:, 1]
```

With `cv=k`, `CalibratedFlagGAM` fits the estimator on each of `k` stratified folds,
collects out-of-fold predictions, and fits a single pooled calibrator on those
out-of-fold predictions — keeping the calibrator's training data disjoint from the
head-fitting data for every observation. Pass `cv="prefit"` to calibrate an
already-fitted estimator against held-out data supplied to `fit()`. Calibration is
defined for binary targets only.

## Monotonicity Constraints

Regulators often require the predicted probability of default (PD) to be monotone in a
feature (e.g., non-increasing in age or income). Because FlagGAM's numerical
contributions are step/ramp basis functions (threshold, hinge, and trend), constraining
the sign of each basis's coefficient gives *exact* monotonicity of the additive
contribution — no post-hoc isotonic projection is needed.

```python
clf_mono = FlagGAMClassifier(monotonic_constraints={"age": -1}).fit(X, y)  # PD non-increasing in age
```

`monotonic_constraints` is a dict mapping feature name to `+1` (non-decreasing), `-1`
(non-increasing), or `0`/absent (unconstrained). Categorical and missing-indicator bases
are never constrained — a categorical level has no defined "direction." Monotonicity is
supported for `representation="full"` binary classification and regression only; it is
incompatible with `representation="compact"` (compact-score columns don't map 1:1 to a
single basis coefficient).

## Fairness / Proxy Audit

`flaggam.fairness` provides group-level performance metrics for a protected attribute and
a rule-level audit that ranks bases by their association with it — operationalizing the
paper's own warning that selected rules may encode bias or proxies for protected
attributes.

```python
from flaggam import ProxyAudit, group_metrics

A = X["purpose"].astype(str)  # protected attribute (illustrative)
metrics = group_metrics(y, clf.predict_proba(X)[:, 1], A)
report = ProxyAudit(clf).report(X, A)                       # ranked candidate proxies
clean_clf, trade = ProxyAudit(clf).drop_proxies(X, y, A, threshold=0.3)
```

`group_metrics` reports, per level of `A`: `n`, `base_rate`, `mean_predicted`,
`selection_rate`, `tpr`, `auroc`, and `ece`, plus three gap summaries
(`demographic_parity_diff`, `equal_opportunity_diff`, `auroc_gap`) computed as the
max-minus-min across groups. `ProxyAudit.report` ranks every fitted basis by its
association with `A` (absolute point-biserial correlation for numeric `A`, Cramer's V
otherwise) and flags those above `threshold`. `ProxyAudit.drop_proxies` refits only the
head after removing flagged bases, returning the new estimator alongside a one-row
trade-off summary (`n_dropped`, AUROC and demographic-parity-gap before/after). Both
methods require a fitted binary classifier with `representation="full"` and
`head="additive"` (no monotonic constraints).

See [Calibration](../api/calibration.md), [Monotonic](../api/monotonic.md), and
[Fairness](../api/fairness.md) for the full API, and
[`docs/DECISIONS.md`](../DECISIONS.md) for the design rationale behind each module.
