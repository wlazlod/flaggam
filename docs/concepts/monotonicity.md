# Monotonicity

Regulators often require the predicted probability of default to be monotone in a
feature — non-increasing in income, say: no applicant should be scored riskier *because*
they earn more. FlagGAM can enforce this **exactly**, not approximately.

This module is an original addition — not part of Zhao & Welsch (2026); the full
derivation is recorded in [DECISIONS 20](../DECISIONS.md).

## Why the constraint is exact

A feature's additive contribution is a sum of its bases: step indicators
(`threshold_low/high`), ramps (`hinge_low/high`), and the centered `trend` term. Each of
these is itself a monotone function of the raw feature value. A sum of monotone
functions with sign-controlled weights is monotone — so constraining the *sign* of each
basis coefficient makes the feature's whole fitted shape monotone by construction. No
post-hoc isotonic projection, no penalty tuning, no residual violations.

```python
from flaggam import FlagGAMClassifier

clf_mono = FlagGAMClassifier(monotonic_constraints={"age": -1}).fit(X, y)
# PD non-increasing in age — exactly, at every value of age
```

## Usage

`monotonic_constraints` is a dict mapping feature name to `+1` (non-decreasing), `-1`
(non-increasing), or `0`/absent (unconstrained). Under the hood the additive head is
replaced by a box-constrained optimization (`scipy.optimize.minimize` with L-BFGS-B)
that fits the same L2-penalized objective with per-coefficient sign bounds.

Scope and limits:

- Categorical and missing-indicator bases are never constrained — a categorical level
  has no defined "direction."
- Supported for `representation="full"` binary classification and regression only;
  incompatible with `representation="compact"` (compact-score columns don't map 1:1 to a
  single basis coefficient).
- A list-valued `C`/`alpha` falls back to `1.0` — CV tuning of the constrained head is
  out of scope.

The [German Credit walkthrough](../notebooks/german_credit.ipynb) constrains
`duration_months` and shows the resulting monotone shape with essentially unchanged
AUROC. See the [API reference](../api.md#monotonic) for the full API.
