# FlagGAM

**Rule-basis generalized additive models for interpretable tabular prediction.**

`flaggam` answers: *which concrete, statistically vetted conditions in my data drive the
prediction — and by how much?* It screens each feature for threshold and category cuts
where the outcome distribution genuinely shifts, turns the survivors into binary "flags",
and fits a linear or logistic head on top — so every prediction is an exact sum of named,
human-readable rule contributions.

```python
from flaggam import FlagGAMClassifier

clf = FlagGAMClassifier(random_state=0).fit(X_train, y_train)

clf.export_rules()[["feature", "rule", "weight"]]
#      feature             rule    weight
#          age   age <= 27.4074  1.589593
#      purpose purpose == 'edu'  0.906901
clf.explain(x_row)   # per-row reason codes: which flags fired, each one's contribution
```

This package is a from-scratch Python implementation of FlagGAM (Zhao & Welsch,
arXiv:2605.31189) and the Univariate Flagging Algorithm it builds on (Sheth et al.,
PLOS ONE 2019).

## Highlights

- **Exact rule extraction** — every basis function is a concrete threshold, hinge, or
  category condition; `export_rules()` returns the full rule table with support, effect
  size, adjusted p-value, and fitted weight
  ([rules and screening](concepts/rules-and-screening.md)).
- **Row-level attribution without approximation** — `explain(X)` decomposes each
  prediction into the flags that fired and their individual contributions; the terms sum
  exactly to the model's score ([how it works](how-it-works.md)).
- **Screening you can defend** — two-proportion/chi-square/Welch tests with a
  Fisher-exact fallback, Benjamini–Hochberg FDR correction, and a support floor on both
  sides of every rule ([how it works](how-it-works.md)).
- **scikit-learn compatible** — `FlagGAMClassifier` / `FlagGAMRegressor` pass
  `check_estimator` and drop into pipelines and `GridSearchCV`
  ([getting started](getting-started.md)).
- **Missing values as first-class citizens** — by default a missing value never triggers
  a flag; opt in to screened missing-indicator rules
  ([missing values](concepts/missing-values.md)).
- **Extensions for regulated settings** — leak-free PD
  [calibration](concepts/calibration.md), *exact* per-feature
  [monotonicity](concepts/monotonicity.md), and a rule-level
  [fairness / proxy audit](concepts/fairness.md).
- **See the model** — six matplotlib plots and a dependency-free interactive HTML rules
  explorer ([visualization](concepts/visualization.md)), plus runners that reproduce the
  paper's benchmark tables ([benchmarks](concepts/benchmarks.md)).

## Where to start

1. [Getting started](getting-started.md) — install and your first rule basis in five
   minutes.
2. [How it works](how-it-works.md) — the full pipeline, from candidate cutoffs to the
   fitted additive head.
3. [German Credit walkthrough](notebooks/german_credit.ipynb) — a runnable notebook:
   rules, reason codes, calibration, monotonicity, fairness.

## Citation

If you use this package in research, please cite the papers it implements:

```
Zhao, Z. & Welsch, R. E. (2026).
FlagGAM: Rule-Basis Generalized Additive Models for Explainable Tabular Prediction.
arXiv:2605.31189.
```

```
Sheth, M., Gerovitch, A., Welsch, R. E., Markuzon, N. (2019).
The Univariate Flagging Algorithm (UFA): An interpretable approach for predictive modeling.
PLOS ONE 14(10): e0223161.
https://doi.org/10.1371/journal.pone.0223161
```

A machine-readable citation file is available at
[`CITATION.cff`](https://github.com/wlazlod/flaggam/blob/main/CITATION.cff).

## Project

[Changelog](changelog.md) · [Design decisions](DECISIONS.md) · [Licensing](LICENSING.md)
