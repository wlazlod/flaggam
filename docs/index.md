# FlagGAM

**Rule-basis generalized additive models for interpretable tabular prediction**

---

FlagGAM builds interpretable generalized additive models from tabular data. It works by
first running a univariate screening pass — the Univariate Flagging Algorithm (Sheth et
al., 2019) — that identifies threshold and category cuts where a feature's distribution
shifts meaningfully relative to the outcome. Each surviving cut becomes a binary basis
function ("flag"), and the flags together form a compact, human-readable rule basis.

A linear or logistic head is then fitted on top of the rule basis, producing a model
whose predictions are sums of individually interpretable flag contributions. Because
every flag corresponds to a concrete data condition (e.g., "age >= 55"), the resulting
model supports exact rule extraction and feature-level attribution without approximation. See
[Algorithm](user-guide/algorithm.md) for the full pipeline walkthrough, from candidate
generation through the fitted additive head.

This package is a from-scratch Python implementation of FlagGAM (Zhao & Welsch,
arXiv:2605.31189) and the Univariate Flagging Algorithm (Sheth et al., PLOS ONE 2019). It
provides sklearn-compatible `FlagGAMClassifier` and `FlagGAMRegressor` estimators that
integrate directly into standard scikit-learn pipelines.

## Why FlagGAM?

- **Exact rule extraction** — every basis function is a concrete threshold, hinge, or
  category condition; `export_rules()` returns the full rule table with support,
  effect size, p-value, and fitted weight.
- **Row-level attribution** — `explain(X)` decomposes each prediction into the flags
  that fired and their individual contributions, with no post-hoc approximation.
- **scikit-learn compatible** — `FlagGAMClassifier` / `FlagGAMRegressor` pass
  `check_estimator` and drop directly into pipelines, `GridSearchCV`, etc.
- **Extensible** — optional PD calibration, exact monotonicity constraints, and a
  fairness/proxy audit, each an original addition documented in
  [Extensions](user-guide/extensions.md).

## Quick Example

```python
import numpy as np
import pandas as pd
from flaggam import FlagGAMClassifier

rng = np.random.default_rng(0)
n = 600
age = rng.normal(40, 10, n)
purpose = rng.choice(["car", "tv", "edu"], n)
logit = -1.5 + 2.0 * (age <= 30) + 1.5 * (purpose == "edu")
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
X = pd.DataFrame({"age": age, "purpose": pd.Categorical(purpose)})

clf = FlagGAMClassifier(random_state=0).fit(X, y)

rules = clf.export_rules()
print(rules[["feature", "rule", "weight"]])
```

## Project Structure

```
flaggam/
├── src/flaggam/
│   ├── __init__.py          # Public API exports
│   ├── estimator.py          # FlagGAMClassifier / FlagGAMRegressor
│   ├── core.py                # Rule discovery and Z(X) construction
│   ├── screening.py           # UFA screening statistics
│   ├── bases.py                # Basis objects (one column of Z(X) each)
│   ├── missing.py             # Missing-indicator discovery
│   ├── heads.py                # Additive / flexible prediction heads
│   ├── weighting.py           # Feature weights and compact score
│   ├── inspection.py          # Rule export and per-row explanations
│   ├── calibration.py         # PD calibration (extension)
│   ├── monotonic.py           # Monotonicity constraints (extension)
│   ├── fairness.py             # Group metrics and proxy audit (extension)
│   ├── datasets.py             # Benchmark dataset loaders
│   └── plots.py                 # Matplotlib visualization helpers (optional viz extra)
├── tests/                    # Test suite
├── benchmarks/                # Paper-table reproduction runners
└── pyproject.toml
```

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
