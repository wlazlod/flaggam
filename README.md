# FlagGAM

[![Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

FlagGAM is a method for building interpretable generalized additive models from tabular data.
It works by first running a univariate screening pass — the Univariate Flagging Algorithm
(Sheth et al., 2019) — that identifies threshold and category cuts where a feature's
distribution shifts meaningfully relative to the outcome.  Each surviving cut becomes a binary
basis function ("flag"), and the flags together form a compact, human-readable rule basis.

A linear or logistic head is then fitted on top of the rule basis, producing a model whose
predictions are sums of individually interpretable flag contributions.  Because every flag
corresponds to a concrete data condition (e.g., "age ≥ 55 and in the high-risk tail"), the
resulting model supports exact rule extraction and feature-level attribution without
approximation.

This package is a from-scratch Python implementation of FlagGAM (Zhao & Welsch, arXiv:2605.31189)
and the Univariate Flagging Algorithm (Sheth et al., PLOS ONE 2019).  It provides
sklearn-compatible `FlagGAMClassifier` and `FlagGAMRegressor` estimators that integrate
directly into standard scikit-learn pipelines.

## Installation

```bash
# Editable install (development)
pip install -e .

# Or with uv
uv sync --extra dev
```

## Quickstart

```python
import pandas as pd
from flaggam import FlagGAMClassifier

# Toy dataset
df = pd.DataFrame({
    "age":    [25, 45, 55, 62, 34, 70, 28, 51],
    "income": [30, 55, 80, 95, 40, 110, 35, 75],
})
y = [0, 0, 1, 1, 0, 1, 0, 1]

# Fit
clf = FlagGAMClassifier(random_state=0)
clf.fit(df, y)

# Predict
proba = clf.predict_proba(df)

# Inspect the rule basis
rules = clf.export_rules()
print(rules)

# Attribution for one observation
explanation = clf.explain(df.iloc[[3]])
print(explanation)
```

## Citation

If you use this package in research, please cite the papers it implements:

**FlagGAM method:**

```
Zhao, Z. & Welsch, R. E. (2026).
FlagGAM: Rule-Basis Generalized Additive Models for Explainable Tabular Prediction.
arXiv:2605.31189.
```

**Univariate Flagging Algorithm:**

```
Sheth, M., Gerovitch, A., Welsch, R. E., Markuzon, N. (2019).
The Univariate Flagging Algorithm (UFA): An interpretable approach for predictive modeling.
PLOS ONE 14(10): e0223161.
https://doi.org/10.1371/journal.pone.0223161
```

A machine-readable citation file is available at [`CITATION.cff`](CITATION.cff).

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Status

Core estimators, screening, basis construction, missing-indicator discovery, feature weighting,
and rule inspection are implemented and pass `check_estimator`.  Benchmarks (German Credit and
other datasets), calibration, monotonic constraints, and fairness extensions are forthcoming.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).  See [`docs/LICENSING.md`](docs/LICENSING.md) for notes
on paper copyrights and dataset licences.
