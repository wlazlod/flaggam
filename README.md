# FlagGAM

[![Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

FlagGAM is a method for building interpretable generalized additive models from tabular data.
It works by first running a univariate screening pass — the Univariate Flagging Algorithm
(Sheth et al., 2019) — that identifies threshold and category cuts where a feature's
distribution shifts meaningfully relative to the outcome.  Each surviving cut becomes a binary
basis function ("flag"), and the flags together form a compact, human-readable rule basis.

A linear or logistic head is then fitted on top of the rule basis, producing a model whose
predictions are sums of individually interpretable flag contributions.  Because every flag
corresponds to a concrete data condition (e.g., "age ≥ 55"), the
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

Rule discovery requires enough rows per tail (min_support).  The example below
uses 600 synthetic rows with a planted signal so that `export_rules()` returns
non-trivial rules.

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

# Inspect the rule basis (5 rules on this seed)
rules = clf.export_rules()
print(rules[["feature", "rule", "weight"]])
#      feature             rule    weight
#          age   age <= 27.4074  1.589593
#          age   age >= 46.7581 -0.387614
#      purpose purpose == 'edu'  0.906901
#      purpose  purpose == 'tv' -0.418348
#      purpose purpose == 'car' -0.486362

# Attribution for a young 'edu' applicant
x_young = pd.DataFrame({"age": [22.0], "purpose": pd.Categorical(["edu"])})
explanation = clf.explain(x_young)
print(explanation)
#    row     feature             rule  value  contribution
#      0         age   age <= 27.4074    1.0      1.589593
#      0     purpose purpose == 'edu'    1.0      0.906901
#      0 <intercept>      <intercept>    1.0     -0.624096
```

## Extensions (beyond the paper)

Three optional modules extend the paper's method; each is an original
addition not present in Zhao & Welsch (2026) and lives in its own module.

**PD calibration** — diagnostics (reliability curve, Brier, ECE,
calibration-in-the-large) and recalibration (`platt`, `isotonic`,
`base_rate`) fitted on data disjoint from head fitting:

```python
from flaggam import CalibratedFlagGAM, expected_calibration_error

cal = CalibratedFlagGAM(FlagGAMClassifier(random_state=0), method="platt", cv=5)
cal.fit(X, y)
pd_hat = cal.predict_proba(X)[:, 1]
```

**Monotonicity constraints** — regulators often require PD monotone in a
feature. Because FlagGAM's numerical contributions are step/ramp bases,
sign constraints give exact monotonicity:

```python
clf = FlagGAMClassifier(monotonic_constraints={"age": -1}).fit(X, y)  # PD non-increasing in age
```

**Fairness / proxy audit** — group metrics for a protected attribute and a
rule-level audit that ranks bases by association with it:

```python
from flaggam import ProxyAudit, group_metrics

metrics = group_metrics(y, clf.predict_proba(X)[:, 1], A)
report = ProxyAudit(clf).report(X, A)          # ranked candidate proxies
clean_clf, trade = ProxyAudit(clf).drop_proxies(X, y, A, threshold=0.3)
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

## Benchmarks

Reproducing the paper's tables requires the `benchmarks` optional dependency group:

```bash
uv sync --extra benchmarks
```

Each runner produces one paper table as a tidy results CSV:

```bash
python -m benchmarks.run_classification   # Table 3 (classification AUROC)
python -m benchmarks.run_regression       # Table 4 (regression RMSE/R2)
python -m benchmarks.run_robustness       # Table 5 (missingness/noise robustness)
python -m benchmarks.run_ablation         # Table 7 (FlagGAM ablations)
python -m benchmarks.run_sensitivity      # Table 8 (hyperparameter sensitivity)
```

All runners default to `--n-splits 1000`, matching the paper, which takes hours per table.
Pass `--n-splits 25` for a quick pass while developing or sanity-checking a change.

Rows are always APPENDED to `--out` if it already exists (this supports chunked
`--seed-start` resumption); delete the file first if you want a fresh run.

```bash
python -m benchmarks.render_tables benchmarks/results/classification.csv --table 3
```

`render_tables.py` compares a results CSV against the paper's reported values
(Zhao & Welsch, arXiv:2605.31189, `benchmarks/paper_targets.py`) and flags deltas beyond
tolerance. Results CSVs are written under `benchmarks/results/` and are gitignored — they are
run artifacts, not tracked outputs.

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
