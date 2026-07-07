# Getting Started

## Requirements

- Python >= 3.11

## Installation

### From Source

```bash
git clone https://github.com/wlazlod/flaggam.git
cd flaggam

# Editable install
pip install -e .

# Or with uv
uv sync --extra dev
```

### Optional Extras

```bash
uv sync --extra viz          # plotting helpers (matplotlib)
uv sync --extra benchmarks   # paper-table reproduction runners
uv sync --extra docs         # this documentation site
```

## Quick Start

Rule discovery requires enough rows per tail (`min_support`). The example below uses 600
synthetic rows with a planted signal so that `export_rules()` returns non-trivial rules.

### 1. Fit an Estimator

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
```

### 2. Inspect the Rule Basis

```python
rules = clf.export_rules()
print(rules[["feature", "rule", "weight"]])
#      feature             rule    weight
#          age   age <= 27.4074  1.589593
#          age   age >= 46.7581 -0.387614
#      purpose purpose == 'edu'  0.906901
#      purpose  purpose == 'tv' -0.418348
#      purpose purpose == 'car' -0.486362
```

Each row of `export_rules()` is one discovered flag: `feature`, `kind`, the rendered
`rule` string, its `cutoff`/`level`, `support`, `effect_size`, `p_value`/`p_adj`, and the
fitted `weight`.

### 3. Explain a Prediction

```python
x_young = pd.DataFrame({"age": [22.0], "purpose": pd.Categorical(["edu"])})
explanation = clf.explain(x_young)
print(explanation)
#    row     feature             rule  value  contribution
#      0         age   age <= 27.4074    1.0      1.589593
#      0     purpose purpose == 'edu'    1.0      0.906901
#      0 <intercept>      <intercept>    1.0     -0.624096
```

`explain(X)` decomposes each row's prediction into the flags that fired and their
individual contribution; the intercept row uses `feature == "<intercept>"`.

## What's Next?

- Learn how rules are discovered in [Rules & Screening](user-guide/rules-and-screening.md)
- Explore [Extensions](user-guide/extensions.md): calibration, monotonicity, fairness
- Reproduce the paper's tables with [Benchmarks](user-guide/benchmarks.md)
- Plot fitted models with [Visualization](user-guide/visualization.md)
- Browse the [API Reference](api/index.md)
