# Getting started

## Install

```bash
pip install flaggam                  # core: numpy, pandas, scipy, scikit-learn
pip install "flaggam[viz]"           # + matplotlib plotting helpers
```

Python >= 3.11. For development, clone the repository and use the extras:

```bash
git clone https://github.com/wlazlod/flaggam.git
cd flaggam
uv sync --extra dev                  # tests, linting, type checking
uv sync --extra benchmarks           # paper-table reproduction runners
uv sync --extra docs                 # this documentation site
```

## First model

Rule discovery needs enough rows per tail (the `min_support` floor), so the example uses
600 synthetic rows with a planted signal:

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

Fitting runs the whole pipeline — candidate cutoffs, statistical screening with FDR
correction, winner selection, and an L2-penalized logistic head on the surviving flags.
[How it works](how-it-works.md) walks every stage.

## Read the rules

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

Each row of `export_rules()` is one discovered flag:

| Column | Meaning |
|---|---|
| `feature`, `kind` | source feature and basis kind (`threshold_low/high`, `category`, ...) |
| `rule` | the rendered condition, e.g. `age <= 27.4074` |
| `cutoff` / `level` | the numeric cutoff or categorical level |
| `support` | rows satisfying the condition in the training data |
| `effect_size`, `p_value`, `p_adj` | screening statistics; `p_adj` is BH-FDR adjusted |
| `weight` | the fitted head coefficient — the flag's additive contribution |

## Explain a prediction

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
individual contribution; the printed contributions sum exactly to the model's logit for
that row. The intercept row uses `feature == "<intercept>"`.

## Visualize it

```python
from flaggam import plot_shape, plot_rule_importance, export_rules_html

plot_shape(clf, "age")               # fitted additive contribution vs. value
plot_rule_importance(clf, top_n=20)  # top rules by |weight|
export_rules_html(clf, path="rules.html")   # interactive explorer, opens in any browser
```

The plots need the `viz` extra; `export_rules_html` produces a single dependency-free
HTML file — see [Visualization](concepts/visualization.md).

## Where next

- [How it works](how-it-works.md) — the pipeline from candidate cutoffs to the fitted
  additive head.
- [Concepts](concepts/rules-and-screening.md) — one page per topic: screening, missing
  values, calibration, monotonicity, fairness, benchmarks, visualization.
- [German Credit walkthrough](notebooks/german_credit.ipynb) — a runnable notebook.
- [API reference](api.md).
