# Visualization

## Interactive rules explorer

`export_rules_html` renders a fitted estimator's rules as a single dependency-free HTML
file: no network access, no external CSS/JS, safe to open offline or embed in an iframe.
It shows a feature-selector dropdown, the fitted shape curve (or level bars for categorical
features), and a table of the rules contributing to that feature. The embed below is
generated from a `FlagGAMClassifier` fit on the German Credit dataset — regenerate it
with `uv run python scripts/make_rules_explorer.py`.

```python
from flaggam import export_rules_html

export_rules_html(clf, path="rules.html")  # open in any browser
```

<iframe src="../../assets/rules-explorer.html" width="100%" height="640"
        style="border:1px solid var(--md-default-fg-color--lightest);border-radius:6px;"></iframe>

`flaggam.plots` provides matplotlib helpers for fitted estimators and diagnostics. This
module requires the optional `viz` extra:

```bash
uv sync --extra viz
```

Importing `flaggam` never imports matplotlib eagerly — each plotting function calls a
lazy import helper first, raising a clear `ImportError` naming `pip install flaggam[viz]`
if matplotlib is not installed. Every function accepts an optional `ax` and returns the
`Axes` it drew on, so plots compose with existing matplotlib figures.

## Shape function — `plot_shape`

Plots the fitted additive contribution for one feature: a step curve of contribution vs.
value (with a rug of the discovered cutoffs) for numeric features, or one bar per
discovered level for categorical features. Requires `representation="full"`.

```python
from flaggam import plot_shape

ax = plot_shape(clf, "age")
```

## Rule importance — `plot_rule_importance`

Horizontal bar chart of the top-`N` rules by `|weight|`, read directly from
`export_rules()`.

```python
from flaggam import plot_rule_importance

ax = plot_rule_importance(clf, top_n=20)
```

## Waterfall — `plot_waterfall`

Cumulative horizontal bars from the intercept to the total score for a single row,
sourced from `explain(x_row)`. Rules beyond `max_rules` collapse into a single
"(other rules)" bucket so the chart stays readable.

```python
from flaggam import plot_waterfall

x_row = X.iloc[[0]]
ax = plot_waterfall(clf, x_row, max_rules=15)
```

## Reliability diagram — `plot_reliability`

Mean predicted probability vs. observed positive fraction per bin, with a per-bin count
overlay, built on `reliability_curve` from `flaggam.calibration`.

```python
from flaggam import plot_reliability

ax = plot_reliability(y_test, clf.predict_proba(X_test)[:, 1], n_bins=10)
```

## Proxy association — `plot_proxy_association`

Horizontal bars of rule-level association with a protected attribute, from a
`ProxyAudit(...).report(...)` DataFrame; bars whose association exceeds the audit
threshold are highlighted, with a dashed line marking the approximate cutoff.

```python
from flaggam import plot_proxy_association, ProxyAudit

report = ProxyAudit(clf).report(X, A)
ax = plot_proxy_association(report, top_n=20)
```

## Group metrics — `plot_group_metrics`

Grouped bar chart of `selection_rate`, `tpr`, and `auroc` per level of a protected
attribute, from a `group_metrics(...)` result; the title annotates the computed fairness
gaps.

```python
from flaggam import plot_group_metrics, group_metrics

metrics = group_metrics(y_test, clf.predict_proba(X_test)[:, 1], A_test)
ax = plot_group_metrics(metrics)
```

See the [API reference](../api.md#plots) for the full API.
