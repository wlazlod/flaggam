# Fairness

Zhao & Welsch (2026) warn that selected rules may encode bias or act as proxies for
protected attributes. `flaggam.fairness` operationalizes that warning: group-level
performance metrics for a protected attribute, plus a rule-level audit that ranks every
fitted basis by its association with it. Because FlagGAM's model *is* its rules, the
audit inspects the actual decision logic — not a surrogate.

This module is an original addition — not part of Zhao & Welsch (2026); the design
rationale is recorded in [DECISIONS](../DECISIONS.md).

## Group metrics

```python
from flaggam import group_metrics

A = X["purpose"].astype(str)   # protected attribute (illustrative)
metrics = group_metrics(y, clf.predict_proba(X)[:, 1], A)
```

`group_metrics` reports, per level of `A`: `n`, `base_rate`, `mean_predicted`,
`selection_rate`, `tpr`, `auroc`, and `ece`, plus three gap summaries
(`demographic_parity_diff`, `equal_opportunity_diff`, `auroc_gap`) computed as the
max-minus-min across groups.

## Proxy audit

```python
from flaggam import ProxyAudit

report = ProxyAudit(clf).report(X, A)                        # ranked candidate proxies
clean_clf, trade = ProxyAudit(clf).drop_proxies(X, y, A, threshold=0.3)
```

`ProxyAudit.report` ranks every fitted basis by its association with `A` — absolute
point-biserial correlation for numeric `A`, Cramer's V otherwise — and flags those above
`threshold`. `ProxyAudit.drop_proxies` refits only the head after removing flagged
bases, returning the new estimator alongside a one-row trade-off summary (`n_dropped`,
AUROC and demographic-parity-gap before/after), so the cost of removing a proxy is a
number, not a guess.

Both methods require a fitted binary classifier with `representation="full"` and
`head="additive"` (no monotonic constraints).

`plot_group_metrics` and `plot_proxy_association` visualize both outputs — see
[Visualization](visualization.md); the
[German Credit walkthrough](../notebooks/german_credit.ipynb) runs the full audit on an
illustrative sex attribute. See the [API reference](../api.md#fairness) for the full API.
