# FAQ

**Why did screening find no rules?**
Every candidate must clear the support floor on *both* sides — by default
`min_support = min(200, max(20, ceil(0.02 * n_train)))` — and survive BH-FDR at
`fdr_alpha` (default 0.05). On small datasets or weak signals nothing may qualify; a
feature is skipped outright if it has fewer than `2 * min_support` non-missing values.
Pass a smaller explicit `min_support`, raise `fdr_alpha`, or bring more rows. See
[rules and screening](concepts/rules-and-screening.md).

**Why does `export_rules()` / `explain()` raise `ValueError`?**
You fitted with `representation="compact"`. The compact head's weights are per-class
scores over collapsed flag counts, not per-rule coefficients, so there is no rule table
to export. Refit with `representation="full"` (the default) to get rules and reason
codes.

**How are missing values handled — do I need to impute?**
No. By default (`missing="no_evidence"`) a missing value never triggers a flag and
contributes nothing to the score. With `missing="indicator"`, missingness itself is
screened like any other candidate and can become a `missing_indicator` rule. See
[missing values](concepts/missing-values.md).

**Why does importing the plots fail with an `ImportError`?**
The matplotlib helpers live behind the optional `viz` extra:
`pip install "flaggam[viz]"`. Importing `flaggam` itself never requires matplotlib —
the error is raised lazily, only when a plotting function is called.

**Why does `export_rules_html` refuse my estimator?**
The explorer needs per-rule weights, so it requires `representation="full"` and the
additive head, and supports binary classification and regression only (a multiclass
head has one coefficient vector per class, so there is no single shape per feature).
The same limits apply to `plot_shape`.

**Can I use a tree ensemble on top of the rules?**
Yes — pass any scikit-learn estimator as `head="flexible"`. It is fit on the rule basis
`Z(X)` with no access to the raw features, but you trade away additive
interpretability: `export_rules()` and `explain()` mark the result
`additive_interpretable=False` with `weight=NaN`.

**Is the monotonicity constraint exact or approximate?**
Exact. A feature's bases are themselves monotone step/ramp functions, so
sign-constraining their coefficients makes the fitted shape monotone at every value —
no post-hoc projection. Supported for `representation="full"` binary classification and
regression; a list-valued `C`/`alpha` falls back to `1.0`. See
[monotonicity](concepts/monotonicity.md).

**Can I put `CalibratedFlagGAM` inside a `Pipeline` or `GridSearchCV`?**
Not currently — it is a thin wrapper, not a full scikit-learn estimator, so it does not
support `clone()`. Tune the underlying `FlagGAMClassifier` first, then calibrate the
chosen configuration at the top level. The base estimators themselves pass
`check_estimator` and work anywhere scikit-learn estimators do.

**How close does this implementation get to the paper's numbers?**
The German Credit smoke benchmark lands at 0.773 AUROC against the paper's reported
0.775, and the runners reproduce the protocol of Tables 3, 4, 5, 7, and 8 with
`--n-splits 1000` defaults matching the paper. See
[benchmarks](concepts/benchmarks.md).
