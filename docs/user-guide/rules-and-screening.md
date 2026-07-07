# Rules & Screening

FlagGAM turns each feature into a small set of binary "flags" — threshold, hinge, or
category conditions — through a univariate screening pass, then fits a linear/logistic
head on top of the resulting basis matrix `Z(X)`. This page explains how that screening
pass works: the Univariate Flagging Algorithm (UFA), the multiple-testing correction
applied across candidates, and the support floor that keeps individual rules statistically
meaningful. See [Algorithm](algorithm.md) for the full walkthrough, including the pipeline
schema and how the additive head is fit on top of the resulting basis matrix.

## The Univariate Flagging Algorithm (UFA)

For each feature, UFA scans candidate cutoffs (numeric features) or levels (categorical
features) and tests whether the outcome distribution differs between the "tail" (rows on
one side of the cutoff, or in that category) and the "baseline" (everything else):

- **Binary outcome** — a two-sided two-proportion z-test compares the positive-class rate
  in the tail vs. the baseline. When any expected cell count in the 2x2 table falls below
  5, the test automatically falls back to Fisher's exact test.
- **Multiclass outcome** — a Pearson chi-square test on the 2xK (tail/baseline x class)
  contingency table, since no single directional effect is defined for K > 2 classes.
- **Regression outcome** — a Welch t-test (unequal variances) on the continuous response,
  tail vs. baseline.

A candidate only becomes a basis function ("flag") if it survives screening: its p-value
(after FDR correction, below) must be at or under `fdr_alpha`, and it must satisfy the
support floor described below.

Candidates are ranked by effect size so that practically meaningful cuts are preferred
over merely significant ones:

- Binary same-side selection ranks by absolute risk difference (`effect_size="risk_difference"`,
  the default) or by absolute log-odds ratio (`effect_size="log_odds_ratio"`).
- Multiclass same-side selection ranks by Cramer's V of the 2xK contingency table.
- Regression screening ranks by absolute standardized mean difference (SMD).

## Basis kinds

Each surviving candidate becomes one column of `Z(X)`, tagged with a `kind`:

| Kind | Meaning |
|------|---------|
| `threshold_low` / `threshold_high` | Step indicator: feature <= cutoff / >= cutoff |
| `hinge_low` / `hinge_high` | Ramp (hinge) function anchored at a cutoff |
| `trend` | Centered linear term for the feature |
| `category` | Indicator for one categorical level |
| `missing_indicator` | Indicator that the feature was missing for this row |

`est.core_.bases_` holds the full list of fitted `Basis` objects (see the
[Bases API](../api/bases.md)); each has `.feature`, `.kind`, `.name`, and a `.transform(x)`
method, plus kind-specific attributes (`.cutoff`/`.side` for threshold/hinge bases,
`.level` for category bases, `.mean` for trend bases).

## Multiple-testing correction (BH-FDR)

Screening a feature typically produces several candidate cutoffs, which inflates the
false-positive rate if each is tested at a flat significance threshold. FlagGAM applies
Benjamini-Hochberg (BH) FDR correction across a feature's candidates before filtering by
`fdr_alpha` (default `0.05`), so the reported `p_adj` in `export_rules()` already accounts
for the number of candidates considered for that feature. Missing-indicator screening
applies the same BH correction, but across features (one candidate per feature) rather
than within a feature.

## Support floor (`min_support`)

A candidate cutoff or level is only tested if both the tail and the baseline have at
least `min_support` observations — a symmetric floor that keeps both sides of every test
adequately powered, and prevents rules from being fit against a near-empty group. By
default (`min_support="auto"`), this floor is computed from the training size:

```python
min_support = min(200, max(20, ceil(0.02 * n_train)))
```

i.e. 2% of the training rows, floored at 20 and capped at 200. Pass an explicit integer to
override it. For categorical features, the "rest" group (all rows not in the level under
test) must also satisfy `min_support`.

## Inspecting the result

Once fitted, `export_rules()` returns one row per surviving basis with its `feature`,
`kind`, rendered `rule` string, `cutoff`/`level`, `support`, `effect_size`, `p_value`,
`p_adj`, `enriched_class` (classification only), fitted `weight`, and whether the flag is
`additive_interpretable`. `explain(X)` decomposes individual predictions the same way,
listing the fired flags and their per-row contributions. See
[Inspection](../api/inspection.md) for the full column reference.
