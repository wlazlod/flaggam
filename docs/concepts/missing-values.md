# Missing values

FlagGAM never imputes. The `missing` parameter controls what happens to observations
where a feature is NaN (numeric) or `None`/NaN (categorical), and takes one of two
values.

## `"no_evidence"` (the default)

Every ordinary basis — `threshold_low/high`, `hinge_low/high`, `category`, `trend` —
evaluates to `0.0` on missing input, so a missing value never triggers a flag and never
contributes to the additive score. This is the conservative choice: a missing observation
is treated as carrying no evidence either way, rather than being silently imputed into
whichever side of a cutoff happens to contain zero.

For the `trend` basis (regression's centered linear term), mapping missing input to
`0.0` is equivalent to imputing the feature mean — the one place where "no evidence"
coincides with a mean fill ([DECISIONS 9](../DECISIONS.md)).

## `"indicator"`

Missingness itself can carry signal — an empty income field on a credit application is
information. With `missing="indicator"`, in addition to the ordinary bases,
`missing.discover_missing_indicators` screens each feature's *missingness pattern*
against the outcome:

- the missing/non-missing split is tested like any other candidate (two-proportion test
  for binary outcomes, chi-square for multiclass and regression),
- both the missing and non-missing groups must satisfy `min_support`,
- the resulting p-values are BH-adjusted **across features** — one candidate per feature
  — rather than within one feature ([DECISIONS 13](../DECISIONS.md)).

Survivors become `missing_indicator` bases: `1{x is missing}`. This is the only basis
kind that fires *because* a value is missing rather than despite it. It shows up in
`export_rules()` and `explain()` like any other flag, with its own screening statistics
and fitted weight.

## Screening ignores missing rows

Candidate generation and screening for ordinary bases run on the observed (non-missing)
training values only: quantile cutoffs are computed over observed values, and a feature
is skipped entirely if it has fewer than `2 * min_support` non-missing observations. See
[How it works](../how-it-works.md) for where this sits in the pipeline.
