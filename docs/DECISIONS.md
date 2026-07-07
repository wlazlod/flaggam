# Implementation Decisions

Resolved ambiguities and design choices made during implementation.
Each entry references the source that drove the decision.

---

1. **Effect size for same-side selection = absolute risk difference; `"log_odds_ratio"` also accepted.**
   Binary same-side selection ranks candidates by absolute risk difference (proportion in tail minus
   proportion in baseline) unless the caller passes `effect_size="log_odds_ratio"`.  Multiclass
   same-side selection uses Cramér's V of the 2×K contingency table (tail vs. baseline), because no
   directional effect is defined for K > 2.
   *(spec §6.1; this plan, Task 4)*

2. **Baseline region = complement of the tail; central alternative noted, not implemented.**
   The baseline for every statistical test is all observations that do not fall in the tail region.
   A "central only" baseline (excluding both tails in a two-sided setting) was considered but is not
   implemented; the complement definition is simpler and matches the paper.
   *(spec §6.2)*

3. **Tail test for binary outcomes = two-proportion z-test (two-sided); Fisher's exact fallback.**
   When any expected cell count in the 2×2 table is below 5 the implementation switches to
   Fisher's exact test automatically.  No user-facing switch is provided.
   *(spec §6.3)*

4. **Multiclass tail test = Pearson chi-square on the 2×K contingency table.**
   The 2×K table has one row for the tail and one for the baseline, with one column per class.
   Pearson chi-square is used rather than likelihood-ratio chi-square because it matches the
   standard two-proportion z-test in the binary case (K = 2).
   *(spec §6.4)*

5. **Regression screening = Welch t-test; ranking by absolute standardised mean difference.**
   The continuous response is discretised into tail vs. baseline by the threshold/category under
   test; the Welch t-test (unequal variances) tests whether the means differ.  Candidates are
   ranked by |SMD| so that effect size drives selection even when power varies.  The centered trend
   term is added unconditionally and is not subject to screening.  Only the Welch t-test is
   implemented in v1; the spec's "expose the test choice" parameter is deferred (documented
   omission).
   *(spec §6.5; paper Fig. 1)*

6. **Feature weights: point-biserial for binary, correlation ratio (η) for regression, Cramér's V for multiclass.**
   Each weight measures the association strength between the flag indicator and the outcome.
   Point-biserial correlation is equivalent to Pearson correlation when one variable is binary.
   *(spec §6.6)*

7. **Compact score = per-class weighted sum of triggered flags mapped to their enriched class; classification only.**
   The compact score sums, for each class c, the feature-weighted contributions of all flags whose
   enriched class is c.  Hinge and trend bases are excluded because they are not class-attributed.
   The default model uses full Z(X) (all basis evaluations) rather than the compact score.
   *(spec §6.7)*

8. **Z(X) enters the additive head unstandardised.**
   No additional standardisation is applied to the basis matrix before it is passed to the linear
   or logistic head.  Table 6 of the paper implies direct use of Z(X) outputs without a
   re-scaling step.
   *(paper Table 6)*

9. **Missing `TrendBasis` input contributes 0 (== feature mean).**
   When the feature required by a `TrendBasis` is absent from X at transform time, the basis
   returns 0 for all observations.  This is equivalent to imputing the feature mean before
   centering, which is the neutral (no-signal) contribution.  The paper is silent on this edge
   case.
   *(paper silent; defensive choice)*

10. **Baseline region must satisfy `min_support`; support floor is symmetric.**
    The `min_support` threshold applies to both the tail and the baseline.  A candidate is
    discarded if either region has fewer than `min_support` observations.  The paper states the
    tail-support constraint only; the baseline floor was added to prevent degenerate tests against
    near-empty baselines.
    *(paper states tail support only; extended symmetrically for robustness)*

11. **Ties in same-side selection resolved by: larger effect → smaller p → lower cutoff.**
    When two candidates have identical effect size and p-value the one with the lower cutoff
    value is preferred.  This rule is fully deterministic and requires no tie-breaking
    randomness.  The lower-cutoff preference is more extreme on the low side (tighter low
    tail) and less extreme on the high side (wider high tail).
    *(determinism requirement)*

12. **`min_support` applies to categorical "rest" group.**
    For a categorical feature with K levels, the "rest" group (all observations not in the level
    under test) must also satisfy `min_support`.  This ensures that the two-proportion test has
    adequate power on both sides and is consistent with decision 10 (symmetric support floor).
    *(extension of min_support semantics to level-vs-rest comparisons)*

13. **Missing-indicator screening applies Benjamini–Hochberg across features.**
    `discover_missing_indicators` generates one candidate per feature (one test each) and
    applies BH correction across those candidates before filtering by `fdr_alpha`.  This
    matches the per-feature-one-candidate structure in `missing.py`'s docstring.
    *(implementation decision; paper is silent on multiple-comparison correction for missing
    indicators)*

14. **Tuning validation carve = 80/20 of the training split, stratified for classification.**
    Methods that need internal hyperparameter tuning (e.g. `xgboost`) carve their validation
    fold from the training split only, never from the held-out test split.  The carve is 80%
    train / 20% validation, stratified on the label for classification tasks.  Spec §10 leaves
    the exact fraction open; 80/20 is the conventional default.
    *(spec §10)*

15. **Smoke-benchmark tolerance ±0.02 at 25 splits; the spec's ±0.010 applies to full
    1000-split reproductions.**
    `tests/test_smoke_benchmark.py` runs 25 stratified splits on German Credit, not the
    paper's 1000.  The 95% CI half-width of a 25-split mean (≈1.96·sd/√25 ≈ 0.011 at the
    observed sd≈0.029) exceeds the ±0.010 spec tolerance, so the smoke test uses a wider
    ±0.02 band around the paper's reported 0.775.  The tighter ±0.010 spec tolerance is
    reserved for full 1000-split reproduction runs.
    *(spec §14; test tolerance discussion)*

16. **GLRM/aix360 status: wrapped but unavailable in this environment; excluded from the
    benchmarks extra.**
    `GLRMMethod` in `benchmarks/_method_impls.py` is fully implemented behind a guarded
    import of `aix360.algorithms.rbm`.  `aix360` was added to the `benchmarks` optional
    dependency group and installed successfully, but importing it transitively requires
    `cvxpy`, which `aix360` does not declare as a dependency and which is not installed;
    the import therefore fails.  The `aix360` addition was reverted from
    `pyproject.toml`/`uv.lock` rather than shipping a non-functional optional dependency, so
    `aix360` is not installed at all in this environment.  `get_methods()` reports `"glrm"` in
    its `skipped` dict with reason `"aix360 not installed: No module named 'aix360'"`, and
    `glrm` is silently excluded from any run rather than raising.
    *(guarded-import design)*

17. **Pima: physiologically impossible zeros treated as missing (NaN) for all methods.**
    In the Pima Indians Diabetes dataset, zero values in `{glucose, blood_pressure,
    skin_thickness, insulin, bmi}` are not valid measurements (they cannot be zero in a living
    patient) and are recoded to `NaN` at load time.  This recoding is applied once in the
    loader, so every method sees the same missingness pattern uniformly — no method benefits
    from treating impossible zeros as real values.
    *(dataset loader design)*

18. **Table 7 ablation uses the ρ=0.50 corruption setting for its "missing"/"noisy" columns.**
    `run_ablation.py` reports FlagGAM ablation variants (rule count, head type, etc.) across a
    `clean` column and `missing`/`noisy` columns.  The `missing`/`noisy` columns use the
    ρ=0.50 corruption conditions (`miss50`/`noise50`), matching the more stressful of the two
    corruption levels used elsewhere (Table 5), rather than ρ=0.25.
    *(spec Table 7; consistent with Table 5's stress-test intent)*

19. **PD calibration (`flaggam/calibration.py`): cross-fitting with a single pooled
    calibrator; logit-space Platt; base_rate as a brentq logit offset; binary-only scope.**
    `CalibratedFlagGAM` wraps any fitted-or-unfitted `FlagGAMClassifier`-compatible estimator
    to recalibrate `P(y=1)`. For `cv=k` (int), `StratifiedKFold(k, shuffle=True,
    random_state=0)` fits `sklearn.clone(estimator)` on each fold's k−1 training splits and
    collects out-of-fold predictions; exactly one calibrator is then fitted on the pooled
    out-of-fold predictions across all folds (not one calibrator per fold), and the final
    `estimator_` is a fresh clone refit on the full `(X, y)`. This keeps the calibrator's
    training data disjoint from the head-fitting data for every observation, satisfying the
    no-leakage requirement, while still producing a single deployable calibrator rather than
    an ensemble of fold-specific ones. `cv="prefit"` treats `estimator` as already fitted and
    uses `(X, y)` passed to `fit()` purely as held-out calibration data, so the caller is
    responsible for disjointness from whatever data trained `estimator`. All three methods
    operate on the model logit `log(p/(1-p))` with `p` clipped to `[1e-12, 1-1e-12]` before
    the log: `platt` fits a near-unregularized `LogisticRegression(C=1e6, solver="lbfgs")` of
    `y` on the logit; `isotonic` fits `IsotonicRegression(out_of_bounds="clip", y_min=0.0,
    y_max=1.0)` directly on `p`; `base_rate` solves for a scalar logit offset `delta` with
    `scipy.optimize.brentq` on `[-30, 30]` such that `mean(expit(logit + delta))` equals
    `target_rate` (raises `ValueError` if `target_rate` is not supplied), and applies that
    offset to the final model's logits at predict time — an intercept shift estimated from
    the leak-free pooled out-of-fold logits, so no per-fold pooling of the offset itself is
    needed. `CalibratedFlagGAM` raises `ValueError` for non-binary targets: PD calibration is
    defined only for `P(y=1)`. This module is an original addition; Zhao & Welsch (2026)
    evaluates only ranking metrics and does not address probability calibration.
    *(spec §8.1; original addition, not in paper)*

20. **Monotonicity (`flaggam/monotonic.py`): sign-constrained bases give exact
    monotonicity; box-constrained L-BFGS-B head; binary + regression scope; list-valued
    C/alpha fall back to 1.0.**
    `monotonic_constraints` is a dict `{feature_name: +1|-1|0}`; `+1` means risk
    non-decreasing in the feature, `-1` non-increasing, `0` (or an absent key)
    unconstrained. Because every numerical feature's bases are themselves monotone
    step/ramp functions of x — tail flags (`threshold_low`/`threshold_high`), tail
    hinges (`hinge_low`/`hinge_high`), and the linear `trend` term — constraining each
    coefficient's sign is sufficient (not merely heuristic) to make the resulting
    additive contribution exactly monotone in x; no post-hoc isotonic projection is
    needed. `category` and `missing`(`_indicator`) bases are never constrained (spec
    §8.2): a categorical level or an is-missing flag has no defined "direction" in x.
    `bounds_for_bases(bases, constraints)` maps each basis to a `(low, high)` box bound
    on its coefficient per this table (for `+1`; `-1` mirrors by swapping the tuple):

    | basis kind        | active for | bound on θ    |
    |-------------------|------------|---------------|
    | `threshold_low`   | small x    | `(None, 0.0)` |
    | `threshold_high`  | large x    | `(0.0, None)` |
    | `hinge_low`       | small x    | `(None, 0.0)` |
    | `hinge_high`      | large x    | `(0.0, None)` |
    | `trend`           | linear     | `(0.0, None)` |
    | `category`, `missing` | —      | `(None, None)`|

    `MonotonicAdditiveHead` is a drop-in replacement for `AdditiveHead` when
    `monotonic_constraints is not None`; it fits an intercept-unpenalized,
    intercept-unbounded L2-penalized loss with `scipy.optimize.minimize(method=
    "L-BFGS-B")` under the per-coefficient box bounds, matching sklearn's primal
    objective forms: binary is `F(θ, b) = 0.5·θᵀθ + C·Σᵢ[logaddexp(0, mᵢ) − yᵢmᵢ]` with
    `m = Zθ + b` (gradients `∂θ = θ + C·Zᵀ(expit(m) − y)`, `∂b = C·Σ(expit(m) − y)`);
    regression is ridge, `F(θ, b) = ‖y − m‖² + α·θᵀθ` (gradients `∂θ = −2Zᵀr + 2αθ`,
    `∂b = −2Σr` with `r = y − m`). The estimator routes `C`/`alpha` straight through
    when scalar; when the caller passed a list/tuple (the `AdditiveHead` CV-tuning
    convention), the constrained path falls back to the spec default of `1.0` — CV
    tuning of the constrained head is out of scope for this task. `FlagGAMClassifier`/
    `FlagGAMRegressor.fit` raise `ValueError` when `monotonic_constraints` is set with
    `head != "additive"` (a flexible estimator has no per-basis coefficient to
    constrain) or, for the classifier, `representation == "compact"` (compact score
    columns don't map 1:1 to feature bases, so there is nothing to constrain a sign
    on); they raise `ValueError` for unknown feature names or out-of-`{+1,-1,0}`
    values, and `NotImplementedError` for a resolved multiclass task — monotonic
    constraints support binary classification and regression only.
    *(spec §8.2; original addition, not in paper)*

21. **Fairness (`flaggam/fairness.py`): group metrics with max-minus-min gaps;
    binarized-indicator association (Cramér's V vs. |point-biserial|) for the
    proxy audit; `threshold=0.2` is a screening heuristic, not a legal
    standard; `drop_proxies` scoped to fitted binary/`representation="full"`/
    `head="additive"` classifiers.**
    `group_metrics(y_true, y_prob, A, threshold=0.5, n_bins=10)` computes, per
    level of the protected attribute `A`, `n`, `base_rate`, `mean_predicted`,
    `selection_rate` (fraction with `y_prob >= threshold`), `tpr` (selection
    rate among true positives), `auroc`, and `ece` (via `expected_calibration_
    error` from Task E1); it raises `ValueError` for non-binary `y_true`,
    matching the binary-only PD scope used elsewhere in the fairness/
    calibration modules. `auroc` and `tpr` are `NaN` for single-class groups
    (no ranking/positive-rate signal is defined there) and such groups are
    excluded from the corresponding gap. The three top-level `gaps` —
    `demographic_parity_diff`, `equal_opportunity_diff`, `auroc_gap` — are
    each the max-minus-min of the per-group column across the non-`NaN`
    groups; this is the simplest, most common operationalization of "gap"
    fairness metrics and requires no reference/privileged group to be named.

    `ProxyAudit(estimator).report(X, A, threshold=0.2)` ranks every fitted
    rule basis in `estimator.core_.bases_` by its association with `A`. The
    association is computed on the BINARIZED basis indicator `z = (Z(X)[:,
    j] > 0)`, not on the raw (possibly continuous) basis output: this is
    exact for `threshold_low/high`, `category`, and `missing_indicator`
    bases, which are already 0/1-valued, but is a documented approximation
    for `hinge_low/high` and `trend` bases, which are continuous — the audit
    only asks whether the rule fires, not by how much. When `A` is numeric
    (`np.issubdtype(A.dtype, np.number)`), association is `|point-biserial
    r|` between the binary indicator and `A`; otherwise (categorical/object
    `A`) association is Cramér's V of the indicator-by-`A` contingency table.
    Both statistics degenerate to `0.0` when the indicator has fewer than two
    distinct values (a basis that never/always fires carries no association
    signal, avoiding a divide-by-zero). The report is sorted descending by
    association (stable sort, ties keep basis-discovery order) and
    `flagged = association > threshold`. The default `threshold=0.2` is a
    heuristic screening level borrowed from conventional "small-to-moderate
    effect" cutoffs for these statistics — it is not a legal or regulatory
    fair-lending standard (e.g., not the four-fifths rule) and callers doing
    compliance work must supply their own threshold.

    `ProxyAudit.drop_proxies(X, y, A, threshold=0.2)` deep-copies the fitted
    estimator, drops every basis whose `rule` (its unique `.name`) is flagged
    by `report`, and refits ONLY the head — a fresh `AdditiveHead("binary",
    C=estimator.C, random_state=estimator.random_state)` — on the reduced
    `Z(X)`; rule discovery itself is not re-run. It raises `ValueError`
    unless the estimator is a fitted binary classifier (`len(classes_) ==
    2`) with `representation="full"` and `head="additive"`: compact-score
    columns are feature-weighted sums across multiple bases and don't map
    1:1 back to a single dropped basis, a `FlexibleHead` has no per-basis
    coefficient to drop, and multiclass PD gaps are out of scope for this
    module (consistent with `group_metrics` and `CalibratedFlagGAM`).  It
    returns `(new_estimator, trade)` where `trade` is a one-row DataFrame of
    `n_dropped`, `auroc_before/after`, and `dp_diff_before/after` (the
    `demographic_parity_diff` gap from `group_metrics`), so a caller can see
    the fairness/accuracy trade-off of the drop in one call. The original
    estimator's `core_.bases_` is untouched because `drop_proxies` mutates
    only the deep-copied `new_est.core_.bases_` list.
    *(original addition, not in paper; operationalizes the paper's Impact
    Statement on proxy/bias risk in selected rules)*
