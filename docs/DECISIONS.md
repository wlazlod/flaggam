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
