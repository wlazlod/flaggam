# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `export_rules_html`: self-contained interactive HTML rules explorer (feature dropdown, shape curves, rule table; no dependencies)
- Documentation: detailed algorithm walkthrough with pipeline schema (`user-guide/algorithm.md`)

## [0.1.0] - 2026-07-07

### Added

- Core estimators: sklearn-compatible `FlagGAMClassifier` and `FlagGAMRegressor` with univariate screening, Benjamini–Hochberg FDR control, automatic missing-value basis discovery, feature weighting, rule export, and per-observation attribution
- Rule basis construction: threshold flags, categorical level flags, hinge and trend terms (regression), and missing-value indicators
- Benchmark suite: repeated-split protocol with training-only tuning carves; method registry with runners for paper Tables 3 (classification AUROC), 4 (regression RMSE/R²), 5 (robustness to missing values and feature noise), 7 (FlagGAM ablations), and 8 (hyperparameter sensitivity); German Credit smoke-acceptance test
- Dataset loaders with parquet caching: Pima, Breast Cancer, Heart, German Credit, Adult, Bank Marketing (classification); Ames, California Housing, Wine Quality (regression)
- PD calibration extension: diagnostics (reliability curves, Brier score, ECE, calibration-in-the-large) and recalibration methods (Platt, isotonic, base-rate offset) with cross-fitting to prevent data leakage
- Monotonicity constraints extension: sign-constrained additive heads for exact feature monotonicity in classification and regression, via box-constrained L-BFGS-B optimization
- Fairness extension: group fairness metrics (demographic parity, equal opportunity, AUROC gap) and rule-level proxy audit with binarized-indicator association ranking
- Visualization module: six matplotlib plots — feature shapes, basis importance, explanation waterfall, prediction reliability, fairness group metrics, protected-attribute association
- Documentation: MkDocs site with API reference via mkdocstrings, getting-started guide, user guides for rules/screening, extensions, benchmarks, and visualization; German Credit credit-approval guidebook as interactive Jupyter notebook
