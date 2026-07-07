# API Reference

Full auto-generated API documentation from source code docstrings.

## Core

| Module | Description |
|--------|-------------|
| [`Estimators`](estimator.md) | `FlagGAMClassifier` / `FlagGAMRegressor` — the sklearn-compatible entry points |
| [`Core`](core.md) | Rule discovery and `Z(X)` basis-matrix construction |
| [`Screening`](screening.md) | UFA screening statistics used to select candidate rules |
| [`Bases`](bases.md) | Basis objects — one column of `Z(X)` each, with screening metadata |
| [`Missing`](missing.md) | Missing-indicator discovery |
| [`Heads`](heads.md) | Prediction heads (additive / flexible) fit on `Z(X)` |
| [`Weighting`](weighting.md) | Feature-weight statistics and the compact score representation |
| [`Inspection`](inspection.md) | Rule export (`export_rules`) and per-row reason codes (`explain`) |

## Extensions

Original additions beyond Zhao & Welsch (2026); see [Extensions](../user-guide/extensions.md).

| Module | Description |
|--------|-------------|
| [`Calibration`](calibration.md) | PD calibration diagnostics and recalibration |
| [`Monotonic`](monotonic.md) | Exact monotonicity constraints for the additive head |
| [`Fairness`](fairness.md) | Group metrics and rule-level proxy audit |

## Utilities

| Module | Description |
|--------|-------------|
| [`Datasets`](datasets.md) | Benchmark dataset loaders with local caching |
| [`Plots`](plots.md) | Matplotlib visualization helpers (optional `viz` extra) |
| [`Explorer`](explorer.md) | Self-contained interactive HTML rules explorer |
