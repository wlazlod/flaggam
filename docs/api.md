# API reference

## Estimators

`FlagGAMClassifier` / `FlagGAMRegressor` — the sklearn-compatible entry points.

::: flaggam.estimator

## Core

Rule discovery and `Z(X)` basis-matrix construction.

::: flaggam.core

## Screening

UFA screening statistics used to select candidate rules.

::: flaggam.screening

## Bases

Basis objects — one column of `Z(X)` each, with screening metadata.

::: flaggam.bases

## Missing

Missing-indicator discovery.

::: flaggam.missing

## Heads

Prediction heads (additive / flexible) fit on `Z(X)`.

::: flaggam.heads

## Weighting

Feature-weight statistics and the compact score representation.

::: flaggam.weighting

## Inspection

Rule export (`export_rules`) and per-row reason codes (`explain`).

::: flaggam.inspection

## Calibration

PD calibration diagnostics and recalibration. An original addition — see
[Calibration](concepts/calibration.md).

::: flaggam.calibration

## Monotonic

Exact monotonicity constraints for the additive head. An original addition — see
[Monotonicity](concepts/monotonicity.md).

::: flaggam.monotonic

## Fairness

Group metrics and rule-level proxy audit. An original addition — see
[Fairness](concepts/fairness.md).

::: flaggam.fairness

## Datasets

Benchmark dataset loaders with local caching.

::: flaggam.datasets

## Plots

Matplotlib visualization helpers (optional `viz` extra).

::: flaggam.plots

## Explorer

Self-contained interactive HTML rules explorer.

::: flaggam.explorer
