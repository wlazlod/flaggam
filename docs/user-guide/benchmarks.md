# Benchmarks

Reproducing the paper's tables requires the `benchmarks` optional dependency group:

```bash
uv sync --extra benchmarks
```

Each runner produces one paper table as a tidy results CSV:

```bash
python -m benchmarks.run_classification   # Table 3 (classification AUROC)
python -m benchmarks.run_regression       # Table 4 (regression RMSE/R2)
python -m benchmarks.run_robustness       # Table 5 (missingness/noise robustness)
python -m benchmarks.run_ablation         # Table 7 (FlagGAM ablations)
python -m benchmarks.run_sensitivity      # Table 8 (hyperparameter sensitivity)
```

All runners default to `--n-splits 1000`, matching the paper, which takes hours per
table. Pass `--n-splits 25` for a quick pass while developing or sanity-checking a change.

Rows are always **appended** to `--out` if it already exists (this supports chunked
`--seed-start` resumption); delete the file first if you want a fresh run.

## Comparing Against the Paper

```bash
python -m benchmarks.render_tables benchmarks/results/classification.csv --table 3
```

`render_tables.py` compares a results CSV against the paper's reported values (Zhao &
Welsch, arXiv:2605.31189, `benchmarks/paper_targets.py`) and flags deltas beyond
tolerance. Results CSVs are written under `benchmarks/results/` and are gitignored — they
are run artifacts, not tracked outputs.

## Benchmark Protocol

The benchmark harness (`benchmarks/methods.py`, `benchmarks/_method_impls.py`) wraps
FlagGAM alongside comparison methods (EBM, RuleFit, XGBoost, and others) behind a common
`get_methods()` registry, so every runner exercises the same repeated-split, corruption,
and imputation protocol regardless of which method is under test. Methods that are not
installed are reported in a `skipped` dict with a human-readable reason rather than
raising — see [`docs/DECISIONS.md`](../DECISIONS.md) for the `GLRM`/`aix360` example.
