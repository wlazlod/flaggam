"""CLI: Table 8 sensitivity benchmark (flaggam refit under one-at-a-time overrides)."""

import argparse
from pathlib import Path
from typing import Any

from benchmarks.methods import flaggam_factory
from benchmarks.runner import RunConfig, _setup_logging, add_common_args, run_benchmark

# One-at-a-time overrides of the paper grid (spec Appendix A). Every value gets its
# own labeled row, including values that coincide with a FlagGAM default; "default"
# (no overrides) is added separately as the all-defaults baseline.
SENSITIVITY_GRID: list[tuple[str, dict[str, Any]]] = [
    ("fdr=0.01", {"fdr_alpha": 0.01}),
    ("fdr=0.05", {"fdr_alpha": 0.05}),
    ("fdr=0.10", {"fdr_alpha": 0.10}),
    ("step=0.025", {"quantile_step": 0.025}),
    ("step=0.05", {"quantile_step": 0.05}),
    ("step=0.10", {"quantile_step": 0.10}),
    ("minsup=100", {"min_support": 100}),
    ("minsup=200", {"min_support": 200}),
    ("minsup=400", {"min_support": 400}),
]

_CONDITIONS = ["clean", "miss50", "noise50"]

_DATASETS = ["bank_marketing"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Table 8 sensitivity benchmark.")
    add_common_args(
        p, conditions_help="Fixed to clean/miss50/noise50 for this runner; not overridable."
    )
    p.set_defaults(out="benchmarks/results/sensitivity.csv")
    return p


def main(argv: list[str] | None = None, registry: dict | None = None) -> None:
    """Run FlagGAM refit under each SENSITIVITY_GRID override, one label per row's method.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None).
        registry: Optional dataset registry for testing (injected by test harness).
    """
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.conditions is not None:
        parser.error("--conditions is fixed to clean/miss50/noise50 for the sensitivity runner")

    factories = {"default": flaggam_factory("binary")}
    factories.update(
        {label: flaggam_factory("binary", overrides) for label, overrides in SENSITIVITY_GRID}
    )

    cfg = RunConfig(
        datasets=args.datasets or _DATASETS,
        methods=args.methods,
        seeds=range(args.seed_start, args.seed_start + args.n_splits),
        conditions=_CONDITIONS,
        out=Path(args.out),
    )
    run_benchmark(cfg, task="binary", registry=registry, factories=factories)


if __name__ == "__main__":
    main()
