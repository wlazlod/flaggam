"""CLI: Table 7 ablation benchmark (compact vs. full representation x additive vs. RF head)."""

import argparse
from pathlib import Path

from benchmarks.methods import get_methods
from benchmarks.runner import RunConfig, _setup_logging, add_common_args, run_benchmark

_ABLATION_VARIANTS = [
    "flaggam_compact_equal",
    "flaggam_compact_weighted",
    "flaggam_full_additive",
    "flaggam_full_rf",
]

# Table 7 reports clean/missing/noisy at rho=0.50, not the full corruption sweep
# (spec DECISIONS entry 18).
_CONDITIONS = ["clean", "miss50", "noise50"]

_DATASETS = ["heart", "adult", "bank_marketing"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Table 7 ablation benchmark.")
    add_common_args(
        p, conditions_help="Fixed to clean/miss50/noise50 for this runner; not overridable."
    )
    p.set_defaults(out="benchmarks/results/ablation.csv")
    return p


def main(argv: list[str] | None = None, registry: dict | None = None) -> None:
    """Run the ablation benchmark over the four FlagGAM representation/head variants.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None).
        registry: Optional dataset registry for testing (injected by test harness).
    """
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.conditions is not None:
        parser.error("--conditions is fixed to clean/miss50/noise50 for the ablation runner")

    all_factories, _ = get_methods("binary", include_ablation=True)
    factories = {name: all_factories[name] for name in _ABLATION_VARIANTS}

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
