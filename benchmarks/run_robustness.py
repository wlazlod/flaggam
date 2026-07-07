"""CLI: Table 5 robustness benchmark (paired AUROC drops over corruption conditions)."""

import argparse
from pathlib import Path

from benchmarks.runner import RunConfig, _setup_logging, add_common_args, run_benchmark
from flaggam.datasets import CLASSIFICATION


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Table 5 robustness benchmark.")
    add_common_args(
        p, conditions_help="Fixed to all five conditions for this runner; not overridable."
    )
    p.set_defaults(out="benchmarks/results/robustness.csv")
    return p


def main(argv: list[str] | None = None, registry: dict | None = None) -> None:
    """Run robustness benchmark with paired corruption conditions.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] if None).
        registry: Optional dataset registry for testing (injected by test harness).
    """
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.conditions is not None:
        parser.error("--conditions is fixed to all five conditions for the robustness runner")
    cfg = RunConfig(
        datasets=args.datasets or list(CLASSIFICATION),
        methods=args.methods,
        seeds=range(args.seed_start, args.seed_start + args.n_splits),
        conditions=["clean", "miss25", "miss50", "noise25", "noise50"],
        out=Path(args.out),
    )
    run_benchmark(cfg, task="binary", registry=registry)


if __name__ == "__main__":
    main()
