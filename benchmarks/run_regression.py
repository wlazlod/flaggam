"""CLI: Table 4 regression benchmark (RMSE + R2, clean condition by default)."""

import argparse
from pathlib import Path

from benchmarks.runner import RunConfig, _setup_logging, add_common_args, run_benchmark
from flaggam.datasets import REGRESSION


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Table 4 regression benchmark.")
    add_common_args(p)
    p.set_defaults(out="benchmarks/results/regression.csv")
    return p


def main(argv: list[str] | None = None, registry: dict | None = None) -> None:
    _setup_logging()
    args = build_parser().parse_args(argv)
    cfg = RunConfig(
        datasets=args.datasets or list(REGRESSION),
        methods=args.methods,
        seeds=range(args.seed_start, args.seed_start + args.n_splits),
        conditions=args.conditions or ["clean"],
        out=Path(args.out),
    )
    run_benchmark(cfg, task="regression", registry=registry)


if __name__ == "__main__":
    main()
