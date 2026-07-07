"""Render a tidy benchmark results CSV into comparison tables.

Prints (and writes `<csv>.md`) a per-dataset mean+/-sd pivot per method on the
clean condition. With `--table {3,4}`, also joins `paper_targets` by
(method, dataset[, metric]) and flags `|delta| > 0.010`. With `--table 5`,
computes per-(dataset, method, seed) paired AUROC drops (clean vs each
corrupted condition) first, macro-averages them, then joins `paper_targets.TABLE5`.
A method/condition/metric combination with fewer than 95% of its group's max
row count gets a footnote (a hint that some fits failed and were skipped).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from benchmarks.paper_targets import CITATION, TABLE3, TABLE4, TABLE5

logger = logging.getLogger(__name__)

_TOLERANCE = 0.010


def _footnotes(df: pd.DataFrame) -> list[str]:
    counts = df.groupby(["dataset", "condition", "metric", "method"]).size().rename("n")
    counts = counts.reset_index()
    notes = []
    for (ds, cond, metric), group in counts.groupby(["dataset", "condition", "metric"]):
        expected = group["n"].max()
        for _, r in group.iterrows():
            if r["n"] < 0.95 * expected:
                notes.append(
                    f"* {r['method']} on {ds}/{cond}/{metric}: {r['n']}/{expected} rows "
                    f"({r['n'] / expected:.0%}) — some splits likely failed"
                )
    return notes


def _paired_drops(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method, seed) AUROC drop = clean − corrupted, long form."""
    auroc = df[df.metric == "auroc"]
    clean = auroc[auroc.condition == "clean"].set_index(["dataset", "method", "seed"])["value"]
    rows = []
    for cond in sorted(c for c in auroc.condition.unique() if c != "clean"):
        sub = auroc[auroc.condition == cond]
        for _, r in sub.iterrows():
            key = (r["dataset"], r["method"], r["seed"])
            if key in clean.index:
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "method": r["method"],
                        "condition": cond,
                        "drop": clean.loc[key] - r["value"],
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "method", "condition", "drop"])


def _delta_lines(ours: dict[tuple, float], target: dict[tuple, float]) -> list[str]:
    lines = [f"-- delta vs paper targets ({CITATION}) --"]
    for key, ours_value in ours.items():
        if key not in target:
            continue
        delta = ours_value - target[key]
        flag = " [FLAG]" if abs(delta) > _TOLERANCE else ""
        label = " / ".join(str(k) for k in key)
        paper_value = target[key]
        lines.append(
            f"  {label}: ours={ours_value:.3f} paper={paper_value:.3f} delta={delta:+.3f}{flag}"
        )
    return lines


def render(csv_path: Path, table: int | None) -> None:
    df = pd.read_csv(csv_path)
    lines: list[str] = []

    if table == 5:
        drops = _paired_drops(df)
        # Two-stage macro-average: mean per (dataset, method, condition) first, then
        # mean across datasets, so datasets with unequal row counts (e.g. from a
        # partial method failure) don't get row-count-weighted in the average.
        per_dataset = drops.groupby(["dataset", "method", "condition"])["drop"].mean()
        macro = per_dataset.groupby(["method", "condition"]).mean()
        pivot = macro.unstack()
        lines.append("Table 5: mean paired AUROC drop (macro-averaged over datasets)")
        lines.append(pivot.to_string())
        lines.extend(_delta_lines(macro.to_dict(), TABLE5))
    else:
        for metric in sorted(df.metric.unique()):
            sub = df[(df.condition == "clean") & (df.metric == metric)]
            if sub.empty:
                continue
            mean = sub.groupby(["dataset", "method"])["value"].mean()
            sd = sub.groupby(["dataset", "method"])["value"].std()
            lines.append(f"--- metric={metric} (clean) ---")
            for ds in sorted(sub.dataset.unique()):
                parts = [
                    f"{meth}={mean[(ds, meth)]:.3f}+/-{sd[(ds, meth)]:.3f}"
                    for meth in sorted(sub[sub.dataset == ds].method.unique())
                ]
                lines.append(f"{ds}: " + ", ".join(parts))
            if table == 3:
                ours: dict[tuple, float] = {(meth, ds): v for (ds, meth), v in mean.items()}
                lines.extend(_delta_lines(ours, TABLE3))
            elif table == 4:
                ours = {(meth, ds, metric): v for (ds, meth), v in mean.items()}
                lines.extend(_delta_lines(ours, TABLE4))

    notes = _footnotes(df)
    if notes:
        lines.append("Footnotes:")
        lines.extend(notes)

    text = "\n".join(lines)
    print(text)
    Path(csv_path).with_suffix(".md").write_text(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a tidy benchmark results CSV into tables.")
    p.add_argument("csv", type=str, help="Path to a tidy results CSV.")
    p.add_argument("--table", type=int, choices=[3, 4, 5], default=None, help="Join paper targets.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    render(Path(args.csv), args.table)


if __name__ == "__main__":
    main()
