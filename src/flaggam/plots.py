"""Matplotlib visualization helpers for fitted FlagGAM estimators and diagnostics.

This module is an ORIGINAL ADDITION and is not part of Zhao & Welsch
(arXiv:2605.31189): the paper specifies no plotting API. matplotlib is an
OPTIONAL dependency (`pip install flaggam[viz]`); importing this module never
imports matplotlib eagerly. Every function calls `_plt()` first, which raises
a helpful `ImportError` if matplotlib is not installed, so `import flaggam`
and `import flaggam.plots` both work without matplotlib present.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .calibration import reliability_curve
from .inspection import _ADDITIVE_HEADS

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib — install with: pip install flaggam[viz]"
        ) from exc
    return plt


def plot_shape(
    estimator: Any, feature: str, ax: "Axes | None" = None, grid_points: int = 200
) -> "Axes":
    """Plot the fitted additive contribution for one feature.

    Numeric features get a step curve of contribution vs. value over the
    range spanned by that feature's discovered cutoffs (padded 10%), plus a
    rug of the cutoffs themselves. Categorical features get one bar per
    discovered level, height equal to that level's coefficient. Requires
    `representation='full'` (the compact head's coefficients are per-class
    scores, not per-basis weights).
    """
    plt = _plt()
    if getattr(estimator, "representation", "full") == "compact":
        raise ValueError(
            "plot_shape requires representation='full'; under 'compact' the head "
            "coefficients are per-class scores, not per-rule weights"
        )
    if hasattr(estimator, "classes_") and len(estimator.classes_) > 2:
        raise ValueError("plot_shape supports binary classification and regression only")
    if not isinstance(estimator.head_, _ADDITIVE_HEADS):
        raise ValueError("plot_shape requires the additive head")
    bases = estimator.core_.bases_
    coef = np.ravel(estimator.head_.coef_)
    feat_bases = [(j, b) for j, b in enumerate(bases) if b.feature == feature]
    if not feat_bases:
        available = sorted({b.feature for b in bases})
        raise ValueError(
            f"no bases discovered for feature {feature!r}; features with bases: {available}"
        )
    y_label = "contribution (log-odds)" if hasattr(estimator, "classes_") else "contribution"

    if ax is None:
        _, ax = plt.subplots()

    if feature in estimator.core_.categorical_features_:
        labels = [str(getattr(b, "level", b.name)) for _, b in feat_bases]
        heights = [coef[j] for j, _ in feat_bases]
        ax.bar(labels, heights)
    else:
        anchors = [
            v
            for _, b in feat_bases
            for v in (getattr(b, "cutoff", None), getattr(b, "mean", None))
            if v is not None
        ]
        if not anchors:
            raise ValueError(
                f"feature {feature!r} has no threshold/hinge/trend bases to plot a shape for"
            )
        lo, hi = min(anchors), max(anchors)
        pad = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1.0)
        grid = np.linspace(lo - pad, hi + pad, grid_points)
        contribution = np.zeros_like(grid)
        for j, b in feat_bases:
            contribution += coef[j] * b.transform(grid)
        ax.step(grid, contribution, where="post")
        rug_y = ax.get_ylim()[0]
        ax.plot(anchors, [rug_y] * len(anchors), "|", color="black", markersize=12)

    ax.set_xlabel(feature)
    ax.set_ylabel(y_label)
    ax.set_title(f"{feature} shape function")
    return ax


def plot_rule_importance(estimator: Any, top_n: int = 20, ax: "Axes | None" = None) -> "Axes":
    """Horizontal bar chart of the top `top_n` rules by |weight| from `export_rules()`."""
    plt = _plt()
    if not isinstance(estimator.head_, _ADDITIVE_HEADS):
        raise ValueError("plot_rule_importance requires the additive head")
    rules = estimator.export_rules()
    top = rules.reindex(rules["weight"].abs().sort_values(ascending=False).index).head(top_n)
    if ax is None:
        _, ax = plt.subplots()
    ax.barh(top["rule"].iloc[::-1], top["weight"].iloc[::-1])
    ax.set_xlabel("weight")
    ax.set_title("Rule importance")
    return ax


def plot_waterfall(
    estimator: Any, x_row: Any, ax: "Axes | None" = None, max_rules: int = 15
) -> "Axes":
    """Cumulative horizontal bars from intercept to total score for one row.

    Rules are sorted by |contribution| and collapsed beyond `max_rules` into
    a single "(other rules)" bucket; the final bar marks the total score.
    """
    plt = _plt()
    exp = estimator.explain(x_row)
    row0 = exp[exp["row"] == 0]
    intercept = float(row0.loc[row0["feature"] == "<intercept>", "contribution"].sum())
    rules = row0[row0["feature"] != "<intercept>"]
    rules = rules.reindex(rules["contribution"].abs().sort_values(ascending=False).index)
    if len(rules) > max_rules:
        head, tail = rules.iloc[:max_rules], rules.iloc[max_rules:]
        names = [*head["rule"], "(other rules)"]
        contributions = [*head["contribution"].astype(float), float(tail["contribution"].sum())]
    else:
        names = list(rules["rule"])
        contributions = list(rules["contribution"].astype(float))

    labels = ["<intercept>", *names, "total"]
    values = [intercept, *contributions]
    running = 0.0
    lefts: list[float] = []
    widths: list[float] = []
    for v in values:
        lefts.append(min(running, running + v))
        widths.append(abs(v))
        running += v
    total = running
    lefts.append(min(0.0, total))
    widths.append(abs(total))

    if ax is None:
        _, ax = plt.subplots()
    y_pos = np.arange(len(labels))
    colors = (
        ["tab:blue"] + ["tab:green" if v >= 0 else "tab:red" for v in values[1:]] + ["black"]
    )
    ax.barh(y_pos, widths, left=lefts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("contribution")
    ax.set_title(f"Waterfall (total = {total:.3f})")
    return ax


def plot_reliability(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: "Axes | None" = None,
) -> "Axes":
    """Reliability diagram: mean predicted vs. observed rate, with per-bin counts."""
    plt = _plt()
    table = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(table["mean_predicted"], table["fraction_positive"], marker="o", label="model")
    count_ax = ax.twinx()
    count_ax.bar(
        table["mean_predicted"], table["count"], width=1.0 / n_bins, alpha=0.2, color="gray"
    )
    count_ax.set_ylabel("count")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    return ax


def plot_proxy_association(
    report: pd.DataFrame, top_n: int = 20, ax: "Axes | None" = None
) -> "Axes":
    """Horizontal bars of rule-level proxy association, flagged rules highlighted."""
    plt = _plt()
    top = report.reindex(report["association"].sort_values(ascending=False).index).head(top_n)
    if ax is None:
        _, ax = plt.subplots()
    colors = ["tab:red" if f else "tab:blue" for f in top["flagged"]]
    ax.barh(top["rule"].iloc[::-1], top["association"].iloc[::-1], color=list(reversed(colors)))
    if top["flagged"].any():
        # Exact audit threshold isn't carried in `report`; approximate the
        # boundary with the smallest association among the flagged rows.
        line_x = float(top.loc[top["flagged"], "association"].min())
        ax.axvline(line_x, linestyle="--", color="black")
    ax.set_xlabel("association")
    ax.set_title("Proxy association by rule")
    return ax


def plot_group_metrics(metrics: dict[str, Any], ax: "Axes | None" = None) -> "Axes":
    """Grouped bar chart of selection_rate/tpr/auroc per protected-attribute group."""
    plt = _plt()
    by = metrics["by_group"][["selection_rate", "tpr", "auroc"]]
    if ax is None:
        _, ax = plt.subplots()
    groups = list(by.index)
    cols = list(by.columns)
    n_metrics = len(cols)
    x = np.arange(len(groups))
    width = 0.8 / n_metrics
    for j, col in enumerate(cols):
        ax.bar(x + j * width, by[col].to_numpy(dtype=float), width, label=col)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels([str(g) for g in groups])
    gap_str = ", ".join(
        f"{k}={v:.3f}" if not np.isnan(v) else f"{k}=nan" for k, v in metrics["gaps"].items()
    )
    ax.set_title(f"Group metrics (gaps: {gap_str})")
    ax.legend()
    return ax
