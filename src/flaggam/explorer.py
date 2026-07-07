"""Self-contained interactive HTML rules explorer for fitted FlagGAM estimators.

This module is an ORIGINAL ADDITION (not part of Zhao & Welsch, arXiv:2605.31189):
the paper specifies no explorer/export API. `export_rules_html` renders a single
HTML document with inlined CSS and vanilla JS and NO external resources (no CDN,
fonts, or images), so the result is fully self-contained: it works offline and
can be embedded in an iframe (e.g. the docs site) with no network access.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .inspection import _ADDITIVE_HEADS

logger = logging.getLogger(__name__)


def _sig6(x: float) -> float:
    """Round to 6 decimal places (keeps the embedded JSON payload small)."""
    xf = float(x)
    return xf if not np.isfinite(xf) else round(xf, 6)


def _feature_payload(
    feat_bases: list[tuple[int, Any]], coef: np.ndarray, is_categorical: bool
) -> dict[str, Any] | None:
    rules = [
        {
            "rule": b.name,
            "kind": b.kind,
            "weight": _sig6(coef[j]),
            "support": int(b.support),
            "p_adj": _sig6(b.p_adj),
        }
        for j, b in feat_bases
    ]
    if is_categorical:
        bars = [
            {"level": str(b.level), "weight": _sig6(coef[j])}
            for j, b in feat_bases
            if b.kind == "category"
        ]
        return {"type": "categorical", "bars": bars, "rules": rules} if bars else None

    # Numeric: same math/anchors as plots.plot_shape (cutoff for threshold/hinge
    # bases, mean for trend bases); missing_indicator bases contribute 0 on the
    # (finite) grid so including them in the sum below is a no-op.
    anchors = [
        v
        for _, b in feat_bases
        for v in (getattr(b, "cutoff", None), getattr(b, "mean", None))
        if v is not None
    ]
    if not anchors:
        return None
    lo, hi = min(anchors), max(anchors)
    pad = 0.1 * (hi - lo) if hi > lo else max(abs(lo), 1.0)
    grid = np.linspace(lo - pad, hi + pad, 200)
    contribution = np.zeros_like(grid)
    for j, b in feat_bases:
        contribution += coef[j] * b.transform(grid)
    curve = {"x": [_sig6(v) for v in grid], "y": [_sig6(v) for v in contribution]}
    cutoffs = [_sig6(v) for v in sorted(set(anchors))]
    return {"type": "numeric", "curve": curve, "cutoffs": cutoffs, "rules": rules}


def _build_payload(estimator: Any, title: str) -> dict[str, Any]:
    bases = estimator.core_.bases_
    coef = np.ravel(estimator.head_.coef_)
    is_clf = hasattr(estimator, "classes_")
    intercept = _sig6(float(np.ravel(estimator.head_.intercept_)[0]))
    categorical = set(estimator.core_.categorical_features_)

    all_names = sorted({b.feature for b in bases})
    numeric_names = sorted(f for f in all_names if f not in categorical)
    categorical_names = sorted(f for f in all_names if f in categorical)

    features = []
    for name in [*numeric_names, *categorical_names]:
        feat_bases = [(j, b) for j, b in enumerate(bases) if b.feature == name]
        payload = _feature_payload(feat_bases, coef, name in categorical)
        if payload is not None:
            features.append({"name": name, **payload})

    return {
        "title": title,
        "task": "binary" if is_clf else "regression",
        "intercept": intercept,
        "n_rules": len(bases),
        "features": features,
    }


def export_rules_html(
    estimator: Any, path: "str | Path | None" = None, title: str = "FlagGAM rules explorer"
) -> str:
    """Render a self-contained interactive HTML explorer of the fitted rules.

    Requires an estimator fitted with `representation='full'` and the additive
    head (binary classification or regression only; the compact head's
    coefficients are per-class scores, not per-rule weights). Returns the
    complete HTML document as a string; if `path` is given, also writes it
    (UTF-8) and still returns the string.
    """
    check_is_fitted(estimator, "core_")
    if getattr(estimator, "representation", "full") == "compact":
        raise ValueError(
            "export_rules_html requires representation='full'; under 'compact' the head "
            "coefficients are per-class scores, not per-rule weights"
        )
    if hasattr(estimator, "classes_") and len(estimator.classes_) > 2:
        raise ValueError("export_rules_html supports binary classification and regression only")
    if not isinstance(estimator.head_, _ADDITIVE_HEADS):
        raise ValueError("export_rules_html requires the additive head")

    payload = _build_payload(estimator, title)
    data = json.dumps(payload).replace("</", "<\\/")
    html = _TEMPLATE.replace("{{TITLE}}", title).replace("{{DATA}}", data)

    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  margin:0;padding:1.5rem;background:#f7f7f9;color:#1a1a1a}
header{margin-bottom:1rem}
h1{font-size:1.3rem;margin:0 0 .4rem}
#meta{color:#555;font-size:.9rem}
#meta span{margin-right:1.2rem}
select{font-size:1rem;padding:.3rem .5rem;margin-bottom:1rem;max-width:100%}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:.5rem;margin-bottom:1rem}
svg{width:100%;height:auto;display:block}
.axis-label{font-size:12px;fill:#333}
.tick-label{font-size:11px;fill:#666}
.zero-line{stroke:#999;stroke-width:1}
.cutoff-line{stroke:#999;stroke-width:1;stroke-dasharray:4 3}
.curve{fill:none;stroke:#2166ac;stroke-width:2}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;
  border-radius:6px;overflow:hidden;font-size:.9rem}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid #eee}
th{background:#fafafa;font-weight:600}
.pos{color:#2166ac}
.neg{color:#b2182b}
</style>
</head>
<body>
<header>
<h1>{{TITLE}}</h1>
<div id="meta"></div>
</header>
<select id="feature"></select>
<div class="panel"><svg id="panel" viewBox="0 0 700 360"></svg></div>
<table id="rules">
<thead><tr><th>Rule</th><th>Weight</th><th>Support</th><th>adj. p-value</th></tr></thead>
<tbody></tbody>
</table>
<script type="application/json" id="flaggam-data">{{DATA}}</script>
<script>
const DATA = JSON.parse(document.getElementById("flaggam-data").textContent);
const NS = document.getElementById("panel").namespaceURI;
const yLabel = DATA.task === "binary" ? "contribution (log-odds)" : "contribution";

function el(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function fmtWeight(w) { return w.toFixed(3); }
function fmtP(p) { return p < 0.001 ? p.toExponential(2) : p.toFixed(4); }

function renderMeta() {
  document.getElementById("meta").innerHTML =
    `<span>task: ${DATA.task}</span><span>intercept: ${DATA.intercept.toFixed(3)}</span>` +
    `<span>rules: ${DATA.n_rules}</span>`;
}

function renderTable(rules) {
  const body = document.querySelector("#rules tbody");
  body.innerHTML = "";
  for (const r of rules) {
    const tr = document.createElement("tr");
    const cls = r.weight >= 0 ? "pos" : "neg";
    tr.innerHTML = `<td>${r.rule}</td><td class="${cls}">${fmtWeight(r.weight)}</td>` +
      `<td>${r.support}</td><td>${fmtP(r.p_adj)}</td>`;
    body.appendChild(tr);
  }
}

function renderNumeric(svg, feat) {
  const W = 700, H = 360, mL = 60, mR = 20, mT = 20, mB = 50;
  const xs = feat.curve.x, ys = feat.curve.y;
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(0, ...ys), yMax = Math.max(0, ...ys);
  const sx = x => mL + (x - xMin) / (xMax - xMin || 1) * (W - mL - mR);
  const sy = y => H - mB - (y - yMin) / (yMax - yMin || 1) * (H - mT - mB);

  svg.appendChild(el("line", {class: "zero-line", x1: mL, x2: W - mR, y1: sy(0), y2: sy(0)}));
  for (const c of feat.cutoffs) {
    svg.appendChild(el("line", {
      class: "cutoff-line", x1: sx(c), x2: sx(c), y1: mT, y2: H - mB,
    }));
  }
  let d = `M${sx(xs[0])} ${sy(ys[0])}`;
  for (let i = 1; i < xs.length; i++) {
    d += ` L${sx(xs[i])} ${sy(ys[i - 1])} L${sx(xs[i])} ${sy(ys[i])}`;
  }
  svg.appendChild(el("path", {class: "curve", d}));

  svg.appendChild(el("text", {class: "axis-label", x: W / 2, y: H - 10, "text-anchor": "middle"}))
    .textContent = feat.name;
  const yLbl = el("text", {
    class: "axis-label", x: 14, y: H / 2, "text-anchor": "middle",
    transform: `rotate(-90 14 ${H / 2})`,
  });
  yLbl.textContent = yLabel;
  svg.appendChild(yLbl);
  svg.appendChild(el("text", {class: "tick-label", x: mL, y: H - mB + 16})).textContent =
    xMin.toFixed(2);
  svg.appendChild(el("text", {
    class: "tick-label", x: W - mR, y: H - mB + 16, "text-anchor": "end",
  })).textContent = xMax.toFixed(2);
  svg.appendChild(el("text", {class: "tick-label", x: mL - 6, y: sy(yMax), "text-anchor": "end"}))
    .textContent = yMax.toFixed(2);
  svg.appendChild(el("text", {class: "tick-label", x: mL - 6, y: sy(yMin), "text-anchor": "end"}))
    .textContent = yMin.toFixed(2);
}

function renderCategorical(svg, feat) {
  const W = 700, H = 360, mL = 60, mR = 20, mT = 20, mB = 60;
  const bars = feat.bars;
  const wMax = Math.max(...bars.map(b => Math.abs(b.weight)), 1e-9);
  const yMin = Math.min(0, ...bars.map(b => b.weight));
  const yMax = Math.max(0, ...bars.map(b => b.weight));
  const sy = y => H - mB - (y - yMin) / (yMax - yMin || 1) * (H - mT - mB);
  const zeroY = sy(0);
  const slot = (W - mL - mR) / bars.length;
  svg.appendChild(el("line", {class: "zero-line", x1: mL, x2: W - mR, y1: zeroY, y2: zeroY}));
  bars.forEach((b, i) => {
    const x = mL + i * slot + slot * 0.15;
    const barW = slot * 0.7;
    const y = sy(Math.max(0, b.weight));
    const barH = Math.abs(sy(b.weight) - zeroY);
    svg.appendChild(el("rect", {
      x, y, width: barW, height: Math.max(barH, 0),
      fill: b.weight >= 0 ? "#2166ac" : "#b2182b",
    }));
    svg.appendChild(el("text", {
      class: "tick-label", x: x + barW / 2, y: H - mB + 16, "text-anchor": "middle",
    })).textContent = b.level;
  });
  svg.appendChild(el("text", {class: "axis-label", x: W / 2, y: H - 10, "text-anchor": "middle"}))
    .textContent = feat.name;
  const yLbl = el("text", {
    class: "axis-label", x: 14, y: H / 2, "text-anchor": "middle",
    transform: `rotate(-90 14 ${H / 2})`,
  });
  yLbl.textContent = yLabel;
  svg.appendChild(yLbl);
}

function render(name) {
  const feat = DATA.features.find(f => f.name === name);
  const svg = document.getElementById("panel");
  svg.innerHTML = "";
  if (feat.type === "numeric") renderNumeric(svg, feat); else renderCategorical(svg, feat);
  renderTable(feat.rules);
}

function init() {
  renderMeta();
  const select = document.getElementById("feature");
  for (const f of DATA.features) {
    const opt = document.createElement("option");
    opt.value = f.name;
    opt.textContent = f.name;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => render(select.value));
  if (DATA.features.length > 0) render(DATA.features[0].name);
}

init();
</script>
</body>
</html>
"""
