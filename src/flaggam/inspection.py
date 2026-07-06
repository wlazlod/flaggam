"""Rule export and per-row reason codes for fitted FlagGAM estimators."""

import logging

import numpy as np
import pandas as pd

from .heads import AdditiveHead

logger = logging.getLogger(__name__)


def export_rules(estimator) -> pd.DataFrame:
    meta = estimator.core_.metadata()
    head = estimator.head_
    if not isinstance(head, AdditiveHead) or head.coef_.size == 0:
        if not isinstance(head, AdditiveHead):
            logger.warning("flexible head: additive interpretability is not preserved; weight=NaN")
        meta["weight"] = np.nan
        meta["additive_interpretable"] = isinstance(head, AdditiveHead)
        return meta
    coef = np.atleast_2d(head.coef_)
    if coef.shape[0] == 1:
        meta["weight"] = coef[0]
    else:
        meta["weight"] = np.abs(coef).max(axis=0)
        for k, cls in enumerate(estimator.classes_):
            meta[f"weight_{cls}"] = coef[k]
    meta["additive_interpretable"] = True
    return meta


def explain(estimator, X) -> pd.DataFrame:
    head = estimator.head_
    if not isinstance(head, AdditiveHead):
        raise ValueError("reason codes require the additive head")
    Z_sparse = estimator.transform(X)
    Z = Z_sparse.toarray()
    coef = np.atleast_2d(head.coef_)
    intercept = np.atleast_1d(head.intercept_)
    is_clf = hasattr(estimator, "classes_")
    if is_clf and coef.shape[0] > 1:
        target_row = np.argmax(estimator.head_.predict_proba(Z_sparse), axis=1)
    else:
        target_row = np.zeros(len(Z), dtype=int)
    records = []
    bases = estimator.core_.bases_
    # TODO: vectorize if explain() becomes hot on large n
    for i in range(Z.shape[0]):
        k = target_row[i]
        # k is always in bounds; min() kept as a cheap invariant guard
        records.append(
            dict(
                row=i,
                feature="<intercept>",
                rule="<intercept>",
                value=1.0,
                contribution=float(intercept[min(k, len(intercept) - 1)]),
            )
        )
        for j, b in enumerate(bases):
            if Z[i, j] != 0.0:
                records.append(
                    dict(
                        row=i,
                        feature=b.feature,
                        rule=b.name,
                        value=float(Z[i, j]),
                        contribution=float(coef[k, j] * Z[i, j]),
                    )
                )
    out = pd.DataFrame.from_records(records)
    return out.sort_values(
        ["row", "contribution"],
        key=lambda s: s.abs() if s.name == "contribution" else s,
        ascending=[True, False],
    ).reset_index(drop=True)
