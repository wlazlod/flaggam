"""German Credit smoke benchmark — the acceptance gate for the core package.

25 stratified splits (not the paper's 1000), so tolerance is +/-0.02 around the
paper's reported 0.775 mean AUROC (arXiv:2605.31189, Table 3); the +/-0.010
spec tolerance applies only to full 1000-split runs. See DECISIONS entry 15.
"""

import numpy as np
import pandas as pd
import pytest
from benchmarks.runner import RunConfig, run_benchmark

N_SPLITS = 25


@pytest.mark.network
@pytest.mark.slow
def test_german_credit_smoke(tmp_path) -> None:
    out = tmp_path / "smoke.csv"
    cfg = RunConfig(
        datasets=["german_credit"],
        methods=["flaggam", "ebm", "xgboost"],
        seeds=range(N_SPLITS),
        conditions=["clean", "miss50"],
        out=out,
    )
    run_benchmark(cfg, task="binary")
    df = pd.read_csv(out)
    piv = df.pivot_table(index=["method", "seed"], columns="condition", values="value")

    flag_clean = piv.loc["flaggam", "clean"]
    ebm_clean = piv.loc["ebm", "clean"]
    assert len(flag_clean) == N_SPLITS, "every split must produce a flaggam row"

    # (1) FlagGAM mean AUROC near the paper's 0.775
    assert 0.755 <= flag_clean.mean() <= 0.795, f"flaggam mean {flag_clean.mean():.4f}"
    # (2) below EBM (paper: 0.792) — allow a half-sd margin at 25 splits
    assert flag_clean.mean() < ebm_clean.mean() + 0.5 * ebm_clean.std() / np.sqrt(N_SPLITS)
    # (3) smaller 50%-missing drop than XGBoost
    drop = (piv["clean"] - piv["miss50"]).groupby("method").mean()
    assert drop["flaggam"] < drop["xgboost"], f"drops: {drop.to_dict()}"
