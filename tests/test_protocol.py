from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.protocol import (
    apply_impute,
    corrupt_missing,
    corrupt_noise,
    impute_stats,
    make_split,
    result_row,
    score_binary,
    score_regression,
    train_val_split,
    write_rows,
)


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=400),
        "b": rng.normal(10, 2, size=400),
        "c": pd.Categorical(rng.choice(["u", "v"], 400)),
    })
    y = pd.Series((rng.uniform(size=400) < 0.3).astype(int))
    return X, y


def test_make_split_stratified(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    tr, te = make_split(y, seed=0, task="binary")
    assert len(tr) == 320 and len(te) == 80
    assert set(tr) & set(te) == set()
    assert abs(y.iloc[tr].mean() - y.iloc[te].mean()) < 0.08   # stratification
    tr2, _ = make_split(y, seed=0, task="binary")
    np.testing.assert_array_equal(tr, tr2)                     # deterministic
    tr3, _ = make_split(y, seed=1, task="binary")
    assert not np.array_equal(tr, tr3)


def test_make_split_regression(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    _, y = xy
    y_float = y.astype(float)
    tr, te = make_split(y_float, seed=0, task="regression")
    assert set(tr) & set(te) == set()  # disjoint indices


def test_train_val_split_shapes(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    X_tr, X_val, y_tr, y_val = train_val_split(X, y, seed=0, task="binary")
    assert len(X_tr) == 320 and len(X_val) == 80
    assert list(X_tr.columns) == list(X.columns)


def test_imputation(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    Xm = X.copy()
    Xm.loc[Xm.index[:50], "a"] = np.nan
    Xm.loc[Xm.index[:50], "c"] = np.nan
    stats = impute_stats(Xm)
    out = apply_impute(Xm, stats)
    assert out.isna().sum().sum() == 0
    assert out.loc[out.index[0], "a"] == pytest.approx(Xm["a"].median())
    # apply to unseen frame with same columns
    assert apply_impute(X, stats).isna().sum().sum() == 0


def test_corrupt_missing_fraction(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = xy
    rng = np.random.default_rng(0)
    Xc = corrupt_missing(X, rho=0.25, rng=rng)
    frac = Xc.isna().sum().sum() / (X.shape[0] * X.shape[1])
    assert frac == pytest.approx(0.25, abs=0.03)
    assert X.isna().sum().sum() == 0                       # original untouched


def test_corrupt_noise_numeric_only(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = xy
    X_orig = X.copy()
    rng = np.random.default_rng(0)
    sd = {"a": float(X["a"].std()), "b": float(X["b"].std())}
    Xc = corrupt_noise(X, rho=0.5, train_sd=sd, rng=rng)
    assert (Xc["c"] == X["c"]).all()                       # categorical untouched
    changed = (Xc["a"] != X["a"]).mean()
    assert changed == pytest.approx(0.5, abs=0.08)
    resid = (Xc["a"] - X["a"])[Xc["a"] != X["a"]]
    assert resid.std() == pytest.approx(0.5 * sd["a"], rel=0.25)
    # original frame untouched (corrupt_noise must copy)
    pd.testing.assert_frame_equal(X, X_orig)


def test_scores_and_rows(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    s = score_binary(y, np.where(y == 1, 0.9, 0.1))
    assert s["auroc"] == pytest.approx(1.0)
    r = score_regression(pd.Series([1.0, 2.0]), np.array([1.0, 2.0]))
    assert r["rmse"] == pytest.approx(0.0) and r["r2"] == pytest.approx(1.0)
    out = tmp_path / "res.csv"
    write_rows([result_row("d", "m", 0, "clean", "auroc", 0.8)], out)
    write_rows([result_row("d", "m", 1, "clean", "auroc", 0.9)], out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["dataset", "method", "seed", "condition", "metric", "value"]
    assert len(df) == 2                                     # append, header once
