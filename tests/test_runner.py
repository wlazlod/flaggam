import numpy as np
import pandas as pd
from benchmarks.runner import RunConfig, run_benchmark

from flaggam.datasets import DatasetSpec


def _toy_binary() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-(2 * (x <= -0.5) - 0.5)))).astype(int))
    return pd.DataFrame({"x": x, "c": pd.Categorical(rng.choice(["a", "b"], n))}), y


TOY = {"toy": DatasetSpec("toy", "binary", _toy_binary)}


def test_run_benchmark_writes_tidy_rows(tmp_path) -> None:
    out = tmp_path / "res.csv"
    cfg = RunConfig(datasets=["toy"], methods=["logistic", "flaggam"],
                    seeds=range(2), conditions=["clean", "miss50"], out=out)
    run_benchmark(cfg, task="binary", registry=TOY)
    df = pd.read_csv(out)
    # 2 methods x 2 seeds x 2 conditions x 1 metric
    assert len(df) == 8
    assert set(df.method) == {"logistic", "flaggam"}
    assert set(df.condition) == {"clean", "miss50"}
    assert df.value.between(0.3, 1.0).all()
    # paired: same (method, seed) has clean >= miss50 on average across the frame
    piv = df.pivot_table(index=["method", "seed"], columns="condition", values="value")
    assert (piv["clean"] - piv["miss50"]).mean() > -0.05


def test_run_benchmark_deterministic(tmp_path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    for out in (a, b):
        run_benchmark(RunConfig(["toy"], ["flaggam"], range(2), ["clean"], out),
                      task="binary", registry=TOY)
    pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))


def test_run_benchmark_no_partial_rows_on_method_failure(tmp_path, monkeypatch) -> None:
    """A method that fails partway through its condition loop must yield zero rows,
    not partial rows from the conditions it completed before failing."""
    from benchmarks.methods import get_methods as real_get_methods

    class _FlakyMethod:
        name = "flaky"
        needs_imputation = False

        def __init__(self) -> None:
            self._calls = 0

        def fit(self, X_train, y_train, seed):
            return self

        def predict_scores(self, X):
            self._calls += 1
            if self._calls > 1:
                raise RuntimeError("boom on second condition")
            return np.full(len(X), 0.5)

    def fake_get_methods(task):
        factories, skipped = real_get_methods(task)
        factories["flaky"] = _FlakyMethod
        return factories, skipped

    monkeypatch.setattr("benchmarks.runner.get_methods", fake_get_methods)

    out = tmp_path / "res.csv"
    cfg = RunConfig(datasets=["toy"], methods=["logistic", "flaky"],
                    seeds=range(1), conditions=["clean", "miss50"], out=out)
    run_benchmark(cfg, task="binary", registry=TOY)
    df = pd.read_csv(out)
    assert (df.method == "flaky").sum() == 0
    assert set(df[df.method == "logistic"].condition) == {"clean", "miss50"}


def test_render_tables_smoke(tmp_path, capsys) -> None:
    from benchmarks.render_tables import render
    out = tmp_path / "res.csv"
    run_benchmark(RunConfig(["toy"], ["flaggam"], range(2), ["clean"], out),
                  task="binary", registry=TOY)
    render(out, table=None)
    text = capsys.readouterr().out
    assert "flaggam" in text and "toy" in text


def test_cli_parses() -> None:
    from benchmarks.run_classification import build_parser
    args = build_parser().parse_args(
        ["--datasets", "german_credit", "--n-splits", "3", "--out", "/tmp/x.csv"]
    )
    assert args.datasets == ["german_credit"] and args.n_splits == 3


def test_robustness_runner_and_drop_table(tmp_path, capsys) -> None:
    import benchmarks.run_robustness as rr
    from benchmarks.render_tables import render

    out = tmp_path / "rob.csv"
    args = ["--datasets", "toy", "--methods", "flaggam", "--n-splits", "2", "--out", str(out)]
    # registry injection for the CLI path: rr.main accepts a registry kwarg for tests
    rr.main(args, registry=TOY)
    df = pd.read_csv(out)
    assert set(df.condition) == {"clean", "miss25", "miss50", "noise25", "noise50"}
    render(out, table=5)
    text = capsys.readouterr().out
    assert "miss50" in text and "drop" in text.lower()
