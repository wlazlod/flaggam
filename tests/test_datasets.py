import pandas as pd
import pytest

from flaggam.datasets import CLASSIFICATION, REGRESSION, DatasetSpec, data_dir


def test_registries_complete() -> None:
    assert set(CLASSIFICATION) == {
        "pima", "breast_cancer", "heart", "german_credit", "adult", "bank_marketing"
    }
    assert set(REGRESSION) == {"ames", "california", "wine_white"}
    for spec in list(CLASSIFICATION.values()) + list(REGRESSION.values()):
        assert isinstance(spec, DatasetSpec)
        assert spec.task in {"binary", "regression"}


def test_data_dir_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLAGGAM_DATA_DIR", str(tmp_path / "cache"))
    d = data_dir()
    assert d == tmp_path / "cache" and d.is_dir()


def test_breast_cancer_offline() -> None:
    X, y = CLASSIFICATION["breast_cancer"].loader()
    assert X.shape == (569, 30) and set(y.unique()) == {0, 1}
    assert all(pd.api.types.is_float_dtype(X[c]) for c in X.columns)
    assert y.sum() == 212  # malignant count — pins positive=malignant polarity


@pytest.mark.network
def test_california_loader() -> None:
    X, y = REGRESSION["california"].loader()
    assert X.shape == (20640, 8) and pd.api.types.is_float_dtype(y)


def test_loader_docstrings_have_license_note() -> None:
    for spec in list(CLASSIFICATION.values()) + list(REGRESSION.values()):
        doc = spec.loader.__doc__ or ""
        assert "License" in doc and "http" in doc, spec.name


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

NETWORK_CASES = [
    ("pima", (768, 8), "binary"),
    ("heart", (303, 13), "binary"),
    ("german_credit", (1000, 20), "binary"),
    ("adult", None, "binary"),           # ~48842 rows; assert 45000 < n < 50000, 14 cols
    ("bank_marketing", (41188, 19), "binary"),
    ("ames", None, "regression"),        # log target: 10 < y.mean() < 13
    ("wine_white", (4898, 11), "regression"),
]


@pytest.mark.network
@pytest.mark.parametrize("name,shape,task", NETWORK_CASES)
def test_network_loader(name, shape, task) -> None:
    spec = (CLASSIFICATION | REGRESSION)[name]
    X, y = spec.loader()
    if shape is not None:
        assert X.shape == shape
    assert len(X) == len(y)
    if task == "binary":
        assert set(y.unique()) <= {0, 1} and 0 < y.mean() < 1
    if name == "adult":
        assert 45000 < len(X) < 50000 and X.shape[1] == 14
    if name == "bank_marketing":
        assert "duration" not in X.columns
        assert "euribor3m" in X.columns
    if name == "ames":
        assert 10 < float(y.mean()) < 13  # log-scale sanity
    if name == "pima":
        assert X["insulin"].isna().sum() > 0  # zeros converted to NaN


@pytest.mark.network
def test_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLAGGAM_DATA_DIR", str(tmp_path))
    X1, _ = CLASSIFICATION["german_credit"].loader()
    assert any(tmp_path.iterdir())          # cache file written
    X2, _ = CLASSIFICATION["german_credit"].loader()
    pd.testing.assert_frame_equal(X1, X2)   # second call served from cache
