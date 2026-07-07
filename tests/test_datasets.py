import pandas as pd

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


def test_california_offline() -> None:
    X, y = REGRESSION["california"].loader()
    assert X.shape == (20640, 8) and pd.api.types.is_float_dtype(y)


def test_loader_docstrings_have_license_note() -> None:
    for spec in list(CLASSIFICATION.values()) + list(REGRESSION.values()):
        doc = spec.loader.__doc__ or ""
        assert "License" in doc and "http" in doc, spec.name
