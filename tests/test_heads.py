import numpy as np
import pytest
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from flaggam.heads import AdditiveHead, FlexibleHead


@pytest.fixture()
def zy() -> tuple[sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(0)
    Z = sparse.csr_matrix((rng.uniform(size=(500, 3)) < 0.3).astype(float))
    logit = -1.0 + 2.5 * Z.toarray()[:, 0] - 1.5 * Z.toarray()[:, 1]
    y = (rng.uniform(size=500) < 1 / (1 + np.exp(-logit))).astype(int)
    return Z, y


def test_additive_classification(zy) -> None:
    Z, y = zy
    head = AdditiveHead(task="binary", C=1.0).fit(Z, y)
    proba = head.predict_proba(Z)
    assert proba.shape == (500, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert head.coef_.shape == (1, 3)
    assert head.coef_[0, 0] > 0 > head.coef_[0, 1]  # signs recovered


def test_additive_internal_cv(zy) -> None:
    Z, y = zy
    head = AdditiveHead(task="binary", C=[0.01, 0.1, 1.0, 10.0], random_state=0).fit(Z, y)
    assert head.predict(Z).shape == (500,)


def test_additive_regression() -> None:
    rng = np.random.default_rng(0)
    Z = sparse.csr_matrix(rng.normal(size=(300, 2)))
    y = 3.0 * Z.toarray()[:, 0] + rng.normal(0, 0.1, 300)
    head = AdditiveHead(task="regression", alpha=1.0).fit(Z, y)
    assert head.coef_[0] == pytest.approx(3.0, abs=0.2)


def test_empty_z_fallback(zy) -> None:
    _, y = zy
    Z0 = sparse.csr_matrix((500, 0))
    head = AdditiveHead(task="binary").fit(Z0, y)
    proba = head.predict_proba(Z0)
    assert proba.shape == (500, 2)
    np.testing.assert_allclose(proba[:, 1], y.mean(), atol=1e-9)
    reg = AdditiveHead(task="regression").fit(Z0, y.astype(float))
    np.testing.assert_allclose(reg.predict(Z0), y.mean(), atol=1e-9)


def test_flexible_head(zy) -> None:
    Z, y = zy
    est = RandomForestClassifier(n_estimators=20)
    head = FlexibleHead(est, task="binary", random_state=0).fit(Z, y)
    assert head.predict_proba(Z).shape == (500, 2)
    assert not hasattr(est, "classes_")  # original estimator untouched (was cloned)
