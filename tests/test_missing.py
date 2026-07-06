import numpy as np
import pandas as pd

from flaggam.core import FlagCoreModule


def test_indicator_mode_flags_informative_missingness() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 0.2).astype(int)
    miss = rng.uniform(size=n) < np.where(y == 1, 0.6, 0.05)  # MNAR: missing when y=1
    x = np.where(miss, np.nan, x)
    X = pd.DataFrame({"x": x})
    meta_no = FlagCoreModule(task="binary").fit(X, y).metadata()
    assert "missing_indicator" not in set(meta_no.kind)
    meta_ind = FlagCoreModule(task="binary", missing="indicator").fit(X, y).metadata()
    assert "missing_indicator" in set(meta_ind.kind)
