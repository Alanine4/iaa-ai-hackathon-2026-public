import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import oof_predictions  # noqa: E402


@pytest.mark.parametrize("name, kwarg", [("lgbm", "random_state"), ("xgb", "random_state"), ("cat", "random_seed")])
def test_seed_param_uses_each_trees_own_keyword(name, kwarg):
    assert oof_predictions.seed_param(name, 44) == {kwarg: 44}


@pytest.mark.parametrize("name", ["glm", "gam"])
def test_seed_param_is_empty_for_deterministic_models(name):
    assert oof_predictions.seed_param(name, 44) == {}


def test_with_target_encoding_adds_one_column_to_fit_and_apply_rows():
    import numpy as np
    import pandas as pd
    x_fit = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=[10, 11, 12, 13, 14])
    x_app = pd.DataFrame({"a": [6.0, 7.0]}, index=[20, 21])
    key_fit = pd.Series(["A", "A", "B", "B", "B"])
    xf, xa, spread = oof_predictions.with_target_encoding(
        x_fit, x_app, key_fit, pd.Series(["B", "Z"]), np.arange(1.0, 6.0), np.ones(5), "make_model")
    assert list(xf.columns) == ["a", "make_model_te"] and list(xa.columns) == ["a", "make_model_te"]
    assert list(xf.index) == list(x_fit.index) and list(xa.index) == list(x_app.index)
    assert xa["make_model_te"].iloc[1] == 0.0 and xa["make_model_te"].iloc[0] > 0.0  # Z unseen; B above the mean
    assert set(spread) == {"n_codes", "min", "max", "std"} and spread["n_codes"] == 2


def test_emerging_run_dirs_keep_variants_apart():
    # C036/C038: emerging-only results in output/emerging/, pooled variants in their own sub-folder
    base = os.path.join("output", "emerging")
    assert os.path.normpath(oof_predictions.run_dir("emerging", 0.3)).endswith(base)
    assert os.path.normpath(oof_predictions.run_dir("pooled", 0.3)).endswith(os.path.join(base, "pooled_w0.3"))


def test_emerging_model_lowers_one_hot_threshold_only_for_emerging_only_linear_models():
    # C036
    assert oof_predictions.emerging_model("contents", "glm", "emerging").min_frequency == 30
    assert oof_predictions.emerging_model("contents", "gam", "emerging").min_frequency == 30
    assert oof_predictions.emerging_model("contents", "glm", "pooled").min_frequency == 100
    assert oof_predictions.emerging_model("motor", "lgbm", "emerging", {"n_estimators": 7}).model.n_estimators == 7


def test_pooled_fit_is_rebalanced_on_the_emerging_fit_rows():
    # C038: developed rows shape the fit, the level is set on the emerging rows of the fit set
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 4000
    x = pd.DataFrame({"market": pd.Categorical(np.where(np.arange(n) < 3000, "developed", "emerging")),
                      "v": rng.uniform(0, 1, n)})
    y = rng.gamma(0.3, 300.0, n) * np.where(x["market"] == "emerging", 0.6, 1.0) * (1 + x["v"].to_numpy())
    w = np.ones(n)
    em = (x["market"] == "emerging").to_numpy()
    m = oof_predictions.emerging_model("contents", "glm", "pooled")
    factor = oof_predictions.fit_emerging(m, x, y, w, np.ones(n, dtype=bool), em, "pooled", valid=None)
    assert factor > 0
    assert np.sum(m.predict(x[em]) * w[em]) == pytest.approx(np.sum(y[em] * w[em]), rel=1e-9)
    m2 = oof_predictions.emerging_model("contents", "glm", "emerging")
    assert oof_predictions.fit_emerging(m2, x, y, w, np.ones(n, dtype=bool), em, "emerging", valid=None) == 1.0
