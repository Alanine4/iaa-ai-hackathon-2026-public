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
