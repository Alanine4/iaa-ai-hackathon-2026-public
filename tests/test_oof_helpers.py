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
