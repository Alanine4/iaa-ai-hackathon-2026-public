import pytest

from aiiaa import models, params


def test_tuned_params_cover_every_line_and_model():
    assert set(params.TUNED) == {"motor", "buildings", "contents"}
    for line, by_model in params.TUNED.items():
        assert set(by_model) <= {"glm", "gam", "lgbm", "xgb"}


@pytest.mark.parametrize("line", ["motor", "buildings", "contents"])
@pytest.mark.parametrize("name", ["glm", "gam", "lgbm", "xgb"])
def test_tuned_params_are_accepted_by_models(line, name):
    model = models.make(name, **params.for_model(line, name))
    assert model is not None


def test_for_model_returns_empty_dict_when_default_kept():
    assert params.for_model("buildings", "xgb") == {}


def test_for_model_returns_a_copy():
    p = params.for_model("motor", "lgbm")
    p["num_leaves"] = -1
    assert params.for_model("motor", "lgbm")["num_leaves"] != -1
