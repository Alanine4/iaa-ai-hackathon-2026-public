import pytest

from aiiaa import models, params


def test_tuned_params_cover_every_line_and_model():
    # C036: emerging-market lines are separate keys, empty (model defaults) unless tuned on emerging data
    assert set(params.TUNED) == {"motor", "buildings", "contents", "emerging-motor", "emerging-buildings",
                                 "emerging-contents"}
    for line, by_model in params.TUNED.items():
        # C046: emerging keys hold emerging-only tree tunings (CatBoost included)
        allowed = {"lgbm", "xgb", "cat"} if line.startswith("emerging-") else {"glm", "gam", "lgbm", "xgb"}
        assert set(by_model) <= allowed


@pytest.mark.parametrize("line", ["motor", "buildings", "contents"])
@pytest.mark.parametrize("name", ["glm", "gam", "lgbm", "xgb"])
def test_tuned_params_are_accepted_by_models(line, name):
    model = models.make(name, **params.for_model(line, name))
    assert model is not None


def test_for_model_returns_empty_dict_when_default_kept():
    assert params.for_model("buildings", "xgb") == {}


@pytest.mark.parametrize("line", ["emerging-motor", "emerging-buildings", "emerging-contents"])
@pytest.mark.parametrize("name", ["glm", "gam"])
def test_emerging_linear_models_use_model_defaults(line, name):
    assert params.for_model(line, name) == {}


@pytest.mark.parametrize("line", ["emerging-motor", "emerging-buildings", "emerging-contents"])
@pytest.mark.parametrize("name", ["lgbm", "xgb", "cat"])
def test_emerging_tuned_params_are_accepted_by_models(line, name):
    # C046: picks of the emerging-only search (tune_cv --market emerging), defaults where nothing beat them
    assert models.make(name, **params.for_model(line, name)) is not None


def test_unknown_line_raises():
    with pytest.raises(KeyError):
        params.for_model("emerging-pets", "glm")


def test_for_model_returns_a_copy():
    p = params.for_model("motor", "lgbm")
    p["num_leaves"] = -1
    assert params.for_model("motor", "lgbm")["num_leaves"] != -1
