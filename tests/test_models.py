import numpy as np
import pandas as pd
import pytest

from aiiaa import models


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 20000
    x = pd.DataFrame({
        "risky": pd.Categorical(rng.choice(["no", "yes"], n)),
        "age": rng.uniform(18, 80, n),
        "flag": rng.random(n) < 0.3,
    })
    x.loc[rng.random(n) < 0.05, "age"] = np.nan
    exposure = rng.uniform(0.2, 1.0, n)
    rate = 100 * np.where(x["risky"] == "yes", 3.0, 1.0)
    counts = rng.poisson(0.2 * exposure * np.where(x["risky"] == "yes", 3.0, 1.0))
    cost = np.array([rng.gamma(2.0, 250.0, k).sum() for k in counts])
    return x, cost / exposure, exposure, rate


@pytest.mark.parametrize("name", ["glm", "gam", "lgbm", "xgb", "cat"])
def test_model_predicts_positive_rates_and_ranks_risky_group_higher(name, data):
    x, y, w, _ = data
    model = models.make(name, n_estimators=150)
    model.fit(x, y, w)
    pred = model.predict(x)
    assert pred.shape == (len(x),)
    assert (pred > 0).all()
    ratio = pred[x["risky"] == "yes"].mean() / pred[x["risky"] == "no"].mean()
    assert 2.0 < ratio < 4.5


def test_tree_hyperparameters_can_be_overridden():
    lg = models.make("lgbm", num_leaves=15, min_child_samples=500, learning_rate=0.03)
    assert lg.model.get_params()["num_leaves"] == 15
    assert lg.model.get_params()["min_child_samples"] == 500
    assert lg.model.get_params()["learning_rate"] == 0.03
    xg = models.make("xgb", max_depth=5, min_child_weight=10)
    assert xg.params["max_depth"] == 5 and xg.params["min_child_weight"] == 10
    ct = models.make("cat", depth=6, l2_leaf_reg=10, n_estimators=200)
    assert ct.params["depth"] == 6 and ct.params["l2_leaf_reg"] == 10
    assert ct.params["iterations"] == 200 and "n_estimators" not in ct.params


def test_gam_recovers_smooth_u_shaped_effect():
    rng = np.random.default_rng(1)
    n = 30000
    age = rng.uniform(18, 85, n)
    x = pd.DataFrame({"age": age, "region": pd.Categorical(rng.choice(["a", "b"], n))})
    mu = 50 * (1 + ((age - 50) / 20) ** 2)
    y = rng.gamma(1.0, mu)
    model = models.make("gam").fit(x, y, np.ones(n))
    grid = pd.DataFrame({"age": [20.0, 50.0, 80.0], "region": pd.Categorical(["a"] * 3, categories=["a", "b"])})
    young, middle, old = model.predict(grid)
    assert young > 1.8 * middle and old > 1.8 * middle


@pytest.mark.parametrize("name", ["glm", "gam", "lgbm", "xgb", "cat"])
def test_model_mean_prediction_close_to_weighted_mean_rate(name, data):
    x, y, w, _ = data
    model = models.make(name, n_estimators=150)
    model.fit(x, y, w)
    assert np.average(model.predict(x), weights=w) == pytest.approx(np.average(y, weights=w), rel=0.05)


@pytest.mark.parametrize("name", ["glm", "gam", "lgbm", "xgb", "cat"])
def test_model_is_exactly_balanced_on_training_data(name, data):
    # C013: log-link Tweedie fits are not balance-preserving; a training-data correction restores it
    x, y, w, _ = data
    model = models.make(name, n_estimators=150)
    model.fit(x, y, w)
    # XGBoost predicts in float32, so totals agree to ~1e-8 rather than to double precision
    assert np.sum(model.predict(x) * w) == pytest.approx(np.sum(y * w), rel=1e-6)
