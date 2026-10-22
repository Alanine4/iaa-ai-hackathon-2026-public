import numpy as np
import pytest

from aiiaa import evaluate


def test_d2_is_one_for_perfect_prediction():
    y = np.array([0.0, 0.0, 120.0, 3000.0])
    assert evaluate.tweedie_d2(y, np.where(y == 0, 1e-9, y)) == pytest.approx(1.0, abs=1e-4)


def test_d2_is_zero_for_constant_mean_prediction():
    y = np.array([0.0, 0.0, 120.0, 3000.0])
    assert evaluate.tweedie_d2(y, np.full(4, y.mean())) == pytest.approx(0.0, abs=1e-12)


def test_d2_matches_manual_deviance_ratio():
    y = np.array([0.0, 50.0, 200.0])
    mu = np.array([10.0, 40.0, 150.0])

    def dev(y, mu, p=1.5):
        return np.sum(2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p)))

    expected = 1 - dev(y, mu) / dev(y, np.full(3, y.mean()))
    assert evaluate.tweedie_d2(y, mu) == pytest.approx(expected)


def test_trend_factor_extrapolates_log_linear_trend_from_exposure_weighted_mean_year():
    years = np.array([2021, 2022, 2023])
    pure_premium = np.array([100.0, 200.0, 400.0])
    exposure = np.array([1.0, 1.0, 1.0])
    # mean year 2022, trend x2 per year -> 2024 is two years ahead
    assert evaluate.trend_factor(years, pure_premium, exposure, 2024) == pytest.approx(4.0)


def test_rescale_trend_moves_prediction_from_one_trend_share_to_another():
    # C020: prediction made with the full factor 1.21 -> half share uses sqrt(1.21) = 1.1
    pred = np.array([121.0, 242.0])
    out = evaluate.rescale_trend(pred, full_factor=1.21, from_share=1.0, to_share=0.5)
    assert out == pytest.approx([110.0, 220.0])


def test_rescale_trend_is_identity_for_equal_shares():
    pred = np.array([5.0])
    assert evaluate.rescale_trend(pred, 1.3, 0.5, 0.5) == pytest.approx([5.0])


def test_blend_weights_put_all_weight_on_the_perfect_model():
    rng = np.random.default_rng(0)
    actual = rng.gamma(0.3, 500.0, 2000)
    folds = [{"actual": actual, "preds": {"good": actual + 1e-6, "bad": np.full(2000, actual.mean())}}]
    w = evaluate.blend_weights(folds, step=0.25)
    assert w == {"good": 1.0, "bad": 0.0}


def test_blend_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(1)
    mu = rng.gamma(2.0, 100.0, 3000)
    actual = rng.gamma(0.5, mu / 0.5)
    folds = [{"actual": actual, "preds": {"a": mu * rng.uniform(0.7, 1.3, 3000), "b": mu * rng.uniform(0.7, 1.3, 3000),
                                          "c": np.full(3000, mu.mean())}}]
    w = evaluate.blend_weights(folds, step=0.1)
    assert all(v >= 0 for v in w.values()) and sum(w.values()) == pytest.approx(1.0)
    assert w["c"] < max(w["a"], w["b"])


def test_trend_factor_is_one_without_trend():
    years = np.array([2021, 2022, 2023])
    assert evaluate.trend_factor(years, np.array([5.0, 5.0, 5.0]), np.ones(3), 2025) == pytest.approx(1.0)


def test_blend_report_gives_per_fold_calibrated_d2_for_weighted_equal_and_single_models():
    # C028
    rng = np.random.default_rng(2)
    folds = []
    for _ in range(2):
        mu = rng.gamma(2.0, 100.0, 2000)
        actual = rng.gamma(0.5, mu / 0.5)
        folds.append({"actual": actual, "preds": {"good": mu * 3.0, "flat": np.full(2000, mu.mean())}})
    rep = evaluate.blend_report(folds, {"good": 1.0, "flat": 0.0}, ["good", "flat"])

    def cal(f, p):
        return evaluate.tweedie_d2(f["actual"], p * f["actual"].sum() / p.sum())

    assert rep["single"]["good"] == pytest.approx([cal(f, f["preds"]["good"]) for f in folds])
    assert rep["single"]["flat"] == pytest.approx([0.0, 0.0], abs=1e-9)  # constant prediction has no ranking power
    assert rep["weighted"] == pytest.approx(rep["single"]["good"])  # all weight on "good"; scale does not matter
    assert rep["equal"] == pytest.approx([cal(f, 0.5 * f["preds"]["good"] + 0.5 * f["preds"]["flat"]) for f in folds])
    assert len(rep["equal"]) == 2
