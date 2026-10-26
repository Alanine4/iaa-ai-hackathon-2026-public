"""C044: test-mix importance weights and the NB / test-mix validation views (src/aiiaa/evaluate.py)."""
import numpy as np
import pytest

from aiiaa import evaluate


def _rows(rng, n=2000, nb_share=0.08):
    nb = rng.uniform(size=n) < nb_share
    exposure = rng.uniform(0.1, 1.0, n)
    return nb, exposure


def test_testmix_weights_move_the_nb_exposure_share_to_the_target_and_keep_total_exposure():
    rng = np.random.default_rng(0)
    nb, e = _rows(rng)
    w = evaluate.testmix_weights(nb, e, 0.77)
    assert (w * e)[nb].sum() / (w * e).sum() == pytest.approx(0.77, rel=1e-12)
    assert (w * e).sum() == pytest.approx(e.sum(), rel=1e-12)  # weights keep the total exposure
    assert len(np.unique(w[nb])) == 1 and len(np.unique(w[~nb])) == 1  # one weight per group
    assert w[nb][0] > 1 > w[~nb][0]


def test_nb_exposure_share_is_exposure_weighted():
    assert evaluate.nb_exposure_share([True, False, False], [3.0, 1.0, 2.0]) == pytest.approx(0.5)


def test_testmix_weights_are_one_when_the_share_already_matches():
    nb = np.array([True, False, False, True])
    e = np.ones(4)
    np.testing.assert_array_equal(evaluate.testmix_weights(nb, e, 0.5), np.ones(4))


@pytest.mark.parametrize("nb", [[False, False, False], [True, True, True]])
def test_testmix_weights_fail_when_one_group_is_missing(nb):
    with pytest.raises(ValueError, match="new-business"):
        evaluate.testmix_weights(np.array(nb), np.ones(3), 0.77)


@pytest.mark.parametrize("share", [-0.1, 1.5])
def test_testmix_weights_reject_a_target_outside_0_1(share):
    with pytest.raises(ValueError, match="target"):
        evaluate.testmix_weights(np.array([True, False]), np.ones(2), share)


def test_testmix_weights_allow_the_boundary_targets():
    nb = np.array([True, False, False])
    np.testing.assert_allclose(evaluate.testmix_weights(nb, np.ones(3), 1.0), [3.0, 0.0, 0.0])


def test_d2_cal_matches_the_unweighted_definition_and_weights_both_deviances():
    rng = np.random.default_rng(1)
    y = np.where(rng.uniform(size=500) < 0.3, rng.gamma(1.5, 200.0, 500), 0.0)
    p = rng.gamma(2.0, 30.0, 500)
    assert evaluate.d2_cal(y, p) == pytest.approx(evaluate.tweedie_d2(y, p * y.sum() / p.sum()))
    w = rng.uniform(0.5, 2.0, 500)
    c = y.sum() / p.sum()  # calibration on unweighted sums: the model's own NB / renewal split is kept

    def dev(mu):
        return 2 * (-4 * np.sqrt(y) + 2 * y / np.sqrt(mu) + 2 * np.sqrt(mu))
    expected = 1 - (w * dev(c * p)).sum() / (w * dev(np.full(500, np.average(y, weights=w)))).sum()
    assert evaluate.d2_cal(y, p, w) == pytest.approx(expected, rel=1e-10)


def test_nb_views_of_one_fold():
    rng = np.random.default_rng(2)
    n = 3000
    nb, e = _rows(rng, n)
    mu = rng.gamma(2.0, 50.0, n) * np.where(nb, 1.5, 1.0)
    y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, mu / 0.45), 0.0)
    claims = (y > 0).astype(int) + (rng.uniform(size=n) < 0.02) * (y > 0)
    v = evaluate.nb_views(y, mu, nb, e, claims, 0.77)
    assert set(v) == {"d2_cal_testmix", "d2_cal_nb", "nb_claims"}
    assert v["nb_claims"] == int(claims[nb].sum())
    assert v["d2_cal_nb"] == pytest.approx(evaluate.d2_cal(y[nb], mu[nb]))
    assert v["d2_cal_testmix"] == pytest.approx(evaluate.d2_cal(y, mu, evaluate.testmix_weights(nb, e, 0.77)))


def test_view_report_has_the_blend_report_layout():
    rng = np.random.default_rng(3)
    folds = []
    for _ in range(3):
        n = 800
        nb, e = _rows(rng, n, 0.3)
        mu = rng.gamma(2.0, 50.0, n)
        y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, mu / 0.45), 0.0)
        folds.append({"actual": y, "preds": {"a": mu, "b": mu * np.exp(rng.normal(0, 0.5, n))},
                      "new_business": nb, "exposure": e, "claim_count": (y > 0).astype(int)})
    rep = evaluate.view_report(folds, {"a": 0.7, "b": 0.3}, ["a", "b"], 0.77)
    assert set(rep) == {"weighted", "equal", "single"} and len(rep["weighted"]) == 3
    f = folds[1]
    assert rep["single"]["b"][1] == evaluate.nb_views(f["actual"], f["preds"]["b"], f["new_business"], f["exposure"],
                                                      f["claim_count"], 0.77)
    blend = 0.7 * f["preds"]["a"] + 0.3 * f["preds"]["b"]
    assert rep["weighted"][1]["d2_cal_testmix"] == pytest.approx(
        evaluate.nb_views(f["actual"], blend, f["new_business"], f["exposure"], f["claim_count"], 0.77)["d2_cal_testmix"])
