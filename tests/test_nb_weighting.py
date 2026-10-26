"""C044 / C045: NB-weighted pooled fits, emerging-only early stopping and the test-mix views in emerging_main."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import oof_predictions  # noqa: E402
from aiiaa import markets, models  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _xy(rng, n=3000):
    x = pd.DataFrame({"v": rng.uniform(0, 1, n), "c": pd.Categorical(rng.choice(["a", "b"], n))})
    y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, 100.0 * (1 + x["v"].to_numpy())), 0.0)
    return x, y, np.ones(n)


def test_inner_split_is_the_same_as_its_explicit_mask():
    # C045: "inner" is now the seeded mask passed through the explicit-mask path; behaviour unchanged (C027)
    rng = np.random.default_rng(0)
    x, y, w = _xy(rng)
    mask = np.zeros(len(x), dtype=bool)
    mask[np.random.default_rng(models.SEED).permutation(len(x))[:len(x) // 10]] = True
    a = models.make("lgbm", n_estimators=300).fit(x, y, w, valid="inner")
    b = models.make("lgbm", n_estimators=300).fit(x, y, w, valid=mask)
    assert a.best_iteration == b.best_iteration
    np.testing.assert_allclose(a.predict(x), b.predict(x))


def test_explicit_mask_holds_its_rows_out_of_the_tree_fit(monkeypatch):
    rng = np.random.default_rng(1)
    x, y, w = _xy(rng, 500)
    mask = np.zeros(len(x), dtype=bool)
    mask[:50] = True
    seen = {}
    m = models.make("xgb", n_estimators=20)

    def spy(xf, yf, wf, valid=None):
        seen["fit"], seen["valid"] = xf.index.tolist(), valid[0].index.tolist()
    monkeypatch.setattr(m, "_fit", spy)
    monkeypatch.setattr(m, "_raw", lambda xx: np.ones(len(xx)))
    m.fit(x, y, w, valid=mask)
    assert seen["valid"] == list(range(50)) and seen["fit"] == list(range(50, 500))


def test_explicit_mask_is_ignored_by_models_without_early_stopping(monkeypatch):
    rng = np.random.default_rng(2)
    x, y, w = _xy(rng, 400)
    m = models.make("glm")
    seen = {}
    monkeypatch.setattr(m, "_fit", lambda xf, yf, wf, valid=None: seen.update(n=len(xf), valid=valid))
    monkeypatch.setattr(m, "_raw", lambda xx: np.ones(len(xx)))
    m.fit(x, y, w, valid=np.arange(400) < 40)
    assert seen == {"n": 400, "valid": None}


def test_explicit_mask_must_be_boolean_over_every_row():
    rng = np.random.default_rng(3)
    x, y, w = _xy(rng, 100)
    with pytest.raises(ValueError, match="mask"):
        models.make("lgbm").fit(x, y, w, valid=np.zeros(99, dtype=bool))
    with pytest.raises(ValueError, match="mask"):
        models.make("lgbm").fit(x, y, w, valid=np.zeros(100, dtype=int))


def test_inner_emerging_holdout_selects_ten_percent_of_the_emerging_fit_rows_only():
    n = 5000
    is_em = np.arange(n) >= 4000
    em_fit = is_em & (np.arange(n) % 5 != 0)  # 800 emerging fit rows
    fit = ~is_em | em_fit
    mask = oof_predictions.inner_emerging_holdout(fit, em_fit)
    assert mask.dtype == bool and len(mask) == fit.sum()
    assert mask.sum() == em_fit.sum() // 10
    assert em_fit[fit][mask].all()  # only emerging fit rows are held out; every developed row stays in the fit
    np.testing.assert_array_equal(mask, oof_predictions.inner_emerging_holdout(fit, em_fit))  # seeded


def test_nb_weights_auto_give_new_business_the_target_share_of_the_fit_weight():
    rng = np.random.default_rng(4)
    n = 10000
    nb = rng.uniform(size=n) < 0.08
    w = rng.uniform(0.1, 1.0, n) * np.where(np.arange(n) < 8000, 0.3, 1.0)  # developed rows carry x0.3
    fit = np.arange(n) < 9500
    wk, factor, share = oof_predictions.nb_weights(w, nb, fit, "auto", 0.77)
    assert share == pytest.approx(0.77, rel=1e-12)
    assert wk[fit & nb].sum() / wk[fit].sum() == pytest.approx(0.77, rel=1e-12)
    expected = 0.77 * w[fit & ~nb].sum() / (0.23 * w[fit & nb].sum())
    assert factor == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(wk, w * np.where(nb, factor, 1.0))  # NB rows of both markets, renewals unchanged


def test_nb_weights_with_a_fixed_factor_and_without_weighting():
    nb = np.array([True, False, False, True])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    fit = np.array([True, True, True, False])
    wk, factor, share = oof_predictions.nb_weights(w, nb, fit, 3.0, 0.77)
    np.testing.assert_allclose(wk, [3.0, 2.0, 3.0, 12.0])
    assert factor == 3.0 and share == pytest.approx(3.0 / 8.0)
    wk, factor, share = oof_predictions.nb_weights(w, nb, fit, None, 0.77)
    assert wk is w and factor == 1.0 and share == pytest.approx(1.0 / 6.0)


def _design(rng):
    n_dev, n_em, n_test = 3000, 1500, 200
    n = n_dev + n_em + n_test
    part = np.array(["dev"] * n_dev + ["em"] * n_em + ["test"] * n_test, dtype=object)
    nb = rng.uniform(size=n) < np.where(part == "test", 0.77, 0.08)
    x = pd.DataFrame({"market": pd.Categorical(np.where(part == "dev", "developed", "emerging")),
                      "new_business": nb, "v": rng.uniform(0, 1, n)})
    y = np.where(rng.uniform(size=n) < 0.25, rng.gamma(1.0, 500.0, n), 0.0) * (part != "test") * np.where(nb, 1.4, 1.0)
    w = np.where(part == "test", 0.0, np.where(part == "dev", 0.3, 1.0))
    year = np.where(np.arange(n_em) < n_em // 2, 2023, 2024)
    train = pd.DataFrame({"policy_id": [f"P{i % 900:04d}" for i in range(n_em)]})
    t = pd.DataFrame({"row_id": [f"E{i:05d}" for i in range(n_em)], "claim_cost": y[part == "em"],
                      "claim_cost_capped": y[part == "em"], "claim_count": (y[part == "em"] > 0).astype(int),
                      "exposure": 1.0, "completeness": 1.0})
    test = pd.DataFrame({"row_id": [f"T{i:05d}" for i in range(n_test)], "exposure": 1.0})
    return x, part, y, w, train, test, t, year


def test_emerging_main_nb_weighted_inner_emerging_run(tmp_path, monkeypatch):
    # C044 + C045 plumbing on a synthetic pooled design
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    rng = np.random.default_rng(5)
    design = _design(rng)
    x, part = design[0], design[1]
    nb, is_em = x["new_business"].to_numpy(), part == "em"
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda *a, **k: design)
    fits = []
    real_make = models.make

    def spy_make(name, **kw):
        m = real_make(name, **kw)
        real_fit = m.fit

        def fit(xf, yf, wf, valid=None):
            fits.append({"name": name, "index": xf.index.to_numpy(), "w": np.asarray(wf), "valid": valid})
            return real_fit(xf, yf, wf, valid=valid)
        m.fit = fit
        return m
    monkeypatch.setattr(oof_predictions.models, "make", spy_make)
    oof_predictions.emerging_main("contents", ["glm", "lgbm"], "pooled", 0.3, "none", nb_weight="auto",
                                  inner_emerging=True)
    out = tmp_path / "output" / "emerging" / "pooled_w0.3_nb"
    assert not (tmp_path / "output" / "emerging" / "pooled_w0.3").exists()
    target = float(nb[part == "test"].mean())  # test exposure is 1.0 per row
    for f in fits:
        rows = f["index"]
        assert f["w"][nb[rows]].sum() / f["w"].sum() == pytest.approx(target, rel=1e-9)
        if f["name"] == "lgbm" and f["valid"] is not None:
            assert isinstance(f["valid"], np.ndarray) and is_em[rows][f["valid"]].all()
            assert f["valid"].sum() == is_em[rows].sum() // 10
    assert sum(f["valid"] is None for f in fits if f["name"] == "lgbm") == 1  # the refit only
    with open(out / "blend_weights_contents.json", encoding="utf-8") as fh:
        res = json.load(fh)
    assert res["test_nb_share"] == pytest.approx(target)
    assert res["nb_weight"]["option"] == "auto" and res["inner_emerging"] is True
    assert len(res["nb_weight"]["factor"]["group"]) == 5 and res["nb_weight"]["refit_factor"] > 1
    for key in ("d2_cal_testmix", "d2_cal_nb", "nb_claims"):
        assert key in res["cv_time"]["lgbm"] and key in res["cv_time"]["blend_equal"]
        assert f"{key}_folds" in res["cv_group"]["lgbm"] and f"{key}_folds" in res["cv_group"]["blend_searched"]
    assert set(res["guard_scores"]["testmix"]) == {"index", "weighted", "equal"}
    oof = pd.read_parquet(out / "oof_contents.parquet")
    assert {"new_business", "exposure", "claim_count"} <= set(oof.columns)
    em_nb = pd.Series(nb[is_em], index=design[6]["row_id"])
    assert (oof["new_business"].to_numpy() == em_nb.reindex(oof["row_id"]).to_numpy()).all()


def test_run_dir_suffix_per_option():
    assert oof_predictions.run_suffix(False, None, False) == ""
    assert oof_predictions.run_suffix(False, "auto", True) == "_nb"
    assert oof_predictions.run_suffix(False, None, True) == "_ie"
    assert oof_predictions.run_suffix(False, 3.0, False) == "_nbw"
    assert oof_predictions.run_suffix(True, None, False) == "_glmx"


def _cli(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "oof_predictions.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_lists_the_nb_options():
    r = _cli("--help")
    assert r.returncode == 0 and "--nb-weight" in r.stdout and "--inner-emerging" in r.stdout


@pytest.mark.parametrize("args", [["--inner-emerging"],
                                  ["--train-set", "pooled", "--nb-weight", "abc"],
                                  ["--train-set", "pooled", "--nb-weight", "-1"]])
def test_cli_rejects_bad_nb_options(args):
    r = _cli("contents", "--market", "emerging", *args)
    assert r.returncode != 0 and "C045" in r.stderr
