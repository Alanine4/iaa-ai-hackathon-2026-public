import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import oof_predictions  # noqa: E402
import tune_cv  # noqa: E402
from aiiaa import models  # noqa: E402


def test_emerging_trees_search_has_default_first_and_distinct_configs_per_model():
    # C039: LGBM 3 x 4 grid, XGB 3 x 3, CatBoost 3 x 3, each preceded by the model defaults ({})
    cfgs = tune_cv.emerging_configs("trees")
    for name, n in (("lgbm", 13), ("xgb", 10), ("cat", 10)):
        own = [c for m, c in cfgs if m == name]
        assert len(own) == n and own[0] == {}
        assert len({tuple(sorted(c.items())) for c in own}) == n
    assert {m for m, _ in cfgs} == {"lgbm", "xgb", "cat"}


def test_emerging_search_grid_values():
    lgbm = [c for m, c in tune_cv.emerging_configs("trees") if m == "lgbm" and c]
    assert {c["num_leaves"] for c in lgbm} == {7, 15, 31}
    assert {c["min_child_samples"] for c in lgbm} == {50, 200, 1000, 3000}
    assert all(c["learning_rate"] == 0.05 and c["colsample_bytree"] == 0.8 and c["subsample"] == 0.8 for c in lgbm)


def test_emerging_search_configs_are_accepted_by_models():
    for name, cfg in tune_cv.emerging_configs("trees"):
        assert models.make(name, **cfg) is not None


def test_emerging_linear_search_is_refused():
    with pytest.raises(ValueError, match="linear"):
        tune_cv.emerging_configs("linear")


def test_emerging_schemes_validate_every_row_once_and_time_fold_fits_2023_only():
    # C039: scheme construction shared with oof_predictions.emerging_main
    policy = np.repeat([f"P{i}" for i in range(10)], 2)
    year = np.tile([2023, 2024], 10)
    train = pd.DataFrame({"policy_id": policy})
    schemes = oof_predictions.emerging_schemes(train, year)
    assert [s[0] for s in schemes] == ["time"] + ["group"] * oof_predictions.GROUP_FOLDS
    _, k, fit, val = schemes[0]
    assert k == 0 and (year[fit] == 2023).all() and fit.sum() == 10 and (year[val] == 2024).all() and val.sum() == 10
    group = schemes[1:]
    assert [s[1] for s in group] == list(range(oof_predictions.GROUP_FOLDS))
    assert (sum(s[3].astype(int) for s in group) == 1).all()
    for _, _, fit_k, val_k in group:
        assert not (fit_k & val_k).any() and (fit_k | val_k).all()
        # a policy never sits on both sides of a grouped fold
        assert not set(policy[fit_k]) & set(policy[val_k])


def test_emerging_summary_ranks_by_grouped_mean_and_keeps_time_fold_and_gain_over_defaults():
    rows = []
    for cfg, group, time_fold in (("{}", [0.04, 0.06], 0.05), ("{'depth': 6}", [0.07, 0.09], 0.03)):
        rows += [dict(line="motor", model="cat", config=cfg, scheme="group", fold=k, d2=d, d2_cal=d, pred_actual=1.0,
                      iterations=100) for k, d in enumerate(group)]
        rows.append(dict(line="motor", model="cat", config=cfg, scheme="time", fold=0, d2=time_fold,
                         d2_cal=time_fold, pred_actual=1.0, iterations=90))
    s = tune_cv.emerging_summary(pd.DataFrame(rows))
    assert list(s["config"]) == ["{'depth': 6}", "{}"]
    assert s["d2_cal_group"].round(4).tolist() == [0.08, 0.05] and s["d2_cal_time"].round(4).tolist() == [0.03, 0.05]
    assert s["gain_group"].round(4).tolist() == [0.03, 0.0] and s["gain_time"].round(4).tolist() == [-0.02, 0.0]


def test_emerging_main_writes_one_csv_row_per_fit_and_the_ranked_summary(tmp_path, monkeypatch):
    # C039: the whole loop on a synthetic design (no data files): 2 configs x (1 time + 5 grouped) schemes
    design = _synthetic_design()[0]  # C046: the design carries new_business and test rows for the views
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda line, train_set, dev_weight: design)
    monkeypatch.setattr(oof_predictions, "run_dir", lambda train_set, dev_weight: str(tmp_path))
    monkeypatch.setattr(tune_cv, "ROOT", str(tmp_path))
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(tune_cv, "emerging_configs",
                        lambda family, cat_depths=None: [("lgbm", {}),
                                                         ("lgbm", dict(min_child_samples=5, n_estimators=20))])
    tune_cv.emerging_main("trees", ["motor"])
    fits = pd.read_csv(tmp_path / "cv_trees.csv")
    assert len(fits) == 2 * 6 and set(fits["scheme"]) == {"time", "group"}
    assert {"line", "model", "config", "scheme", "fold", "d2", "d2_cal", "pred_actual", "iterations", "sec"} <= set(fits)
    md = (tmp_path / "docs" / "cv_emerging_trees.md").read_text(encoding="utf-8")
    assert "d2_cal_group" in md and "d2_cal_time" in md


def _synthetic_design(n_pol=300, seed=1):
    rng = np.random.default_rng(seed)
    policy = np.repeat([f"P{i}" for i in range(n_pol)], 2)
    year = np.tile([2023, 2024], n_pol)
    n = len(policy)
    nb = rng.uniform(size=n) < 0.3
    x = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(size=n), "new_business": nb})
    expo = rng.uniform(0.5, 1.0, n)
    cost = rng.gamma(0.5, 400.0, n) * (rng.uniform(size=n) < 0.2 + 0.2 * x["a"].to_numpy())
    t = pd.DataFrame({"row_id": np.arange(n).astype(str), "exposure": expo, "exposure_eff": expo,
                      "completeness": 0.95, "claim_cost": cost, "claim_cost_capped": np.minimum(cost, 800.0),
                      "claim_count": (cost > 0).astype(int)})
    x_all = pd.concat([x, x.iloc[:40].assign(new_business=np.arange(40) < 30)], ignore_index=True)
    part = np.array(["em"] * n + ["test"] * 40, dtype=object)
    y = np.concatenate([t["claim_cost_capped"].to_numpy() / expo, np.zeros(40)])
    w = np.concatenate([expo, np.zeros(40)])
    test = pd.DataFrame({"row_id": [f"T{i}" for i in range(40)], "exposure": 1.0})
    return (x_all, part, y, w, pd.DataFrame({"policy_id": policy}), test, t, year), nb, t


def test_emerging_main_adds_the_c044_views_per_fit_and_to_the_summary(tmp_path, monkeypatch):
    # C046: every fit row carries the test-mix and NB-only d2_cal (C044 nb_views with the test NB exposure share)
    design, nb, t = _synthetic_design()
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda line, train_set, dev_weight: design)
    monkeypatch.setattr(oof_predictions, "run_dir", lambda train_set, dev_weight: str(tmp_path))
    monkeypatch.setattr(tune_cv, "ROOT", str(tmp_path))
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(tune_cv, "emerging_configs",
                        lambda family, cat_depths=None: [("lgbm", {}), ("lgbm", dict(min_child_samples=5))])
    seen = []
    real = tune_cv.evaluate.nb_views

    def spy(actual, pred, new_business, exposure, claim_count, share):
        seen.append(share)
        return real(actual, pred, new_business, exposure, claim_count, share)
    monkeypatch.setattr(tune_cv.evaluate, "nb_views", spy)
    tune_cv.emerging_main("trees", ["motor"])
    fits = pd.read_csv(tmp_path / "cv_trees.csv")
    assert {"d2_cal_testmix", "d2_cal_nb", "nb_claims"} <= set(fits)
    assert len(seen) == 12 and all(s == pytest.approx(0.75) for s in seen)  # 30 of 40 test rows NB, exposure 1
    # the views are those of the fold's validation rows: NB claim count per fold
    schemes = oof_predictions.emerging_schemes(design[4], design[7])
    claims = [int(t["claim_count"].to_numpy()[val & nb].sum()) for _, _, _, val in schemes]
    assert fits.groupby("config", sort=False)["nb_claims"].apply(list).tolist() == [claims, claims]
    md = (tmp_path / "docs" / "cv_emerging_trees.md").read_text(encoding="utf-8")
    for col in ("d2_cal_testmix_group", "d2_cal_nb_group", "gain_testmix_group", "gain_nb_group"):
        assert col in md


def test_emerging_summary_carries_the_views_and_their_gains_over_defaults():
    rows = []
    for cfg, raw, mix, nbv in (("{}", 0.05, 0.10, 0.20), ("{'depth': 6}", 0.06, 0.08, 0.25)):
        for scheme, k in (("time", 0), ("group", 0), ("group", 1)):
            rows.append(dict(line="motor", model="cat", config=cfg, scheme=scheme, fold=k, d2=raw, d2_cal=raw,
                             d2_cal_testmix=mix + 0.01 * k, d2_cal_nb=nbv, nb_claims=10, pred_actual=1.0,
                             iterations=100))
    s = tune_cv.emerging_summary(pd.DataFrame(rows)).set_index("config")
    assert s.loc["{}", "d2_cal_testmix_group"] == pytest.approx(0.105)
    assert s.loc["{'depth': 6}", "gain_testmix_group"] == pytest.approx(-0.02)
    assert s.loc["{'depth': 6}", "gain_nb_group"] == pytest.approx(0.05)
    assert s.loc["{'depth': 6}", "d2_cal_testmix_time"] == pytest.approx(0.08)
    assert s.loc["{}", "nb_claims_group"] == 20


def test_cat_depth_cap_keeps_the_default_and_drops_its_duplicate():
    # C046: --cat-depths 4,6 caps the CatBoost grid; depth 4 / l2 3 equals the CatBoost defaults and is skipped
    cat = [c for m, c in tune_cv.emerging_configs("trees", cat_depths=[4, 6]) if m == "cat"]
    assert cat[0] == {} and len(cat) == 6
    assert {c["depth"] for c in cat[1:]} == {4, 6} and {"depth": 4, "l2_leaf_reg": 3} not in cat
    others = [(m, c) for m, c in tune_cv.emerging_configs("trees", cat_depths=[4, 6]) if m != "cat"]
    assert others == [(m, c) for m, c in tune_cv.emerging_configs("trees") if m != "cat"]
