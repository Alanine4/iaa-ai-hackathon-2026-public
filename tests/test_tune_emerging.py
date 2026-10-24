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
    rng = np.random.default_rng(1)
    n_pol = 300
    policy = np.repeat([f"P{i}" for i in range(n_pol)], 2)
    year = np.tile([2023, 2024], n_pol)
    n = len(policy)
    x = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(size=n)})
    expo = rng.uniform(0.5, 1.0, n)
    cost = rng.gamma(0.5, 400.0, n) * (rng.uniform(size=n) < 0.2 + 0.2 * x["a"].to_numpy())
    t = pd.DataFrame({"row_id": np.arange(n).astype(str), "exposure": expo, "exposure_eff": expo,
                      "completeness": 0.95, "claim_cost": cost, "claim_cost_capped": np.minimum(cost, 800.0)})
    design = (x, np.array(["em"] * n, dtype=object), t["claim_cost_capped"].to_numpy() / expo, expo,
              pd.DataFrame({"policy_id": policy}), None, t, year)
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda line, train_set, dev_weight: design)
    monkeypatch.setattr(oof_predictions, "run_dir", lambda train_set, dev_weight: str(tmp_path))
    monkeypatch.setattr(tune_cv, "ROOT", str(tmp_path))
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(tune_cv, "emerging_configs",
                        lambda family: [("lgbm", {}), ("lgbm", dict(min_child_samples=5, n_estimators=20))])
    tune_cv.emerging_main("trees", ["motor"])
    fits = pd.read_csv(tmp_path / "cv_trees.csv")
    assert len(fits) == 2 * 6 and set(fits["scheme"]) == {"time", "group"}
    assert {"line", "model", "config", "scheme", "fold", "d2", "d2_cal", "pred_actual", "iterations", "sec"} <= set(fits)
    md = (tmp_path / "docs" / "cv_emerging_trees.md").read_text(encoding="utf-8")
    assert "d2_cal_group" in md and "d2_cal_time" in md
