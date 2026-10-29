"""C046 / C047: emerging-only tuned parameters, `--tag` run folders and `--rows nb` (new-business fit rows)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import oof_predictions  # noqa: E402
from aiiaa import markets, models, params  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_emerging_tuned_parameters_reach_emerging_only_fits_but_not_pooled_fits(monkeypatch):
    # C046: TUNED["emerging-<line>"] is tuned on emerging-only data; pooled fits keep the defaults (C039)
    monkeypatch.setitem(params.TUNED, "emerging-motor", {"xgb": {"max_depth": 6}})
    assert oof_predictions.emerging_model("motor", "xgb", "emerging").params["max_depth"] == 6
    assert oof_predictions.emerging_model("motor", "xgb", "pooled").params["max_depth"] == models.make("xgb").params[
        "max_depth"]
    # an explicit config (tune_cv) still overrides
    assert oof_predictions.emerging_model("motor", "xgb", "pooled", {"max_depth": 4}).params["max_depth"] == 4


def test_run_folder_with_and_without_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    base = markets.get("emerging").out_dir
    assert oof_predictions.run_folder("emerging", 0.3, "", None) == base
    assert oof_predictions.run_folder("emerging", 0.3, "", "c046") == os.path.join(base, "emerging_c046")
    assert oof_predictions.run_folder("pooled", 0.3, "", None) == os.path.join(base, "pooled_w0.3")
    assert oof_predictions.run_folder("pooled", 1.0, "_nbrows", None) == os.path.join(base, "pooled_w1_nbrows")
    assert oof_predictions.run_folder("pooled", 0.3, "_nbrows", "x") == os.path.join(base, "pooled_w0.3_nbrows_x")


def test_run_suffix_for_new_business_rows():
    assert oof_predictions.run_suffix(False, None, False, rows="nb") == "_nbrows"
    assert oof_predictions.run_suffix(False, None, False) == ""


def _design(rng):
    n_dev, n_em, n_test = 3000, 1500, 200
    n = n_dev + n_em + n_test
    part = np.array(["dev"] * n_dev + ["em"] * n_em + ["test"] * n_test, dtype=object)
    nb = rng.uniform(size=n) < np.where(part == "test", 0.77, 0.25)
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


def test_emerging_main_new_business_rows_fit_only_nb_rows_and_score_every_validation_row(tmp_path, monkeypatch):
    # C047: fit sets (folds and refit) hold only NB rows of both markets; the C038 re-balance uses the emerging NB
    # fit rows; validation scores every emerging validation row; the refit predicts every test row
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    design = _design(np.random.default_rng(7))
    x, part = design[0], design[1]
    nb, is_em = x["new_business"].to_numpy(), part == "em"
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda *a, **k: design)
    calls = []
    real = oof_predictions.fit_emerging

    def spy(m, xx, y, w, fit, em_fit, train_set, valid):
        calls.append((fit.copy(), em_fit.copy()))
        return real(m, xx, y, w, fit, em_fit, train_set, valid)
    monkeypatch.setattr(oof_predictions, "fit_emerging", spy)
    oof_predictions.emerging_main("contents", ["glm", "lgbm"], "pooled", 0.3, "none", rows="nb")
    out = tmp_path / "output" / "emerging" / "pooled_w0.3_nbrows"
    assert len(calls) == 2 * 6 + 2  # 6 schemes + refit, two models
    for fit, em_fit in calls:
        assert nb[fit].all() and fit[(part == "dev") & nb].all()  # every developed NB row, nothing else
        assert (em_fit == (fit & is_em)).all() and em_fit.any()
    refit_fit, refit_em = calls[-1]
    assert (refit_em == (is_em & nb)).all()
    oof = pd.read_parquet(out / "oof_contents.parquet")
    assert len(oof) == (design[7] == 2024).sum() + is_em.sum()  # time fold: 2024 rows; grouped: every row once
    assert oof["lgbm"].notna().all() and (~oof["new_business"]).any()
    pred = pd.read_parquet(out / "pred_contents_lgbm.parquet")
    assert len(pred) == (part == "test").sum()
    with open(out / "blend_weights_contents.json", encoding="utf-8") as fh:
        res = json.load(fh)
    assert res["rows"] == "nb"
    assert res["fit_rows"]["refit"] == {"developed": int(((part == "dev") & nb).sum()), "emerging": int(refit_em.sum())}


def test_emerging_main_tag_writes_to_the_tagged_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    x, part, y, w, train, test, t, year = _design(np.random.default_rng(8))
    keep = part != "dev"
    design = (x[keep].reset_index(drop=True), part[keep], y[keep], w[keep], train, test, t, year)
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda *a, **k: design)
    oof_predictions.emerging_main("motor", ["glm"], "emerging", 0.3, "none", refit=False, tag="c046")
    base = tmp_path / "output" / "emerging"
    assert (base / "emerging_c046" / "oof_motor.parquet").exists()
    assert not (base / "oof_motor.parquet").exists()


def _cli(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "oof_predictions.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_lists_tag_and_rows():
    r = _cli("--help")
    assert r.returncode == 0 and "--tag" in r.stdout and "--rows" in r.stdout


def test_cli_rejects_new_business_rows_outside_pooled_runs():
    r = _cli("contents", "--market", "emerging", "--rows", "nb")
    assert r.returncode != 0 and "C047" in r.stderr
