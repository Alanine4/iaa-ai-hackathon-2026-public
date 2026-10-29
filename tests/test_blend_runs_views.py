"""C044: test-mix views, the testmix adoption gate, the candidate filter and the recipe view report (blend_runs.py)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import blend  # noqa: E402
import blend_runs  # noqa: E402
from aiiaa import evaluate, markets  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _oof(rng, n=600, models=("a", "b"), meta=True):
    """Synthetic OOF frame in the emerging_main layout (1 time fold + 5 grouped folds), model a is the truth;
    NB rows (10%) are riskier than renewal rows and model b misses that."""
    rows = []
    for scheme, folds in (("time", [0]), ("group", range(5))):
        for k in folds:
            nb = np.arange(n) % 10 == 0
            mu = rng.gamma(2.0, 50.0, n) * np.where(nb, 2.0, 1.0)
            y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, mu / 0.45), 0.0)
            frame = pd.DataFrame({"row_id": [f"{scheme}{k}_{i:04d}" for i in range(n)], "scheme": scheme, "fold": k,
                                  "actual": y})
            if meta:
                frame["new_business"] = nb
                frame["exposure"] = 1.0
                frame["claim_count"] = (y > 0).astype(int)
            for j, m in enumerate(models):
                frame[m] = mu if m == "a" else mu / np.where(nb, 2.0, 1.0) * np.exp(rng.normal(0, 0.3 + 0.2 * j, n))
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def emerging_root(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend_runs, "nb_share_2025", lambda line: 0.77)
    return tmp_path / "output" / "emerging"


def _write_oof(base, run, line, frame):
    os.makedirs(base / run, exist_ok=True)
    frame.to_parquet(base / run / f"oof_{line}.parquet", index=False)


def test_load_runs_keeps_row_info_out_of_the_candidates(emerging_root):
    rng = np.random.default_rng(0)
    oof = _oof(rng)
    _write_oof(emerging_root, "new", "motor", oof)
    _write_oof(emerging_root, "old", "motor", oof.drop(columns=blend_runs.META))  # a run from before C044
    frame, names = blend_runs.load_runs("motor", ["old", "new"])
    assert names == ["old:a", "old:b", "new:a", "new:b"]
    assert list(frame.columns[:7]) == blend_runs.KEYS + blend_runs.META
    srt = oof.sort_values(["scheme", "fold", "row_id"]).reset_index(drop=True)
    assert (frame["new_business"].to_numpy() == srt["new_business"].to_numpy()).all()


def test_load_runs_takes_row_info_from_the_emerging_tables_for_old_runs(emerging_root, monkeypatch):
    rng = np.random.default_rng(1)
    oof = _oof(rng)
    _write_oof(emerging_root, "old", "motor", oof.drop(columns=blend_runs.META))
    info = oof[["row_id"] + blend_runs.META].drop_duplicates("row_id").set_index("row_id")
    monkeypatch.setattr(blend_runs, "row_info", lambda line: info.iloc[::-1])  # order must not matter
    frame, _ = blend_runs.load_runs("motor", ["old"])
    srt = oof.sort_values(["scheme", "fold", "row_id"]).reset_index(drop=True)
    assert (frame["new_business"].to_numpy() == srt["new_business"].to_numpy()).all()
    np.testing.assert_array_equal(frame["claim_count"].to_numpy(), srt["claim_count"].to_numpy())


def test_load_runs_fails_when_row_info_differs_between_runs(emerging_root):
    rng = np.random.default_rng(2)
    oof = _oof(rng)
    other = oof.copy()
    other.loc[3, "new_business"] = not other.loc[3, "new_business"]
    _write_oof(emerging_root, "r1", "motor", oof)
    _write_oof(emerging_root, "r2", "motor", other)
    with pytest.raises(AssertionError, match="row info"):
        blend_runs.load_runs("motor", ["r1", "r2"])


def test_scores_report_the_four_views_with_the_evaluate_definitions(emerging_root):
    rng = np.random.default_rng(3)
    _write_oof(emerging_root, "r1", "motor", _oof(rng))
    frame, names = blend_runs.load_runs("motor", ["r1"])
    folds = blend_runs.scheme_folds(frame, names)
    s = blend_runs.blend_scores(folds, {"r1:a": 0.6, "r1:b": 0.4}, 0.77)
    g = folds["group"][2]
    p = 0.6 * g["preds"]["r1:a"] + 0.4 * g["preds"]["r1:b"]
    v = evaluate.nb_views(g["actual"], p, g["new_business"], g["exposure"], g["claim_count"], 0.77)
    assert s["group_testmix_folds"][2] == pytest.approx(v["d2_cal_testmix"])
    assert s["group_nb_folds"][2] == pytest.approx(v["d2_cal_nb"])
    assert s["group_testmix"] == pytest.approx(np.mean(s["group_testmix_folds"]))
    assert s["group_mean"] == pytest.approx(np.mean([evaluate.d2_cal(f["actual"], 0.6 * f["preds"]["r1:a"]
                                                                     + 0.4 * f["preds"]["r1:b"]) for f in folds["group"]]))
    assert set(s) >= {"time", "group_mean", "time_testmix", "group_testmix", "time_nb", "group_nb", "nb_claims_time",
                      "nb_claims_group"}


@pytest.mark.parametrize("gain, raw, testmix", [
    ({"time": 0.003, "group_mean": 0.002, "mean": 0.0025, "group_testmix": 0.0}, True, True),
    ({"time": 0.003, "group_mean": 0.002, "mean": 0.0025, "group_testmix": -0.0001}, True, False),
    ({"time": 0.001, "group_mean": 0.001, "mean": 0.001, "group_testmix": 0.0025}, False, True),
    ({"time": 0.001, "group_mean": 0.001, "mean": 0.001, "group_testmix": 0.0015}, False, False),
    ({"time": -0.0001, "group_mean": 0.005, "mean": 0.00245, "group_testmix": 0.004}, False, False),
    ({"time": 0.004, "group_mean": -0.0001, "mean": 0.00195, "group_testmix": 0.004}, False, False),
])
def test_adoption_gates(gain, raw, testmix):
    assert blend_runs.adopt(gain, "raw") is raw
    assert blend_runs.adopt(gain, "testmix") is testmix


def test_evaluate_line_searches_only_the_candidates_and_scores_the_baseline_on_all_columns(emerging_root):
    rng = np.random.default_rng(4)
    _write_oof(emerging_root, "r1", "motor", _oof(rng, models=("a", "b", "c")))
    frame, names = blend_runs.load_runs("motor", ["r1"])
    folds = blend_runs.scheme_folds(frame, names)
    res = blend_runs.evaluate_line(folds, ["r1:b", "r1:c"], {"r1:a": 1.0}, 0.1, 0.77, gate="testmix")
    assert set(res["weights"]) == {"r1:b", "r1:c"} and set(res["single"]) == {"r1:b", "r1:c"}
    assert res["baseline"]["weights"] == {"r1:a": 1.0}
    assert res["gain"]["group_testmix"] == pytest.approx(res["blend"]["group_testmix"] - res["baseline"]["group_testmix"])
    assert res["adoption"]["gate"] == "testmix" and res["adoption"]["passes"] is False  # the truth model is the baseline
    assert set(res["guard_scores"]["testmix"]) == {"index", "weighted", "equal"}
    assert {"time_testmix", "group_nb"} <= set(res["single"]["r1:b"])


def test_candidate_filter_must_name_loaded_columns(emerging_root):
    rng = np.random.default_rng(5)
    _write_oof(emerging_root, "r1", "motor", _oof(rng, n=200))
    with pytest.raises(AssertionError, match="candidates"):
        blend_runs.main(["motor"], ["r1"], "none", "unused", 0.25, "xrun_t", write=False,
                        baseline={"motor": {"r1:b": 1.0}}, candidates=["r1:a", "r9:x"])
    blend_runs.main(["motor"], ["r1"], "none", "unused", 0.25, "xrun_t", write=False,
                    baseline={"motor": {"r1:b": 1.0}}, candidates=["r1:a"], gate="testmix")
    with open(emerging_root / "xrun_t" / "blend_weights_motor.json", encoding="utf-8") as f:
        res = json.load(f)
    assert res["candidates"] == ["r1:a"] and res["weights"] == {"r1:a": 1.0} and res["test_nb_share"] == 0.77
    assert res["adoption"]["gate"] == "testmix"


def test_pooled_view_of_one_line_equals_the_line_view(emerging_root):
    rng = np.random.default_rng(6)
    _write_oof(emerging_root, "r1", "motor", _oof(rng))
    frame, names = blend_runs.load_runs("motor", ["r1"])
    folds = {"motor": blend_runs.scheme_folds(frame, names)}
    recipe = {"motor": {"r1:a": 0.5, "r1:b": 0.5}}
    pooled = blend_runs.pooled_scores(folds, recipe, {"motor": 0.77})
    line = blend_runs.blend_scores(folds["motor"], recipe["motor"], 0.77)
    for k in ("time", "group_mean", "time_testmix", "group_testmix", "time_nb", "group_nb"):
        assert pooled[k] == pytest.approx(line[k], rel=1e-10), k


def test_pooled_view_stacks_lines_with_their_own_calibration_and_weights(emerging_root):
    rng = np.random.default_rng(7)
    folds, recipe = {}, {}
    for line in ("motor", "contents"):
        _write_oof(emerging_root, "r1", line, _oof(rng, n=300))
        frame, names = blend_runs.load_runs(line, ["r1"])
        folds[line] = blend_runs.scheme_folds(frame, names)
        recipe[line] = {"r1:a": 1.0}
    pooled = blend_runs.pooled_scores(folds, recipe, {"motor": 0.77, "contents": 0.5})
    ys, ps, ws = [], [], []
    for line, share in (("motor", 0.77), ("contents", 0.5)):
        f = folds[line]["time"][0]
        p = f["preds"]["r1:a"]
        ys.append(f["actual"]); ps.append(p * f["actual"].sum() / p.sum())
        ws.append(evaluate.testmix_weights(f["new_business"], f["exposure"], share))
    y, p, w = np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)
    assert pooled["time"] == pytest.approx(evaluate.tweedie_d2(y, p))
    assert pooled["time_testmix"] == pytest.approx(evaluate.tweedie_d2(y, p, w))


def test_recipes_name_the_runs_and_sum_to_one():
    assert set(blend_runs.RECIPES) >= {"P2-S02", "P2-S03", "P2-S04", "P2-S05"}
    assert blend_runs.RECIPES["P2-S05"] == blend_runs.BASELINE
    for rec in blend_runs.RECIPES.values():
        for line, w in rec.items():
            assert sum(w.values()) == pytest.approx(1.0) and all(":" in c for c in w)


def test_cli_lists_the_new_options():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "blend_runs.py"), "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    for opt in ("--gate", "--candidates", "--report-views", "mixed-full"):
        assert opt in r.stdout


def _write_pred(base, run, line, model, ids, values, trend_factor):
    os.makedirs(base / run, exist_ok=True)
    pd.DataFrame({"row_id": ids, "prediction": values, "trend": "x", "trend_factor": trend_factor}).to_parquet(
        base / run / f"pred_{line}_{model}.parquet", index=False)


def test_write_from_builds_the_submission_from_each_lines_result_folder(emerging_root, monkeypatch):
    # C044: per-line result folders (different runs / candidates per line) -> one submission
    sample = pd.DataFrame({"row_id": ["r1", "r2", "r3", "r4"], "prediction": 0.0})
    monkeypatch.setattr(blend, "load", lambda name, market=None: sample)
    monkeypatch.setattr(blend, "EMERGING_ROWS", len(sample))
    _write_pred(emerging_root, ".", "motor", "xgb", ["r1", "r2"], [10.0, 20.0], 1.0)
    _write_pred(emerging_root, "w03", "motor", "lgbm", ["r1", "r2"], [30.0, 40.0], 2.0)
    _write_pred(emerging_root, "w1", "buildings", "cat", ["r3"], [100.0], 1.0)
    _write_pred(emerging_root, "w03_nb", "contents", "glm", ["r4"], [5.0], 1.0)
    picks = {"motor": ("xm", {".:xgb": 0.5, "w03:lgbm": 0.5}), "buildings": ("xb", {"w1:cat": 1.0}),
             "contents": ("xc", {"w03_nb:glm": 1.0})}
    for line, (out, w) in picks.items():
        os.makedirs(emerging_root / out, exist_ok=True)
        with open(emerging_root / out / f"blend_weights_{line}.json", "w", encoding="utf-8") as f:
            json.dump({"submission_weights": w}, f)
    spec = blend_runs.parse_write_from("motor=xm,buildings=xb,contents=xc")
    sub = blend_runs.write_from(spec, "mixed-full", "wf_test")
    fac = markets.get("emerging").trends["mixed-full"]
    expected = [(0.5 * 10 + 0.5 * 30 / 2) * fac["motor"], (0.5 * 20 + 0.5 * 40 / 2) * fac["motor"],
                100 * fac["buildings"], 5 * fac["contents"]]
    np.testing.assert_allclose(sub.to_numpy(), expected)
    got = pd.read_csv(emerging_root.parent.parent / "work" / "wf_test.csv")
    np.testing.assert_allclose(got["prediction"].to_numpy(), expected)


@pytest.mark.parametrize("text", ["motor=a,buildings=b", "motor=a,buildings=b,contents=c,extra=d", "motor"])
def test_parse_write_from_needs_every_line_once(text):
    with pytest.raises(ValueError, match="--write-from"):
        blend_runs.parse_write_from(text)


def test_write_from_takes_the_baseline_weights_for_a_line_marked_baseline(emerging_root, monkeypatch):
    seen = {}
    monkeypatch.setattr(blend_runs, "line_prediction", lambda line, w, trend, pred_tag=None: seen.setdefault(
        line, w) and pd.Series([1.0], index=[line]))  # C057: the real function takes pred_tag too
    monkeypatch.setattr(blend_runs, "write_submission", lambda by_line, name: by_line)
    os.makedirs(emerging_root / "xc", exist_ok=True)
    with open(emerging_root / "xc" / "blend_weights_contents.json", "w", encoding="utf-8") as f:
        json.dump({"submission_weights": {"r:glm": 1.0}}, f)
    blend_runs.write_from(blend_runs.parse_write_from("motor=baseline,buildings=baseline,contents=xc"), "none", "x")
    assert seen == {"motor": blend_runs.BASELINE["motor"], "buildings": blend_runs.BASELINE["buildings"],
                    "contents": {"r:glm": 1.0}}
