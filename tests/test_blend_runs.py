"""C041: cross-run blend of the emerging pooled runs (scripts/blend_runs.py)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import blend  # noqa: E402
import blend_runs  # noqa: E402
from aiiaa import markets  # noqa: E402


def _oof(rng, n=600, models=("a", "b")):
    """Synthetic OOF frame in the emerging_main layout: 1 time fold + 5 grouped folds; model a is the truth."""
    rows = []
    for scheme, folds in (("time", [0]), ("group", range(5))):
        for k in folds:
            mu = rng.gamma(2.0, 50.0, n)
            y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, mu / 0.45), 0.0)
            frame = pd.DataFrame({"row_id": [f"{scheme}{k}_{i:04d}" for i in range(n)], "scheme": scheme, "fold": k,
                                  "actual": y, "new_business": np.arange(n) % 10 == 0, "exposure": 1.0,
                                  "claim_count": (y > 0).astype(int)})  # C044 row info
            for j, m in enumerate(models):
                frame[m] = mu if m == "a" else mu * np.exp(rng.normal(0, 0.6 + 0.2 * j, n))
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def emerging_root(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend_runs, "nb_share_2025", lambda line: 0.77)  # C044
    return tmp_path / "output" / "emerging"


def _write_oof(base, run, line, frame):
    os.makedirs(base / run, exist_ok=True)
    frame.to_parquet(base / run / f"oof_{line}.parquet", index=False)


def test_load_runs_aligns_on_sorted_keys_and_names_candidates_by_run(emerging_root):
    rng = np.random.default_rng(0)
    oof = _oof(rng)
    _write_oof(emerging_root, "r1", "motor", oof)
    # same keys in a different row order; this run has only one model, as a GLM-only run has
    _write_oof(emerging_root, "r2", "motor",
               oof[["row_id", "scheme", "fold", "actual", "b"]].sample(frac=1, random_state=1))
    frame, names = blend_runs.load_runs("motor", ["r1", "r2"])
    assert names == ["r1:a", "r1:b", "r2:b"]
    np.testing.assert_allclose(frame["r1:b"].to_numpy(), frame["r2:b"].to_numpy())
    assert list(frame.columns[:4]) == ["row_id", "scheme", "fold", "actual"]


@pytest.mark.parametrize("break_it", ["actual", "fold", "drop_row"])
def test_load_runs_fails_on_mismatched_oof_frames(emerging_root, break_it):
    rng = np.random.default_rng(0)
    oof = _oof(rng)
    other = oof.copy()
    if break_it == "actual":
        other.loc[5, "actual"] += 1.0
    elif break_it == "fold":
        grouped = other["scheme"] == "group"
        other.loc[grouped, "fold"] = (other.loc[grouped, "fold"] + 1) % 5
    else:
        other = other.drop(index=7)
    _write_oof(emerging_root, "r1", "motor", oof)
    _write_oof(emerging_root, "r2", "motor", other)
    with pytest.raises(AssertionError, match="align"):
        blend_runs.load_runs("motor", ["r1", "r2"])


def test_evaluate_line_searches_on_grouped_folds_and_applies_guard_and_adoption_rule(emerging_root):
    rng = np.random.default_rng(3)
    _write_oof(emerging_root, "r1", "motor", _oof(rng, models=("a", "b")))
    frame, names = blend_runs.load_runs("motor", ["r1"])
    second = frame[["r1:b"]].rename(columns={"r1:b": "r2:c"}) * 1.3  # a second run with a worse model
    frame = pd.concat([frame, second], axis=1)
    names = names + ["r2:c"]
    folds = blend_runs.scheme_folds(frame, names)
    assert [len(folds["time"]), len(folds["group"])] == [1, 5]

    res = blend_runs.evaluate_line(folds, names, {"r1:b": 1.0}, 0.1, 0.77)
    w = res["weights"]
    assert set(w) == set(names) and sum(w.values()) == pytest.approx(1.0)
    assert max(w, key=w.get) == "r1:a"  # the truth model carries the blend
    assert res["rule"] == "searched weights beat equal weights on all folds"
    gains = res["gain"]
    assert gains["time"] == pytest.approx(res["blend"]["time"] - res["baseline"]["time"])
    assert gains["group_mean"] == pytest.approx(res["blend"]["group_mean"] - res["baseline"]["group_mean"])
    assert gains["mean"] == pytest.approx((gains["time"] + gains["group_mean"]) / 2)
    assert res["adoption"]["passes"] is True and gains["time"] > 0.002
    assert res["concentration"]["n_nonzero"] == sum(v > 0 for v in w.values())
    # same baseline as the searched blend -> zero gain -> not adopted
    same = blend_runs.evaluate_line(folds, names, res["weights"], 0.1, 0.77)
    assert same["gain"]["mean"] == pytest.approx(0.0, abs=1e-12) and same["adoption"]["passes"] is False


def test_evaluate_line_requires_baseline_candidates_to_be_loaded():
    folds = {"time": [{"actual": np.ones(3), "preds": {"r1:a": np.ones(3)}}],
             "group": [{"actual": np.ones(3), "preds": {"r1:a": np.ones(3)}}]}
    with pytest.raises(AssertionError, match="baseline"):
        blend_runs.evaluate_line(folds, ["r1:a"], {"r9:a": 1.0}, 0.5, 0.77)


def _write_pred(base, run, line, model, ids, values, trend_factor):
    os.makedirs(base / run, exist_ok=True)
    pd.DataFrame({"row_id": ids, "prediction": values, "trend": "x", "trend_factor": trend_factor}).to_parquet(
        base / run / f"pred_{line}_{model}.parquet", index=False)


def test_submission_mixes_runs_and_reexpresses_each_file_at_the_requested_trend(emerging_root, monkeypatch):
    sample = pd.DataFrame({"row_id": ["r1", "r2", "r3", "r4", "r5", "r6"], "prediction": 0.0})
    monkeypatch.setattr(blend, "load", lambda name, market=None: sample)
    monkeypatch.setattr(blend, "EMERGING_ROWS", len(sample))
    ids = {"motor": ["r5", "r1"], "buildings": ["r3", "r2"], "contents": ["r6", "r4"]}
    _write_pred(emerging_root, "w03", "motor", "glm", ids["motor"], [10.0, 20.0], 1.0)
    _write_pred(emerging_root, "w1", "motor", "cat", ids["motor"], [30.0, 40.0], 2.0)
    _write_pred(emerging_root, "w1", "buildings", "cat", ids["buildings"], [100.0, 200.0], 1.0)
    _write_pred(emerging_root, "w03", "contents", "lgbm", ids["contents"], [1.0, 2.0], 1.0)
    weights = {"motor": {"w03:glm": 0.4, "w1:cat": 0.6, "w1:xgb": 0.0},  # zero weight: no file needed
               "buildings": {"w1:cat": 1.0}, "contents": {"w03:lgbm": 1.0}}
    sub = blend_runs.write_submission({ln: blend_runs.line_prediction(ln, w, "developed-half")
                                       for ln, w in weights.items()}, "xrun_test")
    f = markets.get("emerging").trends["developed-half"]
    expected = {"r5": (0.4 * 10 + 0.6 * 30 / 2.0) * f["motor"], "r1": (0.4 * 20 + 0.6 * 40 / 2.0) * f["motor"],
                "r3": 100 * f["buildings"], "r2": 200 * f["buildings"], "r6": 1 * f["contents"], "r4": 2 * f["contents"]}
    work = emerging_root.parent.parent / "work"
    got = pd.read_csv(work / "xrun_test.csv")
    assert list(got["row_id"]) == list(sample["row_id"])
    for row_id, value in zip(got["row_id"], got["prediction"]):
        assert value == pytest.approx(expected[row_id])
    assert (work / "xrun_test.zip").exists()
    assert sub.index.tolist() == sample["row_id"].tolist()


def test_line_prediction_fails_when_a_positive_weight_candidate_has_no_pred_file(emerging_root):
    _write_pred(emerging_root, "w03", "motor", "glm", ["r1"], [1.0], 1.0)
    with pytest.raises(AssertionError, match="w1:cat"):
        blend_runs.line_prediction("motor", {"w03:glm": 0.5, "w1:cat": 0.5}, "none")


def test_main_writes_one_json_per_line_with_the_reported_keys(emerging_root):
    rng = np.random.default_rng(5)
    _write_oof(emerging_root, "r1", "motor", _oof(rng, n=200))
    blend_runs.main(["motor"], ["r1"], "none", "unused", 0.25, "xrun_t", write=False,
                    baseline={"motor": {"r1:b": 1.0}})
    with open(emerging_root / "xrun_t" / "blend_weights_motor.json", encoding="utf-8") as f:
        res = json.load(f)
    for key in ("weights", "weights_searched", "rule", "guard_scores", "baseline", "blend", "gain", "adoption",
                "concentration", "single", "submission_weights", "runs", "candidates", "step", "trend"):
        assert key in res
    assert res["submission_weights"] == (res["weights"] if res["adoption"]["passes"] else {"r1:b": 1.0})


def test_load_runs_skips_a_run_folder_without_this_lines_oof_but_not_a_missing_folder(emerging_root):
    rng = np.random.default_rng(0)
    _write_oof(emerging_root, "r1", "motor", _oof(rng))
    os.makedirs(emerging_root / "r2_glmx")  # holds another line only
    _write_oof(emerging_root, "r2_glmx", "contents", _oof(rng))
    frame, names = blend_runs.load_runs("motor", ["r1", "r2_glmx"])
    assert names == ["r1:a", "r1:b"]
    with pytest.raises(AssertionError, match="no run folder"):
        blend_runs.load_runs("motor", ["r1", "typo_run"])
