"""C047: segment-aware (new business / renewal) blend search, gate and assembly in blend_runs.py; P2-S12 baseline."""
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
from aiiaa import markets  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def emerging_root(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend_runs, "nb_share_2025", lambda line: 0.77)
    return tmp_path / "output" / "emerging"


def _folds(rng, n=3000):
    """Synthetic folds (1 time + 5 grouped): r:base is exact on renewal rows and noisy on NB rows, r:nbgood the
    reverse; NB rows are 30% of the rows and riskier."""
    out = {}
    for scheme, ks in (("time", [0]), ("group", range(5))):
        out[scheme] = []
        for _ in ks:
            nb = rng.uniform(size=n) < 0.3
            mu = rng.gamma(2.0, 50.0, n) * np.where(nb, 2.0, 1.0)
            y = np.where(rng.uniform(size=n) < 0.3, rng.gamma(1.5, mu / 0.45), 0.0)
            noise = np.exp(rng.normal(0, 0.8, n))
            out[scheme].append({"actual": y, "new_business": nb, "exposure": np.ones(n),
                                "claim_count": (y > 0).astype(int),
                                "preds": {"r:base": np.where(nb, mu * noise, mu),
                                          "r:nbgood": np.where(nb, mu, mu * noise)}})
    return out


def test_segment_split_covers_every_row_exactly_once():
    folds = _folds(np.random.default_rng(0), n=500)
    nb_f, ren_f = blend_runs.segment_folds(folds, "nb"), blend_runs.segment_folds(folds, "renewal")
    for scheme in blend_runs.SCHEMES:
        for f, a, b in zip(folds[scheme], nb_f[scheme], ren_f[scheme]):
            assert len(a["actual"]) + len(b["actual"]) == len(f["actual"])
            assert a["new_business"].all() and not b["new_business"].any()
            assert sorted(np.concatenate([a["actual"], b["actual"]])) == sorted(f["actual"])
            for key in ("exposure", "claim_count"):
                assert len(a[key]) == len(a["actual"]) and len(b[key]) == len(b["actual"])
            np.testing.assert_array_equal(a["preds"]["r:base"], f["preds"]["r:base"][f["new_business"]])


def test_fold_prediction_takes_the_segment_blend_per_row():
    f = {"new_business": np.array([True, False, True]),
         "preds": {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 20.0, 30.0])}}
    p = blend_runs.fold_pred(f, {"nb": {"a": 1.0}, "renewal": {"a": 0.5, "b": 0.5}})
    np.testing.assert_allclose(p, [1.0, 11.0, 3.0])
    np.testing.assert_allclose(blend_runs.fold_pred(f, {"a": 0.5, "b": 0.5}), [5.5, 11.0, 16.5])  # flat unchanged


def test_segment_gate_adopts_the_nb_segment_and_keeps_the_renewal_baseline():
    folds = _folds(np.random.default_rng(1))
    res = blend_runs.evaluate_segments(folds, ["r:base", "r:nbgood"], {"r:base": 1.0}, 0.1, 0.77, "testmix")
    nb, ren = res["segments"]["nb"], res["segments"]["renewal"]
    assert nb["weights"]["r:nbgood"] >= 0.8 and nb["adoption"]["passes"]
    assert nb["gain"]["group_testmix"] > 0.002 and nb["gain"]["group_nb"] > 0
    assert ren["weights"] == {"r:base": 1.0, "r:nbgood": 0.0} and not ren["adoption"]["passes"]
    assert res["submission_weights"] == {"nb": nb["weights"], "renewal": {"r:base": 1.0}}
    # the final view is the assembled prediction scored on the whole line
    final = blend_runs.blend_scores(folds, res["submission_weights"], 0.77)
    assert res["final"]["group_testmix"] == pytest.approx(final["group_testmix"])
    # per-segment model table: within-segment d2_cal of every candidate
    assert nb["single"]["r:nbgood"]["group_mean"] > nb["single"]["r:base"]["group_mean"]
    assert ren["single"]["r:base"]["group_mean"] > ren["single"]["r:nbgood"]["group_mean"]


@pytest.mark.parametrize("segment, nb_gain, passes", [("nb", 0.001, True), ("nb", -0.001, False),
                                                      ("renewal", -0.001, True)])
def test_nb_segment_needs_the_nb_view_to_agree_with_the_test_mix_view(segment, nb_gain, passes):
    gain = {"time": 0.0, "group_mean": 0.0, "group_testmix": 0.003, "group_nb": nb_gain, "mean": 0.0}
    assert blend_runs.adopt_segment(gain, "testmix", segment) is passes


def test_segment_line_prediction_uses_the_nb_blend_for_nb_test_rows_only(emerging_root, monkeypatch):
    for model, values in (("base", [1.0, 2.0, 3.0]), ("nbgood", [10.0, 20.0, 30.0])):
        os.makedirs(emerging_root / "r", exist_ok=True)
        pd.DataFrame({"row_id": ["t3", "t1", "t2"], "prediction": values, "trend": "none", "trend_factor": 1.0}
                     ).to_parquet(emerging_root / "r" / f"pred_motor_{model}.parquet", index=False)
    monkeypatch.setattr(blend_runs, "nb_flags_2025",
                        lambda line: pd.Series([True, False, True], index=["t1", "t2", "t3"]))
    p = blend_runs.line_prediction("motor", {"nb": {"r:nbgood": 1.0}, "renewal": {"r:base": 1.0}}, "none")
    assert p.to_dict() == {"t3": 10.0, "t1": 20.0, "t2": 3.0}


def test_main_with_segments_writes_the_per_segment_result_and_write_from_assembles_it(emerging_root, monkeypatch):
    rng = np.random.default_rng(3)
    folds = _folds(rng, n=400)
    rows = []
    for scheme in blend_runs.SCHEMES:
        for k, f in enumerate(folds[scheme]):
            rows.append(pd.DataFrame({"row_id": [f"{scheme}{k}_{i}" for i in range(len(f["actual"]))],
                                      "scheme": scheme, "fold": k, "actual": f["actual"],
                                      "new_business": f["new_business"], "exposure": 1.0,
                                      "claim_count": f["claim_count"],
                                      "base": f["preds"]["r:base"], "nbgood": f["preds"]["r:nbgood"]}))
    os.makedirs(emerging_root / "r", exist_ok=True)
    pd.concat(rows).to_parquet(emerging_root / "r" / "oof_motor.parquet", index=False)
    blend_runs.main(["motor"], ["r"], "none", "unused", 0.1, "seg_t", write=False,
                    baseline={"motor": {"r:base": 1.0}}, gate="testmix", segment="new_business")
    with open(emerging_root / "seg_t" / "blend_weights_motor.json", encoding="utf-8") as fh:
        res = json.load(fh)
    assert set(res["segments"]) == {"nb", "renewal"} and res["segment"] == "new_business"
    assert set(res["submission_weights"]) == {"nb", "renewal"}
    for model, values in (("base", [1.0, 2.0]), ("nbgood", [10.0, 20.0])):
        pd.DataFrame({"row_id": ["t1", "t2"], "prediction": values, "trend": "none", "trend_factor": 1.0}
                     ).to_parquet(emerging_root / "r" / f"pred_motor_{model}.parquet", index=False)
    monkeypatch.setattr(blend_runs, "nb_flags_2025", lambda line: pd.Series([True, False], index=["t1", "t2"]))
    p = blend_runs.line_prediction("motor", res["submission_weights"], "none")
    w_nb, w_ren = res["submission_weights"]["nb"], res["submission_weights"]["renewal"]
    assert p["t1"] == pytest.approx(w_nb.get("r:base", 0) * 1.0 + w_nb.get("r:nbgood", 0) * 10.0)
    assert p["t2"] == pytest.approx(w_ren.get("r:base", 0) * 2.0 + w_ren.get("r:nbgood", 0) * 20.0)


def test_p2s12_baseline_differs_from_p2s05_in_contents_only():
    b = blend_runs.BASELINES
    assert b["P2-S05"] is blend_runs.BASELINE
    assert b["P2-S12"]["contents"] == {"pooled_w0.3:glm": 0.35, "pooled_w1:lgbm": 0.65}
    assert b["P2-S12"]["motor"] == blend_runs.BASELINE["motor"]
    assert b["P2-S12"]["buildings"] == blend_runs.BASELINE["buildings"]


def test_cli_lists_segment_and_baseline():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "blend_runs.py"), "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--segment" in r.stdout and "--baseline" in r.stdout
