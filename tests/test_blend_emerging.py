import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import blend  # noqa: E402
from aiiaa import markets  # noqa: E402

PER_LINE = {"motor": 0.3, "buildings": 1.0, "contents": 0.3}


def test_emerging_market_carries_the_adopted_developed_weights_per_line():
    # C040
    assert markets.get("emerging").developed_weights == PER_LINE
    assert markets.get("developed").developed_weights == {}


def test_parse_dev_weight_single_number_applies_to_every_line():
    assert blend.parse_dev_weight("0.3", markets.get("emerging")) == 0.3
    assert blend.parse_dev_weight("1", markets.get("emerging")) == 1.0


def test_parse_dev_weight_per_line_reads_the_market_table():
    assert blend.parse_dev_weight("per-line", markets.get("emerging")) == PER_LINE


def test_parse_dev_weight_explicit_pairs():
    got = blend.parse_dev_weight("motor=0.3,buildings=1.0,contents=0.5", markets.get("emerging"))
    assert got == {"motor": 0.3, "buildings": 1.0, "contents": 0.5}


@pytest.mark.parametrize("text", ["motor=0.3,buildings=1.0", "motor=0.3,buildings=1,contents=1,extra=1", "abc",
                                  "motor=x,buildings=1,contents=1"])
def test_parse_dev_weight_rejects_incomplete_or_malformed_input(text):
    with pytest.raises(ValueError):
        blend.parse_dev_weight(text, markets.get("emerging"))


def _write_run(folder, line, weights, preds, trend_factor):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": weights}, f)
    for model, frame in preds.items():
        d = frame.copy()
        d["trend"] = 1.0
        d["trend_factor"] = trend_factor
        d.to_parquet(os.path.join(folder, f"pred_{line}_{model}.parquet"), index=False)


@pytest.fixture
def synthetic_runs(tmp_path, monkeypatch):
    """Two pooled run folders: w0.3 (motor, contents) and w1 (buildings); a zero-weight model has no pred file."""
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    base = tmp_path / "output" / "emerging"
    ids = {"motor": ["r5", "r1"], "buildings": ["r3", "r2"], "contents": ["r6", "r4"]}
    sample = pd.DataFrame({"row_id": ["r1", "r2", "r3", "r4", "r5", "r6"], "prediction": 0.0})
    monkeypatch.setattr(blend, "load", lambda name, market=None: sample)
    monkeypatch.setattr(blend, "EMERGING_ROWS", len(sample))

    def preds(line, a, b):
        return {"lgbm": pd.DataFrame({"row_id": ids[line], "prediction": a}),
                "xgb": pd.DataFrame({"row_id": ids[line], "prediction": b})}

    # trend_factor 1.0 in every file except buildings (2.0), so the division by the file's factor is exercised
    _write_run(str(base / "pooled_w0.3"), "motor", {"glm": 0.0, "lgbm": 0.25, "xgb": 0.75}, preds("motor", [10.0, 20.0], [30.0, 40.0]), 1.0)
    _write_run(str(base / "pooled_w0.3"), "contents", {"glm": 0.0, "lgbm": 0.5, "xgb": 0.5}, preds("contents", [1.0, 2.0], [3.0, 5.0]), 1.0)
    _write_run(str(base / "pooled_w1"), "buildings", {"glm": 0.0, "lgbm": 0.6, "xgb": 0.4}, preds("buildings", [100.0, 200.0], [300.0, 400.0]), 2.0)
    # a second, different run for buildings at 0.3 proves the dict picks the w1 folder, not this one
    _write_run(str(base / "pooled_w0.3"), "buildings", {"glm": 0.0, "lgbm": 1.0, "xgb": 0.0}, preds("buildings", [-1.0, -1.0], [-1.0, -1.0]), 2.0)
    return tmp_path


def test_emerging_blend_reads_each_line_from_its_own_run_folder(synthetic_runs):
    # C040: per-line developed weight -> per-line run folder; zero-weight model without a pred file is skipped
    blend.emerging_blend(["glm", "lgbm", "xgb"], "synthetic", True, "pooled", PER_LINE, "developed-half")
    got = pd.read_csv(synthetic_runs / "work" / "synthetic.csv")
    f = markets.get("emerging").trends["developed-half"]
    assert list(got["row_id"]) == ["r1", "r2", "r3", "r4", "r5", "r6"]
    expected = {
        "r5": (0.25 * 10 + 0.75 * 30) * f["motor"], "r1": (0.25 * 20 + 0.75 * 40) * f["motor"],
        "r3": (0.6 * 100 + 0.4 * 300) * f["buildings"] / 2.0, "r2": (0.6 * 200 + 0.4 * 400) * f["buildings"] / 2.0,
        "r6": (0.5 * 1 + 0.5 * 3) * f["contents"], "r4": (0.5 * 2 + 0.5 * 5) * f["contents"],
    }
    assert got["prediction"].notna().all()
    for row_id, value in zip(got["row_id"], got["prediction"]):
        assert value == pytest.approx(expected[row_id])
    assert (synthetic_runs / "work" / "synthetic.zip").exists()


def test_emerging_blend_single_number_uses_one_folder_for_all_lines(synthetic_runs):
    # the buildings pred files of pooled_w0.3 have negative predictions -> the non-negative check must fire
    with pytest.raises(AssertionError):
        blend.emerging_blend(["glm", "lgbm", "xgb"], "synthetic", True, "pooled", 0.3, "none")


def test_emerging_blend_fails_when_a_positive_weight_model_has_no_pred_file(synthetic_runs):
    path = synthetic_runs / "output" / "emerging" / "pooled_w0.3" / "pred_motor_xgb.parquet"
    path.unlink()
    with pytest.raises(AssertionError, match="xgb"):
        blend.emerging_blend(["glm", "lgbm", "xgb"], "synthetic", True, "pooled", PER_LINE, "none")
