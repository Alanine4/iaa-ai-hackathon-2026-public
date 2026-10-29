"""C058: `blend_runs.py --write-from ... --level-from <submission file>:<line>` re-scales listed lines to the total of
the same rows in a submission file (the C054 convention: an adopted line at the confirmed line level)."""
import json
import os
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import blend  # noqa: E402
import blend_runs  # noqa: E402
from aiiaa import markets  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED = {"motor": [1.0, 3.0], "buildings": [10.0, 30.0], "contents": [5.0, 15.0]}  # candidate r:lgbm, two rows per line
REF = {"motor": [100.0, 100.0], "buildings": [60.0, 140.0], "contents": [7.0, 9.0]}  # reference file values


@pytest.fixture
def emerging_root(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    return tmp_path / "output" / "emerging"


@pytest.fixture
def spec(emerging_root):
    """One result folder per line, each with a single candidate r:lgbm at weight 1 and its pred file (rows <line>1, <line>2)."""
    os.makedirs(emerging_root / "r", exist_ok=True)
    spec = {}
    for line, values in PRED.items():
        pd.DataFrame({"row_id": [f"{line}1", f"{line}2"], "prediction": values, "trend": "none", "trend_factor": 1.0}
                     ).to_parquet(emerging_root / "r" / f"pred_{line}_lgbm.parquet", index=False)
        os.makedirs(emerging_root / f"res_{line}", exist_ok=True)
        (emerging_root / f"res_{line}" / f"blend_weights_{line}.json").write_text(
            json.dumps({"submission_weights": {"r:lgbm": 1.0}}), encoding="utf-8")
        spec[line] = f"res_{line}"
    return spec


def _reference(tmp_path, kind, drop=None):
    """Submission file with the rows of all three lines (row_id text, prediction REF); `drop` removes one row."""
    frame = pd.DataFrame({"row_id": [f"{line}{k}" for line in REF for k in (1, 2)],
                          "prediction": [v for values in REF.values() for v in values]})
    if drop:
        frame = frame[frame["row_id"] != drop]
    if kind == "csv":
        path = tmp_path / "ref.csv"
        frame.to_csv(path, index=False)
    else:
        path = tmp_path / "ref.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("ref.csv", frame.to_csv(index=False))
    return str(path)


def _write(spec, level_from, monkeypatch):
    captured = {}
    monkeypatch.setattr(blend_runs, "write_submission", lambda by_line, name: captured.update(by_line=by_line))
    blend_runs.write_from(spec, "none", "unused", level_from=level_from)
    return {line: p.to_dict() for line, p in captured["by_line"].items()}


@pytest.mark.parametrize("kind", ["zip", "csv"])
def test_listed_line_is_rescaled_to_the_file_total_and_the_others_are_unchanged(spec, tmp_path, monkeypatch, capsys, kind):
    out = _write(spec, (_reference(tmp_path, kind), ["buildings"]), monkeypatch)
    assert out["buildings"] == pytest.approx({"buildings1": 10.0 * 5.0, "buildings2": 30.0 * 5.0})  # 40 -> 200, factor 5
    assert out["motor"] == {"motor1": 1.0, "motor2": 3.0} and out["contents"] == {"contents1": 5.0, "contents2": 15.0}
    printed = capsys.readouterr().out
    assert "buildings: level from" in printed and "factor 5.0000, mean 20.00 -> 100.00" in printed
    assert "motor: level from" not in printed and "contents: level from" not in printed


def test_several_lines_are_each_rescaled_to_their_own_file_total(spec, tmp_path, monkeypatch):
    out = _write(spec, (_reference(tmp_path, "zip"), ["motor", "contents"]), monkeypatch)
    assert sum(out["motor"].values()) == pytest.approx(200.0) and sum(out["contents"].values()) == pytest.approx(16.0)
    assert out["contents"]["contents2"] / out["contents"]["contents1"] == pytest.approx(3.0)  # the shape is kept
    assert out["buildings"] == {"buildings1": 10.0, "buildings2": 30.0}


def test_without_level_from_nothing_is_rescaled(spec, monkeypatch):
    out = _write(spec, None, monkeypatch)
    assert out == {line: {f"{line}1": v[0], f"{line}2": v[1]} for line, v in PRED.items()}


def test_a_prediction_row_missing_in_the_file_raises_and_names_it(spec, tmp_path, monkeypatch):
    path = _reference(tmp_path, "csv", drop="buildings2")
    with pytest.raises(AssertionError, match=r"buildings: 1 of 2 row_ids are not in .*ref\.csv.*buildings2"):
        _write(spec, (path, ["buildings"]), monkeypatch)
    _write(spec, (path, ["motor"]), monkeypatch)  # the other lines still work from the same file


def test_parse_level_from_splits_at_the_last_colon_and_rejects_bad_input():
    assert blend_runs.parse_level_from(r"submissions\p.zip:buildings") == (r"submissions\p.zip", ["buildings"])
    assert blend_runs.parse_level_from(r"C:\x\p.zip:motor,contents") == (r"C:\x\p.zip", ["motor", "contents"])
    for bad in ("p.zip", "p.zip:", "p.zip:bikes", "p.zip:motor,motor", ":motor", r"C:\x\p.zip"):
        with pytest.raises(ValueError, match="--level-from"):
            blend_runs.parse_level_from(bad)


def _cli(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "blend_runs.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_rejects_level_from_without_write_from_and_bad_values():
    r = _cli("motor,buildings,contents", "--level-from", "p.zip:buildings")
    assert r.returncode == 2 and "C058" in r.stderr and "--write-from" in r.stderr
    r = _cli("motor,buildings,contents", "--write-from", "motor=a,buildings=b,contents=c", "--level-from", "p.zip:bikes")
    assert r.returncode == 2 and "--level-from needs" in r.stderr
    assert "--level-from" in _cli("--help").stdout
