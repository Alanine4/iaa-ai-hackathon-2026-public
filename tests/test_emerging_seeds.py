"""C057: seed-averaged refit for the emerging market (`--refit-only --seeds N`) and `blend_runs.py --pred-tag`."""
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
import oof_predictions  # noqa: E402
from aiiaa import markets, models  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LINE = "contents"
NEW_KEYS = {"refit_seeds", "refit_pred_tag", "refit_rebalance_factor_seeds", "refit_only_at"}


def _design(rng):
    """Synthetic pooled design (developed, emerging and test rows); the claim rate rises with `v` so trees can split."""
    n_dev, n_em, n_test = 3000, 1500, 200
    n = n_dev + n_em + n_test
    part = np.array(["dev"] * n_dev + ["em"] * n_em + ["test"] * n_test, dtype=object)
    nb = rng.uniform(size=n) < np.where(part == "test", 0.77, 0.25)
    v = rng.uniform(0, 1, n)
    x = pd.DataFrame({"market": pd.Categorical(np.where(part == "dev", "developed", "emerging")),
                      "new_business": nb, "v": v})
    y = np.where(rng.uniform(size=n) < 0.1 + 0.4 * v, rng.gamma(1.0, 500.0, n), 0.0) * (part != "test")
    w = np.where(part == "test", 0.0, np.where(part == "dev", 0.3, 1.0))
    year = np.where(np.arange(n_em) < n_em // 2, 2023, 2024)
    train = pd.DataFrame({"policy_id": [f"P{i % 900:04d}" for i in range(n_em)]})
    t = pd.DataFrame({"row_id": [f"E{i:05d}" for i in range(n_em)], "claim_cost": y[part == "em"],
                      "claim_cost_capped": y[part == "em"], "claim_count": (y[part == "em"] > 0).astype(int),
                      "exposure": 1.0, "completeness": 1.0})
    test = pd.DataFrame({"row_id": [f"T{i:05d}" for i in range(n_test)], "exposure": 1.0})
    return x, part, y, w, train, test, t, year


@pytest.fixture
def pooled(tmp_path, monkeypatch):
    """Run folder of a pooled synthetic run; the default LightGBM (3000 rows per leaf) cannot split a design this small,
    so every seed would give the same constant -- the models get 20 rows per leaf instead."""
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    design = _design(np.random.default_rng(11))
    monkeypatch.setattr(oof_predictions, "emerging_design", lambda *a, **k: design)
    real = oof_predictions.emerging_model
    monkeypatch.setattr(oof_predictions, "emerging_model", lambda line, name, train_set, extra=None: real(
        line, name, train_set, {"min_child_samples": 20, **(extra or {})} if name == "lgbm" else extra))
    return tmp_path / "output" / "emerging" / "pooled_w0.3"


def _run(names, **kwargs):
    oof_predictions.emerging_main(LINE, names, "pooled", 0.3, "none", **kwargs)


def _pred(folder, name, suffix=""):
    return pd.read_parquet(folder / f"pred_{LINE}_{name}{suffix}.parquet")


def _json(folder):
    with open(folder / f"blend_weights_{LINE}.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_refit_only_seeds_writes_cache_and_tagged_files_and_never_touches_the_untagged_run(pooled):
    seed = models.SEED
    _run(["glm", "lgbm"])
    assert list(_pred(pooled, "lgbm").columns) == ["row_id", "prediction", "trend", "trend_factor"]
    assert [f for f in os.listdir(pooled) if "_seed" in f] == []  # a plain run writes today's files only
    before = {n: (pooled / f"pred_{LINE}_{n}.parquet").read_bytes() for n in ("glm", "lgbm")}
    json_before = _json(pooled)

    _run(["lgbm"], refit_only=True, seeds=2)

    for n in ("glm", "lgbm"):  # the untagged files are byte-identical (nothing rewritten)
        assert (pooled / f"pred_{LINE}_{n}.parquet").read_bytes() == before[n]
    untagged = _pred(pooled, "lgbm")
    s0, s1 = _pred(pooled, "lgbm", f"_seed{seed}"), _pred(pooled, "lgbm", f"_seed{seed + 1}")
    assert list(s0.columns) == ["row_id", "prediction", "trend", "trend_factor", "seed", "rebalance_factor"]
    assert (s0["seed"] == seed).all() and (s1["seed"] == seed + 1).all()
    # the default seed reproduces today's untagged prediction; the next seed is a different model
    assert (s0["row_id"] == untagged["row_id"]).all()
    assert np.allclose(s0["prediction"], untagged["prediction"], atol=1e-9)
    assert not np.allclose(s1["prediction"], untagged["prediction"], atol=1e-9)
    avg = _pred(pooled, "lgbm", "_s2")
    assert list(avg.columns) == ["row_id", "prediction", "trend", "trend_factor", "seeds"] and (avg["seeds"] == 2).all()
    assert (avg["row_id"] == untagged["row_id"]).all()
    np.testing.assert_allclose(avg["prediction"], (s0["prediction"] + s1["prediction"]) / 2)
    assert not (pooled / f"pred_{LINE}_glm_seed{seed}.parquet").exists()
    assert not (pooled / f"pred_{LINE}_glm_s2.parquet").exists()

    after = _json(pooled)
    for key, value in json_before.items():  # every pre-existing key (weights, cv_time, cv_group, ...) is unchanged
        assert after[key] == value, key
    assert set(after) - set(json_before) == NEW_KEYS
    assert after["refit_seeds"] == 2 and after["refit_pred_tag"] == "s2"
    factors = after["refit_rebalance_factor_seeds"]["lgbm"]
    assert set(factors) == {str(seed), str(seed + 1)} and factors[str(seed)] != factors[str(seed + 1)]
    assert factors[str(seed)] == pytest.approx(json_before["refit_rebalance_factor"]["lgbm"])
    assert (s0["rebalance_factor"] == factors[str(seed)]).all()
    assert isinstance(after["refit_only_at"], str) and after["refit_only_at"]


def test_refit_only_fits_only_the_seeds_that_are_not_cached(pooled, monkeypatch):
    seed = models.SEED
    _run(["lgbm"])
    _run(["lgbm"], refit_only=True, seeds=2)
    files = {f: (pooled / f).read_bytes() for f in os.listdir(pooled) if f.startswith(f"pred_{LINE}_lgbm_")}
    assert set(files) == {f"pred_{LINE}_lgbm_seed{seed}.parquet", f"pred_{LINE}_lgbm_seed{seed + 1}.parquet",
                          f"pred_{LINE}_lgbm_s2.parquet"}

    calls = []
    real = oof_predictions.fit_emerging

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(oof_predictions, "fit_emerging", spy)
    _run(["lgbm"], refit_only=True, seeds=2)  # everything is cached: nothing is fitted, nothing changes
    assert calls == []
    assert {f: (pooled / f).read_bytes() for f in files} == files

    _run(["lgbm"], refit_only=True, seeds=3)  # one new seed
    assert len(calls) == 1
    s = [_pred(pooled, "lgbm", f"_seed{seed + k}")["prediction"] for k in range(3)]
    np.testing.assert_allclose(_pred(pooled, "lgbm", "_s3")["prediction"], (s[0] + s[1] + s[2]) / 3)
    assert (pooled / f"pred_{LINE}_lgbm_s2.parquet").read_bytes() == files[f"pred_{LINE}_lgbm_s2.parquet"]
    res = _json(pooled)
    assert res["refit_seeds"] == 3 and res["refit_pred_tag"] == "s3"
    assert set(res["refit_rebalance_factor_seeds"]["lgbm"]) == {str(seed + k) for k in range(3)}


def test_refit_only_with_one_seed_writes_the_cache_file_only(pooled):
    seed = models.SEED
    _run(["glm", "lgbm"])
    before = {n: (pooled / f"pred_{LINE}_{n}.parquet").read_bytes() for n in ("glm", "lgbm")}
    _run(["glm", "lgbm"], refit_only=True, seeds=1)
    for n in ("glm", "lgbm"):
        assert (pooled / f"pred_{LINE}_{n}.parquet").read_bytes() == before[n]
        # the default seed reproduces the untagged file (the GLM is deterministic, so it is fit once)
        assert np.allclose(_pred(pooled, n, f"_seed{seed}")["prediction"], _pred(pooled, n)["prediction"], atol=1e-9)
    assert [f for f in os.listdir(pooled) if f.endswith("_s1.parquet")] == []
    res = _json(pooled)
    assert res["refit_seeds"] == 1 and res["refit_pred_tag"] is None
    assert set(res["refit_rebalance_factor_seeds"]) == {"glm", "lgbm"}


def test_deterministic_models_are_fit_once_whatever_the_seed_count(pooled):
    seed = models.SEED
    _run(["glm"])
    glm_before = (pooled / f"pred_{LINE}_glm.parquet").read_bytes()
    _run(["glm"], refit_only=True, seeds=3)
    assert (pooled / f"pred_{LINE}_glm.parquet").read_bytes() == glm_before
    assert sorted(f for f in os.listdir(pooled) if f.startswith(f"pred_{LINE}_glm_")) == [
        f"pred_{LINE}_glm_seed{seed}.parquet"]
    assert list(_json(pooled)["refit_rebalance_factor_seeds"]["glm"]) == [str(seed)]


def test_refit_only_without_a_full_run_fails_before_building_anything(pooled):
    with pytest.raises(FileNotFoundError, match=r"blend_weights_contents\.json"):
        _run(["lgbm"], refit_only=True, seeds=2)
    assert not pooled.exists()


def test_refit_only_needs_every_requested_model_in_the_stored_run(pooled):
    _run(["lgbm"])
    with pytest.raises(KeyError, match="xgb"):
        _run(["lgbm", "xgb"], refit_only=True, seeds=2)


def test_refit_only_rejects_a_nb_weight_that_differs_from_the_stored_run(pooled):
    _run(["lgbm"])
    path = pooled / f"blend_weights_{LINE}.json"
    res = _json(pooled)
    res["nb_weight"]["option"] = "auto"  # as if the stored run had been made with --nb-weight auto
    path.write_text(json.dumps(res), encoding="utf-8")
    with pytest.raises(AssertionError, match="nb-weight"):
        _run(["lgbm"], refit_only=True, seeds=2)


def test_refit_only_takes_the_trend_from_the_stored_run(pooled):
    _run(["lgbm"])
    res = _json(pooled)
    assert res["trend"] == "none"
    oof_predictions.emerging_main(LINE, ["lgbm"], "pooled", 0.3, "developed-half", refit_only=True, seeds=2)
    assert _json(pooled)["trend"] == "none"
    assert (_pred(pooled, "lgbm", "_s2")["trend"] == "none").all()


def test_several_seeds_without_refit_only_are_rejected(pooled):
    with pytest.raises(AssertionError, match="C057"):
        _run(["lgbm"], seeds=2)


def _cli(script, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", script), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_emerging_seeds_need_refit_only():
    r = _cli("oof_predictions.py", LINE, "--market", "emerging", "--seeds", "2")
    assert r.returncode == 2 and "C057" in r.stderr and "--refit-only" in r.stderr


def test_cli_accepts_refit_only_with_seeds_for_the_emerging_market_and_still_rejects_te():
    # the parser accepts the combination: the run then stops at the missing blend_weights json of the (unused) tag
    r = _cli("oof_predictions.py", LINE, "--market", "emerging", "--refit-only", "--seeds", "2", "--tag", "c048_none")
    assert r.returncode != 0 and "FileNotFoundError" in r.stderr and "usage:" not in r.stderr
    assert not os.path.exists(os.path.join(ROOT, "output", "emerging", "emerging_c048_none"))
    r = _cli("oof_predictions.py", LINE, "--market", "emerging", "--refit-only", "--te", "make_model")
    assert r.returncode != 0 and "C036: --te is developed-only" in r.stderr


@pytest.fixture
def emerging_root(tmp_path, monkeypatch):
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    monkeypatch.setattr(blend, "ROOT", str(tmp_path))
    return tmp_path / "output" / "emerging"


def _pred_file(folder, line, model, values, tag=None):
    os.makedirs(folder, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    pd.DataFrame({"row_id": ["t1", "t2"], "prediction": values, "trend": "none", "trend_factor": 1.0}
                 ).to_parquet(folder / f"pred_{line}_{model}{suffix}.parquet", index=False)


def test_line_prediction_reads_tagged_files_and_falls_back_to_untagged_for_deterministic_models(emerging_root):
    r = emerging_root / "r"
    _pred_file(r, "motor", "lgbm", [1.0, 2.0])
    _pred_file(r, "motor", "lgbm", [3.0, 4.0], tag="s2")
    _pred_file(r, "motor", "glm", [10.0, 20.0])  # no tagged glm file: the GLM is not re-fit per seed
    weights = {"r:lgbm": 0.5, "r:glm": 0.5}
    assert blend_runs.line_prediction("motor", weights, "none").to_dict() == {"t1": 5.5, "t2": 11.0}
    assert blend_runs.line_prediction("motor", weights, "none", pred_tag="s2").to_dict() == {"t1": 6.5, "t2": 12.0}
    _pred_file(r, "motor", "glm", [100.0, 200.0], tag="s2")  # a tagged glm file wins when it exists
    assert blend_runs.line_prediction("motor", weights, "none", pred_tag="s2").to_dict() == {"t1": 51.5, "t2": 102.0}


def test_line_prediction_fails_when_a_tree_model_has_no_tagged_file(emerging_root):
    r = emerging_root / "r"
    _pred_file(r, "motor", "xgb", [1.0, 2.0])  # untagged exists, tagged does not
    with pytest.raises(AssertionError, match=r"pred_motor_xgb_s2\.parquet"):
        blend_runs.line_prediction("motor", {"r:xgb": 1.0}, "none", pred_tag="s2")
    assert blend_runs.line_prediction("motor", {"r:xgb": 1.0}, "none").to_dict() == {"t1": 1.0, "t2": 2.0}
    # a zero-weight candidate needs no file at all
    _pred_file(r, "motor", "lgbm", [3.0, 4.0], tag="s2")
    assert blend_runs.line_prediction("motor", {"r:lgbm": 1.0, "r:xgb": 0.0}, "none", pred_tag="s2").to_dict() == {
        "t1": 3.0, "t2": 4.0}


def test_line_prediction_passes_the_tag_through_the_segment_recursion(emerging_root, monkeypatch):
    r = emerging_root / "r"
    _pred_file(r, "motor", "lgbm", [1.0, 2.0])
    _pred_file(r, "motor", "lgbm", [3.0, 4.0], tag="s2")
    _pred_file(r, "motor", "xgb", [5.0, 6.0])
    _pred_file(r, "motor", "xgb", [7.0, 8.0], tag="s2")
    monkeypatch.setattr(blend_runs, "nb_flags_2025", lambda line: pd.Series([True, False], index=["t1", "t2"]))
    weights = {"nb": {"r:lgbm": 1.0}, "renewal": {"r:xgb": 1.0}}
    assert blend_runs.line_prediction("motor", weights, "none", pred_tag="s2").to_dict() == {"t1": 3.0, "t2": 8.0}
    assert blend_runs.line_prediction("motor", weights, "none").to_dict() == {"t1": 1.0, "t2": 6.0}


def test_write_from_passes_the_tag_to_every_line_and_logs_the_fallbacks(emerging_root, monkeypatch, capsys):
    r = emerging_root / "r"
    spec = {}
    for line in blend_runs.LINES:
        _pred_file(r, line, "lgbm", [1.0, 2.0])
        _pred_file(r, line, "lgbm", [3.0, 4.0], tag="s2")
        _pred_file(r, line, "glm", [10.0, 20.0])
        os.makedirs(emerging_root / f"res_{line}", exist_ok=True)
        (emerging_root / f"res_{line}" / f"blend_weights_{line}.json").write_text(
            json.dumps({"submission_weights": {"r:lgbm": 0.5, "r:glm": 0.5}}), encoding="utf-8")
        spec[line] = f"res_{line}"
    captured = {}
    monkeypatch.setattr(blend_runs, "write_submission", lambda by_line, name: captured.update(by_line=by_line))
    blend_runs.write_from(spec, "none", "unused", pred_tag="s2")
    assert all(p.to_dict() == {"t1": 6.5, "t2": 12.0} for p in captured["by_line"].values())
    out = capsys.readouterr().out
    for line in blend_runs.LINES:
        assert f"{line}: pred tag s2: 1 of 2 candidate file(s) tagged; untagged fallback: ['r:glm']" in out
    blend_runs.write_from(spec, "none", "unused")  # no tag: untagged files, no tag line
    assert all(p.to_dict() == {"t1": 5.5, "t2": 11.0} for p in captured["by_line"].values())
    assert "pred tag" not in capsys.readouterr().out


def test_cli_rejects_pred_tag_without_write_from():
    r = _cli("blend_runs.py", "motor,buildings,contents", "--pred-tag", "s2")
    assert r.returncode == 2 and "C057" in r.stderr and "--write-from" in r.stderr
    assert "--pred-tag" in _cli("blend_runs.py", "--help").stdout
