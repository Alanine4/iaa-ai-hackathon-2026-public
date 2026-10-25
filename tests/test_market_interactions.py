"""C042: emerging-specific GLM relativities in the pooled fit (features.market_interactions, --glm-interactions)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from aiiaa import features, markets, models

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import oof_predictions  # noqa: E402


def _frame():
    return pd.DataFrame({
        "market": pd.Categorical(["developed", "developed", "emerging", "emerging", "emerging"]),
        "channel": pd.Categorical(["broker", "direct", "direct", None, "broker"]),
        "new_business": [True, False, False, True, False],
        "sum_insured": [100.0, 200.0, 300.0, np.nan, 500.0],
        "other": [1, 2, 3, 4, 5],
    }, index=[10, 11, 12, 13, 14])


def test_market_interactions_appends_one_column_per_factor_and_keeps_the_rest():
    x = _frame()
    out = features.market_interactions(x, ["channel", "new_business", "sum_insured"])
    assert list(out.columns) == list(x.columns) + ["channel__em", "new_business__em", "sum_insured__em"]
    pd.testing.assert_frame_equal(out[x.columns], x)
    assert list(out.index) == list(x.index)


def test_categorical_interaction_has_emerging_levels_plus_one_developed_level():
    out = features.market_interactions(_frame(), ["channel", "new_business"])
    ch = out["channel__em"]
    assert isinstance(ch.dtype, pd.CategoricalDtype)
    assert list(ch.cat.categories) == ["broker", "direct", "developed"]
    assert ch.iloc[0] == "developed" and ch.iloc[1] == "developed"
    assert ch.iloc[2] == "direct" and pd.isna(ch.iloc[3]) and ch.iloc[4] == "broker"  # emerging missing stays missing
    nb = out["new_business__em"]
    assert isinstance(nb.dtype, pd.CategoricalDtype)
    assert list(nb.cat.categories) == ["False", "True", "developed"]
    assert list(nb.astype(str)) == ["developed", "developed", "False", "True", "False"]


def test_numeric_interaction_is_the_emerging_value_and_missing_on_developed_rows():
    out = features.market_interactions(_frame(), ["sum_insured"])
    s = out["sum_insured__em"]
    assert s.dtype == float
    assert s.iloc[:2].isna().all()
    assert s.iloc[2] == 300.0 and np.isnan(s.iloc[3]) and s.iloc[4] == 500.0


def test_market_interactions_need_both_markets_and_known_columns():
    x = _frame()
    with pytest.raises(ValueError, match="both markets"):
        features.market_interactions(x[x["market"] == "emerging"], ["channel"])
    with pytest.raises(KeyError):
        features.market_interactions(x, ["not_a_column"])


def test_interaction_glm_recovers_an_emerging_relativity_the_main_effect_glm_misses():
    # developed relativities of f: a 1.0, b 1.5, c 2.0; emerging: a 1.0, b 0.7, c 3.0 (plus a market level)
    rng = np.random.default_rng(0)
    n_dev, n_em = 20000, 6000
    market = np.array(["developed"] * n_dev + ["emerging"] * n_em)
    f = rng.choice(["a", "b", "c"], n_dev + n_em)
    rel = {"developed": {"a": 1.0, "b": 1.5, "c": 2.0}, "emerging": {"a": 1.0, "b": 0.7, "c": 3.0}}
    mu = 100.0 * np.array([rel[m][v] for m, v in zip(market, f)]) * np.where(market == "emerging", 0.8, 1.0)
    claims = rng.poisson(0.2, len(mu))
    y = np.where(claims > 0, rng.gamma(2.0 * np.maximum(claims, 1), mu / 0.2 / 2.0 / np.maximum(claims, 1)), 0.0)
    x = pd.DataFrame({"market": pd.Categorical(market), "f": pd.Categorical(f)})
    w = np.ones(len(x))
    em = market == "emerging"

    def emerging_log_error(frame):
        m = models.make("glm").fit(frame, y, w)
        pred = m.predict(frame[em])
        return float(np.mean(np.abs(np.log(pred / mu[em]))))

    main_only = emerging_log_error(x)
    with_interaction = emerging_log_error(features.market_interactions(x, ["f"]))
    # ~400 emerging claims per level: sampling error about 0.06 per relativity; pooled main effect is far off
    assert with_interaction < 0.1 and main_only > 0.3


def test_glm_interaction_columns_are_chosen_per_line_and_exist_in_the_harmonised_schema():
    cols = oof_predictions.GLM_INTERACTIONS
    assert sorted(cols) == ["buildings", "contents", "motor"]
    for line, chosen in cols.items():
        assert 1 <= len(chosen) <= 6 and len(set(chosen)) == len(chosen)
        dropped = features.DROP + features.DROP_BY_LINE["all"] + features.DROP_BY_LINE[line]
        assert not set(chosen) & set(dropped), f"{line}: interaction on a dropped column"
        assert "region" not in chosen  # region codes already differ by market (R01-R12 vs E01-E06)


def test_emerging_main_writes_the_glm_interaction_run_into_its_own_folder(tmp_path, monkeypatch):
    # C042 plumbing on a synthetic pooled design: new folder <run_dir>_glmx, interaction columns reach the GLM
    monkeypatch.setattr(markets, "ROOT", str(tmp_path))
    rng = np.random.default_rng(1)
    n_dev, n_em, n_test = 3000, 1200, 200
    n = n_dev + n_em + n_test
    part = np.array(["dev"] * n_dev + ["em"] * n_em + ["test"] * n_test, dtype=object)
    x = pd.DataFrame({"market": pd.Categorical(np.where(part == "dev", "developed", "emerging")),
                      "channel": pd.Categorical(rng.choice(["broker", "direct"], n)), "v": rng.uniform(0, 1, n)})
    y = np.where(rng.uniform(size=n) < 0.2, rng.gamma(1.0, 500.0, n), 0.0) * (part != "test")
    w = np.where(part == "test", 0.0, 1.0)
    year = np.where(np.arange(n_em) < n_em // 2, 2023, 2024)
    train = pd.DataFrame({"policy_id": [f"P{i % 700:04d}" for i in range(n_em)]})
    t = pd.DataFrame({"row_id": [f"E{i:05d}" for i in range(n_em)], "claim_cost": y[part == "em"],
                      "claim_cost_capped": y[part == "em"], "exposure": 1.0, "completeness": 1.0})
    test = pd.DataFrame({"row_id": [f"T{i:05d}" for i in range(n_test)], "exposure": 1.0})
    seen = {}

    def design(line, train_set, dev_weight, glm_interactions=False):
        seen["glm_interactions"] = glm_interactions
        xx = features.market_interactions(x, ["channel"]) if glm_interactions else x
        return xx, part, y, w, train, test, t, year

    real_make = models.make

    def spy_make(name, **kw):
        m = real_make(name, **kw)
        real_fit = m.fit

        def fit(xf, *a, **k):
            seen.setdefault("columns", list(xf.columns))
            return real_fit(xf, *a, **k)
        m.fit = fit
        return m

    monkeypatch.setattr(oof_predictions, "emerging_design", design)
    monkeypatch.setattr(oof_predictions.models, "make", spy_make)
    oof_predictions.emerging_main("contents", ["glm"], "pooled", 0.3, "none", glm_interactions=True)
    out = tmp_path / "output" / "emerging" / "pooled_w0.3_glmx"
    assert seen["glm_interactions"] is True and "channel__em" in seen["columns"]
    assert (out / "oof_contents.parquet").exists() and (out / "pred_contents_glm.parquet").exists()
    with open(out / "blend_weights_contents.json", encoding="utf-8") as f:
        res = json.load(f)
    assert res["glm_interactions"] == oof_predictions.GLM_INTERACTIONS["contents"]
    assert not (tmp_path / "output" / "emerging" / "pooled_w0.3").exists()  # the plain run folder is untouched


def _cli(*args):
    return subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "oof_predictions.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_cli_lists_the_glm_interactions_flag():
    r = _cli("--help")
    assert r.returncode == 0 and "--glm-interactions" in r.stdout


@pytest.mark.parametrize("args", [["--train-set", "pooled", "--models", "glm,lgbm"],
                                  ["--train-set", "emerging", "--models", "glm"]])
def test_cli_rejects_glm_interactions_outside_a_pooled_glm_only_run(args):
    r = _cli("contents", "--market", "emerging", "--glm-interactions", *args)
    assert r.returncode != 0 and "C042" in r.stderr
