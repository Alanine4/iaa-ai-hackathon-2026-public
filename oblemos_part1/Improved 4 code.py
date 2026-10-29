"""
IAA AI Assisted Hackathon 2026 - Part 1
Improvement 4: CatBoost as a fourth model, blended with the GLM, GAM and LightGBM of Improved 3.

Run (from the D: environment, after "Improved 3 code.py" has been run once):
    D:\\iaa-hackathon\\venv\\Scripts\\python.exe "Improved 4 code.py"
Uses the corrected data of Improved 3 (prepared/improved3) and its saved 2024 validation and 2025 test
predictions, so only CatBoost is fitted here. Nothing produced by the earlier scripts is changed.

CatBoost (no tuning, time limited):
    Tweedie power 1.5 on capped claim cost, starting point log(exposure) + log(average rate) as baseline
    (boost_from_average off, same fix as LightGBM in Improved 1), depth 6, learning rate 0.1.
    Categorical variables use CatBoost's own ordered target statistics - a different way of handling
    many-level variables than LightGBM, which is what makes it useful in a blend.
    Trees: fit on 2021-2023 with early stopping on 2024, then refit on 2021-2024 with trees x data ratio.
    (The 2024 score of CatBoost is therefore slightly optimistic; the other models' 2024 scores are honest.)

Outputs (next to this script):
    submission_improved4_Blend4.parquet    (GLM + GAM + LightGBM + CatBoost) / 4   <- main submission
    submission_improved4_CatBoost.parquet  CatBoost alone
    prepared/improved4/comparison.csv, valid_predictions.parquet
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("first_code", HERE / "First code.py")
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

DATA3 = fc.OUT / "improved3"
OUT = fc.OUT / "improved4"
SUB_PREFIX = "submission_improved4"
TEST_YEAR = 2024
# fast settings (time limit): no combinations of categoricals (max_ctr_complexity=1), 32 bins
CB_PARAMS = dict(loss_function=f"Tweedie:variance_power={fc.METRIC_POWER}", depth=6, learning_rate=0.15,
                 l2_leaf_reg=10, border_count=32, max_ctr_complexity=1, boost_from_average=False,
                 random_seed=fc.SEED, thread_count=-1, verbose=100)
MAX_TREES = 600
# small tuning grid (time limited), best per line on 2024. Leaderboard: Blend4 (50% trees) 0.12736 < Blend3
# (33% trees) 0.12758, so the main file keeps trees at 33%: GLM + GAM + (LightGBM + CatBoost) / 2.
CB_GRID = [
    {},                                                                  # settings of the first run
    {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 30},               # smaller trees, more regularisation
    {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 30, "rsm": 0.5},  # half the variables per split
]
OLD = ["GLM", "GAM", "LightGBM"]
BLENDS = ["CatBoost", "Blend3", "Blend4", "Blend_trees33"]


def pool(df, feats, cats, base, label=True):
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return Pool(X, label=df["claim_cost_capped"].to_numpy(float) if label else None, cat_features=cats,
                baseline=np.log(df["exposure_adj"].to_numpy(float)) + base)


def predict(model, df, feats, cats, base, exposure):
    """Trees only (no baseline), then add back the starting point and the exposure offset."""
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype(str)
    raw = model.predict(Pool(X, cat_features=cats), prediction_type="RawFormulaVal")
    return np.exp(raw + base + np.log(exposure))


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((DATA3 / "meta.json").read_text())
    valid_old = pd.read_parquet(DATA3 / "valid_predictions.parquet")
    valid_all, test_all = [], []

    for line in fc.LINES:
        t0 = time.time()
        fc.section(f"CATBOOST {line.upper()}")
        info = meta["lines"][line]
        feats = [f for f in info["features"] if f not in fc.LEAKY_FEATURES]
        cats = [c for c in info["categorical"] if c in feats]
        loading = info["large_loss_loading"]
        fr = pd.read_parquet(DATA3 / f"freq_{line}.parquet")
        sv = pd.read_parquet(DATA3 / f"sev_{line}.parquet")
        te = pd.read_parquet(DATA3 / f"test_{line}.parquet")
        fr["exposure_adj"] = fr["exposure_adj"].clip(lower=1e-4)
        tr, va = fr[fr["year"] < TEST_YEAR], fr[fr["year"] == TEST_YEAR]

        # ---------------- fit on 2021-2023, early stopping on 2024
        base = np.log(tr["claim_cost_capped"].sum() / tr["exposure_adj"].sum())
        trend_b, _ = fc.severity_trend(sv[sv["year"] < TEST_YEAR], TEST_YEAR)
        p_tr, p_va = pool(tr, feats, cats, base), pool(va, feats, cats, base)
        best = None
        for i, extra in enumerate(CB_GRID):
            params = {**CB_PARAMS, **extra, "verbose": 0}
            m = CatBoostRegressor(iterations=MAX_TREES, **params)
            m.fit(p_tr, eval_set=p_va, early_stopping_rounds=50, use_best_model=True)
            p = predict(m, va, feats, cats, base, va["exposure_adj"].to_numpy()) * loading * trend_b
            score = fc.d2(va["claim_cost"].to_numpy(), p)
            print(f"  setting {i} {extra}: {m.get_best_iteration() + 1} trees, 2024 D2 {score:.4f}", flush=True)
            if best is None or score > best[0]:
                best = (score, params, m.get_best_iteration() + 1, p)
        _, params, n_trees, p_cb = best
        v = valid_old[valid_old["line"] == line].merge(
            pd.DataFrame({"row_id": va["row_id"].to_numpy(), "CatBoost": p_cb}), on="row_id")
        v["Blend3"] = v[OLD].mean(axis=1)
        v["Blend4"] = v[OLD + ["CatBoost"]].mean(axis=1)
        v["Blend_trees33"] = (v["GLM"] + v["GAM"] + (v["LightGBM"] + v["CatBoost"]) / 2) / 3
        valid_all.append(v)
        print(f"  chosen: {n_trees} trees | 2024 D2: " + " | ".join(
            f"{k} {fc.d2(v['y'], v[k]):.4f}" for k in OLD + BLENDS))

        # ---------------- refit on 2021-2024, predict 2025
        base_f = np.log(fr["claim_cost_capped"].sum() / fr["exposure_adj"].sum())
        mf = CatBoostRegressor(iterations=int(n_trees * len(fr) / len(tr)) + 1, **params)
        mf.fit(pool(fr, feats, cats, base_f))
        trend_te, _ = fc.severity_trend(sv, 2025)
        t_cb = predict(mf, te, feats, cats, base_f, te["exposure"].to_numpy()) * loading * trend_te
        test_all.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "line": line, "CatBoost": t_cb}))
        print(f"  2025 CatBoost total {t_cb.sum() / 1e6:.1f}M   [{(time.time() - t0) / 60:.1f} min]")

    # ---------------- comparison
    fc.section("COMPARISON - 2024 validation D2")
    v = pd.concat(valid_all, ignore_index=True)
    rows = [{"line": line, "model": k, "D2": fc.d2(g["y"], g[k]), "pred/actual": g[k].sum() / g["y"].sum()}
            for k in OLD + BLENDS for line, g in [("ALL", v)] + list(v.groupby("line"))]
    comp = pd.DataFrame(rows)
    print(comp.pivot(index="model", columns="line", values="D2").reindex(OLD + BLENDS).round(4).to_string())
    comp.to_csv(OUT / "comparison.csv", index=False)
    v.to_parquet(OUT / "valid_predictions.parquet", index=False)

    # ---------------- submissions
    fc.section("SUBMISSIONS")
    sample = fc.rd("sample_submission")
    t = pd.concat(test_all, ignore_index=True)
    for k in OLD:
        old = pd.read_parquet(HERE / f"submission_improved3_{k}.parquet").set_index("row_id")["prediction"]
        t[k] = old.reindex(t["row_id"]).to_numpy()
    t["Blend4"] = t[OLD + ["CatBoost"]].mean(axis=1)
    t["Blend_trees33"] = (t["GLM"] + t["GAM"] + (t["LightGBM"] + t["CatBoost"]) / 2) / 3
    for name in ["Blend_trees33", "Blend4", "CatBoost"]:
        path = HERE / f"{SUB_PREFIX}_{name}.parquet"
        sub = sample[["row_id"]].merge(t[["row_id", name]].rename(columns={name: "prediction"}), on="row_id",
                                       how="left")
        assert sub["prediction"].notna().all() and len(sub) == len(sample), f"missing predictions ({name})"
        sub.to_parquet(path, index=False)
        print(f"saved {path.name:40s} mean {sub['prediction'].mean():7.2f}")


if __name__ == "__main__":
    main()
