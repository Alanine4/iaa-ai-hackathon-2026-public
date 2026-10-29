"""
IAA AI Assisted Hackathon 2026 - Part 1
Improvement 2: honest validation, LightGBM without weather variables, weighted blend.

Run (from the D: environment, after "First code.py" and "Improved 1 code.py" have been run once):
    D:\\iaa-hackathon\\venv\\Scripts\\python.exe "Improved 2 code.py"
Uses the prepared files, the GLM/GAM submissions of "First code.py" and the LightGBM settings
tuned by "Improved 1 code.py". Nothing from those files is changed or overwritten.

Why (findings after the leaderboard results of Improved 1):
    - Leaderboard: LightGBM alone 0.12272 < Improved 1 best 0.12440 < blend 1/3 each 0.12742,
      while the 2024 validation ranked LightGBM alone first. A paired bootstrap shows 2024 separates
      these models to within +/-0.0007, so the reversal is real: 2025 behaves differently for LightGBM.
    - Validation was made honest (tuning and tree count on 2023 only, refit on 2021-2023 like the final
      model, score on untouched 2024). LightGBM still wins on 2024, so the tuning itself was not the problem.
    - Smaller trees (3 or 5 leaves) score the same as 7 leaves -> the Improved 1 settings are kept.
    - Weather variables identify the year: a classifier separates 2024 from 2025 rows with AUC 0.999 using
      them alone. They give LightGBM a hidden "year" variable, so it can learn year-specific shocks, e.g.
      the immature (not yet fully reported/settled) 2024 claims that enter only the final 2021-2024 fit.
      The GLM never selected a weather variable. Without them LightGBM scores the same on 2024
      (buildings 0.1149 vs 0.1149, contents 0.0945 vs 0.0940, motor 0.0854 vs 0.0859) -> removed.

Honest validation used everywhere here:
    fold 2023: train 2021-2022, early stopping on 2023 -> number of trees
    then refit on 2021-2023 with trees x (data ratio), no early stopping, score on 2024.

Outputs (next to this script; each file answers one question on the leaderboard):
    submission_improved2_Blend_equal.parquet     (GLM + GAM + new LightGBM) / 3
                                                 -> compare with 0.12742: does removing weather help?
    submission_improved2_Blend_weighted.parquet  weights fitted on the honest 2024 predictions
                                                 -> compare with Blend_equal: does more LightGBM weight help now?
    submission_improved2_LightGBM.parquet        new LightGBM alone, fitted on 2021-2024
                                                 -> compare with 0.12272: is LightGBM alone fixed?
    submission_improved2_LightGBM_2021_2023.parquet  same, fitted on 2021-2023 only (no immature 2024 claims)
                                                 -> compare with the previous file: does 2024 data hurt?
    prepared/improved2/valid_predictions.parquet, comparison.csv, blend_weights.json
"""
import importlib.util
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("improved_1", HERE / "Improved 1 code.py")
im = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(im)
fc = im.fc

OUT = fc.OUT / "improved2"
SUB_PREFIX = "submission_improved2"
WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index",
           "rainfall_mm", "mean_temperature_c"]
TUNE_YEAR, TEST_YEAR = 2023, 2024    # honest validation: trees chosen on 2023, scored on 2024
WEIGHT_STEP = 0.05                   # grid step for the blend weights


# ----------------------------------------------------------------------------- helpers
def lgbm_settings(line):
    cfg = json.loads((fc.OUT / "improved1" / "best_params.json").read_text())[line]
    return {k: cfg[k] for k in im.SEARCH_SPACE}


def fit_blend_weights(y, preds):
    """Non-negative weights summing to 1 that maximise D2, searched on a grid."""
    names = list(preds)
    grid = np.round(np.arange(0, 1 + 1e-9, WEIGHT_STEP), 4)
    best = (-np.inf, None)
    for w in product(grid, repeat=len(names) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = (*w, 1 - sum(w))
        score = fc.d2(y, sum(wi * preds[n] for wi, n in zip(w, names)))
        best = max(best, (score, w), key=lambda t: t[0])
    return dict(zip(names, [float(x) for x in best[1]])), best[0]


# ----------------------------------------------------------------------------- driver
def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((fc.OUT / "meta.json").read_text())
    selections = json.loads((fc.MODELS / "selected_variables.json").read_text())
    valid_all, test_all, comparison = [], [], []

    for line in fc.LINES:
        t0 = time.time()
        fc.section(f"IMPROVED 2 - {line.upper()}")
        info = meta["lines"][line]
        feats_all = [f for f in info["features"] if f not in fc.LEAKY_FEATURES]
        feats = [f for f in feats_all if f not in WEATHER]
        cats = [c for c in info["categorical"] if c in feats]
        loading = info["large_loss_loading"]
        cfg = lgbm_settings(line)
        fr = pd.read_parquet(fc.OUT / f"freq_{line}.parquet")
        sv = pd.read_parquet(fc.OUT / f"sev_{line}.parquet")
        te = pd.read_parquet(fc.OUT / f"test_{line}.parquet")
        fr["exposure_adj"] = fr["exposure_adj"].clip(lower=1e-4)
        tr_a, va_a = fr[fr["year"] < TUNE_YEAR], fr[fr["year"] == TUNE_YEAR]
        tr_b, va_b = fr[fr["year"] < TEST_YEAR], fr[fr["year"] == TEST_YEAR]
        y_b = va_b["claim_cost"].to_numpy()
        trend_b, _ = fc.severity_trend(sv[sv["year"] < TEST_YEAR], TEST_YEAR)

        # ---------------- LightGBM without weather: trees on 2023, refit 2021-2023, score 2024
        booster, _ = im.fit_lgbm(tr_a, va_a, feats, cats, cfg, im.FINAL_LR)
        n_trees = booster.best_iteration
        n_b = int(n_trees * len(tr_b) / len(tr_a)) + 1
        _, predict_b = im.fit_lgbm(tr_b, None, feats, cats, cfg, im.FINAL_LR, rounds=n_b)
        p_lgb = predict_b(va_b, va_b["exposure_adj"].to_numpy()) * loading * trend_b
        print(f"  LightGBM (no weather): {n_trees} trees on 2021-{TUNE_YEAR - 1}, {n_b} on 2021-{TEST_YEAR - 1}, "
              f"honest 2024 D2 {fc.d2(y_b, p_lgb):.4f}")

        # ---------------- GLM and GAM of "First code.py" (they never used 2024 for any choice)
        p_glm, p_gam = im.glm_gam_valid(fr, sv, selections[line]["frequency"], selections[line]["severity"],
                                        feats_all)
        p_glm, p_gam = p_glm * loading * trend_b, p_gam * loading * trend_b
        valid_all.append(pd.DataFrame({"line": line, "row_id": va_b["row_id"].to_numpy(), "y": y_b,
                                       "GLM": p_glm, "GAM": p_gam, "LightGBM": p_lgb}))
        print(f"  honest 2024 D2: GLM {fc.d2(y_b, p_glm):.4f} | GAM {fc.d2(y_b, p_gam):.4f} | "
              f"LightGBM {fc.d2(y_b, p_lgb):.4f} | equal blend {fc.d2(y_b, (p_glm + p_gam + p_lgb) / 3):.4f}")

        # ---------------- 2025 predictions
        expo = te["exposure"].to_numpy()
        trend_full, _ = fc.severity_trend(sv, 2025)
        trend_old, _ = fc.severity_trend(sv[sv["year"] < TEST_YEAR], 2025)
        n_full = int(n_trees * len(fr) / len(tr_a)) + 1
        _, predict_full = im.fit_lgbm(fr, None, feats, cats, cfg, im.FINAL_LR, rounds=n_full)
        test_all.append(pd.DataFrame({
            "row_id": te["row_id"].to_numpy(), "line": line,
            "LightGBM": predict_full(te, expo) * loading * trend_full,
            "LightGBM_2021_2023": predict_b(te, expo) * loading * trend_old}))
        print(f"  2025 LightGBM: {n_full} trees on 2021-2024   [{(time.time() - t0) / 60:.1f} min]")

    # ---------------- blend weights fitted on the honest 2024 predictions, all lines together
    fc.section("BLEND WEIGHTS AND COMPARISON (honest 2024 validation)")
    v = pd.concat(valid_all, ignore_index=True)
    weights, _ = fit_blend_weights(v["y"].to_numpy(), {m: v[m].to_numpy() for m in ["LightGBM", "GLM", "GAM"]})
    print("fitted weights:", {k: round(w, 2) for k, w in weights.items()})
    v["Blend_equal"] = (v["GLM"] + v["GAM"] + v["LightGBM"]) / 3
    v["Blend_weighted"] = sum(w * v[m] for m, w in weights.items())
    for name in ["GLM", "GAM", "LightGBM", "Blend_equal", "Blend_weighted"]:
        for line, g in [("ALL", v)] + list(v.groupby("line")):
            comparison.append({"line": line, "model": name, "D2": fc.d2(g["y"], g[name]),
                               "pred/actual": g[name].sum() / g["y"].sum()})
    comp = pd.DataFrame(comparison)
    print(comp.pivot(index="model", columns="line", values="D2")
          .reindex(["GLM", "GAM", "LightGBM", "Blend_equal", "Blend_weighted"]).round(4).to_string())
    comp.to_csv(OUT / "comparison.csv", index=False)
    v.to_parquet(OUT / "valid_predictions.parquet", index=False)
    (OUT / "blend_weights.json").write_text(json.dumps(weights, indent=2))

    # ---------------- submissions (new names, earlier files are kept)
    fc.section("SUBMISSIONS")
    sample = fc.rd("sample_submission")
    t = pd.concat(test_all, ignore_index=True)
    for m in ["GLM", "GAM"]:
        old = pd.read_parquet(HERE / f"submission_{m}.parquet").set_index("row_id")["prediction"]
        t[m] = old.reindex(t["row_id"]).to_numpy()
    t["Blend_equal"] = (t["GLM"] + t["GAM"] + t["LightGBM"]) / 3
    t["Blend_weighted"] = sum(w * t[m] for m, w in weights.items())
    for name in ["Blend_equal", "Blend_weighted", "LightGBM", "LightGBM_2021_2023"]:
        path = HERE / f"{SUB_PREFIX}_{name}.parquet"
        sub = sample[["row_id"]].merge(t[["row_id", name]].rename(columns={name: "prediction"}), on="row_id",
                                       how="left")
        assert sub["prediction"].notna().all() and len(sub) == len(sample), f"missing predictions ({name})"
        sub.to_parquet(path, index=False)
        totals = ", ".join(f"{line} {g[name].sum() / 1e6:.1f}M" for line, g in t.groupby("line"))
        print(f"saved {path.name:50s} mean {sub['prediction'].mean():7.2f}   totals: {totals}")


if __name__ == "__main__":
    main()
