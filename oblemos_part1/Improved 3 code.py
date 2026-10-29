"""
IAA AI Assisted Hackathon 2026 - Part 1
Improvement 3: fix the customer claim history variables, then refit GLM, GAM and LightGBM.

Run (from the D: environment, after "First code.py" and "Improved 1 code.py" have been run once):
    D:\\iaa-hackathon\\venv\\Scripts\\python.exe "Improved 3 code.py"
Nothing produced by the earlier scripts is changed or overwritten.

The problem (found after the Improved 2 leaderboard results):
    cust_claims_prev_1y = claims of the customer in the 365 days before the period start.
    For training rows every such claim is in the data. For a 2025 test row most of that window lies after
    the data cut-off (2024-12-30), so its claims cannot be known: the variable is far too low in test.
    Motor, share of rows with >= 1 claim in the previous 12 months, by start quarter:
        2022-2024 training: Q1 0.13  Q2 0.043  Q3 0.041  Q4 0.040
        2025 test         : Q1 0.117 Q2 0.024  Q3 0.015  Q4 0.004
    Customers with 1 claim last year have 1.7x the motor claim frequency (2+ claims: 3x), so in test
    these risky customers look claim-free. The 2024 validation could not see this (its history was complete),
    which is why LightGBM, the model using it most, looked better on 2024 than it was on 2025.

The fix: only count claims that would be known at the same point in time in training and test.
    For a period starting in year Y the information date is 1 January of Y (for 2025 this is the data
    cut-off), so a claim is counted only if it was reported before that date.
        cust_claims_prev_1y[_same_line]        loss in the 365 days before the start, reported before 1 Jan Y
        cust_claims_prev_cal_year[_same_line]  NEW: loss in calendar year Y-1, reported before 1 Jan Y
                                               (same meaning for every start date, also useful for Q4 starts)
    Weather variables are used again (removing them did not change the leaderboard score).

Modelling (as before, with the honest validation of Improved 2: no choice is made with 2024):
    GLM : variables selected again by forward stepwise BIC on 2021-2023 (the history variables changed)
    GAM : same variables, smoothing chosen by AIC on 2021-2023
    LightGBM : settings tuned in Improved 1, trees chosen on 2023 (train 2021-2022), refit on 2021-2023
    Scored on 2024, then every model is refitted on 2021-2024 and predicts 2025.

Outputs (next to this script):
    submission_improved3_Blend.parquet      (GLM + GAM + LightGBM) / 3   <- main submission
    submission_improved3_GLM.parquet, submission_improved3_GAM.parquet, submission_improved3_LightGBM.parquet
    prepared/improved3/  corrected freq/sev/test files, meta.json, selected_variables.json,
                         comparison.csv, valid_predictions.parquet
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("improved_1", HERE / "Improved 1 code.py")
im = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(im)
fc = im.fc

OUT = fc.OUT / "improved3"
SUB_PREFIX = "submission_improved3"
HISTORY_COLS = ["cust_claims_prev_1y", "cust_claims_prev_1y_same_line",
                "cust_claims_prev_cal_year", "cust_claims_prev_cal_year_same_line"]
TUNE_YEAR, TEST_YEAR = 2023, 2024    # honest validation: trees chosen on 2023, scored on 2024
MODELS = ["GLM", "GAM", "LightGBM"]
# The two history variables share their "missing" column (2021 rows). That duplicate made the score test
# blow up and fill the 3 refit slots, so the motor selection stopped after 10 variables.
fc.SCORE_RCOND = 1e-6
fc.SELECTION_TRIES = 10


# ----------------------------------------------------------------------------- step 1: corrected data
def claim_history_known(rows, claims, policy_customer, line):
    """Customer claim counts using only claims reported before 1 January of the period's year
    (the same information a 2025 row has at the data cut-off). NaN for 2021 periods (no history)."""
    c = claims[["policy_id", "line", "loss_date", "report_date"]].merge(policy_customer, on="policy_id")
    m = rows[["row_id", "customer_id", "period_start"]].merge(c, on="customer_id")
    start = m["period_start"]
    known = m["report_date"] < pd.to_datetime(start.dt.year.astype(str) + "-01-01")
    window = {
        "prev_1y": known & (m["loss_date"] < start) & (m["loss_date"] >= start - pd.Timedelta(days=365)),
        "prev_cal_year": known & (m["loss_date"].dt.year == start.dt.year - 1),
    }
    out = pd.DataFrame(index=rows.index)
    for name, mask in window.items():
        out[f"cust_claims_{name}"] = rows["row_id"].map(m[mask].groupby("row_id").size()).fillna(0)
        same = mask & (m["line"] == line)
        out[f"cust_claims_{name}_same_line"] = rows["row_id"].map(m[same].groupby("row_id").size()).fillna(0)
    out.loc[(rows["period_start"] < pd.Timestamp("2022-01-01")).to_numpy(), HISTORY_COLS] = np.nan
    return out.astype("float32")


def prepare_corrected():
    """Copies of the prepared files with the claim history variables recomputed."""
    if (OUT / "meta.json").exists():
        print(f"corrected files already in {OUT} (delete the folder to rebuild)")
        return
    fc.section("CORRECTED CLAIM HISTORY VARIABLES")
    OUT.mkdir(exist_ok=True)
    claims = fc.clean_claims(fc.rd("claims"))
    pol_cust = []
    for line in fc.LINES:
        for kind in ("policies", "test"):
            h = fc.rd(f"{kind}_{line}", columns=["policy_id", "customer_id"])
            h["customer_id"] = h["customer_id"].astype(str).str.zfill(8)
            pol_cust.append(h)
    policy_customer = pd.concat(pol_cust).drop_duplicates("policy_id")
    meta = json.loads((fc.OUT / "meta.json").read_text())

    for line in fc.LINES:
        fr = pd.read_parquet(fc.OUT / f"freq_{line}.parquet")
        te = pd.read_parquet(fc.OUT / f"test_{line}.parquet")
        sv = pd.read_parquet(fc.OUT / f"sev_{line}.parquet")
        for df in (fr, te):
            df["customer_id"] = df["customer_id"].astype(str)
            new = claim_history_known(df, claims, policy_customer, line)
            df[HISTORY_COLS] = new[HISTORY_COLS].to_numpy()
        sv = sv.drop(columns=[c for c in HISTORY_COLS if c in sv.columns]).merge(
            fr[["row_id"] + HISTORY_COLS], on="row_id", how="left")

        # check: the variables must now look the same in training and test
        a = pd.concat([fr.assign(src=fr["year"].astype(str)), te.assign(src="2025 test")])
        a = a[a["year"] >= 2022]
        share = (a.groupby(["src", a["period_start"].dt.quarter])["cust_claims_prev_1y"]
                 .apply(lambda s: (s > 0).mean()).unstack().round(3))
        share.columns = [f"Q{q}" for q in share.columns]
        print(f"\n{line}: share of rows with >= 1 known claim in the previous 12 months, by start quarter")
        print(share.to_string())
        print("  previous calendar year, share >= 1:",
              a.groupby("src")["cust_claims_prev_cal_year"].apply(lambda s: round((s > 0).mean(), 3)).to_dict())

        meta["lines"][line]["features"] = (meta["lines"][line]["features"]
                                           + [c for c in HISTORY_COLS if c not in meta["lines"][line]["features"]])
        fr.to_parquet(OUT / f"freq_{line}.parquet", compression="zstd", index=False)
        te.to_parquet(OUT / f"test_{line}.parquet", compression="zstd", index=False)
        sv.to_parquet(OUT / f"sev_{line}.parquet", compression="zstd", index=False)
        del fr, te, sv
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------- step 2: models
def fit_glm_gam(tr, sv_tr, feats, label):
    """Variable selection + GLM + GAM on the given training years, as in "First code.py"."""
    samp = tr.sample(min(len(tr), fc.SELECTION_SAMPLE), random_state=fc.SEED)
    f_feats, _ = fc.forward_select(samp, feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float),
                                   1, np.log(samp["exposure_adj"].to_numpy()), f"{label} frequency")
    s_feats, phi_s = fc.forward_select(sv_tr, feats, np.ones(len(sv_tr)), sv_tr["net_cost_capped"].to_numpy(float),
                                       2, None, f"{label} severity")
    lam_f = fc.choose_lambda(samp, f_feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float),
                             1, np.log(samp["exposure_adj"].to_numpy()), 1.0, "GAM frequency")
    lam_s = fc.choose_lambda(sv_tr, s_feats, np.ones(len(sv_tr)), sv_tr["net_cost_capped"].to_numpy(float),
                             2, None, phi_s, "GAM severity")
    glm = fc.fit_freq_sev(tr, sv_tr, f_feats, s_feats)
    gam = fc.fit_freq_sev(tr, sv_tr, f_feats, s_feats, smooth=True, lam_f=lam_f * len(tr) / len(samp), lam_s=lam_s)
    choice = {"frequency": f_feats, "severity": s_feats, "lam_f": lam_f / len(samp), "lam_s": lam_s / len(sv_tr)}
    return glm, gam, choice


def main():
    prepare_corrected()
    meta = json.loads((OUT / "meta.json").read_text())
    lgb_cfg = json.loads((fc.OUT / "improved1" / "best_params.json").read_text())
    valid_all, test_all, selections = [], [], {}

    for line in fc.LINES:
        t0 = time.time()
        fc.section(f"IMPROVED 3 - {line.upper()}")
        info = meta["lines"][line]
        feats = [f for f in info["features"] if f not in fc.LEAKY_FEATURES]
        cats = [c for c in info["categorical"] if c in feats]
        loading = info["large_loss_loading"]
        cfg = {k: lgb_cfg[line][k] for k in im.SEARCH_SPACE}
        fr = pd.read_parquet(OUT / f"freq_{line}.parquet")
        sv = pd.read_parquet(OUT / f"sev_{line}.parquet")
        te = pd.read_parquet(OUT / f"test_{line}.parquet")
        fr["exposure_adj"] = fr["exposure_adj"].clip(lower=1e-4)
        tr_a, va_a = fr[fr["year"] < TUNE_YEAR], fr[fr["year"] == TUNE_YEAR]
        tr_b, va_b = fr[fr["year"] < TEST_YEAR], fr[fr["year"] == TEST_YEAR]
        sv_b = sv[sv["year"] < TEST_YEAR]
        y_b = va_b["claim_cost"].to_numpy()
        expo_b = va_b["exposure_adj"].to_numpy()
        trend_b, _ = fc.severity_trend(sv_b, TEST_YEAR)

        # ---------------- honest 2024 validation
        glm, gam, choice = fit_glm_gam(tr_b, sv_b, feats, "2021-2023")
        selections[line] = choice
        booster, _ = im.fit_lgbm(tr_a, va_a, feats, cats, cfg, im.FINAL_LR)
        n_trees = booster.best_iteration
        _, predict_b = im.fit_lgbm(tr_b, None, feats, cats, cfg, im.FINAL_LR,
                                   rounds=int(n_trees * len(tr_b) / len(tr_a)) + 1)
        p = {"GLM": fc.predict_freq_sev(glm, va_b, expo_b), "GAM": fc.predict_freq_sev(gam, va_b, expo_b),
             "LightGBM": predict_b(va_b, expo_b)}
        p = {k: v * loading * trend_b for k, v in p.items()}
        p["Blend"] = sum(p[m] for m in MODELS) / 3
        valid_all.append(pd.DataFrame({"line": line, "row_id": va_b["row_id"].to_numpy(), "y": y_b, **p}))
        print("  honest 2024 D2: " + " | ".join(f"{k} {fc.d2(y_b, v):.4f}" for k, v in p.items()))
        imp = pd.Series(booster.feature_importance("gain"), index=feats)
        print(f"  LightGBM gain share of the claim history variables: "
              f"{(imp[HISTORY_COLS].sum() / imp.sum()):.3f}")

        # ---------------- refit on 2021-2024 and predict 2025
        expo = te["exposure"].to_numpy()
        trend_te, _ = fc.severity_trend(sv, 2025)
        f_feats, s_feats = choice["frequency"], choice["severity"]
        t = {"GLM": fc.predict_freq_sev(fc.fit_freq_sev(fr, sv, f_feats, s_feats), te, expo)}
        # smoothing per row kept constant (same smoothness on more data, as in "First code.py")
        gam_full = fc.fit_freq_sev(fr, sv, f_feats, s_feats, smooth=True,
                                   lam_f=choice["lam_f"] * len(fr), lam_s=choice["lam_s"] * len(sv))
        t["GAM"] = fc.predict_freq_sev(gam_full, te, expo)
        _, predict_full = im.fit_lgbm(fr, None, feats, cats, cfg, im.FINAL_LR,
                                      rounds=int(n_trees * len(fr) / len(tr_a)) + 1)
        t["LightGBM"] = predict_full(te, expo)
        t = {k: v * loading * trend_te for k, v in t.items()}
        t["Blend"] = sum(t[m] for m in MODELS) / 3
        test_all.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "line": line, **t}))
        print(f"  2025 totals: " + ", ".join(f"{k} {v.sum() / 1e6:.1f}M" for k, v in t.items())
              + f"   [{(time.time() - t0) / 60:.1f} min]")
        del fr, sv, te, tr_a, va_a, tr_b, va_b, sv_b

    # ---------------- comparison with the earlier (honest) results
    fc.section("COMPARISON - honest 2024 validation D2")
    v = pd.concat(valid_all, ignore_index=True)
    rows = []
    for name in MODELS + ["Blend"]:
        for line, g in [("ALL", v)] + list(v.groupby("line")):
            rows.append({"line": line, "model": name, "D2": fc.d2(g["y"], g[name]),
                         "pred/actual": g[name].sum() / g["y"].sum()})
    comp = pd.DataFrame(rows)
    print(comp.pivot(index="model", columns="line", values="D2").reindex(MODELS + ["Blend"]).round(4).to_string())
    old = fc.OUT / "improved2" / "comparison.csv"
    if old.exists():
        o = pd.read_csv(old).replace({"Blend_equal": "Blend"})
        print("\nImproved 2 (old history variables, no weather in LightGBM), same validation:")
        print(o[o["model"].isin(MODELS + ["Blend"])].pivot(index="model", columns="line", values="D2")
              .reindex(MODELS + ["Blend"]).round(4).to_string())
    print("\nNote: the 2024 score can go DOWN here. The old variable was complete in 2024 but not in 2025,"
          "\nso 2024 flattered it. The leaderboard is the real test of this fix.")
    comp.to_csv(OUT / "comparison.csv", index=False)
    v.to_parquet(OUT / "valid_predictions.parquet", index=False)
    (OUT / "selected_variables.json").write_text(json.dumps(selections, indent=2))

    # ---------------- submissions (new names, earlier files are kept)
    fc.section("SUBMISSIONS")
    sample = fc.rd("sample_submission")
    t = pd.concat(test_all, ignore_index=True)
    for name in ["Blend"] + MODELS:
        path = HERE / f"{SUB_PREFIX}_{name}.parquet"
        sub = sample[["row_id"]].merge(t[["row_id", name]].rename(columns={name: "prediction"}), on="row_id",
                                       how="left")
        assert sub["prediction"].notna().all() and len(sub) == len(sample), f"missing predictions ({name})"
        sub.to_parquet(path, index=False)
        print(f"saved {path.name:42s} mean {sub['prediction'].mean():7.2f}")


if __name__ == "__main__":
    main()
