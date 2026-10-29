"""
IAA AI Assisted Hackathon 2026 - Part 1
Improvement 1: fix LightGBM, then tune it.

Run (from the D: environment, after "First code.py" has been run once):
    D:\\iaa-hackathon\\venv\\Scripts\\python.exe "Improved 1 code.py"
Uses the prepared files and helper functions of "First code.py" (nothing there is changed or overwritten).

What changed compared with the LightGBM of "First code.py":
    1. Starting point fixed. The offset log(exposure) was given as init_score, and with an init_score
       LightGBM does not start from the average: every policy started at 1 per year instead of ~300,
       so the trees spent their effort on the overall level. Now init_score = log(exposure) + log(average rate).
    2. Categorical variables are given as integer codes (ordinary splits) instead of LightGBM's
       categorical splits, which overfit badly on many-level variables (make_model has 300 levels,
       previous_insurer 40). Option tried by the tuning: codes in alphabetical order or ordered by the
       training claim cost rate of each level.
    3. Hyperparameters chosen by random search, scored on TWO validation years (mean D2 of
       train 2021-2022 -> 2023 and train 2021-2023 -> 2024) to avoid tuning on the noise of one year.
       The training Tweedie power is tuned too (the metric stays power 1.5).
    4. Option tried by the tuning: written_premium divided by the median premium of its year
       (premiums rise every year, so raw premium thresholds drift).

Outputs (next to this script, the files of "First code.py" are not touched):
    submission_improved1_LightGBM.parquet   improved LightGBM alone
    submission_improved1_Blend.parquet      (GLM + GAM of "First code.py" + improved LightGBM) / 3
    submission_improved1.parquet            best of the two per line on 2024
    prepared/improved1/tuning_<line>.csv    every tuning trial and its scores
    prepared/improved1/best_params.json, comparison.csv
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("first_code", HERE / "First code.py")
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)
lgb = fc.lgb

OUT = fc.OUT / "improved1"
SUB_PREFIX = "submission_improved1"

N_TRIALS = 30                     # random search trials per line (finished trials are kept on re-run)
SEARCH_LR = 0.05                  # learning rate during the search (faster)
FINAL_LR = 0.02                   # learning rate of the final model
VALID_YEARS = [2023, 2024]        # fold "2023" = train 2021-2022, fold "2024" = train 2021-2023
SEARCH_SPACE = {
    "tweedie_variance_power": [1.2, 1.3, 1.4, 1.5],
    "num_leaves": [7, 15, 31, 63, 127],
    "min_data_in_leaf": [100, 300, 1000, 3000, 10000],
    "feature_fraction": [0.3, 0.5, 0.7, 0.9],
    "bagging_fraction": [0.5, 0.7, 0.9, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0, 100.0],
    "max_bin": [63, 255],
    "cat_encoding": ["alphabetical", "ordered"],
    "premium_relative": [False, True],
}
BASE_PARAMS = dict(objective="tweedie", bagging_freq=1, verbose=-1, seed=fc.SEED, num_threads=0)
OPTIONS = ["cat_encoding", "premium_relative"]          # data options, not LightGBM parameters


# ----------------------------------------------------------------------------- data for LightGBM
def category_maps(tr, cats, encoding):
    """Integer code of every level, learned on the training rows only."""
    maps = {}
    for c in cats:
        s = tr[c].astype(str)
        if encoding == "ordered":       # levels sorted by their claim cost rate
            g = pd.DataFrame({"k": s.to_numpy(), "y": tr["claim_cost_capped"].to_numpy(),
                              "e": tr["exposure_adj"].to_numpy()}).groupby("k").sum()
            levels = list((g["y"] / g["e"]).sort_values().index)
        else:
            levels = sorted(s.unique())
        maps[c] = {lvl: i for i, lvl in enumerate(levels)}
    return maps


def features_matrix(df, feats, cats, maps, premium_relative):
    X = pd.DataFrame(index=df.index)
    for f in feats:
        if f in cats:
            X[f] = df[f].astype(str).map(maps[f]).fillna(-1).astype(np.float32)
        else:
            X[f] = df[f].astype(np.float32)
    if premium_relative:
        X["written_premium"] = (df["written_premium"]
                                / df.groupby("year")["written_premium"].transform("median")).astype(np.float32)
    return X


def fit_lgbm(tr, va, feats, cats, cfg, lr, rounds=None):
    """Tweedie LightGBM on capped claim cost, starting point log(exposure) + log(average rate).
    With va: early stopping on va. Returns the booster and a function that predicts claim cost."""
    params = {**BASE_PARAMS, **{k: v for k, v in cfg.items() if k not in OPTIONS}, "learning_rate": lr}
    maps = category_maps(tr, cats, cfg["cat_encoding"])
    base = np.log(tr["claim_cost_capped"].sum() / tr["exposure_adj"].sum())
    dtr = lgb.Dataset(features_matrix(tr, feats, cats, maps, cfg["premium_relative"]),
                      label=tr["claim_cost_capped"].to_numpy(float),
                      init_score=np.log(tr["exposure_adj"].to_numpy(float)) + base)
    if va is None:
        booster = lgb.train(params, dtr, rounds)
    else:
        dva = lgb.Dataset(features_matrix(va, feats, cats, maps, cfg["premium_relative"]),
                          label=va["claim_cost_capped"].to_numpy(float),
                          init_score=np.log(va["exposure_adj"].to_numpy(float)) + base, reference=dtr)
        booster = lgb.train(params, dtr, 20_000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(200, verbose=False)])

    def predict(df, exposure):
        X = features_matrix(df, feats, cats, maps, cfg["premium_relative"])
        return np.exp(booster.predict(X, raw_score=True, num_iteration=booster.best_iteration or None)
                      + base + np.log(exposure))
    return booster, predict


# ----------------------------------------------------------------------------- tuning
def random_configs(n, rng):
    """First config = old LightGBM settings (with the fixes), then random draws from SEARCH_SPACE."""
    old = {k: fc.LGB_PARAMS[k] for k in ["tweedie_variance_power", "num_leaves", "min_data_in_leaf",
                                         "feature_fraction", "bagging_fraction", "lambda_l2"]}
    configs = [{**old, "max_bin": 255, "cat_encoding": "alphabetical", "premium_relative": False}]
    while len(configs) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
        if cfg not in configs:
            configs.append(cfg)
    return configs


def tune(line, fr, feats, cats, loading):
    path = OUT / f"tuning_{line}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    configs = random_configs(N_TRIALS, np.random.default_rng(fc.SEED + fc.LINES.index(line)))
    folds = {y: (fr[fr["year"] < y], fr[fr["year"] == y]) for y in VALID_YEARS}
    rows = done.to_dict("records")
    for i, cfg in enumerate(configs):
        if i < len(rows):
            continue
        t0 = time.time()
        row = {"trial": i, **cfg}
        for y, (tr, va) in folds.items():
            booster, predict = fit_lgbm(tr, va, feats, cats, cfg, SEARCH_LR)
            p = predict(va, va["exposure_adj"].to_numpy()) * loading
            row[f"D2_{y}"] = fc.d2(va["claim_cost"].to_numpy(), p)
            row[f"trees_{y}"] = booster.best_iteration
        row["D2_mean"] = np.mean([row[f"D2_{y}"] for y in VALID_YEARS])
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        best = max(r["D2_mean"] for r in rows)
        print(f"  trial {i:2d}: D2 2023={row['D2_2023']:.4f} 2024={row['D2_2024']:.4f} mean={row['D2_mean']:.4f}"
              f"  (best so far {best:.4f})  [{time.time() - t0:.0f}s]  "
              + " ".join(f"{k}={v}" for k, v in cfg.items()), flush=True)
    res = pd.DataFrame(rows).sort_values("D2_mean", ascending=False)
    best = res.iloc[0]
    cfg = {k: best[k] for k in SEARCH_SPACE}
    cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
    cfg["premium_relative"] = bool(cfg["premium_relative"])
    for k in ["num_leaves", "min_data_in_leaf", "max_bin"]:
        cfg[k] = int(cfg[k])
    print(f"\n  best of {len(res)} trials: mean D2 {best['D2_mean']:.4f} (trial 0 = old settings with fixes: "
          f"{res.loc[res['trial'] == 0, 'D2_mean'].iloc[0]:.4f})")
    return cfg


# ----------------------------------------------------------------------------- GLM / GAM of "First code.py"
def glm_gam_valid(fr, sv, f_feats, s_feats, feats_all):
    """2024 validation predictions of the GLM and GAM, rebuilt exactly as in "First code.py"
    (same selected variables, same sample and smoothing choice). Test predictions are read from its files."""
    tr, va = fr[fr["split"] == "train"], fr[fr["split"] == "valid"]
    sv_tr = sv[sv["split"] == "train"]
    expo = va["exposure_adj"].to_numpy()
    glm = fc.fit_freq_sev(tr, sv_tr, f_feats, s_feats)
    samp = tr.sample(min(len(tr), fc.SELECTION_SAMPLE), random_state=fc.SEED)
    # Gamma dispersion from the model with all candidate variables, as in the variable selection
    y_s = sv_tr["net_cost_capped"].to_numpy(float)
    X_all, _ = fc.Design(sv_tr, feats_all, np.ones(len(sv_tr))).matrix(sv_tr)
    full = fc.fit_glm(X_all, y_s, 2)
    phi_s = fc.pearson_dispersion(y_s, full["mu"], 2, full["beta"].size)
    lam_f = fc.choose_lambda(samp, f_feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float),
                             1, np.log(samp["exposure_adj"].to_numpy()), 1.0, "GAM frequency")
    lam_s = fc.choose_lambda(sv_tr, s_feats, np.ones(len(sv_tr)), sv_tr["net_cost_capped"].to_numpy(float),
                             2, None, phi_s, "GAM severity")
    gam = fc.fit_freq_sev(tr, sv_tr, f_feats, s_feats, smooth=True, lam_f=lam_f * len(tr) / len(samp), lam_s=lam_s)
    return fc.predict_freq_sev(glm, va, expo), fc.predict_freq_sev(gam, va, expo)


# ----------------------------------------------------------------------------- driver
def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((fc.OUT / "meta.json").read_text())
    selections = json.loads((fc.MODELS / "selected_variables.json").read_text())
    first = pd.read_csv(fc.MODELS / "model_comparison.csv").set_index(["line", "model"])
    first_trend = {k: v for k, v in (first["D2_with_trend"] > first["D2"]).items()}
    best_params, valid_all, test_all, comparison = {}, [], [], []

    for line in fc.LINES:
        t0 = time.time()
        fc.section(f"LIGHTGBM TUNING {line.upper()}")
        info = meta["lines"][line]
        feats = [f for f in info["features"] if f not in fc.LEAKY_FEATURES]
        cats = [c for c in info["categorical"] if c in feats]
        loading = info["large_loss_loading"]
        fr = pd.read_parquet(fc.OUT / f"freq_{line}.parquet")
        sv = pd.read_parquet(fc.OUT / f"sev_{line}.parquet")
        te = pd.read_parquet(fc.OUT / f"test_{line}.parquet")
        fr["exposure_adj"] = fr["exposure_adj"].clip(lower=1e-4)

        cfg = tune(line, fr, feats, cats, loading)

        # ---------------- final learning rate: 2024 validation, trees by early stopping
        tr, va = fr[fr["split"] == "train"], fr[fr["split"] == "valid"]
        y_va = va["claim_cost"].to_numpy()
        booster, predict = fit_lgbm(tr, va, feats, cats, cfg, FINAL_LR)
        n_trees = booster.best_iteration
        p_lgb = predict(va, va["exposure_adj"].to_numpy()) * loading
        trend_va, yearly = fc.severity_trend(sv[sv["split"] == "train"], fc.VALID_YEAR)
        lgb_trend = fc.d2(y_va, p_lgb * trend_va) > fc.d2(y_va, p_lgb)
        p_lgb = p_lgb * (trend_va if lgb_trend else 1.0)
        print(f"  final LightGBM (learning rate {FINAL_LR}): {n_trees} trees, 2024 D2 {fc.d2(y_va, p_lgb):.4f}"
              f" (inflation trend {'on' if lgb_trend else 'off'})")

        # ---------------- blend with the GLM and GAM of "First code.py"
        p_glm, p_gam = glm_gam_valid(fr, sv, selections[line]["frequency"], selections[line]["severity"], feats)
        p_glm = p_glm * loading * (trend_va if first_trend[(line, "GLM")] else 1.0)
        p_gam = p_gam * loading * (trend_va if first_trend[(line, "GAM")] else 1.0)
        p_blend = (p_glm + p_gam + p_lgb) / 3
        preds = {"GLM": p_glm, "GAM": p_gam, "LightGBM improved": p_lgb, "Blend improved": p_blend}
        best = max(["LightGBM improved", "Blend improved"], key=lambda k: fc.d2(y_va, preds[k]))
        preds["Best improved"] = preds[best]
        valid_all.append(pd.DataFrame({"line": line, "y": y_va, **preds}))
        for name, p in preds.items():
            comparison.append({"line": line, "model": name, "D2": fc.d2(y_va, p), "pred/actual": p.sum() / y_va.sum()})
        for name in ["LightGBM", "Blend"]:
            r = first.loc[(line, name)]
            comparison.append({"line": line, "model": f"{name} (First code)",
                               "D2": max(r["D2"], r["D2_with_trend"]), "pred/actual": r["pred/actual"]})
        print(f"  2024 D2: LightGBM {fc.d2(y_va, p_lgb):.4f} | Blend {fc.d2(y_va, p_blend):.4f}"
              f"  (First code: LightGBM {max(first.loc[(line, 'LightGBM'), ['D2', 'D2_with_trend']]):.4f},"
              f" Blend {max(first.loc[(line, 'Blend'), ['D2', 'D2_with_trend']]):.4f})  -> best: {best}")

        # ---------------- refit on 2021-2024 and predict 2025
        n_full = int(n_trees * len(fr) / len(tr)) + 1          # more data -> proportionally more trees
        _, predict = fit_lgbm(fr, None, feats, cats, cfg, FINAL_LR, rounds=n_full)
        t_lgb = predict(te, te["exposure"].to_numpy()) * loading
        if lgb_trend:
            t_lgb *= fc.severity_trend(sv, 2025)[0]
        test_all.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), "LightGBM": t_lgb, "line": line,
                                      "best": best}))
        best_params[line] = {**cfg, "learning_rate": FINAL_LR, "n_trees_2021_2023": n_trees,
                             "n_trees_final": n_full, "inflation_trend": bool(lgb_trend)}
        print(f"  2025 LightGBM total {t_lgb.sum():,.0f}   [{(time.time() - t0) / 60:.1f} min]")

    # ---------------- comparison, all lines together (like the leaderboard)
    fc.section("COMPARISON - 2024 validation D2")
    v = pd.concat(valid_all)
    for name in ["GLM", "GAM", "LightGBM improved", "Blend improved", "Best improved"]:
        comparison.append({"line": "ALL", "model": name, "D2": fc.d2(v["y"], v[name]),
                           "pred/actual": v[name].sum() / v["y"].sum()})
    for name, key in [("LightGBM (First code)", "LightGBM"), ("Blend (First code)", "Best")]:
        r = first.loc[("ALL", key)]
        comparison.append({"line": "ALL", "model": name, "D2": max(r["D2"], r["D2_with_trend"]),
                           "pred/actual": r["pred/actual"]})
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT / "comparison.csv", index=False)
    order = ["LightGBM (First code)", "LightGBM improved", "GLM", "GAM", "Blend (First code)", "Blend improved",
             "Best improved"]
    print(comp.pivot(index="model", columns="line", values="D2").reindex(order).round(4).to_string())
    (OUT / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # ---------------- submissions (new names, "First code.py" files are kept)
    fc.section("SUBMISSIONS")
    sample = fc.rd("sample_submission")
    t = pd.concat(test_all)
    old = {m: pd.read_parquet(HERE / f"submission_{m}.parquet").set_index("row_id")["prediction"]
           for m in ["GLM", "GAM"]}
    t["Blend"] = (old["GLM"].reindex(t["row_id"]).to_numpy() + old["GAM"].reindex(t["row_id"]).to_numpy()
                  + t["LightGBM"].to_numpy()) / 3
    t["Best"] = np.where(t["best"] == "Blend improved", t["Blend"], t["LightGBM"])
    for name, path in [("LightGBM", f"{SUB_PREFIX}_LightGBM.parquet"), ("Blend", f"{SUB_PREFIX}_Blend.parquet"),
                       ("Best", f"{SUB_PREFIX}.parquet")]:
        sub = sample[["row_id"]].merge(t[["row_id", name]].rename(columns={name: "prediction"}), on="row_id",
                                       how="left")
        assert sub["prediction"].notna().all() and len(sub) == len(sample), f"missing predictions ({name})"
        sub.to_parquet(HERE / path, index=False)
        print(f"saved {path:40s} mean {sub['prediction'].mean():7.2f}")


if __name__ == "__main__":
    main()
