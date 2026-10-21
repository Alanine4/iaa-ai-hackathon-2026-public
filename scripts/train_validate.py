"""Validate GLM / LightGBM / XGBoost per line (train 2021-2023, validate 2024), refit on 2021-2024,
predict 2025 and write Kaggle submission files.

Validation compares reported 2024 cost with prediction x completeness, so undeveloped claims are not
counted against the model (C006). Requires output/target_<line>.parquet from scripts/build_target.py.

Usage: .venv\\Scripts\\python.exe scripts\\train_validate.py [glm,lgbm,xgb]
Writes: docs/validation_results.md, output/pred_<line>_<model>.parquet, output/submissions/sub_<model>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from aiiaa import clean, evaluate, features, models, params  # noqa: E402

DATA = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-1")
OUT = os.path.join(ROOT, "output")
LINES = ["motor", "buildings", "contents"]
VALID_YEAR, TEST_YEAR = 2024, 2025
TREND_SHARE = 0.5  # C019: half of the log-linear trend (public board: none 0.12376, half 0.12464, full 0.12453)


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def line_trend(t, years, fit_years, target_year):
    """C009: trend from developed pure premium of fit_years."""
    s = t.assign(year=years).loc[lambda d: d["year"].isin(fit_years)].groupby("year").agg(
        ult=("claim_cost_ult", "sum"), exposure=("exposure", "sum"))
    full = evaluate.trend_factor(s.index, s["ult"] / s["exposure"], s["exposure"], target_year)
    return full ** TREND_SHARE


def run_line(line, names, shared):
    train = clean.clean_policies(load(f"policies_{line}"))
    test = clean.clean_policies(load(f"test_{line}"))
    t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
    assert (t["row_id"].to_numpy() == train["row_id"].to_numpy()).all()
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()

    history = pd.DataFrame({"policy_id": train["policy_id"], "year": year, "claim_count": t["claim_count"],
                            "claim_cost": t["claim_cost"], "exposure": train["exposure"]})
    history = history.groupby(["policy_id", "year"], as_index=False).sum()
    rows = pd.concat([train, test], ignore_index=True)
    x_all = features.build_features(rows, *shared, history=history)
    x, x_test = x_all.iloc[:len(train)], x_all.iloc[len(train):]
    log(f"{line}: features {x.shape[1]} cols, train {len(x):,}, test {len(x_test):,}")

    # C010: rate target on completeness-weighted exposure (equivalent to developed cost / exposure)
    # C014: models learn capped cost; the capped excess is spread back with a per-line load factor
    w = t["exposure_eff"].to_numpy()
    y = t["claim_cost_capped"].to_numpy() / w
    fit_mask, val_mask = year < VALID_YEAR, year == VALID_YEAR
    load_val = t.loc[fit_mask, "claim_cost"].sum() / t.loc[fit_mask, "claim_cost_capped"].sum()
    load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
    log(f"{line}: large-loss load factor validation {load_val:.4f}, test {load_test:.4f}")
    trend_val = line_trend(t, year, [2021, 2022, 2023], VALID_YEAR)
    trend_test = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR)
    log(f"{line}: trend factor validation {trend_val:.4f}, test {trend_test:.4f}")

    actual = t.loc[val_mask, "claim_cost"].to_numpy()
    expo_val = t.loc[val_mask, "exposure"].to_numpy()
    comp_val = t.loc[val_mask, "completeness"].to_numpy()
    results, val_preds = [], {}
    for name in names:
        t0 = time.time()
        tuned = params.for_model(line, name)  # C018
        m = models.make(name, **tuned)
        m.fit(x[fit_mask], y[fit_mask], w[fit_mask], valid=(x[val_mask], y[val_mask], w[val_mask]))
        rate = m.predict(x[val_mask])
        pred = rate * expo_val * comp_val * load_val  # expected reported cost at valuation date
        val_preds[name] = pred * trend_val
        res = {"line": line, "model": name,
               "d2_no_trend": evaluate.tweedie_d2(actual, pred),
               "d2": evaluate.tweedie_d2(actual, pred * trend_val),
               "pred/actual": (pred * trend_val).sum() / actual.sum(),
               "iterations": getattr(m, "best_iteration", None), "fit_sec": round(time.time() - t0)}
        results.append(res)
        log(f"{line}: {res}")

        t0 = time.time()
        n_full = int(m.best_iteration * 1.1) if getattr(m, "best_iteration", None) else None
        full = models.make(name, **{**tuned, **({"n_estimators": n_full} if n_full else {})})
        full.fit(x, y, w)
        pred_test = full.predict(x_test) * test["exposure"].to_numpy() * trend_test * load_test
        pd.DataFrame({"row_id": test["row_id"], "prediction": pred_test, "trend_share": TREND_SHARE}).to_parquet(
            os.path.join(OUT, f"pred_{line}_{name}.parquet"), index=False)
        log(f"{line}: {name} refit on all years in {time.time() - t0:.0f}s, test mean {pred_test.mean():.2f}")

    if len(names) > 1:
        blend = np.mean([val_preds[n] for n in names], axis=0)
        results.append({"line": line, "model": "blend", "d2_no_trend": np.nan, "d2": evaluate.tweedie_d2(actual, blend),
                        "pred/actual": blend.sum() / actual.sum(), "iterations": None, "fit_sec": None})
    return results, actual, val_preds


def main(names, lines=LINES):
    os.makedirs(os.path.join(OUT, "submissions"), exist_ok=True)
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    all_results, actual_all, preds_all = [], [], {n: [] for n in names}
    for line in lines:
        res, actual, val_preds = run_line(line, names, shared)
        all_results += res
        actual_all.append(actual)
        for n in names:
            preds_all[n].append(val_preds[n])
    if list(lines) != LINES:
        # Partial run: refresh only the per-line prediction files, keep the full results table and submissions
        log("\n" + pd.DataFrame(all_results).round(4).to_string(index=False))
        return

    actual = np.concatenate(actual_all)
    for n in names:
        p = np.concatenate(preds_all[n])
        all_results.append({"line": "ALL", "model": n, "d2_no_trend": np.nan, "d2": evaluate.tweedie_d2(actual, p),
                            "pred/actual": p.sum() / actual.sum(), "iterations": None, "fit_sec": None})
    if len(names) > 1:
        p = np.mean([np.concatenate(preds_all[n]) for n in names], axis=0)
        all_results.append({"line": "ALL", "model": "blend", "d2_no_trend": np.nan, "d2": evaluate.tweedie_d2(actual, p),
                            "pred/actual": p.sum() / actual.sum(), "iterations": None, "fit_sec": None})

    table = pd.DataFrame(all_results)
    with open(os.path.join(ROOT, "docs", "validation_results.md"), "w", encoding="utf-8") as f:
        f.write("# Validation results (train 2021-2023, validate 2024)\n\n")
        f.write("Metric: Tweedie D2 (power 1.5) of reported 2024 cost vs prediction x completeness. "
                "`d2` includes the trend factor (C009). Generated by `scripts/train_validate.py`.\n\n")
        f.write(table.round(4).to_markdown(index=False) if hasattr(table, "to_markdown") else table.round(4).to_string(index=False))
        f.write("\n")
    log("\n" + table.round(4).to_string(index=False))

    sample = load("sample_submission")
    for n in names + (["blend"] if len(names) > 1 else []):
        if n == "blend":
            parts = [pd.read_parquet(os.path.join(OUT, f"pred_{l}_{m}.parquet")).set_index("row_id")["prediction"]
                     for l in LINES for m in names]
            sub = pd.concat(parts).groupby(level=0).mean()
        else:
            sub = pd.concat([pd.read_parquet(os.path.join(OUT, f"pred_{l}_{n}.parquet")) for l in LINES]).set_index("row_id")["prediction"]
        sub = sub.reindex(sample["row_id"])
        assert sub.notna().all() and (sub >= 0).all() and len(sub) == len(sample), f"invalid submission {n}"
        path = os.path.join(OUT, "submissions", f"sub_{n}.csv")
        sub.rename("prediction").reset_index().to_csv(path, index=False)
        log(f"wrote {path}: rows {len(sub):,}, mean {sub.mean():.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["glm", "gam", "lgbm", "xgb"],
         sys.argv[2].split(",") if len(sys.argv) > 2 else LINES)
