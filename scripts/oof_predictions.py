"""C021 / C028: one pass per line -- out-of-fold validation predictions, per-line blend weights, and the final
refit on all training years with 2025 test predictions.

Same folds and scoring as `tune_cv.py`: train 2021-22 -> validate 2023, train 2021-23 -> validate 2024.
OOF predictions are on the reported-cost scale (x exposure x completeness x large-loss load).
After the folds every model is refit on all training years (n_estimators = mean fold best_iteration x 1.1 for
models with early stopping) and predicts the 2025 test rows (x exposure x trend x large-loss load), written in the
same format as `train_validate.py`, so that script is no longer needed for submissions.

Usage: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line> [--models glm,gam,lgbm,xgb,cat] [--skip-refit]
Writes: output/oof_<line>.parquet, output/blend_weights_<line>.json, output/pred_<line>_<model>.parquet
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import (OUT, TEST_YEAR, TREND_SHARE, clean, evaluate, features, history_frame,  # noqa: E402
                            line_trend, load, log, models, params)
from tune_cv import FOLDS  # noqa: E402

MODELS = ["glm", "gam", "lgbm", "xgb", "cat"]  # C025: CatBoost added


def train_test_features(line, shared, t, train):
    """C028: features of train and test rows built together (identical categorical levels), as in run_line."""
    test = clean.clean_policies(load(f"test_{line}"))
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    history = history_frame(train, t, year, "policy_id")
    rows = pd.concat([train, test], ignore_index=True)
    x_all = features.build_features(rows, *shared, history=history, line=line)  # C026
    return x_all.iloc[:len(train)], x_all.iloc[len(train):], test


def main(line, names, refit=True):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    train = clean.clean_policies(load(f"policies_{line}"))
    t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
    assert (t["row_id"].to_numpy() == train["row_id"].to_numpy()).all()
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    x, x_test, test = train_test_features(line, shared, t, train)
    log(f"{line}: features {x.shape[1]} cols, train {len(x):,}, test {len(x_test):,}")
    w = t["exposure_eff"].to_numpy()
    y = t["claim_cost_capped"].to_numpy() / w

    frames, folds = [], []
    best_its = {n: [] for n in names}
    for val_year, fit_years in FOLDS:
        fit, val = np.isin(year, fit_years), year == val_year
        load_f = t.loc[fit, "claim_cost"].sum() / t.loc[fit, "claim_cost_capped"].sum()
        actual = t.loc[val, "claim_cost"].to_numpy()
        scale = (t["exposure"] * t["completeness"]).to_numpy()[val] * load_f
        frame = pd.DataFrame({"row_id": t.loc[val, "row_id"].to_numpy(), "val_year": val_year, "actual": actual})
        for name in names:
            t0 = time.time()
            m = models.make(name, **params.for_model(line, name))
            m.fit(x[fit], y[fit], w[fit], valid="inner")  # C027: inner split; GLM/GAM ignore it
            frame[name] = m.predict(x[val]) * scale
            best_its[name].append(getattr(m, "best_iteration", None))
            log(f"{line} {val_year} {name}: d2={evaluate.tweedie_d2(actual, frame[name]):.5f} "
                f"it={best_its[name][-1]} {time.time() - t0:.0f}s")
        frames.append(frame)
        folds.append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in names}})
    pd.concat(frames).to_parquet(os.path.join(OUT, f"oof_{line}.parquet"), index=False)

    weights = evaluate.blend_weights(folds, step=0.05)
    report = evaluate.blend_report(folds, weights, names)
    result = {"line": line, "weights": weights,
              "cv_d2_cal_weighted": float(np.mean(report["weighted"])), "cv_d2_cal_equal": float(np.mean(report["equal"])),
              "cv_d2_cal_single": {n: float(np.mean(v)) for n, v in report["single"].items()},
              # C028: per-fold values [fold 2023, fold 2024] for the "weighted must beat equal on both folds" rule
              "cv_d2_cal_both_folds": {**report["single"], "blend_weighted": report["weighted"],
                                       "blend_equal": report["equal"]},
              "best_iteration": best_its}

    if refit:
        # C028: refit on all training years; same arithmetic as train_validate.run_line
        load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
        trend_test = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR)  # already includes TREND_SHARE
        log(f"{line}: large-loss load test {load_test:.4f}, trend test {trend_test:.4f}")
        n_est = {}
        for name in names:
            t0 = time.time()
            its = best_its[name]
            n_est[name] = int(np.mean(its) * 1.1) if all(its) else None
            full = models.make(name, **{**params.for_model(line, name), **({"n_estimators": n_est[name]} if n_est[name] else {})})
            full.fit(x, y, w, valid=None)
            pred_test = full.predict(x_test) * test["exposure"].to_numpy() * trend_test * load_test
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred_test, "trend_share": TREND_SHARE}).to_parquet(
                os.path.join(OUT, f"pred_{line}_{name}.parquet"), index=False)
            log(f"{line}: {name} refit on all years (n_estimators {n_est[name]}) in {time.time() - t0:.0f}s, "
                f"test mean {pred_test.mean():.2f}")
        result["refit_n_estimators"] = n_est

    with open(os.path.join(OUT, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--models", default=",".join(MODELS), help="comma list of models (default: all five)")
    ap.add_argument("--skip-refit", action="store_true", help="only OOF predictions and blend weights")
    a = ap.parse_args()
    main(a.line, a.models.split(","), refit=not a.skip_refit)
