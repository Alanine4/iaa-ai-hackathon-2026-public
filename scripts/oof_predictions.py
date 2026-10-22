"""C021 / C028: one pass per line -- out-of-fold validation predictions, per-line blend weights, and the final
refit on all training years with 2025 test predictions.

Same folds and scoring as `tune_cv.py`: train 2021-22 -> validate 2023, train 2021-23 -> validate 2024.
OOF predictions are on the reported-cost scale (x exposure x completeness x large-loss load).
After the folds every model is refit on all training years (n_estimators = mean fold best_iteration x 1.1 for
models with early stopping) and predicts the 2025 test rows (x exposure x trend x large-loss load), written in the
same format as `train_validate.py`, so that script is no longer needed for submissions.

C029: `--refit-only` skips the folds and reuses `refit_n_estimators` / `weights` from the existing
blend_weights_<line>.json; `--seeds N` averages N seeds (SEED .. SEED+N-1) of each tree model in the refit.
C030: the both-folds guard is applied here (`evaluate.guarded_weights`): `weights` are the searched weights only if they
beat equal weights on every fold, otherwise equal weights; the grid result is kept as `weights_searched` and the reason
as `rule`. With `--refit-only` the stored searched weights are re-guarded from `cv_d2_cal_both_folds`.

Usage: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line> [--models glm,gam,lgbm,xgb,cat] [--skip-refit]
                                                               [--refit-only] [--seeds N]
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
SEED_KWARG = {"lgbm": "random_state", "xgb": "random_state", "cat": "random_seed"}  # C029


def train_test_features(line, shared, t, train):
    """C028: features of train and test rows built together (identical categorical levels), as in run_line."""
    test = clean.clean_policies(load(f"test_{line}"))
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    history = history_frame(train, t, year, "policy_id")
    rows = pd.concat([train, test], ignore_index=True)
    x_all = features.build_features(rows, *shared, history=history, line=line)  # C026
    return x_all.iloc[:len(train)], x_all.iloc[len(train):], test


def seed_param(name, seed):
    """C029: the seed keyword of a tree model as a params dict; GLM/GAM are deterministic and take none."""
    return {SEED_KWARG[name]: seed} if name in SEED_KWARG else {}


def refit_all(line, names, n_est, x, y, w, x_test, test, t, year, seeds):
    """C028/C029: refit on all training years and write the 2025 test predictions; tree models are averaged over seeds."""
    # same arithmetic as train_validate.run_line
    load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
    trend_test = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR)  # already includes TREND_SHARE
    log(f"{line}: large-loss load test {load_test:.4f}, trend test {trend_test:.4f}")
    scale = test["exposure"].to_numpy() * trend_test * load_test
    for name in names:
        t0 = time.time()
        base = {**params.for_model(line, name), **({"n_estimators": n_est[name]} if n_est[name] else {})}
        n_seeds = seeds if name in SEED_KWARG else 1
        rates = []
        for k in range(n_seeds):
            t1 = time.time()
            full = models.make(name, **{**base, **seed_param(name, models.SEED + k)})
            full.fit(x, y, w, valid=None)
            rates.append(full.predict(x_test))
            log(f"{line}: {name} seed {models.SEED + k} refit in {time.time() - t1:.0f}s, "
                f"test mean {(rates[-1] * scale).mean():.2f}")
        pred_test = np.mean(rates, axis=0) * scale
        pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred_test, "trend_share": TREND_SHARE}).to_parquet(
            os.path.join(OUT, f"pred_{line}_{name}.parquet"), index=False)
        log(f"{line}: {name} refit on all years (n_estimators {n_est[name]}, {n_seeds} seed(s)) in "
            f"{time.time() - t0:.0f}s, test mean {pred_test.mean():.2f}")


def main(line, names, refit=True, refit_only=False, seeds=1):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    train = clean.clean_policies(load(f"policies_{line}"))
    t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
    assert (t["row_id"].to_numpy() == train["row_id"].to_numpy()).all()
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    x, x_test, test = train_test_features(line, shared, t, train)
    log(f"{line}: features {x.shape[1]} cols, train {len(x):,}, test {len(x_test):,}")
    w = t["exposure_eff"].to_numpy()
    y = t["claim_cost_capped"].to_numpy() / w

    if refit_only:
        with open(os.path.join(OUT, f"blend_weights_{line}.json"), encoding="utf-8") as f:
            result = json.load(f)
        n_est = {n: result["refit_n_estimators"][n] for n in names}
        both = result["cv_d2_cal_both_folds"]  # C030: re-apply the guard to the stored searched weights
        searched = result.get("weights_searched", result["weights"])
        result["weights"], result["rule"] = evaluate.guarded_weights(
            {"weighted": both["blend_weighted"], "equal": both["blend_equal"]}, searched, list(searched))
        result["weights_searched"] = searched
    else:
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

        searched = evaluate.blend_weights(folds, step=0.05)
        report = evaluate.blend_report(folds, searched, names)
        weights, rule = evaluate.guarded_weights(report, searched, names)  # C030
        result = {"line": line, "weights": weights, "weights_searched": searched, "rule": rule,
                  "cv_d2_cal_weighted": float(np.mean(report["weighted"])), "cv_d2_cal_equal": float(np.mean(report["equal"])),
                  "cv_d2_cal_single": {n: float(np.mean(v)) for n, v in report["single"].items()},
                  # C028: per-fold values [fold 2023, fold 2024] for the "weighted must beat equal on both folds" rule
                  "cv_d2_cal_both_folds": {**report["single"], "blend_weighted": report["weighted"],
                                           "blend_equal": report["equal"]},
                  "best_iteration": best_its}

    if refit or refit_only:
        if not refit_only:
            n_est = {n: int(np.mean(best_its[n]) * 1.1) if all(best_its[n]) else None for n in names}
        refit_all(line, names, n_est, x, y, w, x_test, test, t, year, seeds)
        result["refit_n_estimators"] = {**result.get("refit_n_estimators", {}), **n_est}
        result["refit_seeds"] = seeds

    with open(os.path.join(OUT, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--models", default=",".join(MODELS), help="comma list of models (default: all five)")
    ap.add_argument("--skip-refit", action="store_true", help="only OOF predictions and blend weights")
    ap.add_argument("--refit-only", action="store_true",
                    help="skip the folds; take refit_n_estimators and weights from the existing blend_weights json")
    ap.add_argument("--seeds", type=int, default=1, help="seeds averaged per tree model in the refit (default 1)")
    a = ap.parse_args()
    assert a.seeds >= 1 and not (a.refit_only and a.skip_refit)
    main(a.line, a.models.split(","), refit=not a.skip_refit, refit_only=a.refit_only, seeds=a.seeds)
