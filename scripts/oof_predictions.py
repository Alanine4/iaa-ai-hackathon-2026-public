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
C031: `--te <column>` adds `<column>_te`, the smoothed log risk relativity of the raw code (features.target_encode):
fit rows get the 5-fold out-of-fold encoding within the fit rows, validation / test rows the encoding fitted on all fit
rows. Off by default; the json records the column in `target_encoding` and `--refit-only` requires the same `--te`.

C036 (`--market emerging`): emerging-market pass (`emerging_main`). Validation is always on emerging rows: (a) time
fold, train 2023 -> score 2024; (b) grouped 5-fold CV over all 2023-2024 rows, groups = policy_id, seed 42. Both are
scored like the developed folds (reported cost vs prediction x exposure x completeness x large-loss load, level-
calibrated d2_cal). Blend weights are searched on the grouped folds and kept only if they beat equal weights on the time
fold AND on the grouped-CV mean. `--train-set pooled` (C038) adds every developed training row (harmonised, own target
and history, sample weight x `--developed-weight`) to every fit set; the fitted level is then re-balanced on the
emerging fit rows. `--trend none|developed-half` (C037) sets the 2025 factor. Outputs go to output/emerging/ (emerging
only) or output/emerging/pooled_w<w>/ (pooled).

Usage: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line> [--models glm,gam,lgbm,xgb,cat] [--skip-refit]
                                                               [--refit-only] [--seeds N] [--te make_model]
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
                            line_trend, load, log, markets, models, params)
from tune_cv import FOLDS  # noqa: E402

MODELS = ["glm", "gam", "lgbm", "xgb", "cat"]  # C025: CatBoost added
SEED_KWARG = {"lgbm": "random_state", "xgb": "random_state", "cat": "random_seed"}  # C029
EMERGING_TIME_FOLD = (2024, [2023])  # C036: train 2023 -> score 2024
GROUP_FOLDS, GROUP_SEED = 5, 42  # C036: grouped CV on policy_id
EMERGING_MIN_FREQUENCY = 30  # C036: GLM/GAM one-hot threshold for emerging-only fits (default 100)


def train_test_features(line, shared, t, train):
    """C028: features of train and test rows built together (identical categorical levels), as in run_line."""
    test = clean.clean_policies(load(f"test_{line}"))
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    history = history_frame(train, t, year, "policy_id")
    rows = pd.concat([train, test], ignore_index=True)
    x_all = features.build_features(rows, *shared, history=history, line=line)  # C026
    return x_all.iloc[:len(train)], x_all.iloc[len(train):], test


def with_target_encoding(x_fit, x_apply, key_fit, key_apply, y_fit, w_fit, col):
    """C031: copies of x_fit / x_apply with `<col>_te` (out-of-fold for the fit rows, fitted on all fit rows for the
    apply rows), plus the spread of the encoding across the codes seen in the fit rows."""
    te = f"{col}_te"
    codes = pd.Series(key_fit).dropna().unique()
    enc = features.target_encode(key_fit, y_fit, w_fit, codes)
    spread = {"n_codes": int(len(codes)), "min": float(enc.min()), "max": float(enc.max()), "std": float(enc.std())}
    return (x_fit.assign(**{te: features.target_encode_oof(key_fit, y_fit, w_fit)}),
            x_apply.assign(**{te: features.target_encode(key_fit, y_fit, w_fit, key_apply)}), spread)


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


def main(line, names, refit=True, refit_only=False, seeds=1, te=None):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    train = clean.clean_policies(load(f"policies_{line}"))
    t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
    assert (t["row_id"].to_numpy() == train["row_id"].to_numpy()).all()
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    x, x_test, test = train_test_features(line, shared, t, train)
    log(f"{line}: features {x.shape[1]} cols, train {len(x):,}, test {len(x_test):,}")
    w = t["exposure_eff"].to_numpy()
    y = t["claim_cost_capped"].to_numpy() / w
    if te:  # C031: raw code column of the cleaned policy tables, positionally aligned with x / x_test
        key, key_test = train[te].to_numpy(dtype=object), test[te].to_numpy(dtype=object)

    if refit_only:
        with open(os.path.join(OUT, f"blend_weights_{line}.json"), encoding="utf-8") as f:
            result = json.load(f)
        assert result.get("target_encoding", {}).get("column") == te, "C031: --te must match the stored OOF run"
        n_est = {n: result["refit_n_estimators"][n] for n in names}
        both = result["cv_d2_cal_both_folds"]  # C030: re-apply the guard to the stored searched weights
        searched = result.get("weights_searched", result["weights"])
        result["weights"], result["rule"] = evaluate.guarded_weights(
            {"weighted": both["blend_weighted"], "equal": both["blend_equal"]}, searched, list(searched))
        result["weights_searched"] = searched
    else:
        frames, folds = [], []
        best_its = {n: [] for n in names}
        te_spread = []
        for val_year, fit_years in FOLDS:
            fit, val = np.isin(year, fit_years), year == val_year
            load_f = t.loc[fit, "claim_cost"].sum() / t.loc[fit, "claim_cost_capped"].sum()
            actual = t.loc[val, "claim_cost"].to_numpy()
            scale = (t["exposure"] * t["completeness"]).to_numpy()[val] * load_f
            frame = pd.DataFrame({"row_id": t.loc[val, "row_id"].to_numpy(), "val_year": val_year, "actual": actual})
            x_fit, x_val = x[fit], x[val]
            if te:
                x_fit, x_val, spread = with_target_encoding(x_fit, x_val, key[fit], key[val], y[fit], w[fit], te)
                te_spread.append(spread)
                log(f"{line} {val_year} {te}_te across fit codes: {spread}")
            for name in names:
                t0 = time.time()
                m = models.make(name, **params.for_model(line, name))
                m.fit(x_fit, y[fit], w[fit], valid="inner")  # C027: inner split; GLM/GAM ignore it
                frame[name] = m.predict(x_val) * scale
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
        if te:
            result["target_encoding"] = {"column": te, "spread_by_fold": te_spread}  # C031

    if refit or refit_only:
        if not refit_only:
            n_est = {n: int(np.mean(best_its[n]) * 1.1) if all(best_its[n]) else None for n in names}
        x_ref, x_test_ref = x, x_test
        if te:
            x_ref, x_test_ref, spread = with_target_encoding(x, x_test, key, key_test, y, w, te)
            log(f"{line} refit {te}_te across training codes: {spread}")
        refit_all(line, names, n_est, x_ref, y, w, x_test_ref, test, t, year, seeds)
        result["refit_n_estimators"] = {**result.get("refit_n_estimators", {}), **n_est}
        result["refit_seeds"] = seeds

    with open(os.path.join(OUT, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


def run_dir(train_set, dev_weight):
    """C036/C038: output folder of an emerging run."""
    base = markets.get("emerging").out_dir
    return base if train_set == "emerging" else os.path.join(base, f"pooled_w{dev_weight:g}")


def market_features(market, line, with_test):
    """C035/C036: harmonised features of one market's training rows (and test rows), built within that market: its own
    customers, addresses, weather, target table (output/ or output/emerging/) and per-policy history."""
    customers = clean.fix_customer_id(load("customers", market))
    addresses, weather = load("addresses", market), load("weather", market)
    train = clean.clean_policies(load(f"policies_{line}", market))
    t = pd.read_parquet(os.path.join(markets.get(market).out_dir, f"target_{line}.parquet"))
    assert (t["row_id"].to_numpy() == train["row_id"].to_numpy()).all()
    year = pd.to_datetime(train["period_start"]).dt.year.to_numpy()
    history = history_frame(train, t, year, "policy_id")
    test = clean.clean_policies(load(f"test_{line}", market)) if with_test else None
    rows = pd.concat([train, test], ignore_index=True) if with_test else train
    rows, cust = features.harmonise(rows, customers, market)
    x = features.build_features(rows, cust, addresses, weather, history=history, line=line)  # C026 drops
    return x, train, test, t, year


def emerging_design(line, train_set, dev_weight):
    """C036/C038: one feature frame for every row the emerging pass needs, with per-row arrays.

    part: "dev" (developed training rows, pooled only), "em" (emerging training rows), "test" (emerging 2025 rows).
    y = capped cost / exposure_eff, w = exposure_eff (x dev_weight on developed rows).
    """
    x_em, train, test, t, year = market_features("emerging", line, with_test=True)
    frames = [x_em]
    part = np.array(["em"] * len(train) + ["test"] * len(test), dtype=object)
    y = np.concatenate([t["claim_cost_capped"].to_numpy() / t["exposure_eff"].to_numpy(), np.zeros(len(test))])
    w = np.concatenate([t["exposure_eff"].to_numpy(), np.zeros(len(test))])
    if train_set == "pooled":
        x_dev, _, _, t_dev, _ = market_features("developed", line, with_test=False)
        frames = [x_dev, x_em]
        part = np.concatenate([np.array(["dev"] * len(x_dev), dtype=object), part])
        y = np.concatenate([t_dev["claim_cost_capped"].to_numpy() / t_dev["exposure_eff"].to_numpy(), y])
        w = np.concatenate([t_dev["exposure_eff"].to_numpy() * dev_weight, w])  # C038: developed weight multiplier
    x = features.concat_markets(frames) if len(frames) > 1 else x_em.reset_index(drop=True)
    return x, part, y, w, train, test, t, year


def emerging_model(line, name, train_set, extra=None):
    """C036: params.for_model("emerging-<line>") (defaults unless tuned); emerging-only GLM/GAM pool rare levels at 30."""
    kw = params.for_model(f"emerging-{line}", name)
    if train_set == "emerging" and name in ("glm", "gam"):
        kw["min_frequency"] = EMERGING_MIN_FREQUENCY
    return models.make(name, **{**kw, **(extra or {})})


def fit_emerging(m, x, y, w, fit, em_fit, train_set, valid):
    """Fit on rows `fit`; for the pooled set re-balance the level on the emerging rows of the fit set (C038).
    Returns the re-balance factor (1.0 for emerging-only fits, which Balanced already balances)."""
    m.fit(x[fit], y[fit], w[fit], valid=valid)
    if train_set != "pooled":
        return 1.0
    factor = float(np.sum(y[em_fit] * w[em_fit]) / np.sum(m.predict(x[em_fit]) * w[em_fit]))
    m.balance_ *= factor
    return factor


def scores(actual, pred):
    return {"d2": float(evaluate.tweedie_d2(actual, pred)),
            "d2_cal": float(evaluate.tweedie_d2(actual, pred * actual.sum() / pred.sum())),
            "pred_actual": float(pred.sum() / actual.sum())}


def emerging_schemes(train, year):
    """C036/C039: validation schemes over the emerging training rows as (scheme, k, fit_mask, val_mask): the time fold
    (fit EMERGING_TIME_FOLD years, validate its year) and GROUP_FOLDS grouped folds on policy_id."""
    group = evaluate.group_folds(train["policy_id"].to_numpy(), GROUP_FOLDS, GROUP_SEED)
    val_year, fit_years = EMERGING_TIME_FOLD
    schemes = [("time", 0, np.isin(year, fit_years), year == val_year)]
    schemes += [("group", k, group != k, group == k) for k in range(GROUP_FOLDS)]
    return schemes


def emerging_main(line, names, train_set="emerging", dev_weight=0.3, trend="none", refit=True):
    out_dir = run_dir(train_set, dev_weight)
    os.makedirs(out_dir, exist_ok=True)
    x, part, y, w, train, test, t, year = emerging_design(line, train_set, dev_weight)
    is_dev, is_em = part == "dev", part == "em"
    em_idx = np.flatnonzero(is_em)  # positions of the emerging training rows, in the order of t
    log(f"emerging {line} [{train_set}]: features {x.shape[1]} cols, developed fit rows {is_dev.sum():,}, "
        f"emerging rows {is_em.sum():,}, test rows {(part == 'test').sum():,}")
    schemes = emerging_schemes(train, year)
    comp_expo = (t["exposure"] * t["completeness"]).to_numpy()

    frames, folds = [], {"time": [], "group": []}
    best_its = {n: {"time": [], "group": []} for n in names}
    rebal = {n: {"time": [], "group": []} for n in names}
    for scheme, k, em_fit_mask, em_val_mask in schemes:
        em_fit = np.zeros(len(x), dtype=bool)
        em_fit[em_idx[em_fit_mask]] = True
        fit = is_dev | em_fit
        val_rows = em_idx[em_val_mask]
        load_f = t.loc[em_fit_mask, "claim_cost"].sum() / t.loc[em_fit_mask, "claim_cost_capped"].sum()
        actual = t.loc[em_val_mask, "claim_cost"].to_numpy()
        scale = comp_expo[em_val_mask] * load_f
        frame = pd.DataFrame({"row_id": t.loc[em_val_mask, "row_id"].to_numpy(), "scheme": scheme, "fold": k,
                              "actual": actual})
        for name in names:
            t0 = time.time()
            m = emerging_model(line, name, train_set)
            rebal[name][scheme].append(fit_emerging(m, x, y, w, fit, em_fit, train_set, valid="inner"))  # C027
            frame[name] = m.predict(x.iloc[val_rows]) * scale
            best_its[name][scheme].append(getattr(m, "best_iteration", None))
            s = scores(actual, frame[name].to_numpy())
            log(f"emerging {line} {scheme} {k} {name}: d2={s['d2']:.5f} d2_cal={s['d2_cal']:.5f} "
                f"p/a={s['pred_actual']:.3f} it={best_its[name][scheme][-1]} {time.time() - t0:.0f}s")
        frames.append(frame)
        folds[scheme].append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in names}})
    pd.concat(frames).to_parquet(os.path.join(out_dir, f"oof_{line}.parquet"), index=False)

    searched = evaluate.blend_weights(folds["group"], step=0.05)  # C036: searched on the grouped folds
    rep = {s: evaluate.blend_report(folds[s], searched, names) for s in folds}
    guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
    weights, rule = evaluate.guarded_weights(guard, searched, names)  # index 0 time fold, index 1 grouped mean
    cv_time = {n: {**scores(folds["time"][0]["actual"], folds["time"][0]["preds"][n]),
                   "best_iteration": best_its[n]["time"][0]} for n in names}
    cv_group = {n: {"d2_cal_folds": rep["group"]["single"][n], "d2_cal_mean": float(np.mean(rep["group"]["single"][n])),
                    "d2_mean": float(np.mean([evaluate.tweedie_d2(f["actual"], f["preds"][n]) for f in folds["group"]])),
                    "best_iteration": best_its[n]["group"]} for n in names}
    for b, key in (("weighted", "blend_searched"), ("equal", "blend_equal")):
        cv_time[key] = {"d2_cal": rep["time"][b][0]}
        cv_group[key] = {"d2_cal_folds": rep["group"][b], "d2_cal_mean": float(np.mean(rep["group"][b]))}
    result = {"line": line, "market": "emerging", "train_set": train_set,
              "developed_weight": dev_weight if train_set == "pooled" else None,
              "weights": weights, "weights_searched": searched, "rule": rule,
              "guard_scores": {"index": ["time fold 2023->2024", f"grouped {GROUP_FOLDS}-fold mean"], **guard},
              "cv_time": cv_time, "cv_group": cv_group, "rebalance_factor": rebal}

    if refit:
        n_est = {n: int(np.mean(best_its[n]["group"]) * 1.1) if all(best_its[n]["group"]) else None for n in names}
        factor = markets.get("emerging").trends[trend][line]  # C037
        load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
        fit = is_dev | is_em
        test_rows = np.flatnonzero(part == "test")
        scale = test["exposure"].to_numpy() * factor * load_test
        log(f"emerging {line}: large-loss load test {load_test:.4f}, trend {trend} {factor:.4f}")
        result["refit_rebalance_factor"] = {}
        for name in names:
            t0 = time.time()
            m = emerging_model(line, name, train_set, {"n_estimators": n_est[name]} if n_est[name] else None)
            result["refit_rebalance_factor"][name] = fit_emerging(m, x, y, w, fit, is_em, train_set, valid=None)
            pred = m.predict(x.iloc[test_rows]) * scale
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                          "trend_factor": factor}).to_parquet(os.path.join(out_dir, f"pred_{line}_{name}.parquet"),
                                                              index=False)
            log(f"emerging {line}: {name} refit (n_estimators {n_est[name]}) in {time.time() - t0:.0f}s, "
                f"test mean {pred.mean():.2f}")
        result.update(refit_n_estimators=n_est, trend=trend, trend_factor=factor, load_test=float(load_test))

    with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--market", default=None, choices=sorted(markets.MARKETS),
                    help="C034: default from AIIAA_MARKET, else developed")
    ap.add_argument("--train-set", default="emerging", choices=["emerging", "pooled"], help="C038 (emerging market)")
    ap.add_argument("--developed-weight", type=float, default=0.3, help="C038: weight multiplier on developed rows")
    ap.add_argument("--trend", default="none", choices=["none", "developed-half"], help="C037 (emerging market)")
    ap.add_argument("--models", default=",".join(MODELS), help="comma list of models (default: all five)")
    ap.add_argument("--skip-refit", action="store_true", help="only OOF predictions and blend weights")
    ap.add_argument("--refit-only", action="store_true",
                    help="skip the folds; take refit_n_estimators and weights from the existing blend_weights json")
    ap.add_argument("--seeds", type=int, default=1, help="seeds averaged per tree model in the refit (default 1)")
    ap.add_argument("--te", default=None, help="C031: raw code column to target-encode, e.g. make_model (default off)")
    a = ap.parse_args()
    assert a.seeds >= 1 and not (a.refit_only and a.skip_refit)
    if markets.get(a.market).name == "emerging":
        assert not (a.refit_only or a.te or a.seeds != 1), "C036: --refit-only, --te and --seeds are developed-only"
        emerging_main(a.line, a.models.split(","), a.train_set, a.developed_weight, a.trend, refit=not a.skip_refit)
    else:
        main(a.line, a.models.split(","), refit=not a.skip_refit, refit_only=a.refit_only, seeds=a.seeds, te=a.te)
