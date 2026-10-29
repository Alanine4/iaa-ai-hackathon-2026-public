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
C042: `--glm-interactions` (pooled, `--models glm` only) appends the GLM_INTERACTIONS market interaction columns
(features.market_interactions) and writes to output/emerging/pooled_w<w>_glmx/.
C044: every emerging model and blend is also scored on two test-mix views (`evaluate.nb_views`): `d2_cal_testmix`
(validation rows importance-weighted so new business carries the 2025 test file's NB exposure share of the line) and
`d2_cal_nb` (NB rows only, with their claim count); the OOF file keeps new_business, exposure and claim_count. Weights
are still searched and guarded on the raw grouped folds; `guard_scores.testmix` is reported only.
C045: `--nb-weight auto|<factor>` multiplies the fit weight of new-business rows (both markets) in every fit (`auto`:
NB rows get the test NB exposure share of the fit weight); the C038 re-balance uses the same weights.
`--inner-emerging` (pooled) draws the C027 early-stopping holdout from the emerging fit rows only. Run folders get the
suffix `_nb` (both options), `_nbw` (weighting only) or `_ie` (holdout only).
C046: `params.TUNED["emerging-<line>"]` (tuned on emerging-only data) applies to emerging-only fits only; pooled fits
keep the model defaults (C039). `--tag <name>` appends `_<name>` to the run folder; an emerging-only run with a tag goes
to output/emerging/emerging_<name>/ (the untagged emerging-only folder is output/emerging/ itself).
C047: `--rows nb` (pooled) restricts every fit set (folds and refit) to the new-business rows of both markets; the C038
re-balance uses the emerging NB fit rows; validation still scores every emerging validation row and the refit predicts
every test row. Run folder suffix `_nbrows`.

C057: `--market emerging --refit-only [--seeds N]` re-fits the emerging models without the folds, from the run folder's
blend_weights json (`refit_n_estimators`, `trend`; `--trend` is ignored), and averages N seeds (models.SEED .. SEED+N-1) of
each tree model (GLM / GAM are deterministic and fit once). Per seed it writes the cache file
pred_<line>_<model>_seed<SEED>.parquet (a rerun fits only the seeds without a cache file) and, for N > 1 and tree models,
the average pred_<line>_<model>_s<N>.parquet (blend_runs.py --write-from --pred-tag s<N> reads these). The untagged
pred_<line>_<model>.parquet of the full run is never rewritten in this mode; `--seeds 1` writes the seed-<SEED> cache file
only (it reproduces the untagged file). The json gains refit_seeds, refit_pred_tag, refit_rebalance_factor_seeds and
refit_only_at; every other key stays as it was. `--seeds N` > 1 needs `--refit-only` for the emerging market.

Usage: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line> [--models glm,gam,lgbm,xgb,cat] [--skip-refit]
                                                               [--refit-only] [--seeds N] [--te make_model]
       C057: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line> --market emerging --train-set pooled
             --developed-weight 1 --models lgbm,xgb,cat --refit-only --seeds 5
Writes: output/oof_<line>.parquet, output/blend_weights_<line>.json, output/pred_<line>_<model>.parquet
        C057 (emerging, in the run folder): pred_<line>_<model>_seed<SEED>.parquet, pred_<line>_<model>_s<N>.parquet
"""
import argparse
import datetime
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
# C042: factors that get an emerging-only GLM relativity in the pooled fit (features.market_interactions).
# region is left out: its codes already differ by market (R01-R12 vs E01-E06), so its main effect is market-specific.
GLM_INTERACTIONS = {
    "motor": ["channel", "new_business", "driver_age_band", "fuel_type", "vehicle_age", "hail_days"],
    "buildings": ["channel", "new_business", "policyholder_age_band", "construction_type", "sum_insured",
                  "wind_exposure_index"],
    "contents": ["channel", "new_business", "policyholder_age_band", "tenure", "security_level", "sum_insured"],
}


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


def emerging_design(line, train_set, dev_weight, glm_interactions=False):
    """C036/C038: one feature frame for every row the emerging pass needs, with per-row arrays.

    part: "dev" (developed training rows, pooled only), "em" (emerging training rows), "test" (emerging 2025 rows).
    y = capped cost / exposure_eff, w = exposure_eff (x dev_weight on developed rows).
    C042: glm_interactions appends the GLM_INTERACTIONS[line] market interaction columns (pooled GLM-only runs).
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
    if glm_interactions:
        x = features.market_interactions(x, GLM_INTERACTIONS[line])  # C042
    return x, part, y, w, train, test, t, year


def emerging_model(line, name, train_set, extra=None):
    """C036: params.for_model("emerging-<line>") (defaults unless tuned); emerging-only GLM/GAM pool rare levels at 30.
    C046: the emerging keys are tuned on emerging-only fits, so pooled fits keep the model defaults (C039: no pooled
    configuration beat them); `extra` (tune_cv configs, refit n_estimators) applies to both."""
    kw = params.for_model(f"emerging-{line}", name) if train_set == "emerging" else {}
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


def run_suffix(glm_interactions, nb_weight, inner_emerging, rows="all"):
    """C042 / C045: run-folder suffix, one folder per option set so existing runs stay untouched: `_glmx` (C042),
    then `_nb` (NB weighting with emerging-only early stopping), `_nbw` (NB weighting only), `_ie` (emerging-only early
    stopping only); C047: `_nbrows` (fit on new-business rows only)."""
    tail = ("_nb" if inner_emerging else "_nbw") if nb_weight is not None else ("_ie" if inner_emerging else "")
    return ("_glmx" if glm_interactions else "") + tail + ("_nbrows" if rows == "nb" else "")


def run_folder(train_set, dev_weight, suffix, tag):
    """C046: the run folder: run_dir + suffix, plus `_<tag>`; a tagged emerging-only run goes to
    output/emerging/emerging<suffix>_<tag>/ because the emerging-only run_dir is output/emerging/ itself."""
    if tag is None:
        return run_dir(train_set, dev_weight) + suffix
    if train_set == "emerging":
        return os.path.join(run_dir(train_set, dev_weight), f"emerging{suffix}_{tag}")
    return run_dir(train_set, dev_weight) + f"{suffix}_{tag}"


def inner_emerging_holdout(fit, em_fit):
    """C045: early-stopping holdout as a mask over the fit rows (length fit.sum()): a seeded 10% of the EMERGING fit
    rows, the C027 draw (numpy default_rng(models.SEED), n // 10 rows) applied to them only; every developed row stays
    in the fit part."""
    em_pos = np.flatnonzero(em_fit[fit])
    mask = np.zeros(int(fit.sum()), dtype=bool)
    mask[em_pos[np.random.default_rng(models.SEED).permutation(len(em_pos))[:len(em_pos) // 10]]] = True
    return mask


def nb_weights(w, nb, fit, option, target_share):
    """C045: sample weights with the new-business rows of both markets multiplied by one factor. option "auto": the
    factor that gives NB rows `target_share` of the total weight of the fit rows; a number: that factor; None: w
    unchanged. Returns (weights, factor, NB share of the fit rows' weight after weighting)."""
    if option is None:
        out, factor = w, 1.0
    else:
        factor = (target_share * w[fit & ~nb].sum() / ((1 - target_share) * w[fit & nb].sum())
                  if option == "auto" else float(option))
        out = w * np.where(nb, factor, 1.0)
    return out, float(factor), float(out[fit & nb].sum() / out[fit].sum())


def view_entries(rep, single_fold=False):
    """C044: per-fold `evaluate.view_report` entries as JSON fields: <key>_folds and <key>_mean, or (one fold) <key>."""
    keys = ("d2_cal_testmix", "d2_cal_nb", "nb_claims")
    if single_fold:
        return {k: rep[0][k] for k in keys}
    return {**{f"{k}_folds": [f[k] for f in rep] for k in keys},
            **{f"{k}_mean": float(np.mean([f[k] for f in rep])) for k in keys if k != "nb_claims"}}


def load_refit_json(out_dir, line, names):
    """C057: the blend_weights json of the full run in `out_dir` for `--refit-only`; it must hold the fold-derived
    refit_n_estimators of every requested model."""
    path = os.path.join(out_dir, f"blend_weights_{line}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"C057: --refit-only needs the full run's {path}; run {line} once without --refit-only first")
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    stored = result.get("refit_n_estimators", {})
    missing = [n for n in names if n not in stored]
    if missing:
        raise KeyError(f"C057: {missing} not in refit_n_estimators of {path} (stored: {sorted(stored)})")
    return result


def refit_emerging(result, line, names, n_est, trend, train_set, design, fit_rows, nb_weight, test_share, out_dir,
                   seeds=1, refit_only=False):
    """C036/C057: refit every model on all training years and write the 2025 test predictions to `out_dir`; `result` (the
    run's json dict) is updated in place.

    Full run (refit_only False, seeds 1): one fit per model at the default seed, written to the untagged
    pred_<line>_<model>.parquet; result gets refit_rebalance_factor, refit_n_estimators, trend, trend_factor, load_test.
    Refit-only: per seed k of a tree model (GLM / GAM: one fit) the file pred_<line>_<model>_seed<SEED + k>.parquet is
    loaded if it exists, otherwise fitted and written; for seeds > 1 the tree models' average goes to
    pred_<line>_<model>_s<seeds>.parquet. The untagged files are never touched; result only gains refit_seeds,
    refit_pred_tag, refit_rebalance_factor_seeds and refit_only_at."""
    x, part, y, w, _, test, t, _ = design
    is_dev, is_em = part == "dev", part == "em"
    nb = x["new_business"].to_numpy()
    factor = markets.get("emerging").trends[trend][line]  # C037
    load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
    fit = (is_dev | is_em) & fit_rows  # C047
    w_fit, nb_f, nb_s = nb_weights(w, nb, fit, nb_weight, test_share)  # C045: same weighting in the refit
    if not refit_only:
        result["fit_rows"]["refit"] = {"developed": int((fit & is_dev).sum()), "emerging": int((fit & is_em).sum())}
        result["nb_weight"].update(refit_factor=nb_f, refit_nb_weight_share=nb_s)
    scale = test["exposure"].to_numpy() * factor * load_test
    x_test = x.iloc[np.flatnonzero(part == "test")]
    log(f"emerging {line}: large-loss load test {load_test:.4f}, trend {trend} {factor:.4f}; "
        f"refit NB factor {nb_f:.3f}, NB share of fit weight {nb_s:.4f}")
    rebalance = {}
    for name in names:
        t0 = time.time()
        extra = {"n_estimators": n_est[name]} if n_est[name] else {}
        if not refit_only:
            m = emerging_model(line, name, train_set, extra or None)
            rebalance[name] = fit_emerging(m, x, y, w_fit, fit, fit & is_em, train_set,
                                           valid=None)  # C047: emerging NB rows with --rows nb
            pred = m.predict(x_test) * scale
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                          "trend_factor": factor}).to_parquet(os.path.join(out_dir, f"pred_{line}_{name}.parquet"),
                                                              index=False)
            log(f"emerging {line}: {name} refit (n_estimators {n_est[name]}) in {time.time() - t0:.0f}s, "
                f"test mean {pred.mean():.2f}")
            continue
        n_seeds = seeds if name in SEED_KWARG else 1  # C057: GLM / GAM are deterministic
        preds, rebalance[name] = [], {}
        for k in range(n_seeds):
            seed = models.SEED + k
            cache = os.path.join(out_dir, f"pred_{line}_{name}_seed{seed}.parquet")
            if os.path.exists(cache):
                c = pd.read_parquet(cache)
                assert (c["row_id"].to_numpy() == test["row_id"].to_numpy()).all() and np.allclose(
                    c["trend_factor"], factor), f"C057: cache {cache} belongs to another run"
                pred, rebalance[name][str(seed)] = c["prediction"].to_numpy(), float(c["rebalance_factor"].iloc[0])
                log(f"emerging {line}: {name} seed {seed} cached, test mean {pred.mean():.2f}")
            else:
                t1 = time.time()
                m = emerging_model(line, name, train_set, {**extra, **seed_param(name, seed)})
                rebalance[name][str(seed)] = fit_emerging(m, x, y, w_fit, fit, fit & is_em, train_set, valid=None)
                pred = m.predict(x_test) * scale
                pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                              "trend_factor": factor, "seed": seed, "rebalance_factor": rebalance[name][str(seed)]}
                             ).to_parquet(cache, index=False)
                log(f"emerging {line}: {name} seed {seed} refit (n_estimators {n_est[name]}) in "
                    f"{time.time() - t1:.0f}s, test mean {pred.mean():.2f}")
            preds.append(pred)
        pred_mean = np.mean(preds, axis=0)
        if seeds > 1 and name in SEED_KWARG:
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred_mean, "trend": trend,
                          "trend_factor": factor, "seeds": n_seeds}).to_parquet(
                os.path.join(out_dir, f"pred_{line}_{name}_s{seeds}.parquet"), index=False)
        log(f"emerging {line}: {name} {n_seeds} seed(s) in {time.time() - t0:.0f}s, test mean {pred_mean.mean():.2f}")
    if refit_only:
        old = result.get("refit_rebalance_factor_seeds", {})
        result.update(refit_seeds=seeds, refit_pred_tag=f"s{seeds}" if seeds > 1 else None,
                      refit_rebalance_factor_seeds={**old, **{n: {**old.get(n, {}), **f} for n, f in rebalance.items()}},
                      refit_only_at=datetime.datetime.now().isoformat(timespec="seconds"))
    else:
        result["refit_rebalance_factor"] = rebalance
        result.update(refit_n_estimators=n_est, trend=trend, trend_factor=factor, load_test=float(load_test))


def write_result(out_dir, line, result):
    """C036: the run's blend_weights json."""
    with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


def emerging_main(line, names, train_set="emerging", dev_weight=0.3, trend="none", refit=True, glm_interactions=False,
                  nb_weight=None, inner_emerging=False, rows="all", tag=None, refit_only=False, seeds=1):
    if glm_interactions:  # C042: the interaction columns must reach the GLM only; tree runs keep today's frame
        assert train_set == "pooled" and list(names) == ["glm"], "C042: --glm-interactions is for pooled GLM-only runs"
    assert not inner_emerging or train_set == "pooled", "C045: --inner-emerging is for pooled runs"
    assert rows in ("all", "nb") and (rows == "all" or train_set == "pooled"), "C047: --rows nb is for pooled runs"
    assert seeds >= 1 and (seeds == 1 or refit_only), "C057: for the emerging market several seeds need refit_only"
    assert refit or not refit_only, "C057: refit_only is a refit"
    out_dir = run_folder(train_set, dev_weight, run_suffix(glm_interactions, nb_weight, inner_emerging, rows),
                         tag)  # C042/C045/C046/C047
    if refit_only:  # C057: the stored full run comes first, so a missing run fails before the design is built
        result = load_refit_json(out_dir, line, names)
        assert result.get("nb_weight", {}).get("option") == nb_weight, "C057: --nb-weight must match the stored run"
    else:
        os.makedirs(out_dir, exist_ok=True)
    design = emerging_design(line, train_set, dev_weight, glm_interactions)
    x, part, y, w, train, test, t, year = design
    is_dev, is_em = part == "dev", part == "em"
    em_idx = np.flatnonzero(is_em)  # positions of the emerging training rows, in the order of t
    assert x["new_business"].dtype == bool, "C044: new_business must be a boolean feature column"
    nb = x["new_business"].to_numpy()
    # C047: rows a fit may use (every row, or new-business rows of both markets); validation rows are not restricted
    fit_rows = nb if rows == "nb" else np.ones(len(x), dtype=bool)
    test_rows = np.flatnonzero(part == "test")
    test_share = evaluate.nb_exposure_share(nb[test_rows], test["exposure"].to_numpy())  # C044: test NB exposure share
    log(f"emerging {line} [{train_set}]: features {x.shape[1]} cols, developed fit rows {is_dev.sum():,}, "
        f"emerging rows {is_em.sum():,}, test rows {len(test_rows):,}; test NB exposure share {test_share:.4f}; "
        f"NB weight {nb_weight}; early stopping on emerging rows only: {inner_emerging}; fit rows: {rows}")
    if refit_only:  # C057: no folds, blend search or views; the stored json keeps its keys and gains the refit_* ones
        trend_stored = result["trend"]
        log(f"emerging {line}: refit-only, trend {trend_stored} taken from the stored run (--trend {trend} ignored); "
            f"{seeds} seed(s) per tree model")
        refit_emerging(result, line, names, {n: result["refit_n_estimators"][n] for n in names}, trend_stored, train_set,
                       design, fit_rows, nb_weight, test_share, out_dir, seeds, refit_only=True)
        write_result(out_dir, line, result)
        return
    schemes = emerging_schemes(train, year)
    comp_expo = (t["exposure"] * t["completeness"]).to_numpy()
    # C044: per emerging training row, kept in the OOF file so the views can be recomputed offline
    row_info = {"new_business": nb[em_idx], "exposure": t["exposure"].to_numpy(),
                "claim_count": t["claim_count"].to_numpy()}

    frames, folds = [], {"time": [], "group": []}
    best_its = {n: {"time": [], "group": []} for n in names}
    rebal = {n: {"time": [], "group": []} for n in names}
    nb_factor, nb_share = {"time": [], "group": []}, {"time": [], "group": []}
    n_fit = {"time": [], "group": []}  # C047: fit rows per market
    for scheme, k, em_fit_mask, em_val_mask in schemes:
        em_fit = np.zeros(len(x), dtype=bool)
        em_fit[em_idx[em_fit_mask]] = True
        em_fit &= fit_rows  # C047: with --rows nb the re-balance (C038) and the holdout use emerging NB fit rows
        fit = (is_dev & fit_rows) | em_fit
        n_fit[scheme].append({"developed": int((fit & is_dev).sum()), "emerging": int(em_fit.sum())})
        val_rows = em_idx[em_val_mask]
        load_f = t.loc[em_fit_mask, "claim_cost"].sum() / t.loc[em_fit_mask, "claim_cost_capped"].sum()
        actual = t.loc[em_val_mask, "claim_cost"].to_numpy()
        scale = comp_expo[em_val_mask] * load_f
        val_info = {c: v[em_val_mask] for c, v in row_info.items()}
        frame = pd.DataFrame({"row_id": t.loc[em_val_mask, "row_id"].to_numpy(), "scheme": scheme, "fold": k,
                              "actual": actual, **val_info})
        # C045: NB-weighted fit weights (the C038 re-balance in fit_emerging then uses them too); emerging-only holdout
        w_fit, factor, share = nb_weights(w, nb, fit, nb_weight, test_share)
        nb_factor[scheme].append(factor)
        nb_share[scheme].append(share)
        valid = inner_emerging_holdout(fit, em_fit) if inner_emerging else "inner"  # C027 / C045
        if nb_weight is not None or inner_emerging:
            n_hold = int(valid.sum()) if inner_emerging else int(fit.sum()) // 10
            log(f"emerging {line} {scheme} {k}: NB factor {factor:.3f}, NB share of fit weight {share:.4f}; "
                f"early-stopping rows {n_hold:,}")
        for name in names:
            t0 = time.time()
            m = emerging_model(line, name, train_set)
            rebal[name][scheme].append(fit_emerging(m, x, y, w_fit, fit, em_fit, train_set, valid=valid))
            frame[name] = m.predict(x.iloc[val_rows]) * scale
            best_its[name][scheme].append(getattr(m, "best_iteration", None))
            s = scores(actual, frame[name].to_numpy())
            v = evaluate.nb_views(actual, frame[name].to_numpy(), val_info["new_business"], val_info["exposure"],
                                  val_info["claim_count"], test_share)
            log(f"emerging {line} {scheme} {k} {name}: d2={s['d2']:.5f} d2_cal={s['d2_cal']:.5f} "
                f"testmix={v['d2_cal_testmix']:.5f} nb={v['d2_cal_nb']:.5f} "
                f"p/a={s['pred_actual']:.3f} it={best_its[name][scheme][-1]} {time.time() - t0:.0f}s")
        frames.append(frame)
        folds[scheme].append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in names}, **val_info})
    pd.concat(frames).to_parquet(os.path.join(out_dir, f"oof_{line}.parquet"), index=False)

    searched = evaluate.blend_weights(folds["group"], step=0.05)  # C036: searched on the grouped folds
    rep = {s: evaluate.blend_report(folds[s], searched, names) for s in folds}
    views = {s: evaluate.view_report(folds[s], searched, names, test_share) for s in folds}  # C044
    guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
    weights, rule = evaluate.guarded_weights(guard, searched, names)  # index 0 time fold, index 1 grouped mean
    cv_time = {n: {**scores(folds["time"][0]["actual"], folds["time"][0]["preds"][n]),
                   **view_entries(views["time"]["single"][n], single_fold=True),
                   "best_iteration": best_its[n]["time"][0]} for n in names}
    cv_group = {n: {"d2_cal_folds": rep["group"]["single"][n], "d2_cal_mean": float(np.mean(rep["group"]["single"][n])),
                    "d2_mean": float(np.mean([evaluate.tweedie_d2(f["actual"], f["preds"][n]) for f in folds["group"]])),
                    **view_entries(views["group"]["single"][n]),
                    "best_iteration": best_its[n]["group"]} for n in names}
    for b, key in (("weighted", "blend_searched"), ("equal", "blend_equal")):
        cv_time[key] = {"d2_cal": rep["time"][b][0], **view_entries(views["time"][b], single_fold=True)}
        cv_group[key] = {"d2_cal_folds": rep["group"][b], "d2_cal_mean": float(np.mean(rep["group"][b])),
                         **view_entries(views["group"][b])}
    # C044: the test-mix view of the guard, reported only: weights are neither searched nor guarded on it
    testmix_guard = {b: [views["time"][b][0]["d2_cal_testmix"],
                         float(np.mean([f["d2_cal_testmix"] for f in views["group"][b]]))] for b in ("weighted", "equal")}
    result = {"line": line, "market": "emerging", "train_set": train_set,
              "developed_weight": dev_weight if train_set == "pooled" else None,
              "weights": weights, "weights_searched": searched, "rule": rule,
              "guard_scores": {"index": ["time fold 2023->2024", f"grouped {GROUP_FOLDS}-fold mean"], **guard,
                               "testmix": {"index": ["time fold test-mix", "grouped mean test-mix"], **testmix_guard}},
              "test_nb_share": test_share,
              "cv_time": cv_time, "cv_group": cv_group, "rebalance_factor": rebal,
              "nb_weight": {"option": nb_weight, "factor": nb_factor, "nb_weight_share": nb_share},
              "inner_emerging": inner_emerging, "rows": rows, "fit_rows": n_fit}
    if glm_interactions:
        result["glm_interactions"] = GLM_INTERACTIONS[line]  # C042

    if refit:
        n_est = {n: int(np.mean(best_its[n]["group"]) * 1.1) if all(best_its[n]["group"]) else None for n in names}
        refit_emerging(result, line, names, n_est, trend, train_set, design, fit_rows, nb_weight, test_share, out_dir)

    write_result(out_dir, line, result)


def parse_nb_weight(text):
    """C045: `--nb-weight` value: None (off), "auto" or a positive factor."""
    if text is None or text == "auto":
        return text
    value = float(text)  # ValueError on anything else
    if not value > 0:
        raise ValueError(f"factor must be positive, got {value}")
    return value


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
    ap.add_argument("--seeds", type=int, default=1,
                    help="seeds averaged per tree model in the refit (default 1); C057 (emerging): N > 1 needs "
                         "--refit-only and writes pred_<line>_<model>_s<N>.parquet")
    ap.add_argument("--te", default=None, help="C031: raw code column to target-encode, e.g. make_model (default off)")
    ap.add_argument("--glm-interactions", action="store_true",
                    help="C042 (emerging, pooled, --models glm): emerging-only GLM relativities for GLM_INTERACTIONS; "
                         "writes to <run folder>_glmx")
    ap.add_argument("--nb-weight", default=None,
                    help="C045 (emerging): multiply the weight of new-business rows (both markets) in every fit; "
                         "`auto` gives them the test NB exposure share of the fit weight, a number is the factor")
    ap.add_argument("--inner-emerging", action="store_true",
                    help="C045 (emerging, pooled): early-stopping holdout = 10%% of the emerging fit rows only")
    ap.add_argument("--rows", default="all", choices=["all", "nb"],
                    help="C047 (emerging, pooled): nb = fit on new-business rows of both markets only (folder _nbrows)")
    ap.add_argument("--tag", default=None, help="C046 (emerging): append _<tag> to the run folder")
    a = ap.parse_args()
    if a.glm_interactions and not (a.train_set == "pooled" and a.models == "glm"):
        ap.error("C042: --glm-interactions needs --train-set pooled and --models glm")
    try:
        nb_weight = parse_nb_weight(a.nb_weight)
    except ValueError as e:
        ap.error(f"C045: --nb-weight must be `auto` or a positive number ({e})")
    if a.inner_emerging and a.train_set != "pooled":
        ap.error("C045: --inner-emerging needs --train-set pooled")
    if a.rows == "nb" and a.train_set != "pooled":
        ap.error("C047: --rows nb needs --train-set pooled")
    assert a.seeds >= 1 and not (a.refit_only and a.skip_refit)
    if markets.get(a.market).name == "emerging":
        assert not a.te, "C036: --te is developed-only"
        if a.seeds > 1 and not a.refit_only:
            ap.error("C057: for the emerging market --seeds N > 1 needs --refit-only")
        emerging_main(a.line, a.models.split(","), a.train_set, a.developed_weight, a.trend, refit=not a.skip_refit,
                      glm_interactions=a.glm_interactions, nb_weight=nb_weight, inner_emerging=a.inner_emerging,
                      rows=a.rows, tag=a.tag, refit_only=a.refit_only, seeds=a.seeds)
    else:
        assert nb_weight is None and not a.inner_emerging, "C045: --nb-weight and --inner-emerging are emerging-only"
        assert a.rows == "all" and a.tag is None, "C046/C047: --rows and --tag are emerging-only"
        main(a.line, a.models.split(","), refit=not a.skip_refit, refit_only=a.refit_only, seeds=a.seeds, te=a.te)
