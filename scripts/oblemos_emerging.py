"""C054: the OBLEMOS Part 1 GLM and GAM (oblemos_part1/First code.py, settings of Improved 3) on the emerging market, as
blend candidates for P2-S13.

The models are OBLEMOS's own code, called unchanged:
- `forward_select`: forward stepwise selection on BIC by score test (frequency on a SELECTION_SAMPLE-row sample, severity
  on the claim rows), Improved 3 settings SCORE_RCOND 1e-6 / SELECTION_TRIES 10;
- `fit_freq_sev` / `predict_freq_sev`: Poisson frequency (offset log exposure_adj) x Gamma severity, own IRLS and `Design`
  (numeric variables in quantile bins, rare levels grouped, missing = own level);
- GAM: the same variables with cubic B-splines, one smoothing parameter per part chosen by AIC (`choose_lambda`) on the
  selection data and scaled to the fit size, as in `run_models`.
What Part 2 adds: the teammate's harmonised two-market feature matrix and targets (`oof_predictions.emerging_design`,
C035-C037); pooled rows at full weight (the code has no row weights), with `market` forced into both parts so each
market keeps its own level; the level re-balanced on the emerging fit rows (C038); and the teammate's validation folds,
views and file layout (run `oblemos`, models `ob_glm`, `ob_gam`), so `blend_runs.py` can blend it with P2-S13.
Selection and lambdas are chosen once per line on the time fold's fit rows (developed + emerging 2023), so the 2024 fold
stays out of the variable choice; every fold and the refit then refit the coefficients.
Resumable: selection, every fold and the refit are cached in output/emerging/oblemos/cache/.

C055 `--dev-weight w` (default 1): developed rows at weight w without changing the OBLEMOS functions. Poisson
frequency: a row weight w equals counts and exposure both times w (w y log(mu) - w mu = y' log(mu') - mu' + const with
y' = w y, mu' = w mu), so the developed claim_count and exposure_adj are scaled. Gamma severity has no such identity:
a seeded random share w of the developed claims is kept (an unbiased stand-in for weight w). Run folder `oblemos_w<w>`.

Usage: .venv\\Scripts\\python.exe -I -u scripts\\oblemos_emerging.py <line> [--models glm,gam] [--trend mixed-full]
Writes: output/emerging/oblemos/oof_<line>.parquet, blend_weights_<line>.json, pred_<line>_ob_<model>.parquet
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
from oof_predictions import emerging_design, emerging_schemes, scores, view_entries  # noqa: E402
from train_validate import clean, evaluate, load, log, markets  # noqa: E402
from aiiaa import target  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("first_code", os.path.join(ROOT, "oblemos_part1", "First code.py"))
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)
fc.SCORE_RCOND = 1e-6  # Improved 3 settings (the GLM / GAM behind Blend4)
fc.SELECTION_TRIES = 10

RUN = "oblemos"
CAP_QUANTILE = 0.995  # C014 / C037: each market's own per-claim cap, as in build_target.py
FORCED = ["market"]


def claim_rows(line, n_dev):
    """One row per non-repudiated claim with capped net cost > 0: position of its policy row in emerging_design's
    developed + emerging order, and net_cost_capped (each market's own cap)."""
    out, offset = [], 0
    for market in ("developed", "emerging"):
        pol = clean.clean_policies(load(f"policies_{line}", market)).reset_index(drop=True)
        claims = load("claims", market)
        c = claims[claims["line"] == line]
        cap = target.large_loss_cap(c, CAP_QUANTILE)
        m = target.match_claims(pol, c[c["status"] != "repudiated"]).dropna(subset=["row_id"])
        cost = np.minimum(target.net_cost(m).to_numpy(dtype=float), cap)
        pos = pd.Series(np.arange(len(pol)), index=pol["row_id"]).reindex(m["row_id"]).to_numpy()
        keep = cost > 0
        out.append(pd.DataFrame({"pos": pos[keep].astype(int) + offset, "net_cost_capped": cost[keep]}))
        offset += len(pol)
        if market == "developed":
            assert offset == n_dev, f"{line}: developed policies {offset} vs design rows {n_dev}"
    return pd.concat(out, ignore_index=True)


def frames(x, rows, counts, expo, claims):
    """`fr` (policy rows: features, claim_count, exposure_adj) and `sv` (claims of those rows: features, cost)."""
    fr = x.iloc[rows].reset_index(drop=True).assign(claim_count=counts[rows], exposure_adj=expo[rows])
    sel = claims[np.isin(claims["pos"], rows)]
    sv = x.iloc[sel["pos"].to_numpy()].reset_index(drop=True).assign(net_cost_capped=sel["net_cost_capped"].to_numpy())
    return fr, sv


def select(x, counts, expo, claims, sel_rows, feats, cache):
    """OBLEMOS forward selection and GAM lambdas on the selection rows, cached."""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    fr, sv = frames(x, sel_rows, counts, expo, claims)
    samp = fr.sample(min(len(fr), fc.SELECTION_SAMPLE), random_state=fc.SEED)
    f_feats, _ = fc.forward_select(samp, feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float),
                                   1, np.log(samp["exposure_adj"].to_numpy()), "frequency")
    s_feats, phi_s = fc.forward_select(sv, feats, np.ones(len(sv)), sv["net_cost_capped"].to_numpy(float), 2, None,
                                       "severity")
    f_feats += [f for f in FORCED if f not in f_feats]
    s_feats += [f for f in FORCED if f not in s_feats]
    lam_f = fc.choose_lambda(samp, f_feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float), 1,
                             np.log(samp["exposure_adj"].to_numpy()), 1.0, "GAM frequency")
    lam_s = fc.choose_lambda(sv, s_feats, np.ones(len(sv)), sv["net_cost_capped"].to_numpy(float), 2, None, phi_s,
                             "GAM severity")
    res = {"frequency": f_feats, "severity": s_feats, "lam_f": lam_f, "lam_s": lam_s, "n_samp": len(samp)}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def fit_predict(x, counts, expo, claims, fit_rows, pred_rows, sel, kinds):
    """Rates per unit exposure_adj on pred_rows for each model kind ("glm", "gam")."""
    fr, sv = frames(x, fit_rows, counts, expo, claims)
    xp = x.iloc[pred_rows].reset_index(drop=True)
    out = {}
    for kind in kinds:
        smooth = kind == "gam"
        lam_f = sel["lam_f"] * len(fr) / sel["n_samp"] if smooth else 0.0  # run_models: same smoothness, more data
        m = fc.fit_freq_sev(fr, sv, sel["frequency"], sel["severity"], smooth=smooth, lam_f=lam_f,
                            lam_s=sel["lam_s"] if smooth else 0.0)
        out[kind] = fc.predict_freq_sev(m, xp, np.ones(len(xp)))
    return out


def main(line, kinds, trend, refit=True, dev_weight=1.0, nb_only=False):
    run = (RUN if dev_weight == 1.0 else f"{RUN}_w{dev_weight:g}") + ("_nbrows" if nb_only else "")
    out_dir = os.path.join(markets.get("emerging").out_dir, run)
    cache = os.path.join(out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    x, part, y, w, train, test, t, year = emerging_design(line, "pooled", 1.0)
    is_dev, is_em = part == "dev", part == "em"
    dev_idx, em_idx, test_rows = np.flatnonzero(is_dev), np.flatnonzero(is_em), np.flatnonzero(part == "test")
    t_dev = pd.read_parquet(os.path.join(markets.get("developed").out_dir, f"target_{line}.parquet"))
    counts = np.concatenate([t_dev["claim_count"].to_numpy(float), t["claim_count"].to_numpy(float),
                             np.zeros(len(test_rows))])
    expo = np.concatenate([t_dev["exposure_eff"].to_numpy(), t["exposure_eff"].to_numpy(), np.ones(len(test_rows))])
    expo = np.clip(expo, 1e-4, None)  # as run_models: periods starting on the cut-off day
    claims = claim_rows(line, len(dev_idx))
    if dev_weight != 1.0:  # C055: Poisson weight = scaled counts and exposure; Gamma: seeded share of developed claims
        counts[dev_idx] *= dev_weight
        expo[dev_idx] *= dev_weight
        dev_claim = claims["pos"].to_numpy() < len(dev_idx)
        keep = ~dev_claim | (np.random.default_rng(fc.SEED).random(len(claims)) < dev_weight)
        claims = claims[keep].reset_index(drop=True)
    feats = list(x.columns)
    nb_all = x["new_business"].astype(bool).to_numpy()
    # C056 --nb-only: every fit (selection, folds, refit) on the new-business rows of both markets only (as C047 --rows nb);
    # validation still scores every emerging row, the re-balance uses the emerging NB fit rows
    dev_fit = dev_idx[nb_all[dev_idx]] if nb_only else dev_idx
    em_ok = nb_all[em_idx] if nb_only else np.ones(len(em_idx), dtype=bool)
    log(f"oblemos {line}: developed weight {dev_weight:g}, developed rows {len(dev_idx):,}, emerging {len(em_idx):,}, claims {len(claims):,}, "
        f"{len(feats)} candidate features")

    t0 = time.time()
    sel = select(x, counts, expo, claims, np.concatenate([dev_fit, em_idx[(year == 2023) & em_ok]]), feats,
                 os.path.join(cache, f"{line}_selection.json"))
    log(f"oblemos {line}: selection ({time.time() - t0:.0f}s) frequency {sel['frequency']}; severity {sel['severity']}; "
        f"GAM lambda freq {sel['lam_f']} sev {sel['lam_s']}")

    nb = x["new_business"].to_numpy()
    test_share = evaluate.nb_exposure_share(nb[test_rows], test["exposure"].to_numpy())
    comp_expo = (t["exposure"] * t["completeness"]).to_numpy()
    row_info = {"new_business": nb[em_idx], "exposure": t["exposure"].to_numpy(), "claim_count": t["claim_count"].to_numpy()}
    y_em, w_em = y[em_idx], w[em_idx]
    names = [f"ob_{k}" for k in kinds]
    out_frames, folds = [], {"time": [], "group": []}
    for scheme, k, fit_em, val_em in emerging_schemes(train, year):
        path = os.path.join(cache, f"{line}_{scheme}_{k}.parquet")
        c = pd.read_parquet(path) if os.path.exists(path) else None
        if c is not None and all(f"rate_{n}" in c for n in names):
            rates = {n: c[f"rate_{n}"].to_numpy() for n in names}
            log(f"oblemos {line} {scheme} {k}: cached")
        else:
            t0 = time.time()
            r = fit_predict(x, counts, expo, claims, np.concatenate([dev_fit, em_idx[fit_em & em_ok]]), em_idx, sel, kinds)
            rates = {f"ob_{kd}": v for kd, v in r.items()}
            pd.DataFrame({f"rate_{n}": v for n, v in rates.items()}).to_parquet(path, index=False)
            log(f"oblemos {line} {scheme} {k}: fitted in {time.time() - t0:.0f}s")
        load_f = t.loc[fit_em, "claim_cost"].sum() / t.loc[fit_em, "claim_cost_capped"].sum()
        actual = t.loc[val_em, "claim_cost"].to_numpy()
        val_info = {c: v[val_em] for c, v in row_info.items()}
        frame = pd.DataFrame({"row_id": t.loc[val_em, "row_id"].to_numpy(), "scheme": scheme, "fold": k,
                              "actual": actual, **val_info})
        for n in names:
            rb = fit_em & em_ok
            factor = float(np.sum(y_em[rb] * w_em[rb]) / np.sum(rates[n][rb] * w_em[rb]))  # C038
            frame[n] = rates[n][val_em] * factor * comp_expo[val_em] * load_f
            s = scores(actual, frame[n].to_numpy())
            v = evaluate.nb_views(actual, frame[n].to_numpy(), val_info["new_business"], val_info["exposure"],
                                  val_info["claim_count"], test_share)
            log(f"oblemos {line} {scheme} {k} {n}: d2_cal={s['d2_cal']:.5f} testmix={v['d2_cal_testmix']:.5f} "
                f"nb={v['d2_cal_nb']:.5f} p/a={s['pred_actual']:.3f} re-balance={factor:.3f}")
        out_frames.append(frame)
        folds[scheme].append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in names}, **val_info})
    pd.concat(out_frames).to_parquet(os.path.join(out_dir, f"oof_{line}.parquet"), index=False)

    eq = {n: 1 / len(names) for n in names}
    rep = {s: evaluate.blend_report(folds[s], eq, names) for s in folds}
    views = {s: evaluate.view_report(folds[s], eq, names, test_share) for s in folds}
    cv_time = {n: {**scores(folds["time"][0]["actual"], folds["time"][0]["preds"][n]),
                   **view_entries(views["time"]["single"][n], single_fold=True)} for n in names}
    cv_group = {n: {"d2_cal_folds": rep["group"]["single"][n], "d2_cal_mean": float(np.mean(rep["group"]["single"][n])),
                    **view_entries(views["group"]["single"][n])} for n in names}
    result = {"line": line, "market": "emerging", "train_set": run, "developed_weight": dev_weight, "weights": eq,
              "selection": sel, "settings": {"SCORE_RCOND": fc.SCORE_RCOND, "SELECTION_TRIES": fc.SELECTION_TRIES,
                                             "SELECTION_SAMPLE": fc.SELECTION_SAMPLE, "N_BINS": fc.N_BINS,
                                             "N_KNOTS": fc.N_KNOTS, "MIN_LEVEL_SHARE": fc.MIN_LEVEL_SHARE},
              "test_nb_share": test_share, "cv_time": cv_time, "cv_group": cv_group}

    if refit:
        path = os.path.join(cache, f"{line}_refit.parquet")
        c = pd.read_parquet(path) if os.path.exists(path) else None
        if c is not None and all(f"rate_{n}" in c for n in names):
            rates = {n: c[f"rate_{n}"].to_numpy() for n in names}
            log(f"oblemos {line} refit: cached")
        else:
            t0 = time.time()
            r = fit_predict(x, counts, expo, claims, np.concatenate([dev_fit, em_idx[em_ok]]),
                            np.concatenate([em_idx, test_rows]), sel, kinds)
            rates = {f"ob_{kd}": v for kd, v in r.items()}
            pd.DataFrame({f"rate_{n}": v for n, v in rates.items()}).to_parquet(path, index=False)
            log(f"oblemos {line} refit: fitted in {time.time() - t0:.0f}s")
        factor_t = markets.get("emerging").trends[trend][line]
        load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
        n_em = len(em_idx)
        for n in names:
            level = float(np.sum((y_em * w_em)[em_ok]) / np.sum((rates[n][:n_em] * w_em)[em_ok]))
            pred = rates[n][n_em:] * level * test["exposure"].to_numpy() * factor_t * load_test
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                          "trend_factor": factor_t}).to_parquet(os.path.join(out_dir, f"pred_{line}_{n}.parquet"),
                                                                index=False)
            log(f"oblemos {line}: {n} refit re-balance {level:.3f}, test mean {pred.mean():.2f}")
        result.update(trend=trend, trend_factor=factor_t, load_test=float(load_test))

    with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps({"line": line, "time": {n: cv_time[n]["d2_cal"] for n in names},
                    "group": {n: cv_group[n]["d2_cal_mean"] for n in names},
                    "group_testmix": {n: cv_group[n]["d2_cal_testmix_mean"] for n in names}}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line", choices=["buildings", "contents", "motor"])
    ap.add_argument("--models", default="glm,gam")
    ap.add_argument("--trend", default="mixed-full", choices=sorted(markets.get("emerging").trends))
    ap.add_argument("--skip-refit", action="store_true")
    ap.add_argument("--dev-weight", type=float, default=1.0)
    ap.add_argument("--nb-only", action="store_true", help="C056: fit on new-business rows only")
    a = ap.parse_args()
    main(a.line, a.models.split(","), a.trend, refit=not a.skip_refit, dev_weight=a.dev_weight, nb_only=a.nb_only)
