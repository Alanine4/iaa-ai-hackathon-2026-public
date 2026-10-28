"""C052: per-peril frequency x severity GLM for the emerging property lines (option B).

For every peril p of the line, on the pooled rows (developed rows at the line's C040 developed weight, plus the emerging
rows; the `market` column is a factor, so every peril has its own market level):
- frequency: Poisson GLM (models.GLM banding and one-hot, power 1) on n_p / exposure_eff, weight exposure_eff;
- severity: Gamma GLM (power 2, stronger ridge SEV_ALPHA) on capped cost / n_p of the rows with a claim of p, weight n_p;
- rate = sum over perils of frequency x severity (capped cost per exposure_eff, the scale of the other runs), then
  re-balanced on the emerging fit rows (C038).
Why: the residual scan (2026-10-28) left its largest property signals in peril-shaped places (freezing days x region:
escape of water; elevation: flood), and each peril can borrow its own weather and property structure from the
developed market. Part 1 tried per-peril LightGBM (C023, not adopted), never a per-peril GLM.

Validation, scoring and file layout are those of `transfer_emerging.py` / `oof_predictions.emerging_main` (time fold
2023 -> 2024, grouped 5-fold, C044 views), run name `peril`, model name `pglm`, so `blend_runs.py` can blend it.
Resumable: every fold's validation predictions and the refit are cached in output/emerging/peril/cache/ and reused.

Usage: .venv\\Scripts\\python.exe -I -u scripts\\peril_glm.py <line> [--dev-weight w] [--trend mixed-full]
Writes: output/emerging/peril/oof_<line>.parquet, blend_weights_<line>.json, pred_<line>_pglm.parquet
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
from oof_predictions import emerging_design, emerging_schemes, scores, view_entries  # noqa: E402
from train_validate import clean, evaluate, load, log, markets, models  # noqa: E402
from aiiaa import freqsev  # noqa: E402

RUN = "peril"
NAME = "pglm"
FREQ_ALPHA = 1e-4  # same ridge as the pooled GLM (models.GLM default)
SEV_ALPHA = 1e-3  # severity: far fewer rows (claims only)


class PerilGLM(models.GLM):
    """models.GLM with a chosen Tweedie power: 1 = Poisson frequency, 2 = Gamma severity."""

    def __init__(self, power, alpha, **kw):
        super().__init__(alpha=alpha, **kw)
        self.power = power

    def _fit(self, x, y, w, valid=None):
        num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and x[c].dtype != bool]
        self.edges_ = {c: np.unique(np.nanquantile(x[c].astype(float), np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                       for c in num}
        self.enc_ = OneHotEncoder(handle_unknown="ignore", min_frequency=self.min_frequency)
        xb = self.enc_.fit_transform(self._band(x))
        self.model_ = TweedieRegressor(power=self.power, link="log", alpha=self.alpha, max_iter=self.max_iter)
        self.model_.fit(xb, y, sample_weight=w)


def peril_frame(line):
    """Per-peril counts and capped costs of the developed then the emerging training rows (emerging_design order),
    with the line's own C014 / C037 cap of each market; checked against the target tables."""
    out = []
    for market in ("developed", "emerging"):
        pol = clean.clean_policies(load(f"policies_{line}", market))
        claims = load("claims", market)
        pt = freqsev.peril_targets(pol, claims[claims["line"] == line])
        t = pd.read_parquet(os.path.join(markets.get(market).out_dir, f"target_{line}.parquet"))
        capped = pt[[c for c in pt.columns if c.startswith("cost_") and not c.endswith("_uncapped")]].sum(axis=1)
        assert np.allclose(capped.to_numpy(), t["claim_cost_capped"].to_numpy()), f"{market} {line}: peril costs differ"
        out.append(pt.reset_index(drop=True))
    return pd.concat(out, ignore_index=True).fillna(0)


def fit_rates(x, eff, wmult, pt, perils, fit, pred_rows):
    """Fit every peril on rows `fit`; return the capped-cost rate (sum over perils) on `pred_rows`."""
    xf, xp = x.iloc[np.flatnonzero(fit)], x.iloc[pred_rows]
    rate = np.zeros(len(pred_rows))
    for p in perils:
        n, c = pt[f"n_{p}"].to_numpy(dtype=float), pt[f"cost_{p}"].to_numpy(dtype=float)
        nf, cf, ef, mf = n[fit], c[fit], eff[fit], wmult[fit]
        freq = PerilGLM(1, FREQ_ALPHA).fit(xf, nf / ef, ef * mf)
        pos = (nf > 0) & (cf > 0)
        sev = PerilGLM(2, SEV_ALPHA).fit(xf.iloc[np.flatnonzero(pos)], cf[pos] / nf[pos], nf[pos] * mf[pos])
        rate += freq.predict(xp) * sev.predict(xp)
    return rate


def main(line, dev_weight, trend, refit=True):
    out_dir = os.path.join(markets.get("emerging").out_dir, RUN)
    cache = os.path.join(out_dir, "cache")
    os.makedirs(cache, exist_ok=True)
    x, part, y, w, train, test, t, year = emerging_design(line, "pooled", dev_weight)
    is_dev, is_em = part == "dev", part == "em"
    em_idx, test_rows = np.flatnonzero(is_em), np.flatnonzero(part == "test")
    n_train = int(is_dev.sum() + is_em.sum())
    pt = peril_frame(line)
    pt = pd.concat([pt, pd.DataFrame(0, index=range(len(test_rows)), columns=pt.columns)], ignore_index=True)
    perils = [c[2:] for c in pt.columns if c.startswith("n_")]
    t_dev = pd.read_parquet(os.path.join(markets.get("developed").out_dir, f"target_{line}.parquet"))
    eff = np.concatenate([t_dev["exposure_eff"].to_numpy(), t["exposure_eff"].to_numpy(), np.ones(len(test_rows))])
    wmult = np.where(is_dev, dev_weight, 1.0)
    assert len(eff) == len(x) == len(pt) and n_train == len(t_dev) + len(t)
    log(f"peril {line}: perils {perils}, developed rows {is_dev.sum():,} (weight {dev_weight}), emerging {is_em.sum():,}")

    nb = x["new_business"].to_numpy()
    test_share = evaluate.nb_exposure_share(nb[test_rows], test["exposure"].to_numpy())
    comp_expo = (t["exposure"] * t["completeness"]).to_numpy()
    row_info = {"new_business": nb[em_idx], "exposure": t["exposure"].to_numpy(), "claim_count": t["claim_count"].to_numpy()}
    y_em, w_em = y[em_idx], w[em_idx]
    frames, folds, factors = [], {"time": [], "group": []}, {"time": [], "group": []}
    for scheme, k, fit_em, val_em in emerging_schemes(train, year):
        path = os.path.join(cache, f"{line}_w{dev_weight}_{scheme}_{k}.parquet")
        if os.path.exists(path):
            c = pd.read_parquet(path)
            rate_em, factor = c["rate_em"].to_numpy(), float(c["factor"].iloc[0])
            log(f"peril {line} {scheme} {k}: cached")
        else:
            t0 = time.time()
            fit = is_dev.copy()
            fit[em_idx[fit_em]] = True
            rate_em = fit_rates(x, eff, wmult, pt, perils, fit, em_idx)
            factor = float(np.sum(y_em[fit_em] * w_em[fit_em]) / np.sum(rate_em[fit_em] * w_em[fit_em]))  # C038
            pd.DataFrame({"rate_em": rate_em, "factor": factor}).to_parquet(path, index=False)
            log(f"peril {line} {scheme} {k}: fitted in {time.time() - t0:.0f}s, re-balance {factor:.3f}")
        load_f = t.loc[fit_em, "claim_cost"].sum() / t.loc[fit_em, "claim_cost_capped"].sum()
        actual = t.loc[val_em, "claim_cost"].to_numpy()
        pred = rate_em[val_em] * factor * comp_expo[val_em] * load_f
        val_info = {c: v[val_em] for c, v in row_info.items()}
        frames.append(pd.DataFrame({"row_id": t.loc[val_em, "row_id"].to_numpy(), "scheme": scheme, "fold": k,
                                    "actual": actual, **val_info, NAME: pred}))
        folds[scheme].append({"actual": actual, "preds": {NAME: pred}, **val_info})
        factors[scheme].append(factor)
        s = scores(actual, pred)
        v = evaluate.nb_views(actual, pred, val_info["new_business"], val_info["exposure"], val_info["claim_count"],
                              test_share)
        log(f"peril {line} {scheme} {k} {NAME}: d2_cal={s['d2_cal']:.5f} testmix={v['d2_cal_testmix']:.5f} "
            f"nb={v['d2_cal_nb']:.5f} p/a={s['pred_actual']:.3f}")
    pd.concat(frames).to_parquet(os.path.join(out_dir, f"oof_{line}.parquet"), index=False)

    names = [NAME]
    rep = {s: evaluate.blend_report(folds[s], {NAME: 1.0}, names) for s in folds}
    views = {s: evaluate.view_report(folds[s], {NAME: 1.0}, names, test_share) for s in folds}
    cv_time = {NAME: {**scores(folds["time"][0]["actual"], folds["time"][0]["preds"][NAME]),
                      **view_entries(views["time"]["single"][NAME], single_fold=True)}}
    cv_group = {NAME: {"d2_cal_folds": rep["group"]["single"][NAME],
                       "d2_cal_mean": float(np.mean(rep["group"]["single"][NAME])),
                       **view_entries(views["group"]["single"][NAME])}}
    result = {"line": line, "market": "emerging", "train_set": RUN, "developed_weight": dev_weight,
              "weights": {NAME: 1.0}, "perils": perils, "freq_alpha": FREQ_ALPHA, "sev_alpha": SEV_ALPHA,
              "test_nb_share": test_share, "cv_time": cv_time, "cv_group": cv_group, "rebalance_factor": factors}

    if refit:
        path = os.path.join(cache, f"{line}_w{dev_weight}_refit.parquet")
        if os.path.exists(path):
            rate_test = pd.read_parquet(path)["rate_test"].to_numpy()
            log(f"peril {line} refit: cached")
        else:
            t0 = time.time()
            fit = is_dev | is_em
            rate = fit_rates(x, eff, wmult, pt, perils, fit, np.concatenate([em_idx, test_rows]))
            level = float(np.sum(y_em * w_em) / np.sum(rate[:len(em_idx)] * w_em))
            rate_test = rate[len(em_idx):] * level
            pd.DataFrame({"rate_test": rate_test}).to_parquet(path, index=False)
            log(f"peril {line} refit: fitted in {time.time() - t0:.0f}s, re-balance {level:.3f}")
        factor = markets.get("emerging").trends[trend][line]
        load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
        pred = rate_test * test["exposure"].to_numpy() * factor * load_test
        pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                      "trend_factor": factor}).to_parquet(os.path.join(out_dir, f"pred_{line}_{NAME}.parquet"),
                                                          index=False)
        log(f"peril {line}: refit test mean {pred.mean():.2f}")
        result.update(trend=trend, trend_factor=factor, load_test=float(load_test))

    with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps({"line": line, "time": cv_time[NAME]["d2_cal"], "group": cv_group[NAME]["d2_cal_mean"],
                    "group_testmix": cv_group[NAME]["d2_cal_testmix_mean"]}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line", choices=["buildings", "contents", "motor"])
    ap.add_argument("--dev-weight", type=float, default=None, help="default: the line's C040 developed weight")
    ap.add_argument("--trend", default="mixed-full", choices=sorted(markets.get("emerging").trends))
    ap.add_argument("--skip-refit", action="store_true")
    a = ap.parse_args()
    dw = a.dev_weight if a.dev_weight is not None else markets.get("emerging").developed_weights[a.line]
    main(a.line, dw, a.trend, refit=not a.skip_refit)
