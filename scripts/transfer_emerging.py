"""C046: transfer model for the emerging market (audit P2-F5, option b): a model fitted on the developed market only
gives every emerging row a base rate; an emerging-only booster then learns the emerging departures from it.

Why: in the pooled runs (C038) the developed rows carry 86-96% of the fit weight, so the trees and their early stopping
optimise the developed fit and the emerging structure enters only through `market` splits (audit P2-F5, H4a). Here the
developed fit and the emerging fit are separated:
  1. base: LightGBM (params.for_model(line, "lgbm"), the Part 1 settings) on every developed training row, rate =
     capped cost / exposure_eff, weight exposure_eff, C027 inner early stopping. It never sees an emerging row, so the
     same base serves every emerging fold without leakage.
  2. level: per fold, the base is re-levelled to the emerging fit rows (factor = actual / base on those rows), like the
     C038 re-balance.
  3. booster: LightGBM or CatBoost on the emerging fit rows only, started from log(level x base rate) (LightGBM
     init_score, CatBoost Pool baseline), small trees and strong regularisation (BOOSTERS); trees chosen by early
     stopping on a seeded 10% of the emerging fit rows (the C027 draw). The booster's own level is then balanced on
     the emerging fit rows (C013).
Validation, scoring and file layout are those of `oof_predictions.emerging_main` (C036/C044): time fold 2023 -> 2024 and
grouped 5-fold on policy_id, reported cost vs prediction x exposure x completeness x fit-rows large-loss load, d2_cal plus
the test-mix and NB-only views; the OOF file has the same columns, so `blend_runs.py` can blend this run with the pooled
runs (run name `transfer`). Refit: base unchanged, booster on all emerging rows with mean grouped best iteration x 1.1.

Usage: .venv\\Scripts\\python.exe -I -u scripts\\transfer_emerging.py <line> [--trend mixed-full] [--skip-refit]
Writes: output/emerging/transfer/oof_<line>.parquet, blend_weights_<line>.json, pred_<line>_<model>.parquet
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
from oof_predictions import emerging_design, emerging_schemes, scores, view_entries  # noqa: E402
from train_validate import evaluate, log, markets, models, params  # noqa: E402

RUN = "transfer"
POWER = models.POWER
# C046: emerging boosters. Small trees and strong regularisation: about 38k fit rows and 400-3,000 claims per fold.
BOOSTERS = {
    "tr_base": None,  # the re-levelled developed model alone
    "tr_lgbm": dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.02, num_leaves=4,
                    min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                    n_estimators=3000, random_state=models.SEED, verbose=-1),
    "tr_cat": dict(loss_function=f"Tweedie:variance_power={POWER}", learning_rate=0.03, depth=3, l2_leaf_reg=10,
                   iterations=3000, random_seed=models.SEED, verbose=False, thread_count=-1, allow_writing_files=False),
    # ridge GLM relativities on a few factors (GLM_FACTORS), penalty in exposure-years: the booster trees fitted noise
    "tr_glm_1k": dict(glm=1000.0),
    "tr_glm_5k": dict(glm=5000.0),
    "tr_glm_20k": dict(glm=20000.0),
}
# C046: factors of the GLM correction. region: the base never saw the emerging codes E01-E06; the others are the largest
# market differences (docs/part2_market_comparison.md: channel, age mix, new business, tenure / security, cover).
GLM_FACTORS = {
    "motor": ["region", "new_business", "channel", "driver_age_band", "urban_density", "cover_type", "fuel_type"],
    "buildings": ["region", "new_business", "channel", "policyholder_age_band", "urban_density", "construction_type",
                  "flood_zone"],
    "contents": ["region", "new_business", "channel", "policyholder_age_band", "urban_density", "tenure",
                 "security_level"],
}


class RidgeGLM:
    """C046: Tweedie (power 1.5) log-link GLM with an offset, one-hot factors (missing = own level, no base level) and
    an L2 penalty `lam` on every coefficient except the intercept, fitted by IRLS. With the penalty the relativities are
    shrunk towards the offset, like a credibility weight."""

    def __init__(self, factors, lam, max_iter=50):
        self.factors, self.lam, self.max_iter = factors, lam, max_iter

    def _design(self, x):
        cols = [np.ones(len(x))]
        for f, levels in self.levels_.items():
            v = x[f].astype(object).where(x[f].notna(), "NA").astype(str).to_numpy()
            cols += [(v == lvl).astype(float) for lvl in levels]
        return np.column_stack(cols)

    def fit(self, x, y, w, offset):
        self.levels_ = {f: sorted(x[f].astype(object).where(x[f].notna(), "NA").astype(str).unique())
                        for f in self.factors}
        X = self._design(x)
        # lam is in exposure-years: the IRLS weights are w x mu^(2-p), so scale the penalty by the mean rate^(2-p)
        ybar = np.sum(y * w) / np.sum(w)
        pen = np.full(X.shape[1], self.lam * ybar ** (2 - POWER))
        pen[0] = 0.0
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(np.sum(y * w) / np.sum(np.exp(offset) * w))
        for _ in range(self.max_iter):
            eta = offset + X @ beta
            mu = np.exp(eta)
            W = w * mu ** (2 - POWER)
            z = X @ beta + (y - mu) / mu
            new = np.linalg.solve(X.T @ (X * W[:, None]) + np.diag(pen), X.T @ (W * z))
            done = np.max(np.abs(new - beta)) < 1e-7
            beta = new
            if done:
                break
        self.beta_ = beta
        self.best_iteration = None
        return self

    def predict(self, x, offset):
        return np.exp(offset + self._design(x) @ self.beta_)


def inner_mask(n):
    """C027 draw: a seeded 10% of n rows."""
    mask = np.zeros(n, dtype=bool)
    mask[np.random.default_rng(models.SEED).permutation(n)[:n // 10]] = True
    return mask


# C046 outcome on contents: every emerging correction lost to the re-levelled base (tree boosters; ridge GLM on
# GLM_FACTORS improves monotonically with the penalty and only ties the base at 20,000 exposure-years). So the run also
# scores developed-only bases of the other tree types (Part 1 settings), re-levelled the same way.
BASES = {"tr_base": "lgbm", "tr_base_xgb": "xgb", "tr_base_cat": "cat"}


def base_model(x, y, w, is_dev, line, kind):
    """Step 1: one model type on the developed rows only (Part 1 settings, inner early stopping)."""
    t0 = time.time()
    m = models.make(kind, **params.for_model(line, kind))
    m.fit(x[is_dev], y[is_dev], w[is_dev], valid="inner")
    log(f"transfer {line}: developed base {kind} {m.best_iteration} trees in {time.time() - t0:.0f}s")
    return m


def cat_frame(x, cats):
    out = models._bools_to_int(x).copy()
    for c in cats:
        out[c] = out[c].astype(object).where(out[c].notna(), "NA").astype(str)
    return out


class Booster:
    """Step 3: one emerging booster started from a log base rate (offset). predict(x, offset) returns the rate."""

    def __init__(self, name, n_estimators=None):
        self.name, self.n_estimators = name, n_estimators

    def fit(self, x, y, w, offset, early_stop=True):
        p = dict(BOOSTERS[self.name])
        if self.name == "tr_lgbm":
            if self.n_estimators:
                p["n_estimators"] = self.n_estimators
            self.model = lgb.LGBMRegressor(**p)
            xi = models._bools_to_int(x)
            if early_stop:
                hold = inner_mask(len(x))
                self.model.fit(xi[~hold], y[~hold], sample_weight=w[~hold], init_score=offset[~hold],
                               eval_set=[(xi[hold], y[hold])], eval_sample_weight=[w[hold]],
                               eval_init_score=[offset[hold]], callbacks=[lgb.early_stopping(100, verbose=False)])
                self.best_iteration = self.model.best_iteration_ or p["n_estimators"]
            else:
                self.model.fit(xi, y, sample_weight=w, init_score=offset)
                self.best_iteration = p["n_estimators"]
        else:
            self.cats = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
            if self.n_estimators:
                p["iterations"] = self.n_estimators
            xc = cat_frame(x, self.cats)
            if early_stop:
                hold = inner_mask(len(x))
                p.update(od_type="Iter", od_wait=100)
                self.model = CatBoostRegressor(**p)
                self.model.fit(Pool(xc[~hold], y[~hold], weight=w[~hold], baseline=offset[~hold], cat_features=self.cats),
                               eval_set=Pool(xc[hold], y[hold], weight=w[hold], baseline=offset[hold],
                                             cat_features=self.cats))
                self.best_iteration = self.model.get_best_iteration() + 1
            else:
                self.model = CatBoostRegressor(**p)
                self.model.fit(Pool(xc, y, weight=w, baseline=offset, cat_features=self.cats))
                self.best_iteration = p["iterations"]
        self.balance_ = 1.0
        self.balance_ = float(np.sum(y * w) / np.sum(self.predict(x, offset) * w))  # C013 on the fit rows
        return self

    def raw_margin(self, x):
        if self.name == "tr_lgbm":
            return self.model.predict(models._bools_to_int(x), raw_score=True)
        return self.model.predict(Pool(cat_frame(x, self.cats), cat_features=self.cats), prediction_type="RawFormulaVal")

    def predict(self, x, offset):
        return np.exp(offset + self.raw_margin(x)) * self.balance_


def make_booster(name, line):
    spec = BOOSTERS[name]
    return RidgeGLM(GLM_FACTORS[line], spec["glm"]) if "glm" in spec else Booster(name)


def main(line, trend, refit=True, boosters=()):
    out_dir = os.path.join(markets.get("emerging").out_dir, RUN)
    os.makedirs(out_dir, exist_ok=True)
    x, part, y, w, train, test, t, year = emerging_design(line, "pooled", 1.0)
    is_dev, is_em = part == "dev", part == "em"
    em_idx, test_rows = np.flatnonzero(is_em), np.flatnonzero(part == "test")
    nb = x["new_business"].to_numpy()
    test_share = evaluate.nb_exposure_share(nb[test_rows], test["exposure"].to_numpy())
    log(f"transfer {line}: developed rows {is_dev.sum():,}, emerging rows {is_em.sum():,}, test rows {len(test_rows):,}")

    not_dev = np.flatnonzero(~is_dev)
    base_rates, base_its = {}, {}
    for name, kind in BASES.items():
        m = base_model(x, y, w, is_dev, line, kind)
        r = np.zeros(len(x))
        r[not_dev] = m.predict(x.iloc[not_dev])
        base_rates[name], base_its[name] = r, m.best_iteration
    base_rate = base_rates["tr_base"]  # the boosters start from the LightGBM base
    x_em, y_em, w_em, b_em = x.iloc[em_idx], y[em_idx], w[em_idx], base_rate[em_idx]
    x_test = x.iloc[test_rows]
    del x
    names = list(BASES) + [n for n in BOOSTERS if n != "tr_base" and n in boosters]

    comp_expo = (t["exposure"] * t["completeness"]).to_numpy()
    row_info = {"new_business": nb[em_idx], "exposure": t["exposure"].to_numpy(), "claim_count": t["claim_count"].to_numpy()}
    frames, folds = [], {"time": [], "group": []}
    best_its = {n: {"time": [], "group": []} for n in names}
    for scheme, k, fit, val in emerging_schemes(train, year):
        load_f = t.loc[fit, "claim_cost"].sum() / t.loc[fit, "claim_cost_capped"].sum()
        actual = t.loc[val, "claim_cost"].to_numpy()
        scale = comp_expo[val] * load_f
        val_info = {c: v[val] for c, v in row_info.items()}
        frame = pd.DataFrame({"row_id": t.loc[val, "row_id"].to_numpy(), "scheme": scheme, "fold": k, "actual": actual,
                              **val_info})
        level = float(np.sum(y_em[fit] * w_em[fit]) / np.sum(b_em[fit] * w_em[fit]))  # step 2
        off_fit, off_val = np.log(b_em[fit] * level), np.log(b_em[val] * level)
        for name in names:
            t0 = time.time()
            if name in BASES:  # step 2 for this base: re-levelled on the emerging fit rows
                b = base_rates[name][em_idx]
                rate = b[val] * float(np.sum(y_em[fit] * w_em[fit]) / np.sum(b[fit] * w_em[fit]))
                it = None
            else:
                m = make_booster(name, line).fit(x_em.iloc[np.flatnonzero(fit)], y_em[fit], w_em[fit], off_fit)
                rate, it = m.predict(x_em.iloc[np.flatnonzero(val)], off_val), m.best_iteration
            frame[name] = rate * scale
            best_its[name][scheme].append(it)
            s = scores(actual, frame[name].to_numpy())
            v = evaluate.nb_views(actual, frame[name].to_numpy(), val_info["new_business"], val_info["exposure"],
                                  val_info["claim_count"], test_share)
            log(f"transfer {line} {scheme} {k} {name}: d2_cal={s['d2_cal']:.5f} testmix={v['d2_cal_testmix']:.5f} "
                f"nb={v['d2_cal_nb']:.5f} p/a={s['pred_actual']:.3f} level={level:.3f} it={it} {time.time() - t0:.0f}s")
        frames.append(frame)
        folds[scheme].append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in names}, **val_info})
    pd.concat(frames).to_parquet(os.path.join(out_dir, f"oof_{line}.parquet"), index=False)

    # same report layout as emerging_main (C036/C044); weights searched on the grouped folds, guarded on time + grouped
    searched = evaluate.blend_weights(folds["group"], step=0.05)
    rep = {s: evaluate.blend_report(folds[s], searched, names) for s in folds}
    views = {s: evaluate.view_report(folds[s], searched, names, test_share) for s in folds}
    guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
    weights, rule = evaluate.guarded_weights(guard, searched, names)
    cv_time = {n: {**scores(folds["time"][0]["actual"], folds["time"][0]["preds"][n]),
                   **view_entries(views["time"]["single"][n], single_fold=True),
                   "best_iteration": best_its[n]["time"][0]} for n in names}
    cv_group = {n: {"d2_cal_folds": rep["group"]["single"][n], "d2_cal_mean": float(np.mean(rep["group"]["single"][n])),
                    **view_entries(views["group"]["single"][n]), "best_iteration": best_its[n]["group"]} for n in names}
    result = {"line": line, "market": "emerging", "train_set": RUN, "weights": weights, "weights_searched": searched,
              "rule": rule, "guard_scores": guard, "test_nb_share": test_share, "cv_time": cv_time, "cv_group": cv_group,
              "base_best_iteration": base_its}

    if refit:
        factor = markets.get("emerging").trends[trend][line]
        load_test = t["claim_cost"].sum() / t["claim_cost_capped"].sum()
        scale = test["exposure"].to_numpy() * factor * load_test
        level = float(np.sum(y_em * w_em) / np.sum(b_em * w_em))
        off_all, off_test = np.log(b_em * level), np.log(base_rate[test_rows] * level)
        n_est = {}
        for name in names:
            if name in BASES:
                b = base_rates[name]
                rate = b[test_rows] * float(np.sum(y_em * w_em) / np.sum(b[em_idx] * w_em))
                n_est[name] = None
            elif "glm" in BOOSTERS[name]:
                n_est[name] = None
                rate = make_booster(name, line).fit(x_em, y_em, w_em, off_all).predict(x_test, off_test)
            else:
                n_est[name] = int(np.mean(best_its[name]["group"]) * 1.1)
                m = Booster(name, n_est[name]).fit(x_em, y_em, w_em, off_all, early_stop=False)
                rate = m.predict(x_test, off_test)
            pred = rate * scale
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "prediction": pred, "trend": trend,
                          "trend_factor": factor}).to_parquet(os.path.join(out_dir, f"pred_{line}_{name}.parquet"),
                                                              index=False)
            log(f"transfer {line}: {name} refit (n_estimators {n_est[name]}), test mean {pred.mean():.2f}")
        result.update(refit_n_estimators=n_est, refit_level=level, trend=trend, trend_factor=factor,
                      load_test=float(load_test))

    with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps({"line": line, "time": {n: cv_time[n]["d2_cal"] for n in names},
                    "group": {n: cv_group[n]["d2_cal_mean"] for n in names},
                    "group_testmix": {n: cv_group[n]["d2_cal_testmix_mean"] for n in names},
                    "weights": weights, "rule": rule}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--trend", default="mixed-full", choices=sorted(markets.get("emerging").trends))
    ap.add_argument("--skip-refit", action="store_true")
    ap.add_argument("--boosters", default="", help="comma list of emerging corrections on the LightGBM base, e.g. "
                                                    "tr_lgbm,tr_cat,tr_glm_5k (default none: they lost on contents)")
    a = ap.parse_args()
    main(a.line, a.trend, refit=not a.skip_refit, boosters=[b for b in a.boosters.split(",") if b])
