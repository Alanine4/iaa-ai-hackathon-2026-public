"""C023: per-peril frequency x severity model of claim cost.

For every peril p of a line:
- frequency: LightGBM Poisson on the claim rate n_p / exposure_eff with weight exposure_eff. This is the
  same likelihood as a count model with a log(exposure_eff) offset, but keeps LightGBM's boost-from-average
  start (it is skipped when an init_score offset is supplied).
- severity: LightGBM Gamma on the mean capped cost per claim of rows with positive capped cost of p,
  weight = number of claims; a constant (claim-weighted mean) when the peril has fewer than `min_claims`
  claims in the training data.
- both parts are rebalanced on the training data only (as `models.Balanced`): sum of predicted counts =
  actual count, then sum of predicted capped cost = actual capped cost (the second factor also absorbs
  the few zero-cost claims left out of the gamma fit).
- the capped excess is spread back with a per-peril load (uncapped / capped cost on the training data).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from aiiaa import target
from aiiaa.models import SEED, _bools_to_int

CAP_QUANTILE = 0.995  # C014: same per-claim large-loss cap as scripts/build_target.py

FREQ_DEFAULTS = dict(objective="poisson", num_leaves=15, min_child_samples=3000, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, verbose=-1)
SEV_DEFAULTS = dict(objective="gamma", num_leaves=7, min_child_samples=1000, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, verbose=-1)


def _peril_exposure(exposure, peril):
    """Exposure of one peril: a column of a per-peril DataFrame, or the same array for every peril."""
    if isinstance(exposure, pd.DataFrame):
        return exposure[peril].to_numpy(dtype=float)
    return np.asarray(exposure, dtype=float)


def peril_targets(policies, claims_line, cap=None):
    """Per-row claim count, capped cost and uncapped cost of every peril of the line (0 where none).

    Repudiated claims and claims outside every row are not counted (same rules as target.row_target).
    cap defaults to the line's C014 cap (99.5th percentile of non-repudiated net claim costs).
    """
    if cap is None:
        cap = target.large_loss_cap(claims_line, CAP_QUANTILE)
    valid = claims_line[claims_line["status"] != "repudiated"]
    m = target.match_claims(policies, valid).dropna(subset=["row_id"])
    m = m.assign(cost_uncapped=target.net_cost(m))
    m["cost"] = np.minimum(m["cost_uncapped"], cap)
    agg = m.groupby(["row_id", "peril"]).agg(n=("claim_id", "size"), cost=("cost", "sum"),
                                             cost_uncapped=("cost_uncapped", "sum"))
    out = pd.DataFrame(index=policies.index)
    for p in sorted(claims_line["peril"].unique()):
        rows = agg.xs(p, level="peril") if p in agg.index.get_level_values("peril") else agg.iloc[:0]
        for col in ["n", "cost", "cost_uncapped"]:
            name = f"{col}_{p}" if col != "cost_uncapped" else f"cost_{p}_uncapped"
            out[name] = policies["row_id"].map(rows[col]).fillna(0).to_numpy()
        out[f"n_{p}"] = out[f"n_{p}"].astype(int)
    return out


class FreqSev:
    """Sum over perils of frequency x severity x large-loss load; see the module docstring."""

    def __init__(self, n_estimators=2000, learning_rate=0.05, min_claims=1500, freq_params=None, sev_params=None):
        self.min_claims = min_claims
        common = dict(n_estimators=n_estimators, learning_rate=learning_rate)
        self.freq_params = {**FREQ_DEFAULTS, **common, **(freq_params or {})}
        self.sev_params = {**SEV_DEFAULTS, **common, **(sev_params or {})}

    @staticmethod
    def _lgbm(params, x, y, w, valid):
        model = lgb.LGBMRegressor(**params)
        kw = {}
        if valid is not None:
            xv, yv, wv = valid
            kw = dict(eval_X=xv, eval_y=yv, eval_sample_weight=[wv], callbacks=[lgb.early_stopping(100, verbose=False)])
        model.fit(x, y, sample_weight=w, **kw)
        return model

    def fit(self, x, pt, exposure_eff, valid=None):
        """x: features; pt: peril_targets rows of x; exposure_eff: completeness-weighted exposure, one array
        for all perils or a DataFrame with one column per peril (per-peril reporting completeness).

        valid = (x, pt, exposure_eff) of the validation rows enables early stopping.
        """
        xi = _bools_to_int(x)
        if valid is not None:
            xv, ptv = _bools_to_int(valid[0]), valid[1]
        self.perils_ = [c[2:] for c in pt.columns if c.startswith("n_")]
        self.components_ = {}
        for p in self.perils_:
            w = _peril_exposure(exposure_eff, p)
            if valid is not None:
                wv = _peril_exposure(valid[2], p)
            n, cost = pt[f"n_{p}"].to_numpy(dtype=float), pt[f"cost_{p}"].to_numpy(dtype=float)
            uncapped = pt[f"cost_{p}_uncapped"].to_numpy(dtype=float)

            vf = None if valid is None else (xv, ptv[f"n_{p}"].to_numpy(dtype=float) / wv, wv)
            freq = self._lgbm(self.freq_params, xi, n / w, w, vf)
            freq_raw = freq.predict(xi)
            freq_balance = float(n.sum() / np.sum(freq_raw * w))

            if n.sum() >= self.min_claims:
                pos = cost > 0
                vs = None
                if valid is not None:
                    nv, cv = ptv[f"n_{p}"].to_numpy(dtype=float), ptv[f"cost_{p}"].to_numpy(dtype=float)
                    vpos = cv > 0
                    vs = (xv[vpos], cv[vpos] / nv[vpos], nv[vpos])
                sev = self._lgbm(self.sev_params, xi[pos], cost[pos] / n[pos], n[pos], vs)
                sev_raw = sev.predict(xi)
            else:
                sev = float(cost.sum() / n.sum())
                sev_raw = np.full(len(xi), sev)
            sev_balance = float(cost.sum() / np.sum(freq_raw * freq_balance * w * sev_raw))

            self.components_[p] = dict(
                freq=freq, sev=sev, freq_balance=freq_balance, sev_balance=sev_balance,
                load=float(uncapped.sum() / cost.sum()), n_claims=int(n.sum()),
                freq_iterations=freq.best_iteration_ or freq.n_estimators,
                sev_iterations=None if isinstance(sev, float) else (sev.best_iteration_ or sev.n_estimators))
        return self

    def rates(self, x):
        """Balanced claim frequency per unit exposure (freq_<p>) and capped severity per claim (sev_<p>)."""
        xi = _bools_to_int(x)
        out = {}
        for p, c in self.components_.items():
            out[f"freq_{p}"] = c["freq"].predict(xi) * c["freq_balance"]
            sev = np.full(len(xi), c["sev"]) if isinstance(c["sev"], float) else c["sev"].predict(xi)
            out[f"sev_{p}"] = sev * c["sev_balance"]
        return pd.DataFrame(out, index=x.index)

    def predict(self, x, exposure):
        """Expected uncapped claim cost per row (sum over perils, load included).

        exposure: one array for all perils or a DataFrame with one column per peril, as in fit.
        """
        r = self.rates(x)
        return sum(r[f"freq_{p}"].to_numpy() * _peril_exposure(exposure, p) * r[f"sev_{p}"].to_numpy() * c["load"]
                   for p, c in self.components_.items())
