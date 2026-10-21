"""Develop reported claim cost to ultimate (C006).

Claims are a single snapshot valued at 2024-12-31; there are no historical valuations, so a classic
incurred triangle cannot be built. Development uses the reporting-lag distribution instead:
- IBNR: a cost-weighted reporting CDF per peril (C022; per line before), estimated from accident year 2021 (observed >= 3 years),
  gives the share of ultimate cost reported by the valuation date for each policy row; a row's
  completeness mixes the per-peril values by each peril's share of the line's cost.
- IBNER: open-claim case estimates are assumed unbiased (no valuation history to test this).
"""
import numpy as np
import pandas as pd

from aiiaa import target

VALUATION = pd.Timestamp("2024-12-31").date()
MAX_LAG = 1500  # longest observed reporting lag is 1361 days
MIN_PERIL_CLAIMS = 200  # C022: perils with fewer non-repudiated claims fall back to the pooled CDF


def reporting_cdf(lags, weights, max_lag=MAX_LAG):
    """Weighted share of claims reported within 0..max_lag days of the loss date."""
    totals = np.bincount(np.clip(lags.to_numpy(), 0, max_lag), weights=weights.to_numpy(), minlength=max_lag + 1)
    return np.cumsum(totals) / totals.sum()


def reporting_cdf_by_peril(claims, max_lag=MAX_LAG):
    """Cost-weighted reporting CDF for each peril (C022); repudiated claims are excluded.

    Lag is report_date - loss_date in days, weights are net cost. A peril with fewer than
    MIN_PERIL_CLAIMS claims gets the pooled CDF of all the claims passed in.
    """
    valid = claims[claims["status"] != "repudiated"]
    lag = (pd.to_datetime(valid["report_date"]) - pd.to_datetime(valid["loss_date"])).dt.days
    weight = target.net_cost(valid)
    pooled = reporting_cdf(lag, weight, max_lag)
    out = {}
    for peril, idx in valid.groupby("peril").groups.items():
        out[peril] = reporting_cdf(lag[idx], weight[idx], max_lag) if len(idx) >= MIN_PERIL_CLAIMS else pooled
    return out


def peril_cost_shares(claims):
    """Share of total net cost by peril (C022); repudiated claims are excluded; shares sum to 1."""
    valid = claims[claims["status"] != "repudiated"]
    cost = target.net_cost(valid).groupby(valid["peril"]).sum()
    return (cost / cost.sum()).to_dict()


def row_completeness(rows, cdf, valuation=VALUATION):
    """Mean of cdf(valuation - loss_day) over every day in each row's period (uniform loss timing)."""
    start = pd.to_datetime(rows["period_start"])
    end = pd.to_datetime(rows["period_end"])
    v = pd.Timestamp(valuation)
    lo = (v - end).dt.days.to_numpy()
    hi = (v - start).dt.days.to_numpy()
    if (lo < 0).any():
        raise ValueError("Row periods must end on or before the valuation date")
    ext = np.concatenate([cdf, np.full(max(0, hi.max() + 1 - len(cdf)), cdf[-1])])
    cum = np.concatenate([[0.0], np.cumsum(ext)])
    return pd.Series((cum[hi + 1] - cum[lo]) / (hi - lo + 1), index=rows.index)


def row_completeness_mixed(rows, cdfs, shares, valuation=VALUATION):
    """Cost-share-weighted mix of per-peril completeness: sum_p shares[p] * row_completeness(cdfs[p]) (C022)."""
    missing = sorted(set(shares) - set(cdfs))
    if missing:
        raise ValueError(f"No reporting CDF for perils: {missing}")
    out = pd.Series(0.0, index=rows.index)
    for peril, share in shares.items():
        out = out + share * row_completeness(rows, cdfs[peril], valuation)
    return out


def develop(t):
    """Ultimate cost estimate and completeness-weighted exposure for each row.

    claim_cost_ult is unbiased in aggregate; exposure_eff is the alternative for models that keep
    reported cost as the target and shrink exposure instead.
    """
    out = t.copy()
    out["claim_cost_ult"] = out["claim_cost"] / out["completeness"]
    out["exposure_eff"] = out["exposure"] * out["completeness"]
    return out
