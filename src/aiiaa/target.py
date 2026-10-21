"""Build the reported (not yet developed) claim cost for each policy row.

Claim cost = incurred_to_date - recoveries, excluding repudiated claims (competition definition).
paid_to_date and incurred_to_date are already net of the policy excess (Kaggle data page).
"""
import numpy as np
import pandas as pd


def match_claims(policies, claims):
    """Attach the row_id whose period covers each claim's loss_date (same policy_id).

    C005: claims that fall outside every row of their policy keep row_id = NaN so they can be audited.
    """
    rows = policies[["row_id", "policy_id", "period_start", "period_end"]]
    m = claims.merge(rows, on="policy_id", how="left")
    inside = (m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])
    matched = m[inside].drop(columns=["period_start", "period_end"])
    unmatched = claims[~claims["claim_id"].isin(matched["claim_id"])].assign(row_id=pd.NA)
    return pd.concat([matched, unmatched], ignore_index=True)


def net_cost(claims):
    """Incurred minus recoveries; repudiated (declined) claims cost nothing."""
    cost = claims["incurred_to_date"] - claims["recoveries"]
    return cost.where(claims["status"] != "repudiated", 0.0)


def large_loss_cap(claims, q):
    """C014: per-claim cap at quantile q of non-repudiated net claim costs."""
    valid = claims[claims["status"] != "repudiated"]
    return float(np.quantile(net_cost(valid), q))


def row_target(policies, claims, cap=np.inf):
    """Per-row reported claim cost (uncapped and capped per claim at `cap`) and claim count."""
    valid = claims[claims["status"] != "repudiated"]
    m = match_claims(policies, valid).dropna(subset=["row_id"])
    m = m.assign(claim_cost=net_cost(m))
    m["claim_cost_capped"] = np.minimum(m["claim_cost"], cap)
    agg = m.groupby("row_id").agg(claim_cost=("claim_cost", "sum"), claim_cost_capped=("claim_cost_capped", "sum"),
                                  claim_count=("claim_id", "size"))
    out = policies[["row_id"]].merge(agg, on="row_id", how="left")
    out[["claim_cost", "claim_cost_capped"]] = out[["claim_cost", "claim_cost_capped"]].fillna(0.0)
    out["claim_count"] = out["claim_count"].fillna(0).astype(int)
    return out
