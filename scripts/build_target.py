"""Build the developed target for every training row and reconcile it against the claims file.

Outputs (completeness mixes per-peril reporting CDFs by cost share, C022): output/target_<line>.parquet (row_id, exposure, claim_count, claim_cost, completeness,
claim_cost_ult, exposure_eff) and a reconciliation report on stdout.

Usage: .venv\\Scripts\\python.exe scripts\\build_target.py > docs\\target_summary.txt
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from aiiaa import clean, development, target  # noqa: E402

DATA = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-1")
OUT = os.path.join(ROOT, "output")
CDF_ACCIDENT_YEAR = 2021  # C006: oldest full year, observed for >= 3 years at valuation
CAP_QUANTILE = 0.995  # C014: per-claim large-loss cap; motor sweep favoured ~50k (99.5th percentile)


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def main():
    os.makedirs(OUT, exist_ok=True)
    claims = load("claims")
    for line in ["motor", "buildings", "contents"]:
        p = clean.clean_policies(load(f"policies_{line}"))
        c = claims[claims["line"] == line]
        valid = c[c["status"] != "repudiated"]

        cap = target.large_loss_cap(c, CAP_QUANTILE)
        t = target.row_target(p, c, cap=cap)
        print(f"\n{line}: large-loss cap (q={CAP_QUANTILE}) = {cap:,.0f}; capped share of cost = "
              f"{1 - t['claim_cost_capped'].sum() / t['claim_cost'].sum():.4f}")
        base = valid[pd.to_datetime(valid["loss_date"]).dt.year == CDF_ACCIDENT_YEAR]
        cdfs = development.reporting_cdf_by_peril(base)  # C022: per-peril CDFs, mixed by cost share
        shares = development.peril_cost_shares(base)
        t["completeness"] = development.row_completeness_mixed(p, cdfs, shares).to_numpy()
        t["exposure"] = p["exposure"].to_numpy()
        t = development.develop(t)
        final = os.path.join(OUT, f"target_{line}.parquet")
        tmp = os.path.join(OUT, f"target_{line}.tmp.parquet")
        t.to_parquet(tmp, index=False)
        os.replace(tmp, final)  # atomic swap: other processes may be reading the old file

        # Reconciliation: every non-repudiated cost is either on a row or listed as unmatched (C005)
        m = target.match_claims(p, valid)
        unmatched = m[m["row_id"].isna()]
        total = target.net_cost(valid).sum()
        on_rows = t["claim_cost"].sum()
        off_rows = target.net_cost(unmatched).sum()
        print(f"\n######## {line}")
        print(f"rows={len(t):,}  claims(non-repudiated)={len(valid):,}  matched={len(m) - len(unmatched):,}  unmatched={len(unmatched):,}")
        print(f"net cost total={total:,.2f}  on rows={on_rows:,.2f}  unmatched={off_rows:,.2f}  difference={total - on_rows - off_rows:,.4f}")
        print(f"unmatched by loss year: {pd.to_datetime(unmatched['loss_date']).dt.year.value_counts().sort_index().to_dict()}")
        print(f"per-peril reporting CDF (cost-weighted, AY{CDF_ACCIDENT_YEAR}) at 30/90/365 days, and cost share:")
        for peril in sorted(shares, key=shares.get, reverse=True):
            vals = [round(float(cdfs[peril][x]), 4) for x in (30, 90, 365)]
            print(f"  {peril:<18} share={shares[peril]:.4f}  cdf30/90/365={vals}")
        y = pd.to_datetime(p["period_start"]).dt.year.to_numpy()
        s = t.assign(year=y).groupby("year").agg(
            exposure=("exposure", "sum"), completeness=("completeness", "mean"),
            reported=("claim_cost", "sum"), ultimate=("claim_cost_ult", "sum"))
        s["pp_reported"] = s["reported"] / s["exposure"]
        s["pp_ultimate"] = s["ultimate"] / s["exposure"]
        print(s.round(4).to_string())
        print(f"min completeness={t['completeness'].min():.4f}  rows with completeness<0.5: {(t['completeness'] < 0.5).sum():,}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
