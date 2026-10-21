"""Build the developed target for every training row and reconcile it against the claims file.

Outputs: output/target_<line>.parquet (row_id, exposure, claim_count, claim_cost, completeness,
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


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def main():
    os.makedirs(OUT, exist_ok=True)
    claims = load("claims")
    for line in ["motor", "buildings", "contents"]:
        p = clean.clean_policies(load(f"policies_{line}"))
        c = claims[claims["line"] == line]
        valid = c[c["status"] != "repudiated"]

        t = target.row_target(p, c)
        lag = (pd.to_datetime(valid["report_date"]) - pd.to_datetime(valid["loss_date"])).dt.days
        base = pd.to_datetime(valid["loss_date"]).dt.year == CDF_ACCIDENT_YEAR
        cdf = development.reporting_cdf(lag[base], target.net_cost(valid[base]))
        t["completeness"] = development.row_completeness(p, cdf).to_numpy()
        t["exposure"] = p["exposure"].to_numpy()
        t = development.develop(t)
        t.to_parquet(os.path.join(OUT, f"target_{line}.parquet"), index=False)

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
        print(f"reporting CDF (cost-weighted, AY{CDF_ACCIDENT_YEAR}) at 30/90/365/730 days: "
              f"{[round(float(cdf[x]), 4) for x in (30, 90, 365, 730)]}")
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
