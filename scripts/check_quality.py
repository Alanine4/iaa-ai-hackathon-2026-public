"""Follow-up checks on data-quality issues spotted in docs/data_profile.txt.

Usage: .venv\\Scripts\\python.exe scripts\\check_quality.py > docs\\data_quality_checks.txt
"""
import os
import sys

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "iaa-ai-assisted-hackathon-2026-part-1")


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def main():
    cust = load("customers")
    claims = load("claims")
    for line, name in [("motor", "policies_motor"), ("buildings", "policies_buildings"), ("contents", "policies_contents")]:
        p = load(name)
        days = (pd.to_datetime(p["period_end"]) - pd.to_datetime(p["period_start"])).dt.days + 1
        bad = p["exposure"] > 1.0001
        print(f"\n### {name}")
        print(f"exposure > 1: {bad.sum():,} rows; their exposure values (top): {p.loc[bad, 'exposure'].value_counts().head(5).to_dict()}")
        print(f"  of these, exposure == period days: {(p.loc[bad, 'exposure'] == days[bad]).mean():.1%}")
        ok = ~bad
        print(f"exposure vs days/365 max abs diff (good rows): {(p.loc[ok, 'exposure'] - days[ok] / 365).abs().max():.4f}")
        print(f"rows per policy-year > 1: {(p.groupby([p['policy_id'], p['period_start'].astype(str).str[:4]]).size() > 1).sum():,}")
        if "sum_insured" in p:
            q = p["sum_insured"].quantile([.5, .99, .999]).to_dict()
            print(f"sum_insured quantiles {q}; > 5,000,000: {(p['sum_insured'] > 5e6).sum():,} rows; sample: {sorted(p.loc[p['sum_insured'] > 5e6, 'sum_insured'].head(8).tolist())}")
        if "construction_type" in p:
            print("construction_type vs written_premium mean:\n", p.groupby("construction_type")["written_premium"].agg(["count", "mean"]).round(1))
        print(f"customer_service_contacts > 10: {(p['customer_service_contacts'] > 10).sum():,}")
        miss = ~p["customer_id"].isin(set(cust["customer_id"]))
        if miss.any():
            fixed = p.loc[miss, "customer_id"].str.zfill(8).isin(set(cust["customer_id"]))
            print(f"customer_id missing: {miss.sum():,}; lengths {p.loc[miss, 'customer_id'].str.len().value_counts().to_dict()}; resolved by zfill(8): {fixed.mean():.1%}")

        # match claims to rows by policy_id + loss_date within period
        c = claims[claims["line"] == line][["claim_id", "policy_id", "loss_date"]]
        m = c.merge(p[["row_id", "policy_id", "period_start", "period_end"]], on="policy_id", how="left")
        inside = (m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])
        hits = m[inside].groupby("claim_id").size()
        n = len(c)
        print(f"claims {n:,}: matched to exactly 1 row {(hits == 1).sum():,}, >1 rows {(hits > 1).sum():,}, unmatched {n - len(hits):,}")
        un = c[~c["claim_id"].isin(hits.index)]
        print(f"  unmatched by loss year: {un['loss_date'].astype(str).str[:4].value_counts().sort_index().to_dict()}")

    t = load("test_buildings")
    print("\ntest_buildings construction_type:", t["construction_type"].value_counts().to_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
