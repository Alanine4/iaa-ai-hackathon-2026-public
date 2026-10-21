"""Evidence for cleaning rules Q1 (exposure), Q3 (sum_insured outliers), Q4 (construction_type codes).

Usage: .venv\\Scripts\\python.exe scripts\\investigate_q1_q3_q4.py > docs\\investigate_q1_q3_q4.txt
"""
import os
import sys

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "iaa-ai-assisted-hackathon-2026-part-1")
pd.set_option("display.width", 200)


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def claim_counts(p, line, claims):
    c = claims[(claims["line"] == line) & (claims["status"] != "repudiated")]
    m = c.merge(p[["row_id", "policy_id", "period_start", "period_end"]], on="policy_id")
    m = m[(m["loss_date"] >= m["period_start"]) & (m["loss_date"] <= m["period_end"])]
    return m.groupby("row_id").size().reindex(p["row_id"]).fillna(0).to_numpy()


def main():
    claims = load("claims")
    for line in ["motor", "buildings", "contents"]:
        p = load(f"policies_{line}")
        days = (pd.to_datetime(p["period_end"]) - pd.to_datetime(p["period_start"])).dt.days + 1
        p["days"] = days
        p["year"] = pd.to_datetime(p["period_start"]).dt.year
        p["n_claims"] = claim_counts(p, line, claims)
        diff = p["exposure"] - days / 365
        print(f"\n######## {line}")
        print("Q1 exposure - days/365 buckets:")
        print(pd.cut(diff, [-2, -0.01, -0.0001, 0.0001, 0.01, 0.02, 1, 400]).value_counts().sort_index().to_string())
        odd = p[diff.abs() > 0.0001]
        print("Q1 odd rows: exposure value counts (top 8):", odd["exposure"].value_counts().head(8).to_dict())
        print("Q1 odd rows: days value counts (top 8):", odd["days"].value_counts().head(8).to_dict())
        print("Q1 odd rows by year:", odd["year"].value_counts().sort_index().to_dict())
        sub = odd[odd["exposure"] < 1]
        if len(sub):
            print("Q1 odd rows with exposure<1 sample:\n", sub[["period_start", "period_end", "days", "exposure"]].head(8).to_string())
        print("Q1 leap year check: rows with days==366:", (days == 366).sum(), " days>366:", (days > 366).sum())
        print("Q1 claim frequency per exposure-year, odd vs normal:",
              round(odd["n_claims"].sum() / (odd["days"] / 365).sum(), 4), round(p["n_claims"].sum() / (days / 365).sum(), 4))

        if "sum_insured" in p:
            big = p["sum_insured"] > 5e6
            print(f"Q3 sum_insured > 5m: {big.sum():,}")
            print("Q3 written_premium mean, big vs normal:", round(p.loc[big, "written_premium"].mean(), 1), round(p.loc[~big, "written_premium"].mean(), 1))
            print("Q3 big / 1000 quantiles:", (p.loc[big, "sum_insured"] / 1000).quantile([0, .25, .5, .75, 1]).to_dict())
            print("Q3 normal quantiles:", p.loc[~big, "sum_insured"].quantile([0, .25, .5, .75, .99, 1]).to_dict())
            print("Q3 big rows divisible by 1000*1000:", ((p.loc[big, "sum_insured"] % 1e6) == 0).mean().round(3),
                  " normal rows divisible by 1000:", ((p.loc[~big, "sum_insured"] % 1000) == 0).mean().round(3))
            print("Q3 big by year:", p.loc[big, "year"].value_counts().sort_index().to_dict())
            print("Q3 same policy other years - sum_insured ratio (big/normal):")
            pol = p.loc[big, "policy_id"].unique()
            g = p[p["policy_id"].isin(pol)].assign(big=lambda d: d["sum_insured"] > 5e6)
            r = g.groupby("policy_id").apply(lambda d: d.loc[d.big, "sum_insured"].mean() / d.loc[~d.big, "sum_insured"].mean() if (~d.big).any() else None, include_groups=False).dropna()
            print("   n policies with both:", len(r), " ratio quantiles:", r.quantile([0, .5, 1]).round(1).to_dict())
            print("Q3 claim freq big vs normal:", round(p.loc[big, "n_claims"].sum() / p.loc[big, "days"].sum() * 365, 4),
                  round(p.loc[~big, "n_claims"].sum() / p.loc[~big, "days"].sum() * 365, 4))

        if "construction_type" in p:
            ct = p["construction_type"]
            print("Q4 construction_type by year:\n", pd.crosstab(ct, p["year"]).to_string())
            agg = p.groupby(ct).agg(rows=("row_id", "size"), prem=("written_premium", "mean"),
                                    freq=("n_claims", lambda s: s.sum()), expo=("days", lambda s: s.sum() / 365))
            agg["freq"] = (agg["freq"] / agg["expo"]).round(4)
            print("Q4 premium & claim frequency by code:\n", agg.round(2).to_string())
            # same policy seen with different codes across years?
            multi = p.groupby("policy_id")["construction_type"].agg(lambda s: tuple(sorted(set(s))))
            combos = multi[multi.map(len) > 1].value_counts().head(15)
            print("Q4 codes seen together on the same policy (top 15):\n", combos.to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
