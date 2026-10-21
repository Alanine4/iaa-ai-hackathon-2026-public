"""Evidence for claim development: reporting lags (IBNR), case-estimate adequacy of open claims (IBNER),
and year-on-year frequency / severity.

Usage: .venv\\Scripts\\python.exe scripts\\investigate_development.py > docs\\investigate_development.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aiiaa import clean, target  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "iaa-ai-assisted-hackathon-2026-part-1")
VALUATION = pd.Timestamp("2024-12-31")


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def main():
    claims = load("claims")
    claims = claims[claims["status"] != "repudiated"].copy()
    claims["lag"] = (pd.to_datetime(claims["report_date"]) - pd.to_datetime(claims["loss_date"])).dt.days
    claims["net"] = target.net_cost(claims)
    claims["ay"] = pd.to_datetime(claims["loss_date"]).dt.year

    print("== Reporting lag by line (accident year 2021 only, observed for >= 3 years)")
    for line, g in claims[claims["ay"] == 2021].groupby("line"):
        for w in ["count", "cost"]:
            weights = np.ones(len(g)) if w == "count" else g["net"].to_numpy()
            order = np.argsort(g["lag"].to_numpy())
            lags = g["lag"].to_numpy()[order]
            cum = np.cumsum(weights[order]) / weights.sum()
            pts = {L: round(float(cum[lags <= L][-1]) if (lags <= L).any() else 0.0, 4) for L in [7, 30, 90, 180, 365, 730, 1095]}
            print(f"  {line:9s} {w:5s}-weighted share reported within L days: {pts}")

    print("\n== Reporting lag by peril (AY 2021): median / p90 / p99 days, mean cost")
    print(claims[claims["ay"] == 2021].groupby(["line", "peril"]).agg(
        n=("lag", "size"), med=("lag", "median"), p90=("lag", lambda s: s.quantile(.9)),
        p99=("lag", lambda s: s.quantile(.99)), mean_cost=("net", "mean")).round(0).to_string())

    print("\n== Lag stability: count share reported within 90 days by accident year (only lags fully observable)")
    for line, g in claims.groupby("line"):
        row = {}
        for ay in [2021, 2022, 2023]:
            gg = g[g["ay"] == ay]
            row[ay] = round(float((gg["lag"] <= 90).mean()), 4)
        print(f"  {line:9s} {row}")

    print("\n== Open-claim case estimates (IBNER): mean net cost, closed vs open, by line x AY")
    print(claims.pivot_table(index=["line", "ay"], columns="status", values="net", aggfunc=["mean", "size"]).round(0).to_string())

    print("\n== Mean paid / incurred for open claims by AY (how much of the estimate is already paid)")
    o = claims[claims["status"] == "open"]
    print((o.groupby(["line", "ay"])["paid_to_date"].sum() / o.groupby(["line", "ay"])["incurred_to_date"].sum()).round(3).unstack().to_string())

    print("\n== Frequency and severity by line x AY (reported only; 2024 is incomplete)")
    for line in ["motor", "buildings", "contents"]:
        p = clean.clean_policies(load(f"policies_{line}"))
        p["ay"] = pd.to_datetime(p["period_start"]).dt.year
        t = target.row_target(p, claims[claims["line"] == line].drop(columns=["lag", "net", "ay"]))
        p = p.merge(t, on="row_id")
        s = p.groupby("ay").agg(expo=("exposure", "sum"), n=("claim_count", "sum"), cost=("claim_cost", "sum"))
        s["freq"] = s["n"] / s["expo"]
        s["sev"] = s["cost"] / s["n"]
        s["pure_prem"] = s["cost"] / s["expo"]
        print(f"-- {line}\n{s.round(4).to_string()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
