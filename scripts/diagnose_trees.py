"""Diagnose early stopping and under-prediction of the tree models (motor, LightGBM).

Compares: baseline; large-loss capping of individual claims at several thresholds; and checks
whether the fitted model is balanced on its own training data.

Usage: .venv\\Scripts\\python.exe scripts\\diagnose_trees.py > docs\\diagnose_trees.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from aiiaa import clean, evaluate, features, models, target  # noqa: E402

DATA = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-1")
LINE = sys.argv[1] if len(sys.argv) > 1 else "motor"


def load(name):
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def capped_row_cost(p, claims, cap):
    valid = claims[claims["status"] != "repudiated"]
    m = target.match_claims(p, valid).dropna(subset=["row_id"])
    m = m.assign(cost=np.minimum(target.net_cost(m), cap))
    return p[["row_id"]].merge(m.groupby("row_id")["cost"].sum().rename("cost"), on="row_id", how="left")["cost"].fillna(0).to_numpy()


def main():
    claims = load("claims")
    c = claims[claims["line"] == LINE]
    p = clean.clean_policies(load(f"policies_{LINE}"))
    t = pd.read_parquet(os.path.join(ROOT, "output", f"target_{LINE}.parquet"))
    year = pd.to_datetime(p["period_start"]).dt.year.to_numpy()
    history = pd.DataFrame({"policy_id": p["policy_id"], "year": year, "claim_count": t["claim_count"],
                            "claim_cost": t["claim_cost"], "exposure": p["exposure"]}).groupby(["policy_id", "year"], as_index=False).sum()
    x = features.build_features(p, clean.fix_customer_id(load("customers")), load("addresses"), load("weather"), history)
    w = t["exposure_eff"].to_numpy()
    fit, val = year < 2024, year == 2024
    actual = t.loc[val, "claim_cost"].to_numpy()
    scale_val = (t["exposure"] * t["completeness"]).to_numpy()[val]

    net = target.net_cost(c[c["status"] != "repudiated"])
    print(f"{LINE}: claim net cost quantiles", net.quantile([.9, .99, .999, .9999, 1]).round(0).to_dict())
    print(f"share of total cost above 100k: {net[net > 1e5].sum() / net.sum():.3f}, above 250k: {net[net > 2.5e5].sum() / net.sum():.3f}")
    print(f"exposure_eff < 0.01 rows: {(w < 0.01).sum():,}, their share of cost: {t.loc[w < 0.01, 'claim_cost'].sum() / t['claim_cost'].sum():.4f}")

    for cap in [np.inf, 250_000, 100_000, 50_000]:
        cost = t["claim_cost"].to_numpy() if np.isinf(cap) else capped_row_cost(p, c, cap)
        load_factor = t.loc[fit, "claim_cost"].sum() / cost[fit].sum()  # spread capped excess back
        y = cost / w
        m = models.make("lgbm")
        m.fit(x[fit], y[fit], w[fit], valid=(x[val], y[val], w[val]))
        train_ratio = np.average(m.predict(x[fit]), weights=w[fit]) / np.average(y[fit], weights=w[fit])
        pred = m.predict(x[val]) * scale_val * load_factor
        print(f"cap={cap:>9}: best_iter={m.best_iteration:4d}  train balance={train_ratio:.3f}  load={load_factor:.4f}  "
              f"val D2={evaluate.tweedie_d2(actual, pred):.5f}  val pred/actual={pred.sum() / actual.sum():.3f}  "
              f"val D2 rebalanced={evaluate.tweedie_d2(actual, pred / train_ratio):.5f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
