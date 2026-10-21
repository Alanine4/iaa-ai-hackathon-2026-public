"""Small LightGBM sweep on one line: tree complexity, learning rate, large-loss cap.

Every configuration is rebalanced on its own training data (C013) before scoring on 2024.
Usage: .venv\\Scripts\\python.exe scripts\\sweep_lgbm.py motor > docs\\sweep_lgbm_motor.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"), )
sys.path.insert(0, os.path.dirname(__file__))
from aiiaa import clean, evaluate, features, models  # noqa: E402
from diagnose_trees import capped_row_cost, load  # noqa: E402

LINE = sys.argv[1] if len(sys.argv) > 1 else "motor"
CONFIGS = [
    dict(n_estimators=1),  # initial score only: is the starting level balanced?
    dict(num_leaves=63, min_child_samples=200, learning_rate=0.05),
    dict(num_leaves=15, min_child_samples=1000, learning_rate=0.05),
    dict(num_leaves=7, min_child_samples=3000, learning_rate=0.05),
    dict(num_leaves=15, min_child_samples=1000, learning_rate=0.02),
    dict(num_leaves=31, min_child_samples=2000, learning_rate=0.02, reg_lambda=10.0),
]
CAPS = [np.inf, 50_000]


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

    for cap in CAPS:
        cost = t["claim_cost"].to_numpy() if np.isinf(cap) else capped_row_cost(p, c, cap)
        load_factor = t.loc[fit, "claim_cost"].sum() / cost[fit].sum()
        y = cost / w
        for cfg in CONFIGS:
            cfg = dict(cfg)
            lr = cfg.pop("learning_rate", 0.05)
            n = cfg.pop("n_estimators", 5000)
            m = models.LGBM(n_estimators=n, learning_rate=lr)
            m.model.set_params(**cfg)
            m.fit(x[fit], y[fit], w[fit], valid=None if n == 1 else (x[val], y[val], w[val]))
            balance = np.average(m.predict(x[fit]), weights=w[fit]) / np.average(y[fit], weights=w[fit])
            pred = m.predict(x[val]) * scale_val * load_factor / balance
            print(f"cap={cap:>7} lr={lr} n={n} {cfg}: iters={m.best_iteration:5d} train balance={balance:.3f} "
                  f"val D2 (rebalanced)={evaluate.tweedie_d2(actual, pred):.5f} pred/actual={pred.sum() / actual.sum():.3f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
