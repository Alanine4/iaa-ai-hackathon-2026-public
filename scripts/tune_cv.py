"""C017: time-based cross-validation tuning per line and model family.

Folds (expanding window): train 2021-2022 -> validate 2023; train 2021-2023 -> validate 2024.
Score: Tweedie D2 of reported cost vs prediction x completeness x large-loss load.
`d2_cal` rescales predictions to the fold's actual total first, so it measures risk ranking only;
the overall level is decided separately (trend share, C016). Configurations are ranked by mean d2_cal.

Usage: .venv\\Scripts\\python.exe scripts\\tune_cv.py trees|linear [line,...]
Writes: output/cv_<family>.csv (appended per fit) and docs/cv_<family>.md (summary)
"""
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import (LINES, OUT, ROOT, clean, cross_line_tables, evaluate, features, history_frame,  # noqa: E402
                            load, log, models)

FOLDS = [(2023, [2021, 2022]), (2024, [2021, 2022, 2023])]
N_RANDOM = 12
random.seed(7)

LGBM_SPACE = dict(num_leaves=[7, 15, 31], min_child_samples=[500, 1000, 3000, 6000], learning_rate=[0.03, 0.05],
                  colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0])
XGB_SPACE = dict(max_depth=[2, 3, 4, 5], min_child_weight=[10, 50, 200, 1000], learning_rate=[0.03, 0.05],
                 colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0])
LINEAR = [("glm", dict(n_bins=5)), ("glm", dict(n_bins=10)), ("glm", dict(n_bins=20)),
          ("glm", dict(n_bins=10, alpha=1e-3)),
          ("gam", dict(n_knots=5)), ("gam", dict(n_knots=8)), ("gam", dict(n_knots=12)),
          ("gam", dict(n_knots=8, alpha=1e-3))]


def sample(space, n):
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: random.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def configs(family):
    if family == "trees":
        return ([("lgbm", {})] + [("lgbm", c) for c in sample(LGBM_SPACE, N_RANDOM)]
                + [("xgb", {})] + [("xgb", c) for c in sample(XGB_SPACE, N_RANDOM // 2)])
    return LINEAR


def line_data(line, shared):
    p = clean.clean_policies(load(f"policies_{line}"))
    t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
    year = pd.to_datetime(p["period_start"]).dt.year.to_numpy()
    history = history_frame(p, t, year, "policy_id")
    history_all, portfolio = cross_line_tables()
    x = features.build_features(p, *shared, history=history, history_all=history_all, portfolio=portfolio)
    return x, t, year


def main(family, lines):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    csv = os.path.join(OUT, f"cv_{family}.csv")
    rows = []
    for line in lines:
        x, t, year = line_data(line, shared)
        w = t["exposure_eff"].to_numpy()
        y = t["claim_cost_capped"].to_numpy() / w
        for name, cfg in configs(family):
            for val_year, fit_years in FOLDS:
                fit, val = np.isin(year, fit_years), year == val_year
                load_f = t.loc[fit, "claim_cost"].sum() / t.loc[fit, "claim_cost_capped"].sum()
                actual = t.loc[val, "claim_cost"].to_numpy()
                scale = (t["exposure"] * t["completeness"]).to_numpy()[val] * load_f
                t0 = time.time()
                m = models.make(name, **cfg)
                m.fit(x[fit], y[fit], w[fit], valid=(x[val], y[val], w[val]) if name in ("lgbm", "xgb") else None)
                pred = m.predict(x[val]) * scale
                row = dict(line=line, model=name, config=str(cfg), val_year=val_year,
                           d2=evaluate.tweedie_d2(actual, pred),
                           d2_cal=evaluate.tweedie_d2(actual, pred * actual.sum() / pred.sum()),
                           pred_actual=pred.sum() / actual.sum(), iterations=getattr(m, "best_iteration", None),
                           sec=round(time.time() - t0))
                rows.append(row)
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
                log(f"{line} {name} {cfg} {val_year}: d2={row['d2']:.5f} d2_cal={row['d2_cal']:.5f} "
                    f"p/a={row['pred_actual']:.3f} it={row['iterations']} {row['sec']}s")

    df = pd.DataFrame(rows)
    summary = (df.groupby(["line", "model", "config"])
               .agg(d2_cal=("d2_cal", "mean"), d2=("d2", "mean"), pred_actual=("pred_actual", "mean"),
                    iterations=("iterations", "mean"), folds=("val_year", "size"))
               .reset_index().sort_values(["line", "model", "d2_cal"], ascending=[True, True, False]))
    with open(os.path.join(ROOT, "docs", f"cv_{family}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Time-based CV tuning: {family}\n\nFolds: 2021-22 -> 2023, 2021-23 -> 2024. Ranked by mean `d2_cal` "
                "(level-calibrated D2). Generated by `scripts/tune_cv.py` (C017).\n\n")
        f.write(summary.round(5).to_markdown(index=False))
        f.write("\n")
    log("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fam = sys.argv[1]
    main(fam, sys.argv[2].split(",") if len(sys.argv) > 2 else LINES)
