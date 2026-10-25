"""C017: time-based cross-validation tuning per line and model family.

Folds (expanding window): train 2021-2022 -> validate 2023; train 2021-2023 -> validate 2024.
Score: Tweedie D2 of reported cost vs prediction x completeness x large-loss load.
`d2_cal` rescales predictions to the fold's actual total first, so it measures risk ranking only;
the overall level is decided separately (trend share, C016). Configurations are ranked by mean d2_cal.

Usage: .venv\\Scripts\\python.exe scripts\\tune_cv.py trees|linear [line,...]
Writes: output/cv_<family>.csv (appended per fit) and docs/cv_<family>.md (summary)

C039 (`--market emerging`): hyper-parameter search of the tree models on the emerging data, whose 34k-90k training rows
per line are far fewer than the ~1,000,000 the Part 1 defaults were chosen for (`min_child_samples` 3000 is a very
coarse leaf there). Only family `trees` is searched (`EMERGING_*_SPACE`: full grids plus the model defaults `{}`).
Every configuration is scored on the same schemes as `oof_predictions.py --market emerging`: the time fold (train
2023 -> score 2024) and the grouped 5-fold CV on policy_id, with the same scoring. `--train-set pooled` /
`--developed-weight` add the developed rows to every fit as in C038. The script adopts nothing: the summary shows the
grouped-fold mean d2_cal (the ranking) and the time-fold d2_cal side by side, each with its gain over the defaults.

Usage: .venv\\Scripts\\python.exe scripts\\tune_cv.py trees <line>[,<line>] --market emerging
                                                      [--train-set pooled] [--developed-weight 0.3]
Writes: output/emerging[/pooled_w<w>]/cv_trees.csv (appended per fit) and docs/cv_emerging_trees.md (every line in the csv)
"""
import argparse
import itertools
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import (LINES, OUT, ROOT, clean, evaluate, features, history_frame, load, log,  # noqa: E402
                            markets, models)

FOLDS = [(2023, [2021, 2022]), (2024, [2021, 2022, 2023])]
N_RANDOM = 12
random.seed(7)

LGBM_SPACE = dict(num_leaves=[7, 15, 31], min_child_samples=[500, 1000, 3000, 6000], learning_rate=[0.03, 0.05],
                  colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0])
XGB_SPACE = dict(max_depth=[2, 3, 4, 5], min_child_weight=[10, 50, 200, 1000], learning_rate=[0.03, 0.05],
                 colsample_bytree=[0.5, 0.8], subsample=[0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0])
# C039: emerging-market search spaces (full grids; the model defaults {} are added in emerging_configs)
EMERGING_LGBM_SPACE = dict(num_leaves=[7, 15, 31], min_child_samples=[50, 200, 1000, 3000],
                           learning_rate=[0.05], colsample_bytree=[0.8], subsample=[0.8])
EMERGING_XGB_SPACE = dict(max_depth=[3, 4, 6], min_child_weight=[1, 10, 100])
EMERGING_CAT_SPACE = dict(depth=[4, 6, 8], l2_leaf_reg=[1, 3, 10])
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
    x = features.build_features(p, *shared, history=history, line=line)  # C026: no cross-line tables needed
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
                m.fit(x[fit], y[fit], w[fit], valid="inner")  # C027: inner split; GLM/GAM ignore it
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


def grid(space):
    """C039: every combination of a search space, in a stable order."""
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def emerging_configs(family):
    """C039: [(model, config), ...] of the emerging-market search; each model's defaults ({}) come first."""
    if family != "trees":
        raise ValueError(f"C039: the emerging search covers family 'trees' only, got {family!r} "
                         "(GLM/GAM over-fit on emerging data for other reasons)")
    spaces = (("lgbm", EMERGING_LGBM_SPACE), ("xgb", EMERGING_XGB_SPACE), ("cat", EMERGING_CAT_SPACE))
    return [(name, cfg) for name, space in spaces for cfg in [{}] + grid(space)]


def emerging_summary(df):
    """C039: one row per line, model and config: grouped-fold mean d2_cal (the ranking), time-fold d2_cal and the gain
    of both over that model's defaults ({}); ranked within line and model by the grouped-fold mean."""
    df = df.drop_duplicates(["line", "model", "config", "scheme", "fold"], keep="last")  # a re-run replaces a fit
    keys = ["line", "model", "config"]
    group, time_fold = df[df["scheme"] == "group"], df[df["scheme"] == "time"]
    out = (group.groupby(keys).agg(d2_cal_group=("d2_cal", "mean"), d2_cal_group_min=("d2_cal", "min"),
                                   d2_group=("d2", "mean"), pred_actual_group=("pred_actual", "mean"),
                                   iterations=("iterations", "mean"), folds=("fold", "size")).reset_index()
           .merge(time_fold[keys + ["d2_cal", "pred_actual"]].rename(
               columns={"d2_cal": "d2_cal_time", "pred_actual": "pred_actual_time"}), on=keys, how="left"))
    base = out[out["config"] == str({})].set_index(["line", "model"])
    index = pd.MultiIndex.from_frame(out[["line", "model"]])
    for fold in ("group", "time"):
        out[f"gain_{fold}"] = out[f"d2_cal_{fold}"] - base[f"d2_cal_{fold}"].reindex(index).to_numpy()
    return out.sort_values(["line", "model", "d2_cal_group"], ascending=[True, True, False])


def emerging_main(family, lines, train_set="emerging", dev_weight=0.3, models=None):
    """C039: search the emerging-market configurations of `family` on the C036 schemes (see module docstring)."""
    import oof_predictions as oof  # here, not at the top: oof_predictions imports FOLDS from this module

    todo = [(n, cfg) for n, cfg in emerging_configs(family) if models is None or n in models]  # C039: --models filter
    out_dir = oof.run_dir(train_set, dev_weight)
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"cv_{family}.csv")
    for line in lines:
        x, part, y, w, train, _, t, year = oof.emerging_design(line, train_set, dev_weight)
        is_dev, em_idx = part == "dev", np.flatnonzero(part == "em")
        comp_expo = (t["exposure"] * t["completeness"]).to_numpy()
        schemes = oof.emerging_schemes(train, year)
        log(f"emerging {line} [{train_set}]: {len(todo)} configs x {len(schemes)} schemes, {x.shape[1]} cols, "
            f"emerging rows {len(em_idx):,}")
        for name, cfg in todo:
            for scheme, k, em_fit_mask, em_val_mask in schemes:
                em_fit = np.zeros(len(x), dtype=bool)
                em_fit[em_idx[em_fit_mask]] = True
                load_f = t.loc[em_fit_mask, "claim_cost"].sum() / t.loc[em_fit_mask, "claim_cost_capped"].sum()
                actual = t.loc[em_val_mask, "claim_cost"].to_numpy()
                scale = comp_expo[em_val_mask] * load_f
                t0 = time.time()
                m = oof.emerging_model(line, name, train_set, cfg)
                oof.fit_emerging(m, x, y, w, is_dev | em_fit, em_fit, train_set, valid="inner")  # C027
                pred = m.predict(x.iloc[em_idx[em_val_mask]]) * scale
                row = dict(line=line, model=name, config=str(cfg), scheme=scheme, fold=k,
                           **oof.scores(actual, pred), iterations=getattr(m, "best_iteration", None),
                           sec=round(time.time() - t0))
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
                log(f"{line} {name} {cfg} {scheme} {k}: d2={row['d2']:.5f} d2_cal={row['d2_cal']:.5f} "
                    f"p/a={row['pred_actual']:.3f} it={row['iterations']} {row['sec']}s")
                del m

    summary = emerging_summary(pd.read_csv(csv, dtype={"config": str}))
    suffix = "" if train_set == "emerging" else f"_pooled_w{dev_weight:g}"
    with open(os.path.join(ROOT, "docs", f"cv_emerging_{family}{suffix}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Emerging-market CV tuning: {family} [{train_set}]\n\nSchemes: time fold (train 2023 -> score 2024) "
                f"and grouped {oof.GROUP_FOLDS}-fold CV on policy_id (seed {oof.GROUP_SEED}). Ranked within line and "
                "model by `d2_cal_group` (mean of the grouped folds, level-calibrated D2); `d2_cal_time` is the time "
                "fold; `gain_*` is the difference to that model's defaults (config `{}`). Generated by "
                "`scripts/tune_cv.py --market emerging` (C039); the script adopts no configuration.\n\n")
        f.write(summary.round(5).to_markdown(index=False))
        f.write("\n")
    log("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["trees", "linear"])
    ap.add_argument("lines", nargs="?", default=None, help="comma list of lines (default: all three)")
    ap.add_argument("--market", default=None, choices=sorted(markets.MARKETS),
                    help="C034: default from AIIAA_MARKET, else developed")
    ap.add_argument("--train-set", default="emerging", choices=["emerging", "pooled"], help="C039 (emerging market)")
    ap.add_argument("--developed-weight", type=float, default=0.3, help="C039: weight multiplier on developed rows")
    ap.add_argument("--models", default=None, help="C039: comma list of model names to search (default: all)")
    a = ap.parse_args()
    selected = a.lines.split(",") if a.lines else LINES
    if markets.get(a.market).name == "emerging":
        emerging_main(a.family, selected, a.train_set, a.developed_weight, a.models.split(",") if a.models else None)
    else:
        if a.train_set != "emerging":
            ap.error("--train-set pooled is an emerging-market option")
        main(a.family, selected)
