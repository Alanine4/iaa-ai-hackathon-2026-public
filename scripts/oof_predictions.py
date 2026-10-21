"""C021: out-of-fold validation predictions of every model (tuned settings) and per-line blend weights.

Same folds and scoring as `tune_cv.py`: train 2021-22 -> validate 2023, train 2021-23 -> validate 2024.
Predictions are on the reported-cost scale (x exposure x completeness x large-loss load).

Usage: .venv\\Scripts\\python.exe scripts\\oof_predictions.py <line>
Writes: output/oof_<line>.parquet, output/blend_weights_<line>.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import OUT, clean, evaluate, load, log, models, params  # noqa: E402
from tune_cv import FOLDS, line_data  # noqa: E402

MODELS = ["glm", "gam", "lgbm", "xgb", "cat"]  # C025: CatBoost added


def main(line):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    x, t, year = line_data(line, shared)
    w = t["exposure_eff"].to_numpy()
    y = t["claim_cost_capped"].to_numpy() / w
    frames, folds = [], []
    for val_year, fit_years in FOLDS:
        fit, val = np.isin(year, fit_years), year == val_year
        load_f = t.loc[fit, "claim_cost"].sum() / t.loc[fit, "claim_cost_capped"].sum()
        actual = t.loc[val, "claim_cost"].to_numpy()
        scale = (t["exposure"] * t["completeness"]).to_numpy()[val] * load_f
        frame = pd.DataFrame({"row_id": t.loc[val, "row_id"].to_numpy(), "val_year": val_year, "actual": actual})
        for name in MODELS:
            t0 = time.time()
            m = models.make(name, **params.for_model(line, name))
            m.fit(x[fit], y[fit], w[fit], valid=(x[val], y[val], w[val]) if name in ("lgbm", "xgb") else None)
            frame[name] = m.predict(x[val]) * scale
            log(f"{line} {val_year} {name}: d2={evaluate.tweedie_d2(actual, frame[name]):.5f} {time.time() - t0:.0f}s")
        frames.append(frame)
        folds.append({"actual": actual, "preds": {n: frame[n].to_numpy() for n in MODELS}})
    pd.concat(frames).to_parquet(os.path.join(OUT, f"oof_{line}.parquet"), index=False)

    weights = evaluate.blend_weights(folds, step=0.05)
    equal = {n: 0.25 for n in MODELS}

    def score(wts):
        return np.mean([evaluate.tweedie_d2(f["actual"], p * f["actual"].sum() / p.sum())
                        for f in folds for p in [sum(wts[n] * f["preds"][n] for n in MODELS)]])

    result = {"line": line, "weights": weights, "cv_d2_cal_weighted": score(weights), "cv_d2_cal_equal": score(equal),
              "cv_d2_cal_single": {n: score({m: float(m == n) for m in MODELS}) for n in MODELS}}
    with open(os.path.join(OUT, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(json.dumps(result))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
