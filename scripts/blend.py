"""C020/C021: blend saved test predictions, each re-expressed at the current trend share.

Weights: equal by default; with `--weighted`, per-line weights from output/blend_weights_<line>.json
(written by scripts/oof_predictions.py, C021).

Saved prediction files written after C021 carry a `trend_share` column. Older files do not;
FILE_TREND_SHARE records the share they were written with (checked against the run logs
`output/train_validate.log` and `output/train_trees_tuned.log`).

Usage: .venv\\Scripts\\python.exe scripts\\blend.py glm,gam,lgbm,xgb <name> [--weighted]
Writes: work/<name>.csv and work/<name>.zip
"""
import json
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from aiiaa import evaluate  # noqa: E402
from train_validate import LINES, OUT, TEST_YEAR, TREND_SHARE, line_trend, load  # noqa: E402

# glm/gam: run finished 16:34 (full trend, before C019); lgbm/xgb: tuned run finished 19:03 (half trend)
FILE_TREND_SHARE = {"glm": 1.0, "gam": 1.0, "lgbm": 0.5, "xgb": 0.5}


def line_weights(line, names, weighted):
    if not weighted:
        return {n: 1 / len(names) for n in names}
    with open(os.path.join(OUT, f"blend_weights_{line}.json"), encoding="utf-8") as f:
        w = json.load(f)["weights"]
    return {n: w.get(n, 0.0) for n in names}


def main(names, out_name, weighted=False):
    sample = load("sample_submission")
    parts = []
    for line in LINES:
        t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
        year = pd.to_datetime(load(f"policies_{line}")["period_start"]).dt.year.to_numpy()
        full = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR) ** (1 / TREND_SHARE)
        w = line_weights(line, names, weighted)
        assert abs(sum(w.values()) - 1) < 1e-9, f"{line} weights do not sum to 1: {w}"
        cols = []
        for n in names:
            f = pd.read_parquet(os.path.join(OUT, f"pred_{line}_{n}.parquet")).set_index("row_id")
            share = float(f["trend_share"].iloc[0]) if "trend_share" in f else FILE_TREND_SHARE[n]
            cols.append(pd.Series(evaluate.rescale_trend(f["prediction"].to_numpy(), full, share, TREND_SHARE),
                                  index=f.index, name=n))
        df = pd.concat(cols, axis=1)
        print(f"{line}: full trend {full:.4f}; weights {w}; mean by model "
              + ", ".join(f"{c} {df[c].mean():.2f}" for c in df))
        parts.append(sum(w[n] * df[n] for n in names))

    sub = pd.concat(parts).reindex(sample["row_id"])
    assert sub.notna().all() and (sub >= 0).all() and len(sub) == len(sample)
    os.makedirs(os.path.join(ROOT, "work"), exist_ok=True)
    csv = os.path.join(ROOT, "work", f"{out_name}.csv")
    sub.rename("prediction").reset_index().to_csv(csv, index=False)
    with zipfile.ZipFile(os.path.join(ROOT, "work", f"{out_name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, f"{out_name}.csv")
    print(f"{out_name}: rows {len(sub):,}, mean {sub.mean():.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1].split(","), sys.argv[2], weighted="--weighted" in sys.argv[3:])
