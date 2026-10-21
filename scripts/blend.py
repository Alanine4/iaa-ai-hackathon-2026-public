"""C020: equal-weight blend of saved test predictions, each re-expressed at the current trend share.

Saved prediction files come from different runs; FILE_TREND_SHARE records the trend share each one was
written with (checked against the run logs `output/train_validate.log` and `output/train_trees_tuned.log`).

Usage: .venv\\Scripts\\python.exe scripts\\blend.py glm,gam,lgbm,xgb <name>
Writes: work/<name>.csv and work/<name>.zip
"""
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


def main(names, out_name):
    sample = load("sample_submission")
    parts = []
    for line in LINES:
        t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
        year = pd.to_datetime(load(f"policies_{line}")["period_start"]).dt.year.to_numpy()
        full = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR) ** (1 / TREND_SHARE)
        cols = []
        for n in names:
            p = pd.read_parquet(os.path.join(OUT, f"pred_{line}_{n}.parquet")).set_index("row_id")["prediction"]
            cols.append(pd.Series(evaluate.rescale_trend(p.to_numpy(), full, FILE_TREND_SHARE[n], TREND_SHARE),
                                  index=p.index, name=n))
        df = pd.concat(cols, axis=1)
        print(f"{line}: full trend {full:.4f}; mean by model " + ", ".join(f"{c} {df[c].mean():.2f}" for c in df))
        parts.append(df.mean(axis=1))

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
    main(sys.argv[1].split(","), sys.argv[2])
