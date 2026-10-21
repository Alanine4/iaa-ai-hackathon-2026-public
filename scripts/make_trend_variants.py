"""C016: rescale the saved blend predictions to test how much of the trend factor (C009) to apply.

Validation showed predictions 5-21% above actual with the trend and close to actual without it,
so the public leaderboard is used to compare full, half (square root) and no trend.
Rescaling the saved test predictions needs no retraining: pred_variant = pred * trend ** (share - 1).

Usage: .venv\\Scripts\\python.exe scripts\\make_trend_variants.py
Writes: work/sub_blend_trend<share>.csv and .zip
"""
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import LINES, OUT, TEST_YEAR, line_trend, load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["glm", "gam", "lgbm", "xgb"]
SHARES = [0.0, 0.5]


def main():
    sample = load("sample_submission")
    parts = {}
    for line in LINES:
        t = pd.read_parquet(os.path.join(OUT, f"target_{line}.parquet"))
        year = pd.to_datetime(load(f"policies_{line}")["period_start"]).dt.year.to_numpy()
        trend = line_trend(t, year, [2021, 2022, 2023, 2024], TEST_YEAR)
        blend = pd.concat([pd.read_parquet(os.path.join(OUT, f"pred_{line}_{m}.parquet")).set_index("row_id")["prediction"]
                           for m in MODELS], axis=1).mean(axis=1)
        parts[line] = (blend, trend)
        print(f"{line}: full trend {trend:.4f}")

    os.makedirs(os.path.join(ROOT, "work"), exist_ok=True)
    for share in SHARES:
        sub = pd.concat([b * tr ** (share - 1) for b, tr in parts.values()]).reindex(sample["row_id"])
        assert sub.notna().all() and (sub >= 0).all() and len(sub) == len(sample)
        name = f"sub_blend_trend{int(share * 100):03d}"
        csv = os.path.join(ROOT, "work", f"{name}.csv")
        sub.rename("prediction").reset_index().to_csv(csv, index=False)
        with zipfile.ZipFile(os.path.join(ROOT, "work", f"{name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(csv, f"{name}.csv")
        print(f"{name}: mean {sub.mean():.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
