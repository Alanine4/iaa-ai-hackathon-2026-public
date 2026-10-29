"""Part 1 blend test: OBLEMOS improved4_Blend4 mixed into the team's Part 1 final S09.

The OBLEMOS prediction is first put on S09's level per line (row_id prefix B / C / M), so the blend tests ranking only;
then prediction = (1 - w) x S09 + w x OBLEMOS for w = 0.3 and 0.5.
Late-submission scores (Kaggle Part 1, 2026-10-28): w 0.3 -> private 0.13718 / public 0.12949, w 0.5 -> 0.13707 / 0.12971,
S09 alone 0.13689 / 0.12863, OBLEMOS alone 0.13526 / 0.12736.

Usage (repository root): .venv\\Scripts\\python.exe -I oblemos_part1\\blend_with_S09.py
Writes: work/part1_blend_S09_oblemos_<w>.csv
"""
import io
import os
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S09 = os.path.join(ROOT, "submissions", "part1_final_S09_sub_v3_seeds.zip")
OBLEMOS = os.path.join(ROOT, "oblemos_part1", "submissions", "part1_oblemos_improved4_Blend4.zip")


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))


def main():
    m = read_zip(S09).merge(read_zip(OBLEMOS), on="row_id", suffixes=("_s09", "_ob"), validate="one_to_one")
    line = m["row_id"].str[0]
    level = m.groupby(line)["prediction_s09"].transform("sum") / m.groupby(line)["prediction_ob"].transform("sum")
    m["ob_level"] = m["prediction_ob"] * level
    os.makedirs(os.path.join(ROOT, "work"), exist_ok=True)
    for w in (0.3, 0.5):
        out = m[["row_id"]].assign(prediction=(1 - w) * m["prediction_s09"] + w * m["ob_level"])
        path = os.path.join(ROOT, "work", f"part1_blend_S09_oblemos_{int(w * 100)}.csv")
        out.to_csv(path, index=False)
        print(f"{path}: {len(out):,} rows, mean {out['prediction'].mean():.2f}")


if __name__ == "__main__":
    main()
