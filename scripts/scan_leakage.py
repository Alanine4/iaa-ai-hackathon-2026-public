"""Scan every numeric/boolean policy column for in-period leakage (C008).

A leaking column looks different on rows with claims, and the test distribution matches the training
rows WITHOUT claims (test rows are written before any 2025 claim happens).

Usage: .venv\\Scripts\\python.exe scripts\\scan_leakage.py > docs\\scan_leakage.txt
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "iaa-ai-assisted-hackathon-2026-part-1")


def main():
    for line in ["motor", "buildings", "contents"]:
        p = pd.read_parquet(os.path.join(DATA, f"policies_{line}.parquet"))
        te = pd.read_parquet(os.path.join(DATA, f"test_{line}.parquet"))
        t = pd.read_parquet(os.path.join(ROOT, "output", f"target_{line}.parquet"), columns=["row_id", "claim_count"])
        p = p.merge(t, on="row_id")
        has = p["claim_count"] > 0
        rows = []
        for col in te.columns:
            if col in ("exposure",) or not (pd.api.types.is_numeric_dtype(te[col]) or pd.api.types.is_bool_dtype(te[col])):
                continue
            a, b = p[col].astype(float), te[col].astype(float)
            rows.append({
                "col": col,
                "no_claim_mean": a[~has].mean(), "claim_mean": a[has].mean(), "train_mean": a.mean(), "test_mean": b.mean(),
                "claim_vs_noclaim_%": 100 * (a[has].mean() / a[~has].mean() - 1) if a[~has].mean() else float("nan"),
                "test_closer_to_noclaim": abs(b.mean() - a[~has].mean()) < abs(b.mean() - a.mean()),
            })
        df = pd.DataFrame(rows).sort_values("claim_vs_noclaim_%", key=lambda s: s.abs(), ascending=False)
        print(f"\n######## {line}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
