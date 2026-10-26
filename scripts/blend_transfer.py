"""C047: cross-run blend of the C046 transfer run with the runs of the current best submission (P2-S12).

P2-S12 per line (docs/SUBMISSIONS.md): motor pooled_w0.3 glm/lgbm/xgb/cat 0.25 each, buildings pooled_w1 cat .95 /
lgbm .05, contents pooled_w0.3 glm .35 + pooled_w1 lgbm .65 (C044), trend mixed-full. Per line the candidates are the
P2-S12 components plus the three re-levelled developed bases of the transfer run; `blend_runs.main` searches the weights
on the grouped folds, guards them on the time fold and the grouped mean (C036) and adopts them only under the C044
test-mix gate against the P2-S12 recipe as baseline; otherwise the line keeps P2-S12. `--check` first rebuilds P2-S12
from the local runs and compares it with the submitted file.

Usage: .venv\\Scripts\\python.exe -I -u scripts\\blend_transfer.py [--name p2_transfer] [--gate testmix] [--check]
Writes: output/emerging/xrun_transfer/blend_weights_<line>.json, work/<name>.csv and .zip
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
import blend_runs as br  # noqa: E402

TREND = "mixed-full"
P2S12 = {"motor": dict(br.BASELINE["motor"]), "buildings": dict(br.BASELINE["buildings"]),
         "contents": {"pooled_w0.3:glm": 0.35, "pooled_w1:lgbm": 0.65}}
TRANSFER = ["transfer:tr_base", "transfer:tr_base_xgb", "transfer:tr_base_cat"]
OUT = "xrun_transfer"
SUBMITTED = os.path.join(br.markets.ROOT, "submissions", "part2_candidate_P2-S12_p2_c044_contents_xrun.zip")


def runs_of(weights):
    return sorted({c.split(":", 1)[0] for c in weights})


def check_p2s12():
    """Rebuild P2-S12 from the local run files and compare it with the submitted file, per line."""
    with zipfile.ZipFile(SUBMITTED) as z:
        sub = pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))).set_index("row_id")["prediction"]
    for line in br.LINES:
        local = br.line_prediction(line, P2S12[line], TREND)
        ref = sub.reindex(local.index)
        rel = (local / ref - 1).abs()
        print(f"P2-S12 check {line}: rows {len(local):,}, mean local {local.mean():.2f} vs submitted {ref.mean():.2f}, "
              f"median |rel diff| {rel.median():.4f}, corr {np.corrcoef(local, ref)[0, 1]:.5f}", flush=True)


def main(name, gate, check):
    if check:
        check_p2s12()
    for line in br.LINES:
        cands = list(P2S12[line]) + TRANSFER
        br.main([line], runs_of(cands), TREND, name, 0.05, OUT, False, baseline=P2S12, gate=gate, candidates=cands)
    br.write_from({line: OUT for line in br.LINES}, TREND, name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p2_transfer")
    ap.add_argument("--gate", default="testmix", choices=br.GATES)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    main(a.name, a.gate, a.check)
