"""C052: cross-run blend of the per-peril GLM run (`peril:pglm`, scripts/peril_glm.py) with the property recipes of the
current best submission P2-S13 (= P2-S12 for buildings and contents).

Per line `blend_runs.main` searches the weights over the P2-S13 components plus `peril:pglm` on the grouped folds,
guards them on the time fold and the grouped mean (C036) and adopts them only under the C044 testmix gate against the
P2-S13 recipe; otherwise the line keeps P2-S13. The submission is P2-S13 (teammate's file, motor NB segment included)
with only the adopted lines replaced by their blend, each re-expressed at trend mixed-full like every other run.

Usage: .venv\\Scripts\\python.exe -I -u scripts\\blend_peril.py [--lines contents,buildings] [--name p2_s13_peril]
Writes: output/emerging/xrun_peril/blend_weights_<line>.json, work/<name>.csv and .zip (when a line is adopted)
"""
import argparse
import io
import json
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
import blend_runs as br  # noqa: E402

TREND = "mixed-full"
P2S13 = {"buildings": {"pooled_w1:cat": 0.95, "pooled_w1:lgbm": 0.05},
         "contents": {"pooled_w0.3:glm": 0.35, "pooled_w1:lgbm": 0.65}}
PERIL = ["peril:pglm"]
OUT = "xrun_peril"
WORK = os.path.join(br.markets.ROOT, "work")
P2S13_FILE = os.path.join(WORK, "P2-S13_p2_c047.zip")  # teammate's submissions/part2_candidate_P2-S13_p2_c047.zip


def runs_of(weights):
    return sorted({c.split(":", 1)[0] for c in weights})


def main(lines, name):
    adopted = {}
    for line in lines:
        cands = list(P2S13[line]) + PERIL
        br.main([line], runs_of(cands), TREND, name, 0.05, OUT, False, baseline=P2S13, gate="testmix",
                candidates=cands)
        with open(os.path.join(br.run_path(OUT), f"blend_weights_{line}.json"), encoding="utf-8") as f:
            res = json.load(f)
        weights = res.get("submission_weights") or {}
        if weights and weights != P2S13[line]:
            adopted[line] = weights
        print(f"{line}: submission weights {weights}", flush=True)
    if not adopted:
        print("no line adopted: no submission written", flush=True)
        return
    with zipfile.ZipFile(P2S13_FILE) as z:
        sub = pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))).set_index("row_id")
    for line, weights in adopted.items():
        pred = br.line_prediction(line, weights, TREND)
        old = sub.loc[pred.index, "prediction"]
        print(f"{line}: {len(pred):,} rows replaced, mean {old.mean():.2f} -> {pred.mean():.2f}", flush=True)
        sub.loc[pred.index, "prediction"] = pred.to_numpy()
    sub = sub.reset_index()
    sub.to_csv(os.path.join(WORK, name + ".csv"), index=False)
    with zipfile.ZipFile(os.path.join(WORK, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name + ".csv", sub.to_csv(index=False))
    print(f"wrote work/{name}.zip: {len(sub):,} rows, mean {sub['prediction'].mean():.2f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="contents,buildings")
    ap.add_argument("--name", default="p2_s13_peril")
    a = ap.parse_args()
    main(a.lines.split(","), a.name)
