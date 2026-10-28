"""C051: emerging-market policyholder-age correction for the property lines on top of P2-S13.

The pooled runs take most of their age effect from the developed rows (buildings: developed weight 1.0). The emerging
OOF residuals show a different age pattern on both schemes (buildings: 35-49 over-predicted, 50+ under-predicted),
and the 2025 test rows are 50% under 35. The correction is a credibility-shrunk actual / expected relativity per
`policyholder_age_band` on the residuals of the line's P2-S13 recipe: rel = 1 + Z (A/E - 1), Z = claims / (claims + K),
so it shrinks towards 1, i.e. towards the pooled (developed-led) age effect.

Cross-fitting (no row scores a relativity learned from its own outcome):
- grouped folds: relativities from the other four grouped folds' OOF residuals (each fold calibrated to its own total);
- time fold (2023 -> 2024): relativities from the grouped OOF residuals of the 2023 rows only.
K is picked on the raw grouped mean from K_GRID; the line is adopted under the C044 testmix gate against the
uncorrected recipe. Test: relativities from all grouped OOF rows, applied to the P2-S13 line rows and re-scaled so the
line total is unchanged (the property levels were confirmed by the P2-S13 probes on 2026-10-28).

Resumable: each line's result is written to output/emerging/age_adjust/result_<line>.json and skipped when present
(`--redo` recomputes). Usage: .venv\\Scripts\\python.exe -I -u scripts\\age_adjust.py [--lines buildings,contents]
Writes: output/emerging/age_adjust/result_<line>.json, work/p2_s13_age.csv and .zip (adopted lines only)
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("AIIAA_MARKET", "emerging")
import blend_runs as br  # noqa: E402

RECIPE = {"buildings": {"pooled_w1:cat": 0.95, "pooled_w1:lgbm": 0.05},  # P2-S12 = P2-S13 property lines
          "contents": {"pooled_w0.3:glm": 0.35, "pooled_w1:lgbm": 0.65}}
AGE = "policyholder_age_band"
K_GRID = (25, 50, 100, 200, 400)
OUT = os.path.join(br.markets.get("emerging").out_dir, "age_adjust")
WORK = os.path.join(br.markets.ROOT, "work")
# teammate's submissions/part2_candidate_P2-S13_p2_c047.zip (origin/main 443f6d5), copied to work/
P2S13 = os.path.join(WORK, "P2-S13_p2_c047.zip")


def relativities(band, actual, pred, k):
    """Credibility-shrunk A/E per band; `pred` must already be on the actual's level."""
    g = pd.DataFrame({"b": band, "A": actual, "E": pred, "n": actual > 0}).groupby("b").sum()
    return 1 + g["n"] / (g["n"] + k) * (g["A"] / g["E"] - 1)


def line_frame(line):
    """OOF of the recipe per (scheme, fold, row) with age band, start year and the C044 row info."""
    runs = sorted({c.split(":", 1)[0] for c in RECIPE[line]})
    frame, _ = br.load_runs(line, runs)
    frame["pred"] = sum(w * frame[c] for c, w in RECIPE[line].items())
    pol = pd.read_parquet(os.path.join(br.markets.get("emerging").data_dir, f"policies_{line}.parquet"),
                          columns=["row_id", AGE, "period_start"]).set_index("row_id")
    frame["band"] = pol[AGE].astype(str).reindex(frame["row_id"]).to_numpy()
    frame["year"] = pd.to_datetime(pol["period_start"]).dt.year.reindex(frame["row_id"]).to_numpy()
    # per-fold calibration so the residuals measure ranking only
    frame["pred_cal"] = frame["pred"] * frame.groupby(["scheme", "fold"])["actual"].transform("sum") \
        / frame.groupby(["scheme", "fold"])["pred"].transform("sum")
    return frame


def corrected(frame, k):
    """Cross-fitted corrected predictions (column aligned with frame)."""
    out = frame["pred"].to_numpy().copy()
    grp = frame[frame["scheme"] == "group"]
    for fold in sorted(grp["fold"].unique()):
        fit = grp[grp["fold"] != fold]
        rel = relativities(fit["band"], fit["actual"], fit["pred_cal"], k)
        idx = (frame["scheme"] == "group") & (frame["fold"] == fold)
        out[idx.to_numpy()] *= frame.loc[idx, "band"].map(rel).fillna(1.0).to_numpy()
    fit = grp[grp["year"] == 2023]
    rel = relativities(fit["band"], fit["actual"], fit["pred_cal"], k)
    idx = (frame["scheme"] == "time").to_numpy()
    out[idx] *= frame.loc[idx, "band"].map(rel).fillna(1.0).to_numpy()
    return out


def folds_of(frame, preds):
    """The blend_runs folds structure with one prediction column per name."""
    out = {}
    for scheme in br.SCHEMES:
        part = frame["scheme"] == scheme
        out[scheme] = [{"actual": frame.loc[i, "actual"].to_numpy(),
                        "preds": {n: p[i.to_numpy()] for n, p in preds.items()},
                        **{c: frame.loc[i, c].to_numpy() for c in br.META}}
                       for f in sorted(frame.loc[part, "fold"].unique())
                       for i in [part & (frame["fold"] == f)]]
    return out


def evaluate_line(line):
    frame = line_frame(line)
    share = br.nb_share_2025(line)
    preds = {"base": frame["pred"].to_numpy()}
    preds.update({f"k{k}": corrected(frame, k) for k in K_GRID})
    folds = folds_of(frame, preds)
    scores = {n: br.blend_scores(folds, {n: 1.0}, share) for n in preds}
    keys = ("time", "group_mean", "time_testmix", "group_testmix", "time_nb", "group_nb")
    gains = {n: {key: scores[n][key] - scores["base"][key] for key in keys} for n in preds if n != "base"}
    for g in gains.values():
        g["mean"] = (g["time"] + g["group_mean"]) / 2
    best = max(gains, key=lambda n: gains[n]["group_mean"])
    k = int(best[1:])
    grp = frame[frame["scheme"] == "group"]
    rel = relativities(grp["band"], grp["actual"], grp["pred_cal"], k)
    res = {"line": line, "recipe": RECIPE[line], "test_nb_share": share, "k_grid": list(K_GRID), "picked_k": k,
           "adopted": br.adopt(gains[best], "testmix"), "gate": br.GATE_RULES["testmix"],
           "base_scores": {key: scores["base"][key] for key in keys},
           "gains": gains, "relativities_all_rows": rel.round(4).to_dict()}
    return res


def write_submission(results, name):
    with zipfile.ZipFile(P2S13) as z:
        sub = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    data = br.markets.get("emerging").data_dir
    for line, res in results.items():
        if not res["adopted"]:
            continue
        test = pd.read_parquet(os.path.join(data, f"test_{line}.parquet"), columns=["row_id", AGE])
        rel = test.set_index("row_id")[AGE].astype(str).map(res["relativities_all_rows"]).fillna(1.0)
        sel = sub["row_id"].isin(test["row_id"])
        before = sub.loc[sel, "prediction"].sum()
        sub.loc[sel, "prediction"] *= rel.reindex(sub.loc[sel, "row_id"]).to_numpy()
        sub.loc[sel, "prediction"] *= before / sub.loc[sel, "prediction"].sum()
        print(f"{line}: age correction applied to {sel.sum():,} test rows, line total unchanged", flush=True)
    sub.to_csv(os.path.join(WORK, name + ".csv"), index=False)
    with zipfile.ZipFile(os.path.join(WORK, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name + ".csv", sub.to_csv(index=False))
    print(f"wrote work/{name}.zip: {len(sub):,} rows, mean {sub['prediction'].mean():.2f}", flush=True)


def main(lines, redo, name):
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for line in lines:
        path = os.path.join(OUT, f"result_{line}.json")
        if os.path.exists(path) and not redo:
            print(f"{line}: result exists, loaded ({path})", flush=True)
            with open(path) as f:
                results[line] = json.load(f)
        else:
            results[line] = evaluate_line(line)
            with open(path, "w") as f:
                json.dump(results[line], f, indent=2)
        r = results[line]
        print(f"\n{line}: picked K={r['picked_k']}, adopted={r['adopted']}; relativities {r['relativities_all_rows']}")
        for n, g in r["gains"].items():
            print(f"   {n:5s} " + "  ".join(f"{key} {v:+.4f}" for key, v in g.items()), flush=True)
    if any(r["adopted"] for r in results.values()):
        write_submission(results, name)
    else:
        print("no line passed the gate: no submission written", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="buildings,contents")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--name", default="p2_s13_age")
    a = ap.parse_args()
    main(a.lines.split(","), a.redo, a.name)
