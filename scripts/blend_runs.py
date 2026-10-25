"""C041: emerging blend over the models of several pooled run folders (cross-run blend), from existing files only.

For each line the OOF files `output/emerging/<run>/oof_<line>.parquet` of every run are aligned on
(row_id, scheme, fold, actual) (the runs used the same folds), every model column becomes a candidate `<run>:<model>`,
and the C036 rule is applied unchanged: weights searched on the grouped folds (0.05 grid), kept only if they beat equal
weights on the time fold AND on the grouped mean (`evaluate.guarded_weights`, index 0 time fold, index 1 grouped mean).
The result is compared with the current best blend (BASELINE, P2-S05) on the same folds; the adoption rule is a mean
gain over the two schemes of at least ADOPT_GAIN with no loss on either scheme. Runs may hold only some models.

`--write` builds work/<name>.csv/.zip from the runs' pred files, each re-expressed at `--trend`
(trend factor / the file's trend_factor), checked against the emerging sample_submission like blend.py. Per line the
submission uses the cross-run weights when they pass the adoption rule, otherwise the BASELINE weights.

Usage: .venv\\Scripts\\python.exe scripts\\blend_runs.py motor,buildings,contents --runs pooled_w0.3,pooled_w1
       [--trend developed-half] [--name p2_xrun] [--step 0.05] [--out xrun] [--write]
Writes: output/emerging/<out>/blend_weights_<line>.json; with --write work/<name>.csv and work/<name>.zip
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import blend  # noqa: E402
from train_validate import LINES, evaluate, markets  # noqa: E402

KEYS = ["row_id", "scheme", "fold", "actual"]
SCHEMES = ("time", "group")
ADOPT_GAIN = 0.002  # mean d2_cal gain over the two schemes needed for adoption (no loss on either scheme)
# P2-S05 (work/p2_pooled_bw1.zip): motor and contents from pooled_w0.3, buildings from pooled_w1 (C040)
BASELINE = {
    "motor": {"pooled_w0.3:glm": 0.25, "pooled_w0.3:lgbm": 0.25, "pooled_w0.3:xgb": 0.25, "pooled_w0.3:cat": 0.25},
    "buildings": {"pooled_w1:cat": 0.95, "pooled_w1:lgbm": 0.05},
    "contents": {"pooled_w0.3:glm": 0.55, "pooled_w0.3:xgb": 0.25, "pooled_w0.3:lgbm": 0.2},
}


def run_path(run):
    return os.path.join(markets.get("emerging").out_dir, run)


def load_runs(line, runs):
    """OOF frames of the runs aligned on KEYS (sorted by scheme, fold, row_id), one column per `<run>:<model>`.
    A run folder without this line's OOF file (e.g. a GLM-only run of another line) is skipped; a missing folder fails."""
    base, cols, names = None, [], []
    for run in runs:
        assert os.path.isdir(run_path(run)), f"no run folder {run_path(run)}"
        path = os.path.join(run_path(run), f"oof_{line}.parquet")
        if not os.path.exists(path):
            print(f"{line}: run {run} has no oof_{line}.parquet, skipped", flush=True)
            continue
        o = pd.read_parquet(path)
        o = o.sort_values(["scheme", "fold", "row_id"]).reset_index(drop=True)
        keys = o[KEYS]
        if base is None:
            base = keys
        else:
            assert keys.equals(base), f"{line}: OOF frames of {names[0].split(':')[0]} and {run} do not align on {KEYS}"
        models = [c for c in o.columns if c not in KEYS]
        cols.append(o[models].rename(columns={m: f"{run}:{m}" for m in models}))
        names += [f"{run}:{m}" for m in models]
    return pd.concat([base] + cols, axis=1), names


def scheme_folds(frame, names):
    """The `folds` structure of emerging_main per scheme: fold order 0..k, {"actual", "preds": {name: array}}."""
    out = {}
    for scheme in SCHEMES:
        part = frame[frame["scheme"] == scheme]
        out[scheme] = [{"actual": g["actual"].to_numpy(), "preds": {n: g[n].to_numpy() for n in names}}
                       for _, g in part.groupby("fold", sort=True)]
    return out


def _time_group(folds, weights, names):
    """d2_cal of one blend as {"time": time fold, "group_mean": grouped-fold mean}."""
    full = {n: weights.get(n, 0.0) for n in names}
    rep = {s: evaluate.blend_report(folds[s], full, names)["weighted"] for s in SCHEMES}
    return {"time": rep["time"][0], "group_mean": float(np.mean(rep["group"]))}


def evaluate_line(folds, names, baseline, step):
    """C036 search and guard over the candidates, plus the comparison with the baseline blend on the same folds."""
    missing = sorted(set(baseline) - set(names))
    assert not missing, f"baseline candidates {missing} are not in the loaded runs"
    searched = evaluate.blend_weights_grid(folds["group"], step=step)
    rep = {s: evaluate.blend_report(folds[s], searched, names) for s in SCHEMES}
    guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
    weights, rule = evaluate.guarded_weights(guard, searched, names)  # index 0 time fold, index 1 grouped mean
    blend_s, base_s = _time_group(folds, weights, names), _time_group(folds, baseline, names)
    gain = {s: blend_s[s] - base_s[s] for s in blend_s}
    gain["mean"] = (gain["time"] + gain["group_mean"]) / 2
    passes = bool(gain["mean"] >= ADOPT_GAIN and gain["time"] >= 0 and gain["group_mean"] >= 0)
    nonzero = {n: v for n, v in weights.items() if v > 0}
    return {"weights": weights, "weights_searched": searched, "rule": rule,
            "guard_scores": {"index": ["time fold 2023->2024", "grouped 5-fold mean"], **guard},
            "baseline": {"weights": dict(baseline), **base_s}, "blend": blend_s, "gain": gain,
            "adoption": {"passes": passes,
                         "rule": f"mean gain over time fold and grouped mean >= {ADOPT_GAIN} and no loss on either"},
            "concentration": {"n_nonzero": len(nonzero), "max_weight": max(nonzero.values()),
                              "effective_n": float(1 / sum(v * v for v in nonzero.values()))},
            "single": {n: {"time": rep["time"]["single"][n][0],
                           "group_mean": float(np.mean(rep["group"]["single"][n]))} for n in names}}


def line_prediction(line, weights, trend):
    """Blended 2025 prediction of one line from the runs' pred files, each re-expressed at `trend` (C037)."""
    factor = markets.get("emerging").trends[trend][line]
    cols = []
    for cand, wt in weights.items():
        if wt <= 0:
            continue
        run, model = cand.split(":", 1)
        path = os.path.join(run_path(run), f"pred_{line}_{model}.parquet")
        assert os.path.exists(path), f"{line}: candidate {cand} has weight {wt} but no pred file {path}"
        f = pd.read_parquet(path).set_index("row_id")
        assert f["trend_factor"].nunique() == 1, f"{line} {cand}: mixed trend factors"
        cols.append((f["prediction"] * factor / f["trend_factor"].iloc[0]).rename(cand))
    df = pd.concat(cols, axis=1)
    assert df.notna().all().all(), f"{line}: pred files cover different rows"
    return sum(weights[c] * df[c] for c in df)


def write_submission(by_line, name):
    """Stack the line predictions, check them against the emerging sample_submission and write work/<name>."""
    sample = blend.load("sample_submission", "emerging")
    stacked = pd.concat(list(by_line.values()))
    assert stacked.index.is_unique and set(stacked.index) == set(sample["row_id"]), "row_ids differ from sample"
    sub = stacked.reindex(sample["row_id"])
    assert sub.notna().all() and (sub >= 0).all() and len(sub) == len(sample) == blend.EMERGING_ROWS
    assert (sub.index == sample["row_id"].to_numpy()).all()
    blend.write_submission(sub, name)
    return sub


def main(lines, runs, trend, name, step, out, write, baseline=None):
    baseline = baseline or BASELINE
    out_dir = os.path.join(markets.get("emerging").out_dir, out)
    os.makedirs(out_dir, exist_ok=True)
    chosen = {}
    for line in lines:
        frame, names = load_runs(line, runs)
        folds = scheme_folds(frame, names)
        t0 = time.time()
        res = evaluate_line(folds, names, baseline[line], step)
        chosen[line] = res["weights"] if res["adoption"]["passes"] else dict(baseline[line])
        res = {"line": line, "runs": runs, "candidates": names, "step": step, "trend": trend,
               "search_seconds": round(time.time() - t0, 1), **res, "submission_weights": chosen[line]}
        with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        b, x, g = res["baseline"], res["blend"], res["gain"]
        print(f"{line}: {len(names)} candidates, search {res['search_seconds']}s; baseline time {b['time']:.5f} "
              f"group {b['group_mean']:.5f}; cross-run time {x['time']:.5f} group {x['group_mean']:.5f}; "
              f"gain time {g['time']:+.5f} group {g['group_mean']:+.5f} mean {g['mean']:+.5f}; "
              f"adopted {res['adoption']['passes']}; rule: {res['rule']}", flush=True)
        print(f"{line}: weights {{{', '.join(f'{k}: {v:g}' for k, v in res['weights'].items() if v > 0)}}}; "
              f"concentration {res['concentration']}", flush=True)
    if write:
        assert sorted(lines) == sorted(LINES), "--write needs all three lines"
        by_line = {line: line_prediction(line, chosen[line], trend) for line in lines}
        for line, pred in by_line.items():
            print(f"{line}: submission weights {chosen[line]}; mean {pred.mean():.2f} rows {len(pred):,}")
        write_submission(by_line, name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("lines", help="comma list, e.g. motor,buildings,contents")
    ap.add_argument("--runs", default="pooled_w0.3,pooled_w1", help="run folders under output/emerging/")
    ap.add_argument("--trend", default="developed-half", choices=["none", "developed-half"], help="C037")
    ap.add_argument("--name", default="p2_xrun", help="submission name under work/ (with --write)")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default="xrun", help="result folder under output/emerging/")
    ap.add_argument("--write", action="store_true", help="also write work/<name>.csv/.zip")
    a = ap.parse_args()
    main(a.lines.split(","), a.runs.split(","), a.trend, a.name, a.step, a.out, a.write)
