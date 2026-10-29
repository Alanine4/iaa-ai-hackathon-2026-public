"""C041: emerging blend over the models of several pooled run folders (cross-run blend), from existing files only.

For each line the OOF files `output/emerging/<run>/oof_<line>.parquet` of every run are aligned on
(row_id, scheme, fold, actual) (the runs used the same folds), every model column becomes a candidate `<run>:<model>`,
and the C036 rule is applied unchanged: weights searched on the grouped folds (0.05 grid), kept only if they beat equal
weights on the time fold AND on the grouped mean (`evaluate.guarded_weights`, index 0 time fold, index 1 grouped mean).
The result is compared with the current best blend (BASELINE, P2-S05) on the same folds; the adoption rule is a mean
gain over the two schemes of at least ADOPT_GAIN with no loss on either scheme. Runs may hold only some models.
The run "." is output/emerging/ itself (the emerging-only run).

C044: every blend, baseline and candidate is also scored on the test-mix views (`evaluate.nb_views`): d2_cal with
importance weights that give new-business rows the 2025 test file's NB exposure share of the line, and d2_cal on the NB
rows only. The OOF row info (new_business, exposure, claim_count) comes from the OOF files (runs from C044 on) or from
the emerging policy / target tables. Weights are still searched and guarded on the raw grouped folds only (77-697 NB
claims per line are too few to tune on). `--gate testmix`: adopt only if the raw time fold, the raw grouped mean AND the
test-mix grouped mean do not lose, and the mean raw gain >= ADOPT_GAIN OR the test-mix grouped gain >= ADOPT_GAIN.
`--candidates a:x,b:y` restricts the search to those candidates (the baseline is scored from all loaded columns).
`--report-views` prints the six views of the submitted recipes (RECIPES) per line and for the three lines stacked
(Kaggle-like D2), on the exposure basis and on the row basis of the audit (audit_p2 a21: NB rows to 77% of rows).

C047: `--baseline P2-S12` compares with the current recipe (contents pooled_w0.3 glm .35 + pooled_w1 lgbm .65; motor and
buildings as P2-S05); the default stays P2-S05 (BASELINE). `--segment new_business` searches, guards and gates the new
business rows (segment "nb") and the renewal rows ("renewal") of each line separately: per segment the weights are
searched on that segment's rows of the raw grouped folds and guarded on its time fold and grouped mean (both within the
segment); the gate scores the whole line with that segment's blend on its rows and the baseline on the other segment's
rows, with every view as before; the NB segment must also not lose on the NB grouped view (the test-mix and NB views
agree). The submission weights are then {"nb": ..., "renewal": ...}: NB test rows take the nb blend, the others the
renewal blend (`line_prediction`, `--write-from`).

`--write` builds work/<name>.csv/.zip from the runs' pred files, each re-expressed at `--trend`
(trend factor / the file's trend_factor), checked against the emerging sample_submission like blend.py. Per line the
submission uses the cross-run weights when they pass the adoption rule, otherwise the BASELINE weights.

C057: `--pred-tag s<N>` (with `--write-from`) reads the seed-averaged files pred_<line>_<model>_s<N>.parquet written by
`oof_predictions.py --market emerging --refit-only --seeds N`. GLM / GAM are not re-fit per seed, so a missing tagged
glm / gam file falls back to the untagged one; a missing tagged file of any other model is an error. One log line per
line gives the tag, the number of tagged candidate files and the fallbacks.

C058: `--level-from <submission zip or csv>:<line>[,<line>]` (with `--write-from`) re-scales each listed line so that
its total equals the total of the same rows in the given submission file. This is the C054 convention: an adopted line
replaces a line of the current best submission at that line's confirmed level (the P2-S13 level probes). The rows of a
line are the row_ids of its prediction and all of them must be in the file; unlisted lines are untouched. One log line
per listed line gives the factor and the mean before and after.

Usage: .venv\\Scripts\\python.exe scripts\\blend_runs.py motor,buildings,contents --runs pooled_w0.3,pooled_w1
       [--trend developed-half] [--name p2_xrun] [--step 0.05] [--out xrun] [--write]
       [--gate raw|testmix] [--candidates run:model,...] [--report-views]
       .venv\\Scripts\\python.exe scripts\\blend_runs.py motor,buildings,contents
       --write-from motor=<out>,buildings=<out>,contents=<out> --trend mixed-full --name <name>
       C047: [--baseline P2-S05|P2-S12] [--segment new_business]
       C057: [--pred-tag s5]   (with --write-from: read the seed-averaged pred files)
       C058: [--level-from submissions\\part2_candidate_P2-S13_p2_c047.zip:buildings]   (with --write-from)
Writes: output/emerging/<out>/blend_weights_<line>.json; with --write work/<name>.csv and work/<name>.zip;
        with --report-views output/emerging/<out>/views_recipes.csv
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import blend  # noqa: E402
from oof_predictions import SEED_KWARG  # noqa: E402  (C059)
from train_validate import LINES, clean, evaluate, load, markets  # noqa: E402

KEYS = ["row_id", "scheme", "fold", "actual"]
META = ["new_business", "exposure", "claim_count"]  # C044: OOF row info, not candidates
SCHEMES = ("time", "group")
ADOPT_GAIN = 0.002  # mean d2_cal gain over the two schemes needed for adoption (no loss on either scheme)
GATES = ("raw", "testmix")  # C044
# P2-S05 (work/p2_pooled_bw1.zip): motor and contents from pooled_w0.3, buildings from pooled_w1 (C040)
BASELINE = {
    "motor": {"pooled_w0.3:glm": 0.25, "pooled_w0.3:lgbm": 0.25, "pooled_w0.3:xgb": 0.25, "pooled_w0.3:cat": 0.25},
    "buildings": {"pooled_w1:cat": 0.95, "pooled_w1:lgbm": 0.05},
    "contents": {"pooled_w0.3:glm": 0.55, "pooled_w0.3:xgb": 0.25, "pooled_w0.3:lgbm": 0.2},
}
# C047: the current recipe (P2-S12, work/p2_c044_contents_xrun): contents from the C044 two-run blend
BASELINES = {"P2-S05": BASELINE,
             "P2-S12": {**BASELINE, "contents": {"pooled_w0.3:glm": 0.35, "pooled_w1:lgbm": 0.65}}}
SEGMENTS = ("nb", "renewal")  # C047: new-business rows, renewal rows
# C044: the OOF recipes of the submitted Part 2 files; "." = emerging-only run (output/emerging/blend_weights_*.json)
EMERGING_ONLY = {"motor": {".:gam": 0.1, ".:lgbm": 0.1, ".:xgb": 0.2, ".:cat": 0.6},
                 "buildings": {".:gam": 0.05, ".:xgb": 0.3, ".:cat": 0.65},
                 "contents": {".:glm": 0.15, ".:gam": 0.05, ".:lgbm": 0.4, ".:cat": 0.4}}
RECIPES = {
    "P2-S02": EMERGING_ONLY,  # P2-S01 / S02 differ only in trend
    "P2-S03": {**EMERGING_ONLY, "contents": BASELINE["contents"]},
    "P2-S04": {"motor": BASELINE["motor"], "buildings": {"pooled_w0.3:cat": 0.8, "pooled_w0.3:lgbm": 0.2},
               "contents": BASELINE["contents"]},
    "P2-S05": BASELINE,
}
# audit_p2 a21 (row basis): grouped test-mix steps, pooled three lines
AUDIT_STEPS = {("P2-S03", "P2-S04"): 0.0051, ("P2-S04", "P2-S05"): 0.0004}
AUDIT_ROW_SHARE = 0.77
VIEW_KEYS = ("time", "group_mean", "time_testmix", "group_testmix", "time_nb", "group_nb")


def run_path(run):
    return os.path.join(markets.get("emerging").out_dir, run)


def row_info(line):
    """C044: new_business (emerging policies), exposure (C001, target) and claim_count of the emerging training rows,
    indexed by row_id."""
    em = markets.get("emerging")
    pol = pd.read_parquet(os.path.join(em.data_dir, f"policies_{line}.parquet"), columns=["row_id", "new_business"])
    t = pd.read_parquet(os.path.join(em.out_dir, f"target_{line}.parquet"), columns=["row_id", "exposure", "claim_count"])
    assert (pol["row_id"].to_numpy() == t["row_id"].to_numpy()).all(), f"{line}: policies and target rows differ"
    return pd.concat([pol.set_index("row_id"), t.set_index("row_id")], axis=1)[META]


def nb_share_2025(line):
    """C044: new-business share of the 2025 test exposure of a line (C001 recomputed exposure)."""
    test = clean.clean_policies(load(f"test_{line}", "emerging"))
    return evaluate.nb_exposure_share(test["new_business"].to_numpy(dtype=bool), test["exposure"].to_numpy())


def load_runs(line, runs):
    """OOF frames of the runs aligned on KEYS (sorted by scheme, fold, row_id), then the META row info, then one
    column per `<run>:<model>`. A run folder without this line's OOF file (e.g. a GLM-only run of another line) is
    skipped; a missing folder fails. C044: row info comes from the first run whose OOF holds it (later runs holding it
    must agree), otherwise from `row_info`."""
    base, meta, cols, names = None, None, [], []
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
        if set(META) <= set(o.columns):
            if meta is None:
                meta = o[META]
            else:
                assert o[META].equals(meta), f"{line}: row info {META} of {run} differs from an earlier run"
        models = [c for c in o.columns if c not in KEYS + META]
        cols.append(o[models].rename(columns={m: f"{run}:{m}" for m in models}))
        names += [f"{run}:{m}" for m in models]
    if meta is None:
        meta = row_info(line).loc[base["row_id"]].reset_index(drop=True)
    return pd.concat([base, meta] + cols, axis=1), names


def scheme_folds(frame, names):
    """The `folds` structure of emerging_main per scheme: fold order 0..k, {"actual", "preds": {name: array}} plus the
    META arrays (C044)."""
    out = {}
    for scheme in SCHEMES:
        part = frame[frame["scheme"] == scheme]
        out[scheme] = [{"actual": g["actual"].to_numpy(), "preds": {n: g[n].to_numpy() for n in names},
                        **{c: g[c].to_numpy() for c in META}} for _, g in part.groupby("fold", sort=True)]
    return out


def restrict(folds, names):
    """Folds with only the named candidates (C044 `--candidates`)."""
    return {s: [{**f, "preds": {n: f["preds"][n] for n in names}} for f in folds[s]] for s in folds}


def segment_mask(new_business, segment):
    """C047: rows of one segment: "nb" = new business, "renewal" = the others."""
    nb = np.asarray(new_business, dtype=bool)
    return nb if segment == "nb" else ~nb


def segment_folds(folds, segment):
    """C047: the folds restricted to the rows of one segment (every array of a fold, candidates included)."""
    out = {}
    for scheme in folds:
        out[scheme] = []
        for f in folds[scheme]:
            m = segment_mask(f["new_business"], segment)
            out[scheme].append({**{k: np.asarray(v)[m] for k, v in f.items() if k != "preds"},
                                "preds": {n: np.asarray(p)[m] for n, p in f["preds"].items()}})
    return out


def fold_pred(f, weights):
    """Blend of one fold (names missing from `weights` weigh 0). C047: by-segment weights {"nb": {...}, "renewal":
    {...}} take the nb blend on new-business rows and the renewal blend on the others."""
    if set(weights) == set(SEGMENTS):
        nb = segment_mask(f["new_business"], "nb")
        return np.where(nb, fold_pred(f, weights["nb"]), fold_pred(f, weights["renewal"]))
    return sum(wt * f["preds"][n] for n, wt in weights.items() if wt > 0)


def blend_scores(folds, weights, share):
    """C044: one blend (names missing from `weights` weigh 0; C047: or by-segment weights, `fold_pred`) on the raw
    d2_cal and the two test-mix views: time fold, grouped mean (and per-fold lists), NB claim counts."""
    out = {}
    for scheme in SCHEMES:
        raw, mix, nbv, claims = [], [], [], 0
        for f in folds[scheme]:
            p = fold_pred(f, weights)
            raw.append(evaluate.d2_cal(f["actual"], p))
            v = evaluate.nb_views(f["actual"], p, f["new_business"], f["exposure"], f["claim_count"], share)
            mix.append(v["d2_cal_testmix"])
            nbv.append(v["d2_cal_nb"])
            claims += v["nb_claims"]
        if scheme == "time":
            out.update(time=raw[0], time_testmix=mix[0], time_nb=nbv[0], nb_claims_time=claims)
        else:
            out.update(group_mean=float(np.mean(raw)), group_testmix=float(np.mean(mix)), group_nb=float(np.mean(nbv)),
                       group_folds=raw, group_testmix_folds=mix, group_nb_folds=nbv, nb_claims_group=claims)
    return out


def pooled_scores(folds_by_line, recipe, shares):
    """C044: Kaggle-like D2 of the lines stacked (audit_p2 a20 / a21): per fold each line's blend is level-calibrated on
    its own rows, the lines are stacked and scored raw, with each line's test-mix weights, and on the NB rows (each
    line calibrated within its NB rows). Time fold and grouped mean."""
    out = {}
    for scheme in SCHEMES:
        raw, mix, nbv = [], [], []
        for k in range(len(folds_by_line[next(iter(recipe))][scheme])):
            ys, ps, ws, yn, pn = [], [], [], [], []
            for line, w in recipe.items():
                f = folds_by_line[line][scheme][k]
                y, nb = f["actual"], f["new_business"].astype(bool)
                p = fold_pred(f, w)  # C047: flat or by-segment weights
                ys.append(y)
                ps.append(p * y.sum() / p.sum())
                ws.append(evaluate.testmix_weights(nb, f["exposure"], shares[line]))
                yn.append(y[nb])
                pn.append(p[nb] * y[nb].sum() / p[nb].sum())
            y, p = np.concatenate(ys), np.concatenate(ps)
            raw.append(float(evaluate.tweedie_d2(y, p)))
            mix.append(float(evaluate.tweedie_d2(y, p, np.concatenate(ws))))
            nbv.append(float(evaluate.tweedie_d2(np.concatenate(yn), np.concatenate(pn))))
        if scheme == "time":
            out.update(time=raw[0], time_testmix=mix[0], time_nb=nbv[0])
        else:
            out.update(group_mean=float(np.mean(raw)), group_testmix=float(np.mean(mix)), group_nb=float(np.mean(nbv)))
    return out


def adopt(gain, gate):
    """C041 / C044 adoption rule on the gains over the baseline. raw: mean raw gain (time fold, grouped mean) >=
    ADOPT_GAIN and no loss on either. testmix: no loss on the raw time fold, the raw grouped mean AND the test-mix
    grouped mean, and the mean raw gain >= ADOPT_GAIN OR the test-mix grouped gain >= ADOPT_GAIN."""
    raw_ok = gain["time"] >= 0 and gain["group_mean"] >= 0
    if gate == "raw":
        return bool(raw_ok and gain["mean"] >= ADOPT_GAIN)
    assert gate == "testmix", f"unknown gate {gate}"
    return bool(raw_ok and gain["group_testmix"] >= 0
                and (gain["mean"] >= ADOPT_GAIN or gain["group_testmix"] >= ADOPT_GAIN))


def adopt_segment(gain, gate, segment):
    """C047: the gate of one segment, on the whole-line gains of the assembled prediction (this segment's blend on its
    rows, the baseline elsewhere); the NB segment must also not lose on the NB grouped view, so the test-mix and NB
    views agree."""
    return bool(adopt(gain, gate) and (segment != "nb" or gain["group_nb"] >= 0))


GATE_RULES = {"raw": f"mean gain over time fold and grouped mean >= {ADOPT_GAIN} and no loss on either",
              "testmix": f"no loss on raw time fold, raw grouped mean and test-mix grouped mean; mean raw gain >= "
                         f"{ADOPT_GAIN} or test-mix grouped gain >= {ADOPT_GAIN}"}


def evaluate_line(folds, names, baseline, step, share, gate="raw"):
    """C036 search and guard over the candidates `names`, plus the comparison with the baseline blend on the same folds
    (the baseline may use any loaded column). C044: all six views for blend, baseline and candidates."""
    loaded = list(folds["group"][0]["preds"])
    missing = sorted(set(baseline) - set(loaded))
    assert not missing, f"baseline candidates {missing} are not in the loaded runs"
    sub = restrict(folds, names)
    searched = evaluate.blend_weights_grid(sub["group"], step=step)
    rep = {s: evaluate.blend_report(sub[s], searched, names) for s in SCHEMES}
    guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
    weights, rule = evaluate.guarded_weights(guard, searched, names)  # index 0 time fold, index 1 grouped mean
    views = {s: evaluate.view_report(sub[s], searched, names, share) for s in SCHEMES}  # C044, reported only
    testmix_guard = {b: [views["time"][b][0]["d2_cal_testmix"],
                         float(np.mean([f["d2_cal_testmix"] for f in views["group"][b]]))] for b in ("weighted", "equal")}
    blend_s, base_s = blend_scores(folds, weights, share), blend_scores(folds, baseline, share)
    gain = {k: blend_s[k] - base_s[k] for k in VIEW_KEYS}
    gain["mean"] = (gain["time"] + gain["group_mean"]) / 2
    nonzero = {n: v for n, v in weights.items() if v > 0}
    single = {}
    for n in names:
        s = blend_scores(folds, {n: 1.0}, share)
        single[n] = {k: s[k] for k in VIEW_KEYS}
    return {"weights": weights, "weights_searched": searched, "rule": rule,
            "guard_scores": {"index": ["time fold 2023->2024", "grouped 5-fold mean"], **guard,
                             "testmix": {"index": ["time fold test-mix", "grouped mean test-mix"], **testmix_guard}},
            "baseline": {"weights": dict(baseline), **base_s}, "blend": blend_s, "gain": gain,
            "adoption": {"passes": adopt(gain, gate), "gate": gate, "rule": GATE_RULES[gate]},
            "concentration": {"n_nonzero": len(nonzero), "max_weight": max(nonzero.values()),
                              "effective_n": float(1 / sum(v * v for v in nonzero.values()))},
            "single": single}


def segment_scores(folds, weights):
    """C047: raw d2_cal within one segment's folds (calibrated within the segment): time fold, grouped folds, mean."""
    group = [evaluate.d2_cal(f["actual"], fold_pred(f, weights)) for f in folds["group"]]
    return {"time": evaluate.d2_cal(folds["time"][0]["actual"], fold_pred(folds["time"][0], weights)),
            "group_mean": float(np.mean(group)), "group_folds": group}


def evaluate_segments(folds, names, baseline, step, share, gate="testmix"):
    """C047: per segment ("nb", "renewal") the C036 search (on the segment's rows of the raw grouped folds) and guard
    (the segment's time fold and grouped mean, against equal weights), then the gate (`adopt_segment`) on the whole line
    with this segment's guarded blend on its rows and the baseline on the other segment's rows. Also the within-segment
    scores of the blend, the baseline and every candidate (the per-segment model table), the claims of the segment, and
    the final assembled prediction (each segment's blend if adopted, otherwise the baseline) on every view."""
    loaded = list(folds["group"][0]["preds"])
    missing = sorted(set(baseline) - set(loaded))
    assert not missing, f"baseline candidates {missing} are not in the loaded runs"
    base_s = blend_scores(folds, baseline, share)
    out, chosen = {"baseline": {"weights": dict(baseline), **base_s}, "segments": {}}, {}
    for seg in SEGMENTS:
        seg_folds = segment_folds(folds, seg)
        sub = restrict(seg_folds, names)
        searched = evaluate.blend_weights_grid(sub["group"], step=step)
        rep = {s: evaluate.blend_report(sub[s], searched, names) for s in SCHEMES}
        guard = {b: [rep["time"][b][0], float(np.mean(rep["group"][b]))] for b in ("weighted", "equal")}
        weights, rule = evaluate.guarded_weights(guard, searched, names)
        other = [s for s in SEGMENTS if s != seg][0]
        blend_s = blend_scores(folds, {seg: weights, other: dict(baseline)}, share)
        gain = {k: blend_s[k] - base_s[k] for k in VIEW_KEYS}
        gain["mean"] = (gain["time"] + gain["group_mean"]) / 2
        passes = adopt_segment(gain, gate, seg)
        chosen[seg] = weights if passes else dict(baseline)
        nonzero = {n: v for n, v in weights.items() if v > 0}
        out["segments"][seg] = {
            "weights": weights, "weights_searched": searched, "rule": rule,
            "guard_scores": {"index": [f"{seg} rows: time fold", f"{seg} rows: grouped mean"], **guard},
            "within_segment": {"blend": segment_scores(seg_folds, weights),
                               "baseline": segment_scores(seg_folds, baseline)},
            "claims": {s: int(sum(f["claim_count"].sum() for f in seg_folds[s])) for s in SCHEMES},
            "rows": {s: int(sum(len(f["actual"]) for f in seg_folds[s])) for s in SCHEMES},
            "assembled": blend_s, "gain": gain,
            "adoption": {"passes": passes, "gate": gate,
                         "rule": GATE_RULES[gate] + ("; NB grouped view must not lose" if seg == "nb" else "")},
            "concentration": {"n_nonzero": len(nonzero), "max_weight": max(nonzero.values())},
            "single": {n: segment_scores(seg_folds, {n: 1.0}) for n in names}}
    final = blend_scores(folds, chosen, share)
    gain = {k: final[k] - base_s[k] for k in VIEW_KEYS}
    gain["mean"] = (gain["time"] + gain["group_mean"]) / 2
    out.update(final=final, final_gain=gain, submission_weights=chosen)
    return out


def nb_flags_2025(line):
    """C047: new_business flag of the 2025 test rows of a line, indexed by row_id."""
    test = clean.clean_policies(load(f"test_{line}", "emerging"))
    return pd.Series(test["new_business"].to_numpy(dtype=bool), index=test["row_id"].to_numpy())


def pred_path(line, cand, pred_tag=None):
    """C057: pred file of one candidate `<run>:<model>` and whether it is a tagged file. With a tag the file is
    pred_<line>_<model>_<tag>.parquet. C059: only the models that are re-fit per seed (SEED_KWARG: lgbm, xgb, cat) have
    a tagged file; every other model (glm, gam, the OBLEMOS ob_glm / ob_gam, ...) is deterministic, so a missing
    tagged file falls back to the untagged one; a missing tagged file of a seeded model is an error."""
    run, model = cand.split(":", 1)
    base = os.path.join(run_path(run), f"pred_{line}_{model}")
    if pred_tag is None:
        return f"{base}.parquet", False
    tagged = f"{base}_{pred_tag}.parquet"
    if os.path.exists(tagged):
        return tagged, True
    seeded = model in SEED_KWARG
    assert not seeded, f"{line}: candidate {cand} has no tagged pred file {tagged}"
    return f"{base}.parquet", False


def used_candidates(weights):
    """C057: the candidates with a positive weight in flat or by-segment weights."""
    flat = weights.values() if set(weights) == set(SEGMENTS) else [weights]
    return sorted({cand for w in flat for cand, wt in w.items() if wt > 0})


def line_prediction(line, weights, trend, pred_tag=None):
    """Blended 2025 prediction of one line from the runs' pred files, each re-expressed at `trend` (C037).
    C047: by-segment weights take the nb blend on new-business test rows and the renewal blend on the others.
    C057: `pred_tag` selects the seed-averaged files (`pred_path`)."""
    if set(weights) == set(SEGMENTS):
        parts = {s: line_prediction(line, weights[s], trend, pred_tag) for s in SEGMENTS}
        renewal = parts["renewal"].reindex(parts["nb"].index)
        nb = nb_flags_2025(line).reindex(parts["nb"].index)
        assert renewal.notna().all() and nb.notna().all(), f"{line}: segment predictions cover different rows"
        return parts["nb"].where(nb.astype(bool), renewal)
    factor = markets.get("emerging").trends[trend][line]
    cols = []
    for cand, wt in weights.items():
        if wt <= 0:
            continue
        path, _ = pred_path(line, cand, pred_tag)
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


def parse_write_from(text):
    """C044: `--write-from motor=<out>,buildings=<out>,contents=<out>` as {line: result folder}, every line once."""
    pairs = [p.split("=", 1) for p in text.split(",")]
    spec = {k.strip(): v.strip() for k, v in (p for p in pairs if len(p) == 2)}
    if len(pairs) != len(LINES) or sorted(spec) != sorted(LINES):
        raise ValueError(f"--write-from needs line=<result folder> for exactly the lines {LINES}, got {text!r}")
    return spec


def parse_level_from(text):
    """C058: `--level-from <submission zip or csv>:<line>[,<line>]` as (path, [lines]). The path ends at the LAST colon
    (a Windows drive letter has one as well)."""
    path, _, names = text.rpartition(":")
    lines = [n.strip() for n in names.split(",")]
    if not path or not set(lines) <= set(LINES) or len(set(lines)) != len(lines):
        raise ValueError(f"--level-from needs <submission zip or csv>:<line>[,<line>] with lines from {LINES}, "
                         f"got {text!r}")
    return path, lines


def read_level_file(path):
    """C058: row_id -> prediction of a submission file (a zip holding one csv, or a csv); row_id is read as text."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            assert len(members) == 1, f"{path}: expected one csv in the zip, found {members}"
            frame = pd.read_csv(io.BytesIO(z.read(members[0])), dtype={"row_id": str})
    else:
        frame = pd.read_csv(path, dtype={"row_id": str})
    assert frame["row_id"].is_unique, f"{path}: duplicate row_id"
    return frame.set_index("row_id")["prediction"]


def level_to_file(pred, reference, line, path):
    """C058: `pred` (the rows of one line) scaled so that its total equals the total of the same rows in `reference`
    (a row_id -> prediction series read from `path`); returns the scaled prediction and the factor."""
    absent = pred.index.difference(reference.index)
    assert len(absent) == 0, f"{line}: {len(absent)} of {len(pred)} row_ids are not in {path}, e.g. {list(absent[:3])}"
    assert pred.sum() > 0, f"{line}: the prediction total is {pred.sum()}, no level factor"
    factor = reference.loc[pred.index].sum() / pred.sum()
    return pred * factor, factor


def write_from(spec, trend, name, pred_tag=None, level_from=None):
    """C044: build work/<name> from the adopted `submission_weights` of one result folder per line (each written by
    an earlier evaluation run of this script), so lines may use different runs, candidates and gates; the folder name
    `baseline` takes the line's BASELINE weights. C057: `pred_tag` reads the seed-averaged pred files (`pred_path`).
    C058: `level_from` = (submission file, [lines]) re-scales those lines to the file's total of their rows."""
    reference = read_level_file(level_from[0]) if level_from else None
    by_line = {}
    for line in LINES:
        if spec[line] == "baseline":
            weights = dict(BASELINE[line])
        else:
            path = os.path.join(markets.get("emerging").out_dir, spec[line], f"blend_weights_{line}.json")
            with open(path, encoding="utf-8") as f:
                weights = json.load(f)["submission_weights"]
        by_line[line] = line_prediction(line, weights, trend, pred_tag)
        if pred_tag:  # C057
            used = used_candidates(weights)
            fallback = [c for c in used if not pred_path(line, c, pred_tag)[1]]
            print(f"{line}: pred tag {pred_tag}: {len(used) - len(fallback)} of {len(used)} candidate file(s) tagged; "
                  f"untagged fallback: {fallback}", flush=True)
        print(f"{line}: from {spec[line]}: submission weights {weights}; trend {trend}; "
              f"mean {by_line[line].mean():.2f} rows {len(by_line[line]):,}", flush=True)
        if level_from and line in level_from[1]:  # C058
            before = by_line[line].mean()
            by_line[line], factor = level_to_file(by_line[line], reference, line, level_from[0])
            print(f"{line}: level from {level_from[0]}: factor {factor:.4f}, mean {before:.2f} -> "
                  f"{by_line[line].mean():.2f}", flush=True)
    return write_submission(by_line, name)


def _fmt(d):
    return " ".join(f"{k} {d[k]:.5f}" for k in VIEW_KEYS)


def main(lines, runs, trend, name, step, out, write, baseline=None, gate="raw", candidates=None, segment=None):
    baseline = baseline or BASELINE
    out_dir = os.path.join(markets.get("emerging").out_dir, out)
    os.makedirs(out_dir, exist_ok=True)
    chosen = {}
    for line in lines:
        frame, loaded = load_runs(line, runs)
        names = loaded
        if candidates:  # C044: search over a subset of the loaded candidates
            missing = sorted(set(candidates) - set(loaded))
            assert not missing, f"{line}: candidates {missing} are not in the loaded runs"
            names = [n for n in loaded if n in candidates]
        folds = scheme_folds(frame, loaded)
        share = nb_share_2025(line)
        t0 = time.time()
        if segment:  # C047
            assert segment == "new_business", f"unknown segment {segment}"
            res = evaluate_segments(folds, names, baseline[line], step, share, gate)
            chosen[line] = res["submission_weights"]
            res = {"line": line, "runs": runs, "candidates": names, "step": step, "trend": trend,
                   "test_nb_share": share, "segment": segment, "search_seconds": round(time.time() - t0, 1), **res}
            with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print_segments(line, res)
            continue
        res = evaluate_line(folds, names, baseline[line], step, share, gate)
        chosen[line] = res["weights"] if res["adoption"]["passes"] else dict(baseline[line])
        res = {"line": line, "runs": runs, "candidates": names, "step": step, "trend": trend, "test_nb_share": share,
               "search_seconds": round(time.time() - t0, 1), **res, "submission_weights": chosen[line]}
        with open(os.path.join(out_dir, f"blend_weights_{line}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        b, x, g = res["baseline"], res["blend"], res["gain"]
        print(f"{line}: {len(names)} candidates, search {res['search_seconds']}s, test NB share {share:.4f}", flush=True)
        print(f"{line}: baseline  {_fmt(b)}\n{line}: cross-run {_fmt(x)}\n{line}: gain      {_fmt(g)} mean {g['mean']:+.5f}; "
              f"gate {gate} adopted {res['adoption']['passes']}; rule: {res['rule']}", flush=True)
        print(f"{line}: weights {{{', '.join(f'{k}: {v:g}' for k, v in res['weights'].items() if v > 0)}}}; "
              f"concentration {res['concentration']}", flush=True)
        for n, s in res["single"].items():
            print(f"{line}:   single {n:<22} {_fmt(s)}", flush=True)
    if write:
        assert sorted(lines) == sorted(LINES), "--write needs all three lines"
        by_line = {line: line_prediction(line, chosen[line], trend) for line in lines}
        for line, pred in by_line.items():
            print(f"{line}: submission weights {chosen[line]}; mean {pred.mean():.2f} rows {len(pred):,}")
        write_submission(by_line, name)


def _fmt_seg(d):
    return f"time {d['time']:.5f} group_mean {d['group_mean']:.5f}"


def print_segments(line, res):
    """C047: per-segment tables of a segment run."""
    print(f"{line}: {len(res['candidates'])} candidates, search {res['search_seconds']}s, "
          f"test NB share {res['test_nb_share']:.4f}", flush=True)
    print(f"{line}: baseline (whole line)  {_fmt(res['baseline'])}", flush=True)
    for seg, r in res["segments"].items():
        print(f"{line} [{seg}] rows {r['rows']} claims {r['claims']}; rule: {r['rule']}", flush=True)
        print(f"{line} [{seg}] weights {{{', '.join(f'{k}: {v:g}' for k, v in r['weights'].items() if v > 0)}}}",
              flush=True)
        print(f"{line} [{seg}] within segment: blend {_fmt_seg(r['within_segment']['blend'])}; "
              f"baseline {_fmt_seg(r['within_segment']['baseline'])}", flush=True)
        print(f"{line} [{seg}] assembled {_fmt(r['assembled'])}", flush=True)
        print(f"{line} [{seg}] gain      {_fmt(r['gain'])} mean {r['gain']['mean']:+.5f}; adopted "
              f"{r['adoption']['passes']}", flush=True)
        for n, s in r["single"].items():
            print(f"{line} [{seg}]   single {n:<26} {_fmt_seg(s)}", flush=True)
    print(f"{line}: final     {_fmt(res['final'])}\n{line}: final gain {_fmt(res['final_gain'])} "
          f"mean {res['final_gain']['mean']:+.5f}; submission weights {res['submission_weights']}", flush=True)


def report_views(lines, out, recipes=None):
    """C044: the six views of each recipe per line and for the lines stacked, on the exposure basis (test file NB
    exposure share; the C044 definition) and on the audit's row basis (NB rows to 77% of rows, audit_p2 a21); steps
    between consecutive recipes and the check against the audit's grouped test-mix steps."""
    recipes = recipes or RECIPES
    runs = sorted({c.split(":", 1)[0] for rec in recipes.values() for w in rec.values() for c in w})
    folds, shares = {}, {}
    for line in lines:
        frame, names = load_runs(line, runs)
        folds[line] = scheme_folds(frame, names)
        shares[line] = nb_share_2025(line)
    row_basis = {line: {s: [{**f, "exposure": np.ones(len(f["actual"]))} for f in folds[line][s]] for s in SCHEMES}
                 for line in lines}
    rows = []
    for basis, fl, sh in (("exposure", folds, shares), ("rows", row_basis, {ln: AUDIT_ROW_SHARE for ln in lines})):
        for rec_name, rec in recipes.items():
            for line in lines:
                s = blend_scores(fl[line], rec[line], sh[line])
                rows.append({"basis": basis, "recipe": rec_name, "line": line, **{k: s[k] for k in VIEW_KEYS},
                             "nb_claims": s["nb_claims_group"]})
            if len(lines) > 1:
                rows.append({"basis": basis, "recipe": rec_name, "line": "pooled",
                             **pooled_scores(fl, {ln: rec[ln] for ln in lines}, sh)})
    table = pd.DataFrame(rows)
    out_dir = os.path.join(markets.get("emerging").out_dir, out)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "views_recipes.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"test NB exposure share per line: {shares}")
    print(table.round(5).to_string(index=False))
    names = list(recipes)
    for basis in ("exposure", "rows"):
        t = table[(table["basis"] == basis) & (table["line"] == ("pooled" if len(lines) > 1 else lines[0]))]
        t = t.set_index("recipe")
        for a, b in zip(names, names[1:]):
            d = t.loc[b, list(VIEW_KEYS)] - t.loc[a, list(VIEW_KEYS)]
            check = ""
            if basis == "rows" and (a, b) in AUDIT_STEPS and len(lines) == 3:
                ok = abs(d["group_testmix"] - AUDIT_STEPS[(a, b)]) < 0.00005
                check = f"  audit {AUDIT_STEPS[(a, b)]:+.4f}: {'reproduced' if ok else 'NOT reproduced'}"
            print(f"{basis} {a}->{b}: " + " ".join(f"{k} {d[k]:+.5f}" for k in VIEW_KEYS) + check)
    return table


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("lines", help="comma list, e.g. motor,buildings,contents")
    ap.add_argument("--runs", default="pooled_w0.3,pooled_w1", help="run folders under output/emerging/ (. = itself)")
    ap.add_argument("--trend", default="developed-half", choices=sorted(markets.get("emerging").trends),
                    help="C037 / C043")
    ap.add_argument("--name", default="p2_xrun", help="submission name under work/ (with --write)")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default="xrun", help="result folder under output/emerging/")
    ap.add_argument("--write", action="store_true", help="also write work/<name>.csv/.zip")
    ap.add_argument("--gate", default="raw", choices=GATES, help="C044 adoption gate")
    ap.add_argument("--candidates", default=None, help="C044: comma list of <run>:<model> to search over")
    ap.add_argument("--report-views", action="store_true",
                    help="C044: print the six views of the RECIPES (runs taken from the recipes) and exit")
    ap.add_argument("--baseline", default="P2-S05", choices=sorted(BASELINES),
                    help="C047: recipe the gate compares with (P2-S12 = current recipe)")
    ap.add_argument("--segment", default=None, choices=["new_business"],
                    help="C047: search, guard and gate NB rows and renewal rows separately")
    ap.add_argument("--write-from", default=None,
                    help="C044: motor=<out>,buildings=<out>,contents=<out>: write work/<name> from the submission_weights "
                         "of those result folders (no search; <out> = baseline: BASELINE weights) and exit")
    ap.add_argument("--pred-tag", default=None,
                    help="C057 (with --write-from): read pred_<line>_<model>_<tag>.parquet (seed-averaged files, e.g. "
                         "s5); glm / gam fall back to the untagged file, other models need the tagged file")
    ap.add_argument("--level-from", default=None,
                    help="C058 (with --write-from): <submission zip or csv>:<line>[,<line>]: re-scale those lines so "
                         "that each line's total equals the total of the same rows in that file")
    a = ap.parse_args()
    if a.pred_tag and not a.write_from:
        ap.error("C057: --pred-tag needs --write-from")
    if a.level_from and not a.write_from:
        ap.error("C058: --level-from needs --write-from")
    if a.report_views:
        report_views(a.lines.split(","), a.out)
    elif a.write_from:
        try:
            spec = parse_write_from(a.write_from)
        except ValueError as e:
            ap.error(str(e))
        try:
            level_from = parse_level_from(a.level_from) if a.level_from else None
        except ValueError as e:
            ap.error(str(e))
        write_from(spec, a.trend, a.name, pred_tag=a.pred_tag, level_from=level_from)
    else:
        main(a.lines.split(","), a.runs.split(","), a.trend, a.name, a.step, a.out, a.write,
             baseline=BASELINES[a.baseline], gate=a.gate, candidates=a.candidates.split(",") if a.candidates else None,
             segment=a.segment)
