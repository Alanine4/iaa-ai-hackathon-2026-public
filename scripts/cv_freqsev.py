"""C023: time-based CV of the per-peril frequency x severity model (`src/aiiaa/freqsev.py`).

Same folds and scoring as `oof_predictions.py`: train 2021-22 -> validate 2023, train 2021-23 -> validate 2024;
Tweedie D2 of reported cost vs prediction on the reported-cost scale, and `d2_cal` after rescaling the
prediction to the fold's actual total.

Usage: .venv\\Scripts\\python.exe -I scripts\\cv_freqsev.py <line> [<line> ...]
Writes: output/oof_freqsev_<line>.parquet (row_id, val_year, actual, freqsev; same rows and order as
output/oof_<line>.parquet), output/cv_freqsev_<line>.csv, output/cv_freqsev_perils_<line>.csv,
docs/cv_freqsev.md (rebuilt from every line's csv files)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_validate import OUT, ROOT, clean, evaluate, load, log  # noqa: E402
from tune_cv import FOLDS, line_data  # noqa: E402

from aiiaa import freqsev  # noqa: E402  (src/ is on sys.path via train_validate)

# Best single model per line, mean CV d2_cal over the same two folds (C021, output/blend_weights_<line>.json)
BEST_SINGLE = {"motor": ("gam", 0.0893), "buildings": ("lgbm", 0.1094), "contents": ("xgb", 0.0911)}


def run_line(line, shared, claims):
    x, t, year = line_data(line, shared)
    p = clean.clean_policies(load(f"policies_{line}"))
    assert (p["row_id"].to_numpy() == t["row_id"].to_numpy()).all()
    pt = freqsev.peril_targets(p, claims[claims["line"] == line])
    perils = [c[2:] for c in pt.columns if c.startswith("n_")]
    # Peril targets reconcile with the line target table (same matching rules and the same C014 cap)
    assert (pt[[f"n_{q}" for q in perils]].sum(axis=1).to_numpy() == t["claim_count"].to_numpy()).all()
    assert np.allclose(pt[[f"cost_{q}" for q in perils]].sum(axis=1), t["claim_cost_capped"])
    assert np.allclose(pt[[f"cost_{q}_uncapped" for q in perils]].sum(axis=1), t["claim_cost"])
    log(f"{line}: {len(x):,} rows, {x.shape[1]} features, perils {perils}")

    w = t["exposure_eff"].to_numpy()
    frames, fold_rows, peril_rows = [], [], []
    for val_year, fit_years in FOLDS:
        t0 = time.time()
        fit, val = np.isin(year, fit_years), year == val_year
        m = freqsev.FreqSev().fit(x[fit], pt[fit], w[fit], valid=(x[val], pt[val], w[val]))
        # Scoring as oof_predictions.py: rate x exposure x completeness x load_f. exposure x completeness is
        # exposure_eff, so the expected reported cost is predict(x, exposure_eff). No per-line load_f: predict
        # already applies a load per peril, and on the fit rows sum_p load_p * capped_p = sum_p uncapped_p =
        # the line's uncapped total, i.e. the per-peril loads reproduce load_f in total while letting the
        # large-loss share differ by peril. Applying load_f as well would count the capped excess twice.
        pred = m.predict(x[val], w[val])
        actual = t.loc[val, "claim_cost"].to_numpy()
        sec = time.time() - t0
        fold = dict(line=line, val_year=val_year, d2=evaluate.tweedie_d2(actual, pred),
                    d2_cal=evaluate.tweedie_d2(actual, pred * actual.sum() / pred.sum()),
                    pred_actual=pred.sum() / actual.sum(), sec=round(sec))
        fold_rows.append(fold)
        log(f"{line} {val_year}: d2={fold['d2']:.5f} d2_cal={fold['d2_cal']:.5f} p/a={fold['pred_actual']:.3f} "
            f"{sec:.0f}s")

        r_fit, r_val = m.rates(x[fit]), m.rates(x[val])
        for q in perils:
            c = m.components_[q]
            n_f, cost_f = pt.loc[fit, f"n_{q}"].to_numpy(), pt.loc[fit, f"cost_{q}"].to_numpy()
            n_v, cost_v = pt.loc[val, f"n_{q}"].to_numpy(), pt.loc[val, f"cost_{q}"].to_numpy()
            unc_v = pt.loc[val, f"cost_{q}_uncapped"].to_numpy()
            f_f, s_f = r_fit[f"freq_{q}"].to_numpy(), r_fit[f"sev_{q}"].to_numpy()
            f_v, s_v = r_val[f"freq_{q}"].to_numpy(), r_val[f"sev_{q}"].to_numpy()
            pred_q = f_v * w[val] * s_v * c["load"]
            row = dict(
                line=line, val_year=val_year, peril=q, fit_claims=c["n_claims"],
                severity="constant" if isinstance(c["sev"], float) else "gamma",
                freq_it=c["freq_iterations"], sev_it=c["sev_iterations"], load=c["load"],
                # predicted / actual on the training rows (1 by construction for count and cost)
                fit_count=(f_f * w[fit]).sum() / n_f.sum(), fit_cost=(f_f * w[fit] * s_f).sum() / cost_f.sum(),
                fit_sev_given_n=(n_f * s_f).sum() / cost_f.sum(),
                # predicted / actual on the validation rows
                val_count=(f_v * w[val]).sum() / n_v.sum(), val_sev_given_n=(n_v * s_v).sum() / cost_v.sum(),
                val_cost=(pred_q / c["load"]).sum() / cost_v.sum(),
                val_cost_share=unc_v.sum() / actual.sum(),
                val_d2_cal=evaluate.tweedie_d2(unc_v, pred_q * unc_v.sum() / pred_q.sum()))
            peril_rows.append(row)
            log(f"  {q}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in row.items() if k not in ("line", "val_year", "peril")))
        frames.append(pd.DataFrame({"row_id": t.loc[val, "row_id"].to_numpy(), "val_year": val_year,
                                    "actual": actual, "freqsev": pred}))

    oof = pd.concat(frames, ignore_index=True)
    ref_path = os.path.join(OUT, f"oof_{line}.parquet")
    if os.path.exists(ref_path):
        ref = pd.read_parquet(ref_path, columns=["row_id", "val_year", "actual"])
        assert (ref["row_id"].to_numpy() == oof["row_id"].to_numpy()).all()
        assert (ref["val_year"].to_numpy() == oof["val_year"].to_numpy()).all()
        assert np.allclose(ref["actual"], oof["actual"])
        log(f"{line}: row set and order match {ref_path}")
    oof.to_parquet(os.path.join(OUT, f"oof_freqsev_{line}.parquet"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, f"cv_freqsev_{line}.csv"), index=False)
    pd.DataFrame(peril_rows).to_csv(os.path.join(OUT, f"cv_freqsev_perils_{line}.csv"), index=False)


def write_doc():
    folds = [pd.read_csv(os.path.join(OUT, f"cv_freqsev_{ln}.csv")) for ln in BEST_SINGLE
             if os.path.exists(os.path.join(OUT, f"cv_freqsev_{ln}.csv"))]
    perils = [pd.read_csv(os.path.join(OUT, f"cv_freqsev_perils_{ln}.csv")) for ln in BEST_SINGLE
              if os.path.exists(os.path.join(OUT, f"cv_freqsev_perils_{ln}.csv"))]
    folds, perils = pd.concat(folds), pd.concat(perils)
    summary = folds.groupby("line", sort=False).agg(d2=("d2", "mean"), d2_cal=("d2_cal", "mean")).reset_index()
    summary["best_single"] = summary["line"].map(lambda ln: BEST_SINGLE[ln][0])
    summary["best_single_d2_cal"] = summary["line"].map(lambda ln: BEST_SINGLE[ln][1])
    summary["difference"] = summary["d2_cal"] - summary["best_single_d2_cal"]
    with open(os.path.join(ROOT, "docs", "cv_freqsev.md"), "w", encoding="utf-8") as f:
        f.write("# Time-based CV: per-peril frequency x severity (C023)\n\n"
                "Folds: 2021-22 -> 2023, 2021-23 -> 2024. Scored like `scripts/oof_predictions.py`: Tweedie D2 "
                "(power 1.5) of reported cost; `d2_cal` rescales predictions to the fold's actual total. "
                "Generated by `scripts/cv_freqsev.py`.\n\n## Per fold\n\n")
        f.write(folds.round(5).to_markdown(index=False))
        f.write("\n\n## Mean over folds vs best single model (C021 out-of-fold runs)\n\n")
        f.write(summary.round(5).to_markdown(index=False))
        f.write("\n\n## Per peril\n\nRatios are predicted / actual. `fit_*` on the training rows, `val_*` on the "
                "validation fold; `*_sev_given_n` compares severity x actual claim count with actual capped cost; "
                "`val_cost` is capped cost; `val_cost_share` is the peril's share of the fold's uncapped cost; "
                "`val_d2_cal` scores the peril's own uncapped cost.\n\n")
        f.write(perils.round(4).to_markdown(index=False))
        f.write("\n")


def main(lines):
    shared = (clean.fix_customer_id(load("customers")), load("addresses"), load("weather"))
    claims = load("claims")
    for line in lines:
        t0 = time.time()
        run_line(line, shared, claims)
        log(f"{line}: done in {time.time() - t0:.0f}s")
    write_doc()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
