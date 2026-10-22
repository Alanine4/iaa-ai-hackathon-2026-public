# IAA AI Assisted Hackathon 2026, Part 1: pricing model

Code, documentation and audit trail for our Part 1 entry in the IAA AI Assisted Hackathon 2026
(https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1).

## What the competition asks

For each policy row in 2025 (motor, home buildings and home contents; 626,639 rows in total) we predict the
final cost of all claims that occur in the row's period, net of excess and recoveries and excluding repudiated
claims. The training data covers 2021 to 2024. A submission is one CSV with the header `row_id,prediction`, scored by
Tweedie D2 with power 1.5 (higher is better). Part 2 will add a second market; this repository covers Part 1 only.

## Repository layout

| Folder or file | What is in it |
|---|---|
| `src/aiiaa/` | Reusable code: `clean.py`, `target.py`, `development.py`, `features.py`, `models.py`, `params.py`, `evaluate.py`, `freqsev.py` |
| `scripts/` | Pipeline entry points (`build_target.py`, `oof_predictions.py`, `blend.py`, `tune_cv.py`, `train_validate.py`) and one-off analysis and investigation scripts |
| `tests/` | pytest suite, including the raw-data integrity test |
| `data/` | The 11 raw Kaggle parquet files, a SHA256 manifest and `data/README.md` |
| `docs/` | `CHANGELOG.md`, `AI_LOG.md`, `AUDIT_2026-10-22.md`, `SUBMISSIONS.md`, `COMPETITION.md`, `DATA_GUIDE.md`, `DATA_WALKTHROUGH.md`, validation and CV reports, and `references/` (archived Kaggle pages) |
| `output/` | Derived data and predictions written by the scripts. Not version-controlled |
| `work/` | Submission CSV and zip files written by `blend.py`. Not version-controlled |
| `TODO.md`, `CLAUDE.md` | Progress checklist and working conventions for the AI assistant |
| `reproduce.ps1` | One-command reproduction of a submission file |

Project notes in `TODO.md`, `CLAUDE.md` and most of `docs/` are written in Chinese. Code, comments and the change log are in English.

## Environment setup

The project uses Python 3.14 on Windows. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins every package (pandas 3.0.6, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, scikit-learn 1.9.1,
statsmodels 0.15.0 and others). Run all scripts with `.venv\Scripts\python.exe`. The reproduction script uses `-I -u`
(isolated, unbuffered), and the scripts add `src/` to the import path themselves.

## Data

The 11 raw files sit in `data/iaa-ai-assisted-hackathon-2026-part-1/` exactly as downloaded on 2026-10-21. They are
committed as a read-only backup with a SHA256 manifest (`MANIFEST.sha256`), and `tests/test_raw_data_integrity.py` fails
if any byte changes (C007). No correction is ever written back to the raw files. Every fix is applied in code at load
time and has a change-log row. See `data/README.md` for the access rules (the repository must stay private) and for how
to add a new dataset. Field definitions and the data-quality issues Q1 to Q10 are in `docs/DATA_GUIDE.md`.

## Reproducing a submission

Run `reproduce.ps1` from the repository root, or run the steps below by hand. The full run takes about two hours.
It does not need `scripts\make_manifest.py`, which is only used when a new raw dataset is added.

```powershell
.\reproduce.ps1 -Name sub_repro            # add -SkipTests to leave out pytest; -Models to change the model list
```

The same sequence by hand:

```powershell
.venv\Scripts\python.exe -I -u scripts\build_target.py
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py motor
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py buildings
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py contents
.venv\Scripts\python.exe -I -u scripts\blend.py glm,gam,lgbm,xgb,cat sub_repro --weighted
.venv\Scripts\python.exe -I -u -m pytest -q
```

`build_target.py` writes `output\target_<line>.parquet` (reported and developed cost per training row) and prints a
reconciliation of claim cost against the claims file. The report goes to the console. The committed copy is `docs\target_summary.txt`.
`oof_predictions.py` runs one line at a time. It scores two out-of-fold years (train 2021-22 and predict 2023, then
train 2021-23 and predict 2024), writes per-line blend weights to `output\blend_weights_<line>.json`, refits every model on
all training years and writes the 2025 predictions to `output\pred_<line>_<model>.parquet`. `blend.py` combines those
files with the per-line weights and writes `work\<name>.csv` and `work\<name>.zip`, the file to upload to Kaggle.

One manual step is not in the script. The weights file is written with the searched weights. For motor, the
searched weights did not beat equal weights on both folds, so the submitted blend (S07) used equal weights of 0.2 per model, edited
into `blend_weights_motor.json` by hand (C021, `docs/SUBMISSIONS.md`). A fresh run will therefore give a motor blend
that differs from S07 unless that file is edited the same way. Compare `cv_d2_cal_both_folds` in each weights file
before trusting `--weighted`. Exact numbers will also differ slightly between runs if library versions or thread counts differ.

## Modelling pipeline

Cleaning (C001 to C004). The raw tables have four defects that are fixed at load time in `clean.py`. `exposure` is
recomputed from the period dates on a 365-day year because thousands of training rows hold day counts or 370/365
(C001). Motor `customer_id` values lost their leading zeros and are padded to 8 characters so they join to `customers`
(C002). `sum_insured` values above 5 million are divided by 1,000 and flagged, since they match the same policies' other
years at a factor of exactly 1,000 (C003). `construction_type` is lower-cased and the codes T and W both map to timber (C004).

Target and development (C005, C006, C014). `target.py` attaches each claim to the policy row whose period contains the
loss date and sums incurred minus recoveries, excluding repudiated claims. Claims that match no row are listed, and the
reconciliation difference is 0 (C005). The claims file is a single snapshot at 2024-12-31, so recent years are
under-reported. `development.py` builds a cost-weighted reporting-lag curve from accident year 2021, derives a
completeness for every row, and develops the cost to ultimate (C006, with per-peril curves from C022). Each claim is
capped at the 99.5th percentile of its line and the capped excess is added back as a per-line load factor (C014).

Features (C008, C026). `features.py` joins customers, addresses and the weather for the policy start year, and adds
customer age, tenure, property age and the same policy's claims in earlier years. It drops
`customer_service_contacts`, which leaks (it counts contacts caused by claims in the same period, C008), and
`renewal_decision`, which is only known after the row ends. A per-line drop policy (C026) removes columns with no
frequency signal that the models were overfitting: `branch_id`, `previous_insurer`, `email_domain`, the cross-line history and
household counts, motor `make_model`, and three customer columns for buildings and contents.

Models (C010 to C013, C015, C025, C027). All models fit capped cost per unit of completeness-weighted exposure with a
Tweedie loss of power 1.5. The set is a banded one-hot GLM, a spline GAM with one global ridge penalty (C011), LightGBM,
XGBoost and CatBoost (C025). Each model's predictions are rescaled so the training total matches the actual total
(C013). Tree models use shallow trees (C015) and tuned per-line settings from `params.py` (C018). Their number of rounds is
chosen on a random 10% of the fitting rows, never on the scored fold (C027); the final refit uses 1.1 times the mean best round count.

Validation and blending (C017 to C021, C028). Folds are expanding windows, 2021-22 to 2023 and 2021-23 to 2024, scored by
D2 on reported cost against prediction times completeness. `d2_cal` rescales to the fold total first, so it measures ranking
and not level. Hyper-parameters were chosen on this score with a pre-set rule (adopt only a gain of at least 0.002, C017,
C018). Blend weights are non-negative, on a 0.05 grid, sum to 1, and are chosen by mean `d2_cal` over both folds (C021).
The searched weights are adopted only if they beat equal weights on both folds. `oof_predictions.py` now does the folds, the weights and
the refit in one pass (C028).

Trend (C009, C019). Developed pure premium rises over 2021 to 2024, so a per-line log-linear trend factor from the mean
training year to 2025 scales the predictions. The public leaderboard preferred half the fitted trend over none or the full
trend (0.12464 against 0.12376 and 0.12453), so predictions use the square root of the factor (C019).

## Governance

Every change to data handling, code behaviour or modelling assumptions has a row in `docs/CHANGELOG.md` with its reason,
evidence and code location (C000 to C028). Code comments cite the same IDs. `docs/AI_LOG.md` records each piece of work
the AI assistant did, how a person or a test checked it, and whether it was accepted, modified or rejected. The test suite in
`tests/` covers cleaning, target matching, development, features, models, tuned parameters and scoring, and was written before the
code it tests. `tests/test_raw_data_integrity.py` checks the raw files against the manifest. On 2026-10-22 an
independent read-only audit reviewed the pipeline (`docs/AUDIT_2026-10-22.md`). Its findings B1 to B8 fed into C026 and C027;
the remaining ones are listed below. Run the tests with `.venv\Scripts\python.exe -m pytest -q`.

## Known limitations

The trend may be counted twice (audit B4). `written_premium` and `sum_insured` already rise with the year, and they enter
the models, yet the trend is fitted on raw pure premium. The residual trend the audit measured is about 1.017 for motor,
1.048 for buildings and 1.004 for contents, against 1.018, 1.095 and 1.064 applied. The public board still preferred a trend
over none, so the level was kept. The large-loss load for the test year includes under-reported 2024 bodily injury claims and
may be about 1.4% low for motor (B5). Recoveries appear only on closed claims, so open 2024 motor own-damage cost is overstated by
roughly 1.4% (B6). The prior-claims history is only partly put on a developed basis: cost is developed but claim counts are
not, and training history is not cut at the policy's inception date (B7). The audit could not verify the
open-claim reserving of bodily injury, the date of the customer snapshot, or whether `days_unoccupied` is set after a claim.

Several ideas were tested and not adopted. The per-peril frequency by severity model (C023) scored 0.0801 against 0.0893 for the best
single motor model. Cross-line and household features (C024) added nothing and were removed by C026. Per-peril reporting curves
(C022) change nothing numerically and stay as inputs only. The GAM is not an mgcv-style GAM (one penalty for all terms, C011). The C026
drop of `document_preference`, `marital_status` and `occupation` was tested as a group, not column by column. Earlier tuning and
blend decisions (C018, C021) used the old early-stopping setup and were not re-run in full (C027).

The 2025 test data differs from training (DATA_GUIDE Q10): median `renewal_count` falls from 6 to 3, median `vehicle_age` rises from 7 to 9 and
median `ncb_years` from 7 to 8. The audit found this drift internally consistent, with `renewal_count` rising by 1.0 per year. The
public leaderboard uses about a third of the test rows, so the final ranking may move.

## Submissions

`docs/SUBMISSIONS.md` lists each upload with its date, file, blend recipe, validation score and public score. The latest, S07
(2026-10-22, per-line weighted blend of GLM, GAM, LightGBM, XGBoost and CatBoost), scored 0.12851 on the public board.
