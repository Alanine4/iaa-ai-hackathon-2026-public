# IAA AI Assisted Hackathon 2026: pricing model for two markets

Code, documentation and audit trail for our entries in the IAA AI Assisted Hackathon 2026: Part 1, the developed market
(https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1), and Part 2, an emerging market
(https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-2).

## What the competition asks

In Part 1, for each policy row in 2025 (motor, home buildings and home contents; 626,639 rows in total) we predict the
final cost of all claims that occur in the row's period, net of excess and recoveries and excluding repudiated
claims. The training data covers 2021 to 2024. A submission is one CSV with the header `row_id,prediction`, scored by
Tweedie D2 with power 1.5 (higher is better). Part 2 asks for the same prediction in an emerging market that has only
two years of training data (2023 and 2024) and 209,063 test rows for 2025, about 77% of the test exposure being new
business. Scoring is unchanged, and the written report must also build the office premium of 20 reference policies from
the predicted risk premium, which the judges assess and the leaderboard does not score.

## Repository layout

| Folder or file | What is in it |
|---|---|
| `src/aiiaa/` | Reusable code: `clean.py`, `target.py`, `development.py`, `features.py`, `models.py`, `params.py`, `evaluate.py`, `freqsev.py`, and `markets.py` (data folder, training years, output folder and trend options of each market, C034) |
| `scripts/` | Pipeline entry points (`build_target.py`, `oof_predictions.py`, `blend.py`, `blend_runs.py` for the Part 2 cross-run and segment blends, `tune_cv.py`, `train_validate.py`) and one-off analysis and investigation scripts |
| `tests/` | pytest suite, including the raw-data integrity test |
| `data/` | The 11 raw Part 1 parquet files, the Part 2 folder (13 emerging files), a SHA256 manifest for each, and `data/README.md` |
| `docs/` | `CHANGELOG.md`, `AI_LOG.md`, `AUDIT_2026-10-22.md`, `AUDIT_PART2.md`, `SUBMISSIONS.md`, `HUMAN_REVIEW.md` (sign-off sheet for key AI decisions, in Chinese), `OFFICE_PREMIUM_DRAFT.md` (office premium formula chain for the 20 reference policies), `COMPETITION.md`, `DATA_GUIDE.md`, `DATA_WALKTHROUGH.md`, validation and CV reports, and `references/` (archived Kaggle pages) |
| `docs/part2_validation/` | Copy of the Part 2 validation evidence from `output/emerging/`: fold scores and blend weights per run, logs, and its own README |
| `submissions/` | The selected Kaggle files: Part 1 final (S09), Part 2 candidate (P2-S13) and Part 2 hedge (P2-S12) |
| `output/` | Derived data and predictions written by the scripts. Part 2 files go to `output/emerging/`. Not version-controlled |
| `work/` | Submission CSV and zip files written by `blend.py` and `blend_runs.py`. Not version-controlled |
| `TODO.md`, `CLAUDE.md` | Progress checklist and working conventions for the AI assistant |
| `reproduce.ps1` | One-command reproduction of a submission file (Part 1; the `-Part2` switch adds the Part 2 chain) |

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

The Part 2 files were added on 2026-10-24 under `data/iaa-ai-assisted-hackathon-2026-part-2/` (C033). `emerging/` holds
the 13 emerging-market files (the 11 file types of Part 1 plus `expenses.parquet` and `reference_policies.parquet`) and is
committed with its own `MANIFEST.sha256`, which the same integrity test checks. `developed/` is a byte-identical copy of
the Part 1 folder (all 11 SHA256 values match the Part 1 manifest), so its parquet files are not stored twice: the
repository keeps only its manifest, and the code reads the Part 1 folder for the developed market.

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

The script applies the both-folds guard itself (C030): the weights file holds the guarded `weights`, the grid result in
`weights_searched` and the reason in `rule`. The final refit averages 3 seeds for the tree models (C029), so numbers differ slightly between runs.

## Reproducing the Part 2 submission

P2-S13 (`submissions/part2_candidate_P2-S13_p2_c047.zip`, public 0.15577) is built from six training runs, two blend
searches and one write step. The pooled runs read the Part 1 target tables `output\target_<line>.parquet`, so the first
command is the Part 1 `build_target.py`; skip it if those files exist. Run one data process at a time and plan for about
4.5 GB of memory at peak. The five full pooled runs take roughly one to two hours each (the comparable contents runs took
57 to 58 minutes in C045; motor is the largest line and was not timed), about 7 to 8 hours in all; the motor run on
new-business rows took 41 minutes (C047); the blend steps take a few minutes (128 and 257 seconds for the two searches).
`reproduce.ps1 -Part2` runs the same chain after the Part 1 stages.

```powershell
.venv\Scripts\python.exe -I -u scripts\build_target.py
.venv\Scripts\python.exe -I -u scripts\build_target.py --market emerging
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py motor --market emerging --train-set pooled --developed-weight 0.3 --models glm,lgbm,xgb,cat
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py motor --market emerging --train-set pooled --developed-weight 1.0 --models glm,lgbm,xgb,cat
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py motor --market emerging --train-set pooled --developed-weight 0.3 --models glm,lgbm,xgb,cat --rows nb
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py buildings --market emerging --train-set pooled --developed-weight 1.0 --models glm,lgbm,xgb,cat
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py contents --market emerging --train-set pooled --developed-weight 0.3 --models glm,lgbm,xgb,cat
.venv\Scripts\python.exe -I -u scripts\oof_predictions.py contents --market emerging --train-set pooled --developed-weight 1.0 --models glm,lgbm,xgb,cat
.venv\Scripts\python.exe -I -u scripts\blend_runs.py contents --runs pooled_w0.3,pooled_w1 --gate testmix --out xrun_c044_contents_2run
.venv\Scripts\python.exe -I -u scripts\blend_runs.py motor --runs pooled_w0.3,pooled_w1,pooled_w0.3_nbrows --segment new_business --gate testmix --baseline P2-S12 --step 0.1 --out xrun_c047_motor
.venv\Scripts\python.exe -I -u scripts\blend_runs.py motor,buildings,contents --write-from motor=xrun_c047_motor,buildings=baseline,contents=xrun_c044_contents_2run --trend mixed-full --name p2_c047
```

The per-line developed-row weights are motor 0.3, buildings 1.0 and contents 0.3 (C040, `markets.py`). The weight-1.0 runs
for motor and contents are blend candidates, not the main runs of those lines. `--developed-weight 0.3` writes to
`output\emerging\pooled_w0.3\` and `1.0` to `output\emerging\pooled_w1\`; `--rows nb` fits on the new-business rows of both
markets only and adds the suffix `_nbrows` (C047). Each run writes the out-of-fold file, `blend_weights_<line>.json` and
`pred_<line>_<model>.parquet` to its folder. The two searches write `blend_weights_<line>.json` to
`output\emerging\<out>\`. The contents search (8 candidates, step 0.05) picks `pooled_w0.3` GLM 0.35 and `pooled_w1`
LightGBM 0.65 (C044). The motor search uses `--step 0.1` because its 12 candidates on a 0.05 grid are about 85 million
weight vectors; the new-business rows take an equal blend of all 12 candidates after the guard rejected the searched weights,
and the renewal rows keep the P2-S12 blend (C047). `buildings=baseline` takes the P2-S05 buildings blend (`pooled_w1` CatBoost
0.95 and LightGBM 0.05). The last command writes `work\p2_c047.csv` and `work\p2_c047.zip` and checks them against the
emerging `sample_submission`; the archived zip is the file that was uploaded.

## Modelling pipeline

The paragraphs up to "Trend" describe the Part 1 pipeline, which Part 2 reuses; the Part 2 changes follow after them.

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

Part 2: markets and schema (C033 to C037). `src/aiiaa/markets.py` holds one configuration per market: data folder, training
years, output folder and the 2025 trend options. Scripts take `--market developed|emerging` (or the `AIIAA_MARKET`
variable), and emerging outputs go to `output/emerging/`, so Part 1 files are never overwritten (C034). `features.harmonise`
puts both markets on one schema (C035): driver and policyholder ages are cut into the emerging bands, occupation text is
cleaned to 20 base jobs plus a self-employed flag, the new construction codes map to the existing categories, `gender` is
`unknown` for the developed market, and columns that exist only in the developed market are dropped. The emerging target
(C037) uses the same claim matching and a cap at the emerging line's own 99.5th percentile, but borrows the reporting
curves and peril cost shares of the developed market's accident year 2021, because two emerging years cannot support a
curve. The emerging 2023 curve, which has the same truncation as developed 2023, was compared with the borrowed one in
`docs/part2_validation/build_target.log`.

Part 2: pooled training (C038, C040). Fitted on emerging rows alone, the GLM and GAM overfit badly (negative D2 on the
buildings time fold, C036). The main runs are therefore pooled: every developed training row is added to every fit set with
its sample weight multiplied by 0.3 (motor, contents) or 1.0 (buildings), chosen per line by the validation rule below, and
the fitted level is re-balanced on the emerging fit rows so that developed rows do not set the emerging level. Pooled
training beat emerging-only on the time fold on all three lines (d2_cal 0.0915 against 0.0762 for motor, 0.1065 against
0.0410 for buildings, 0.1072 against 0.0779 for contents). Pooled runs use GLM, LightGBM, XGBoost and CatBoost; the GAM is
left out because it exceeded the memory cap on the pooled rows.

Part 2: validation and blending (C036, C041). With two training years the validation uses emerging rows only: a time fold
(train 2023, score 2024) and a grouped 5-fold CV on `policy_id`. Blend weights are searched on the grouped folds and kept
only if they beat equal weights on the time fold and on the grouped mean. `blend_runs.py` applies the same rule to blends
across run folders (C041).

Part 2: test-mix views and gate (C044). The audit (`docs/AUDIT_PART2.md`, P2-F1) found that validation exposure is 92%
renewals while about 77% of the 2025 test exposure is new business. Every model and blend is therefore also scored on a test-mix
view (validation rows re-weighted so that new business carries the test file's share of exposure) and on a new-business-only
view. A change is adopted only if it does not lose on the raw time fold, the raw grouped mean and the test-mix grouped view,
and gains at least 0.002 on the raw mean or on the test-mix grouped view. Under this gate the contents two-run blend was
adopted (P2-S12).

Part 2: segment blend (C047). New-business test rows can take a different blend from renewal rows, and the gate is applied
to each segment on the whole line. For motor, the new-business rows take an equal blend of 12 candidates (pooled runs at
developed weights 0.3 and 1.0, plus a run trained on the new-business rows of both markets only, four models each). It
passed on all six views and scored 0.15577 on the public board (P2-S13, +0.0010). The buildings renewal segment also passed
the gate mechanically but was rejected, because its gain came from applying a model built for new business to renewal rows.
Contents keeps the C044 blend and buildings the P2-S05 blend.

Part 2: level (C043). The 2025 level cannot be validated on 2023 and 2024 data. Motor uses the emerging market's own full
trend (factor 1.1606); buildings and contents use the developed market's half trend (1.0948 and 1.0636). The choice rests
on public-board probes, described under known limitations.

## Governance

Every change to data handling, code behaviour or modelling assumptions has a row in `docs/CHANGELOG.md` with its reason,
evidence and code location (C000 to C047). Code comments cite the same IDs. `docs/AI_LOG.md` records each piece of work
the AI assistant did, how a person or a test checked it, and whether it was accepted, modified or rejected. The test suite in
`tests/` (340 tests) covers cleaning, target matching, development, features, models, tuned parameters, scoring, the market
configuration and the Part 2 blends and validation views, and was written before the code it tests. `tests/test_raw_data_integrity.py` checks the raw files against the manifest. On 2026-10-22 an
independent read-only audit reviewed the pipeline (`docs/AUDIT_2026-10-22.md`). Its findings B1 to B8 fed into C026 and C027;
the remaining ones are listed below. On 2026-10-25 a second read-only audit covered the Part 2 pipeline (`docs/AUDIT_PART2.md`, findings
P2-F1 to P2-F10); it led to the test-mix gate (C044) and the new-business experiments (C045 to C047). `docs/HUMAN_REVIEW.md` is the
sign-off sheet for the key AI decisions, written in Chinese, with the evidence and one question for a team member per decision.
Run the tests with `.venv\Scripts\python.exe -m pytest -q`.

## Known limitations

The trend may be counted twice (audit B4). `written_premium` and `sum_insured` already rise with the year, and they enter
the models, yet the trend is fitted on raw pure premium. The residual trend the audit measured is about 1.017 for motor,
1.048 for buildings and 1.004 for contents, against 1.018, 1.095 and 1.064 applied. The public board still preferred a trend
over none, so the level was kept. A probe submission with buildings and contents at a quarter of the trend (S08) scored 0.12820
against 0.12851 for half trend (S07), so the public board does not support the double-counting hypothesis and half trend was kept. The large-loss load for the test year includes under-reported 2024 bodily injury claims and
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

Part 2 has only two training years, and its weak points follow from that. The trend cannot be fitted on 2023 and 2024: buildings
pure premium in 2024 is 0.63 of 2023 because of a catastrophe year, so the property lines borrow the developed market's half
trend. The reporting curves are borrowed from the developed market's accident year 2021; the audit puts the resulting
development bias under 2.5% (P2-F9). Validation exposure is about 8% new business against about 77% of the test exposure. The
test-mix and new-business views correct the weighting but not the thin evidence: the new-business-only scores rest on 37 to
697 claims per line and scheme (buildings 37 and 77, contents 65 and 102, motor 382 and 697), so the weights are not tuned
on them and they work as a gate only. Part of the 2025 level comes from the public board. Six scalar probes (P2-S06 to P2-S11)
moved the motor level from the developed half trend to the emerging market's own full trend, motor pure premium having risen
10.7% from 2023 to 2024; the public board covers about a third of the test rows, so the final ranking may move. The motor
new-business blend was chosen by a rule fixed in advance, but a control run made after seeing the result (new-business rows
from pooled XGBoost alone, `docs/CHANGELOG.md` C047) scored higher on every view; it is a post-hoc selection, so it is
recorded as a diagnostic and was not submitted.

## Submissions

`docs/SUBMISSIONS.md` lists each upload with its date, file, blend recipe, validation score and public score. For Part 1 the
final file is S09 (public 0.12863, private 0.13689, final rank 12 of 51). For Part 2 the candidate is P2-S13 (`p2_c047.zip`, public 0.15577) and the hedge is
P2-S12 (`p2_c044_contents_xrun.zip`, public 0.15474). `submissions/` holds `part1_final_S09_sub_v3_seeds.zip`,
`part2_candidate_P2-S13_p2_c047.zip` and `part2_hedge_P2-S12_p2_c044_contents_xrun.zip`.
