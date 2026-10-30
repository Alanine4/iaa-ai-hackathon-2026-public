# To-do list

Rule: when an item is done, change `[ ]` to `[x]` and write the completion date and output files at the end of the line. Add newly found items at the end of the matching phase.
Part 1 deadline: about 2026-10-23 (follow the countdown on the Kaggle page, UTC).

## Phase 0: Setup
- [x] Download the 11 data files to `data/` — 2026-10-21
- [x] Set up the Python environment `.venv` (pandas, pyarrow, lightgbm, sklearn, statsmodels, matplotlib) — 2026-10-21
- [x] Archive the original text of the Overview, Data and Rules pages in `docs/references/` — 2026-10-21
- [x] Profile the 11 files one by one → `scripts/profile_data.py`, `docs/data_profile.txt` — 2026-10-21
- [x] Verify the data quality issues → `scripts/check_quality.py`, `docs/data_quality_checks.txt` — 2026-10-21
- [x] Write the data guide and the competition guide → `docs/DATA_GUIDE.md`, `docs/COMPETITION.md` — 2026-10-21
- [x] Initialise the git repository, configure `.gitignore` (excludes `.venv/` and `output/`; data/ is under version control since C007), make the first commit — 2026-10-21
- [x] Create `docs/AI_LOG.md` and the change log `docs/CHANGELOG.md` — 2026-10-21
- [x] Push to the private GitHub repository https://github.com/Alanine4/iaa-ai-hackathon-2026 (without output/) — 2026-10-21
- [x] Commit the raw data to the private repository as a read-only backup, with a SHA256 manifest and an integrity test (C007) — 2026-10-21
- [x] Invite the teammate as a collaborator: GitHub user OBLEMOS already has write access (checked with gh api); already on the same team on Kaggle — 2026-10-26

## Phase 1: Target variable (day 1)
- [x] Cleaning: recompute exposure (C001), zero-pad customer_id (C002), unify construction_type (C004) — 2026-10-21, `src/aiiaa/clean.py`
- [x] Confirm what W and T mean in construction_type: the same class, T is used in 2021–22 and W in 2023–24, both map to timber (C004) — 2026-10-21
- [x] Decide the sum_insured outlier rule: values above 5 million are divided by 1000 and a flag column is added (C003) — 2026-10-21
- [x] Align claims to policy rows, drop declined claims, compute incurred − recoveries, list the unmatched ones separately (C005) — 2026-10-21, `src/aiiaa/target.py`
- [x] Development adjustment: per-line reporting-lag curve weighted by amount (C006) — 2026-10-21, `src/aiiaa/development.py`
- [x] All 18 automated tests pass; the full-total reconciliation difference is 0 — 2026-10-21, `tests/`, `docs/target_summary.txt`
- [ ] Human review: whether the C003 threshold, the C004 mapping, and the buildings 2024 developed level (481, about 458 by trend) are too high
- [x] Split the reporting curve by claim cause for each line (motor bodily injury is reported slowly) to see whether it improves the 2024 estimate — 2026-10-21, C022, numbers unchanged (after mixing by cost share it is identical to the pooled curve), kept as a building block, `docs/CHANGELOG.md`

## Phase 2: Features and validation (day 1)
- [x] Merge customers, addresses and weather (by year) — 2026-10-21, `src/aiiaa/features.py`
- [x] Feature engineering (customer age, customer tenure, building age, prior-year claim experience on the same policy); drop renewal_decision (Q9) and customer_service_contacts (C008) — 2026-10-21
- [x] Leakage scan across all fields → `docs/scan_leakage.txt` — 2026-10-21
- [x] Validation scheme: train on 2021–2023, validate on 2024, Tweedie D² (power 1.5), trend adjustment (C009) — 2026-10-21, `src/aiiaa/evaluate.py`
- [x] Model code: GLM, GAM (C011), LightGBM, XGBoost, all 51 tests pass — 2026-10-21, `src/aiiaa/models.py`
- [x] Run the baselines: GLM / GAM / LightGBM / XGBoost validation + full retraining + 5 submission files (including the average blend) — 2026-10-21, `docs/validation_results.md`, `output/submissions/`
- [ ] Find out why the validation set is 7–11% too high overall (the bias is smaller without trend; suspected cause: trend coefficients extrapolated from 3 years of data are too high; the buildings 2025 trend coefficient 1.199 needs review)
- [ ] Set up the Kaggle API credentials (you need to do this yourself), after which submissions can be made with the CLI
- [x] Model selection: compare validation scores per line, decide between a single model and a blend, write it into the CHANGELOG — 2026-10-21, C021 per-line blend weights (adopted only if not worse than equal weights on both folds), `docs/CHANGELOG.md`
- [x] First Kaggle submission: blend model, public leaderboard 0.12453, rank 19 — 2026-10-21, `docs/SUBMISSIONS.md`
- [ ] Upload the two trend comparison versions (`work/sub_blend_trend000.zip`, `work/sub_blend_trend050.zip`) and tell me the scores, to decide how much trend to use (C016)

- [x] Data overview document: for each of the 11 files, a field-by-field explanation with example, type, source (official / inferred) and handling — 2026-10-21, `docs/DATA_WALKTHROUGH.md`
- [x] Variable type fix: identifiers stored as numbers, such as branch_id, should be treated as categories in GLM/GAM, not as continuous numbers — 2026-10-22, C026 simply dropped branch_id (no frequency signal), `docs/CHANGELOG.md`, `docs/AUDIT_2026-10-22.md`

## Phase 3: Improvement and diagnostics (day 2)
- [x] Tuning; compare GLM and LightGBM, consider blending the two — 2026-10-21, C017/C018, `docs/CHANGELOG.md`
- [ ] Diagnostics: actual vs predicted (by quantile), lift curve, effect of each feature (SHAP / partial dependence) (partly done 2026-10-22: the independent audit checked frequency dispersion by column and the gain share of the tree models, `docs/AUDIT_2026-10-22.md`; quantile comparison, lift and SHAP not done)
- [ ] Check the train/test distribution difference (Q10) and write it into the report (partly done 2026-10-22: the audit confirmed the Q10 drift is internally consistent, renewal_count rises by 1.0 each year, `docs/AUDIT_2026-10-22.md`; not yet written in the report)
- [ ] Retrain on the full 2021–2024 data and produce the final submission
- [ ] Select the Final Submission on Kaggle
- [ ] Human review of the three columns dropped together in C026 (document_preference, marital_status and occupation were tested together, not verified one by one)
- [ ] Rerun the peak-split model with the new features (optional, if time allows)
- [ ] Before the final submission, select the Final Submission on Kaggle

## Phase 4: Preparing for Part 2
- [x] Turn the pipeline into a script that reruns with one command (works by changing the data path) (`reproduce.ps1`, the two "Reproducing" sections of the README; checked 2026-10-30)
- [x] Write the README: environment, run steps, directory structure (`README.md`, checked 2026-10-30)
- [x] Report draft: method, decisions, AI use, assumptions, limitations, controls (2026-10-30, `docs/report/report.tex`, organised by the four grading criteria, 5 graded pages + appendix; `docs/REPORT_DRAFT.md` is the source-material draft)
- [ ] After the Part 2 data is released: compare the differences between the old and new markets, then break down the new to-dos

## Phase 5: Part 2 emerging market
- [x] Switch data paths and output directories by market (`--market` / `AIIAA_MARKET`, the default is still Part 1) — 2026-10-24, C034, `src/aiiaa/markets.py`
- [x] Harmonise the fields of the two markets: age becomes age band, gender, occupation cleaning (split out a self-employed flag), new construction type codes — 2026-10-24, C035, `src/aiiaa/features.py`
- [x] Emerging market target variable: borrow the developed market reporting curve, reconciliation difference 0, and compare the reporting curves of the two markets — 2026-10-24, C037, `docs/part2_emerging_target_summary.txt`
- [x] Emerging market validation (2023→2024 time fold + 5 folds grouped by policy) and the blend weights of the five models — 2026-10-24, C036, `output/emerging/blend_weights_<line>.json`
- [x] Submission file using only emerging market data (not uploaded) — 2026-10-24, C036, `work/p2_emerging_only.csv`, `work/p2_emerging_only.zip`
- [x] Pooled training of both markets (developed weight 0.3) run for three lines and compared: contents 0.1072 vs 0.0779, buildings 0.1065 vs 0.0410, motor 0.0915 vs 0.0762 (time fold d2_cal), the group means also all win, all three lines adopted; GAM did not enter the pooled run because memory exceeded 4 GB — 2026-10-25, `output/emerging/pooled_w0.3/`, `work/p2_pooled_w03_half.zip` (P2-S04)
- [x] Pooled weight probe (contents): weight 1.0 vs 0.3, time fold 0.1069 vs 0.1072, group mean 0.1128 vs 0.1100, average +0.0013 < 0.002, so 0.3 stays; the tree models alone benefit (LGBM +0.0067, CAT +0.0145) — 2026-10-25, `output/emerging/pooled_w1/blend_weights_contents.json`
- [x] C039 tuning (pooled contents, LGBM 13 configs + XGB 10 configs × 6 folds): no config beats the default on both the time fold and the group folds (LGBM with 15 leaves: group +0.0030 / time fold −0.0063; the XGB default is best), parameters unchanged; the hypothesis that "leaves of 3000 rows are too coarse" does not hold on pooled — 2026-10-25, `docs/cv_emerging_trees_pooled_w0.3.md`
- [x] Buildings line weight 1.0 probe: time fold 0.1169 vs 0.1065, group mean 0.1180 vs 0.1132, average +0.0076, buildings adopts 1.0 (contents keeps 0.3) — 2026-10-25, `output/emerging/pooled_w1/blend_weights_buildings.json`, P2-S05
- [x] Motor line weight 1.0 probe: time fold 0.0908 vs 0.0915, group mean 0.0730 vs 0.0740, slightly worse, motor keeps 0.3; the three-line weights {motor 0.3, buildings 1.0, contents 0.3} are written into `markets.py`, blend.py supports `--developed-weight per-line` (C040), and the repository command reproduces P2-S05 byte for byte — 2026-10-25
- [x] Optimisation probes C041 (cross-run blend) and C042 (GLM market interaction terms): neither met the adoption standard, the version of C042 that formally met the standard was rejected because the new business level was not verified; the new business segment check shows the evidence is noisy, so no adjustment was made — 2026-10-25, `docs/CHANGELOG.md` C041/C042, `docs/part2_new_business_check.md`, AI_LOG A23
- [x] Level calibration (C043): 6 scalar probes (P2-S06–S11); the motor level is low, so switch to the emerging market's own full trend 1.1606 (`mixed-full`), buildings and contents unchanged; P2-S10 0.15414 — 2026-10-25, `docs/SUBMISSIONS.md`
- [x] Audit countermeasures (C044/C045): the test-mix-weighted view becomes a gate; new business weighted training, early-stopping-only change, and dropping GLM from motor are all rejected; the two-run contents blend is adopted under the new gate → P2-S12 0.15474 — 2026-10-26, `docs/AUDIT_PART2.md`, CHANGELOG C044/C045
- [x] New model experiments C046 (emerging market tree model tuning → blend candidate, rejected on all three lines), C047 (new-business-specific model + segmented blend: motor new business segment adopted, buildings renewal segment rejected by design) → P2-S13 0.15577 — 2026-10-26, CHANGELOG C046/C047, `docs/part2_validation/`
- [ ] Kaggle final selection: candidate P2-S15 (0.15807, seed-averaged version, the rule fixed in advance has been passed, 2026-10-30, `docs/SUBMISSIONS.md`); hedge P2-S14 (0.15781)
- [x] The Part 1 report and video present the post-deadline blend (0.7 S09 + 0.3 OBLEMOS, late, not ranked, private 0.13718) side by side with the ranked S09 (private 0.13689, rank 12 / 51); the draft is written into `docs/REPORT_DRAFT.md` and `docs/VIDEO_SCRIPT.md`, keep it in the final version (2026-10-30: report table 3, appendix B and the dashboard overview all show both side by side, marked late/not ranked)
- [ ] Paste `docs/report/overleaf_sections.tex` into Overleaf and compile it, confirm that section 4 ends on page 5 and the appendix starts on page 6 (team lead)
- [x] Team lead sign-off of `docs/HUMAN_REVIEW.md` R12–R16 — 2026-10-30, `docs/HUMAN_REVIEW.md` (the team lead replied in the chat "I approve all of these" (said in Chinese), filled in by the coordinator)
- [ ] Record the 5-minute video using `docs/dashboard/index.html` (or the artifact link), script in `docs/VIDEO_SCRIPT.md` (664 words, about 4:55, suggest cutting about 10 words from segment 5)
- [x] Re-check of the developed market default path: the build_target tables for the three lines are identical to `output/target_*.parquet`, the report matches `docs/target_summary.txt`; the S09 blend reproduced with the current blend.py has the same SHA256 as `work/sub_v3_seeds.csv` — 2026-10-25, `output/emerging/check_dev_target.log`
- [x] Recomputed half-trend coefficients: motor 1.0175, buildings 1.0948, contents 1.0636, consistent with `markets.py` — 2026-10-25, `output/emerging/check_dev_target.log`
- [ ] GLM/GAM overfit severely on the emerging market (buildings time-fold D² is negative, a single-row prediction of 400,000), consider stronger regularisation or tree models only
- [x] Upload `work/p2_emerging_only.zip` to Kaggle and record the score — 2026-10-24, P2-S01 public leaderboard 0.14370 (rank 8 / 10), `docs/SUBMISSIONS.md`
- [x] Trend probe P2-S02 (developed-half): 0.14460, 0.0009 higher than no trend, the Part 2 default switches to developed-half — 2026-10-24, `docs/SUBMISSIONS.md`
- [x] Office premium formula draft (8 expense items, full table for the 20 reference policies, formula chain, 8 open questions) — 2026-10-24, `docs/OFFICE_PREMIUM_DRAFT.md`, `scripts/part2_expenses.py`; **open questions are for you and your teammate to decide**
- [x] Human review sign-off sheet: 11 key AI decisions + 7 corrected AI outputs, each with file:line and numbers — 2026-10-24, `docs/HUMAN_REVIEW.md`; **waiting for the team lead to sign item by item**, then copy the results into AI_LOG
- [ ] Fix outdated text: lines 107–108 of `docs/DATA_GUIDE.md` still say "pending / unconfirmed" for Q3/Q4, they should be changed to the conclusions of C003/C004 (found in passing by a worker)
- [x] Report section 7 (new market adaptation: differences, process changes, validation design, result table, office premium formula, limitations), about 590 words — 2026-10-25, `docs/REPORT_DRAFT.md`
- [x] Compress the report to within 5 pages and add section 8 (2026-10-30, `docs/report/report.tex` section 1.4, the premium table uses P2-S15 predictions, C060): office premium table for the 20 reference policies (waiting for the final prediction file and for the teammate to settle the definition)
- [x] Video script (5 minutes, about 700 words) (2026-10-29, `docs/VIDEO_SCRIPT.md`, test count changed to 340)
- [x] Invite the teammate as a GitHub collaborator (OBLEMOS, write access) — 2026-10-26
- [x] Correction: the 0.1356 of the teammate's Part 1 file is a private leaderboard score, the best public score is 0.12758, both below S09; the choice of S09 is correct. On 10/27–28 the teammate submitted 6 variants based on S12/S13 in Part 2 (best 0.15563, did not exceed S13), recorded in SUBMISSIONS T01–T05 — 2026-10-29
