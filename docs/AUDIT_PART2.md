# Independent audit of the Part 2 (emerging market) pipeline (2026-10-25)

Audited commit: fd526f6 at the start (P2-S05 recipe); HEAD 0bb37d4 at the end (C043 level probes P2-S06..S11 logged during the audit, read and taken into account; no code path audited here changed).
Auditor: senior implementer agent (claude-opus-5-5), read-only. No model was trained; every number comes from the raw emerging tables, `output/emerging/**` (targets, OOF, pred files, JSONs) and, twice, the developed target/policy tables (memory-capped runs, peak 366 MB).
Scratch scripts `a01_*.py` ... `a27_*.py` and their outputs `aNN_*.txt` are in the session scratchpad folder `C:\Users\LIN\AppData\Local\Temp\claude\C--Projects-AIIAA\313cf442-d49a-49d6-b972-b4e38a047f4f\scratchpad\audit_p2\` (cited below as `audit_p2\aNN`).
Recipe audited: P2-S05 (motor pooled_w0.3 glm/lgbm/xgb/cat equal; buildings pooled_w1 cat .95 / lgbm .05; contents pooled_w0.3 glm .55 / xgb .25 / lgbm .20). "Kaggle-like D2" below = Tweedie D2 (power 1.5) of the three lines stacked, against the mean of y, as sklearn computes it.

---

## Summary

No scaling or alignment bug was found in the test-time arithmetic (H2 passes on every check). The uniform level has been harvested: validation curvature plus P2-S06..S11 put the remaining uniform-level gain at about 0.0002. So the 0.015-0.018 still between us and the leaders is mostly ranking, not level.

The structural finding is about validation. The emerging validation rows are 92% renewals; the 2025 test is 77% new business (NB). The pooled design's validation gains sit almost entirely on renewal rows. On NB rows the pooled models gain little over the emerging-only models, and on motor NB rows they are worse than the emerging-only XGBoost. When the validation is re-weighted to the test mix, it matches the public board: the validation-to-public shortfall for P2-S01 -> P2-S05 drops from about 3x to about 1.5x, and P2-S04 -> P2-S05 drops to +0.0004 against a public -0.0007. Every Part 2 choice so far (developed weight per line, blend weights, the guard) was optimised on renewal behaviour. The evidence on NB rows is thin (697 motor, 77 buildings, 102 contents NB claims), so it can reject choices but cannot tune them.

## Findings

| ID | Severity | What | Evidence | Recommended action | Expected effect on score |
|---|---|---|---|---|---|
| P2-F1 | high (structural) | Validation population does not represent the test: NB is 7.7-8.4% of validation exposure vs 76.8-77.0% of test exposure, and the pooled gains are concentrated on renewal rows. Unweighted validation over-predicted the public gains of the pooled design by about 2-5x | `audit_p2\a20`, `a21`, `a23`; `docs/part2_market_comparison.md` s.3 | Add a third validation view, test-mix re-weighted folds (NB rows re-weighted to 77% within each line), plus NB-only D2. Report it with time / grouped for every Part 2 decision and reject any change that loses on it. Do not search weights on it (too noisy, see P2-F2) | Indirect. It would have flagged P2-S05's buildings w1 (public -0.0007). Worth 0.001-0.004 through better choices; unproven |
| P2-F2 | medium | Motor blend is weak on NB rows. Motor GLM (25% of the equal-weight fallback) is the worst motor candidate on NB rows; the emerging-only XGBoost is the best on grouped NB rows | `audit_p2\a23`: grouped NB d2_cal glm 0.038, P2-S05 blend 0.054, w0.3:xgb 0.061, em:xgb 0.071; time-fold NB glm 0.094, blend 0.107, w0.3:xgb 0.116 | Candidate motor recipes without GLM (pooled lgbm/xgb, optionally + em-only xgb), scored on time / grouped / test-mix. Run C039 on emerging-only motor first (47k rows, cheap) | 0 to +0.002 (motor is 47% of the null deviance). Test-mix search: time +0.0040 out of search but grouped LOFO -0.0018, so not yet robust (`a24`) |
| P2-F3 | medium | NB relative level per line. The test predictions' NB/renewal ratio per exposure-year (motor 1.39, buildings 1.09, contents 1.04) differs from the emerging experience for buildings and contents; the pooled models inherit the developed NB relativity | `audit_p2\a16` (capped-ultimate NB/REN: motor dev 1.48 / em 1.49; buildings dev 1.12 / em 0.88; contents dev 1.03 / em 1.36), `a22` (test-mix A/E NB-relative factor: buildings 0.76 grouped / 0.72 time; contents 1.22 / 1.51; motor 1.26 / 0.92) | One paired public probe: buildings NB x0.85 and contents NB x1.2 on the adopted file. Motor stays uniform: the two schemes disagree and P2-S11 tied | Up to +0.001 (in-sample estimates on test-mix: buildings x0.85 about +0.0004, contents x1.2 +0.0000 / +0.0005) |
| P2-F4 | low (closed) | Uniform level. On validation, 10% / 15% / 20% uniform level errors cost 0.0013-0.0016 / 0.0026-0.0028 / 0.0045-0.0070 of pooled D2; the Part 1 board implies c about 0.22 (0.0020 / 0.0043 / 0.0073). P2-S06..S11 harvested it: after the motor +14% (P2-S10) the uniform gain left is about 0.0002 | `audit_p2\a08`; SUBMISSIONS P2-S06..S11 | Stop level probing (already decided). Level cannot explain a 0.02 gap: that would need a uniform error of 35% or more | <= +0.0002 left |
| P2-F5 | medium (untested) | Pooled fits: the C027 inner early-stopping holdout carries 86-89% (w 0.3) / 95-96% (w 1) developed weight, so the iteration count is set by the developed fit. Emerging structure enters only through `market` splits in 7-leaf / 3,000-row trees; re-balance factors up to 1.26 show the emerging level is not learned | `audit_p2\a26`; `a03` (best iterations pooled 220-999 vs emerging-only 1-320; rebalance factors); `models.py:39-43`, `oof_predictions.py:249-257` | Training needed: (a) early-stop on the emerging rows of the holdout only; (b) developed-model offset (`init_score` / `base_margin`) plus emerging-only boosting. Validate on all three views | Unknown. The only design change seen here that could plausibly reach >= 0.005. Counter-evidence: no one-way residual signal on emerging OOF (P2-F9) |
| P2-F6 | low | Emerging-only fits used the Part 1 tree defaults for 1M rows (LightGBM min_child_samples 3000, XGBoost depth 3). Buildings xgb stopped at 1 iteration in 2 of 5 grouped folds and lgbm at 6. C039 was run only on pooled contents. P2-S01 and every pooled-vs-emerging-only comparison understate emerging-only | `audit_p2\a03`; `params.py` (empty emerging keys); C039 outcome | Run `tune_cv.py trees motor --market emerging` (emerging-only), then re-score em-only candidates on NB rows | Feeds P2-F2 |
| P2-F7 | low | Blend concentration and board noise. Buildings cat .95 is robust; contents glm .55 is not robust on the grouped folds; the public board is noisy for single scores but fine for differences between close variants | `audit_p2\a17`, `a19`, `a27` (numbers in s.H7) | None for buildings. For contents, prefer weights the time fold and the test-mix view both support | Private evaporation <= 0.001 |
| P2-F8 | low (bug in a diagnostic) | `scripts/part2_new_business_check.py:73-74` reads the RAW policy `exposure` (C001 not applied; 0.45% of training rows hold 1.0139 or 365) and drops the target's recomputed exposure. Section 1 pure premiums and the by-year NB/REN ratios in `docs/part2_new_business_check.md` are wrong (motor 2023 NB 753.5 vs 940.6, renewal 291.3 vs 412.9; motor 2023 ratio 2.587 vs 2.278; buildings 2024 ratio 0.483 vs 0.783). Sections 2 and 4 are unaffected (OOF ratios; test raw exposure equals recomputed) | `audit_p2\a01`, `a05` | Use the target's `exposure`, regenerate, re-read the conclusion text. `REPORT_DRAFT.md:140` quotes only section 2 ratios and stays valid | None on score |
| P2-F9 | low | Development (H3). Within the observable 24-month window the emerging market reports faster than the developed one, so the 2024 ultimate is over-stated by 1.8% motor, 2.5% buildings, -0.2% contents; about 1% on the two-year level | `audit_p2\a14` | Mention in the report; no change | < 0.0002 |
| P2-F10 | info | Not supported / verified: H2 (no scaling bug), H4b (category union), H4c (no leak), H5 (no dropped column with residual signal), H6 (target matches the definition), leakage scan, weather join, seasonality | s.H2-H8 | - | - |

## Top actions, in order

1. **Make test-mix scoring a decision gate (P2-F1).** Extend the OOF scorers (`oof_predictions.scores`, `blend_runs`) to report, per line and pooled: time fold, grouped mean, test-mix re-weighted time / grouped, and NB-only. Validation: recompute the table in `audit_p2\a20/a21` from the code and check that it reproduces the public direction of P2-S01 -> S03 -> S04 -> S05.
2. **Motor NB-aware blend (P2-F2, P2-F6).** Tune emerging-only motor trees (C039 on the emerging-only set). Then run `blend_runs.py motor` over {em:xgb, em:lgbm, w0.3:xgb, w0.3:lgbm, w1:lgbm} with GLM excluded. Adopt only if the blend beats the P2-S05 motor recipe on the time fold, the grouped mean AND the test-mix grouped LOFO. One public probe against P2-S10.
3. **Pooled early stopping on emerging rows / developed offset (P2-F5).** One full pooled run per line with the early-stopping holdout restricted to emerging fit rows; separately, a developed-only model's log prediction as offset plus a shallow emerging-only booster. Same three-view gate.

Cheap side action: the buildings x0.85 / contents x1.2 NB probe (P2-F3), one submission, adopted only if it gains more than the 0.0005 tie band.

---

## H1 Level

**Curvature (validation, `audit_p2\a08`).** The P2-S05 recipe was re-levelled uniformly on the pooled three-line OOF (each line first sum-calibrated):

| factor | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 | 1.15 | 1.20 | 1.30 |
|---|---|---|---|---|---|---|---|---|---|---|
| time fold D2 | 0.13784 | 0.14112 | 0.14327 | 0.14447 | **0.14486** | 0.14455 | 0.14365 | 0.14222 | 0.14033 | 0.13542 |
| grouped D2 | 0.13535 | 0.13858 | 0.14069 | 0.14184 | **0.14218** | 0.14182 | 0.14086 | 0.13938 | 0.13744 | 0.13242 |

Per line, the line's own D2 moves by about c (ln k)^2 with c = 0.16 motor, 0.13 buildings, 0.13 contents. In the pooled metric a 10% error on one line costs 0.0005-0.0010 (motor), 0.0004 (buildings) and 0.0002-0.0003 (contents). The Part 1 probes (S01/S02/S03) fit a parabola with c about 0.22 and the optimum about 6.6% above no-trend. That matches the validation within a factor 1.5.

**Baselines.** Per-line constant rate x exposure scores 0.080 (time) / 0.079 (grouped) on validation. One global rate scores 0.019. The board's "baseline (average cost)" 0.0732 is consistent with the per-line rate version.

**Direction, from the data.**
- Motor: the 2023-trained model under-predicts 2024 by 6% (time-fold pred/actual 0.942). Like-for-like (losses Jan-Jun, reported within 180 days) motor frequency is +3.9% and severity +17% for 2024 vs 2023, with 7% vs 1% still open (`a04`). Motor was the low line, and P2-S07 / S10 confirmed it on the board.
- Buildings: the 2023 vs 2024 drop is fire-driven: 49 vs 21 fire claims, 3.22M vs 1.13M (`a04`). The written-premium medians (+6.1% a year in the emerging market) support the 1.0948 developed-half factor.
- Contents: flat, consistent with 1.0636 (premium medians +4.3% a year).
- The test mix lifts the model's per-exposure rate 22% / 23% / 14% above the 2023-24 training average before trend. Test NB rows look like training NB rows on every profiled column (age bands, marketing segment, ncb, sum insured; `a05`), so the uplift is a mix effect the model applies, not an extrapolation.

**Status after P2-S06..S11.** S06 (x1.10 all lines) +0.0008. With c of 0.14-0.22 that puts the uniform optimum 7-8% above P2-S05. The motor-only probes then moved most of that onto motor (S10 +0.0029). The remaining uniform gain is about 0.0002 (P2-F4).

**Noise of public-fitted levels (`a27`).** On a public-board-sized subsample the deviance-optimal level of one line has a sampling SD of 4.7% (motor), 12% (buildings) and 7% (contents). The motor +14% is therefore well above the noise and backed by the time fold. The buildings decision (S08) is weakly informed; at 12% SD a one-probe verdict on buildings carries little information.

## H2 Test-time scaling (all pass; `audit_p2\a01`, `a03`)

- `oof_predictions.py:341`: pred = rate x `test["exposure"]` x trend factor x `load_test`. There is no completeness term. `test` comes from `clean.clean_policies` (C001 days/365). Raw test exposure equals the recomputed value on every row, so the recomputation is moot for test (the max abs difference stays below 0.01 on every row). Training uses the same C001 basis (target `exposure`).
- Pred files carry `trend none, trend_factor 1.0` in every pooled run folder. `blend.py` multiplies by factor / file factor (`blend.py:113`), so the trend is applied once.
- Rate = prediction / exposure has |corr| <= 0.013 with exposure in every pred file, consistent with exposure entering only as a multiplier.
- P2-S05 rebuilt from the pred files and the per-line weights matches `work/p2_pooled_bw1.csv` to 1.8e-12. Per-line means 510.24 / 509.15 / 97.69; per exposure-year 584.6 / 562.0 / 109.7. `p2_level110.csv` / P2-S05 = 1.1000 on every row.
- `load_test` 1.0529 / 1.0511 / 1.0214 equals reported / capped reported cost over all emerging rows.

## H3 Development (`audit_p2\a14`)

Emerging and developed AY2023 cost-weighted CDFs both carry the 2024-12-31 truncation. Applied to the 2024 rows' periods, emerging/developed completeness = 1.016 (motor), 1.027 (buildings), 0.997 (contents); count-weighted 0.997 / 1.006 / 1.000. Re-developing 2024 with that ratio gives pure premiums of 485.4 vs 494.3 (motor, -1.8%), 319.1 vs 327.3 (buildings, -2.5%) and 89.9 vs 89.7 (contents). On the two-year level this is about -1%, -1%, 0%: worth under 0.0002, no action. The time-fold scale uses the same completeness on both sides, so d2_cal is unaffected. Not testable: the tail beyond 24 months and IBNER of open case estimates (20% of 2024 motor claims are open).

## H4 Pooled design

**(a) Early stopping.** Holdout = seeded 10% of fit rows (`models.py:39-43`), evaluated with the fit weights. The developed share of that weight is 0.867 / 0.885 / 0.860 at w 0.3 and 0.956 / 0.962 / 0.954 at w 1 (`a26`). Best iterations: pooled 220-999 (grouped folds), emerging-only 1-320 (`a03`). The stopping point therefore optimises the developed fit. Whether the emerging part is under- or over-iterated cannot be read from the saved files; it needs one run (P2-F5).

**(b) Category union.** Region codes do not collide (R01-R12 vs E01-E06). Construction codes are mapped onto the shared developed names on purpose (C035). All three tree models see one concatenated frame with one category list: XGBoost uses the codes of that list, LightGBM the same pandas categories at fit and predict, CatBoost strings. No test level is unseen in training (`a09`). Verdict: as intended.

**(c) Re-balance.** Applied after the early-stopped fit (`oof_predictions.py:249-257`). It uses y and w of the emerging fit rows only, not the scored fold, so there is no leak. The C027 holdout is part of the fit rows.

**(d) Year dropped.** Grouped OOF pred/actual by year: motor 1.023 / 0.975, buildings 0.821 / 1.285, contents 0.972 / 1.028 (`a10`). The models average the years and do not recover them through proxies (every 2023 row has zero policy history; year-specific weather). Weighting 2024 more is not defensible for buildings (fire noise in 2023). For motor the own-trend factor (C043) already does the equivalent.

## H5 Features (`audit_p2\a09`, `a12`, `a15`)

Raw emerging columns not used: identifiers and dates, `customer_service_contacts`, `customer_since_year` (used as tenure), `year_built` (property_age), `renewal_decision`, `surname`, `branch_id`, `previous_insurer`, `email_domain`, `make_model` (motor) and `occupation` / `marital_status` / `document_preference` (property). No emerging-only column is unused except those.

Out-of-fold residual test on the P2-S05 OOF: per-level A/E factors with credibility K = 50 claims, estimated on the other grouped folds (or on 2023 for the time fold) and applied to the held-out rows. No dropped column gains >= 0.001 on both schemes. The only both-positive case is buildings `email_domain` (+0.0009 / +0.0024, 5 levels, fire-dominated line), which is likely noise. make_model: -0.0023 / -0.0013. occupation: -0.0022 / -0.0006 (motor), -0.0049 / +0.0024 (buildings). Used features show no residual either (age bands, channel, region, gender, missing-count, written-premium bands).

Weather: per-peril frequency moves strongly with realised weather (motor glass_hail 0.52 -> 1.56 across hail quintiles; property flood 0.25 -> 2.17 across rainfall quintiles), and the residual by weather decile is flat, so the models already use it. 2025 weather exists for every test row.

## H6 Target and metric (`docs/references/part2_kaggle_*`; `audit_p2\a13`)

The scored quantity is the ultimate cost of claims with loss date in the row period, net of excess and recoveries, excluding declined claims, as a row total. Our target matches. Claim amounts are already net of excess: incurred minimum 0.12 CU, many claims below the compulsory 250 / 350, compulsory excess constant per line. Recoveries are netted (5.4% of motor incurred) and repudiated claims are excluded. Claims are valued 2024-12-31. The pages say nothing about inflation or about a 2025 exposure basis. Test exposure is the row length, which is what we use.

## H7 Blend concentration and shake-up (`audit_p2\a17`, `a19`, `a27`)

| line | adopted | leave-one-fold-out re-search vs equal (grouped) | in-sample optimism | bootstrap weights (40 reps) |
|---|---|---|---|---|
| motor | equal (guard) | -0.0001 | 0.0001 | lgbm 0.57 +/- 0.26 |
| buildings | cat .95 / lgbm .05 | **+0.0058** | 0.0007 | cat 0.88 +/- 0.14 (5%: 0.60) |
| contents | glm .55 / xgb .25 / lgbm .20 | -0.0005 | **0.0026** | glm 0.51 +/- 0.15 |

The contents weights are supported by the time fold (0.1072 vs 0.1046 equal) but not out of search on the grouped folds.

Public-board-sized noise (one third of the rows, bootstrap from the time fold): a single submission's D2 has SD 0.015 at 20k rows, about 0.008 at 70k. The P2-S05 - P2-S04 difference has SD 0.0018 (about 0.001 at 70k). The P2-S05 - P2-S01 difference has SD 0.009 (about 0.005).

Deviance shares (time fold): motor 47%, buildings 30%, contents 23% of the null deviance. The top 1% claim rows carry 12% and zero rows 58%. In training the rows later `declined_by_insurer` carry 22% / 52% / 21% of ultimate cost on 0.5% / 0.4% / 0.1% of rows (`a07`). That is where the board noise comes from.

Expected private evaporation: at most about 0.001 (contents optimism x its share, plus level noise).

## H8 Other checks

**The P2-F1 evidence (`audit_p2\a20`, `a21`).** Kaggle-like D2, each line sum-calibrated (level-neutral):

| recipe | time | grouped | grouped NB rows | grouped renewal rows | grouped test-mix | time test-mix | public |
|---|---|---|---|---|---|---|---|
| P2-S01 (emerging-only) | 0.10971 | 0.12306 | 0.12334 | 0.11931 | 0.13006 | 0.12016 | 0.14370 (no trend; S02 0.14460) |
| P2-S03 | 0.11546 | 0.12623 | 0.12803 | 0.12231 | 0.13421 | 0.13675 | 0.14693 |
| P2-S04 | 0.14179 | 0.14165 | 0.12864 | 0.13981 | 0.13934 | 0.14882 | 0.15193 |
| P2-S05 | 0.14486 | 0.14311 | 0.12863 | 0.14151 | 0.13973 | 0.15148 | 0.15123 |

Steps, validation vs public:

| step | time | grouped | grouped test-mix | public |
|---|---|---|---|---|
| S03 -> S04 | +0.0263 | +0.0154 | +0.0051 | +0.0050 |
| S04 -> S05 | +0.0031 | +0.0015 | +0.0004 | -0.0007 |
| S02 -> S05 | - | +0.0201 | +0.0097 | +0.0066 |

The time fold over-states emerging-only further because its emerging-only fit had one year.

**Leakage (`a11`).**
- `customer_service_contacts`: training rows with claims 3.4 vs 0.42 without; the test (0.40) matches the no-claim rows, so there is no test-side leak and C008 holds.
- `renewal_decision` is training-only.
- Policy history: strictly prior years only. Every 2023 row has zero history, and the models do not use that as a year proxy (H4d).
- Test NB customers, policies and addresses are unseen in training: 0.0% customers, 1.5% addresses (`a25`).

**Rows ending early.** 24% of test renewal rows end early vs 16% in training. The claim frequency per exposure-year is the same as for full-length rows (motor 0.204 vs 0.201), so there is no exploitable signal (`a07`).

**Seasonality.** Training NB exposure is concentrated late in the year; test NB runs January to December. The implied bias of the training NB rate is +0.3% frequency / +2.3% cost (motor), +2.5% / +5.0% (buildings) and +1.7% / +6.8% (contents). These come from one year of monthly costs and are small next to the NB sampling error (`a18`).

**Within-NB ranking.** On grouped NB rows the P2-S05 blend scores 0.054 (motor), 0.098 (buildings) and 0.172 (contents); the best single candidates score 0.071 (em:xgb), 0.117 (em:cat) and 0.176 (w0.3:glm). Per-model table in `a23`.

**Test-mix weight search (diagnostic only, `a24`).** Searched on the grouped folds, weights do not survive leave-one-fold-out (motor -0.0018, buildings -0.0064, contents -0.0002 vs P2-S05) but gain on the time fold (+0.0040 / +0.0013 / +0.0016). The NB evidence is too thin to tune on, hence the gate-only use in P2-F1.

## Not tested

- H4a / P2-F5: whether emerging-holdout early stopping or a developed offset improves the emerging fit needs model training.
- Interactions beyond one-way. The residual tests are one-way only; interactions specific to the NB segment cannot be checked with 77-697 NB claims per line.
- How the leaders' models differ. The board shows only their scores.
- Development tail beyond 24 months and the IBNER of open claims (no valuation history).
- Effect of 2025 weather that lies outside the emerging 2023-24 range (freezing days mean 28.9 vs 21.0 / 17.5); there is no truth to score it against.
