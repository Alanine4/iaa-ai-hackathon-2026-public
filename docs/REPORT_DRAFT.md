# Pricing 2025 claim cost for motor, buildings and contents in two markets

## 1. Problem and data

We predict the 2025 claim cost of each policy row in private motor, home buildings and home contents. The target is the ultimate cost of claims with a loss date in the row's period, net of excess and recoveries. Declined claims do not count. It is a row total, not an annual rate. Scoring is Tweedie D² with power 1.5 against a constant-rate baseline.

Part 1 is a developed market with 2,409,311 training rows for 2021 to 2024, 163,150 claims valued at 2024-12-31 and 626,639 test rows. Part 2 is an emerging market with 2 training years and 209,063 test rows. Its construction codes, 80 occupation spellings and age bands differ from Part 1. We fix 10 documented data issues in code at load time (appendix A1), and the raw files are never edited (C007).

| Item | Developed | Emerging |
|---|---|---|
| Motor exposure-years, 2023-24 | 399,468 | 38,969 |
| Motor claim frequency, 2023-24 | 13.48% | 18.09% |
| Motor mean reported severity | 3,862 CU | 2,250 CU |
| New business in 2025 test rows | 15% | 76% |
| Test policies seen in training | 85% | 24% |
| Broker share of motor exposure | 25% | 70% |
| Missing values in affected columns | about 2.5% | about 12% |

## 2. Target and development

A claim attaches to the row of the same policy whose period contains its loss date. The row target is incurred minus recoveries. Repudiated claims are excluded, unmatched claims are listed and left out, and totals reconcile with a difference of 0 (C005, C037).

The claims file is a single snapshot, so no triangle is possible. We fit a cost-weighted reporting-lag curve per line on accident year 2021. Each row's days in force give a completeness factor, and developed cost is reported cost divided by it (C006). Motor 2024 pure premium rises from 454 reported to 585 developed, inside the 564 to 606 range of 2021 to 2023. Motor bodily injury is the slow peril: 43% of cost, 14% reported at 30 days.

The emerging market has no tail of its own, so it borrows the developed accident year 2021 curves and peril shares (C037). At 365 days the motor curve reaches 0.961 for emerging 2023 against 0.922 for the curve used (appendix A4). The audit puts the resulting overstatement of the 2024 ultimate at 1.8% on motor and 2.5% on buildings (P2-F9), and we accepted it.

Each claim is capped at its line's 99.5th percentile. The models learn capped cost, and a per-line load adds back the cost above the cap (C014).

## 3. Modelling approach

Part 1. Every model fits capped cost per unit of completeness-weighted exposure with a Tweedie loss of power 1.5 (C010). The candidates are a banded GLM, a spline GAM (C011), LightGBM, XGBoost and CatBoost (C025). Log-link LightGBM ended at 0.83 of actual cost on its own training rows, so each model is rescaled to the training total (C013). A feature policy dropped columns without frequency signal, among them `branch_id`, motor `make_model` and the cross-line counts (C026). Per-line blend weights come from out-of-fold predictions on a 0.05 grid. They are used only if they beat equal weights on both folds (C021, C030), and motor fell back to equal weights. A per-line log-linear trend takes predictions to 2025 at half its fitted size: motor 1.018, buildings 1.095, contents 1.064 (C019).

Part 2. The same code runs with a market setting (C034), and a shared schema maps ages, construction codes and occupation across markets (C035). On emerging data alone (P2-S01) the GLM and GAM overfit, with buildings time-fold `d2_cal` of -10.5 and -30.7. We therefore trained on both markets together (C038). Each developed row joins the fit at weight w times exposure, with `market` as a feature. After each fit the level is re-balanced on the emerging fit rows. Pooling lifted the time-fold `d2_cal` from 0.0762 to 0.0915 on motor, 0.0410 to 0.1065 on buildings and 0.0779 to 0.1072 on contents. Validation then set w per line: 1.0 for buildings, 0.3 for motor and contents (C040).

Later, 2 changes passed the test-mix gate of section 6. Contents became a blend across the 2 weight runs (P2-S12). For the 68,823 motor new-business test rows we added models trained on new-business rows of both markets. Those rows take an equal-weight blend of 12 pooled and new-business candidates, because searched weights lost the guard (C047, P2-S13). Buildings and contents keep the developed half trend (1.0948 and 1.0636), and motor uses the emerging market's own observed trend, 1.1606 (C043).

| File | Recipe | Public | Private |
|---|---|---|---|
| S09 | Part 1 final: 5 models, guarded weights, half trend | 0.12863 | 0.13689, rank 12 of 51 |
| S09 blend | Post-deadline blend, 0.7 S09 + 0.3 teammate model (OBLEMOS Blend4); late submission, not ranked | 0.12949 | 0.13718 |
| P2-S01 | Emerging data only | 0.14370 | after 2026-10-31 |
| P2-S04 | Pooled training on all lines, w = 0.3 | 0.15193 | |
| P2-S10 | Buildings w = 1.0, motor at its own trend | 0.15414 | |
| P2-S12 | Contents blend across 2 weight runs | 0.15474 | |
| P2-S13 | Motor new-business segment blend | 0.15577 | |
| P2-S14 | Buildings blended with the teammate's Part 1 GAM (C054) | 0.15781 | |
| P2-S15 | P2-S14 recipe with 3-seed averaged tree predictions (C057); final candidate | 0.15807 | |

S09 is the ranked Part 1 entry. The Part 1 method we present is a post-deadline blend of 0.7 S09 and 0.3 of a teammate's model, OBLEMOS Blend4, put on S09's level per line (`oblemos_part1/blend_with_S09.py`). It was scored as a late submission, so it is not ranked. The final Part 2 candidate, P2-S15, is the P2-S14 recipe with LightGBM, XGBoost and CatBoost test predictions averaged over 3 seeds (C057). Seed averaging leaves the out-of-fold predictions unchanged, so no validation gate applies. We adopted it by a rule fixed before the score was seen: keep it unless the public score falls more than 0.0005 below P2-S14. Its public score is 0.15807, 0.00026 above P2-S14, which is under our 0.0005 tie band.

## 4. Major decisions and what was rejected

Every change faced a rule fixed before the run (section 6).

| Decision | Evidence | Outcome |
|---|---|---|
| C022 per-peril curves | Mix equals the pooled curve (max difference 5e-16) | No effect |
| C023 peril frequency x severity | Motor `d2_cal` 0.0801 vs 0.0893 | Rejected |
| C024 cross-line features | Gain rank 60th of 66 or lower; motor -0.001 to -0.002 | Rejected |
| C031 `make_model` encoding | Mean +0.0003; 2024 fold fell | Rejected |
| C032 tree re-tuning | Motor +0.0027 mean, -0.0008 on 2024 | Rejected |
| C039 pooled tuning, 23 settings | None beat defaults on both schemes | Rejected |
| C041 cross-run blend, raw rule | Contents +0.0018, below 0.002 | Rejected |
| C042 GLM market interactions | Worse on all 6 folds; blend gain in-sample | Rejected |
| C043 motor own trend 1.1606 | Motor pure premium +10.7% vs 1.75% applied; P2-S10 +0.0029 | Adopted |
| C044 test-mix gate | Matched the public S03 to S04 step (+0.0051 vs +0.0050) | Adopted |
| C045 new-business weighting | Contents worse on every view | Rejected |
| C046 emerging-only re-tuning | No line passed the gate | Rejected |
| C047 new-business segment | Motor test-mix grouped +0.0050 | Motor adopted |

The motor level is the only decision taken mainly from the public board, though the 2023-trained model had also under-predicted 2024 motor by 6%. The 6 scalar probes (P2-S06 to S11) left little gain beyond the full own trend, about +14% on motor. The property lines keep the developed half trend, because the 2023 fire year (buildings 2024/2023 = 0.63) rules out their own. The risk is that a level read from about a third of the rows does not hold on the rest. On a board of that size a single line's optimal level has a sampling SD of 4.7% for motor (audit H1). The C047 buildings renewal segment passed the gate mechanically. We rejected it because its gain came from a new-business GLM on renewal rows.

## 5. AI-assisted workflow

We used Claude Code in 3 roles. A coordinator planned each step, fixed the decision rule before a run and chose what to submit. Workers wrote a failing test, then the code, ran the experiment and logged the change in `docs/CHANGELOG.md` (C000 to C047). A read-only auditor trained nothing and backed every finding with a script (B1 to B8 in Part 1, A13; P2-F1 to P2-F10 in Part 2, A24). `docs/AI_LOG.md` (A01 to A26) records what each step produced, how it was checked and whether we accepted, modified or rejected it.

The team lead reviewed a sign-off sheet of 11 key AI decisions (`docs/HUMAN_REVIEW.md`, R01 to R11). They run from the cleaning rules (C003, C004) to the Part 2 pooled design and trend. Each item gives its evidence with file and line references. The team lead approved all 11 on 2026-10-25 in a single reply, without item-level reasons. Later Part 2 choices and the section 8 conventions (R12 to R15) were still unsigned on 2026-10-29.

AI output was often corrected or rejected:

- The worker showed that the coordinator's per-peril curves (C022) equal the pooled curve, so they changed nothing (A09).
- The Part 1 audit found that motor trees overfit `make_model` and that `branch_id` lowered every model; both fed the C026 drop policy (A13, A14).
- The C042 blend met the adoption rule (+0.0024). The coordinator still rejected it: the gain was in-sample and the new-business level shifted 21% untested (A23).
- The Part 2 audit overturned our validation design (P2-F1) and found a diagnostic on raw exposure, since fixed (P2-F8, A24).
- A post-hoc control after C047 scored higher on every view. We kept it as a diagnostic and did not submit it (A26).

## 6. Validation and guardrails

Part 1 uses 2 expanding-window folds that score 2023 and 2024. Part 2 has 1 time fold (train 2023, score 2024) and a grouped 5-fold split on `policy_id` (C036), scored on emerging rows only. `d2_cal` rescales each fold to its actual total, so it measures ranking rather than level. There are no random splits and no year feature.

The adoption rules were set before the runs. A change needs a mean gain of at least 0.002 with no fold lost (C018). Searched blend weights are used only if they beat equal weights on every scheme, and this guard is code, not a hand edit (C030). Gaps under 0.0005 count as ties and the simpler option wins (P2-S11).

The Part 2 audit found the main flaw in our validation (P2-F1). Validation rows were 92% renewals, while 77% of the 2025 test rows are new business. Every Part 2 choice up to P2-S05 had been tuned on renewal behaviour. Re-weighting the folds to the test mix reproduced the board. The S03 to S04 step was +0.0154 on raw grouped folds, +0.0051 on the test mix and +0.0050 in public. Since then a change must not lose on the raw time fold, the raw grouped mean or the test-mix grouped mean. It must also gain 0.002 on the raw mean or the test-mix grouped mean (C044). Weights are still searched on raw folds: 77 to 697 new-business claims per line can reject a change but cannot tune one.

A SHA256 manifest, read-only attributes and a test guard every raw byte (C007). The developed files in the Part 2 release matched the Part 1 manifest (C033). Modules are written test-first, and 340 tests pass (A29). Reproduction is checked byte for byte. The current code rebuilds S09 with the SHA256 of the submitted file and P2-S05 exactly, and P2-S13 equals P2-S12 outside motor new business (C047). `reproduce.ps1` rebuilds Part 1 from the raw data, and with `-Part2` also P2-S13. It does not run the later steps: the OBLEMOS buildings blend of P2-S14 (C054, `docs/PART2_OBLEMOS.md`) and the seed-averaged refit and tagged blend behind P2-S15 (C057 to C059) are run by hand with the commands recorded there.

A Part 1 lesson concerns the final choice. After the close our notes recorded a teammate's file as public 0.1356, 0.007 above S09 and not among the 2 selected files. The Kaggle listing shows that 0.13562 is its private score. Its public score was 0.12440, and S09 led on both boards (0.12863 and 0.13689). The choice held, but the comparison mixed 2 boards and the teammate's code was not in the repository at the time. It is now: `oblemos_part1/` for Part 1, and `scripts/oblemos_emerging.py` with the C048 to C056 entries for Part 2 (merged 2026-10-29). The Part 2 selection is therefore a team decision against a checklist. It compares scores from a single board and the 6 validation views per candidate, on files the repository rebuilds.

## 7. Assumptions and limitations

Assumptions in both parts: open-claim case estimates are unbiased, and the accident year 2021 reporting pattern holds for later years and for the emerging market. T and W both mean timber (C004).

- The emerging market's 2 years give 1 time fold with about 500 claims per year on each property line. 2024 is also the least developed year.
- Reporting curves are borrowed. Nothing beyond 24 months can be checked, and 20% of 2024 emerging motor claims are open.
- The level rests partly on public-board probes: the developed half trend (P2-S02, +0.0009, below our 0.002 rule) and the motor level. Part 1 shows the risk. The private board preferred no trend to half trend by 0.0001, after the public board had preferred half by 0.0009. The S08 probe of audit B4 also flipped sign (public -0.0003, private +0.0001). The motor gain (+0.0029) is larger and agrees with the time fold.
- New-business evidence is thin, at 37 to 697 claims per line and scheme. On new business, predicted-to-actual ratios run 0.81 to 1.01 for motor, 1.30 to 2.23 for buildings and 0.74 to 0.84 for contents. The 2 schemes disagree for motor. We made no segment level adjustment; the C047 models lifted the motor new-business level by 4%.

## 8. Office premium build-up for the 20 reference policies

R is the P2-S15 prediction for the row, the final candidate (`work/p2_s15_seeds3.csv`; the tables come from `scripts/part2_office_premium.py --predictions work/p2_s15_seeds3.csv`), matched on `row_id`, and is already a row total. We apply these conventions to the emerging column of `expenses.parquet`:

- Cat = 0.12 R on buildings and contents, 0 on motor. The unit is a share of property risk premium, and we treat contents as property.
- F = 85 CU per policy, charged in full and not pro-rated by exposure.
- k = variable expense 0.15 + commission + premium tax 0.08 + target profit margin 0.08, all shares of the office premium. Commission is 0.20 for broker and 0 for direct; the comparison-site value is blank, and none of the 20 policies uses that channel.
- P = (R + Cat + F) / (1 - k). The divisor is 0.49 for broker and 0.69 for direct. There is no minimum or maximum premium.

| row_id | line | channel | NB | exposure | R | Cat | F | 1 - k | P | WP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M00118569 | motor | direct | no | 0.9397 | 922.76 | 0.00 | 85.00 | 0.69 | 1,460.52 | 1,095.48 |
| M00129413 | motor | broker | yes | 0.3315 | 212.69 | 0.00 | 85.00 | 0.49 | 607.53 | 354.15 |
| M00153851 | motor | broker | yes | 0.9616 | 368.16 | 0.00 | 85.00 | 0.49 | 924.81 | 414.72 |
| M00161473 | motor | broker | yes | 0.9644 | 1,135.24 | 0.00 | 85.00 | 0.49 | 2,490.28 | 1,131.78 |
| M00167496 | motor | broker | yes | 0.9836 | 777.40 | 0.00 | 85.00 | 0.49 | 1,760.00 | 463.95 |
| M00176777 | motor | broker | yes | 0.9918 | 1,072.30 | 0.00 | 85.00 | 0.49 | 2,361.85 | 443.61 |
| M00185311 | motor | direct | yes | 0.9644 | 1,105.96 | 0.00 | 85.00 | 0.69 | 1,726.03 | 872.31 |
| B00215707 | buildings | broker | yes | 0.9863 | 342.12 | 41.05 | 85.00 | 0.49 | 955.46 | 272.44 |
| B00215809 | buildings | direct | yes | 0.9753 | 480.49 | 57.66 | 85.00 | 0.69 | 903.11 | 237.22 |
| B00217584 | buildings | broker | yes | 0.9671 | 825.49 | 99.06 | 85.00 | 0.49 | 2,060.29 | 323.36 |
| B00218282 | buildings | broker | yes | 0.9726 | 391.04 | 46.92 | 85.00 | 0.49 | 1,067.27 | 230.80 |
| B00223369 | buildings | broker | yes | 0.9945 | 515.20 | 61.82 | 85.00 | 0.49 | 1,351.08 | 204.56 |
| B00229821 | buildings | broker | yes | 0.9918 | 387.92 | 46.55 | 85.00 | 0.49 | 1,060.14 | 285.60 |
| B00250679 | buildings | broker | yes | 0.1096 | 38.71 | 4.65 | 85.00 | 0.49 | 261.95 | 180.72 |
| C00259568 | contents | broker | no | 1.0000 | 40.88 | 4.91 | 85.00 | 0.49 | 266.92 | 83.18 |
| C00270331 | contents | broker | yes | 0.9726 | 37.45 | 4.49 | 85.00 | 0.49 | 259.07 | 140.29 |
| C00283470 | contents | broker | yes | 0.9726 | 86.59 | 10.39 | 85.00 | 0.49 | 371.39 | 117.97 |
| C00286287 | contents | broker | yes | 0.9671 | 162.14 | 19.46 | 85.00 | 0.49 | 544.08 | 269.23 |
| C00302985 | contents | broker | yes | 0.9918 | 60.81 | 7.30 | 85.00 | 0.49 | 312.46 | 115.88 |
| C00314425 | contents | broker | yes | 0.0685 | 5.97 | 0.72 | 85.00 | 0.49 | 187.11 | 102.87 |

Amounts in CU; NB = new business; WP = written premium; source `scripts/part2_office_premium.py`. For B00229821, R 387.92 + Cat 46.55 + F 85.00 = 519.47, and 519.47 / 0.49 gives P = 1,060.14 (split by item in appendix A6). The 20 rows sum to R = 8,969.31 and P = 20,931.34 (mean 1,046.57), against 7,340.12 of written premium. Against the P2-S13 table the 20-policy total moved by less than 0.1%, while one buildings policy (B00218282) moved by a quarter, so individual quotes use the seed-averaged file.

Written premium is the current tariff and R is our 2025 cost estimate, so we do not expect them to match. P exceeds written premium on all 20 rows, by 1.33 to 6.60 times (median 2.92). The loadings drive most of it. A broker row with R = 0 would still cost 173.47, and F, Cat and the shares make up 36% to 97% of P. R alone exceeds written premium on 10 rows, 6 buildings and 4 motor, all new business. Our buildings new-business predictions ran 1.30 times actual on the grouped folds (77 claims), so part of that gap may be our own overstatement. Written premium does not scale with exposure (Part 1 audit; the emerging test file agrees). The 3 short rows therefore set a part-year P against an annual tariff, and F dominates them. Pro-rated by exposure, P would fall from 607.53 to 491.56 (M00129413, 0.33 years), from 261.95 to 107.49 (B00250679, 0.11) and from 187.11 to 25.52 (C00314425, 0.07).

---

## Appendix, not part of the 5 pages

### A1. Data quality issues and treatment

| Issue | Size | Treatment |
|---|---|---|
| Q1 `exposure` holds day counts, 370/365 or a 360-day basis | 5,723 motor, 4,176 buildings, 4,305 contents rows | Recomputed from period dates on a 365-day year (C001) |
| Q2 motor `customer_id` lost leading zeros | 98,110 training and 25,554 test rows | Padded to 8 characters; 100% join to customers (C002) |
| Q3 `sum_insured` inflated by 1,000 | Values above 5m, including 370 buildings and 360 contents test rows | Divided by 1,000 and flagged; same policies differ by exactly 1,000 across years (C003) |
| Q4 `construction_type` case and codes T/W | 13,250 policies switch from T to W | Lower-cased; T and W mapped to timber (C004) |
| Q5 claims matching no policy row | 294 motor, 67 buildings, 49 contents | Excluded from the target and listed; reconciliation difference 0 (C005) |
| Q6 `customer_service_contacts` | About 3.0 on claimed rows against 0.41 on unclaimed rows | Dropped as leakage (C008) |
| Q7 about 2.5% random missing values | Many columns | Not imputed for tree models; the GAM uses median imputation plus missing indicators (C011) |
| Q8 `telematics_score` empty for 87% of motor rows | Structural: no device | Not imputed |
| Q9 `renewal_decision` only in training | All rows | Dropped; known only after the row ends |
| Q10 training and test drift | Median `renewal_count` 6 to 3, `vehicle_age` 7 to 9, `ncb_years` 7 to 8 | Year is not a feature; the audit found the drift internally consistent (`renewal_count` +1.0 per year) |

Part 2 re-check (C033): Q1 recurs in the emerging training tables only (253 motor rows); Q2 is absent; Q3 recurs and the 5m threshold still separates; Q4 changes to the BRK, STN, WOD and MIX codes (C035); 36 claims match no row; missing values rise to about 12%; the Q8 columns are absent. `customer_service_contacts` again separates claimed rows (3.4 against 0.42 contacts) and stays out (audit H8).

### A2. Part 1 target and validation detail

- Repudiated claims (1,711, all zero amounts) are excluded (C005). 20.7% of 2024 motor claims were still open at 2024-12-31.
- Per-line caps: motor 51,585, buildings 147,445, contents 28,449 CU in Part 1; 27,578, 220,808 and 28,312 CU in the emerging market (C014, C037). The capped share is 8.8%, 4.2% and 2.1% of cost by line in Part 1 and 5.0%, 4.9% and 2.1% in the emerging market.
- Motor perils other than bodily injury are 76% to 99% reported at 30 days (`docs/target_summary.txt`).
- Trees stop early on a random 10% of the fitting rows (C027), and the S09 refit averages 3 seeds (C029).
- Rescaling each model to the training total raised motor validation D² from 0.066 to 0.073 (C013).
- Per-line tuning was adopted only where the mean gain was at least 0.002: motor LightGBM gained 0.0075, motor XGBoost 0.0031, contents XGBoost 0.0024 (C015, C018).
- The feature policy (C026) raised mean LightGBM `d2_cal` from 0.0832 to 0.0887 on motor, 0.1094 to 0.1134 on buildings and 0.0891 to 0.0936 on contents.
- Motor searched weights lost to equal weights on the 2024 fold (0.09031 against 0.09071), so motor uses equal weights (C021, C030).
- Validation folds compare reported cost with prediction times completeness. The final S09 blends scored a mean 2-fold `d2_cal` of 0.0919 (motor), 0.1144 (buildings) and 0.0945 (contents).
- C023 detail: per-peril frequency by severity scored 0.0801 against 0.0893 on motor, 0.1045 against 0.1094 on buildings and 0.0879 against 0.0911 on contents. 2024 bodily injury counts were over-predicted by 1.43 (`docs/cv_freqsev.md`).
- C031 detail: an out-of-fold target encoding of `make_model` (credibility constant 2,000 exposure-years) carried weak signal: code relativities correlated 0.18 across years, against -0.05 for shuffled codes. The equal blend gained 0.0003 on the mean and fell on 2024.
- Audit B1 and B2: without `make_model`, motor LightGBM on the 2023 fold went from 0.08082 to 0.08868; `branch_id` lowered every model on that fold (buildings LightGBM 0.1071 to 0.1051). Audit B8: trees had stopped early on the scored fold and now stop on a random 10% of the fitting rows (C027).

### A3. Part 2 validation detail

Emerging-only against pooled training, `d2_cal` (time fold / grouped mean):

| Line | Emerging only | Pooled, w = 0.3 | Pooled, w = 1.0 | Adopted w |
|---|---|---|---|---|
| Motor | 0.0762 / 0.0677 | 0.0915 / 0.0740 | 0.0908 / 0.0730 | 0.3 |
| Buildings | 0.0410 / 0.0732 | 0.1065 / 0.1132 | 0.1169 / 0.1180 | 1.0 |
| Contents | 0.0779 / 0.0934 | 0.1072 / 0.1100 | 0.1069 / 0.1128 | 0.3 |

The GAM left the pooled runs after it exceeded the memory cap on 756k rows. Tree models needed level re-balancing on emerging rows, with factors of 1.08 to 1.26 at weight 0.3 on the property lines and 1.01 to 1.10 at weight 1.0. The P2-S12 contents blend takes GLM 0.35 from the w = 0.3 run and LightGBM 0.65 from the w = 1.0 run (C044). P2-S05 (buildings at w = 1.0) scored 0.15123 in public against 0.15193 for P2-S04, so the board disagreed with validation. We kept the validation rule (C040).

Audit P2-F1 evidence: Kaggle-like D² of the 3 lines stacked, each line sum-calibrated (audit H8).

| Recipe | Time | Grouped | Grouped test-mix | Public |
|---|---|---|---|---|
| P2-S01 | 0.10971 | 0.12306 | 0.13006 | 0.14370 |
| P2-S03 | 0.11546 | 0.12623 | 0.13421 | 0.14693 |
| P2-S04 | 0.14179 | 0.14165 | 0.13934 | 0.15193 |
| P2-S05 | 0.14486 | 0.14311 | 0.13973 | 0.15123 |

Gains of the adopted Part 2 changes over the recipe they replaced, in the order raw time / raw grouped / test-mix time / test-mix grouped / new-business time / new-business grouped:

| Change | Gains |
|---|---|
| P2-S12 contents blend, over the P2-S05 contents recipe | +0.0005 / +0.0032 / +0.0020 / +0.0041 / not reported / +0.0039 |
| P2-S13 motor new-business segment, over the P2-S12 motor recipe | +0.0001 / +0.0009 / +0.0008 / +0.0050 / +0.0012 / +0.0032 |

The P2-S12 grouped gains were measured on the folds the weights were searched on; its time-fold gains are out of search (C044).

New-business segment check, predicted-to-actual on reported cost for the P2-S05 blend (`docs/part2_new_business_check.md`):

| Line | Time fold, new / renewal | Grouped, new / renewal | New-business claims, time / grouped |
|---|---|---|---|
| Motor | 1.012 / 0.933 | 0.812 / 1.025 | 382 / 697 |
| Buildings | 2.229 / 1.607 | 1.296 / 0.981 | 37 / 77 |
| Contents | 0.737 / 1.111 | 0.837 / 1.018 | 65 / 102 |

Further Part 2 limitations: buildings 2023 was a fire year (49 fire claims for 3.22m CU against 21 for 1.13m in 2024, audit H1), the large-loss load is uniform within a line, and the emerging caps rest on 36, 6 and 6 claims above the cap.

### A4. Reporting curves, emerging against developed

Cost-weighted cumulative share reported (`docs/part2_emerging_target_summary.txt`, C037):

| Line | Days | Emerging 2023 | Developed 2023, same truncation | Developed 2021, used |
|---|---|---|---|---|
| Motor | 30 | 0.603 | 0.579 | 0.556 |
| Motor | 90 | 0.774 | 0.717 | 0.723 |
| Motor | 365 | 0.961 | 0.970 | 0.922 |
| Buildings | 30 | 0.913 | 0.829 | 0.857 |
| Buildings | 90 | 0.952 | 0.927 | 0.933 |
| Buildings | 365 | 0.998 | 0.992 | 0.992 |
| Contents | 30 | 0.961 | 0.947 | 0.949 |
| Contents | 90 | 0.991 | 0.995 | 0.989 |
| Contents | 365 | 0.993 | 0.996 | 0.990 |

The peril mix differs most on buildings: fire is 0.454 of emerging cost against 0.237 in developed accident year 2021. Mean completeness of the 2024 motor rows is 0.7672 with the developed peril shares and 0.8202 with the emerging ones.

### A5. All submissions

Part 1. Private scores come from the Kaggle submissions listing after the close.

| ID | Recipe | Public | Private |
|---|---|---|---|
| S01 | Equal blend of GLM, GAM, LightGBM, XGBoost; full trend | 0.12453 | 0.13347 |
| S02 | S01 with no trend | 0.12376 | 0.13466 |
| S03 | S01 with half trend | 0.12464 | 0.13456 |
| S04 | Tuned LightGBM and XGBoost, equal weights, half trend | 0.12477 | 0.13279 |
| S05 | Equal blend of GLM, GAM and tuned trees | 0.12464 | 0.13477 |
| S06 | Per-line weights from out-of-fold predictions (C021) | 0.12579 | 0.13505 |
| S07 | Feature policy, inner early stopping, CatBoost, guarded weights | 0.12851 | 0.13693 |
| S08 | S07 with a lower trend on buildings and contents (probe) | 0.12820 | 0.13703 |
| S09 | S07 recipe with seed-averaged refit (final) | 0.12863 | 0.13689 |
| S09 blend | 0.7 S09 + 0.3 teammate model (OBLEMOS Blend4), post-deadline, late submission, not ranked | 0.12949 | 0.13718 |

Part 2, public scores (`docs/SUBMISSIONS.md`). On 2026-10-29 P2-S13 ranked 23 of 35 teams on the public board. On 2026-10-30 P2-S15 ranked 18 of 38 (median 0.15773).

| ID | Recipe | Public |
|---|---|---|
| P2-S01 | Emerging only, no trend | 0.14370 |
| P2-S02 | P2-S01 with developed half trend | 0.14460 |
| P2-S03 | Contents pooled | 0.14693 |
| P2-S04 | All lines pooled, w = 0.3 | 0.15193 |
| P2-S05 | P2-S04 with buildings at w = 1.0 | 0.15123 |
| P2-S06 | Level probe, all lines x 1.10 | 0.15206 |
| P2-S07 | Level probe, motor x 1.10 | 0.15362 |
| P2-S08 | Level probe, buildings x 1.10 | 0.15029 |
| P2-S09 | Motor at the square root of its own trend, 1.0773 | 0.15284 |
| P2-S10 | Motor at its full own trend, 1.1606 | 0.15414 |
| P2-S11 | P2-S10 uplift on motor new business only (tie, not adopted) | 0.15434 |
| P2-S12 | P2-S10 with the contents blend across weight runs | 0.15474 |
| P2-S13 | P2-S12 with the motor new-business segment blend | 0.15577 |
| P2-S14 | P2-S13 with buildings = 0.4 teammate Part 1 GAM + 0.6 pooled CatBoost, at the P2-S13 buildings total (C054) | 0.15781 |
| P2-S15 | P2-S14 recipe with 3-seed averaged LightGBM, XGBoost and CatBoost test predictions (C057); final candidate | 0.15807 |

### A6. Office premium split by item

Each share-of-premium item in CU; every row adds up to P (`scripts/part2_office_premium.py`).

| row_id | R + Cat + F | variable expense | commission | premium tax | profit | P |
|---|---:|---:|---:|---:|---:|---:|
| M00118569 | 1,007.76 | 219.08 | 0.00 | 116.84 | 116.84 | 1,460.52 |
| M00129413 | 297.69 | 91.13 | 121.51 | 48.60 | 48.60 | 607.53 |
| M00153851 | 453.16 | 138.72 | 184.96 | 73.99 | 73.99 | 924.81 |
| M00161473 | 1,220.24 | 373.54 | 498.06 | 199.22 | 199.22 | 2,490.28 |
| M00167496 | 862.40 | 264.00 | 352.00 | 140.80 | 140.80 | 1,760.00 |
| M00176777 | 1,157.30 | 354.28 | 472.37 | 188.95 | 188.95 | 2,361.85 |
| M00185311 | 1,190.96 | 258.90 | 0.00 | 138.08 | 138.08 | 1,726.03 |
| B00215707 | 468.17 | 143.32 | 191.09 | 76.44 | 76.44 | 955.46 |
| B00215809 | 623.14 | 135.47 | 0.00 | 72.25 | 72.25 | 903.11 |
| B00217584 | 1,009.54 | 309.04 | 412.06 | 164.82 | 164.82 | 2,060.29 |
| B00218282 | 522.96 | 160.09 | 213.45 | 85.38 | 85.38 | 1,067.27 |
| B00223369 | 662.03 | 202.66 | 270.22 | 108.09 | 108.09 | 1,351.08 |
| B00229821 | 519.47 | 159.02 | 212.03 | 84.81 | 84.81 | 1,060.14 |
| B00250679 | 128.36 | 39.29 | 52.39 | 20.96 | 20.96 | 261.95 |
| C00259568 | 130.79 | 40.04 | 53.38 | 21.35 | 21.35 | 266.92 |
| C00270331 | 126.94 | 38.86 | 51.81 | 20.73 | 20.73 | 259.07 |
| C00283470 | 181.98 | 55.71 | 74.28 | 29.71 | 29.71 | 371.39 |
| C00286287 | 266.60 | 81.61 | 108.82 | 43.53 | 43.53 | 544.08 |
| C00302985 | 153.10 | 46.87 | 62.49 | 25.00 | 25.00 | 312.46 |
| C00314425 | 91.68 | 28.07 | 37.42 | 14.97 | 14.97 | 187.11 |

### A7. Further Part 1 limitations from the audit

- The trend may count the year level twice (B4). The residual trend implied is 1.017, 1.048 and 1.004 against 1.018, 1.095 and 1.064 applied. S08 scored 0.0001 above S07 on the private board, within noise but in the direction of B4.
- The 2025 motor large-loss load includes under-reported 2024 bodily injury and may be 1.4% low (B5).
- Recoveries appear only on closed claims, so open 2024 own-damage cost is about 1.4% overstated (B6).
- Prior-claim history is developed in cost but not in claim counts (B7).
- Open-claim reserving for bodily injury cannot be verified from one snapshot. Quickly reported severity falls from 20.7k to 16.6k across accident years while other motor perils rise, which could be under-reserving or a mix change.
- The customer snapshot date is unknown, so early training rows may see later information.
- Buildings and contents cost above the cap concentrates in the top risk decile, while the load is uniform within a line.
- The GAM uses a single penalty (C011).
