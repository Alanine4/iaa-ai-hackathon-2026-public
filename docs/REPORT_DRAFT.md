# Pricing 2025 claim cost for motor, buildings and contents: Part 1

## 1. Problem and data

We predict, for each of the 626,639 policy rows of 2025, the ultimate cost of claims with a loss date in the row's period, net of excess and recoveries, excluding repudiated claims. The training data has 2,409,311 policy rows for 2021 to 2024 across private motor, home buildings and home contents, plus 163,150 claims valued at 2024-12-31. Submissions are scored by Tweedie D² with power 1.5.

The raw tables contain ten documented quality issues, each handled in code at load time. The raw files are never edited (C007).

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

## 2. Target construction and development

A claim attaches to the policy row whose period contains its loss date. The row target is incurred minus recoveries, with repudiated claims (1,711, all zero amounts) excluded (C005). The claims file is a single snapshot, so recent accident years are under-reported: 20.7% of 2024 motor claims were still open. An incurred triangle is not possible. Instead we fit a cost-weighted reporting-lag curve per line from accident year 2021, compute a completeness factor for every row from its days in force, and develop reported cost as reported divided by completeness (C006). Motor 2024 pure premium rises from 454 reported to 585 developed, inside the 564 to 606 range of 2021 to 2023. Per-peril curves (C022) show that motor bodily injury is only 14% reported at 30 days against 95% to 99% for other perils.

Each claim is capped at the 99.5th percentile of its line (motor 51,585; buildings 147,445; contents 28,449). Models learn the capped cost, and the excess is added back through a per-line load (C014). The capped share is 8.8% of motor cost, 4.2% of buildings, 2.1% of contents.

## 3. Modelling approach

Every model fits capped cost per unit of completeness-weighted exposure with a Tweedie loss of power 1.5 (C010). The candidates are a banded one-hot GLM, a spline GAM with one global ridge penalty (C011), LightGBM, XGBoost and CatBoost (C025). Three decisions came from diagnostics:

- Log-link Tweedie boosting does not preserve the total: LightGBM ended at 0.83 of actual cost on its own training rows. Rescaling each model to the training total raised motor validation D² from 0.066 to 0.073 (C013).
- Shallow trees beat deep ones. Per-line tuning was adopted only where the mean cross-validated gain was at least 0.002, a threshold fixed beforehand: motor LightGBM gained 0.0075, motor XGBoost 0.0031, contents XGBoost 0.0024 (C015, C018).
- Developed pure premium rises over 2021 to 2024 (buildings 384 to 481), so a per-line log-linear trend scales predictions to 2025. We apply half the fitted trend (motor 1.018, buildings 1.095, contents 1.064) because the public board preferred half to none or full (C009, C019).

Validation uses two expanding-window folds (train 2021-22, score 2023; train 2021-23, score 2024) on reported cost against prediction times completeness. `d2_cal` rescales each fold to its actual total, so it measures ranking, not level. The feature policy (C026) removed columns without frequency signal and raised mean LightGBM `d2_cal` from 0.0832 to 0.0887 on motor, 0.1094 to 0.1134 on buildings and 0.0891 to 0.0936 on contents. Final predictions blend the five models with per-line weights found on out-of-fold predictions (non-negative, 0.05 grid, sum 1). Searched weights are used only if they beat equal weights on both folds (C021, C030); motor falls back to equal weights (0.09031 against 0.09071 on 2024). The tree refit averages three seeds (C029). The final blends score a mean two-fold `d2_cal` of 0.0919 for motor, 0.1144 for buildings and 0.0945 for contents. Two further changes tried after S09, a target encoding of `make_model` (C031) and a re-tuning of the trees (C032), did not pass the adoption rule, so S09 is the Part 1 submission.

| ID | Recipe | Public score |
|---|---|---|
| S01 | Equal blend of GLM, GAM, LightGBM, XGBoost; full trend (rank 19) | 0.12453 |
| S02 | S01 with no trend | 0.12376 |
| S03 | S01 with half trend | 0.12464 |
| S04 | Tuned LightGBM and XGBoost, equal weights, half trend | 0.12477 |
| S05 | Equal blend of GLM, GAM and tuned trees | 0.12464 |
| S06 | Per-line weights from out-of-fold predictions (C021) | 0.12579 |
| S07 | Feature policy, inner early stopping, CatBoost, guarded weights (rank 11) | 0.12851 |
| S08 | S07 with a lower trend on buildings and contents (probe) | 0.12820 |
| S09 | S07 recipe with seed-averaged refit | 0.12863 |

The leader's score at S07 was 0.12996. Validation and leaderboard scores sit on different targets and years, so we compare scores only within a column.

## 4. AI-assisted workflow

We worked with Claude Code throughout (`docs/AI_LOG.md`, A01 to A19). The assistant produced a first version at each stage, and a person or an automated check decided whether to keep it.

- Exploration and data quality: profiling of all 11 files, and flagging Q1 to Q10 with counts. Cleaning rules such as T/W = timber were inferred from the year split and policy transitions, then written as failing tests before implementation (A02 to A04).
- Target and leakage: claim matching and development were designed with AI and checked by full-data reconciliation (difference 0). A scan of every column for leakage found only `customer_service_contacts` (A05, A06).
- Modelling and debugging: models, the D² scorer and the trend factor were written test-first. A crash on a boolean column (C012) and a 9% to 18% under-prediction of tree models (C013) were each diagnosed by experiment before any fix (A07, A08).
- Independent review: a separate read-only AI audit produced findings B1 to B8, each backed by script output (A13).

Verification relied on tests written before the code (129 pass), ablations on time-based folds, and the audit. These cases show AI output being corrected or rejected:

- **C022 caught as a no-op.** The per-peril reporting curves were designed to correct the mix of slow and fast perils. The implementing assistant showed numerically that a cost-share mix of per-peril curves equals the pooled curve (maximum difference 5e-16), so row completeness could not change. The code stays as an input, with no accuracy claim (A09).
- **C023 rejected on evidence.** A per-peril frequency by severity model scored a mean `d2_cal` of 0.0801 on motor against 0.0893 for the best single model, 0.1045 against 0.1094 on buildings and 0.0879 against 0.0911 on contents. The main error was the peril mix: 2024 bodily injury counts were over-predicted by 1.43. It stayed out of the blend (A10, `docs/cv_freqsev.md`).
- **C024 features rejected.** Cross-line and household features ranked 60th or lower of 66 by gain, and re-adding them lowered motor by 0.001 to 0.002. The loss in the full set came from `branch_id` coded as a category (A11, C026).
- **Audit B1 and B2 fixed.** Motor trees overfit the 300-level `make_model`: without it, LightGBM on the 2023 fold went from 0.08082 to 0.08868. `branch_id` lowered every model on the 2023 fold (buildings LightGBM 0.1071 to 0.1051). Both led to the C026 drop policy (A13, A14).
- **Audit B4 probed, not adopted.** The audit argued that the trend counts the year level twice. A probe with a lower trend on buildings and contents (S08) scored 0.12820 against 0.12851, so half trend was kept.
- **Audit B8 fixed.** Tree models stopped early on the fold being scored while GLM and GAM did not. Early stopping now uses a random 10% of the fitting rows (C027).
- **C031 target encoding not adopted.** An out-of-fold target encoding of `make_model` (log relativity with credibility constant 2,000 exposure-years, 5-fold within the fitting rows so no row sees its own target) was built to recover the signal lost when the raw code was dropped. The codes carry real but weak signal (cross-year correlation of code relativities 0.18 against −0.05 after shuffling). The equal blend improved by 0.0003 on the mean and fell on the 2024 fold, below the 0.002 rule, so it was not adopted and the S09 files were restored (C031, A18).
- **C032 re-tuning not adopted.** The tree hyper-parameter search was repeated on the final feature set with inner early stopping. The best motor LightGBM setting gained 0.0027 on the mean but lost 0.0008 on the 2024 fold; the other lines gained 0.0005 to 0.0018. Nothing met the rule, so the parameters stayed (C032, A19).

## 5. Guardrails and governance

- **Change control.** Every change to data handling, code or assumptions has a row in `docs/CHANGELOG.md` (C000 to C032) with reason, evidence and code location. Code comments cite the same ID.
- **AI log.** `docs/AI_LOG.md` records what the assistant did, how it was checked and whether it was accepted, modified or rejected.
- **Raw-data protection.** A SHA256 manifest, read-only attributes and an integrity test fail if any raw byte changes (C007).
- **Tests before code.** The suite covers cleaning, matching, development, features, models, scoring and the blend guard.
- **Time-based validation.** Two expanding-window folds, no random splits, no `year` feature.
- **Adoption rules fixed in advance.** A tuned setting needs a mean gain of at least 0.002 (C018). Searched blend weights are used only if they beat equal weights on both folds, and this rule is now code (C030).
- **Restraint on the leaderboard.** Model, feature and weight choices were made on cross-validation. The leaderboard informed one parameter, the trend share (C016, C019), and one probe (S08).
- **Reproducibility.** `reproduce.ps1` rebuilds a submission from the raw data with pinned package versions.
- **Human review.** Some decisions are logged as pending sign-off: the T/W mapping, the C003 threshold and the bundled C026 drop of three customer columns.

## 6. Assumptions and limitations

Assumptions: open-claim case estimates are unbiased (C006); the reporting-lag pattern of accident year 2021 holds for later years; a year has 365 days; T and W both mean timber.

Limitations, taken from the audit:

- The trend may double count the year level already carried by `written_premium` and `sum_insured` (B4). The residual trend implied is 1.017, 1.048 and 1.004, against 1.018, 1.095 and 1.064 applied. The public board did not support the correction, but the gap was 0.0003 on about a third of the rows.
- The 2025 motor large-loss load includes under-reported 2024 bodily injury and may be 1.4% low (B5). Recoveries appear only on closed claims, so open 2024 own-damage cost is about 1.4% overstated (B6). Prior-claim history is developed in cost but not in claim counts (B7).
- Open-claim reserving for bodily injury cannot be verified from one snapshot. Quickly reported severity falls from 20.7k to 16.6k across accident years while other motor perils rise, which could be under-reserving or a mix change.
- The customer snapshot date is unknown, so early training rows may see later information.
- The large-loss load is uniform within a line, while buildings and contents excess concentrates in the top risk decile.
- Blend weights were re-estimated after the early-stopping fix (C028); the hyper-parameter search was repeated on the final feature set and changed nothing (C032). The GAM uses a single penalty (C011).
- The public score covers about 33% of the test rows. The final ranking uses the other 67% and may move.

## 7. Adaptation to the new market (Part 2)

### 7.1 What differs in the emerging market

The emerging market has about one tenth of the developed exposure (38,969 motor exposure-years against 399,468) and only two accident years. In the 2025 test set 76% of rows are new business against 15%, and 24% of policies appear in training. Motor frequency is 34% higher (18.09% against 13.48%) and severity 42% lower (2,250 against 3,862 CU). Construction type arrives as BRK, STN, WOD and MIX, occupation as 80 free-text spellings and age as bands. About 12% of values in affected columns are missing against 2.5%.

### 7.2 What we changed in the pipeline

Each market is a configuration; the developed path is unchanged (C034). A common schema (C035) cuts developed ages into the emerging bands and maps the four construction codes to brick, stone, timber and non_standard. Two accident years give no tail, so we apply the developed accident year 2021 reporting curves; at 365 days the motor curve reaches 0.961 for emerging 2023 and 0.922 for the curve used. Pooled training adds every developed row at a weight times exposure, with `market` as a feature, and re-balances the level on emerging rows after each fit (C038).

### 7.3 Validation design for a two-year market

Two years allow one time fold (train 2023, score 2024), so we add a grouped 5-fold on `policy_id` (C036). Searched blend weights are kept only if they beat equal weights on both, and the developed-row weight is chosen per line the same way. Pooled runs score emerging rows only. The time fold is weak: about 500 claims per year on each property line, and 2024 is the least developed year.

### 7.4 Results and what we adopted or rejected

Emerging-only GLM and GAM over-fitted (buildings time fold `d2_cal` -10.5 and -30.7). Pooling lifted every line. The GAM was left out of pooled runs because it exceeded the memory cap on 756k rows. Tree models needed level re-balancing on emerging rows, with factors of 1.08 to 1.26 at weight 0.3 on the property lines and 1.01 to 1.10 at weight 1.0, a known limitation of pooling.

| Line | Emerging only (time / grouped) | Pooled, weight 0.3 | Pooled, weight 1.0 | Adopted weight |
|---|---|---|---|---|
| Motor | 0.0762 / 0.0677 | 0.0915 / 0.0740 | 0.0908 / 0.0730 | 0.3 |
| Buildings | 0.0410 / 0.0732 | 0.1065 / 0.1132 | 0.1169 / 0.1180 | 1.0 |
| Contents | 0.0779 / 0.0934 | 0.1072 / 0.1100 | 0.1069 / 0.1128 | 0.3 |

Buildings at 1.0 beat 0.3 by 0.0104 on the time fold and 0.0048 on the grouped mean. Contents at 1.0 lost 0.0003 on the time fold, so 0.3 stayed. The C039 search of 23 tree configurations on pooled contents found none better than the defaults on both columns, so nothing was adopted. The developed market's half trend (C037) was adopted after the P2-S02 probe (+0.0009).

| ID | Recipe | Public score |
|---|---|---|
| P2-S01 | Emerging only, no trend | 0.14370 |
| P2-S02 | P2-S01 with developed half trend | 0.14460 |
| P2-S03 | Contents pooled | 0.14693 |
| P2-S04 | All lines pooled, weight 0.3 | 0.15193 |
| P2-S05 | P2-S04 with buildings at weight 1.0 | 0.15123 |

On P2-S05 the public board disagreed with validation. It covers about a third of the rows, so we kept the validation rule (C040).

### 7.5 Office premium

For the 20 reference policies the office premium is P = (R + Cat + F) / (1 - k). R is our predicted row risk premium, Cat is 0.12 R on buildings and contents and 0 on motor, and F is 85 CU. Here k = 0.15 + c + 0.08 + 0.08 is variable expense, commission c (0.20 broker, 0.00 direct), premium tax and profit, so the divisor is 0.49 for broker and 0.69 for direct. The build-up table is in section 8. The team is deciding four conventions: whether F scales with exposure, whether contents counts as property for reinsurance, where premium tax enters, and the blank comparison-site commission.

### 7.6 Limitations specific to Part 2

- 76% of test rows are new business against about 8% of training exposure. A segment check (`docs/part2_new_business_check.md`) gives new-business prediction-to-actual ratios of 0.81 to 1.01 for motor, 1.30 to 2.23 for buildings and 0.74 to 0.84 for contents, on 37 to 697 claims and with the two validation schemes disagreeing for motor. We judged this too noisy to support a segment adjustment and made none.
- The trend rests on one public-board probe of 0.0009, below our 0.002 threshold.
- Reporting curves are borrowed, and the caps rest on 36 (motor), 6 (buildings) and 6 (contents) claims above the cap.
- The sign-off sheet for Part 2 decisions (`docs/HUMAN_REVIEW.md`, R08 to R11) is pending; the work is logged in `docs/AI_LOG.md` A20 to A22.
