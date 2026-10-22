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

Validation uses two expanding-window folds (train 2021-22, score 2023; train 2021-23, score 2024) on reported cost against prediction times completeness. `d2_cal` rescales each fold to its actual total, so it measures ranking, not level. The feature policy (C026) removed columns without frequency signal and raised mean LightGBM `d2_cal` from 0.0832 to 0.0887 on motor, 0.1094 to 0.1134 on buildings and 0.0891 to 0.0936 on contents. Final predictions blend the five models with per-line weights found on out-of-fold predictions (non-negative, 0.05 grid, sum 1). Searched weights are used only if they beat equal weights on both folds (C021, C030); motor falls back to equal weights (0.09031 against 0.09071 on 2024). The tree refit averages three seeds (C029). The final blends score a mean two-fold `d2_cal` of 0.0919 for motor, 0.1144 for buildings and 0.0945 for contents.

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

We worked with Claude Code throughout (`docs/AI_LOG.md`, A01 to A17). The assistant produced a first version at each stage, and a person or an automated check decided whether to keep it.

- Exploration and data quality: profiling of all 11 files, and flagging Q1 to Q10 with counts. Cleaning rules such as T/W = timber were inferred from the year split and policy transitions, then written as failing tests before implementation (A02 to A04).
- Target and leakage: claim matching and development were designed with AI and checked by full-data reconciliation (difference 0). A scan of every column for leakage found only `customer_service_contacts` (A05, A06).
- Modelling and debugging: models, the D² scorer and the trend factor were written test-first. A crash on a boolean column (C012) and a 9% to 18% under-prediction of tree models (C013) were each diagnosed by experiment before any fix (A07, A08).
- Independent review: a separate read-only AI audit produced findings B1 to B8, each backed by script output (A13).

Verification relied on tests written before the code (122 pass), ablations on time-based folds, and the audit. These cases show AI output being corrected or rejected:

- **C022 caught as a no-op.** The per-peril reporting curves were designed to correct the mix of slow and fast perils. The implementing assistant showed numerically that a cost-share mix of per-peril curves equals the pooled curve (maximum difference 5e-16), so row completeness could not change. The code stays as an input, with no accuracy claim (A09).
- **C023 rejected on evidence.** A per-peril frequency by severity model scored a mean `d2_cal` of 0.0801 on motor against 0.0893 for the best single model, 0.1045 against 0.1094 on buildings and 0.0879 against 0.0911 on contents. The main error was the peril mix: 2024 bodily injury counts were over-predicted by 1.43. It stayed out of the blend (A10, `docs/cv_freqsev.md`).
- **C024 features rejected.** Cross-line and household features ranked 60th or lower of 66 by gain, and re-adding them lowered motor by 0.001 to 0.002. The loss in the full set came from `branch_id` coded as a category (A11, C026).
- **Audit B1 and B2 fixed.** Motor trees overfit the 300-level `make_model`: without it, LightGBM on the 2023 fold went from 0.08082 to 0.08868. `branch_id` lowered every model on the 2023 fold (buildings LightGBM 0.1071 to 0.1051). Both led to the C026 drop policy (A13, A14).
- **Audit B4 probed, not adopted.** The audit argued that the trend counts the year level twice. A probe with a lower trend on buildings and contents (S08) scored 0.12820 against 0.12851, so half trend was kept.
- **Audit B8 fixed.** Tree models stopped early on the fold being scored while GLM and GAM did not. Early stopping now uses a random 10% of the fitting rows (C027).

## 5. Guardrails and governance

- **Change control.** Every change to data handling, code or assumptions has a row in `docs/CHANGELOG.md` (C000 to C030) with reason, evidence and code location. Code comments cite the same ID.
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
- Tuning and blend decisions made before the early-stopping fix (C018, C021) were not fully re-run (C027). The GAM uses a single penalty (C011).
- The public score covers about 33% of the test rows. The final ranking uses the other 67% and may move.

## 7. Adaptation to the new market (Part 2)
