# Part 2 data intake note (2026-10-24)

This document gives a first pass over the downloaded Part 2 data: what is in it, whether it matches Part 1, how the new market differs, what dirty data there is, which parts of the existing pipeline it affects, and what is recommended next. All numbers come from results actually run on 2026-10-24, and the source is given at the end of each section. The official field descriptions in `docs/references/part2_kaggle_data_page.md` are authoritative.

## 1. What the data contains

Location: `data/iaa-ai-assisted-hackathon-2026-part-2/`, which holds two folders.

- `developed/`: the developed market, 11 files; officially described as the original Part 1 data, provided for reference and reuse.
- `emerging/`: the emerging market, 13 files. Two more files than `developed/`: `expenses.parquet` (expense assumptions) and `reference_policies.parquet` (20 sample policies, used for the "office premium" derivation in the report).

Size of each file in `emerging/`:

| File | Rows | Notes |
|---|---|---|
| policies_motor | 47,283 | Motor training, 2023–2024 |
| policies_buildings | 27,447 | Buildings training |
| policies_contents | 34,566 | Contents training |
| claims | 9,258 | Motor 7,183, buildings 1,059, contents 1,016, valuation date 2024-12-31 |
| test_motor / test_buildings / test_contents | 90,521 / 52,295 / 66,247 | 2025, the rows to be predicted, 209,063 in total |
| customers / addresses | 135,856 / 136,914 | Regions E01–E06 |
| weather | 410,742 | Three years per address: 2023, 2024, 2025 (the official table says 2023–2024, but the file contains 2025, the same approach as Part 1, because the test set needs 2025) |
| expenses | 8 | See section 6 |
| reference_policies | 20 | Taken from the test files, 7 motor, 7 buildings, 6 contents |
| sample_submission | 209,063 | Note it is in parquet format, while Kaggle expects a CSV (header `row_id,prediction`), submitting only the emerging-market rows |

Compared with Part 1, the training data is much smaller: motor has 47,000 rows against 980,000 in Part 1. The test rows are conversely about twice the training rows (motor 90,500 vs 47,300), because a large block of broker business comes in during 2025.

Backups made: `emerging/MANIFEST.sha256` (13 files) and `developed/MANIFEST.sha256` (11 files) have been written and the files set to read-only (the C007 approach). At present `tests/test_raw_data_integrity.py` covers only the Part 1 folder; Part 2 has no corresponding integrity test yet.

Source: `docs/part2_emerging_profile.txt`.

## 2. Whether developed matches Part 1

It matches, byte for byte. The SHA256 of each of the 11 files in `developed/` was compared against `data/iaa-ai-assisted-hackathon-2026-part-1/MANIFEST.sha256`, and all 11 are identical (addresses, claims, customers, policies_buildings, policies_contents, policies_motor, sample_submission, test_buildings, test_contents, test_motor, weather). The file sizes are also the same. So the Part 1 code, tests and conclusions can be applied to developed as they are, with no need to re-clean or re-model.

One pitfall to watch: IDs collide between the two markets. `customer_id` has 752 in common, `address_id` has 8,046 in common, the training-table `row_id` (starting at M00000000) has 47,283 that are exactly identical, and `policy_id` also has a few in common, but they are unrelated records. The official text also says tables must not be joined across markets. If the data of the two markets is ever combined for training, a market flag column must be added first, and the IDs cannot be used directly as keys. The test-set `row_id` has no duplicates between the two markets.

## 3. Main differences between emerging and developed

Basis of comparison: 2023–2024 combined, with developed also restricted to 2023–2024, and no development adjustment (the valuation date for both markets is 2024-12-31, so 2024 is equally undeveloped). The full table is in `docs/part2_market_comparison.md`. Differences above 25% are marked "Yes".

### 3.1 Business performance

| Item | Line | developed | emerging | Difference | Over 25% |
|---|---|---|---|---|---|
| Claim frequency (claims per risk year) | Motor | 13.48% | 18.09% | +34% | Yes |
| | Buildings | 4.46% | 4.50% | +1% | No |
| | Contents | 4.05% | 3.43% | −15% | No |
| Average claim amount (CU) | Motor | 3,862 | 2,250 | −42% | Yes |
| | Buildings | 9,874 | 9,052 | −8% | No |
| | Contents | 2,886 | 2,589 | −10% | No |
| Reported pure premium (CU per risk year) | Motor | 521 | 407 | −22% | No (close) |
| | Buildings | 441 | 407 | −8% | No |
| | Contents | 117 | 89 | −24% | No (close) |
| Median premium (CU) | Motor | 493 | 502 | +2% | No |
| | Buildings | 228 | 238 | +4% | No |
| | Contents | 134 | 119 | −11% | No |
| Risk years in the training period | Motor | 399,468 | 38,969 | about 1/10 | Yes |
| | Buildings | 299,905 | 23,468 | about 1/13 | Yes |
| | Contents | 299,942 | 29,058 | about 1/10 | Yes |
| Number of claims in the training period | Motor | 53,854 | 7,048 | | |
| | Buildings | 13,388 | 1,056 | | |
| | Contents | 12,146 | 998 | | |

Key points:

- Motor is the line where the new market differs most: claims are more frequent but each is smaller, the two roughly offset, the pure premium is 22% lower, yet the current premium is about the same as in the developed market (median 502 vs 493).
- emerging's own 2024 pure premium is lower than 2023: motor 433 → 383, buildings 514 → 307, contents 91 → 87. 2024 is not fully developed, and buildings has only just over 500 claims, so volatility is large and this cannot be taken directly as a trend.
- The reporting speed and open-claim proportion are close to the developed market: for 2024 losses the open proportion is motor 20.0% (developed 20.7%), buildings 25.3% (24.7%), contents 11.5% (13.0%); the median reporting delay is 9/8/6 days (developed 9/9/6 days), and the 99th percentile is motor 331 days (377), buildings 182 days (239), contents 72 days (63).

### 3.2 Customer and policy structure

| Item | developed | emerging | Notes |
|---|---|---|---|
| Channel (motor as an example) | broker 25%, comparison site 35%, direct 40% | broker 70%, comparison site 5%, direct 25% | Same for all three lines; difference over 25% |
| New business share in the test set | 15% | 76–77% | In the training set new business is only 8% (about 3,040 risk years for motor) |
| Share of test policies seen in the training set | 85% | 24% | |
| Age band (motor, share aged 17–24) | 13% | training 20%, test 25% | The test set is younger than the training set: aged 25–34, training 19% and test 25%; aged 65 and over, training 29% and test 24% |
| Tenant share (contents) | 30% | 43% | |
| Electric vehicle share | 8.1% | 2.0% | |
| Median vehicle age | 7 years | 9 years (test 10 years) | |
| Median years licensed | 28 years | 24 years (test 13 years) | |
| Missing rate | about 2.5% | about 12% | The same set of columns, e.g. vehicle_age, overnight_parking, soil_type, occupation |
| Weather (2024 average) | hail 6.8 days, frost 39.5 days, wildfire index 0.16, rainfall 1,036 mm | hail 17.1 days, frost 17.5 days, wildfire index 0.55, rainfall 1,375 mm | Storm share of buildings claims 44% (developed 30%) |
| Region | R01–R12 | E01–E06 | The codes are not interchangeable |
| Share by property type, urbanisation and construction type | | | Essentially the same in both markets; none differs by more than 25% |

After the broker business comes in, the channel and region shares of the test set are almost the same as in the training set (brokers are 70% in both), so these fields cannot tell the "newly arrived business" apart from the old business; the main differences are the new business share, age band and years licensed.

### 3.3 Field differences (official description)

| Table | In developed but not in emerging | In emerging but not in developed |
|---|---|---|
| customers | date_of_birth, credit_band | gender (about 50% each for male and female, no missing values) |
| policies_motor / test_motor | driver_age, annual_mileage, telematics_opt_in, telematics_score | driver_age_band (17-24, 25-34, 35-49, 50-64, 65+) |
| policies_buildings / contents and the test tables | policyholder_age | policyholder_age_band |
| construction_type | brick, stone, timber, non_standard (inconsistent case) | BRK, STN, WOD, MIX (all upper case, clean) |
| occupation | 20 canonical spellings | Free text, 80 raw spellings |

Source: `docs/part2_market_comparison.md`, `docs/part2_emerging_quality.txt`.

## 4. Data quality issues found (old vs new comparison table)

The Part 1 IDs Q1–Q10 are in section 7 of `docs/DATA_GUIDE.md`. The full output is in `docs/part2_emerging_quality.txt`.

### 4.1 Whether the Part 1 issues are still present in emerging

| ID | Part 1 issue | Status in emerging | Count |
|---|---|---|---|
| Q1 | exposure errors (365, 1.0139, etc.) | Still present, only in the training tables; test-table exposure is all correct (0 rows > 1). There are three kinds of error: 1.0139 (equal to 365/360, a full year counted on 360 days), days entered directly (e.g. 365), and a partial year counted on 360 days (e.g. 302 days entered as 0.8389 = 302/360) | Motor 253 rows (0.53%: 206 rows on the 360-day basis, including 141 rows of 1.0139, and 47 rows with days entered directly); buildings 153 rows; contents 198 rows. Recomputing with (end date − start date + 1)/365 is enough; the rule is unchanged |
| Q2 | Leading zero lost from motor customer_id | Does not occur. customer_id is 8 digits in every table and all join to customers (0 rows fail to join); the 13,490 customer IDs starting with 0 are fully preserved | 0 |
| Q3 | sum_insured inflated about 1000 times | Still present, and the rule works. Largest normal values: buildings 813,000 (training) / 894,000 (test), contents 235,000 / 283,200; smallest abnormal values: buildings 60,000,000, contents 7,600,000; there is a clear gap between the two, so the 5 million threshold still separates them | Buildings: 53 training rows, 104 test rows; contents: 57 training rows, 135 test rows (about 0.2%) |
| Q4 | construction_type inconsistent case, W/T codes | The spelling has changed: only four values BRK, STN, WOD, MIX, all upper case, no case problem; but the existing `fix_construction_type` does not recognise these codes and will raise an error | All 27,447 training rows and 52,295 test rows need mapping |
| Q5 | Claims that match no policy row | Still present but very few: all are claims whose loss date is before the policy's first row (34 claims with losses in 2021–2022, 2 with losses in 2023; the market only started writing business in 2023), and none has a loss date after the policy's last row | 36 claims (motor 28, buildings 3, contents 5), 92,268 CU in total, 0.3% of total claim amount; no claim matches more than one row |
| Q6 | customer_service_contacts maximum 36 in training, 6 in test | The same kind of phenomenon remains: training maximum 20, test maximum 5; the column was already removed in C008, so there is no impact | Training rows > 10: motor 61, buildings 2, contents 1 |
| Q7 | About 2.5% random missing | The rate is higher: about 12% (campaign_code 20%, vs 51% in Part 1). flood_zone is still about 3% | E.g. occupation 12.16%, vehicle_age 11.85%, soil_type 12.04%, distance_to_river_km 11.99% (2.5% in Part 1) |
| Q8 | telematics_score 87% blank | Not applicable: telematics_opt_in, telematics_score and annual_mileage do not exist in emerging at all | |
| Q9 | renewal_decision only in the training set (leakage) | Same: present in the training tables, absent from the test tables, and still cannot be used as a feature | All three training tables have it |
| Q10 | Difference between training and test distributions | Much larger, see below | See section 3.2 |

Details of Q10: new business is 76–77% of the test set but only 8% of the training set; renewal_count has median 7 and maximum 16 in the training set, and median 0 in the test set; and in this data renewal_count = 0 is exactly equivalent to new_business = yes (0 exceptions).

### 4.2 Newly found issues in emerging

| ID | Issue | Count | Notes |
|---|---|---|---|
| N1 | occupation is free text | 80 raw spellings, 119,339 non-missing values | Leading and trailing spaces (e.g. `Retired ` with a trailing space) and mixed case: after stripping spaces and lower-casing, 40 remain; of these 40, half are the 20 base occupations and the other half are the same occupations with the prefix `Self-employed ` added (11,975 records in total, 10%). Some combinations make no sense (`Self-employed retired`, `Self-employed student`, `Self-employed unemployed`). Part 1 has only 20 canonical spellings |
| N2 | In expenses the comparison-site commission for emerging is blank, but the data has a comparison-site channel | Comparison-site channel is 4.9–5.2% of training rows and 5.0% of test rows | The official text says blank means "not applicable", which contradicts the data. An assumption of our own is needed (see section 6) |
| N3 | renewal_count and customer_since_year contradict "the company has written business for only two years" | renewal_count maximum 16; for 20.2% of customers customer_since_year is before 2023 (earliest 2008) | These two fields cannot be read as "years of renewal with this company"; they can only be treated as years of history brought in by the broker business. An artefact of the synthetic data; take care when using them in modelling |
| N4 | Some claims have loss dates before the market opened | See the 36 claims in Q5 | Same batch as Q5 |
| N5 | Age is given only as an age band, no date of birth | All | The existing `customer_age` cannot be computed (see section 5) |
| N6 | ncb_years is greater than years_licensed | Training motor 7,756 rows (16.4%), test 20,051 rows | Also present in the developed market (about 13%), so not unique to emerging, which shows such inconsistencies come from how the data was generated; no separate handling for emerging is needed |

Items also checked where no problem was found: row_id is all unique; no fully duplicated rows; rows of the same policy do not overlap in time; no rows cross a year boundary; claim report dates are not earlier than the loss date and not later than the valuation date; closed claims have incurred = paid (0 exceptions); incurred − recoveries has no negatives; repudiated claims (120) have both paid and incurred equal to 0; age band, gender and region have no case or whitespace variants (only occupation does); all identifier formats are correct; the row_id values in the sample submission file match the three test tables exactly.

## 5. Impact on the existing pipeline

Component by component, marked "needs change", "may need change" or "no change". This is an assessment only; I have not touched any code.

| Component | Impact | Verdict |
|---|---|---|
| `fix_exposure` in `src/aiiaa/clean.py` | The training tables still contain wrong exposure; the recomputation rule (days/365) is still correct | No change |
| `fix_customer_id` | The emerging IDs are already 8 digits, so padding changes nothing | No change (harmless) |
| `fix_sum_insured` | The 5 million threshold still separates the values (see Q3) | No change |
| `fix_construction_type` | Raises ValueError on BRK, STN, WOD, MIX | Needs change: add an emerging mapping table BRK→brick, STN→stone, WOD→timber, MIX→non_standard (official description: "MIX = mixed or non-standard"). Recommend handling by market separately, and not mixing the old W/T rules and the new codes in one table |
| occupation cleaning | The existing code has no text tidying at all. Motor features still use occupation (buildings and contents were dropped in C026), so the 80 raw spellings would be treated as 80 categories | Needs change: strip spaces and lower-case; whether to split out a "self-employed" flag or merge down to 20 levels is for you to decide |
| Age fields in `src/aiiaa/features.py` | `customer_age` is computed from date_of_birth, which emerging does not have, so it will raise an error; driver_age and policyholder_age (numeric) are replaced by age bands (categorical); credit_band, annual_mileage and telematics_* do not exist in emerging | Needs change: a "common age representation for both markets" is needed. One way is to cut the developed numeric ages into the same 5 age bands (`docs/part2_market_comparison.md` compares them this way); gender exists only in emerging |
| Region etc. in features | The region codes R01–R12 and E01–E06 are not interchangeable; for the meaning of `customer_since_year` and `renewal_count` see N3 | May need change |
| Reporting curve in `src/aiiaa/development.py` | The current curve is estimated from claims with 2021 losses in the developed market (observed for more than 3 years by then). emerging has only two loss years, 2023 and 2024: 2023 is observed for 12–24 months and 2024 for less than 12 months; there are only 36 claims for 2021–2022 | To be decided: emerging data cannot estimate the long tail; either keep the developed curve (emerging's reporting delay and open proportion are close to developed, see section 3.1), or run a check with emerging's 2023-loss data before deciding |
| Trend estimation (`evaluate.trend_factor`, C009) | Currently extrapolates from the four years 2021–2024 to 2025. emerging has only two points, and 2024 is not fully developed (the −40% change in reported pure premium for buildings from 2023 to 2024 is mainly due to under-development and too few claims); a two-point straight-line extrapolation to 2025 is unreliable | Needs change: the trend cannot be estimated from the two emerging points alone; a decision is needed on borrowing the developed trend, setting it to 0, or something else |
| Large-claim capping (C014, 99.5th percentile) | Computed on emerging alone: motor cap 27,578 (only 36 claims above), buildings 220,808 (6 claims), contents 28,312 (6 claims). The developed market has 51,585, 147,445 and 28,449 respectively. The buildings cap is determined by only 6 claims and is unstable | To be decided: use emerging's 99.5th percentile, or borrow the shape of the developed claim-amount distribution. The large-loss loading factor (total / capped total) is also estimated from only this small sample |
| Validation scheme and weight search (2-fold time validation, adopt only if no worse than equal weights on both folds, 0.002 threshold) | The current folds are "train 2021–22 → validate 2023" and "train 2021–23 → validate 2024". emerging can only be cut into one fold (train 2023 → validate 2024), and 2024 is not fully developed. The rule "no worse than equal weights on both folds" cannot be applied | Needs change: the validation scheme and the adoption rule must be redefined, and fixed before looking at results (to avoid overfitting, see the convention recorded in memory) |
| Tuned hyperparameters (`params.py`, C017/C032) | All tuned at the scale of millions of rows; emerging motor has only 47,000 rows and buildings 27,000, so settings such as the minimum samples per leaf may be too large | May need change |
| Target variable (`target.py`) | The logic is unchanged and applies equally to emerging (deduct the excess, deduct recoveries, remove repudiated claims) | No change |
| Prediction of test rows | Test-row exposure is correct and needs no recomputation; 76% is new business, and the training data has only about 3,040 risk years of new business (motor) | No code change, but the model's estimate for new business is very unstable, which is the main risk |
| `expenses.parquet` | Does not affect the Kaggle score; used for the office premium derivation in the report | See section 6 |
| `reference_policies.parquet` | The row_id of all 20 policies are in the test files, and the premiums match the values in the test files (20/20) | See section 6 |

## 6. Purpose of expenses and reference_policies

Neither file affects the Kaggle score. The official description says they are used in the report: for the 20 sample policies, show how to go from your predicted risk premium, using the expense assumptions, to the office premium. It is reviewed by the judges and goes into the report.

`expenses.parquet` in full:

| Item | developed | emerging | Unit |
|---|---|---|---|
| fixed_expense_per_policy (fixed expense per policy) | 45.00 | 85.00 | CU |
| variable_expense (variable expense) | 0.12 | 0.15 | Proportion of premium |
| commission_direct (direct commission) | 0.00 | 0.00 | Proportion of premium |
| commission_comparison_site (comparison-site commission) | 0.10 | blank | Proportion of premium |
| commission_broker (broker commission) | 0.15 | 0.20 | Proportion of premium |
| premium_tax (premium tax) | 0.12 | 0.08 | Proportion of premium |
| cat_reinsurance (catastrophe reinsurance) | 0.06 | 0.12 | Proportion of property risk premium |
| target_profit_margin (target profit margin) | 0.05 | 0.08 | Proportion of premium |

Several things must be decided before reading this table (I have not decided them for you):

- The comparison-site commission is blank in emerging, but the emerging data has about 5% comparison-site channel (N2). None of the 20 sample policies is on the comparison-site channel (17 broker, 3 direct), so the samples themselves are unaffected, but the report needs to state how this item is handled.
- The unit of catastrophe reinsurance is "proportion of property risk premium", so it should be added only to buildings and contents, not to motor.
- The fixed expense is per policy, not per row. The sample policies include rows with exposure of only 0.07 and 0.11; a definition is needed for whether to scale by exposure.
- Whether the ratios are added together and divided into the premium, or each multiplied by the risk premium separately, and whether premium tax is also levied on the other items, the official text does not say; the report needs to state clearly the definition you choose.

`reference_policies.parquet`: 20 rows, with one more column than the test files, `line` (line of business), plus the union of the line-specific columns (columns not belonging to the line are blank). Of these, 18 are new business, 17 are on the broker channel and 3 are direct.

## 7. Suggested next steps

I only list them and do not prioritise for you; the points in brackets are where you need to make the call.

- [ ] Decide whether to commit `emerging/MANIFEST.sha256`, `developed/MANIFEST.sha256` and the Part 2 data together, and add Part 2 checks to `tests/test_raw_data_integrity.py` (these files have not been committed yet).
- [ ] Change the pipeline to be "parameterised by market" (data path, cleaning rules, field list), add the emerging mapping to `fix_construction_type` and tidying rules for occupation; write the tests before the change (TDD).
- [ ] Decide the modelling approach (your call): use only emerging data, apply the developed model directly and calibrate the overall level, or combine the two markets with a market flag. Note the training data is only 47,000 / 27,000 / 35,000 rows.
- [ ] Unify the age representation: cut the developed numeric ages into 5 age bands, and drop the fields emerging does not have (date_of_birth, credit_band, annual_mileage, telematics_*).
- [ ] Use the emerging 2023-loss data to test whether the developed reporting curve applies, then decide whether to keep or re-estimate it.
- [ ] Decide the trend treatment (your call): borrow developed, set to 0, or another way.
- [ ] Decide the large-claim capping (your call): buildings has only 6 claims above the cap.
- [ ] Redesign the validation scheme and the "adoption threshold" and fix them before looking at results (your call).
- [ ] Assess the impact of new business: 76% of the test set is new business, while the training set has only about 3,040 risk years of new business (motor); first see how stable the frequency difference between new and old business is.
- [ ] Write the office premium derivation for the 20 sample policies: first fix the definition of the comparison-site commission, the scope of catastrophe reinsurance and how the fixed expense is scaled (section 6).
- [ ] When submitting, note: the Week 2 leaderboard takes only predictions for the emerging-market test rows, in CSV format `row_id,prediction`, with predictions ≥ 0; sample_submission is parquet in the folder and must be converted to CSV. The Part 2 deadline follows the countdown on the Kaggle page (when the page was captured it showed 7 days remaining).

---

## Files used by this document

- `docs/part2_emerging_profile.txt`: row count, columns, types, missing rate, category counts and numeric distribution of each emerging file; full text of expenses and reference_policies
- `docs/part2_emerging_quality.txt`: full output of the data quality checks
- `docs/part2_market_comparison.md`: comparison tables of the two markets (in English)
- Scripts that generate these files: `scripts/part2_profile.py`, `scripts/part2_quality.py`, `scripts/part2_compare.py` (all take the folder path as an argument)
- Integrity manifests: `data/iaa-ai-assisted-hackathon-2026-part-2/emerging/MANIFEST.sha256`, `data/iaa-ai-assisted-hackathon-2026-part-2/developed/MANIFEST.sha256`
