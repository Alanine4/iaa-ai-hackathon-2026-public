# Data Guide (Part 1, Developed Market)

Sources: `docs/references/kaggle_data_page.md` (official field descriptions), `docs/data_profile.txt` (output of `scripts/profile_data.py`), `docs/data_quality_checks.txt` (output of `scripts/check_quality.py`). All figures come from the actual run on 2026-10-21.

Data location: `data/iaa-ai-assisted-hackathon-2026-part-1/`, 11 parquet files, about 198 MB in total.

## 1. How to open them

Parquet is a column-compressed tabular file format that Excel cannot open; use Python:

```python
import pandas as pd
d = r"C:\Projects\AIIAA\data\iaa-ai-assisted-hackathon-2026-part-1"
m = pd.read_parquet(d + r"\policies_motor.parquet")
m.shape; m.head().T; m.info(); m["cover_type"].value_counts(); m.describe().T
```

Python interpreter: `C:\Projects\AIIAA\.venv\Scripts\python.exe` (pandas, pyarrow, lightgbm, scikit-learn, statsmodels and matplotlib are installed). To look at the data in Excel, export a small sample: `m.head(1000).to_csv("sample.csv")`.

## 2. What one row represents

A row in a policy table is "one coverage period of one policy within one calendar year". `exposure` is the length of that period, in years. If a policy changes during the year (new vehicle, house move), it is split into several rows; if it lapses or is not renewed mid-year, the row ends early. `policy_id` does not change across years.

## 3. Overview of the files

| File | Rows | Columns | One row is | Purpose |
|---|---|---|---|---|
| policies_motor | 979,968 | 39 | policy period | Motor training, 2021–2024 |
| policies_buildings | 707,726 | 27 | policy period | Buildings training |
| policies_contents | 721,617 | 28 | policy period | Contents training |
| claims | 163,150 | 12 | one claim | All claims reported during the training period, evaluation date 2024-12-31 |
| test_motor | 254,986 | 38 | policy period | 2025 motor, to be predicted |
| test_buildings | 184,155 | 26 | policy period | 2025 buildings, to be predicted |
| test_contents | 187,498 | 27 | policy period | 2025 contents, to be predicted |
| customers | 545,240 | 18 | customer | Customer attributes, shared across lines |
| addresses | 575,983 | 16 | address | Location and property features |
| weather | 2,879,915 | 9 | address × year | Annual weather per address, including 2025 |
| sample_submission | 626,639 | 2 | test row | Submission format, prediction is all 0 |

The test sets have one column fewer than the training sets, `renewal_decision`. The row_id in sample_submission is identical to the row_id of the three test tables (checked). First letter of row_id: M motor, B buildings, C contents.

## 4. How the tables join

```
customers ──customer_id──┐
                         ├── policies_* / test_* ──policy_id + loss_date──> claims
addresses ──address_id───┤
weather ──address_id + year┘
```

- Claim to policy row: same `policy_id`, and `period_start ≤ loss_date ≤ period_end`. Measured: each claim matches at most 1 row, with no one-to-many matches.
- Weather joins on `address_id` + policy start year. The test set uses 2025 weather.
- One customer can hold motor, buildings and contents policies at once; several customers may share one address.
- About 85% of the policy_id values in the test set appear in the training set (84.8–84.9% for all three lines), so historical claim experience can be used.

## 5. Field groups (detailed definitions in references/kaggle_data_page.md)

**Common to all three lines**: row_id, policy_id, customer_id, address_id, period_start, period_end, exposure, voluntary_excess (voluntary excess chosen by the customer, 0/250/500/1000), compulsory_excess (compulsory excess, fixed at 250 for motor and 350 for buildings/contents), channel, payment_method, missed_payments_12m, new_business, renewal_count, days_quote_to_inception, discount_applied (0/0.05/0.10), written_premium (premium charged under current pricing), customer_service_contacts, renewal_decision (training set only).

**Motor only**: driver_age, years_licensed, ncb_years, vehicle_value_group(1–20), vehicle_age, make_model(300 values), engine_size_cc, body_type, transmission, seats, vehicle_colour, fuel_type, driver_assist, cover_type, annual_mileage, overnight_parking, vehicle_use, named_young_driver, telematics_opt_in, telematics_score.

**Buildings only**: policyholder_age, sum_insured, construction_type, occupancy, days_unoccupied, smoke_alarms, accidental_damage_cover, mortgage.

**Contents only**: policyholder_age, sum_insured, tenure, security_level, high_value_items, accidental_damage_cover, works_from_home, lodgers, bicycle_cover.

**customers**: date of birth, occupation (20 classes), marital status, household size, credit band A–E, claims in the 5 years before joining, licence penalty points, years at address, year first took out a policy, email domain, marketing consent, document preference, previous insurer (40 insurers), campaign code (51% empty, meaning no campaign), branch (25 branches), marketing segment (12 classes).

**addresses**: region R01–R12, urban density, flood zone 1–5, distance to river, elevation above river level, soil type, year built, property type, roof type, basement, trees within 5 m, listed building, bedrooms, energy rating, solar panels.

**weather**: hail days, wind exposure index, freezing days, dry summer index, wildfire index, annual rainfall, annual mean temperature.

**claims**: line, peril (cause), loss_date, report_date, status (closed/open/repudiated), paid_to_date (paid, net of excess), incurred_to_date (paid + outstanding estimate), recoveries (recoveries / salvage), repairer_id, third_party_id.

Claim causes by line:
- Motor: own_damage 48,964, third_party_prop 33,826, glass_hail 15,732, bodily_injury 7,583, theft 6,501
- Buildings: escape_of_water 10,137, storm 7,816, accidental_damage 3,848, flood 2,232, subsidence 1,294, fire 1,115
- Contents: accidental_damage 9,208, theft 7,863, escape_of_water 5,407, flood 1,624

## 6. How the prediction target is derived from the data

Official definition: the **ultimate cost** of all claims arising in the period of this row (loss_date falls within the period), net of excess, net of recoveries, excluding repudiated claims. This is the risk premium, not the premium charged.

In terms of the data:

```
Net amount per claim = incurred_to_date − recoveries      (status ≠ repudiated)
Target per row       = sum of the net amounts of all claims matched to this row, then development-adjusted to ultimate
```

Verified facts:
- For closed claims, incurred == paid, without exception.
- There are 1,711 repudiated claims in total; paid and incurred are both 0, so removing them or not does not change the amounts, but the treatment must be stated in the documentation.
- incurred − recoveries is never negative.
- The excess has already been deducted in paid (the official text states that paid is net of excess).

Why a development adjustment is needed (the actuarial key point):
- The evaluation date is 2024-12-31. Of the claims with a 2024 loss date, the share not yet closed is 20.7% for motor, 24.7% for buildings and 13.0% for contents, against almost 0 for 2021.
- Reporting is delayed: the median motor reporting delay is 9 days, the 99th percentile 377 days, the maximum 1361 days. Some accidents occurring at the end of 2024 have not been reported yet (IBNR).
- Motor net claim totals: 106 million in 2021, 117 million in 2022, 117 million in 2023, 91 million in 2024. 2024 is clearly lower, mainly because development is incomplete, not because risk has really fallen.

## 7. Data quality issues (found; to be handled before modelling)

| # | Issue | Where | Size | Initial treatment idea |
|---|---|---|---|---|
| Q1 | exposure is wrong: values such as 365 and 289 appear, suspected to be entered in days; there are also many values of 1.0139 | three training tables | motor 3,991 rows, buildings 3,044 rows, contents 3,088 rows with exposure > 1 | Recompute as (period_end − period_start + 1)/365 and check the differences against the original values |
| Q2 | Motor customer_id has lost its leading zeros (e.g. `5180324`) | policies_motor, test_motor | 98,110 training rows, 25,554 test rows | After padding to 8 characters (zfill(8)), 100% match customers |
| Q3 | sum_insured has abnormally large values (e.g. 220 million), suspected to be inflated by about 1000 times | buildings, contents (both training and test) | buildings 1,395 rows, contents 1,402 rows > 5 million | Decision rule to be determined; compare against written_premium of similar policies |
| Q4 | construction_type has inconsistent capitalisation: Brick/brick/BRICK, stone/STONE, non_standard/NON_STANDARD, plus codes W/w, T/t | buildings | 11 spellings, whereas the official list has only 4 values: brick/stone/timber/non-standard | Standardise to lower case. What W and T stand for is not yet confirmed: the test set has W but no T |
| Q5 | Claims with a 2020 loss date match no policy row | claims | 274 claims (motor 198, buildings 46, contents 30), plus about 140 claims from 2021–2023 that also fail to match | Excluded from the target; recorded in the documentation |
| Q6 | customer_service_contacts has a maximum of 36 in the training set and 6 in the test set | training tables | motor 731 rows > 10 | Truncate (capping), or check whether it is related to claims (post-event information) |
| Q7 | About 2.5% random missingness: flood_zone, soil_type, roof_type, energy_rating, vehicle_age, engine_size, annual_mileage, overnight_parking, smoke_alarms, security_level, high_value_items, occupation, marital_status, etc. | multiple tables | about 2.5% per column | Add a "missing" category, or leave it to tree models |
| Q8 | telematics_score is about 87% empty | motor | Customers without a device all have null values | Structural missingness (a value exists only if the device is activated); do not impute |
| Q9 | renewal_decision exists only in the training set and is only known after this row has ended | training tables | all | Must not be used as a feature (data leakage) |
| Q10 | Training and test distributions differ: renewal_count median 6 → 3, vehicle_age median 7 → 9, ncb_years median 7 → 8 | training vs test | overall | State it in the validation design and the report |

## 8. Scoring definition

Tweedie D², power = 1.5, measured against a constant-prediction baseline, higher is better. All predictions must be ≥ 0. At submission the three test tables are combined into one file, with header `row_id,prediction`.
