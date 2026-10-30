# Data walkthrough: every file, every field, and what was done to it

Sources:
- Official field descriptions: `docs/references/kaggle_data_page.md` (archived original text of the Kaggle Data page)
- Profiling and quality-check results: `docs/data_profile.txt`, `docs/data_quality_checks.txt`, `docs/investigate_*.txt`, `docs/scan_leakage.txt`
- Processing logic: `src/aiiaa/clean.py`, `target.py`, `development.py`, `features.py`
- The number of each change (C0xx) refers to `docs/CHANGELOG.md`

The **"Source"** column in the tables:
- **Official**: the Kaggle Data page states the meaning
- **Official+name**: the Data page states the meaning, but the unit or direction is read from the column name (e.g. `_km`, `_mm`)
- **Inferred**: the Data page does not state it; I inferred it from the data, and the basis is given in "Handling / notes"

The **"Type"** column in the tables (used later in modelling):

| Type | Meaning | Examples |
|---|---|---|
| ID | Used to join tables; not itself a model input | customer_id |
| Date | A date | period_start |
| Continuous | A number that can take many values; its size is meaningful | age, sum insured, rainfall |
| Count | An integer count such as 0, 1, 2... | number of prior claims |
| Ordinal | A few categories with a high-to-low order | credit band A–E, flood zone 1–5 |
| Nominal | A few categories with no order | body type, occupation |
| Binary | Yes / no | whether there is a basement |

---

## 0. How the data was read

1. All 11 files are in parquet format (a column-compressed table format) and are read with pandas in Python: `pd.read_parquet(path)`.
2. The original files are in `data/iaa-ai-assisted-hackathon-2026-part-1/`, **set to read-only and never modified**. The SHA256 fingerprint of each file is recorded in `MANIFEST.sha256`, and a test checks it automatically (C007).
3. All corrections are made in code, after the data is read into memory (`src/aiiaa/clean.py`). So it is always possible to compare "what the original was" with "what it was changed to".
4. The official guidance requires ID fields to be read as text, not as numbers (otherwise the leading 0 is lost).

## 1. File overview

| # | File | Rows | Columns | One row represents | Purpose |
|---|---|---|---|---|---|
| 1 | customers | 545,240 | 18 | One customer | Customer attributes, shared by all three lines |
| 2 | addresses | 575,983 | 16 | One address | Geographic and property characteristics |
| 3 | weather | 2,879,915 | 9 | One year for one address | Weather, one row per year for 2021–2025 |
| 4 | policies_motor | 979,968 | 39 | One coverage period of one motor policy | Motor training data, 2021–2024 |
| 5 | policies_buildings | 707,726 | 27 | One coverage period of one buildings policy | Buildings training data |
| 6 | policies_contents | 721,617 | 28 | One coverage period of one contents policy | Contents training data |
| 7 | claims | 163,150 | 12 | One claim | All claims in the training period, as at 2024-12-31 |
| 8 | test_motor | 254,986 | 38 | Same as 4 | 2025 motor, to be predicted |
| 9 | test_buildings | 184,155 | 26 | Same as 5 | 2025 buildings, to be predicted |
| 10 | test_contents | 187,498 | 27 | Same as 6 | 2025 contents, to be predicted |
| 11 | sample_submission | 626,639 | 2 | One test row | Submission format template; all predictions are 0 |

**Terms**
- **Buildings**: covers the building itself, e.g. walls, roof, pipes.
- **Contents**: covers what is inside the building, e.g. furniture, appliances, jewellery.
- **Coverage period**: a stretch of time for one policy within one calendar year. A mid-year change of car or home splits it into several rows, and a mid-term cancellation ends it early. Official wording: "A policy has more than one row in a year if it changed mid-term".

**How the tables join** (stated officially)

```
customers ──customer_id──┐
addresses ──address_id───┼──> policy tables / test tables ──policy_id + loss date──> claims
weather ──address_id+year┘
```

---

## 2. Each file, each field

### 2.1 customers (customer table)

Example (the real first row): a customer born in 1938, retired and married, who first took out a policy in 2011, credit band B, with 3 licence points.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| customer_id | 88505748 | Customer ID | ID | Official | 8-digit text; the motor tables lost the leading 0, which is restored (C002) |
| surname | Surname569 | Surname (anonymised) | Nominal | Official | Only 600 pseudonyms, no meaning, **not used in the model** |
| date_of_birth | 1938-01-05 | Date of birth | Date | Official | Converted to the age at the policy start date, `customer_age` |
| occupation | Retired | Occupation | Nominal (20 levels) | Official | 2.5% missing |
| marital_status | married | Marital status: single / married / divorced / widowed | Nominal | Official | 2.5% missing |
| household_size | 4.0 | Number of people in the household | Count (1–7) | Official | 2.5% missing |
| credit_band | B | Credit band, A best, E worst | Ordinal | Official | |
| prior_claims_5y | 0 | Number of claims in the 5 years before joining (declared by the customer) | Count (0–7) | Official | |
| licence_points | 3 | Driving licence penalty points | Count (0/3/6/9) | Official | |
| years_at_address | NaN | Years living at the current address | Continuous (0–57) | Official | 2.5% missing |
| customer_since_year | 2011 | Year of the first policy with this company | Year | Official | Converted to "years with the company", `customer_tenure` |
| email_domain | posthub | Email provider | Nominal (5 levels) | Official | All are fictitious names with almost equal counts across the five levels, so they probably carry no risk meaning; left to the model to judge |
| marketing_consent | False | Whether the customer consents to receive marketing | Binary | Official | |
| document_preference | app | How documents are received: email / post / App | Nominal | Official | |
| previous_insurer | INS30 | Previous insurance company | Nominal (40 levels) | Official | Which company each code stands for is unknown |
| campaign_code | NaN | Marketing campaign joined at inception | Nominal (15 levels) | Official | 51% blank. The official text says "if any", so **blank = did not take part in a campaign**, not missing |
| branch_id | 2 | Service branch | Nominal (25 branches) | Official | It is a code, not a magnitude, but the code currently treats it as a number (see section 5) |
| marketing_segment | senior_affluent | The marketing department's customer segment, e.g. "senior + affluent" | Nominal (12 levels) | Official | A combination of "age band × income tier" |

### 2.2 addresses (address table)

Example: a terraced house built in 2014, in the suburbs of region R05, flood zone 1 (lowest), 4.1 km from the river, clay soil, energy rating A.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| address_id | A1980663 | Address ID | ID | Official | |
| region | R05 | Region, R01–R12 | Nominal | Official | |
| urban_density | suburban | Degree of urbanisation: rural / suburban / urban / dense urban | Ordinal | Official | |
| flood_zone | 1.0 | Flood risk zone, 1 lowest, 5 highest | Ordinal | Official | 2.9% missing |
| distance_to_river_km | 4.1 | Distance to the nearest river (km) | Continuous | Official+name | Unit read from the column name; 2.5% missing |
| elevation_m | 9.2 | Metres above the local river level | Continuous | Official | |
| soil_type | clay | Soil: clay / sand / loam / chalk / rock | Nominal | Official | 2.5% missing. Clay is linked to subsidence risk |
| year_built | 2014 | Year built | Year | Official | Converted to property age, `property_age` |
| property_type | terraced | Property type: detached / semi-detached / terraced / flat | Nominal | Official | |
| roof_type | pitched | Roof: pitched / flat | Binary | Official | 2.5% missing |
| basement | False | Whether there is a basement | Binary | Official | |
| trees_within_5m | False | Whether there are trees within 5 m | Binary | Official | Linked to subsidence and storm risk |
| listed_building | False | Whether it is a protected historic (listed) building | Binary | Official | Repair costs are usually higher |
| bedrooms | 3 | Number of bedrooms | Count (1–7) | Official | |
| energy_rating | A | Energy efficiency rating, A best, G worst | Ordinal | Official | 2.5% missing |
| solar_panels | False | Whether there are solar panels | Binary | Official | |

### 2.3 weather (weather table)

Example: address A1980663 had 10 hail days and 31 freezing days in 2021, annual rainfall of 1254 mm and a mean annual temperature of 13.9 degrees Celsius.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| address_id | A1980663 | Address ID | ID | Official | |
| year | 2021 | Year | Year | Official | Joined on the **policy start year**; the test set uses 2025 |
| hail_days | 10.0 | Number of days with hail in the year | Count | Official | |
| wind_exposure_index | 4.6 | Damaging-wind exposure index, 0–10 | Continuous | Official | |
| freezing_days | 31.0 | Number of days with frost in the year | Count | Official | Linked to burst pipes |
| dry_summer_index | 0.614 | Summer dryness, 0 wet, 1 very dry | Continuous | Official | Drought is linked to subsidence |
| wildfire_index | 0.248 | Wildfire risk, 0–1 | Continuous | Official | |
| rainfall_mm | 1254.0 | Annual rainfall (mm) | Continuous | Official+name | |
| mean_temperature_c | 13.9 | Mean annual temperature (degrees Celsius) | Continuous | Official+name | |

### 2.4 Policy fields shared by the three lines

Example (row 4 of the motor training table): policy M39362900 for the period 2021-01-01 to 2021-10-12, exposure 0.7808 years. There was a mid-term change (renewal_decision = mid_term_change), so this row ends early.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| row_id | M00000003 | Row ID; the first letter M/B/C indicates the line | ID | Official | The meaning of the first letter was observed from the data |
| policy_id | M39362900 | Policy ID, unchanged across years | ID | Official | Used to join claims and to compute prior-year experience |
| customer_id | 67002274 | Policyholder | ID | Official | Zero-padded for motor (C002) |
| address_id | A7357961 | Insured address (for motor, the policyholder's home address) | ID | Official | |
| period_start / period_end | 2021-01-01 / 2021-10-12 | Start and end dates of this row (inclusive) | Date | Official | |
| exposure | 0.7808 | How many years this row was on cover | Continuous | Official | **Contains errors; recomputed from the dates** (C001, see section 4) |
| voluntary_excess | 250 | Excess chosen by the customer: 0/250/500/1000 CU | Ordinal | Official | CU is a fictitious currency unit |
| compulsory_excess | 250 | Excess set by the insurer | Constant | Official | All 250 for motor and all 350 for buildings and contents; no discriminating power |
| channel | direct | Sales channel: direct / comparison site / broker | Nominal | Official | |
| payment_method | monthly | Payment method: annual / monthly | Binary | Official | |
| missed_payments_12m | 0 | Number of missed payments in the past 12 months | Count | Official | |
| new_business | False | Whether it is the first year with this company | Binary | Official | |
| renewal_count | 4 | Number of renewals so far | Count | Official | Test median 3, training median 6; the distributions differ (Q10) |
| days_quote_to_inception | 19 | Days from quote to inception | Count (0–30) | Official | |
| discount_applied | 0.05 | Premium discount | Ordinal (0 / 0.05 / 0.10) | Official | "0.05 = 5% discount" is inferred |
| written_premium | 430.08 | Premium charged under current pricing (CU) | Continuous | Official | It reflects the company's **existing** pricing judgement and can be used as a feature; it is not what we are predicting |
| customer_service_contacts | 0 | Number of customer service contacts | Count | Official | **Data leakage, removed** (C008, see section 4) |
| renewal_decision | mid_term_change | Outcome after this row ends: renewed / customer did not renew / cancelled / insurer declined / mid-term change | Nominal | Official | **Only in the training set; it is after-the-fact information and cannot be used** (Q9) |

### 2.5 Motor-specific fields (policies_motor / test_motor)

Example: the main driver is 87 years old, has held a licence for 68 years and has 8 years of no-claims bonus; the car is a 3-year-old automatic SUV, petrol, 2400cc, comprehensive cover, driven 10,000 miles/km a year, parked overnight on the driveway and used for commuting.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| driver_age | 87 | Main driver's age (at the start of this row) | Continuous (17–92) | Official | |
| years_licensed | 68 | Years holding a driving licence | Continuous | Official | |
| ncb_years | 8 | No-claims bonus years (consecutive years without a claim) | Count (0–15) | Official | |
| vehicle_value_group | 14 | Vehicle value group, 1 lowest, 20 highest | Ordinal | Official | |
| vehicle_age | 3.0 | Vehicle age | Continuous | Official | 2.5% missing; test median 9, training median 7 (Q10) |
| make_model | MM115 | Make and model code | Nominal (300 levels) | Official | Which car each code stands for is unknown (anonymised) |
| engine_size_cc | 2400.0 | Engine size (cc) | Continuous | Official+name | 2.5% missing |
| body_type | suv | Body: hatchback / saloon / estate / SUV / sports | Nominal | Official | |
| transmission | automatic | Manual / automatic | Binary | Official | |
| seats | 5 | Number of seats | Count (2/4/5/7) | Official | |
| vehicle_colour | red | Body colour | Nominal (8 levels) | Official | Almost equal counts across colours |
| fuel_type | petrol | Petrol / diesel / hybrid / electric | Nominal | Official | |
| driver_assist | False | Whether driver assistance is fitted (e.g. automatic braking) | Binary | Official | |
| cover_type | comprehensive | Cover level: comprehensive / third party, fire and theft / third party only | Ordinal | Official | |
| annual_mileage | 10000.0 | Declared annual mileage | Continuous | Official | Unit not stated; 2.5% missing |
| overnight_parking | driveway | Overnight parking: garage / driveway / street | Nominal | Official | 2.5% missing |
| vehicle_use | commuting | Use: private / commuting / business | Nominal | Official | |
| named_young_driver | False | Whether the policy names a driver under 25 | Binary | Official | |
| telematics_opt_in | False | Whether a telematics device is installed | Binary | Official | |
| telematics_score | NaN | Driving score from the telematics device, 1–100 | Continuous | Official | Blank when no device is installed (87%); **the blank itself carries meaning and is not imputed** (Q8). "A higher score means better driving" is inferred: policyholders with a claim average 58.2, those without 60.2 |

### 2.6 Buildings-specific fields (policies_buildings / test_buildings)

Example (first row of the test table): a 76-year-old owner-occupier, sum insured 89,000, brick construction, with smoke alarms, accidental damage cover added and a mortgage.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| policyholder_age | 76 | Policyholder's age | Continuous | Official | |
| sum_insured | 89000.0 | Sum insured (the maximum payable) | Continuous | Official | **About 1400 rows were inflated by a factor of 1000, now corrected** (C003) |
| construction_type | Brick | Construction: brick / stone / timber / non-standard | Nominal | Official | **Inconsistent spelling, plus two extra codes W and T** (C004) |
| occupancy | owner_occupied | Owner-occupied / let | Binary | Official | |
| days_unoccupied | 10.0 | Longest unoccupied spell (days) | Count | Official+name | 2.5% missing |
| smoke_alarms | True | Whether there are smoke alarms | Binary (with missing) | Official | 2.5% missing; once caused an XGBoost error (C012) |
| accidental_damage_cover | True | Whether "accidental damage" cover was added | Binary | Official | |
| mortgage | True | Whether there is a mortgage | Binary | Official | |

### 2.7 Contents-specific fields (policies_contents / test_contents)

Example (first row of the test table, the same customer and address as above): an owner, sum insured 38,500, with security locks and an alarm fitted, no high-value items, and accidental damage cover added.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| policyholder_age | 76 | Policyholder's age | Continuous | Official | |
| sum_insured | 38500.0 | Contents sum insured | Continuous | Official | Has the same ×1000 problem (C003) |
| tenure | owner | Owner / tenant | Binary | Official | |
| security_level | alarm_and_locks | Security: basic / locks / locks + alarm | Ordinal | Official | 2.5% missing |
| high_value_items | 0.0 | Number of declared high-value items | Count (0–5) | Official | 2.5% missing |
| accidental_damage_cover | True | Whether "accidental damage" cover was added | Binary | Official | |
| works_from_home | False | Whether the policyholder works from home | Binary | Official | |
| lodgers | False | Whether lodgers or students stay in the home | Binary | Official | |
| bicycle_cover | False | Whether bicycle cover was added | Binary | Official | |

### 2.8 claims (claim table)

Example 1 (motor own damage, with recovery): loss on 2024-09-20, reported 4 days later, closed. Paid 1,290.01, recovered 1,032.01, net cost 258.

Example 2 (buildings escape of water, open): loss on 2024-08-08, reported on 2024-09-25. Paid so far 252.22, estimated total payout 611.10.

Example 3 (motor theft, repudiated): status repudiated, paid and incurred are both 0.

| Field | Example | Meaning | Type | Source | Handling / notes |
|---|---|---|---|---|---|
| claim_id | CL75052008 | Claim ID | ID | Official | |
| policy_id | M63370288 | Which policy the claim belongs to | ID | Official | |
| line | motor | Line of business | Nominal | Official | |
| peril | own_damage | Cause of loss | Nominal | Official | 5 for motor, 6 for buildings, 4 for contents |
| loss_date | 2024-09-20 | Loss date | Date | Official | Used to match the claim to a policy row |
| report_date | 2024-09-24 | Report date | Date | Official | Report date − loss date = reporting delay, used for the development adjustment (C006) |
| status | closed | Closed / open / repudiated | Nominal | Official | Repudiated claims are not counted in the target |
| paid_to_date | 1290.01 | Amount paid to date (after deducting the excess) | Continuous | Official | |
| incurred_to_date | 1290.01 | Paid + estimate of what is still to be paid (outstanding reserve) | Continuous | Official | "Also net of excess" is inferred: of the 151,292 closed claims, incurred equals paid in every case |
| recoveries | 1032.01 | Money recovered through subrogation or salvage | Continuous | Official | Observed to **appear only in motor own damage and theft** |
| repairer_id | RP0886 | Repairer ID | ID | Official | Present only for own damage and third-party property; not used in the model (only known after the claim occurs) |
| third_party_id | TP9574811 | Third-party ID | ID | Official | Present only for third-party property and bodily injury; not used in the model |

**Claim peril codes and meanings**

| Code | Meaning | Line | Count |
|---|---|---|---|
| own_damage | Own damage | Motor | 48,964 |
| third_party_prop | Third-party property damage | Motor | 33,826 |
| glass_hail | Glass / hail | Motor | 15,732 |
| bodily_injury | Bodily injury | Motor | 7,583 |
| theft | Theft | Motor 6,501 / Contents 7,863 | 14,364 |
| escape_of_water | Escape of water | Buildings 10,137 / Contents 5,407 | 15,544 |
| accidental_damage | Accidental damage | Buildings 3,848 / Contents 9,208 | 13,056 |
| storm | Storm | Buildings | 7,816 |
| flood | Flood | Buildings 2,232 / Contents 1,624 | 3,856 |
| subsidence | Subsidence | Buildings | 1,294 |
| fire | Fire | Buildings | 1,115 |

### 2.9 Test tables (test_motor / test_buildings / test_contents)

The fields are identical to the corresponding training tables, except that `renewal_decision` is absent. All rows are 2025 coverage periods; the claim cost of each row is to be predicted, and **no answers are provided**.

Example: the first row of test_motor is the 2025 record of the same policy M39362900 as the customer in section 2.5. The driver is 91 years old and has switched to a 19-year-old manual hatchback. About 85% of the policies in the test set appear in the training set.

### 2.10 sample_submission (submission template)

| Field | Example | Meaning |
|---|---|---|
| row_id | M02409311 | Row ID of the test table |
| prediction | 0.0 | Predicted claim cost (CU), must be ≥ 0 |

Checked: its 626,639 row_id values match the row_id values of the three test tables exactly.

---

## 3. What is stated officially and what I inferred

**Everything stated officially**: **every field** in the 11 files has a one-line English description on the Data page, and the value ranges of most categorical fields are also listed (e.g. credit band from A best to E).

**What I inferred (not stated on the Data page)**

| Inference | Basis | How certain |
|---|---|---|
| T and W in construction_type both mean timber | T appears only in 2021–22 and W only in 2023–24; 13,250 policies change from T to W | Fairly certain, but not an official statement |
| A sum_insured above 5 million has been inflated by a factor of 1000 | The median sum insured of the same policy in other years is exactly 1/1000 of it; the premium is the same as for normal rows | Fairly certain |
| exposure is computed on a 365-day year | In 2024 all full-year rows end on 12-30; for 99.4% of rows exposure = days / 365 | Certain |
| incurred_to_date is also net of excess | The official text says only that paid is net of excess; for closed claims incurred equals paid in every case | Certain |
| A higher telematics_score is better | Policyholders with a claim have a lower average score | Fairly certain |
| discount_applied 0.05 = 5% discount | The only values are 0 / 0.05 / 0.10 | Fairly certain |
| Units of distance, rainfall, temperature and engine size | Column name suffixes `_km`, `_mm`, `_c`, `_cc` | Certain |
| Unit of annual_mileage (miles or kilometres) | No basis | **Unknown**; does not affect modelling |
| The first letter M / B / C of row_id indicates the line | Maps one-to-one to the files | Certain |
| What make_model, previous_insurer, branch_id and campaign_code actually stand for | All are anonymised codes | **Unknown**; can only be used as categories |

---

## 4. Issues found and how they were handled

| ID | Issue | Scale | Treatment | Change ID |
|---|---|---|---|---|
| Q1 | exposure entered wrongly: some are entered as a number of days (365), some as 1.0139 (365/360, a full year on a 360-day basis), and some use a 360-day year for part of a year | Training set: motor 5,723 rows, buildings 4,176 rows, contents 4,305 rows; none in the test set | All rows recomputed as (end date − start date + 1) / 365, with the original value kept in a separate column | C001 |
| Q2 | Leading 0 lost from motor customer_id | 98,110 training rows, 25,554 test rows | Padded to 8 digits; after padding 100% match | C002 |
| Q3 | Sum insured inflated by a factor of 1000 | Buildings 1,395 rows, contents 1,402 rows (training set); 730 rows in the test set | Values above 5 million are divided by 1000 and a flag column is added | C003 |
| Q4 | Construction type spelled inconsistently (Brick / brick / BRICK), plus two extra codes W and T | 11 spellings | Lower-cased; T and W mapped to timber; an unseen code raises an error | C004 |
| Q5 | Some claims match no policy row (mostly losses in 2020) | Motor 294 claims, buildings 67, contents 49 | Not counted in the target and listed separately; the reconciliation of total amounts shows a difference of 0 | C005 |
| Q6 | customer_service_contacts is after-the-fact information: in the training set each claim adds 2.5–3 contacts | All rows | Removed from the features; all fields were scanned and only this column has the problem | C008 |
| Q7 | About 2.5% random missingness (in a dozen or so fields) | About 2.5% per column | No mean imputation. The GLM treats missing as its own level; the GAM imputes the median and adds an "is missing" column; tree models handle nulls directly | C011 |
| Q8 | 87% of telematics_score is blank | Motor | Keep the blanks and use them together with "device installed or not" | – |
| Q9 | renewal_decision is after-the-fact information and is absent from the test set | Training set | Not used in the model | – |
| Q10 | Training and test distributions differ (renewal count, vehicle age, etc.) | Overall | Validate in time order and document it in the report | – |
| – | smoke_alarms is "yes / no / blank" and was mistakenly treated as a text category, causing an XGBoost error | Buildings | Converted to numeric 1 / 0 / blank | C012 |
| – | Claims not yet fully developed: 13–25% of claims with 2024 losses are still open, and some have not yet been reported | Most pronounced for 2024 | Estimate each row's "reported proportion" (completeness) from the reporting-delay curve | C006 |
| – | Extreme large claims (the largest is 11.89 million) | Motor: 8.8% of the amount lies above the cap | Cap each line at the 99.5th percentile and spread the trimmed part back evenly | C014 |

Two more observations (the data was not changed, but worth knowing when modelling):
- **compulsory_excess is a constant**: all 250 for motor and all 350 for buildings and contents; it is of no use to the model.
- **About 85% of the test-set policies appear in the training set**, so "how many claims this policy had in earlier years" is a very useful feature.

## 5. New fields (derived variables)

| New field | How it is computed | Type | Notes |
|---|---|---|---|
| customer_age | (policy start date − date of birth) / 365.25 | Continuous | Buildings and contents have policyholder_age and motor has driver_age; this is the customer's own age |
| customer_tenure | Start year − customer_since_year | Count | Number of years insured with the company |
| property_age | Start year − year_built | Continuous | Age of the property |
| prior_claims | Number of claims on the same policy in **earlier years** | Count | Uses only data from before the start year, so it does not peek at the answer |
| prior_cost | Claim amount on the same policy in earlier years | Continuous | As above |
| prior_exposure | Sum of exposure on the same policy in earlier years | Continuous | Used to distinguish "no claims" from "no history" |
| sum_insured_scaled | Whether the sum insured was corrected | Binary | C003 |
| completeness | The proportion of claims that should have been reported for this row by 2024-12-31 | Continuous | Used only on the target, not as a feature (C006) |
| claim_cost / claim_cost_capped | Total claim amount for this row / total after capping | Target | The model learns the capped version (C014) |

**Fields dropped before modelling**: all IDs, dates, exposure (used as a weight rather than a feature), surname, date_of_birth, year_built, customer_since_year (already converted into the derived variables above), renewal_decision, customer_service_contacts.

**A known imperfection**: fields such as `branch_id` (branch code), `flood_zone` and `household_size` are numeric in the data, and the code currently treats them as continuous numbers. `flood_zone` has a natural order, so this is not a big problem; but `branch_id` is only a code, and "branch 5 is larger than branch 3" has no meaning. Tree models are less affected; the GLM and GAM are affected. This must be corrected at the modelling step.
