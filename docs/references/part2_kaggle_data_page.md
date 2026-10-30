# Kaggle Part 2 Data page, original text (archived)

Source: https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-2/data
Capture date: 2026-10-24. The Glossary is the same as Part 1 and is not copied again.

---

## Overview

This dataset contains both markets:

- `developed/`: the Part 1 data for the established market, 2021 to 2024. It is included so you can reuse and compare your Part 1 work.
- `emerging/`: the new market your insurer is expanding into. The insurer has written business there for two policy years (2023 and 2024). At the start of 2025 it takes on a large portfolio from a broker, so most 2025 policies are new business.

In Part 2 you predict the emerging market only. Your submission covers every row in emerging/test_motor, emerging/test_buildings and emerging/test_contents.

All data is synthetic. No real person, address or insurer is represented, so you can use it freely with any tool, including AI assistants. Amounts are in a fictional currency unit (CU). The data comes as it would from an insurer's operational systems: in several linked tables, and not cleaned.

## What am I predicting?

For each row in the three emerging-market test files, predict the expected ultimate claim cost for that row. This is the risk premium, not the premium charged. It covers: all claims with a loss date within the row's period (period_start to period_end); net of the policy excess and of any recoveries; including claims not yet reported or settled; excluding claims the insurer declined.

The prediction is a total for the row, not an annual rate. A row with exposure of 0.5 covers half a year.

Submissions are scored with the D2 Tweedie score (power 1.5).

## Files

Both folders contain the same core tables. The emerging folder adds two files for the commercial premium task.

| File | In folder | One row per | Description |
|---|---|---|---|
| customers | both | Customer | Customer details, shared across all their policies |
| addresses | both | Address | Location and property characteristics |
| weather | both | Address and year | Annual weather and climate measures for each address |
| policies_motor | both | Policy period | Training data for private motor |
| policies_buildings | both | Policy period | Training data for home buildings |
| policies_contents | both | Policy period | Training data for home contents |
| claims | both | Claim | All reported claims on the training policies, as at 31 December 2024 |
| test_motor | both | Policy period | 2025 motor policies (emerging: predict these) |
| test_buildings | both | Policy period | 2025 buildings policies (emerging: predict these) |
| test_contents | both | Policy period | 2025 contents policies (emerging: predict these) |
| sample_submission | emerging | Test row | A Part 2 submission in the correct format (CSV) |
| expenses | emerging | Cost item | Expense, commission, tax, reinsurance and profit assumptions for both markets |
| reference_policies | emerging | Policy | 20 policies for the office premium build-up in your report |

| | Developed | Emerging |
|---|---|---|
| Training years | 2021 to 2024 | 2023 and 2024 |
| Weather years | 2021 to 2024 | 2023 and 2024 |
| Regions | R01 to R12 | E01 to E06 |
| Claims valued at | 31 December 2024 | 31 December 2024 |

Files are provided in Parquet format, except the sample submission, which is CSV.

## How the tables link

Join tables only within the same market folder. Customer, address and policy identifiers are unique within a market, but the same value can appear in both markets for unrelated records. Never join a developed table to an emerging table.

Within a market: `customer_id` links policies to customers; `address_id` links policies to addresses and weather; `policy_id` links claims to policies (match each claim to a policy row using its loss_date); `row_id` identifies each test row in your submission. Test row_ids are unique across both markets. A customer can hold motor, buildings and contents policies, and several customers can share an address.

## Policy periods

Each policy row covers one policy for one calendar year, or part of a year. A policy has more than one row in a year if it changed mid-term. Rows end early when a policy is cancelled or lapses. `exposure` is the length of the row in years.

## Columns

Identifiers (customer_id, address_id, policy_id, row_id, claim_id) are text. Where a column differs between markets, the Market column says so. A blank means the column is the same in both.

### customers

| Column | Market | Description |
|---|---|---|
| customer_id | | Customer identifier |
| surname | | Customer surname (anonymised) |
| date_of_birth | developed only | Date of birth |
| gender | emerging only | M or F. May be used for pricing in the emerging market; not available in the developed market |
| occupation | | Occupation. In the emerging market this is free text as entered by the customer |
| marital_status | | single, married, divorced, widowed |
| household_size | | Number of people in the household |
| credit_band | developed only | Credit score band, A (best) to E |
| prior_claims_5y | | Claims in the five years before joining, as declared by the customer |
| licence_points | | Penalty points on the driving licence |
| years_at_address | | Years the customer has lived at their address |
| customer_since_year | | Year the customer first took out a policy |
| email_domain | | Email provider |
| marketing_consent | | Consented to marketing |
| document_preference | | email, post or app |
| previous_insurer | | Insurer before joining |
| campaign_code | | Marketing campaign at acquisition, if any |
| branch_id | | Servicing branch |
| marketing_segment | | Segment assigned by the marketing team |

### addresses

Same columns as Part 1; `region`: R01 to R12 (developed), E01 to E06 (emerging).

### weather

Same columns as Part 1.

### Columns in all policy and test files

Same columns as Part 1 (row_id, policy_id, customer_id, address_id, period_start, period_end, exposure, voluntary_excess 0/250/500/1,000 CU, compulsory_excess, channel, payment_method, missed_payments_12m, new_business, renewal_count, days_quote_to_inception, discount_applied, written_premium, customer_service_contacts, renewal_decision training only).

### Motor columns

| Column | Market | Description |
|---|---|---|
| driver_age | developed only | Age of the main driver at the start of the row |
| driver_age_band | emerging only | Age band of the main driver: 17-24, 25-34, 35-49, 50-64, 65+ |
| years_licensed, ncb_years, vehicle_value_group, vehicle_age, make_model, engine_size_cc, body_type, transmission, seats, vehicle_colour, fuel_type, driver_assist, cover_type | | as Part 1 |
| annual_mileage | developed only | Declared annual mileage (not collected in the emerging market) |
| overnight_parking, vehicle_use, named_young_driver | | as Part 1 |
| telematics_opt_in | developed only | not offered in the emerging market |
| telematics_score | developed only | |

### Buildings columns

| Column | Market | Description |
|---|---|---|
| policyholder_age | developed only | Age at the start of the row |
| policyholder_age_band | emerging only | Age band at the start of the row: 17-24, 25-34, 35-49, 50-64, 65+ |
| sum_insured | | Buildings sum insured |
| construction_type | | Developed: brick, stone, timber, non-standard. Emerging (local codes): BRK (brick), STN (stone), WOD (timber), MIX (mixed or non-standard) |
| occupancy, days_unoccupied, smoke_alarms, accidental_damage_cover, mortgage | | as Part 1 |

### Contents columns

| Column | Market | Description |
|---|---|---|
| policyholder_age | developed only | |
| policyholder_age_band | emerging only | 17-24, 25-34, 35-49, 50-64, 65+ |
| sum_insured, tenure, security_level, high_value_items, accidental_damage_cover, works_from_home, lodgers, bicycle_cover | | as Part 1 |

### claims

Same columns and perils as Part 1. The policy excess applies to all perils, including third-party perils. Claims in both markets are valued at 31 December 2024. Some 2024 claims had not been reported by then, and open claims may still develop.

### expenses (emerging folder)

| Column | Description |
|---|---|
| item | Cost item |
| developed | Value in the developed market |
| emerging | Value in the emerging market (blank if not applicable) |
| unit | CU, share of premium, or share of property risk premium |

Items: fixed expense per policy, variable expense, commission by channel, premium tax, catastrophe reinsurance and target profit margin.

### reference_policies (emerging folder)

Twenty 2025 emerging-market policies taken from the test files (same columns, plus line). For each one, show in your written report how you build the office premium from your predicted risk premium using the expenses assumptions. This is assessed by the judging panel and does not affect the Kaggle score.

### sample_submission (emerging folder)

row_id from the emerging-market test files; prediction = expected ultimate claim cost for the row, net of excess and recoveries, in CU (must be zero or above). Submit one file covering every row in the three emerging-market test files.
