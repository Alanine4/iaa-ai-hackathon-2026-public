# Office premium formula draft and data check

Purpose: in the Part 2 report, explain for the 20 reference policies how the office premium is derived from the predicted risk premium. This document gives a draft formula chain and the definitions that a person must decide. The data comes from `expenses.parquet` and `reference_policies.parquet` under `data/iaa-ai-assisted-hackathon-2026-part-2/emerging/`, printed and checked by `scripts/part2_expenses.py`.

## 1. Official text (source: `docs/references/part2_kaggle_data_page.md`)

- File list table (lines 44–45):
  - "expenses | emerging | Cost item | Expense, commission, tax, reinsurance and profit assumptions for both markets"
  - "reference_policies | emerging | Policy | 20 policies for the office premium build-up in your report"
- expenses section (lines 140–149): the columns are `item` (Cost item), `developed` (Value in the developed market), `emerging` (Value in the emerging market (blank if not applicable)), `unit` (CU, share of premium, or share of property risk premium). "Items: fixed expense per policy, variable expense, commission by channel, premium tax, catastrophe reinsurance and target profit margin."
- reference_policies section (lines 151–153): "Twenty 2025 emerging-market policies taken from the test files (same columns, plus line). For each one, show in your written report how you build the office premium from your predicted risk premium using the expenses assumptions. This is assessed by the judging panel and does not affect the Kaggle score."
- Prediction target (lines 19–23): "This is the risk premium, not the premium charged." "The prediction is a total for the row, not an annual rate. A row with exposure of 0.5 covers half a year."

`docs/references/part2_kaggle_overview_page.md` contains no text about office premium, expenses or reference policies. The official sources give no combination formula; the formula chain below is a draft.

## 2. Full expenses.parquet table

8 rows, 4 columns: `item`, `developed`, `emerging`, `unit`. This task uses only the `emerging` column; the `developed` column is listed for comparison only.

| item | developed | emerging | unit | Meaning | Dimension | By line? | By channel? |
|---|---|---|---|---|---|---|---|
| fixed_expense_per_policy | 45.00 | 85.00 | CU | Fixed expense per policy | Amount (CU) | Uniform across lines | No |
| variable_expense | 0.12 | 0.15 | share of premium | Variable expense rate | Share of premium | Uniform across lines | No |
| commission_direct | 0.00 | 0.00 | share of premium | Commission rate for the direct channel | Share of premium | Uniform across lines | direct only |
| commission_comparison_site | 0.10 | blank (NaN) | share of premium | Commission rate for the comparison site channel | Share of premium | Uniform across lines | comparison_site only |
| commission_broker | 0.15 | 0.20 | share of premium | Commission rate for the broker channel | Share of premium | Uniform across lines | broker only |
| premium_tax | 0.12 | 0.08 | share of premium | Premium tax rate | Share of premium | Uniform across lines | No |
| cat_reinsurance | 0.06 | 0.12 | share of property risk premium | Catastrophe reinsurance cost | Share of property risk premium | Property lines only (see open question 3) | No |
| target_profit_margin | 0.05 | 0.08 | share of premium | Target profit margin | Share of premium | Uniform across lines | No |

The "By line?" and "By channel?" columns are derived from the item names and the unit; the file has no line column and no channel column.

## 3. The 20 rows of reference_policies.parquet

The file has 50 columns, one more than the test files: `line` (line of business). Line split: motor 7 (row_id starts with M), buildings 7 (B), contents 6 (C). Channel: broker 17, direct 3, no comparison_site. New business (`new_business`) 18 policies, renewal 2. Only the fields relevant to the office premium derivation are listed below. `written_premium` is the premium collected under current pricing (Part 1 data page glossary: "Premium charged under the insurer's current pricing", source `docs/references/kaggle_data_page.md` line 122), not the risk premium for this task.

| row_id | policy_id | line | channel | new_business | period_start | period_end | exposure | written_premium | discount_applied | voluntary_excess | compulsory_excess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B00215707 | B75172079 | buildings | broker | True | 2025-01-06 | 2025-12-31 | 0.9863 | 272.44 | 0.00 | 0 | 350 |
| B00215809 | B88623113 | buildings | direct | True | 2025-01-10 | 2025-12-31 | 0.9753 | 237.22 | 0.05 | 0 | 350 |
| B00217584 | B13799940 | buildings | broker | True | 2025-01-13 | 2025-12-31 | 0.9671 | 323.36 | 0.00 | 0 | 350 |
| B00218282 | B74876592 | buildings | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 230.80 | 0.00 | 250 | 350 |
| B00223369 | B73680946 | buildings | broker | True | 2025-01-03 | 2025-12-31 | 0.9945 | 204.56 | 0.00 | 0 | 350 |
| B00229821 | B92110463 | buildings | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 285.60 | 0.00 | 250 | 350 |
| B00250679 | B05817658 | buildings | broker | True | 2025-11-22 | 2025-12-31 | 0.1096 | 180.72 | 0.00 | 0 | 350 |
| C00259568 | C16955632 | contents | broker | False | 2025-01-01 | 2025-12-31 | 1.0000 | 83.18 | 0.00 | 1000 | 350 |
| C00270331 | C97318122 | contents | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 140.29 | 0.00 | 1000 | 350 |
| C00283470 | C11841593 | contents | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 117.97 | 0.00 | 250 | 350 |
| C00286287 | C12450281 | contents | broker | True | 2025-01-13 | 2025-12-31 | 0.9671 | 269.23 | 0.00 | 0 | 350 |
| C00302985 | C36495409 | contents | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 115.88 | 0.00 | 250 | 350 |
| C00314425 | C17515168 | contents | broker | True | 2025-01-13 | 2025-02-06 | 0.0685 | 102.87 | 0.00 | 250 | 350 |
| M00118569 | M75811658 | motor | direct | False | 2025-01-01 | 2025-12-09 | 0.9397 | 1095.48 | 0.05 | 0 | 250 |
| M00129413 | M41089337 | motor | broker | True | 2025-09-02 | 2025-12-31 | 0.3315 | 354.15 | 0.00 | 250 | 250 |
| M00153851 | M47069725 | motor | broker | True | 2025-01-15 | 2025-12-31 | 0.9616 | 414.72 | 0.00 | 0 | 250 |
| M00161473 | M92909747 | motor | broker | True | 2025-01-14 | 2025-12-31 | 0.9644 | 1131.78 | 0.00 | 250 | 250 |
| M00167496 | M61457154 | motor | broker | True | 2025-01-07 | 2025-12-31 | 0.9836 | 463.95 | 0.00 | 250 | 250 |
| M00176777 | M42023395 | motor | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 443.61 | 0.00 | 0 | 250 |
| M00185311 | M83715550 | motor | direct | True | 2025-01-14 | 2025-12-31 | 0.9644 | 872.31 | 0.05 | 1000 | 250 |

3 rows have exposure below 0.5: B00250679 (0.1096), C00314425 (0.0685), M00129413 (0.3315); the fixed expense definition affects these 3 most.

## 4. Formula chain (draft)

Notation: R = predicted risk premium of the row (row total, not annualised, in CU); exposure = the `exposure` of the row. Each step below states the fields used.

| Step | Content | Formula | Fields used |
|---|---|---|---|
| 1 | Risk premium | R = prediction for that row_id in the final submission file | From the final submission file in `work/`, not an expenses field |
| 2 | Catastrophe reinsurance cost (buildings and contents only; motor takes 0) | Cat = 0.12 × R | `cat_reinsurance` |
| 3 | Fixed expense (per policy) | F = 85 CU; whether to multiply by exposure: see open question 1 | `fixed_expense_per_policy` |
| 4 | Channel commission rate c | broker takes 0.20; direct takes 0.00; comparison_site is blank (see open question 2) | `commission_broker`, `commission_direct`, `commission_comparison_site` |
| 5 | Sum of the items proportional to the office premium, k | k = 0.15 + c + 0.08 + 0.08, i.e. variable_expense + c + premium_tax + target_profit_margin | `variable_expense`, `premium_tax`, `target_profit_margin`, and c from step 4 |
| 6 | Office premium | P = (R + Cat + F) ÷ (1 − k) | Steps 1–5 |

Denominators (addition and subtraction only, values from the table above): for broker, 1 − k = 1 − 0.15 − 0.20 − 0.08 − 0.08 = 0.49; for direct, 1 − k = 1 − 0.15 − 0.00 − 0.08 − 0.08 = 0.69.

The amount each expense item contributes to P can be shown in separate columns in the report table:

- Variable expense = variable_expense × P
- Commission = c × P
- Premium tax = premium_tax × P
- Profit = target_profit_margin × P
- Catastrophe reinsurance = Cat (step 2, property lines only)
- Fixed expense = F (step 3)

Check: risk premium + catastrophe reinsurance + fixed expense + variable expense + commission + premium tax + profit = P.

The formula chain rests on the literal meaning of the unit column: items in `share of premium` are taken as a proportion of the office premium P, so they go in the denominator; `cat_reinsurance`, in `share of property risk premium`, is taken as a proportion of the risk premium R, so it is a cost added to R; `fixed_expense_per_policy` is in CU, so it is an amount added to the numerator. The official text does not say how to combine them; this structure is an interpretation of the unit and needs confirmation by a team member.

## 5. Open questions that a person must decide

| # | Question | AI suggestion | Reason |
|---|---|---|---|
| 1 | The fixed expense of 85 CU is "per policy"; the reference policies include rows with exposure of only 0.0685, 0.1096 and 0.3315. How should it be allocated? | Prorate by exposure: F = 85 × exposure; state the definition in the report and give the unprorated result as a comparison in a footnote | The risk premium R is a row total (official: a row with exposure 0.5 covers half a year), and R and F are in the same numerator, so their time basis should be consistent; when one policy has several rows within a year (mid-term changes), not prorating would charge the fee more than once. Counterargument: the acquisition and set-up costs of a short-term policy do not shrink with time, so not prorating is closer to the true cost |
| 2 | `commission_comparison_site` is blank in emerging; is it reasonable to treat it as 0? | Treat it as 0 and state this in the report; also note that the comparison site channel is about 5% of the emerging data (`docs/PART2_INTAKE.md` N2), and that this assumption has no effect on the 20 reference policies (none of them uses this channel) | The official description of this column is "blank if not applicable"; read literally it means not applicable, so treating it as 0 does not contradict the official text; but the channel exists in the data, so treating it as 0 would understate cost. To be prudent, the sensitivity note can show the difference when the developed value of 0.10 is used instead |
| 3 | Is the catastrophe reinsurance cost added only for buildings/contents, or also for motor? | Add it only for buildings and contents; motor takes 0 | The unit says "share of property risk premium"; motor is not a property line. The official text does not state whether "property" includes contents; we recommend adding it for both lines and stating this in the report |
| 4 | At which step is premium tax `premium_tax` (0.08) / IPT added? | Put it in the denominator, together with variable expense, commission and profit, as a proportion of the office premium | The unit is "share of premium". An alternative reading is to compute the premium excluding tax first, then multiply by (1 + tax rate); the two readings have different tax bases and give different results, so a team member needs to choose one and write out the formula in the report. The official text does not say whether premium tax also applies to the fixed expense |
| 5 | Should a premium cap/floor be applied to the 20 reference policies? | No cap/floor in the main table; add a `written_premium` column for comparison, and explain rows with large deviations in a note | Caps and floors are a commercial decision; the official text provides no cap/floor parameters; applying a cap would mask the model prediction itself. `written_premium` is the result of current pricing and serves only as a reference |
| 6 | Should `discount_applied` (0.05, on 3 of the reference policies) be deducted from the office premium? | No | The office premium is derived from the risk premium and does not depend on current pricing; the discount is a component of `written_premium` and shows only in the comparison column |
| 7 | What if the predicted risk premium is 0? | Calculate by the formula: P = F ÷ (1 − k) (the same for property lines, with Cat = 0) | Submissions must be non-negative (official: "predictions zero or above"); the formula is still defined at R = 0; the fixed expense does not disappear with R |
| 8 | How many decimal places to keep? | No rounding in intermediate steps; display two decimals in the final result | Avoids accumulation of rounding error; `written_premium` itself has two decimals |

## 6. Deliverable layout

The report contains a 20-row table, one reference policy per row, sorted by line, then row_id.

| Column | Source | Status |
|---|---|---|
| row_id | reference_policies | Available |
| line (line of business) | reference_policies | Available |
| channel | reference_policies | Available |
| exposure | reference_policies | Available |
| Risk premium R | prediction for that row_id in the final submission file in `work/` | To be filled (left blank for now, no numbers invented) |
| Catastrophe reinsurance (property lines only) | 0.12 × R | To be calculated |
| Fixed expense | 85 (or 85 × exposure, depending on open question 1) | To be calculated |
| Variable expense | 0.15 × P | To be calculated |
| Commission | c × P (broker 0.20, direct 0.00) | To be calculated |
| Premium tax | 0.08 × P | To be calculated |
| Profit | 0.08 × P | To be calculated |
| Office premium P | (R + catastrophe reinsurance + fixed expense) ÷ (1 − k) | To be calculated |
| written_premium (comparison) | reference_policies | Available |

Below the table state: the currency is CU; the definitions used (final choices for open questions 1, 2, 3 and 4); the source file and version of the risk premium.

## 7. Check results

- expenses: 8 rows, 4 columns; field names, values and units match the parquet (printed by `scripts/part2_expenses.py`). In the emerging column `commission_comparison_site` is NaN, and the other 7 items have values.
- reference_policies: 20 rows, all row_id values are listed; line split motor 7 / buildings 7 / contents 6, channel broker 17 / direct 3, consistent with the description in section 6 of `docs/PART2_INTAKE.md`.
- This document contains no risk premium or office premium values.
