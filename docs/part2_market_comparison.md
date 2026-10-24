# Part 2 market comparison: developed vs emerging

Source: `scripts/part2_compare.py` on `data/iaa-ai-assisted-hackathon-2026-part-2/{developed,emerging}`. Reported (not developed) claim cost, valued at 2024-12-31 in both markets. Exposure-years are (period_end - period_start + 1) / 365, so the exposure typos found in `docs/part2_emerging_quality.txt` (Q1) do not enter. Claim count and cost use the C005 matching (loss_date inside the row period, same policy_id) and exclude repudiated claims; unmatched claims are not counted. Currency is the fictional CU in both markets.

Flag rule: **FLAG** means emerging differs from developed by more than 25% in relative terms (emerging / developed - 1); for category shares the move must also be at least 1 percentage point.

## 1. Headline numbers per line

### motor

| metric | dev 2021 | dev 2022 | dev 2023 | dev 2024 | dev 2023-24 | em 2023 | em 2024 | em 2023-24 | em/dev - 1 (2023-24) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| training rows | 216,008 | 254,685 | 254,577 | 254,698 | 509,275 | 21,680 | 25,603 | 47,283 |  |  |
| exposure-years | 187,991.5 | 199,698.5 | 199,681.9 | 199,785.6 | 399,467.5 | 18,864.5 | 20,104.0 | 38,968.5 |  |  |
| claims (matched, not repudiated) | 27,356 | 29,691 | 27,990 | 25,864 | 53,854 | 3,557 | 3,491 | 7,048 |  |  |
| claim frequency per exposure-year | 14.55% | 14.87% | 14.02% | 12.95% | 13.48% | 18.86% | 17.36% | 18.09% | 34% | **FLAG** |
| mean severity (CU, reported) | 3,871 | 3,931 | 4,191 | 3,506 | 3,862 | 2,295 | 2,204 | 2,250 | -42% | **FLAG** |
| reported pure premium per exposure-year (CU) | 563 | 584 | 587 | 454 | 521 | 433 | 383 | 407 | -22% |  |
| written_premium median, all rows (CU) | 458 | 469 | 485 | 501 | 493 | 498 | 505 | 502 | 2% |  |
| written_premium median, rows > 0.95 year (CU) | 457 | 468 | 483 | 499 | 491 | 496 | 502 | 499 | 2% |  |

Frequency ratio emerging / developed - 1: 2023 35%, 2024 34%. Emerging 2024 vs 2023 frequency -8%; pure premium -12% (developed 2024 vs 2023: frequency -8%, pure premium -23%).

### buildings

| metric | dev 2021 | dev 2022 | dev 2023 | dev 2024 | dev 2023-24 | em 2023 | em 2024 | em 2023-24 | em/dev - 1 (2023-24) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| training rows | 155,729 | 183,864 | 184,132 | 184,001 | 368,133 | 12,580 | 14,867 | 27,447 |  |  |
| exposure-years | 140,782.9 | 149,872.4 | 149,987.3 | 149,917.5 | 299,904.8 | 11,372.6 | 12,095.4 | 23,468.0 |  |  |
| claims (matched, not repudiated) | 6,195 | 6,792 | 6,817 | 6,571 | 13,388 | 519 | 537 | 1,056 |  |  |
| claim frequency per exposure-year | 4.40% | 4.53% | 4.55% | 4.38% | 4.46% | 4.56% | 4.44% | 4.50% | 1% |  |
| mean severity (CU, reported) | 8,725 | 9,059 | 9,510 | 10,250 | 9,874 | 11,269 | 6,909 | 9,052 | -8% |  |
| reported pure premium per exposure-year (CU) | 384 | 411 | 432 | 449 | 441 | 514 | 307 | 407 | -8% |  |
| written_premium median, all rows (CU) | 198 | 210 | 222 | 235 | 228 | 231 | 245 | 238 | 4% |  |
| written_premium median, rows > 0.95 year (CU) | 199 | 210 | 223 | 236 | 229 | 231 | 244 | 237 | 3% |  |

Frequency ratio emerging / developed - 1: 2023 0%, 2024 1%. Emerging 2024 vs 2023 frequency -3%; pure premium -40% (developed 2024 vs 2023: frequency -4%, pure premium 4%).

### contents

| metric | dev 2021 | dev 2022 | dev 2023 | dev 2024 | dev 2023-24 | em 2023 | em 2024 | em 2023-24 | em/dev - 1 (2023-24) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| training rows | 158,848 | 187,632 | 187,646 | 187,491 | 375,137 | 15,834 | 18,732 | 34,566 |  |  |
| exposure-years | 140,931.4 | 149,986.2 | 149,974.9 | 149,966.8 | 299,941.7 | 14,107.7 | 14,950.3 | 29,058.0 |  |  |
| claims (matched, not repudiated) | 5,583 | 6,024 | 6,158 | 5,988 | 12,146 | 491 | 507 | 998 |  |  |
| claim frequency per exposure-year | 3.96% | 4.02% | 4.11% | 3.99% | 4.05% | 3.48% | 3.39% | 3.43% | -15% |  |
| mean severity (CU, reported) | 2,663 | 2,663 | 2,848 | 2,925 | 2,886 | 2,618 | 2,560 | 2,589 | -10% |  |
| reported pure premium per exposure-year (CU) | 106 | 107 | 117 | 117 | 117 | 91 | 87 | 89 | -24% |  |
| written_premium median, all rows (CU) | 118 | 124 | 131 | 138 | 134 | 116 | 121 | 119 | -11% |  |
| written_premium median, rows > 0.95 year (CU) | 118 | 124 | 131 | 138 | 134 | 115 | 121 | 118 | -12% |  |

Frequency ratio emerging / developed - 1: 2023 -15%, 2024 -15%. Emerging 2024 vs 2023 frequency -3%; pure premium -5% (developed 2024 vs 2023: frequency -3%, pure premium -0%).

## 2. Test (2025) rows and exposure

| line | metric | developed | emerging | em/dev - 1 | flag |
|---|---|---|---|---|---|
| motor | test rows | 254,986 | 90,521 | -64% | **FLAG** |
| motor | test exposure-years | 199,984 | 79,007 | -60% | **FLAG** |
| motor | mean exposure per test row | 0.784 | 0.873 | 11% |  |
| motor | new_business share of test rows | 15.2% | 76.0% | 399% | **FLAG** |
| motor | written_premium median, test rows (CU) | 517 | 540 | 4% |  |
| motor | test policy_id seen in training | 84.8% | 24.0% | -72% | **FLAG** |
| buildings | test rows | 184,155 | 52,295 | -72% | **FLAG** |
| buildings | test exposure-years | 150,244 | 47,376 | -68% | **FLAG** |
| buildings | mean exposure per test row | 0.816 | 0.906 | 11% |  |
| buildings | new_business share of test rows | 15.1% | 75.9% | 401% | **FLAG** |
| buildings | written_premium median, test rows (CU) | 249 | 257 | 3% |  |
| buildings | test policy_id seen in training | 84.9% | 24.1% | -72% | **FLAG** |
| contents | test rows | 187,498 | 66,247 | -65% | **FLAG** |
| contents | test exposure-years | 150,124 | 59,016 | -61% | **FLAG** |
| contents | mean exposure per test row | 0.801 | 0.891 | 11% |  |
| contents | new_business share of test rows | 15.2% | 76.1% | 402% | **FLAG** |
| contents | written_premium median, test rows (CU) | 145 | 127 | -12% |  |
| contents | test policy_id seen in training | 84.8% | 23.9% | -72% | **FLAG** |

Ratio of test exposure-years to the 2023-24 training exposure-years (emerging): motor 2.03x, buildings 2.02x, contents 2.03x; developed (to 2021-24): motor 0.25x, buildings 0.25x, contents 0.25x.

## 3. Category mix (share of exposure-years)

`dev` = developed training 2021-24, `em` = emerging training 2023-24, `em test` = emerging 2025 test. construction_type is shown after mapping BRK -> brick, STN -> stone, WOD -> timber, MIX -> non_standard (developed after the C004 clean-up). The developed driver and policyholder ages were cut into the emerging bands. `flag dev` flags em vs dev; `flag test` flags em test vs em training (same rule).

### motor

| variable | level | dev | em | em test | em/dev - 1 | flag dev | test/em - 1 | flag test |
|---|---|---|---|---|---|---|---|---|
| cover_type | comprehensive | 82.1% | 78.9% | 77.5% | -4% |  | -2% |  |
| cover_type | third_party_fire_theft | 9.0% | 10.9% | 11.3% | 20% |  | 4% |  |
| cover_type | third_party_only | 8.9% | 10.2% | 11.2% | 14% |  | 10% |  |
| channel | broker | 25.0% | 70.2% | 70.0% | 181% | **FLAG** | -0% |  |
| channel | comparison_site | 35.0% | 4.9% | 5.0% | -86% | **FLAG** | 3% |  |
| channel | direct | 40.1% | 25.0% | 25.0% | -38% | **FLAG** | 0% |  |
| fuel_type | diesel | 27.6% | 29.6% | 29.2% | 7% |  | -1% |  |
| fuel_type | electric | 8.1% | 2.0% | 2.0% | -75% | **FLAG** | -1% |  |
| fuel_type | hybrid | 13.8% | 14.6% | 14.7% | 6% |  | 0% |  |
| fuel_type | petrol | 50.6% | 53.8% | 54.1% | 6% |  | 1% |  |
| body_type | coupe | 7.5% | 8.5% | 9.6% | 13% |  | 13% |  |
| body_type | estate | 12.4% | 12.1% | 10.4% | -2% |  | -14% |  |
| body_type | hatchback | 37.6% | 39.4% | 43.8% | 5% |  | 11% |  |
| body_type | saloon | 21.3% | 20.2% | 18.0% | -5% |  | -11% |  |
| body_type | suv | 21.2% | 19.8% | 18.2% | -7% |  | -8% |  |
| transmission | automatic | 45.1% | 45.1% | 45.0% | -0% |  | -0% |  |
| transmission | manual | 54.9% | 54.9% | 55.0% | 0% |  | 0% |  |
| vehicle_use | business | 9.9% | 10.0% | 9.9% | 1% |  | -2% |  |
| vehicle_use | commuting | 44.9% | 44.9% | 45.1% | -0% |  | 0% |  |
| vehicle_use | social | 45.2% | 45.1% | 45.1% | -0% |  | -0% |  |
| overnight_parking | (missing) | 2.5% | 12.1% | 11.9% | 388% | **FLAG** | -1% |  |
| overnight_parking | driveway | 48.8% | 44.0% | 44.2% | -10% |  | 0% |  |
| overnight_parking | garage | 19.4% | 17.4% | 17.5% | -10% |  | 1% |  |
| overnight_parking | street | 29.3% | 26.5% | 26.3% | -9% |  | -1% |  |
| payment_method | annual | 54.9% | 55.2% | 55.0% | 1% |  | -0% |  |
| payment_method | monthly | 45.1% | 44.8% | 45.0% | -1% |  | 1% |  |
| new_business | False | 91.6% | 92.2% | 23.1% | 1% |  | -75% | **FLAG** |
| new_business | True | 8.4% | 7.8% | 76.9% | -7% |  | 885% | **FLAG** |
| voluntary_excess | 0 | 40.0% | 40.3% | 40.1% | 1% |  | -1% |  |
| voluntary_excess | 1000 | 9.9% | 9.9% | 9.9% | -0% |  | 0% |  |
| voluntary_excess | 250 | 30.0% | 30.4% | 30.1% | 1% |  | -1% |  |
| voluntary_excess | 500 | 20.1% | 19.4% | 19.9% | -3% |  | 3% |  |
| driver_age_band | 17-24 | 13.3% | 19.6% | 25.1% | 47% | **FLAG** | 28% | **FLAG** |
| driver_age_band | 25-34 | 16.8% | 18.7% | 25.2% | 11% |  | 35% | **FLAG** |
| driver_age_band | 35-49 | 19.2% | 16.4% | 13.0% | -14% |  | -21% |  |
| driver_age_band | 50-64 | 19.3% | 16.8% | 13.1% | -13% |  | -22% |  |
| driver_age_band | 65+ | 31.4% | 28.5% | 23.5% | -9% |  | -17% |  |
| urban_density | dense_urban | 10.1% | 10.0% | 10.1% | -1% |  | 1% |  |
| urban_density | rural | 19.9% | 20.5% | 20.2% | 3% |  | -2% |  |
| urban_density | suburban | 45.0% | 44.8% | 45.0% | -0% |  | 1% |  |
| urban_density | urban | 25.1% | 24.7% | 24.7% | -1% |  | -0% |  |
| property_type | detached | 25.0% | 25.2% | 25.3% | 1% |  | 0% |  |
| property_type | flat | 20.0% | 19.7% | 19.8% | -1% |  | 0% |  |
| property_type | semi_detached | 30.0% | 30.0% | 29.9% | 0% |  | -0% |  |
| property_type | terraced | 25.1% | 25.0% | 25.0% | -0% |  | 0% |  |

### buildings

| variable | level | dev | em | em test | em/dev - 1 | flag dev | test/em - 1 | flag test |
|---|---|---|---|---|---|---|---|---|
| construction_type | brick | 70.0% | 69.9% | 70.2% | -0% |  | 0% |  |
| construction_type | non_standard | 9.9% | 10.2% | 10.0% | 3% |  | -2% |  |
| construction_type | stone | 8.0% | 7.8% | 7.9% | -1% |  | 0% |  |
| construction_type | timber | 12.1% | 12.1% | 11.9% | -0% |  | -1% |  |
| occupancy | let | 9.9% | 10.1% | 10.0% | 2% |  | -1% |  |
| occupancy | owner_occupied | 90.1% | 89.9% | 90.0% | -0% |  | 0% |  |
| channel | broker | 25.1% | 70.2% | 69.8% | 180% | **FLAG** | -0% |  |
| channel | comparison_site | 35.0% | 5.2% | 5.0% | -85% | **FLAG** | -3% |  |
| channel | direct | 39.9% | 24.6% | 25.1% | -38% | **FLAG** | 2% |  |
| payment_method | annual | 54.9% | 55.3% | 55.0% | 1% |  | -1% |  |
| payment_method | monthly | 45.1% | 44.7% | 45.0% | -1% |  | 1% |  |
| new_business | False | 91.6% | 92.3% | 23.2% | 1% |  | -75% | **FLAG** |
| new_business | True | 8.4% | 7.7% | 76.8% | -8% |  | 895% | **FLAG** |
| voluntary_excess | 0 | 50.1% | 49.8% | 50.0% | -0% |  | 0% |  |
| voluntary_excess | 1000 | 4.9% | 5.1% | 5.0% | 3% |  | -2% |  |
| voluntary_excess | 250 | 30.0% | 29.9% | 30.0% | -0% |  | 0% |  |
| voluntary_excess | 500 | 15.0% | 15.2% | 15.1% | 1% |  | -1% |  |
| policyholder_age_band | 17-24 | 13.2% | 19.5% | 24.9% | 48% | **FLAG** | 28% | **FLAG** |
| policyholder_age_band | 25-34 | 16.7% | 18.4% | 25.1% | 10% |  | 36% | **FLAG** |
| policyholder_age_band | 35-49 | 19.4% | 16.3% | 13.1% | -16% |  | -20% |  |
| policyholder_age_band | 50-64 | 19.3% | 16.9% | 13.3% | -13% |  | -21% |  |
| policyholder_age_band | 65+ | 31.4% | 28.9% | 23.5% | -8% |  | -19% |  |
| urban_density | dense_urban | 10.1% | 9.7% | 10.0% | -4% |  | 3% |  |
| urban_density | rural | 19.8% | 20.4% | 20.1% | 3% |  | -2% |  |
| urban_density | suburban | 44.9% | 45.4% | 45.3% | 1% |  | -0% |  |
| urban_density | urban | 25.2% | 24.5% | 24.6% | -3% |  | 0% |  |
| property_type | detached | 25.1% | 24.8% | 25.2% | -1% |  | 2% |  |
| property_type | flat | 19.9% | 19.8% | 19.8% | -0% |  | -0% |  |
| property_type | semi_detached | 29.9% | 29.6% | 29.7% | -1% |  | 0% |  |
| property_type | terraced | 25.1% | 25.8% | 25.4% | 3% |  | -2% |  |
| accidental_damage_cover | False | 60.1% | 60.1% | 60.1% | 0% |  | -0% |  |
| accidental_damage_cover | True | 39.9% | 39.9% | 39.9% | -0% |  | 0% |  |
| mortgage | False | 44.8% | 44.9% | 45.5% | 0% |  | 1% |  |
| mortgage | True | 55.2% | 55.1% | 54.5% | -0% |  | -1% |  |

### contents

| variable | level | dev | em | em test | em/dev - 1 | flag dev | test/em - 1 | flag test |
|---|---|---|---|---|---|---|---|---|
| tenure | owner | 70.0% | 56.9% | 56.2% | -19% |  | -1% |  |
| tenure | renter | 30.0% | 43.1% | 43.8% | 44% | **FLAG** | 2% |  |
| security_level | (missing) | 2.5% | 12.3% | 12.2% | 389% | **FLAG** | -1% |  |
| security_level | alarm_and_locks | 34.0% | 30.4% | 30.6% | -11% |  | 1% |  |
| security_level | locks | 29.3% | 26.6% | 26.4% | -9% |  | -1% |  |
| security_level | standard | 34.2% | 30.7% | 30.8% | -10% |  | 0% |  |
| channel | broker | 25.2% | 70.2% | 69.9% | 179% | **FLAG** | -0% |  |
| channel | comparison_site | 35.0% | 5.1% | 5.0% | -86% | **FLAG** | -2% |  |
| channel | direct | 39.8% | 24.7% | 25.1% | -38% | **FLAG** | 1% |  |
| payment_method | annual | 54.9% | 55.7% | 55.1% | 2% |  | -1% |  |
| payment_method | monthly | 45.1% | 44.3% | 44.9% | -2% |  | 2% |  |
| new_business | False | 91.6% | 92.3% | 23.0% | 1% |  | -75% | **FLAG** |
| new_business | True | 8.4% | 7.7% | 77.0% | -9% |  | 902% | **FLAG** |
| voluntary_excess | 0 | 50.0% | 50.9% | 50.2% | 2% |  | -1% |  |
| voluntary_excess | 1000 | 5.0% | 4.6% | 5.0% | -8% |  | 8% |  |
| voluntary_excess | 250 | 30.0% | 29.9% | 30.0% | -0% |  | 0% |  |
| voluntary_excess | 500 | 15.1% | 14.6% | 14.8% | -3% |  | 2% |  |
| policyholder_age_band | 17-24 | 13.2% | 19.8% | 24.9% | 50% | **FLAG** | 26% | **FLAG** |
| policyholder_age_band | 25-34 | 16.8% | 18.3% | 25.1% | 9% |  | 38% | **FLAG** |
| policyholder_age_band | 35-49 | 19.3% | 16.5% | 13.0% | -15% |  | -21% |  |
| policyholder_age_band | 50-64 | 19.3% | 16.7% | 13.3% | -13% |  | -20% |  |
| policyholder_age_band | 65+ | 31.4% | 28.7% | 23.7% | -9% |  | -18% |  |
| urban_density | dense_urban | 10.1% | 9.7% | 9.9% | -4% |  | 2% |  |
| urban_density | rural | 19.9% | 20.2% | 20.2% | 2% |  | -0% |  |
| urban_density | suburban | 44.9% | 45.7% | 45.4% | 2% |  | -1% |  |
| urban_density | urban | 25.1% | 24.4% | 24.6% | -3% |  | 1% |  |
| property_type | detached | 25.0% | 24.6% | 25.0% | -2% |  | 2% |  |
| property_type | flat | 20.0% | 20.2% | 20.0% | 1% |  | -1% |  |
| property_type | semi_detached | 30.0% | 29.8% | 29.7% | -1% |  | -0% |  |
| property_type | terraced | 25.0% | 25.4% | 25.3% | 2% |  | -1% |  |
| accidental_damage_cover | False | 55.1% | 55.4% | 54.6% | 0% |  | -1% |  |
| accidental_damage_cover | True | 44.9% | 44.6% | 45.4% | -1% |  | 2% |  |
| works_from_home | False | 75.0% | 75.0% | 75.0% | 0% |  | 0% |  |
| works_from_home | True | 25.0% | 25.0% | 25.0% | -0% |  | -0% |  |

## 4. Numeric features (medians over training rows; address fields joined by address_id)

| line | variable | dev | em | em test | em/dev - 1 | flag dev | test/em - 1 | flag test |
|---|---|---|---|---|---|---|---|---|
| motor | vehicle_age | 7.00 | 9.00 | 10.00 | 29% | **FLAG** | 11% |  |
| motor | years_licensed | 28.00 | 24.00 | 13.00 | -14% |  | -46% | **FLAG** |
| motor | ncb_years | 7.00 | 6.00 | 5.00 | -14% |  | -17% |  |
| motor | vehicle_value_group | 11.00 | 11.00 | 11.00 | 0% |  | 0% |  |
| motor | engine_size_cc | 2,100.00 | 2,100.00 | 2,100.00 | 0% |  | 0% |  |
| motor | seats | 5.00 | 5.00 | 5.00 | 0% |  | 0% |  |
| motor | written_premium | 479.54 | 502.06 | 539.50 | 5% |  | 7% |  |
| motor | renewal_count | 6.00 | 7.00 | 0.00 | 17% |  | -100% | **FLAG** |
| motor | year_built | 1,965.00 | 1,965.00 | 1,964.00 | 0% |  | -0% |  |
| motor | bedrooms | 3.00 | 3.00 | 3.00 | 0% |  | 0% |  |
| motor | distance_to_river_km | 2.36 | 2.35 | 2.37 | -0% |  | 1% |  |
| motor | elevation_m | 8.10 | 7.50 | 7.50 | -7% |  | 0% |  |
| motor | flood_zone | 1.00 | 1.00 | 1.00 | 0% |  | 0% |  |
| buildings | sum_insured | 223,000.00 | 229,000.00 | 240,000.00 | 3% |  | 5% |  |
| buildings | days_unoccupied | 7.00 | 7.00 | 7.00 | 0% |  | 0% |  |
| buildings | written_premium | 216.66 | 237.56 | 256.77 | 10% |  | 8% |  |
| buildings | renewal_count | 6.00 | 7.00 | 0.00 | 17% |  | -100% | **FLAG** |
| buildings | year_built | 1,965.00 | 1,965.00 | 1,965.00 | 0% |  | 0% |  |
| buildings | bedrooms | 3.00 | 3.00 | 3.00 | 0% |  | 0% |  |
| buildings | distance_to_river_km | 2.36 | 2.37 | 2.37 | 0% |  | 0% |  |
| buildings | elevation_m | 8.10 | 7.70 | 7.60 | -5% |  | -1% |  |
| buildings | flood_zone | 1.00 | 1.00 | 1.00 | 0% |  | 0% |  |
| contents | sum_insured | 38,900.00 | 35,800.00 | 36,300.00 | -8% |  | 1% |  |
| contents | high_value_items | 1.00 | 0.00 | 0.00 | -100% | **FLAG** |  |  |
| contents | written_premium | 127.48 | 118.95 | 127.03 | -7% |  | 7% |  |
| contents | renewal_count | 6.00 | 7.00 | 0.00 | 17% |  | -100% | **FLAG** |
| contents | year_built | 1,965.00 | 1,964.00 | 1,965.00 | -0% |  | 0% |  |
| contents | bedrooms | 3.00 | 3.00 | 3.00 | 0% |  | 0% |  |
| contents | distance_to_river_km | 2.37 | 2.36 | 2.37 | -0% |  | 0% |  |
| contents | elevation_m | 8.10 | 7.60 | 7.50 | -6% |  | -1% |  |
| contents | flood_zone | 1.00 | 1.00 | 1.00 | 0% |  | 0% |  |

Mean weather by address, year 2024 (all addresses of the market):

| variable | dev | em | em/dev - 1 | flag |
|---|---|---|---|---|
| hail_days | 6.768 | 17.081 | 152% | **FLAG** |
| wind_exposure_index | 4.351 | 5.413 | 24% |  |
| freezing_days | 39.482 | 17.490 | -56% | **FLAG** |
| dry_summer_index | 0.594 | 0.712 | 20% |  |
| wildfire_index | 0.156 | 0.549 | 252% | **FLAG** |
| rainfall_mm | 1,036.392 | 1,375.087 | 33% | **FLAG** |
| mean_temperature_c | 12.972 | 15.777 | 22% |  |

## 5. Missing-value rates (columns where either market is above 0.5%)

| table | column | dev | em |
|---|---|---|---|
| customers | occupation | 2.47% | 12.16% |
| customers | marital_status | 2.52% | 12.03% |
| customers | household_size | 2.47% | 11.80% |
| customers | years_at_address | 2.51% | 11.94% |
| customers | campaign_code | 51.19% | 20.19% |
| customers | (columns only in developed: date_of_birth, credit_band; only in emerging: gender) |  |  |
| addresses | flood_zone | 2.92% | 2.95% |
| addresses | distance_to_river_km | 2.50% | 11.99% |
| addresses | soil_type | 2.50% | 12.04% |
| addresses | roof_type | 2.50% | 11.93% |
| addresses | energy_rating | 2.51% | 12.06% |
| addresses | (columns only in developed: ; only in emerging: ) |  |  |
| policies_motor | vehicle_age | 2.49% | 11.85% |
| policies_motor | engine_size_cc | 2.52% | 12.15% |
| policies_motor | overnight_parking | 2.49% | 12.05% |
| policies_motor | (columns only in developed: driver_age, annual_mileage, telematics_opt_in, telematics_score; only in emerging: driver_age_band) |  |  |
| policies_buildings | days_unoccupied | 2.49% | 11.99% |
| policies_buildings | smoke_alarms | 2.52% | 11.90% |
| policies_buildings | (columns only in developed: policyholder_age; only in emerging: policyholder_age_band) |  |  |
| policies_contents | security_level | 2.51% | 12.26% |
| policies_contents | high_value_items | 2.51% | 11.77% |
| policies_contents | (columns only in developed: policyholder_age; only in emerging: policyholder_age_band) |  |  |

## 6. Claims

| line | set | claims | repudiated | open (all years) | open, loss year 2024 | with recovery | report lag median days | lag p90 | lag p99 | net cost median | p90 | p99 | cap (p99.5) | claims above cap | cost share above cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| motor | developed 2021-24 | 112606 | 1.25% | 6.6% | 20.7% | 13.1% | 9 | 42 | 377 | 1,532 | 8,214 | 35,698 | 51,585 | 556 | 8.8% |
| motor | developed 2023-24 | 54561 | 1.24% | 11.9% | 20.7% | 12.6% | 9 | 38 | 252 | 1,591 | 8,253 | 29,789 | 41,889 | 270 | 11.5% |
| motor | emerging (all claims) | 7183 | 1.49% | 11.7% | 20.0% | 10.5% | 9 | 41 | 331 | 992 | 4,923 | 18,537 | 27,578 | 36 | 5.0% |
| buildings | developed 2021-24 | 26442 | 0.00% | 7.2% | 24.7% | 0.0% | 9 | 34 | 239 | 3,128 | 21,225 | 98,993 | 147,445 | 133 | 4.2% |
| buildings | developed 2023-24 | 13395 | 0.00% | 13.6% | 24.7% | 0.0% | 9 | 33 | 190 | 3,357 | 22,450 | 104,678 | 150,926 | 67 | 4.2% |
| buildings | emerging (all claims) | 1059 | 0.00% | 14.8% | 25.3% | 0.0% | 8 | 31 | 182 | 2,331 | 17,906 | 138,510 | 220,808 | 6 | 4.9% |
| contents | developed 2021-24 | 24102 | 1.24% | 3.4% | 13.0% | 0.0% | 6 | 23 | 63 | 1,191 | 6,896 | 22,050 | 28,449 | 120 | 2.1% |
| contents | developed 2023-24 | 12317 | 1.33% | 6.6% | 13.0% | 0.0% | 6 | 23 | 54 | 1,239 | 7,149 | 23,396 | 29,750 | 61 | 2.4% |
| contents | emerging (all claims) | 1016 | 1.28% | 5.8% | 11.5% | 0.0% | 6 | 22 | 72 | 1,177 | 6,352 | 19,960 | 28,312 | 6 | 2.1% |

Perils (share of claims within line):

| line | market | own_damage | third_party_prop | glass_hail | bodily_injury | theft | escape_of_water | storm | accidental_damage | flood | subsidence | fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| motor | dev | 43.5% | 30.0% | 14.0% | 6.7% | 5.8% |  |  |  |  |  |  |
| motor | em | 37.8% | 31.5% | 18.8% | 6.3% | 5.7% |  |  |  |  |  |  |
| buildings | dev |  |  |  |  |  | 38.3% | 29.6% | 14.6% | 8.4% | 4.9% | 4.2% |
| buildings | em |  |  |  |  |  | 25.3% | 44.0% | 11.5% | 7.6% | 4.9% | 6.6% |
| contents | dev |  |  |  |  | 32.6% | 22.4% |  | 38.2% | 6.7% |  |  |
| contents | em |  |  |  |  | 41.1% | 16.9% |  | 33.5% | 8.5% |  |  |

Caps: the Part 1 pipeline caps each claim at the 99.5th percentile of the line's non-repudiated net costs (C014). Emerging caps are estimated from 2 accident years (about 1,000 property claims and 7,000 motor claims), so each is set by only a handful of claims.

## 7. Everything flagged above (emerging vs developed; emerging test vs emerging training where stated)

- motor: claim frequency per exposure-year emerging 0.1809 vs developed 0.1348 (34%)
- motor: mean severity (CU, reported) emerging 2,250 vs developed 3,862 (-42%)
- motor: test rows emerging 90,521 vs developed 254,986 (-64%)
- motor: test exposure-years emerging 79,007 vs developed 199,984 (-60%)
- motor: new_business share of test rows emerging 76.0% vs developed 15.2% (399%)
- motor: test policy_id seen in training emerging 24.0% vs developed 84.8% (-72%)
- buildings: test rows emerging 52,295 vs developed 184,155 (-72%)
- buildings: test exposure-years emerging 47,376 vs developed 150,244 (-68%)
- buildings: new_business share of test rows emerging 75.9% vs developed 15.1% (401%)
- buildings: test policy_id seen in training emerging 24.1% vs developed 84.9% (-72%)
- contents: test rows emerging 66,247 vs developed 187,498 (-65%)
- contents: test exposure-years emerging 59,016 vs developed 150,124 (-61%)
- contents: new_business share of test rows emerging 76.1% vs developed 15.2% (402%)
- contents: test policy_id seen in training emerging 23.9% vs developed 84.8% (-72%)
- motor: mix channel=broker: emerging 70.2% vs developed 25.0%
- motor: mix channel=comparison_site: emerging 4.9% vs developed 35.0%
- motor: mix channel=direct: emerging 25.0% vs developed 40.1%
- motor: mix fuel_type=electric: emerging 2.0% vs developed 8.1%
- motor: mix overnight_parking=(missing): emerging 12.1% vs developed 2.5%
- motor: mix new_business=False: emerging test 23.1% vs emerging training 92.2%
- motor: mix new_business=True: emerging test 76.9% vs emerging training 7.8%
- motor: mix driver_age_band=17-24: emerging 19.6% vs developed 13.3%
- motor: mix driver_age_band=17-24: emerging test 25.1% vs emerging training 19.6%
- motor: mix driver_age_band=25-34: emerging test 25.2% vs emerging training 18.7%
- buildings: mix channel=broker: emerging 70.2% vs developed 25.1%
- buildings: mix channel=comparison_site: emerging 5.2% vs developed 35.0%
- buildings: mix channel=direct: emerging 24.6% vs developed 39.9%
- buildings: mix new_business=False: emerging test 23.2% vs emerging training 92.3%
- buildings: mix new_business=True: emerging test 76.8% vs emerging training 7.7%
- buildings: mix policyholder_age_band=17-24: emerging 19.5% vs developed 13.2%
- buildings: mix policyholder_age_band=17-24: emerging test 24.9% vs emerging training 19.5%
- buildings: mix policyholder_age_band=25-34: emerging test 25.1% vs emerging training 18.4%
- contents: mix tenure=renter: emerging 43.1% vs developed 30.0%
- contents: mix security_level=(missing): emerging 12.3% vs developed 2.5%
- contents: mix channel=broker: emerging 70.2% vs developed 25.2%
- contents: mix channel=comparison_site: emerging 5.1% vs developed 35.0%
- contents: mix channel=direct: emerging 24.7% vs developed 39.8%
- contents: mix new_business=False: emerging test 23.0% vs emerging training 92.3%
- contents: mix new_business=True: emerging test 77.0% vs emerging training 7.7%
- contents: mix policyholder_age_band=17-24: emerging 19.8% vs developed 13.2%
- contents: mix policyholder_age_band=17-24: emerging test 24.9% vs emerging training 19.8%
- contents: mix policyholder_age_band=25-34: emerging test 25.1% vs emerging training 18.3%
- motor: median vehicle_age: emerging 9.00 vs developed 7.00 (29%)
- motor: median years_licensed: emerging test 13.00 vs emerging training 24.00 (-46%)
- motor: median renewal_count: emerging test 0.00 vs emerging training 7.00 (-100%)
- buildings: median renewal_count: emerging test 0.00 vs emerging training 7.00 (-100%)
- contents: median high_value_items: emerging 0.00 vs developed 1.00 (-100%)
- contents: median renewal_count: emerging test 0.00 vs emerging training 7.00 (-100%)
- weather 2024 mean hail_days: emerging 17.081 vs developed 6.768 (152%)
- weather 2024 mean freezing_days: emerging 17.490 vs developed 39.482 (-56%)
- weather 2024 mean wildfire_index: emerging 0.549 vs developed 0.156 (252%)
- weather 2024 mean rainfall_mm: emerging 1,375.087 vs developed 1,036.392 (33%)

Total flagged items: 52.
