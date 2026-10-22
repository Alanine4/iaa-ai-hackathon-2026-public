# Kaggle submissions (Part 1)

Public leaderboard uses ~33% of the test rows; the final ranking uses the other ~67%.
Validation D2 = combined three-line Tweedie D2 on 2024 (`docs/validation_results.md`).

| # | Date | File | What it is | Validation D2 | Public score | Public rank | Notes |
|---|---|---|---|---|---|---|---|
| S01 | 2026-10-21 | `sub_blend.csv` | Equal-weight blend of GLM, GAM, LightGBM, XGBoost; capped at q99.5 + load (C014), rebalanced (C013), trend per line (C009: motor 1.035, buildings 1.199, contents 1.131) | 0.1336 | 0.12453 | 19 | Uploaded manually. Leader at the time: 0.12996 |
| S02 | 2026-10-21 | `sub_blend_trend000.zip` | S01 with no trend (C016), mean 330.0 | – | 0.12376 | – | −0.00077 vs S01: the trend is real |
| S03 | 2026-10-21 | `sub_blend_trend050.zip` | S01 with half trend (square root of the factor, C016), mean 346.1 | – | 0.12464 | – | +0.00011 vs S01, within public-board noise; half trend adopted as the more cautious choice |
| S04 | 2026-10-21 | `sub_trees_tuned.csv` | Equal blend of tuned LightGBM + XGBoost (C018), half trend (C019); no GLM/GAM | 0.1335 (trees blend) | 0.12477 | – | Uploaded by the user. +0.00013 vs S03 |
| S05 | 2026-10-21 | `sub_blend4_tuned.csv` | Equal blend of GLM + GAM (defaults, rescaled to half trend, C020) + tuned trees; half trend | – | 0.12464 | – | Uploaded by the user. −0.00013 vs S04: adding GLM/GAM at equal weight did not help on the public board |
| S06 | 2026-10-21 | `sub_blend4_weighted.zip` | Per-line blend weights from 2-fold OOF (C021): motor equal (searched weights failed the both-folds guard), buildings GAM 0.3 / LGBM 0.35 / XGB 0.35, contents GAM 0.3 / XGB 0.7; buildings GAM 5 knots; half trend | CV d2_cal weighted: buildings 0.1114, contents 0.0926 | 0.12579 | – | Submitted via CLI. +0.00102 vs S04, the largest gain so far. Leader 0.12996 |
| S07 | 2026-10-22 | `sub_v3_weighted.zip` | C026 feature policy + C027 inner early stopping + CatBoost; per-line weights: motor equal 0.2x5 (searched weights failed the both-folds guard), buildings GAM .2 / LGBM .25 / CAT .55, contents GAM .25 / LGBM .2 / XGB .35 / CAT .2; half trend | CV d2_cal (2-fold mean): motor equal 0.0919, buildings weighted 0.1144, contents weighted 0.0945 | 0.12851 | 11 | Submitted via CLI 2026-10-22 13:35. +0.00272 vs S06 (0.12579). Leader 0.12996 |
