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
| S08 | 2026-10-22 | `sub_v3_trend25bc.zip` | Probe for audit finding B4: S07 with buildings and contents at trend share 0.25 (motor unchanged at 0.5) | – | 0.12820 | – | −0.00031 vs S07: lower trend on buildings/contents hurts; half trend kept for all lines. B4 (residual-trend double counting) not supported by the public board |
| S09 | 2026-10-22 | `sub_v3_seeds.zip` | S07 recipe with seed-averaged tree refits (C029) | same OOF as S07 (refit only) | 0.12863 | – | Submitted via CLI 2026-10-22 18:19. +0.00012 vs S07; best so far. Final submission candidate |

# Kaggle submissions (Part 2, emerging market)

Competition: iaa-ai-assisted-hackathon-2026-part-2 (deadline 2026-10-31 03:59:59 local). Public leaderboard share not stated; treated like Part 1 (about a third of the test rows).
Validation = emerging time fold (train 2023 -> score 2024) and grouped 5-fold CV (`output/emerging/blend_weights_<line>.json`, C036).

| # | Date | File | What it is | Validation (d2_cal) | Public score | Public rank | Notes |
|---|---|---|---|---|---|---|---|
| P2-S01 | 2026-10-24 | `p2_emerging_only.zip` | Emerging data only (C036); five models per line, searched weights (motor cat .6 / xgb .2 / lgbm .1 / gam .1; buildings cat .65 / xgb .3 / gam .05; contents lgbm .4 / cat .4 / glm .15 / gam .05), guard passed on both schemes; developed AY2021 CDFs (C037); cap q99.5 + load; no trend; mean 346.1 | time fold: motor 0.0762, buildings 0.0410, contents 0.0779; grouped mean: 0.0677 / 0.0732 / 0.0934 | 0.14370 | 8 of 10 | Submitted via CLI 2026-10-24 18:29. Leader 0.16446; baseline (average cost) 0.07319 |
| P2-S02 | 2026-10-24 | `p2_emerging_half.zip` | P2-S01 re-expressed at the developed market's half trend to 2025 (motor 1.0175, buildings 1.0948, contents 1.0636; C037 option `developed-half`); mean 361.1 | same OOF as P2-S01 (level only) | 0.14460 | – | Pre-planned probe (decision D4). +0.00090 vs P2-S01: the 2025 level is above the 2023-24 training level (test rows are 76% new business, younger). `developed-half` adopted as the Part 2 default; no further trend probing (public-board chasing rule) |
| P2-S03 | 2026-10-24 | `p2_hybrid_contents_pooled.zip` | Contents from pooled training (C038: developed rows at weight 0.3, GAM left out for memory; glm .55 / lgbm .2 / xgb .25, guard passed); motor and buildings as P2-S02; developed-half trend; mean 360.3 | contents time fold 0.1072 (emerging-only 0.0779), grouped mean 0.1100 (0.0934) | 0.14693 | – | Submitted via CLI 2026-10-24 22:25. +0.00233 vs P2-S02 from one line (32% of the rows). Pooled training confirmed; buildings and motor pooled runs follow |
