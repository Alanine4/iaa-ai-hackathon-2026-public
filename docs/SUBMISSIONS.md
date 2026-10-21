# Kaggle submissions (Part 1)

Public leaderboard uses ~33% of the test rows; the final ranking uses the other ~67%.
Validation D2 = combined three-line Tweedie D2 on 2024 (`docs/validation_results.md`).

| # | Date | File | What it is | Validation D2 | Public score | Public rank | Notes |
|---|---|---|---|---|---|---|---|
| S01 | 2026-10-21 | `sub_blend.csv` | Equal-weight blend of GLM, GAM, LightGBM, XGBoost; capped at q99.5 + load (C014), rebalanced (C013), trend per line (C009: motor 1.035, buildings 1.199, contents 1.131) | 0.1336 | 0.12453 | 19 | Uploaded manually. Leader at the time: 0.12996 |
