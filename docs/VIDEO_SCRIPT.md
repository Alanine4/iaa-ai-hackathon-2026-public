# Video script: IAA AI Assisted Hackathon 2026 (version 4)

Length 5:00. There is no separate opening: Oussama's first sentence opens the video. Speech runs to 4:48 and the last 12 seconds up to 5:00 are a buffer. Pace is 135 words per minute, and each segment length is its word count divided by 135, rounded to the second. Word counts count the spoken column only.

The recording is a screen capture of `docs/dashboard/index.html`, opened section by section from the left navigation: "Part 1: developed market", "Part 2: emerging market", "AI-assisted workflow" and Overview (the arrow keys move between them). The two speakers record their own segments, Oussama the first one and Yichuan the other three, and the cuts fall at section changes. The last 12 seconds up to 5:00 are a buffer.

## Part 1: developed market, Oussama El Hadi, 0:00-2:03 (276 words)

On screen: the "Part 1: developed market" section. Rows 1 and 2 (intro) over the section heading; rows 3 to 8 (preparation) over "Data quality: ten documented issues" (exposure, customer service contacts, renewal decision) and "Target and development, Part 1" (cap, reporting curve), with the duplicates line (rows 4 and 5) on the "Prepare" row of "Oussama's Part 1 pipeline"; rows 9 to 13 (modelling, his four models) over "Oussama's Part 1 pipeline", then rows 14 and 15 (Yichuan's five models) over "Blend recipe, Part 1: weight of each model family"; rows 16 and 17 (validation) over the "Oussama's 2024 validation, Tweedie D2" table inside his panel; row 18 (result) over "Standing on both boards".

| Spoken (English) |
|---|
| In Part 1, we predicted the ultimate 2025 claim cost of every policy in three lines: motor, home buildings and home contents. |
| We worked in three stages, preparation, modelling and validation, with two independent pipelines: mine and my teammate Yichuan's. |
| First, data preparation. |
| We found exposure in mixed units, about 600 duplicate claims, and variables that 2025 policies will not have. |
| So we recomputed exposure from the policy dates and removed the duplicates. |
| We capped large claims at the 99.5th percentile, adding the excess back as a load. |
| We dropped customer service contacts, which are caused by claims, and the renewal decision, known only after the year ends. |
| And since recent claims are not all reported yet, we allowed for them with a reporting curve from 2021 losses. |
| Second, modelling. |
| After joining the customer, address and weather data, I fitted four models. |
| A GLM and a GAM split frequency and severity; the GLM's variables were selected step by step by BIC. |
| LightGBM and CatBoost model cost directly with a Tweedie loss and tuned parameters. |
| My final file averages the four with equal weights (Blend 4). |
| Yichuan fitted five models, the same families plus XGBoost, each modelling cost per unit of exposure directly. |
| His final file is a weighted blend, with weights kept only if they beat equal weights on both validation years. |
| Third, validation. |
| Both pipelines were fitted on earlier years and tested on a later one, not a random split, because pricing predicts the future. |
| Finally, a 70/30 average of our two final files scored 0.1372 on the private board as a late submission, above my file at 0.13526 and Yichuan's ranked entry at 0.13689. |

## Part 2: emerging market, Yichuan Lin, 2:03-3:35 (206 words)

On screen: the "Part 2: emerging market" section. Rows 1 to 3 over "What changed in the new market" and "New business share"; rows 4 to 7 (training) over "Public D2 by submission"; rows 8 to 11 (validation) over "Six validation views" and "Adoption gates"; row 12 (the gate result) and row 13 (the motor trend) over "Blend recipe, Part 2: weight of each model family" and "Six-view gains in D2, last two adopted steps"; row 14 over "Public D2 by submission" (the P2-S15 bar); row 15 over "Limitations and uncertainty".

| Spoken (English) |
|---|
| Part 2 is where our solution changed, when the new market data was introduced. |
| Same pipeline, three changes. |
| First, the data: two training years instead of four, and 76 percent of the 2025 test rows are new business, against 15 percent in Part 1. |
| Second, training. |
| On emerging rows alone the GLM and GAM overfit. |
| So we pooled both markets: developed rows enter at a weight set by validation, market is a feature, and the level is re-balanced on emerging rows. |
| The public score rose from 0.14370 to 0.15193. |
| Third, validation. |
| Our audit found that validation rows were 92 percent renewals. |
| So every fold is now also scored re-weighted to the test mix, and on new-business rows only: six views. |
| A change must gain on average 0.002 and lose on none. |
| Under that gate, contents became a blend across two training runs, motor new business an equal blend of twelve candidates, and buildings 0.4 Oussama's GAM plus 0.6 CatBoost. |
| Motor moved to the market's own trend after six public probes: the one decision the board led. |
| Our final file, P2-S15, scores 0.15807 on the public board, rank 18 of 38. |
| The private score is unknown, and new-business evidence is thin: 37 to 697 claims per view. |

## AI-assisted workflow, Yichuan Lin, 3:35-4:39 (144 words)

On screen: the "AI-assisted workflow" section. Rows 1 to 5 (the roles and the loop) over "The loop on every change"; rows 6 to 9 (rejected output) over "AI output rejected or corrected after checking"; rows 10 and 11 (the audit) over "Independent audits of AI work"; rows 12 and 13 (tests, data and rebuild) over "Guardrails" and "How a reviewer rebuilds each file"; row 14 over "Human review register".

| Spoken (English) |
|---|
| Our AI-assisted workflow used Claude Code in three roles. |
| A coordinator planned each step and fixed the decision rule before any run. |
| Workers wrote a failing test first, then the code. |
| An auditor only read and reported, and never trained a model. |
| Every change went through one loop: propose, check, sign, log. |
| We rejected AI output when the check failed. |
| Per-peril reporting curves changed nothing, so they were dropped. |
| A frequency-times-severity model lost to the best single model. |
| One blend passed our rule and we still overruled it, because its gain was in-sample. |
| The audit found our validation set was 92 percent renewals. |
| That finding became the test-mix gates. |
| Three hundred and forty-two tests were written before the code. |
| The raw data is read-only behind a SHA256 manifest, and every submitted file has a logged rebuild command. |
| Sixteen key decisions carry a human signature. |

## Closing, Yichuan Lin, 4:39-4:48 (20 words)

On screen: the Overview section, with the score tiles in view.

| Spoken (English) |
|---|
| We are Yichuan Lin and Oussama El Hadi. |
| Code, tests, logs and the report are in our repository. |
| Thank you. |

Total spoken words: 646 (Part 1 276, Part 2 206, AI-assisted workflow 144, Closing 20), 4:48 in all at 135 words per minute with each segment rounded to the second, leaving a 12-second buffer up to 5:00.
