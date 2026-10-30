# Video script: IAA AI Assisted Hackathon 2026 (version 3)

Length 5:00. Speech runs to 4:50 and the last 10 seconds are a buffer. Pace is about 135 words per minute. Word counts count the spoken column only; the bracketed rows for Oussama's AI-assisted workflow segment are placeholders and are not counted.

The recording is a screen capture of `docs/dashboard/index.html`, opened section by section from the left navigation: Overview, "Part 1: developed market", "Part 2: emerging market" and "AI-assisted workflow" (the arrow keys move between them). The two speakers record their own segments and the cuts fall at section changes. The last 10 seconds are a buffer.

## Opening, Oussama El Hadi, 0:00-0:13 (29 words)

On screen: the Overview section, the four score tiles and "Judging criteria and weights".

| Spoken (English) |
|---|
| Hello, we are Oussama El Hadi and Yichuan Lin. In five minutes: our approach, our results, our AI-assisted workflow, and what changed when the new market data was introduced. |

## Part 1: developed market, Oussama El Hadi, 0:13-1:38 (191 words)

On screen: the "Part 1: developed market" section. Rows 1 and 2 over "Data quality: ten documented issues"; rows 3 and 4 over "Target and development, Part 1"; rows 5, 7 and 8 over "Oussama's Part 1 pipeline" (row 8 points at its "Claim history" row, row 7 at its validation table); row 6 over "Blend recipe, Part 1: weight of each model family"; row 9 over "Standing on both boards".

| Spoken (English) |
|---|
| In Part 1, we predict the ultimate 2025 claim cost of every motor, buildings and contents policy. |
| We started with a data audit: exposure in mixed units, sums insured a thousand times too large and 600 duplicate claims, all fixed in code. |
| Claims stop at the end of 2024, so only 77 percent of 2024 motor cost is reported. Bodily injury is slowest: 14 percent at 30 days. |
| We develop it with a reporting curve from 2021 losses, and cap claims at the 99.5th percentile with a load, so no single claim sets a rating factor. |
| We built two pipelines. Oussama's models frequency and severity with a GLM selected by BIC and a spline GAM, then averages them with LightGBM and CatBoost. |
| Yichuan's fits five Tweedie models on pure premium, and blend weights must beat equal weights on both time folds. |
| Both validate on a later year, never a random split. |
| The leaderboard caught one gap validation missed: 2025 claim history is incomplete, so I count only claims known on 1 January. |
| S09, our ranked entry, scored 0.13689 privately. Blended 70/30 with Oussama's Blend4, it reaches 0.13718, as a late submission. |

## Part 2: emerging market, Yichuan Lin, 1:38-3:18 (226 words)

On screen: the "Part 2: emerging market" section. "What changed in the new market" and "New business share" for the opening lines and the data sentence; "Public D2 by submission" for the training sentences; "Six validation views" and "Adoption gates" for the validation sentences; "Blend recipe, Part 2: weight of each model family" and "Six-view gains in D2, last two adopted steps" for the gate sentences and the motor trend; "Limitations and uncertainty" for the last two sentences. Office premium has no panel; it is spoken over the recipe panel.

| Spoken (English) |
|---|
| Part 2 moved the same pipeline to the emerging market. |
| Five Tweedie models per line, blended by weights that must beat equal weights on every fold. |
| Three things changed. |
| First, the data: two training years instead of four, and 76 percent of the 2025 test rows are new business. |
| Second, training. |
| On emerging rows alone the GLM and GAM overfit. |
| So we pooled both markets: developed rows enter at a weight set by validation, market is a feature, and the level is re-balanced on emerging rows. |
| The public score rose from 0.14370 to 0.15193. |
| Third, validation. |
| Our audit found that validation rows were 92 percent renewals. |
| So every fold is now also scored re-weighted to the test mix, and on new-business rows only: six views. |
| A change must gain on average 0.002 and lose on none. |
| Under that gate, contents became a blend across two training runs, motor new business an equal blend of twelve candidates, and buildings 0.4 Oussama's GAM plus 0.6 CatBoost. |
| Motor moved to the market's own trend, 1.1606, after six public probes: the one decision the board led. |
| Office premium adds catastrophe and fixed costs and divides by one minus the loadings. |
| Our final file, P2-S15, scores 0.15807 on the public board, rank 18 of 38. |
| The private score is unknown, and new-business evidence is thin: 37 to 697 claims per view. |

## AI-assisted workflow, first half, Oussama El Hadi, 3:18-3:54 (0 words so far, about 80 to come)

Note: this segment is about 80 words, still to be written by Oussama. The third row is the one judges listen for.

On screen: the "AI-assisted workflow" section, "The loop on every change" with "Stages in the AI log".

| Spoken (English) |
|---|
| [Oussama: which parts of the Part 1 pipeline the AI wrote, one sentence] |
| [Oussama: how each piece was re-run on the 2024 validation before it entered Blend4] |
| [Oussama: one concrete case where the AI proposed X, the check showed Y, and he kept, changed or dropped it] |
| [Oussama: what the AI changed for him and what it did not decide] |

## AI-assisted workflow, second half, Yichuan Lin, 3:54-4:41 (107 words)

On screen: the "AI-assisted workflow" section. "The loop on every change", then "AI output rejected or corrected after checking" with "Independent audits of AI work", then "Human review register" and "Guardrails".

| Spoken (English) |
|---|
| Our AI-assisted workflow used Claude Code in three roles. |
| A coordinator planned each step and fixed the decision rule before any run. |
| Workers wrote a failing test first, then the code. |
| An auditor only read and reported, and never trained a model. |
| Every change went through one loop: propose, check, sign, log. |
| We rejected AI output when the check failed. |
| Per-peril reporting curves changed nothing, so they were dropped. |
| One blend passed our rule and we still overruled it, because its gain was in-sample. |
| The audit found our validation set was 92 percent renewals. |
| That finding became the test-mix gates. |
| Sixteen key decisions carry a human signature. |

## Closing, Yichuan Lin, 4:41-4:48 (14 words)

On screen: the Overview section again, with the score tiles in view.

| Spoken (English) |
|---|
| Code, tests, logs and the report are in our repository. Thank you for watching. |

Total spoken words: 567 written (4:12 at 135 words per minute), plus about 80 words still to come for Oussama's AI-assisted workflow segment: about 647 words and 4:48 in all, 2 seconds under the 4:50 target.
