# Video script: IAA AI Assisted Hackathon 2026

Length 5:00. Speech runs to 4:50 and the last 10 seconds are a buffer. Pace is about 135 words per minute. Word counts count the spoken column only (left column).

## Segment 1, 0:00-0:30: who we are and the task (78 words)

On screen: title card, then `README.md` section "What the competition asks".

| Spoken (English) |
|---|
| Hello, we are team Yichuan Lin: Yichuan Lin and Oussama El Hadi, and this is our entry to the IAA AI Assisted Hackathon 2026. |
| The task, in one sentence: predict the final claim cost of every 2025 policy row, for motor, home buildings and home contents. |
| There are two markets. Part 1 is a developed market with four years of history, and Part 2 is a smaller emerging market. |
| Entries are scored with Tweedie D squared, power 1.5. |

## Segment 2, 0:30-1:30: approach (136 words)

On screen: `docs/REPORT_DRAFT.md` sections 2 and 3 (target construction, model list), ending on the adoption-rule sentence.

| Spoken (English) |
|---|
| First, the target. We match each claim to the policy row that contains its loss date, and we subtract recoveries. |
| Recent claims are still being reported, so we develop them with a reporting-lag curve for each line. |
| For motor bodily injury, only 14 percent of the cost is reported after 30 days. |
| Each claim is capped at the 99.5th percentile, and a load adds the capped excess back. |
| Then we fit five Tweedie models: a GLM, a GAM, LightGBM, XGBoost and CatBoost, and blend them with weights chosen per line. |
| Validation is by time, with two expanding folds, never at random. |
| A change is adopted only if the mean gain is at least 0.002 and it loses on neither fold. |
| We fixed that rule before running the experiments, and the blend weight check is now code. |

## Segment 3, 1:30-2:30: AI-assisted workflow (127 words)

On screen: `docs/AI_LOG.md` (rows A09 and A18), a `docs/CHANGELOG.md` row with its code references, and the sign-off note in `docs/HUMAN_REVIEW.md`.

| Spoken (English) |
|---|
| We used Claude Code in three roles. |
| A coordinator plans and decides. Workers implement, and write the tests first. An independent auditor only reads and reports. |
| The suite now has 341 tests. The raw data is never edited, and every fix lives in code. |
| Every change has a CHANGELOG number, and the code comments cite it. |
| Every AI step is logged, with how we checked it. Our team lead signed off sixteen key decisions by 30 October. |
| We also reject AI output. The coordinator designed per-peril reporting curves, C022. The worker implementing them showed they change nothing. |
| Four other ideas, C023, C024, C031 and C032, were rejected on validation. |
| The audits found real issues: B1 to B8 in Part 1, and P2-F1 to F10 in Part 2. |

## Segment 4, 2:30-3:45: what changed for the new market (165 words)

On screen: `docs/REPORT_DRAFT.md` section 7.1 and the section 7.4 table (emerging only against pooled), then the P2-F1 row of `docs/AUDIT_PART2.md`, then `docs/SUBMISSIONS.md` rows P2-S10 and P2-S13.

| Spoken (English) |
|---|
| Then the second market arrived. It has about one tenth of the data, covering only two years. |
| 76 percent of its test rows are new business, against 15 percent before. |
| Motor frequency is 34 percent higher, and severity is 42 percent lower. |
| So each market became a configuration, with one shared schema, and we borrowed the developed reporting curves, because two years give no tail. |
| Pooling the developed rows into training, at a weight of 0.3, lifted contents from 0.078 to 0.107 on the time fold. |
| Then our audit found that validation was 92 percent renewals, while the test is mostly new business. We made a test-mix check part of the gate. Weighted to the test mix, validation matched the public board. |
| Motor level looked low, so we moved to the emerging market's own trend, 1.16. For motor we also added a separate new-business blend, which passed all six validation views. |
| We rejected interaction terms, new-business weighted training, and re-tuning, because none passed the gate. |

## Segment 5, 3:45-4:35: results (119 words)

On screen: the Kaggle leaderboards for Part 1 and Part 2, the Part 1 post-deadline block of `docs/SUBMISSIONS.md`, then the report's office premium formula (`docs/REPORT_DRAFT.md` section 7.5) and the build-up table for the 20 reference policies.

| Spoken (English) |
|---|
| In Part 1, our ranked final scored 0.13689 on the private board, rank 12 of 51. The blend we present, a late submission and not ranked, scored 0.13718. |
| In Part 2, the public score rose from 0.14370 with emerging data only, to 0.15193 with pooled training, and now 0.15807 with seed-averaged trees. At recording time we rank 18th of 38 teams, and the leader has 0.176. |
| For the 20 reference policies, office premium is risk premium, plus catastrophe reinsurance for property and a fixed expense of 85, divided by one minus the loadings for expense, commission, tax and profit. |
| Our reporting curves are borrowed and the new-business evidence is thin. Part of the motor level came from leaderboard probes. |

## Segment 6, 4:35-4:50: closing (39 words)

On screen: the repository file tree: `README.md`, `reproduce.ps1`, `tests/`, `docs/CHANGELOG.md`.

| Spoken (English) |
|---|
| Code, tests and the change log are in the repository. |
| One script rebuilds Part 1 from the raw data, and a market setting runs the emerging data. |
| Next, we would test a developed-model offset, an idea from our audit. |

Total spoken words: 664 (about 4:55 at 135 words per minute, 5 seconds over the 4:50 target; trim about 10 words, segment 5 is the candidate).
