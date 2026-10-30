# Video script: IAA AI Assisted Hackathon 2026

Length 5:00. Speech runs to 4:50 and the last 10 seconds are a buffer. Pace is about 135 words per minute. Word counts count the spoken column only. The recording is a screen capture of `docs/dashboard/index.html`, opened section by section from the "Judging criteria" navigation on the left and scrolled within each section.

## Overview, 0:00-0:25 (59 words)

On screen: Overview. The Part 1 and Part 2 score tiles, "Automated tests" and "Change log" at the top, then "Judging criteria and weights", "What changed in the new market" and "Standing on both boards".

| Spoken (English) |
|---|
| Hello, we are Yichuan Lin and Oussama El Hadi. |
| This board follows the four judging criteria. |
| In the new market, 76 percent of test rows are new business, against 15 percent before. |
| Our ranked Part 1 entry scored 0.13689 on the private board, rank 12 of 51. |
| Part 2 scored 0.15807 on the public board, rank 18 of 38. |

## 1 Pricing Performance & Adaptation (40%), 0:25-2:20 (256 words)

On screen: the Pricing Performance & Adaptation section. "Public D2 by submission", "New business share", "Limitations and uncertainty", "Blend recipes: weight of each model family", then "Data quality: ten documented issues". The office premium sentence is spoken over the last panel.

| Spoken (English) |
|---|
| In Part 2, the public score rose from 0.14370 with emerging rows only to 0.15193 with pooled training. |
| Motor and buildings blends then lifted it to 0.15781, and three-seed trees to 0.15807. |
| New business is 7.8 percent of motor training exposure but 76.9 percent of the 2025 test. |
| Validation rows were 92 percent renewals, so adoption gates now include test-mix views. |
| Limits: new-business evidence is thin, at 37 to 697 claims per view, and the Part 2 private score is unknown. |
| Yichuan's approach blends five Tweedie models: GLM, GAM, LightGBM, XGBoost and CatBoost. Blend weights must beat equal weights on both time folds. |
| Oussama's approach is a frequency and severity GLM with Blend4, an equal average of four models. |
| The Part 1 blend of 0.7 S09 and 0.3 Blend4 scored 0.13718 privately: a late submission, not ranked. |
| In Part 2 we used both. Oussama's GAM takes 0.4 of buildings, and motor new business is an equal blend of twelve candidates. |
| Developed rows train at weight 0.3 or 1.0, and reporting curves are borrowed, because two years give no tail. |
| Motor moved to the market's own full trend, 1.1606, after six public probes: the one decision the board led. |
| Claims match to the policy row containing the loss date, net of recoveries. |
| Motor bodily injury is only 14 percent reported at 30 days, so claims are developed with a reporting-lag curve, then capped at the 99.5th percentile. |
| For 20 reference policies, office premium is risk premium plus catastrophe and fixed costs, divided by one minus the loadings. |

## 2 Effective Use of AI (25%), 2:20-3:25 (148 words)

On screen: the Effective Use of AI section. "The loop on every change" with "Stages in the AI log" below it, then "AI output rejected or corrected after checking" with "Independent audits of AI work", then "Change log outcomes, 61 entries" and "Human review register".

| Spoken (English) |
|---|
| We used Claude Code in three roles. |
| A coordinator plans and decides. Workers write the test first, then the code. Auditors only read and report. |
| Every change follows one loop: propose, check, sign, log. The AI log holds 34 steps. |
| We reject AI output too. Per-peril curves, C022, equal the pooled curve. A frequency times severity model, C023, fell below the best single model. |
| Target encoding, C031, and re-tuning, C032, missed the 0.002 rule. |
| A market interaction blend passed at plus 0.0024, but we overruled it, C042, because of an unvalidated 21 percent new-business shift. |
| Read-only audits raised B1 to B8 and P2-F1 to P2-F10. P2-F1 found validation was mostly renewals, which led to the test-mix views. |
| Of 61 change log entries, 43 were adopted, 3 partly, 11 evaluated but not adopted and 4 rejected or withdrawn. |
| Our team lead signed off sixteen key decisions by 30 October. |

## 3 Trust, Validation & Guardrails (20%), 3:25-4:15 (115 words)

On screen: the Trust, Validation & Guardrails section. "Six-view gains in D2, last two adopted steps", "Six validation views", "Adoption gates" and "Guardrails".

| Spoken (English) |
|---|
| The last two adopted steps gained on all six views and on the public board: buildings 0.00204, motor 0.00103. |
| Only candidates that gained on all six improved the public score. |
| A time fold and five policy-grouped folds are each scored raw, re-weighted to the 77 percent new-business test share, and on new-business rows only. |
| A change needs a mean gain of at least 0.002 and no regression. |
| Seed averaging cannot be tested on folds, so we set its keep rule before the score. |
| We wrote 341 tests before the code, and the raw data is read-only with a SHA256 check. |
| The rebuild script reproduces S09 and P2-S13. Logged commands reproduced P2-S14 to within one billionth. |

## 4 Communication & Documentation (15%), 4:15-4:45 (58 words)

On screen: the Communication & Documentation section. "Documentation set", "Repository layout", "Submissions logged" and "How a reviewer rebuilds each file".

| Spoken (English) |
|---|
| Documentation sits beside the code: README, data guide, change log, AI log, review register, report and two audits. |
| The repository holds code, tests, docs, data, the submitted files and Oussama's Part 1 pipeline. |
| We logged nine Part 1 submissions plus three unranked, and fifteen in Part 2. |
| For each file, a reviewer finds the command that rebuilds it. |

## Closing, 4:45-4:50 (12 words)

On screen: Overview again, with the Part 1 and Part 2 score tiles in view.

| Spoken (English) |
|---|
| Code, tests and logs are in our repository. Thank you for watching. |

Total spoken words: 648 (about 4:48 at 135 words per minute, 2 seconds under the 4:50 target).
