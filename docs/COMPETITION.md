# Competition Summary

Archived originals: `docs/references/kaggle_overview_page.md`, `kaggle_rules_page.md`, `kaggle_data_page.md`. Where this page conflicts with the originals, the originals prevail.

## Basic Information

- Name: IAA AI Assisted Hackathon 2026 – Part 1, organised by the International Actuarial Association, invited participants only.
- Link: https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-1
- Part 1 deadline: on 2026-10-21 the page showed 2 days remaining, i.e. about 2026-10-23. The date on the Rules page is a placeholder; follow the countdown on the page and mind the time zone (UTC).
- Role: you are a member of the pricing team at a developed-market insurer, building claim prediction models for motor, buildings and contents, and demonstrating an AI-assisted actuarial workflow. The model itself can be a GLM, machine learning, or a combination of both.

## Two-Part Process

1. **Part 1 / Week 1 (current)**: use the 2021–2024 synthetic data to predict the ultimate claim cost of every row in 2025, and submit to the Kaggle leaderboard.
2. **Part 2 / Week 2**: when Part 1 closes, "emerging market" data is released on a separate Kaggle page. Explain what changed, how the model was adjusted, and which uncertainties remain. The team must be the same for both parts.
3. **Joint submission after Part 2 closes** (via a form; the link will be announced later):
   - Team name and member list
   - Prediction files for Week 1 and Week 2
   - Video presentation, only the first 5 minutes are assessed. Content: method, results, AI workflow, and the changes made after switching to the new market
   - Written report, only the first 5 pages are assessed. Content: modelling approach, key decisions, use of AI, assumptions, limitations, adaptation to the new market, controls
   - Code (scripts or notebooks), under version control, understandable and reproducible by others

## Scoring (total score determined after Part 2 closes)

| Category | Weight | How it is assessed |
|---|---|---|
| Pricing performance and adaptability | 40% | Part 1 and Part 2 private leaderboard results each count for half, plus the judges' assessment of modelling choices, market adaptation, and handling of data quality and uncertainty |
| Effective use of AI | 25% | Whether AI runs through EDA, feature engineering, modelling, diagnostics and testing; whether it improves quality or efficiency; whether AI output was rigorously checked |
| Trust, validation and governance | 20% | Validation checks, automated tests, model diagnostics; governance controls, reproducibility, human review and escalation process |
| Communication and documentation | 15% | Clarity of the video and report; repository structure and version control |

The organiser first screens the deliverables with AI tools, and the judges make the final decision. Part 1 leaderboard rank alone does not decide the prizes.

## Submission Rules

- Format: one CSV with header `row_id,prediction`, covering all 626,639 rows of the three test tables, predictions ≥ 0.
- Metric: Tweedie D² (power=1.5).
- At most 10 submissions per day. You may select 1 Final Submission; if you do not, the system uses the best submission on the public leaderboard.
- The final ranking uses the private leaderboard. Which rows the public and private leaderboards use is not disclosed, so do not overfit to the public score.

## Key Rules

- Teams have at most 3 members. After the Part 1 team-merger deadline, members cannot be changed.
- **Giving the competition data to AI tools (including hosted large language models) is allowed**, for competition purposes only (Rules 2.5.a).
- Data must not be distributed to non-participants. Code must not be shared privately outside the team; it may only be shared publicly on this competition's Kaggle forum or in a notebook.
- External data and tools must be publicly available to all at low cost. Subscription fees for general AI assistants are acceptable.
- Open-source code used must carry an OSI-approved licence that does not restrict commercial use.
- Test set records must not be labelled or judged manually.
- Winners must hand over reproducible training and inference code and environment instructions, and license them under an open-source licence.
- Prizes: CAD 500 / 100 / 50, paid only to members or students of IAA full or associate member associations (MA).
