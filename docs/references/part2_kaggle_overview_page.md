# Kaggle Part 2 Overview 页原文（存档）

来源：https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-2/overview
抓取日期：2026-10-24。页面显示：Start a day ago，Close 7 days to go。参与情况：27 Entrants · 7 Participants · 7 Teams · 31 Submissions。
Challenge 段、Evaluation 表、Prizes 与 Part 1 相同（见 `kaggle_overview_page.md`），下面只记 Part 2 新增或改动的原文。

---

## Week 2: The New Market

Your insurer is now expanding into an emerging market. It has written business there for two years (2023 and 2024), and at the start of 2025 it takes on a large portfolio from a broker, so most 2025 policies are new business.

The data is in the emerging folder of this competition's dataset. The developed folder contains the Part 1 data for reference. Predict the 2025 claim cost for every row in the three emerging-market test files.

The new market will not necessarily behave like the one you started with: there is less data, and the customers, climate, regulation and costs are different. We want to see how your team identifies what has changed, decides what to carry over from Part 1, and communicates any remaining uncertainty.

## Submissions (unchanged list, with one Part 2 note)

Each team should provide at the end of 2 weeks (in a final submission form - link to be provided): team name and members; Week 1's prediction file; Week 2's prediction file; a short video demo (first 5 minutes graded) explaining approach, results, AI-assisted workflow, and how the solution changed when the new market data was introduced; a concise write-up (first 5 pages graded) covering modelling approach, major decisions, use of AI, assumptions, limitations, adaptation to the new market, and guardrails; the version-controlled code.

Submission file: one CSV, header `row_id,prediction`, predictions zero or above.

**For Week 2 leaderboard, please submit the prediction for Week 2 emerging market test data only.**

Accuracy: Tweedie D² (power 1.5) relative to a constant-rate baseline, on the ultimate cost of claims with a loss date in the row's period, net of excess and recoveries, excluding declined claims.

Prizes are decided after Part 2 by the judging panel, based on the evaluation criteria, not on either leaderboard alone.

## Citation

nightowl and ShaunBennet. IAA AI Assisted Hackathon 2026 - Part 2. https://www.kaggle.com/competitions/iaa-ai-assisted-hackathon-2026-part-2, 2026. Kaggle.
