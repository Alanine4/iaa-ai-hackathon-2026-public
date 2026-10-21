# AI-assisted workflow log

Records where AI (Claude Code, model claude-opus-5-5) contributed, what the human reviewed, and how each AI output was verified.
Status: accepted / modified / rejected.

| # | Date | Stage | What AI did | How it was verified | Status |
|---|---|---|---|---|---|
| A01 | 2026-10-21 | Setup | Read Kaggle Overview, Data and Rules pages; archived them to `docs/references/` | Human compared archived text against the live pages | accepted |
| A02 | 2026-10-21 | EDA | Wrote `scripts/profile_data.py` to profile all 11 files (types, missing rates, ranges, key integrity) | Output in `docs/data_profile.txt`; row counts cross-checked with the sample submission (626,639 rows match exactly) | accepted |
| A04 | 2026-10-21 | Cleaning rules | Investigated Q1/Q3/Q4 and proposed rules C001–C004 (e.g. inferred T/W = timber from year split and policy transitions) | Evidence script `scripts/investigate_q1_q3_q4.py`; rules written as failing tests first (`tests/test_clean.py`), then implemented; human to review C003 threshold and C004 mapping | accepted, pending human review |
| A05 | 2026-10-21 | Target & development | Designed the target (claim matching, repudiated excluded) and the reporting-lag development method C006 | Unit tests (`tests/test_target.py`, `tests/test_development.py`); full-data reconciliation shows 0 difference between claims file and row targets + unmatched list; developed motor 2024 pure premium checked against 2021–2023 | accepted; buildings 2024 developed level (481 vs ~458 trend) flagged for review |
| A03 | 2026-10-21 | Data quality | Flagged issues Q1–Q10 and wrote `scripts/check_quality.py` to confirm each | Each issue backed by counts in `docs/data_quality_checks.txt`; human reviewed the plain-language explanation of every item | accepted |
