# AIIAA: IAA AI Assisted Hackathon 2026

Actuarial pricing competition: predict the final claim cost of every policy row in 2025 for motor, buildings and contents (Tweedie D², power 1.5).

## Read at the start of every session
1. `TODO.md`: the progress checklist. When an item is done, change `[ ]` to `[x]` immediately and write the date and output files.
2. `docs/COMPETITION.md`: competition process, scoring, rules.
3. `docs/DATA_GUIDE.md`: data structure, target definition, data quality issues Q1–Q10.
4. `docs/references/`: archive of the original Kaggle web page text; it is the authoritative source. When it conflicts with other documents, this folder wins.

## Conventions
- For Python use only `.venv\Scripts\python.exe`; do not use R.
- Raw data is in `data/`, committed as a read-only backup in the private repository, with a SHA256 manifest and an integrity test, and is never modified (C007). All corrections are made in code.
- Reusable logic goes in `src/aiiaa/` (clean, target, development); write the tests first and put them in `tests/`; run the tests with: `.venv\Scripts\python.exe -m pytest -q`
- One-off analysis scripts go in `scripts/`, analysis output goes in `docs/`, and generated intermediate data goes in `output/` (not in git).
- Every change must be recorded in `docs/CHANGELOG.md` (numbered C0xx), and the number must be cited in the code comments. Code and comments are always in English.
- When reading identifier columns (customer_id, address_id, policy_id, row_id, claim_id), always treat them as text.
- Every step of AI-assisted work is recorded in `docs/AI_LOG.md`, stating how a human checked it.
