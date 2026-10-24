"""Print the Part 2 expenses table and the 20 reference policies (read-only).

One-off helper for docs/OFFICE_PREMIUM_DRAFT.md. Reads only two small files from
data/iaa-ai-assisted-hackathon-2026-part-2/emerging/ and writes nothing to disk.
Run with the project venv Python and -I: scripts/part2_expenses.py
"""
from pathlib import Path

import pandas as pd

EMERGING = Path(__file__).resolve().parents[1] / "data" / "iaa-ai-assisted-hackathon-2026-part-2" / "emerging"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)

# Identifier columns are read as text per project convention.
exp = pd.read_parquet(EMERGING / "expenses.parquet")
print("expenses: rows=%d cols=%s" % (len(exp), list(exp.columns)))
print(exp.dtypes.to_string())
print(exp.to_string())

ref = pd.read_parquet(EMERGING / "reference_policies.parquet")
print("\nreference_policies: rows=%d cols=%d" % (len(ref), ref.shape[1]))
ref["prefix"] = ref["row_id"].str[0]
print("\nline x row_id prefix:")
print(pd.crosstab(ref["line"], ref["prefix"]).to_string())
print("\nchannel counts:\n" + ref["channel"].value_counts().to_string())
print("\nnew_business counts:\n" + ref["new_business"].value_counts().to_string())

key = ["row_id", "policy_id", "line", "channel", "new_business", "period_start", "period_end",
       "exposure", "written_premium", "discount_applied", "voluntary_excess", "compulsory_excess"]
print("\nkey columns:")
print(ref[key].sort_values(["line", "row_id"]).to_string(index=False))
