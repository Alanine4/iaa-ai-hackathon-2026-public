"""Office premium build-up for the 20 Part 2 reference policies (report table generator).

Report table generator for section 8 of docs/REPORT_DRAFT.md. Read-only: it reads
  - work/p2_c047.csv (P2-S13, the final Part 2 candidate; one row total per row_id),
  - data/iaa-ai-assisted-hackathon-2026-part-2/emerging/reference_policies.parquet,
  - data/iaa-ai-assisted-hackathon-2026-part-2/emerging/expenses.parquet (emerging column),
and prints the section-8 table as markdown, a second table that splits P into its share-of-premium
items (report appendix), and the checks quoted in the report. Nothing is written to disk.

Team conventions (docs/REPORT_DRAFT.md section 8):
  R   = predicted risk premium of the row from P2-S13 (already a row total, not annualised)
  Cat = cat_reinsurance x R on buildings and contents, 0 on motor ("share of property risk premium")
  F   = fixed_expense_per_policy, charged in full per policy (not pro-rated by exposure)
  k   = variable_expense + commission of the row's channel + premium_tax + target_profit_margin,
        all shares of the office premium
  P   = (R + Cat + F) / (1 - k)
No minimum or maximum premium. written_premium (current tariff) is shown for comparison only.

Run: .venv\\Scripts\\python.exe -I scripts\\part2_office_premium.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMERGING = ROOT / "data" / "iaa-ai-assisted-hackathon-2026-part-2" / "emerging"
PREDICTIONS = ROOT / "work" / "p2_c047.csv"

LINE_ORDER = ["motor", "buildings", "contents"]
PROPERTY_LINES = {"buildings", "contents"}
DAYS_PER_YEAR = 365  # C001 basis


def expense_values():
    """Emerging-market expense items as a dict item -> value (NaN where the file is blank)."""
    exp = pd.read_parquet(EMERGING / "expenses.parquet")
    return dict(zip(exp["item"], exp["emerging"]))


def commission_by_channel(e):
    """Commission share per channel; a blank value stays NaN so any use of it fails loudly."""
    return {
        "broker": e["commission_broker"],
        "direct": e["commission_direct"],
        "comparison_site": e["commission_comparison_site"],
    }


def build_table():
    e = expense_values()
    commission = commission_by_channel(e)

    # Identifier columns are read as text (project convention).
    ref = pd.read_parquet(EMERGING / "reference_policies.parquet")
    pred = pd.read_csv(PREDICTIONS, dtype={"row_id": str})
    assert pred["row_id"].is_unique
    t = ref.merge(pred, on="row_id", how="left", validate="one_to_one")
    assert len(t) == 20 and t["prediction"].notna().all(), "every reference row needs a P2-S13 prediction"

    # The stored exposure must equal the C001 recomputation from the period dates (file holds 4 decimals).
    days = (pd.to_datetime(t["period_end"]) - pd.to_datetime(t["period_start"])).dt.days + 1
    assert np.allclose(t["exposure"], days / DAYS_PER_YEAR, atol=1e-4)

    # The emerging comparison-site commission is blank; none of the 20 rows may depend on it.
    t["commission"] = t["channel"].map(commission)
    assert t["commission"].notna().all(), "a reference row uses a channel without an emerging commission"

    t["R"] = t["prediction"]
    t["Cat"] = np.where(t["line"].isin(PROPERTY_LINES), e["cat_reinsurance"] * t["R"], 0.0)
    t["F"] = e["fixed_expense_per_policy"]
    t["k"] = e["variable_expense"] + t["commission"] + e["premium_tax"] + e["target_profit_margin"]
    t["divisor"] = 1.0 - t["k"]
    t["P"] = (t["R"] + t["Cat"] + t["F"]) / t["divisor"]

    # Build-up identity: risk premium + cat + fixed + every share-of-premium item = office premium.
    parts = t["R"] + t["Cat"] + t["F"] + t["k"] * t["P"]
    assert np.allclose(parts, t["P"])

    # Sensitivity only: fixed expense pro-rated by exposure.
    t["P_prorated_F"] = (t["R"] + t["Cat"] + t["F"] * t["exposure"]) / t["divisor"]

    t["line"] = pd.Categorical(t["line"], LINE_ORDER, ordered=True)
    return t.sort_values(["line", "row_id"]).reset_index(drop=True), e


def money(x):
    return f"{x:,.2f}"


def print_table(t):
    # NB = new business, WP = written premium (current tariff, comparison only).
    head = ["row_id", "line", "channel", "NB", "exposure", "R", "Cat", "F", "1 - k", "P", "WP"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * 4 + ["---:"] * 7) + "|")
    for r in t.itertuples():
        cells = [r.row_id, str(r.line), r.channel, "yes" if r.new_business else "no", f"{r.exposure:.4f}",
                 money(r.R), money(r.Cat), money(r.F), f"{r.divisor:.2f}", money(r.P), money(r.written_premium)]
        print("| " + " | ".join(cells) + " |")


def print_split(t, e):
    """Second table (report appendix): each share-of-premium item in CU, so every row adds up to P."""
    head = ["row_id", "R + Cat + F", "variable expense", "commission", "premium tax", "profit", "P"]
    print("\n| " + " | ".join(head) + " |")
    print("|---|" + "|".join(["---:"] * 6) + "|")
    for r in t.itertuples():
        cells = [r.row_id, money(r.R + r.Cat + r.F), money(e["variable_expense"] * r.P), money(r.commission * r.P),
                 money(e["premium_tax"] * r.P), money(e["target_profit_margin"] * r.P), money(r.P)]
        print("| " + " | ".join(cells) + " |")


def print_checks(t, e):
    print("\nExpense items used (emerging column):")
    for item, value in e.items():
        print(f"  {item}: {value}")
    print("Divisor 1 - k by channel:", t.groupby("channel")["divisor"].first().round(4).to_dict())

    print("\nSums and means over the 20 rows:")
    for col in ["R", "Cat", "F", "P", "written_premium"]:
        print(f"  {col}: sum {money(t[col].sum())}, mean {money(t[col].mean())}")
    print("  check: sum(R + Cat + F) / sum(P) =", round((t["R"] + t["Cat"] + t["F"]).sum() / t["P"].sum(), 4))

    print("\nBy line (rows, sum R, sum P, sum written premium):")
    g = t.groupby("line", observed=True).agg(rows=("row_id", "size"), R=("R", "sum"), P=("P", "sum"),
                                             wp=("written_premium", "sum"))
    print(g.round(2).to_string())

    ratio = t["P"] / t["written_premium"]
    print("\nComparison with written premium:")
    print(f"  rows with R > written premium: {(t['R'] > t['written_premium']).sum()} of {len(t)}",
          t.loc[t["R"] > t["written_premium"], "line"].astype(str).value_counts().to_dict())
    print("  of those, new business:", int(t.loc[t["R"] > t["written_premium"], "new_business"].sum()))
    print(f"  rows with P > written premium: {(t['P'] > t['written_premium']).sum()} of {len(t)}")
    floor = (e["fixed_expense_per_policy"] / t.groupby("channel")["divisor"].first()).round(2).to_dict()
    print("  P at R = 0 (fixed expense only) by channel:", floor)
    print(f"  P / written premium: min {ratio.min():.2f} ({t.loc[ratio.idxmin(), 'row_id']}), "
          f"max {ratio.max():.2f} ({t.loc[ratio.idxmax(), 'row_id']}), median {ratio.median():.2f}")
    print("  share of P that is F + loadings (1 - R / P): "
          f"min {(1 - t['R'] / t['P']).min():.2f}, max {(1 - t['R'] / t['P']).max():.2f}")

    print("\nShort rows (exposure < 0.5): F in full vs F pro-rated by exposure (sensitivity only):")
    short = t[t["exposure"] < 0.5]
    for r in short.itertuples():
        print(f"  {r.row_id} {r.line} exposure {r.exposure:.4f}: R {money(r.R)}, P {money(r.P)} "
              f"vs {money(r.P_prorated_F)} pro-rated (F {money(r.F * r.exposure)}); "
              f"written premium {money(r.written_premium)}; P / written {r.P / r.written_premium:.2f}")


if __name__ == "__main__":
    table, expenses = build_table()
    print_table(table)
    print_split(table, expenses)
    print_checks(table, expenses)
