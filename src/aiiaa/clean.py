"""Load-time corrections for policy tables. Raw parquet files are never modified.

Each fix references its change-log ID in docs/CHANGELOG.md.
"""
import pandas as pd

DAYS_PER_YEAR = 365  # the data uses a 365-day year: full 2024 rows end on 2024-12-30
SUM_INSURED_SCALE_THRESHOLD = 5_000_000  # largest genuine value is 953k; smallest scaled value is 8.2m
CONSTRUCTION_TYPES = {
    "brick": "brick",
    "stone": "stone",
    "non_standard": "non_standard",
    "t": "timber",  # code used in 2021-2022
    "w": "timber",  # code used in 2023-2024; same policies switch from T to W
}


def fix_exposure(df):
    """C001: recompute exposure from period dates; keep the stored value in exposure_raw."""
    out = df.copy()
    days = (pd.to_datetime(out["period_end"]) - pd.to_datetime(out["period_start"])).dt.days + 1
    out["exposure_raw"] = out["exposure"]
    out["exposure"] = days / DAYS_PER_YEAR
    return out


def fix_customer_id(df):
    """C002: restore leading zeros dropped from 8-digit customer ids."""
    out = df.copy()
    out["customer_id"] = out["customer_id"].astype(str).str.zfill(8)
    return out


def fix_sum_insured(df):
    """C003: divide sum_insured values scaled up by 1000 and flag the corrected rows."""
    out = df.copy()
    scaled = out["sum_insured"] > SUM_INSURED_SCALE_THRESHOLD
    out["sum_insured_scaled"] = scaled
    out.loc[scaled, "sum_insured"] = out.loc[scaled, "sum_insured"] / 1000
    return out


def fix_construction_type(df):
    """C004: lower-case construction_type and map the T/W codes to timber."""
    out = df.copy()
    lowered = out["construction_type"].str.lower()
    unknown = sorted(set(out.loc[~lowered.isin(CONSTRUCTION_TYPES) & lowered.notna(), "construction_type"]))
    if unknown:
        raise ValueError(f"Unknown construction_type codes: {unknown}")
    out["construction_type"] = lowered.map(CONSTRUCTION_TYPES)
    return out


def clean_policies(df):
    """Apply every fix whose columns exist in the given policy or test table."""
    out = fix_customer_id(fix_exposure(df))
    if "sum_insured" in out:
        out = fix_sum_insured(out)
    if "construction_type" in out:
        out = fix_construction_type(out)
    return out
