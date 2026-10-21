"""Profile every competition parquet file: shape, dtypes, missing, value ranges, categories, key integrity.

Usage: .venv\\Scripts\\python.exe scripts\\profile_data.py > docs\\data_profile.txt
"""
import glob
import os
import sys

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "iaa-ai-assisted-hackathon-2026-part-1")


def profile(name, df):
    print(f"\n{'=' * 100}\n{name}  rows={len(df):,}  cols={df.shape[1]}\n{'=' * 100}")
    rows = []
    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        info = {"col": col, "dtype": str(s.dtype), "missing%": round(s.isna().mean() * 100, 2), "nunique": nun}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info.update(min=s.min(), p50=s.median(), mean=round(s.mean(), 3), max=s.max())
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(min=s.min(), max=s.max())
        rows.append(info)
    print(pd.DataFrame(rows).to_string(index=False))
    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        if (not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or nun <= 12) and nun <= 30:
            vc = s.value_counts(dropna=False)
            print(f"  [{col}] " + ", ".join(f"{k}: {v:,}" for k, v in vc.items()))
        elif not pd.api.types.is_numeric_dtype(s):
            print(f"  [{col}] {nun:,} distinct, e.g. {list(s.dropna().unique()[:5])}")


def main():
    t = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "*.parquet")))}
    for name, df in t.items():
        profile(name, df)

    print(f"\n{'=' * 100}\nKEY INTEGRITY\n{'=' * 100}")
    cust, addr = set(t["customers"]["customer_id"]), set(t["addresses"]["address_id"])
    for name in ["policies_motor", "policies_buildings", "policies_contents", "test_motor", "test_buildings", "test_contents"]:
        df = t[name]
        print(f"{name}: row_id unique={df['row_id'].is_unique}, customer_id missing in customers={(~df['customer_id'].isin(cust)).sum()}, "
              f"address_id missing in addresses={(~df['address_id'].isin(addr)).sum()}, policies={df['policy_id'].nunique():,}, "
              f"rows by start year={df['period_start'].astype(str).str[:4].value_counts().sort_index().to_dict()}, "
              f"exposure sum={df['exposure'].sum():,.0f}")
    for line, train, test in [("motor", "policies_motor", "test_motor"), ("buildings", "policies_buildings", "test_buildings"),
                              ("contents", "policies_contents", "test_contents")]:
        overlap = t[test]["policy_id"].isin(set(t[train]["policy_id"])).mean()
        print(f"{line}: share of test policy_id seen in training = {overlap:.1%}")
    c = t["claims"]
    all_pol = pd.concat([t[n]["policy_id"] for n in ["policies_motor", "policies_buildings", "policies_contents"]])
    print(f"claims: claim_id unique={c['claim_id'].is_unique}, policy_id not in any training policy={(~c['policy_id'].isin(set(all_pol))).sum()}")
    print("claims by line x loss year:\n", pd.crosstab(c["line"], c["loss_date"].astype(str).str[:4]))
    print("claims by line x status:\n", pd.crosstab(c["line"], c["status"]))
    print("claims by line x peril:\n", pd.crosstab(c["peril"], c["line"]))
    closed = c[c["status"] == "closed"]
    print(f"closed claims with incurred != paid: {(closed['incurred_to_date'] - closed['paid_to_date']).abs().gt(0.01).sum():,}")
    rep = c[c["status"] == "repudiated"]
    print(f"repudiated: paid sum={rep['paid_to_date'].sum():,.0f}, incurred sum={rep['incurred_to_date'].sum():,.0f}")
    print("incurred - recoveries < 0 count:", ((c["incurred_to_date"] - c["recoveries"]) < 0).sum())
    lag = (pd.to_datetime(c["report_date"]) - pd.to_datetime(c["loss_date"])).dt.days
    print("report lag days by line:\n", lag.groupby(c["line"]).describe(percentiles=[.5, .9, .99]))
    print("open share by loss year:\n", c.assign(y=c["loss_date"].astype(str).str[:4]).groupby(["line", "y"])["status"].apply(lambda s: round((s == "open").mean(), 3)).unstack())
    print("incurred-recoveries by line x loss year (sum):\n",
          c[c["status"] != "repudiated"].assign(y=c["loss_date"].astype(str).str[:4], net=c["incurred_to_date"] - c["recoveries"])
          .pivot_table(index="line", columns="y", values="net", aggfunc="sum").round(0))

    s = t["sample_submission"]
    test_ids = set(pd.concat([t[n]["row_id"] for n in ["test_motor", "test_buildings", "test_contents"]]))
    print(f"\nsample_submission rows={len(s):,}, ids match test exactly={set(s['row_id']) == test_ids}, prediction values={s['prediction'].unique()[:5]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
