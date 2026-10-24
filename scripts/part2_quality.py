r"""C033: data-quality checks on a Part 2 market folder (Part 1 issues Q1-Q10 re-checked, plus new checks).

Usage: .venv\Scripts\python.exe -I -u scripts\part2_quality.py data\iaa-ai-assisted-hackathon-2026-part-2\emerging > docs\part2_emerging_quality.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from aiiaa.target import large_loss_cap, match_claims, net_cost  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

LINES = {"motor": ("policies_motor", "test_motor"), "buildings": ("policies_buildings", "test_buildings"),
         "contents": ("policies_contents", "test_contents")}
AGE_BAND_MAX_AGE = {"17-24": 24, "25-34": 34, "35-49": 49, "50-64": 64, "65+": 200}


def section(title):
    print(f"\n{'#' * 100}\n# {title}\n{'#' * 100}")


def days_of(p):
    return (pd.to_datetime(p["period_end"]) - pd.to_datetime(p["period_start"])).dt.days + 1


def main(folder):
    t = {f[:-8]: pd.read_parquet(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".parquet")}
    cust, addr, claims, weather = t["customers"], t["addresses"], t["claims"], t["weather"]
    policy_tables = {n: t[n] for pair in LINES.values() for n in pair}

    section("Q1 exposure vs period days (365 or 360 basis?)")
    for name, p in policy_tables.items():
        d = days_of(p)
        bad = p["exposure"] > 1.0001
        e365, e360 = d / 365, d / 360
        ok = ~bad
        print(f"\n{name}: rows={len(p):,}; exposure > 1: {bad.sum():,} rows; top values {p.loc[bad, 'exposure'].value_counts().head(5).to_dict()}")
        if bad.any():
            print(f"  of these, exposure == period days: {(p.loc[bad, 'exposure'] == d[bad]).sum():,}; "
                  f"exposure == 365/360 (1.0139): {(p.loc[bad, 'exposure'].round(4) == round(365 / 360, 4)).sum():,}; "
                  f"exposure == 366/365 (1.0027): {(p.loc[bad, 'exposure'].round(4) == round(366 / 365, 4)).sum():,}")
        print(f"  good rows: max |exposure - days/365| = {(p.loc[ok, 'exposure'] - e365[ok]).abs().max():.4f}; "
              f"max |exposure - days/360| = {(p.loc[ok, 'exposure'] - e360[ok]).abs().max():.4f}")
        print(f"  good rows within 0.0006 of days/365: {((p.loc[ok, 'exposure'] - e365[ok]).abs() < 0.0006).mean():.2%}; "
              f"within 0.0006 of days/360: {((p.loc[ok, 'exposure'] - e360[ok]).abs() < 0.0006).mean():.2%}")
        on365 = (p["exposure"] - e365).abs() < 0.0006
        on360 = ((p["exposure"] - e360).abs() < 0.0006) & ~on365
        print(f"  rows matching days/365: {on365.sum():,} ({on365.mean():.2%}); rows matching days/360 only (360-day basis): {on360.sum():,} ({on360.mean():.2%}); "
              f"rows matching neither: {(~on365 & ~on360).sum():,} ({(~on365 & ~on360).mean():.2%}) of which exposure == days: {((~on365 & ~on360) & (p['exposure'] == d)).sum():,}")
        print(f"  days per row: min {d.min()}, max {d.max()}; rows with days > 365: {(d > 365).sum():,}; "
              f"last period_end by start year: {p.groupby(p['period_start'].astype(str).str[:4])['period_end'].max().astype(str).to_dict()}")
        print(f"  rows per policy-year > 1: {(p.groupby([p['policy_id'], p['period_start'].astype(str).str[:4]]).size() > 1).sum():,}")
        spans = (p["period_start"].astype(str).str[:4] != p["period_end"].astype(str).str[:4]).sum()
        print(f"  rows whose period crosses a calendar year: {spans:,}")

    section("Q2 customer_id leading zeros / joins")
    for name, p in policy_tables.items():
        miss = ~p["customer_id"].isin(set(cust["customer_id"]))
        print(f"{name}: customer_id length counts {p['customer_id'].str.len().value_counts().to_dict()}; "
              f"not in customers: {miss.sum():,}; ids starting with '0': {(p['customer_id'].str[0] == '0').sum():,}; "
              f"address_id not in addresses: {(~p['address_id'].isin(set(addr['address_id']))).sum():,}")
    print(f"customers: customer_id length counts {cust['customer_id'].str.len().value_counts().to_dict()}; ids starting with '0': {(cust['customer_id'].str[0] == '0').sum():,}")
    used_c = set().union(*[set(p["customer_id"]) for p in policy_tables.values()])
    used_a = set().union(*[set(p["address_id"]) for p in policy_tables.values()])
    print(f"customers never on a policy row: {(~cust['customer_id'].isin(used_c)).sum():,} of {len(cust):,}; "
          f"addresses never on a policy row: {(~addr['address_id'].isin(used_a)).sum():,} of {len(addr):,}")
    print(f"weather: addresses not in addresses table: {(~weather['address_id'].isin(set(addr['address_id']))).sum():,}; years {sorted(weather['year'].unique())}; "
          f"rows per year {weather['year'].value_counts().sort_index().to_dict()}; duplicated (address, year): {weather.duplicated(['address_id', 'year']).sum():,}")

    section("Q3 sum_insured outliers (x1000?)")
    for name in ["policies_buildings", "test_buildings", "policies_contents", "test_contents"]:
        s = t[name]["sum_insured"]
        print(f"\n{name}: quantiles {s.quantile([.01, .5, .99, .999]).round(0).to_dict()}; max {s.max():,.0f}")
        for thr in [1e6, 2e6, 5e6]:
            print(f"  rows > {thr:,.0f}: {(s > thr).sum():,}", end="")
        print()
        below, above = s[s <= 5e6], s[s > 5e6]
        amin = above.min() if len(above) else float("nan")
        amax = above.max() if len(above) else float("nan")
        print(f"  largest value <= 5m: {below.max():,.0f}; smallest value > 5m: {amin:,.0f}; "
              f"after dividing by 1000 the scaled values lie in {amin / 1000:,.0f} .. {amax / 1000:,.0f}")
        lg = np.sort(np.log10(s.values))
        gaps = np.diff(lg)
        i = gaps.argmax()
        print(f"  largest gap in log10 sorted values: between {10 ** lg[i]:,.0f} and {10 ** lg[i + 1]:,.0f}")
        print(f"  rows with sum_insured <= 0: {(s <= 0).sum():,}; scaled share of rows: {(s > 5e6).mean():.2%}")

    section("Q4 construction_type codes / case")
    for name in ["policies_buildings", "test_buildings"]:
        print(f"{name}: {t[name]['construction_type'].value_counts(dropna=False).to_dict()}")
    print("reference_policies:", t["reference_policies"]["construction_type"].value_counts(dropna=False).to_dict())
    print("written_premium mean by construction_type (training):\n",
          t["policies_buildings"].groupby("construction_type")["written_premium"].agg(["count", "mean"]).round(1))

    section("Q5 claims -> policy rows (aiiaa.target.match_claims logic)")
    c = claims.copy()
    print(f"claim_id unique: {c['claim_id'].is_unique}; claims by line: {c['line'].value_counts().to_dict()}")
    prefix_ok = c["policy_id"].str[0].map({"M": "motor", "B": "buildings", "C": "contents"}) == c["line"]
    print(f"claims whose policy_id prefix (M/B/C) disagrees with line: {(~prefix_ok).sum():,}")
    for line, (train, _) in LINES.items():
        cl = c[c["line"] == line]
        m = match_claims(t[train], cl)
        hits = m.dropna(subset=["row_id"]).groupby("claim_id").size()
        un = m[m["row_id"].isna()]
        in_pol = cl["policy_id"].isin(set(t[train]["policy_id"]))
        print(f"\n{line}: claims {len(cl):,}; matched to exactly 1 row {(hits == 1).sum():,}; >1 rows {(hits > 1).sum():,}; unmatched {len(un):,} "
              f"({len(un) / len(cl):.2%}); policy_id not in {train}: {(~in_pol).sum():,}")
        print(f"  unmatched by loss year: {un['loss_date'].astype(str).str[:4].value_counts().sort_index().to_dict()}")
        print(f"  unmatched net cost (non-repudiated): {net_cost(un).sum():,.0f} of total {net_cost(cl).sum():,.0f}")
        print(f"  all claims by loss year: {cl['loss_date'].astype(str).str[:4].value_counts().sort_index().to_dict()}")
        first_start = t[train].groupby("policy_id")["period_start"].min()
        last_end = t[train].groupby("policy_id")["period_end"].max()
        print(f"  unmatched claims with loss_date before the policy's first row: {(un['loss_date'] < un['policy_id'].map(first_start)).sum():,}; "
              f"after its last row: {(un['loss_date'] > un['policy_id'].map(last_end)).sum():,}")

    section("Q6 customer_service_contacts, Q8 telematics, Q9 renewal_decision")
    for line, (train, test) in LINES.items():
        print(f"{line}: customer_service_contacts max train {t[train]['customer_service_contacts'].max()} / test {t[test]['customer_service_contacts'].max()}; "
              f"train rows > 10: {(t[train]['customer_service_contacts'] > 10).sum():,}; renewal_decision in train: {'renewal_decision' in t[train]}, in test: {'renewal_decision' in t[test]}")
    for name in ["policies_motor", "test_motor"]:
        print(f"{name}: telematics_opt_in present={'telematics_opt_in' in t[name]}, telematics_score present={'telematics_score' in t[name]}, "
              f"annual_mileage present={'annual_mileage' in t[name]}, driver_age present={'driver_age' in t[name]}")
    print("customers: credit_band present =", "credit_band" in cust, "; date_of_birth present =", "date_of_birth" in cust, "; gender present =", "gender" in cust)
    for name in ["policies_motor", "policies_buildings", "policies_contents"]:
        print(f"{name} renewal_decision: {t[name]['renewal_decision'].value_counts().to_dict()}")

    section("Q7 missing-value rates (columns with any missing; developed Part 1 rate was about 2.5%)")
    for name in ["addresses", "customers", "policies_motor", "policies_buildings", "policies_contents", "test_motor", "test_buildings", "test_contents", "claims"]:
        miss = (t[name].isna().mean() * 100).round(2)
        miss = miss[miss > 0]
        print(f"{name}: " + ", ".join(f"{k} {v}%" for k, v in miss.items()))

    section("occupation (free text), gender, age bands, region, case/whitespace variants in every text column")
    occ = cust["occupation"].dropna()
    print(f"occupation: {occ.nunique()} distinct raw strings of {len(occ):,} non-missing values")
    print("top 20 raw values (repr shows hidden whitespace):")
    for k, v in occ.value_counts().head(20).items():
        print(f"  {k!r}: {v:,}")
    canon = occ.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    se = canon.str.startswith("self-employed ")
    base = canon.str.replace(r"^self-employed ", "", regex=True)
    print(f"after strip + collapse spaces + lower-case: {canon.nunique()} distinct; self-employed prefix on {se.sum():,} values ({se.mean():.1%}); "
          f"distinct base occupations after removing the prefix: {base.nunique()}")
    print("base occupations:", sorted(base.unique()))
    ws = (occ != occ.str.strip()) | occ.str.contains(r"\s{2,}")
    print(f"values with leading/trailing/double whitespace: {ws.sum():,}; all-lower-case values: {(occ == occ.str.lower()).sum():,}")
    print(f"raw strings that are whitespace/case variants of another raw string: {occ.nunique() - canon.nunique()} extra strings")
    print(f"gender levels: {cust['gender'].value_counts(dropna=False).to_dict()}")
    print(f"marital_status levels: {cust['marital_status'].value_counts(dropna=False).to_dict()}")
    print(f"region levels (addresses): {addr['region'].value_counts(dropna=False).sort_index().to_dict()}")
    for name in ["policies_motor", "test_motor"]:
        print(f"{name} driver_age_band: {t[name]['driver_age_band'].value_counts(dropna=False).to_dict()}")
    for name in ["policies_buildings", "test_buildings", "policies_contents", "test_contents"]:
        print(f"{name} policyholder_age_band: {t[name]['policyholder_age_band'].value_counts(dropna=False).to_dict()}")
    print("\nText columns whose number of levels drops after strip + lower-case (hidden variants):")
    found = False
    for name, df in t.items():
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_string_dtype(s) and col not in {"row_id", "policy_id", "customer_id", "address_id", "claim_id", "third_party_id", "repairer_id", "surname"}:
                s = s.dropna()
                if s.nunique() > 5000:
                    continue
                norm = s.str.strip().str.lower()
                if norm.nunique() != s.nunique():
                    found = True
                    print(f"  {name}.{col}: {s.nunique()} raw levels -> {norm.nunique()} after normalising")
    if not found:
        print("  none")
    print("\nidentifier format checks (regex):")
    pat = {"customer_id": r"^\d{8}$", "address_id": r"^A\d{7}$", "policy_id": r"^[MBC]\d{8}$", "row_id": r"^[MBC]\d{8}$"}
    for name, df in t.items():
        for col, rx in pat.items():
            if col in df:
                bad = (~df[col].astype(str).str.match(rx)).sum()
                if bad:
                    print(f"  {name}.{col}: {bad:,} values do not match {rx}")
    print("  (no line printed above = all identifiers match their format)")

    section("duplicated ids, overlapping rows, id overlap train/test/reference")
    for name, p in policy_tables.items():
        full_dup = p.drop(columns=["row_id"]).duplicated().sum()
        s = p.assign(ps=pd.to_datetime(p["period_start"]), pe=pd.to_datetime(p["period_end"])).sort_values(["policy_id", "ps"])
        prev_end = s.groupby("policy_id")["pe"].shift()
        overlap = (s["ps"] <= prev_end).sum()
        print(f"{name}: row_id duplicated {p['row_id'].duplicated().sum():,}; fully duplicated rows (ignoring row_id) {full_dup:,}; rows overlapping the previous row of the same policy {overlap:,}")
    all_ids = pd.concat([p["row_id"] for p in policy_tables.values()])
    print(f"row_id duplicated across all six policy tables: {all_ids.duplicated().sum():,}")
    ref = t["reference_policies"]
    test_all = pd.concat([t[n] for n in ["test_motor", "test_buildings", "test_contents"]])
    print(f"reference_policies row_id found in test files: {ref['row_id'].isin(set(test_all['row_id'])).sum()} of {len(ref)}; "
          f"in sample_submission: {ref['row_id'].isin(set(t['sample_submission']['row_id'])).sum()}")
    chk = ref.merge(test_all[["row_id", "written_premium", "exposure", "period_start"]], on="row_id", suffixes=("", "_test"))
    print(f"  written_premium equal to test file: {(chk['written_premium'] == chk['written_premium_test']).sum()} of {len(chk)}; exposure equal: {(chk['exposure'] == chk['exposure_test']).sum()}")

    section("date ranges (period_start / period_end), per table")
    for name, p in policy_tables.items():
        print(f"{name}: period_start {p['period_start'].min()} .. {p['period_start'].max()}; period_end {p['period_end'].min()} .. {p['period_end'].max()}")
    print(f"claims: loss_date {c['loss_date'].min()} .. {c['loss_date'].max()}; report_date {c['report_date'].min()} .. {c['report_date'].max()}")

    section("training vs test: exposure, new business, seen policies, region mix")
    for line, (train, test) in LINES.items():
        tr, te = t[train], t[test]
        print(f"\n{line}")
        print(f"  test exposure quantiles {te['exposure'].quantile([0, .05, .25, .5, .75, .95, 1]).round(4).to_dict()}; mean {te['exposure'].mean():.4f}; sum {te['exposure'].sum():,.1f}")
        print(f"  test rows with exposure == 1: {(te['exposure'] >= 0.9999).mean():.1%}; exposure < 0.5: {(te['exposure'] < 0.5).mean():.1%}; < 0.1: {(te['exposure'] < 0.1).mean():.1%}")
        by_year = tr.groupby(tr["period_start"].astype(str).str[:4])
        print(f"  train mean exposure by year {by_year['exposure'].mean().round(4).to_dict()} (raw, before fixing exposure>1); median {by_year['exposure'].median().to_dict()}")
        print(f"  new_business share: train by year {by_year['new_business'].mean().round(4).to_dict()}; test {te['new_business'].mean():.4f}; "
              f"by exposure: test {(te['exposure'] * te['new_business']).sum() / te['exposure'].sum():.4f}")
        seen = te["policy_id"].isin(set(tr["policy_id"]))
        print(f"  test rows whose policy_id is in training: {seen.mean():.1%}; of these new_business=True: {te.loc[seen, 'new_business'].mean():.1%}; "
              f"of unseen rows new_business=True: {te.loc[~seen, 'new_business'].mean():.1%}")
        print(f"  renewal_count: train median {tr['renewal_count'].median()}, max {tr['renewal_count'].max()}; test median {te['renewal_count'].median()}, max {te['renewal_count'].max()}; "
              f"test new_business=True & renewal_count>0: {((te['new_business']) & (te['renewal_count'] > 0)).sum():,}; "
              f"train new_business=True & renewal_count>0: {((tr['new_business']) & (tr['renewal_count'] > 0)).sum():,}; "
              f"train new_business=False & renewal_count==0: {((~tr['new_business']) & (tr['renewal_count'] == 0)).sum():,}")
        first_year = tr.groupby("policy_id")["period_start"].transform("min").astype(str).str[:4]
        later_nb = ((tr["period_start"].astype(str).str[:4] > first_year) & tr["new_business"]).sum()
        print(f"  train rows flagged new_business in a later year than the policy's first row: {later_nb:,}")
        reg = addr.set_index("address_id")["region"]
        r_tr = tr.assign(region=tr["address_id"].map(reg)).groupby("region")["exposure"].sum()
        r_te = te.assign(region=te["address_id"].map(reg)).groupby("region")["exposure"].sum()
        mix = pd.DataFrame({"train_share": (r_tr / r_tr.sum()).round(4), "test_share": (r_te / r_te.sum()).round(4)})
        print("  region mix by exposure:\n" + mix.to_string().replace("\n", "\n  "))
        ch = pd.DataFrame({"train": tr["channel"].value_counts(normalize=True).round(4), "test": te["channel"].value_counts(normalize=True).round(4)})
        print("  channel mix by rows:\n" + ch.to_string().replace("\n", "\n  "))
        print(f"  test addresses not on any training row: {(~te['address_id'].isin(set(tr['address_id']))).mean():.1%}")

    section("consistency between columns (new checks)")
    for name in ["policies_motor", "test_motor"]:
        df = t[name]
        max_licensed = df["driver_age_band"].map(AGE_BAND_MAX_AGE) - 17
        over = df["years_licensed"] > max_licensed
        print(f"{name}: years_licensed > (band max age - 17): {over.sum():,} rows ({over.mean():.2%}); "
              f"ncb_years > years_licensed: {(df['ncb_years'] > df['years_licensed']).sum():,}; "
              f"band 17-24 with years_licensed >= 8: {((df['driver_age_band'] == '17-24') & (df['years_licensed'] >= 8)).sum():,}; "
              f"named_young_driver True with band 17-24: {((df['named_young_driver']) & (df['driver_age_band'] == '17-24')).sum():,}")
        print(f"  vehicle_age min {df['vehicle_age'].min()}, max {df['vehicle_age'].max()}; years_licensed by band median: {df.groupby('driver_age_band')['years_licensed'].median().to_dict()}")
    print(f"customers: customer_since_year range {cust['customer_since_year'].min()} .. {cust['customer_since_year'].max()} (market started writing in 2023); "
          f"customers with customer_since_year < 2023: {(cust['customer_since_year'] < 2023).mean():.1%}")
    print(f"customers: years_at_address max {cust['years_at_address'].max()}; licence_points levels {sorted(cust['licence_points'].unique())}; surname distinct {cust['surname'].nunique()}")
    print(f"addresses: year_built range {addr['year_built'].min()} .. {addr['year_built'].max()}; weather rows per address-year unique: {weather.groupby(['address_id', 'year']).size().max() == 1}")
    for name, p in policy_tables.items():
        print(f"{name}: written_premium <= 0: {(p['written_premium'] <= 0).sum():,}; discount_applied levels {sorted(p['discount_applied'].unique())}; "
              f"discount 0.05 iff channel direct: {((p['discount_applied'] == 0.05) == (p['channel'] == 'direct')).mean():.1%} of rows")

    section("claims: perils, status, lags, open share, recoveries, net cost and cap")
    c = claims.assign(lag=(pd.to_datetime(claims["report_date"]) - pd.to_datetime(claims["loss_date"])).dt.days,
                      year=claims["loss_date"].astype(str).str[:4])
    print("claims by line x peril:\n", pd.crosstab(c["peril"], c["line"]))
    print("status share by line:\n", pd.crosstab(c["line"], c["status"], normalize="index").round(4))
    print("status counts by line:\n", pd.crosstab(c["line"], c["status"]))
    print(f"report_date before loss_date: {(c['lag'] < 0).sum():,}; report_date after 2024-12-31: {(pd.to_datetime(c['report_date']) > '2024-12-31').sum():,}; "
          f"loss_date after 2024-12-31: {(pd.to_datetime(c['loss_date']) > '2024-12-31').sum():,}")
    print("report lag days quantiles by line:\n", c.groupby("line")["lag"].quantile([.5, .9, .99, 1.0]).unstack().round(1))
    print("report lag days quantiles by line x loss year (2023, 2024 only):\n",
          c[c["year"].isin(["2023", "2024"])].groupby(["line", "year"])["lag"].quantile([.5, .9, .99, 1.0]).unstack().round(1))
    print("open share by line x loss year:\n", c.groupby(["line", "year"])["status"].apply(lambda s: round((s == "open").mean(), 3)).unstack())
    print("claim counts by line x loss year:\n", pd.crosstab(c["line"], c["year"]))
    closed = c[c["status"] == "closed"]
    print(f"closed claims with incurred != paid: {(closed['incurred_to_date'] - closed['paid_to_date']).abs().gt(0.01).sum():,}")
    rep = c[c["status"] == "repudiated"]
    print(f"repudiated: n={len(rep)}, paid sum={rep['paid_to_date'].sum():,.0f}, incurred sum={rep['incurred_to_date'].sum():,.0f}, recoveries sum={rep['recoveries'].sum():,.0f}")
    print(f"incurred - recoveries < 0: {((c['incurred_to_date'] - c['recoveries']) < 0).sum():,}; paid > incurred: {(c['paid_to_date'] > c['incurred_to_date'] + 0.01).sum():,}")
    print("recoveries > 0 share and total by line:\n",
          c.assign(rec=c["recoveries"] > 0).groupby("line").agg(share_with_recovery=("rec", "mean"), total_recoveries=("recoveries", "sum"), claims=("claim_id", "size")).round(3))
    print("repairer_id / third_party_id missing share by line:\n", c.groupby("line")[["repairer_id", "third_party_id"]].apply(lambda d: d.isna().mean()).round(3))
    valid = c[c["status"] != "repudiated"].assign(net=lambda d: net_cost(d))
    print("net cost quantiles by line (non-repudiated):\n", valid.groupby("line")["net"].quantile([.5, .9, .99, .995, .999, 1.0]).unstack().round(0))
    print("net cost mean and sum by line x loss year:\n", valid.pivot_table(index="line", columns="year", values="net", aggfunc=["mean", "sum"]).round(0))
    for line in LINES:
        cl = valid[valid["line"] == line]
        cap = large_loss_cap(c[c["line"] == line], 0.995)
        above = cl["net"] > cap
        print(f"{line}: cap at 99.5th percentile = {cap:,.0f}; claims above cap {above.sum()} of {len(cl)}; share of net cost above the cap {(cl['net'] - cap).clip(lower=0).sum() / cl['net'].sum():.2%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
