r"""C033: developed vs emerging market comparison table (Markdown).

Usage: .venv\Scripts\python.exe -I -u scripts\part2_compare.py data\iaa-ai-assisted-hackathon-2026-part-2 > docs\part2_market_comparison.md

Reported (not developed) figures; both markets are valued at 2024-12-31, so accident year 2024 is equally immature in both.
Exposure here is always (period_end - period_start + 1) / 365 (C001 basis), never the stored column.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from aiiaa.clean import fix_construction_type, fix_sum_insured  # noqa: E402
from aiiaa.target import large_loss_cap, net_cost, row_target  # noqa: E402

FLAG = 0.25
MIN_PP = 0.01  # a mix share must also move by at least 1 percentage point to be flagged
LINES = {"motor": ("policies_motor", "test_motor"), "buildings": ("policies_buildings", "test_buildings"),
         "contents": ("policies_contents", "test_contents")}
EM_CONSTRUCTION = {"BRK": "brick", "STN": "stone", "WOD": "timber", "MIX": "non_standard"}
MIX_COLS = {
    "motor": ["cover_type", "channel", "fuel_type", "body_type", "transmission", "vehicle_use", "overnight_parking", "payment_method",
              "new_business", "voluntary_excess", "driver_age_band", "urban_density", "property_type"],
    "buildings": ["construction_type", "occupancy", "channel", "payment_method", "new_business", "voluntary_excess", "policyholder_age_band",
                  "urban_density", "property_type", "accidental_damage_cover", "mortgage"],
    "contents": ["tenure", "security_level", "channel", "payment_method", "new_business", "voluntary_excess", "policyholder_age_band",
                 "urban_density", "property_type", "accidental_damage_cover", "works_from_home"],
}
NUM_COLS = {
    "motor": ["vehicle_age", "years_licensed", "ncb_years", "vehicle_value_group", "engine_size_cc", "seats", "written_premium", "renewal_count"],
    "buildings": ["sum_insured", "days_unoccupied", "written_premium", "renewal_count"],
    "contents": ["sum_insured", "high_value_items", "written_premium", "renewal_count"],
}
ADDR_NUM = ["year_built", "bedrooms", "distance_to_river_km", "elevation_m", "flood_zone"]
WEATHER_COLS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm", "mean_temperature_c"]


def md(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def rel(em, dev):
    if dev is None or em is None or pd.isna(dev) or pd.isna(em) or dev == 0:
        return np.nan
    return em / dev - 1


def flag_rel(r):
    return "" if pd.isna(r) else ("**FLAG**" if abs(r) > FLAG else "")


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "" if pd.isna(x) else f"{x:,.{d}f}"


def load(root, market, name, cols=None):
    return pd.read_parquet(os.path.join(root, market, f"{name}.parquet"), columns=cols)


def prep(p):
    p = p.copy()
    p["exp"] = ((pd.to_datetime(p["period_end"]) - pd.to_datetime(p["period_start"])).dt.days + 1) / 365
    p["year"] = pd.to_datetime(p["period_start"]).dt.year
    return p


def mix(df, col):
    s = df[col].astype("object")
    s = s.where(df[col].notna(), "(missing)").astype(str)
    return df.groupby(s)["exp"].sum() / df["exp"].sum()


def main(root):
    m = {}
    for market in ["developed", "emerging"]:
        d = {"addresses": load(root, market, "addresses"), "weather": load(root, market, "weather"), "claims": load(root, market, "claims")}
        for line, (tr, te) in LINES.items():
            d[tr], d[te] = prep(load(root, market, tr)), prep(load(root, market, te))
        m[market] = d

    # harmonise categorical definitions
    for market in m:
        for name in ["policies_buildings", "test_buildings"]:
            df = fix_sum_insured(m[market][name])
            if market == "developed":
                df = fix_construction_type(df)
            else:
                df["construction_type"] = df["construction_type"].map(EM_CONSTRUCTION)
            m[market][name] = df
        for name in ["policies_contents", "test_contents"]:
            m[market][name] = fix_sum_insured(m[market][name])
    for name in ["policies_motor", "test_motor"]:
        df = m["developed"][name]
        df["driver_age_band"] = pd.cut(df["driver_age"], [0, 24, 34, 49, 64, 200], labels=["17-24", "25-34", "35-49", "50-64", "65+"]).astype(object)
    for line, (tr, te) in LINES.items():
        if line != "motor":
            for name in (tr, te):
                df = m["developed"][name]
                df["policyholder_age_band"] = pd.cut(df["policyholder_age"], [0, 24, 34, 49, 64, 200], labels=["17-24", "25-34", "35-49", "50-64", "65+"]).astype(object)
    for market in m:
        addr = m[market]["addresses"].set_index("address_id")
        for line, (tr, te) in LINES.items():
            for name in (tr, te):
                df = m[market][name]
                for c in ["urban_density", "property_type"] + ADDR_NUM:
                    df[c] = df["address_id"].map(addr[c])

    flagged = []
    print("# Part 2 market comparison: developed vs emerging\n")
    print("Source: `scripts/part2_compare.py` on `data/iaa-ai-assisted-hackathon-2026-part-2/{developed,emerging}`. Reported (not developed) claim cost, valued at 2024-12-31 in both markets. "
          "Exposure-years are (period_end - period_start + 1) / 365, so the exposure typos found in `docs/part2_emerging_quality.txt` (Q1) do not enter. "
          "Claim count and cost use the C005 matching (loss_date inside the row period, same policy_id) and exclude repudiated claims; unmatched claims are not counted. "
          "Currency is the fictional CU in both markets.\n")
    print(f"Flag rule: **FLAG** means emerging differs from developed by more than {int(FLAG * 100)}% in relative terms (emerging / developed - 1); "
          f"for category shares the move must also be at least {int(MIN_PP * 100)} percentage point.\n")

    # ---- section 1: headline per line
    print("## 1. Headline numbers per line\n")
    headline = {}
    for line, (tr, te) in LINES.items():
        rows = {}
        for market in m:
            p = m[market][tr]
            cl = m[market]["claims"]
            cl = cl[cl["line"] == line]
            tgt = row_target(p[["row_id", "policy_id", "period_start", "period_end"]], cl)
            p = p.merge(tgt, on="row_id", how="left")
            for label, mask in [("2021", p["year"] == 2021), ("2022", p["year"] == 2022), ("2023", p["year"] == 2023), ("2024", p["year"] == 2024),
                                ("2023-24", p["year"].isin([2023, 2024]))]:
                s = p[mask]
                if len(s) == 0:
                    continue
                expo, n, cost = s["exp"].sum(), s["claim_count"].sum(), s["claim_cost"].sum()
                rows[(market, label)] = {"rows": len(s), "exposure_years": expo, "claims": n, "frequency": n / expo, "mean_severity": cost / n if n else np.nan,
                                         "pure_premium": cost / expo, "wp_median": s["written_premium"].median(),
                                         "wp_full_year_median": s.loc[s["exp"] > 0.95, "written_premium"].median()}
        headline[line] = rows
        print(f"### {line}\n")
        cols = [("developed", "2021"), ("developed", "2022"), ("developed", "2023"), ("developed", "2024"), ("developed", "2023-24"),
                ("emerging", "2023"), ("emerging", "2024"), ("emerging", "2023-24")]
        names = {"rows": "training rows", "exposure_years": "exposure-years", "claims": "claims (matched, not repudiated)", "frequency": "claim frequency per exposure-year",
                 "mean_severity": "mean severity (CU, reported)", "pure_premium": "reported pure premium per exposure-year (CU)",
                 "wp_median": "written_premium median, all rows (CU)", "wp_full_year_median": "written_premium median, rows > 0.95 year (CU)"}
        table = []
        for key, label in names.items():
            r = {"metric": label}
            for mk, yr in cols:
                v = rows.get((mk, yr), {}).get(key)
                r[f"{'dev' if mk == 'developed' else 'em'} {yr}"] = (pct(v, 2) if key == "frequency" else num(v, 1 if key == "exposure_years" else 0)) if v is not None else ""
            dv, ev = rows[("developed", "2023-24")][key], rows[("emerging", "2023-24")][key]
            if key in {"frequency", "mean_severity", "pure_premium", "wp_median", "wp_full_year_median"}:
                r_ = rel(ev, dv)
                r["em/dev - 1 (2023-24)"] = pct(r_, 0)
                r["flag"] = flag_rel(r_)
                if r["flag"]:
                    flagged.append(f"{line}: {label} emerging {num(ev, 4 if key == 'frequency' else 0)} vs developed {num(dv, 4 if key == 'frequency' else 0)} ({pct(r_, 0)})")
            else:
                r["em/dev - 1 (2023-24)"] = ""
                r["flag"] = ""
            table.append(r)
        print(md(pd.DataFrame(table), index=False))
        # per-year frequency and pure premium ratio
        r23 = rel(rows[("emerging", "2023")]["frequency"], rows[("developed", "2023")]["frequency"])
        r24 = rel(rows[("emerging", "2024")]["frequency"], rows[("developed", "2024")]["frequency"])
        print(f"\nFrequency ratio emerging / developed - 1: 2023 {pct(r23, 0)}, 2024 {pct(r24, 0)}. "
              f"Emerging 2024 vs 2023 frequency {pct(rel(rows[('emerging', '2024')]['frequency'], rows[('emerging', '2023')]['frequency']), 0)}; "
              f"pure premium {pct(rel(rows[('emerging', '2024')]['pure_premium'], rows[('emerging', '2023')]['pure_premium']), 0)} "
              f"(developed 2024 vs 2023: frequency {pct(rel(rows[('developed', '2024')]['frequency'], rows[('developed', '2023')]['frequency']), 0)}, "
              f"pure premium {pct(rel(rows[('developed', '2024')]['pure_premium'], rows[('developed', '2023')]['pure_premium']), 0)}).\n")

    # ---- section 2: test sets
    print("## 2. Test (2025) rows and exposure\n")
    table = []
    for line, (tr, te) in LINES.items():
        d, e = m["developed"][te], m["emerging"][te]
        for label, dv, ev in [("test rows", len(d), len(e)), ("test exposure-years", d["exp"].sum(), e["exp"].sum()),
                              ("mean exposure per test row", d["exp"].mean(), e["exp"].mean()),
                              ("new_business share of test rows", d["new_business"].mean(), e["new_business"].mean()),
                              ("written_premium median, test rows (CU)", d["written_premium"].median(), e["written_premium"].median()),
                              ("test policy_id seen in training", d["policy_id"].isin(set(m["developed"][tr]["policy_id"])).mean(), e["policy_id"].isin(set(m["emerging"][tr]["policy_id"])).mean())]:
            r_ = rel(ev, dv)
            share = "share" in label or "seen" in label
            table.append({"line": line, "metric": label, "developed": pct(dv, 1) if share else num(dv, 3 if "mean" in label else 0),
                          "emerging": pct(ev, 1) if share else num(ev, 3 if "mean" in label else 0), "em/dev - 1": pct(r_, 0), "flag": flag_rel(r_)})
            if flag_rel(r_):
                flagged.append(f"{line}: {label} emerging {pct(ev, 1) if share else num(ev, 3 if 'mean' in label else 0)} vs developed {pct(dv, 1) if share else num(dv, 3 if 'mean' in label else 0)} ({pct(r_, 0)})")
    print(md(pd.DataFrame(table), index=False))
    print("\nRatio of test exposure-years to the 2023-24 training exposure-years (emerging): " + ", ".join(
        f"{line} {m['emerging'][te]['exp'].sum() / m['emerging'][tr]['exp'].sum():.2f}x" for line, (tr, te) in LINES.items()) +
        "; developed (to 2021-24): " + ", ".join(f"{line} {m['developed'][te]['exp'].sum() / m['developed'][tr]['exp'].sum():.2f}x" for line, (tr, te) in LINES.items()) + ".\n")

    # ---- section 3: category mix
    print("## 3. Category mix (share of exposure-years)\n")
    print("`dev` = developed training 2021-24, `em` = emerging training 2023-24, `em test` = emerging 2025 test. "
          "construction_type is shown after mapping BRK -> brick, STN -> stone, WOD -> timber, MIX -> non_standard (developed after the C004 clean-up). "
          "The developed driver and policyholder ages were cut into the emerging bands. "
          "`flag dev` flags em vs dev; `flag test` flags em test vs em training (same rule).\n")
    for line, (tr, te) in LINES.items():
        print(f"### {line}\n")
        table = []
        for col in MIX_COLS[line]:
            a, b, c = mix(m["developed"][tr], col), mix(m["emerging"][tr], col), mix(m["emerging"][te], col)
            levels = sorted(set(a.index) | set(b.index) | set(c.index))
            for lv in levels:
                av, bv, cv = a.get(lv, 0.0), b.get(lv, 0.0), c.get(lv, 0.0)
                r1, r2 = rel(bv, av), rel(cv, bv)
                f1 = "**FLAG**" if (not pd.isna(r1) and abs(r1) > FLAG and abs(bv - av) >= MIN_PP) or (av == 0 and bv >= MIN_PP) else ""
                f2 = "**FLAG**" if (not pd.isna(r2) and abs(r2) > FLAG and abs(cv - bv) >= MIN_PP) or (bv == 0 and cv >= MIN_PP) else ""
                table.append({"variable": col, "level": lv, "dev": pct(av), "em": pct(bv), "em test": pct(cv), "em/dev - 1": pct(r1, 0), "flag dev": f1,
                              "test/em - 1": pct(r2, 0), "flag test": f2})
                if f1:
                    flagged.append(f"{line}: mix {col}={lv}: emerging {pct(bv)} vs developed {pct(av)}")
                if f2:
                    flagged.append(f"{line}: mix {col}={lv}: emerging test {pct(cv)} vs emerging training {pct(bv)}")
        print(md(pd.DataFrame(table), index=False))
        print()

    # ---- section 4: numeric medians
    print("## 4. Numeric features (medians over training rows; address fields joined by address_id)\n")
    table = []
    for line, (tr, te) in LINES.items():
        for col in NUM_COLS[line] + ADDR_NUM:
            dv, bv, tv = m["developed"][tr][col].median(), m["emerging"][tr][col].median(), m["emerging"][te][col].median()
            r1, r2 = rel(bv, dv), rel(tv, bv)
            table.append({"line": line, "variable": col, "dev": num(dv, 2), "em": num(bv, 2), "em test": num(tv, 2), "em/dev - 1": pct(r1, 0), "flag dev": flag_rel(r1),
                          "test/em - 1": pct(r2, 0), "flag test": flag_rel(r2)})
            if flag_rel(r1):
                flagged.append(f"{line}: median {col}: emerging {num(bv, 2)} vs developed {num(dv, 2)} ({pct(r1, 0)})")
            if flag_rel(r2):
                flagged.append(f"{line}: median {col}: emerging test {num(tv, 2)} vs emerging training {num(bv, 2)} ({pct(r2, 0)})")
    print(md(pd.DataFrame(table), index=False))
    print("\nMean weather by address, year 2024 (all addresses of the market):\n")
    table = []
    for col in WEATHER_COLS:
        dv = m["developed"]["weather"].query("year == 2024")[col].mean()
        bv = m["emerging"]["weather"].query("year == 2024")[col].mean()
        r1 = rel(bv, dv)
        table.append({"variable": col, "dev": num(dv, 3), "em": num(bv, 3), "em/dev - 1": pct(r1, 0), "flag": flag_rel(r1)})
        if flag_rel(r1):
            flagged.append(f"weather 2024 mean {col}: emerging {num(bv, 3)} vs developed {num(dv, 3)} ({pct(r1, 0)})")
    print(md(pd.DataFrame(table), index=False))

    # ---- section 5: missing
    print("\n## 5. Missing-value rates (columns where either market is above 0.5%)\n")
    table = []
    for name, cols in [("customers", None), ("addresses", None)] + [(tr, None) for tr, _ in LINES.values()]:
        d = load(root, "developed", name)
        e = load(root, "emerging", name)
        for col in [c for c in e.columns if c in d.columns]:
            dv, ev = d[col].isna().mean(), e[col].isna().mean()
            if dv > 0.005 or ev > 0.005:
                table.append({"table": name, "column": col, "dev": pct(dv, 2), "em": pct(ev, 2)})
        only_d = [c for c in d.columns if c not in e.columns]
        only_e = [c for c in e.columns if c not in d.columns]
        table.append({"table": name, "column": "(columns only in developed: " + ", ".join(only_d) + "; only in emerging: " + ", ".join(only_e) + ")", "dev": "", "em": ""})
    print(md(pd.DataFrame(table), index=False))

    # ---- section 6: claims
    print("\n## 6. Claims\n")
    table = []
    cap_rows = []
    for line in LINES:
        for label, market, years in [("developed 2021-24", "developed", None), ("developed 2023-24", "developed", [2023, 2024]), ("emerging (all claims)", "emerging", None)]:
            cl = m[market]["claims"]
            cl = cl[cl["line"] == line].assign(year=lambda d: pd.to_datetime(d["loss_date"]).dt.year,
                                               lag=lambda d: (pd.to_datetime(d["report_date"]) - pd.to_datetime(d["loss_date"])).dt.days)
            if years:
                cl = cl[cl["year"].isin(years)]
            valid = cl[cl["status"] != "repudiated"]
            n = net_cost(valid)
            cap = large_loss_cap(valid, 0.995)
            table.append({"line": line, "set": label, "claims": len(cl), "repudiated": pct((cl["status"] == "repudiated").mean(), 2), "open (all years)": pct((cl["status"] == "open").mean(), 1),
                          "open, loss year 2024": pct((cl.loc[cl["year"] == 2024, "status"] == "open").mean(), 1),
                          "with recovery": pct((cl["recoveries"] > 0).mean(), 1), "report lag median days": num(cl["lag"].median(), 0),
                          "lag p90": num(cl["lag"].quantile(.9), 0), "lag p99": num(cl["lag"].quantile(.99), 0),
                          "net cost median": num(n.median(), 0), "p90": num(n.quantile(.9), 0), "p99": num(n.quantile(.99), 0),
                          "cap (p99.5)": num(cap, 0), "claims above cap": int((n > cap).sum()), "cost share above cap": pct((n - cap).clip(lower=0).sum() / n.sum(), 1)})
            if market == "emerging" or years is None:
                cap_rows.append((line, label, cap))
    print(md(pd.DataFrame(table), index=False))
    print("\nPerils (share of claims within line):\n")
    table = []
    for line in LINES:
        for market in m:
            cl = m[market]["claims"]
            cl = cl[cl["line"] == line]
            sh = cl["peril"].value_counts(normalize=True)
            table.append({"line": line, "market": "dev" if market == "developed" else "em", **{k: pct(v, 1) for k, v in sh.items()}})
    print(md(pd.DataFrame(table).fillna(""), index=False))
    print("\nCaps: the Part 1 pipeline caps each claim at the 99.5th percentile of the line's non-repudiated net costs (C014). "
          "Emerging caps are estimated from 2 accident years (about 1,000 property claims and 7,000 motor claims), so each is set by only a handful of claims.\n")

    # ---- section 7: flag summary
    print("## 7. Everything flagged above (emerging vs developed; emerging test vs emerging training where stated)\n")
    for f in flagged:
        print(f"- {f}")
    print(f"\nTotal flagged items: {len(flagged)}.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
