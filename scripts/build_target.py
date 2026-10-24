"""Build the developed target for every training row and reconcile it against the claims file.

Outputs (completeness mixes per-peril reporting CDFs by cost share, C022): output/target_<line>.parquet (row_id, exposure, claim_count, claim_cost, completeness,
claim_cost_ult, exposure_eff) and a reconciliation report on stdout.

C037 (`--market emerging`): emerging policies and claims, cap at the emerging line's own 99.5th percentile, but the
reporting CDFs and peril shares come from the DEVELOPED market's accident year 2021 (emerging claims are observed for
at most two years). Writes output/emerging/target_<line>.parquet and, per line, the emerging AY2023 cost-weighted CDF
next to the developed CDFs at the same lags.

Usage: .venv\\Scripts\\python.exe scripts\\build_target.py [--market developed|emerging] > docs\\target_summary.txt
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from aiiaa import clean, development, markets, target  # noqa: E402

CDF_ACCIDENT_YEAR = 2021  # C006: oldest full year, observed for >= 3 years at valuation
CAP_QUANTILE = 0.995  # C014: per-claim large-loss cap; motor sweep favoured ~50k (99.5th percentile)
CHECK_LAGS = (30, 90, 180, 365)  # C037: lags of the emerging vs developed CDF consistency table
CHECK_YEAR = 2023  # C037: oldest full emerging accident year


def load(name, market):
    return pd.read_parquet(os.path.join(markets.get(market).data_dir, f"{name}.parquet"))


def main(market):
    m = markets.get(market)
    out_dir = m.out_dir  # C034: output/ (developed) or output/emerging/
    os.makedirs(out_dir, exist_ok=True)
    claims = load("claims", m.name)
    dev_claims = claims if m.name == "developed" else load("claims", "developed")  # C037: CDF source
    for line in ["motor", "buildings", "contents"]:
        p = clean.clean_policies(load(f"policies_{line}", m.name))
        c = claims[claims["line"] == line]
        valid = c[c["status"] != "repudiated"]
        dev_c = dev_claims[dev_claims["line"] == line]
        dev_valid = dev_c[dev_c["status"] != "repudiated"]

        cap = target.large_loss_cap(c, CAP_QUANTILE)
        t = target.row_target(p, c, cap=cap)
        print(f"\n{line}: large-loss cap (q={CAP_QUANTILE}) = {cap:,.0f}; capped share of cost = "
              f"{1 - t['claim_cost_capped'].sum() / t['claim_cost'].sum():.4f}")
        base = dev_valid[pd.to_datetime(dev_valid["loss_date"]).dt.year == CDF_ACCIDENT_YEAR]  # C037: developed AY2021
        cdfs = development.reporting_cdf_by_peril(base)  # C022: per-peril CDFs, mixed by cost share
        shares = development.peril_cost_shares(base)
        t["completeness"] = development.row_completeness_mixed(p, cdfs, shares).to_numpy()
        t["exposure"] = p["exposure"].to_numpy()
        t = development.develop(t)
        final = os.path.join(out_dir, f"target_{line}.parquet")
        tmp = os.path.join(out_dir, f"target_{line}.tmp.parquet")
        t.to_parquet(tmp, index=False)
        os.replace(tmp, final)  # atomic swap: other processes may be reading the old file

        # Reconciliation: every non-repudiated cost is either on a row or listed as unmatched (C005)
        m_claims = target.match_claims(p, valid)
        unmatched = m_claims[m_claims["row_id"].isna()]
        total = target.net_cost(valid).sum()
        on_rows = t["claim_cost"].sum()
        off_rows = target.net_cost(unmatched).sum()
        print(f"\n######## {line}")
        print(f"rows={len(t):,}  claims(non-repudiated)={len(valid):,}  matched={len(m_claims) - len(unmatched):,}  unmatched={len(unmatched):,}")
        print(f"net cost total={total:,.2f}  on rows={on_rows:,.2f}  unmatched={off_rows:,.2f}  difference={total - on_rows - off_rows:,.4f}")
        print(f"unmatched by loss year: {pd.to_datetime(unmatched['loss_date']).dt.year.value_counts().sort_index().to_dict()}")
        print(f"per-peril reporting CDF (cost-weighted, AY{CDF_ACCIDENT_YEAR}) at 30/90/365 days, and cost share:")
        for peril in sorted(shares, key=shares.get, reverse=True):
            vals = [round(float(cdfs[peril][x]), 4) for x in (30, 90, 365)]
            print(f"  {peril:<18} share={shares[peril]:.4f}  cdf30/90/365={vals}")
        y = pd.to_datetime(p["period_start"]).dt.year.to_numpy()
        s = t.assign(year=y).groupby("year").agg(
            exposure=("exposure", "sum"), completeness=("completeness", "mean"),
            reported=("claim_cost", "sum"), ultimate=("claim_cost_ult", "sum"))
        s["pp_reported"] = s["reported"] / s["exposure"]
        s["pp_ultimate"] = s["ultimate"] / s["exposure"]
        print(s.round(4).to_string())
        print(f"min completeness={t['completeness'].min():.4f}  rows with completeness<0.5: {(t['completeness'] < 0.5).sum():,}")
        if m.name != "developed":
            consistency_check(line, valid, dev_valid, p, cdfs, shares)


def pooled_cdf(valid, year):
    """Cost-weighted reporting CDF of all non-repudiated claims with loss year `year` (C037)."""
    v = valid[pd.to_datetime(valid["loss_date"]).dt.year == year]
    lag = (pd.to_datetime(v["report_date"]) - pd.to_datetime(v["loss_date"])).dt.days
    return development.reporting_cdf(lag, target.net_cost(v)), len(v)


def consistency_check(line, valid, dev_valid, p, cdfs, shares):
    """C037: is the developed reporting pattern usable for the emerging market?

    All CDFs are pooled over perils and cost-weighted. Emerging AY2023 is observed for 12-24 months, so its CDF is
    relative to cost reported by 2024-12-31; developed AY2023 has exactly the same truncation (like-for-like);
    developed AY2021 is the pattern used for the target (relative to cost reported within 36-48 months).
    """
    em, n_em = pooled_cdf(valid, CHECK_YEAR)
    dev23, n_dev23 = pooled_cdf(dev_valid, CHECK_YEAR)
    dev21, n_dev21 = pooled_cdf(dev_valid, CDF_ACCIDENT_YEAR)
    print(f"CDF consistency ({line}; cost-weighted, all perils): emerging AY{CHECK_YEAR} n={n_em:,} | "
          f"developed AY{CHECK_YEAR} n={n_dev23:,} (same truncation) | developed AY{CDF_ACCIDENT_YEAR} n={n_dev21:,} (used)")
    for lag in CHECK_LAGS:
        print(f"  lag {lag:>3} days: emerging {em[lag]:.4f} | developed AY{CHECK_YEAR} {dev23[lag]:.4f} | "
              f"developed AY{CDF_ACCIDENT_YEAR} {dev21[lag]:.4f}")
    # Sensitivity: the same developed per-peril CDFs mixed by the emerging market's own (reported) peril cost shares
    em_shares = development.peril_cost_shares(valid)
    usable = {k: v for k, v in em_shares.items() if k in cdfs}
    usable = {k: v / sum(usable.values()) for k, v in usable.items()}
    year = pd.to_datetime(p["period_start"]).dt.year
    used = development.row_completeness_mixed(p, cdfs, shares)
    alt = development.row_completeness_mixed(p, cdfs, usable)
    print(f"  peril cost shares, emerging 2023-24 reported / developed AY{CDF_ACCIDENT_YEAR}: "
          + ", ".join(f"{k} {em_shares[k]:.3f}/{shares.get(k, 0.0):.3f}"
                      for k in sorted(em_shares, key=em_shares.get, reverse=True)))
    print(f"  sensitivity, mean completeness of 2024 rows: developed peril shares {used[year == 2024].mean():.4f}, "
          f"emerging peril shares {alt[year == 2024].mean():.4f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default=None, choices=sorted(markets.MARKETS),
                    help="C034: default from AIIAA_MARKET, else developed")
    main(ap.parse_args().market)
