"""
IAA AI Assisted Hackathon 2026 - Part 1
Step 1: exploratory data analysis (EDA) + data preparation   -> prepare_data()
Step 2: modelling (GLM, GAM, LightGBM), comparison, submission -> run_models()

Run (from the D: environment):
    D:\\iaa-hackathon\\venv\\Scripts\\python.exe "First code.py"
Step 1 only runs when the "prepared" folder does not exist yet (or FORCE_PREP = True).

Step 1 output (folder "prepared" next to this script):
    freq_<line>.parquet   one row per policy period 2021-2024 -> frequency model
                          targets: claim_count, claim_cost, claim_cost_capped
                          offset : exposure_adj (= exposure x reporting completeness)
                          split  : "train" (2021-2023) / "valid" (2024)
    sev_<line>.parquet    one row per claim with cost > 0 -> severity model
                          targets: net_cost, net_cost_capped
    test_<line>.parquet   one row per 2025 policy period to predict (same features)
    meta.json             feature lists, large-loss caps and loadings
    figures/*.png         EDA charts

Data issues found during exploration and how they are fixed here:
    1. Train exposure is in mixed units (years, days, and a 360-day basis -> 1.0139).
       Test exposure always equals (period days + 1) / 365, so we recompute it from dates.
    2. ~0.2% of sum_insured values are x1000 too large (train AND test) -> divided by 1000.
    3. construction_type has inconsistent spellings (Brick/brick/BRICK...). Train has codes
       W and T (~6% each), test only W (~12%) -> both treated as "timber".
    4. ~10% of motor customer_id lost their leading zeros -> zero-padded to 8 characters.
    5. ~600 duplicated claims (same policy, date, peril and amount, different claim_id).
    6. Claims have no row_id -> each claim is matched to the policy period containing loss_date.
       Claims outside any period (mostly losses in Dec 2020) are dropped.
    7. Data is cut at 2024-12-30, so losses late in 2024 are not all reported yet (IBNR).
       We estimate the reporting delay pattern and shrink exposure accordingly.
    8. A few huge claims (one open bodily-injury reserve of 11.9M) -> severity capped per line
       at the 99.5th percentile; the excess is kept as a multiplicative loading in meta.json.
    Claim cost = incurred_to_date - recoveries. incurred is treated as already net of excess:
    20-27% of claims are below the policy excess (some ~15), which only makes sense after excess.

Step 2 (per line: buildings, contents, motor), metric = Tweedie D2 with power 1.5:
    Validation: fit on 2021-2023, score on 2024 (actual = uncapped claim cost of the row).
    1. GLM : Poisson frequency (offset log exposure) x Gamma severity, numeric variables binned.
             Variables chosen by forward stepwise selection on BIC (or AIC, see SELECTION_CRITERION).
    2. GAM : same selected variables, numeric ones as penalised cubic splines (P-splines),
             smoothing parameter chosen by AIC.
    3. LightGBM : gradient boosted trees, Tweedie(1.5) objective directly on claim cost,
             log(exposure) as offset, all variables.
    + Blend : average of the three.
    The best model per line (highest 2024 D2) is refitted on 2021-2024 and predicts 2025.
    Output: submission.parquet (row_id, prediction) next to this script + prepared/models/*.
"""
import gc
import json
import shutil
import time
from pathlib import Path

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("lightgbm is not installed in this Python. Run the script with the project environment:\n"
                     "  VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' > D:\\iaa-hackathon\\venv\\Scripts\\python.exe\n"
                     "  Terminal: & \"D:\\iaa-hackathon\\venv\\Scripts\\python.exe\" \"D:\\iaa-hackathon\\New folder\\First code.py\"")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, sparse, special
from scipy.interpolate import BSpline
from sklearn.metrics import d2_tweedie_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data ptata"
OUT = HERE / "prepared"
FIG = OUT / "figures"
MODELS = OUT / "models"
SUBMISSION = HERE / "submission.parquet"
FORCE_PREP = False                    # True = rebuild the prepared files (step 1)

LINES = ["buildings", "contents", "motor"]
CUTOFF = pd.Timestamp("2024-12-30")   # last report_date in claims = data extraction date
VALID_YEAR = 2024                     # out-of-time validation year (test is 2025)
LARGE_LOSS_Q = 0.995                  # severity cap quantile, per line
RUN_EDA = True                        # set False to only rebuild the prepared files

ID_COLS = ["row_id", "policy_id", "customer_id", "address_id"]
TARGET_COLS = ["claim_count", "claim_cost", "claim_cost_capped"]
# columns kept in the files but NOT to be used as model features
NON_FEATURES = ID_COLS + TARGET_COLS + [
    "period_start", "period_end", "year", "split", "exposure", "exposure_adj",
    "report_completeness", "exposure_was_wrong", "sum_insured_was_wrong",
]
DROP_CUSTOMER_COLS = ["surname", "date_of_birth", "email_domain"]  # noise / redundant with age

ONE_WAY_FEATURES = {
    "buildings": ["policyholder_age", "sum_insured", "construction_type", "flood_zone", "region",
                  "property_type", "year_built", "accidental_damage_cover", "rainfall_mm"],
    "contents": ["policyholder_age", "sum_insured", "tenure", "security_level", "region",
                 "urban_density", "high_value_items", "credit_band", "accidental_damage_cover"],
    "motor": ["driver_age", "ncb_years", "vehicle_value_group", "annual_mileage", "cover_type",
              "region", "urban_density", "named_young_driver", "credit_band"],
}

# modelling settings
# Target leakage: contacts are made BECAUSE of claims (0 contacts -> 0 claims in 2021-2024,
# claimants average ~3 contacts vs 0.4) and test policies cannot contain 2025 claim contacts.
LEAKY_FEATURES = ["customer_service_contacts"]
METRIC_POWER = 1.5                    # Tweedie power of the competition metric
SELECTION_CRITERION = "BIC"           # "BIC" (fewer variables) or "AIC"
SELECTION_SAMPLE = 500_000            # rows used for the frequency variable selection
MAX_SELECTED = 30                     # max variables per GLM
SELECTION_TRIES = 3                   # candidates refitted per step before the selection stops
SCORE_RCOND = 1e-15                   # score test: ignore near-singular directions (duplicate columns)
N_BINS = 10                           # GLM: numeric variables -> up to 10 quantile bins
N_KNOTS = 6                           # GAM: interior knots per spline
LAMBDA_GRID = [0.1, 1, 10, 100, 1_000, 10_000]
MIN_LEVEL_SHARE = 0.005               # GLM/GAM: levels with < 0.5% weight grouped into "other"
SEED = 2026
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=METRIC_POWER, learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, cat_smooth=50, max_cat_to_onehot=8,
                  verbose=-1, seed=SEED)


# ----------------------------------------------------------------------------- helpers
def rd(name, **kw):
    return pd.read_parquet(DATA / f"{name}.parquet", **kw)


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def to_datetime(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df


def bool_like_to_float(df):
    """Boolean columns containing None arrive as object -> 1.0 / 0.0 / NaN."""
    for c in df.columns:
        if df[c].dtype == object:
            vals = set(df[c].dropna().unique())
            if vals and vals <= {True, False}:
                df[c] = df[c].map({True: 1.0, False: 0.0})
    return df


def shrink(df):
    """Smaller dtypes so the files fit in memory and on disk."""
    for c in df.columns:
        if c in ID_COLS or c in ("claim_id",):
            continue
        s = df[c]
        if s.dtype == "float64":
            df[c] = s.astype("float32")
        elif s.dtype == "int64":
            df[c] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == bool:
            df[c] = s.astype("int8")
        elif s.dtype == object:
            df[c] = s.astype("category")
    return df


def align_categories(train, test):
    """Same category levels in train and test (needed by LightGBM / one-hot encoders)."""
    for c in train.columns:
        if isinstance(train[c].dtype, pd.CategoricalDtype) and c in test.columns:
            levels = sorted(set(train[c].cat.categories) | set(test[c].astype("category").cat.categories))
            train[c] = train[c].cat.set_categories(levels)
            test[c] = test[c].astype("category").cat.set_categories(levels)
    return train, test


# ----------------------------------------------------------------------------- cleaning
def clean_claims(claims):
    section("CLEANING CLAIMS")
    c = to_datetime(claims.copy(), ["loss_date", "report_date"])
    n0 = len(c)
    c = c.drop_duplicates(["policy_id", "loss_date", "peril", "incurred_to_date"])
    print(f"duplicate claims removed          : {n0 - len(c):,}")
    c["net_cost"] = c["incurred_to_date"] - c["recoveries"]
    c["report_lag"] = (c["report_date"] - c["loss_date"]).dt.days
    n1 = len(c)
    c = c[c["status"] != "repudiated"]
    print(f"repudiated claims removed         : {n1 - len(c):,}")
    n2 = len(c)
    c = c[c["net_cost"] > 0]
    print(f"nil claims (net cost <= 0) removed: {n2 - len(c):,}")
    print(f"valid claims kept                 : {len(c):,}")
    return c


def clean_policies(p, line):
    p = to_datetime(p.copy(), ["period_start", "period_end"])
    days = (p["period_end"] - p["period_start"]).dt.days + 1
    p["exposure_was_wrong"] = ((p["exposure"] - days / 365).abs() > 0.003).astype("int8")
    p["exposure"] = days / 365
    p["customer_id"] = p["customer_id"].astype(str).str.zfill(8)
    if "sum_insured" in p:
        wrong = p["sum_insured"] > 2_000_000          # all such values are exactly x1000
        p["sum_insured_was_wrong"] = wrong.astype("int8")
        p.loc[wrong, "sum_insured"] /= 1000
    if "construction_type" in p:
        ct = p["construction_type"].str.lower()
        p["construction_type"] = ct.replace({"w": "timber", "t": "timber"})
    p["year"] = p["period_start"].dt.year
    return p.drop(columns=["renewal_decision"], errors="ignore")


# ----------------------------------------------------------------------------- claims -> policy rows
def assign_claims(claims_line, pol):
    """Attach each claim to the policy period (row_id) whose dates contain its loss_date."""
    m = pd.merge_asof(
        claims_line.sort_values("loss_date"),
        pol[["row_id", "policy_id", "period_start", "period_end"]].sort_values("period_start"),
        left_on="loss_date", right_on="period_start", by="policy_id", direction="backward",
    )
    ok = m["row_id"].notna() & (m["loss_date"] <= m["period_end"])
    print(f"claims matched to a policy period : {ok.sum():,} / {len(m):,}"
          f"  (unmatched by loss year: {m.loc[~ok, 'loss_date'].dt.year.value_counts().sort_index().to_dict()})")
    return m[ok].drop(columns=["period_start", "period_end"])


def reporting_curve(claims_line):
    """F[t] = share of claims reported within t days of the loss (from 2021 losses, fully developed)."""
    lags = np.sort(claims_line.loc[claims_line["loss_date"].dt.year == 2021, "report_lag"].to_numpy())
    t = np.arange(0, 1500)
    return np.searchsorted(lags, t, side="right") / len(lags)


def add_completeness(p, F):
    """Average share of claims already reported at CUTOFF over the days of each policy period."""
    start = pd.Timestamp("2021-01-01")
    grid = pd.date_range(start, CUTOFF)
    days_to_cutoff = (CUTOFF - grid).days.to_numpy()
    daily = F[np.minimum(days_to_cutoff, len(F) - 1)]
    cs = np.concatenate([[0.0], np.cumsum(daily)])
    s = (p["period_start"] - start).dt.days.to_numpy()
    e = (p["period_end"] - start).dt.days.to_numpy()
    p["report_completeness"] = (cs[e + 1] - cs[s]) / (e - s + 1)
    return p


# ----------------------------------------------------------------------------- features
def claim_history(rows, valid_claims, policy_customer, line):
    """Number of claims of the customer in the 365 days before the period start (all lines / same line).
    NaN for periods starting before 2022 (no full history available)."""
    c = valid_claims[["policy_id", "line", "loss_date"]].merge(policy_customer, on="policy_id")
    m = rows[["row_id", "customer_id", "period_start"]].merge(c, on="customer_id")
    m = m[(m["loss_date"] < m["period_start"]) & (m["loss_date"] >= m["period_start"] - pd.Timedelta(days=365))]
    out = rows[["row_id"]].copy()
    any_line = m.groupby("row_id").size()
    same_line = m[m["line"] == line].groupby("row_id").size()
    out["cust_claims_prev_1y"] = out["row_id"].map(any_line).fillna(0)
    out["cust_claims_prev_1y_same_line"] = out["row_id"].map(same_line).fillna(0)
    no_history = (rows["period_start"] < pd.Timestamp("2022-01-01")).to_numpy()
    out.loc[no_history, ["cust_claims_prev_1y", "cust_claims_prev_1y_same_line"]] = np.nan
    return out.drop(columns="row_id")


def add_features(p, customers, addresses, weather, holdings):
    p = (p.merge(customers, on="customer_id", how="left")
          .merge(addresses, on="address_id", how="left")
          .merge(weather, on=["address_id", "year"], how="left")
          .merge(holdings, on=["customer_id", "year"], how="left"))
    p["customer_tenure"] = p["year"] - p["customer_since_year"]
    p["building_age"] = p["year"] - p["year_built"]
    p["log_sum_insured"] = np.log(p["sum_insured"]) if "sum_insured" in p else np.nan
    return p.drop(columns=["customer_since_year"])


# ----------------------------------------------------------------------------- EDA helpers
def overview(name, df):
    na = df.isna().mean()
    na = na[na > 0].round(3).to_dict()
    print(f"{name:28s} rows={len(df):>10,}  cols={df.shape[1]:>3}  missing share: {na if na else '-'}")


def one_way(df, col, bins=8):
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        key = pd.qcut(s, bins, duplicates="drop")      # ordered intervals keep numeric order
    else:
        key = s.astype(object).where(s.notna(), "missing").astype(str)
    g = df.groupby(key, observed=True).agg(exposure=("exposure_adj", "sum"),
                                           claims=("claim_count", "sum"),
                                           cost=("claim_cost_capped", "sum"))
    g.index = g.index.astype(str)
    g = g[g["exposure"] > 0.002 * g["exposure"].sum()]        # hide tiny levels
    g["frequency"] = g["claims"] / g["exposure"]
    g["severity"] = g["cost"] / g["claims"].clip(lower=1)
    return g


def plot_one_ways(df, line):
    feats = [f for f in ONE_WAY_FEATURES[line] if f in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, f in zip(axes.ravel(), feats):
        g = one_way(df, f)
        x = np.arange(len(g))
        ax.bar(x, g["frequency"], color="#4C78A8")
        ax.set_ylabel("frequency (claims / year)", color="#4C78A8")
        ax2 = ax.twinx()
        ax2.plot(x, g["severity"], color="#E45756", marker="o")
        ax2.set_ylabel("avg severity (capped)", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(f)
    fig.suptitle(f"{line}: one-way frequency (bars) and severity (line), 2021-2024")
    fig.tight_layout()
    fig.savefig(FIG / f"one_way_{line}.png", dpi=90)
    plt.close(fig)


def drift_report(train, test, line):
    """Standardised mean difference of numeric features between 2024 and 2025 (test)."""
    last = train[train["year"] == VALID_YEAR]
    rows = []
    for c in test.columns:
        if c in NON_FEATURES or not pd.api.types.is_numeric_dtype(test[c]):
            continue
        sd = last[c].std()
        if sd and sd > 0:
            rows.append((c, last[c].mean(), test[c].mean(), (test[c].mean() - last[c].mean()) / sd))
    d = pd.DataFrame(rows, columns=["feature", "mean_2024", "mean_2025_test", "std_diff"])
    d = d.reindex(d["std_diff"].abs().sort_values(ascending=False).index).head(10)
    print(f"\n[{line}] largest train(2024) -> test(2025) shifts (|std_diff| > 0.1 is worth a look):")
    print(d.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- step 1
def prepare_data():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    free_gb = shutil.disk_usage(OUT).free / 1e9
    print(f"Free disk space: {free_gb:.2f} GB")
    if free_gb < 0.5:
        print("WARNING: less than 0.5 GB free - the prepared files (~150-250 MB) may not fit.")

    section("LOADING RAW DATA")
    claims_raw = rd("claims")
    customers = rd("customers")
    addresses = rd("addresses")
    weather = rd("weather")
    for name, df in [("claims", claims_raw), ("customers", customers), ("addresses", addresses), ("weather", weather)]:
        overview(name, df)
    for line in LINES:
        for kind in ("policies", "test"):
            overview(f"{kind}_{line}", rd(f"{kind}_{line}", columns=None))

    # ---------------------------------------------------------------- claims EDA
    if RUN_EDA:
        section("EDA - CLAIMS (raw)")
        print(pd.crosstab(claims_raw["peril"], claims_raw["line"], margins=True))
        print("\nstatus:", claims_raw["status"].value_counts().to_dict())
        print("\nincurred_to_date by line (quantiles):")
        print(claims_raw.groupby("line")["incurred_to_date"]
              .describe(percentiles=[.5, .9, .99, .999]).round(0).to_string())
        print("\nLargest claims:")
        print(claims_raw.nlargest(8, "incurred_to_date")[
            ["claim_id", "line", "peril", "status", "paid_to_date", "incurred_to_date"]].to_string(index=False))

    claims = clean_claims(claims_raw)
    del claims_raw

    # ---------------------------------------------------------------- shared lookups
    # compact dtypes before joining: motor has ~1M rows and the laptop only has 8 GB RAM
    customers = customers.drop(columns=DROP_CUSTOMER_COLS)
    customers["campaign_code"] = customers["campaign_code"].fillna("none")
    customers = shrink(bool_like_to_float(customers))
    addresses = shrink(bool_like_to_float(addresses))
    weather = weather.astype({c: "float32" for c in weather.columns if c not in ("address_id", "year")})

    # customer holdings per year (from train + test policy files: known at pricing time)
    hold = []
    pol_cust = []
    for line in LINES:
        for kind in ("policies", "test"):
            h = rd(f"{kind}_{line}", columns=["policy_id", "customer_id", "period_start"])
            h["customer_id"] = h["customer_id"].astype(str).str.zfill(8)
            h["year"] = pd.to_datetime(h["period_start"]).dt.year
            h["line"] = line
            hold.append(h[["customer_id", "year", "line"]])
            pol_cust.append(h[["policy_id", "customer_id"]])
    hold = pd.concat(hold).drop_duplicates()
    holdings = pd.crosstab([hold["customer_id"], hold["year"]], hold["line"]).clip(upper=1)
    holdings.columns = [f"holds_{c}" for c in holdings.columns]
    holdings["n_lines_held"] = holdings.sum(axis=1)
    holdings = holdings.reset_index().astype({c: "int8" for c in holdings.columns})
    policy_customer = pd.concat(pol_cust).drop_duplicates("policy_id")
    del hold, pol_cust

    if RUN_EDA:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        for line in LINES:
            cl = claims[claims["line"] == line]
            cl.groupby(cl["loss_date"].dt.to_period("Q")).size().plot(ax=axes[0], label=line, marker="o")
            F = reporting_curve(cl)
            axes[1].plot(np.arange(len(F))[:400], F[:400], label=line)
            axes[2].hist(np.log10(cl["net_cost"]), bins=80, alpha=0.5, label=line)
        axes[0].set_title("valid claims by loss quarter (drop at end = not yet reported)")
        axes[1].set_title("reporting delay: share reported within t days")
        axes[2].set_title("log10(net claim cost)")
        for a in axes:
            a.legend()
        fig.tight_layout()
        fig.savefig(FIG / "claims_overview.png", dpi=90)
        plt.close(fig)

    meta = {"cutoff": str(CUTOFF.date()), "valid_year": VALID_YEAR, "lines": {}}

    # ---------------------------------------------------------------- per line
    for line in LINES:
        section(f"PREPARING {line.upper()}")
        pol = clean_policies(rd(f"policies_{line}"), line)
        test = clean_policies(rd(f"test_{line}"), line)
        print(f"exposure recomputed from dates - rows that were wrong: {pol['exposure_was_wrong'].sum():,}")
        if "sum_insured_was_wrong" in pol:
            print(f"sum_insured /1000 fixed: train {pol['sum_insured_was_wrong'].sum():,}, "
                  f"test {test['sum_insured_was_wrong'].sum():,}")

        cl = claims[claims["line"] == line]
        sev = assign_claims(cl, pol)

        # large-loss cap
        cap = float(sev["net_cost"].quantile(LARGE_LOSS_Q))
        sev["net_cost_capped"] = sev["net_cost"].clip(upper=cap)
        loading = float(sev["net_cost"].sum() / sev["net_cost_capped"].sum())
        print(f"severity cap ({LARGE_LOSS_Q:.1%} quantile) = {cap:,.0f};  "
              f"large-loss loading = {loading:.4f}  (multiply capped predictions by this)")

        # row-level targets (+ counts per peril, for optional peril-level models)
        agg = sev.groupby("row_id").agg(claim_count=("claim_id", "size"),
                                        claim_cost=("net_cost", "sum"),
                                        claim_cost_capped=("net_cost_capped", "sum"))
        peril_counts = pd.crosstab(sev["row_id"], sev["peril"]).add_prefix("n_")
        pol = pol.merge(agg, on="row_id", how="left").merge(peril_counts, on="row_id", how="left")
        fill_cols = list(agg.columns) + list(peril_counts.columns)
        pol[fill_cols] = pol[fill_cols].fillna(0)
        count_cols = ["claim_count"] + list(peril_counts.columns)
        pol[count_cols] = pol[count_cols].astype("int16")

        # reporting delay adjustment of exposure
        F = reporting_curve(cl)
        pol = add_completeness(pol, F)
        pol["exposure_adj"] = pol["exposure"] * pol["report_completeness"]
        test["report_completeness"] = 1.0
        test["exposure_adj"] = test["exposure"]
        pol["split"] = np.where(pol["year"] == VALID_YEAR, "valid", "train")

        # features
        pol, test = shrink(bool_like_to_float(pol)), shrink(bool_like_to_float(test))
        gc.collect()
        pol = pd.concat([pol, claim_history(pol, claims, policy_customer, line)], axis=1)
        test = pd.concat([test, claim_history(test, claims, policy_customer, line)], axis=1)
        pol = add_features(pol, customers, addresses, weather, holdings)
        test = add_features(test, customers, addresses, weather, holdings)
        pol, test = bool_like_to_float(pol), bool_like_to_float(test)
        pol, test = shrink(pol), shrink(test)
        pol, test = align_categories(pol, test)

        # severity table: claim + features of its policy period
        row_feats = pol.drop(columns=TARGET_COLS + list(peril_counts.columns))
        sev = sev[["claim_id", "row_id", "peril", "loss_date", "status", "report_lag",
                   "net_cost", "net_cost_capped"]].merge(row_feats, on="row_id", how="left")
        sev["peril"] = sev["peril"].astype("category")

        # ------------------------------------------------ EDA per line
        if RUN_EDA:
            print("\nBy policy year (frequency uses reporting-adjusted exposure):")
            y = pol.groupby("year").agg(rows=("row_id", "size"), exposure=("exposure", "sum"),
                                        exposure_adj=("exposure_adj", "sum"), claims=("claim_count", "sum"),
                                        cost_capped=("claim_cost_capped", "sum"))
            y["freq_raw"] = y["claims"] / y["exposure"]
            y["freq_adj"] = y["claims"] / y["exposure_adj"]
            y["avg_severity"] = y["cost_capped"] / y["claims"]
            y["loss_cost_per_year"] = y["cost_capped"] / y["exposure_adj"]
            print(y.round(4).to_string())
            print("\nBy peril:")
            pp = sev.groupby("peril", observed=True)["net_cost"].agg(["size", "mean", "median", "max"])
            print(pp.round(0).to_string())
            print(f"\nClaims per row: {pol['claim_count'].value_counts().sort_index().to_dict()}")
            plot_one_ways(pol, line)
            drift_report(pol, test, line)

        # ------------------------------------------------ save
        features = [c for c in test.columns if c not in NON_FEATURES]
        categorical = [c for c in features if isinstance(test[c].dtype, pd.CategoricalDtype)]
        meta["lines"][line] = {
            "features": features,
            "categorical": categorical,
            "peril_count_columns": list(peril_counts.columns),
            "severity_cap": cap,
            "large_loss_loading": loading,
            "n_train_rows": int(len(pol)), "n_test_rows": int(len(test)), "n_claims": int(len(sev)),
        }
        pol.to_parquet(OUT / f"freq_{line}.parquet", compression="zstd", index=False)
        sev.to_parquet(OUT / f"sev_{line}.parquet", compression="zstd", index=False)
        test.to_parquet(OUT / f"test_{line}.parquet", compression="zstd", index=False)
        print(f"\nsaved freq_{line} {pol.shape}, sev_{line} {sev.shape}, test_{line} {test.shape}")
        print(f"{len(features)} features ({len(categorical)} categorical)")
        del pol, test, sev, row_feats
        gc.collect()

    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    section("DONE")
    total = sum(p.stat().st_size for p in OUT.glob("*.parquet")) / 1e6
    print(f"Prepared files in {OUT}  ({total:.0f} MB)")
    print(f"Figures in {FIG}")


# ============================================================================= step 2: modelling
# ----------------------------------------------------------------------------- GLM engine
def deviance(y, mu, power):
    """Total Tweedie deviance (power 1 = Poisson, 2 = Gamma)."""
    mu = np.maximum(mu, 1e-12)
    if power == 1:
        d = special.xlogy(y, y / mu) - y + mu
    elif power == 2:
        d = np.log(mu / y) + y / mu - 1
    else:
        d = (np.maximum(y, 0) ** (2 - power) / ((1 - power) * (2 - power))
             - y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power))
    return 2 * np.sum(d)


def solve_sym(H, b):
    """Solve H x = b with a tiny ridge; falls back to least squares for collinear variables
    (e.g. customer_tenure is identical to renewal_count)."""
    H = H + 1e-9 * (np.trace(H) / len(H) + 1) * np.eye(len(H))
    try:
        return linalg.solve(H, b, assume_a="pos")
    except (linalg.LinAlgError, ValueError):
        return linalg.lstsq(H, b)[0]


def fit_glm(X, y, power, offset=None, P=None, beta=None, max_iter=30, tol=1e-7):
    """Log-link GLM (Poisson power=1 / Gamma power=2) by IRLS on a sparse design matrix.
    P = quadratic penalty matrix (used by the GAM); column 0 of X must be the intercept."""
    n, p = X.shape
    off = np.zeros(n) if offset is None else offset
    P = np.zeros((p, p)) if P is None else P
    if beta is None:
        beta = np.zeros(p)
        beta[0] = np.log(y.sum() / np.exp(off).sum())
    eta = np.clip(X @ beta + off, -30, 30)
    mu = np.exp(eta)
    obj = deviance(y, mu, power) + beta @ P @ beta
    for _ in range(max_iter):
        W = mu ** (2 - power)                       # IRLS weights for log link
        z = eta - off + (y - mu) / mu               # working response
        XtW = X.T.multiply(W).tocsr()
        A = (XtW @ X).toarray()
        new = solve_sym(A + P, XtW @ z)
        step = 1.0
        while True:                                 # step halving if the deviance gets worse
            cand = beta + step * (new - beta)
            eta_c = np.clip(X @ cand + off, -30, 30)
            mu_c = np.exp(eta_c)
            obj_c = deviance(y, mu_c, power) + cand @ P @ cand
            if obj_c <= obj * (1 + 1e-10) or step < 1e-3:
                break
            step /= 2
        done = abs(obj - obj_c) < tol * (abs(obj_c) + 1)
        beta, eta, mu, obj = cand, eta_c, mu_c, obj_c
        if done:
            break
    return {"beta": beta, "dev": deviance(y, mu, power), "mu": mu, "A": A, "P": P}


def edf(model):
    """Effective degrees of freedom (= number of parameters for an unpenalised GLM)."""
    A, P = model["A"], model["P"]
    return float(np.trace(solve_sym(A + P, A)))


def pearson_dispersion(y, mu, power, n_params):
    return float(np.sum((y - mu) ** 2 / mu ** power) / max(len(y) - n_params, 1))


class Design:
    """Turns data frame columns into sparse design blocks, learned on the training data.
    - categorical (or numeric with <= 12 values): one-hot, base level = largest weight,
      rare levels (< MIN_LEVEL_SHARE) grouped into "other", missing = own level
    - other numeric: quantile bins (GLM) or cubic B-splines with a difference penalty (GAM)"""

    def __init__(self, df, features, weight, smooth=False):
        self.features, self.smooth, self.spec = list(features), smooth, {}
        for f in self.features:
            s = df[f]
            continuous = (pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)
                          and s.nunique() > 12)
            if continuous and smooth:
                x = s.to_numpy(float)
                lo, hi = np.nanquantile(x, [0.01, 0.99])
                inner = np.unique(np.nanquantile(np.clip(x, lo, hi), np.linspace(0, 1, N_KNOTS + 2)[1:-1]))
                inner = inner[(inner > lo) & (inner < hi)]
                knots = np.r_[[lo] * 4, inner, [hi] * 4]
                self.spec[f] = {"kind": "spline", "knots": knots, "lo": lo, "hi": hi,
                                "median": float(np.nanmedian(x)), "missing": bool(np.isnan(x).any())}
            elif continuous:
                x = s.to_numpy(float)
                edges = np.unique(np.nanquantile(x, np.linspace(0, 1, N_BINS + 1)[1:-1]))
                spec = {"kind": "bin", "edges": edges}
                self.spec[f] = self._levels(self._keys(s, spec), weight, spec)
            else:
                spec = {"kind": "cat"}
                self.spec[f] = self._levels(self._keys(s, spec), weight, spec)

    @staticmethod
    def _keys(s, spec):
        if spec["kind"] == "bin":
            x = s.to_numpy(float)
            k = np.digitize(x, spec["edges"]).astype(object)
            k[np.isnan(x)] = "missing"
            return pd.Series(k).astype(str)
        return s.astype(object).where(s.notna(), "missing").astype(str).reset_index(drop=True)

    @staticmethod
    def _levels(keys, weight, spec):
        share = pd.Series(weight).groupby(keys.to_numpy()).sum()
        share = share / share.sum()
        small = share[share < MIN_LEVEL_SHARE]
        share = share[share >= MIN_LEVEL_SHARE]
        if len(small):
            share.loc["other"] = share.get("other", 0) + small.sum()
        spec["levels"] = list(share.index)
        spec["base"] = share.idxmax()
        return spec

    def block(self, df, f):
        """Sparse columns for one variable (+ its penalty matrix, None if unpenalised)."""
        spec = self.spec[f]
        if spec["kind"] == "spline":
            x = df[f].to_numpy(float)
            miss = np.isnan(x)
            x = np.clip(np.where(miss, spec["median"], x), spec["lo"], spec["hi"])
            B = BSpline.design_matrix(x, spec["knots"], 3).tocsr()[:, 1:]   # drop 1 col (intercept)
            nb = B.shape[1] + 1
            D = np.diff(np.eye(nb), 2, axis=0)[:, 1:]
            P = D.T @ D
            if spec["missing"]:
                B = sparse.hstack([B, sparse.csr_matrix(miss.astype(float)[:, None])], format="csr")
                P = linalg.block_diag(P, np.zeros((1, 1)))
            return B, P
        keys = self._keys(df[f], spec)
        levels, base = spec["levels"], spec["base"]
        keys = keys.where(keys.isin(levels), "other" if "other" in levels else base)
        cols = {lvl: i for i, lvl in enumerate(l for l in levels if l != base)}
        codes = keys.map(cols).fillna(-1).to_numpy(int)
        rows = np.flatnonzero(codes >= 0)
        B = sparse.csr_matrix((np.ones(rows.size), (rows, codes[rows])), shape=(len(keys), len(cols)))
        return B, None

    def matrix(self, df, lam=0.0):
        blocks, pens = [sparse.csr_matrix(np.ones((len(df), 1)))], [np.zeros((1, 1))]
        for f in self.features:
            B, P = self.block(df, f)
            blocks.append(B)
            pens.append(lam * P if P is not None else np.zeros((B.shape[1], B.shape[1])))
        return sparse.hstack(blocks, format="csr"), linalg.block_diag(*pens)


# ----------------------------------------------------------------------------- variable selection
def score_gains(X1, blocks, names, y, mu, power, phi, pen):
    """Approximate criterion improvement of adding each candidate (Rao score test):
    score statistic ~ drop in deviance/phi, minus the penalty for the extra parameters."""
    W = mu ** (2 - power)
    r = mu ** (1 - power) * (y - mu)
    XtW1 = X1.T.multiply(W).tocsr()
    A11_inv = np.linalg.pinv((XtW1 @ X1).toarray())
    gains = {}
    for f in names:
        Xg = blocks[f]
        XgtW = Xg.T.multiply(W).tocsr()
        Ig1 = (XgtW @ X1).toarray()
        I_eff = (XgtW @ Xg).toarray() - Ig1 @ A11_inv @ Ig1.T
        U = Xg.T @ r
        gains[f] = float(U @ np.linalg.pinv(I_eff, rcond=SCORE_RCOND, hermitian=True) @ U) / phi - pen * Xg.shape[1]
    return gains


def forward_select(df, features, weight, y, power, offset, label):
    """Forward stepwise selection on AIC/BIC = deviance/phi + penalty x n_parameters.
    Candidates are ranked by a score test, the best ones are refitted and kept if the criterion improves."""
    design = Design(df, features, weight, smooth=False)
    blocks = {f: design.block(df, f)[0] for f in features}
    blocks = {f: b for f, b in blocks.items() if b.shape[1] > 0}
    n = len(y)
    pen = np.log(n) if SELECTION_CRITERION == "BIC" else 2.0
    one = sparse.csr_matrix(np.ones((n, 1)))
    if power == 1:
        phi = 1.0
    else:   # Gamma dispersion from the model with all candidates
        full = fit_glm(sparse.hstack([one] + list(blocks.values()), format="csr"), y, power, offset)
        phi = pearson_dispersion(y, full["mu"], power, full["beta"].size)
    X, selected = one, []
    fit = fit_glm(X, y, power, offset)
    crit = fit["dev"] / phi + pen * X.shape[1]
    print(f"  [{label}] forward selection on {SELECTION_CRITERION}, n={n:,}, {len(blocks)} candidates, "
          f"dispersion={phi:.3f}, intercept-only {SELECTION_CRITERION}={crit:,.1f}")
    while len(selected) < MAX_SELECTED:
        remaining = [f for f in blocks if f not in selected]
        if not remaining:
            break
        gains = score_gains(X, blocks, remaining, y, fit["mu"], power, phi, pen)
        accepted = False
        for f in sorted(gains, key=gains.get, reverse=True)[:SELECTION_TRIES]:
            if gains[f] <= 0:
                break
            X_new = sparse.hstack([X, blocks[f]], format="csr")
            beta0 = np.r_[fit["beta"], np.zeros(blocks[f].shape[1])]
            fit_new = fit_glm(X_new, y, power, offset, beta=beta0)
            crit_new = fit_new["dev"] / phi + pen * X_new.shape[1]
            if crit_new < crit:
                selected.append(f)
                X, fit, crit = X_new, fit_new, crit_new
                print(f"     + {f:30s} params={blocks[f].shape[1]:3d}   {SELECTION_CRITERION}={crit:,.1f}")
                accepted = True
                break
        if not accepted:
            break
    print(f"  [{label}] {len(selected)} variables selected")
    return selected, phi


def choose_lambda(df, features, weight, y, power, offset, phi, label):
    """GAM smoothing parameter (one for all splines) minimising AIC = deviance/phi + 2 x edf."""
    design = Design(df, features, weight, smooth=True)
    if not any(s["kind"] == "spline" for s in design.spec.values()):
        return 0.0
    best = (np.inf, None)
    for lam in LAMBDA_GRID:
        X, P = design.matrix(df, lam)
        m = fit_glm(X, y, power, offset, P)
        aic = m["dev"] / phi + 2 * edf(m)
        print(f"  [{label}] lambda={lam:>6}: edf={edf(m):6.1f}  AIC={aic:,.1f}")
        best = min(best, (aic, lam))
    return best[1]


# ----------------------------------------------------------------------------- models
def fit_freq_sev(fr, sv, f_feats, s_feats, smooth=False, lam_f=0.0, lam_s=0.0):
    """Frequency (Poisson, offset log exposure) x severity (Gamma) model, GLM or GAM."""
    dF = Design(fr, f_feats, fr["exposure_adj"].to_numpy(), smooth)
    XF, PF = dF.matrix(fr, lam_f)
    mF = fit_glm(XF, fr["claim_count"].to_numpy(float), 1, np.log(fr["exposure_adj"].to_numpy()), PF)
    dS = Design(sv, s_feats, np.ones(len(sv)), smooth)
    XS, PS = dS.matrix(sv, lam_s)
    mS = fit_glm(XS, sv["net_cost_capped"].to_numpy(float), 2, None, PS)
    n_f, n_s = len(fr), len(sv)
    info = {"freq_edf": edf(mF), "sev_edf": edf(mS),
            "freq_AIC": mF["dev"] + 2 * edf(mF), "freq_BIC": mF["dev"] + np.log(n_f) * edf(mF)}
    return {"dF": dF, "bF": mF["beta"], "dS": dS, "bS": mS["beta"], "info": info}


def predict_freq_sev(m, df, exposure):
    XF, _ = m["dF"].matrix(df)
    XS, _ = m["dS"].matrix(df)
    return np.exp(XF @ m["bF"]) * np.exp(XS @ m["bS"]) * exposure


def fit_lgbm(tr, features, cats, va=None, rounds=5000):
    """Tweedie(1.5) boosting on capped claim cost with log(exposure) offset (init_score)."""
    def arr(s):   # LightGBM needs plain contiguous float64 arrays (a pandas float32 slice crashes it)
        return np.ascontiguousarray(s.to_numpy(dtype=np.float64))

    dtr = lgb.Dataset(tr[features], label=arr(tr["claim_cost_capped"]),
                      init_score=np.log(arr(tr["exposure_adj"])), categorical_feature=cats, free_raw_data=False)
    if va is None:
        return lgb.train(LGB_PARAMS, dtr, rounds)
    dva = lgb.Dataset(va[features], label=arr(va["claim_cost_capped"]),
                      init_score=np.log(arr(va["exposure_adj"])), categorical_feature=cats, reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])


def predict_lgbm(booster, df, features, exposure):
    return np.exp(booster.predict(df[features], raw_score=True) + np.log(exposure))


def severity_trend(sv, target_year):
    """Yearly severity inflation (log-linear fit on average capped cost by policy year),
    returned as the factor that moves the average training year to target_year."""
    g = sv.groupby("year")["net_cost_capped"].agg(["mean", "size"])
    slope = np.polyfit(g.index.to_numpy(float), np.log(g["mean"]), 1, w=np.sqrt(g["size"]))[0]
    mean_year = np.average(g.index, weights=g["size"])
    return float(np.exp(slope * (target_year - mean_year))), float(np.exp(slope) - 1)


def d2(y, pred):
    return d2_tweedie_score(y, np.maximum(pred, 1e-9), power=METRIC_POWER)


def d2_bootstrap_sd(y, pred, n_boot=200):
    """Standard deviation of D2 when the rows are resampled: how much the score moves by chance alone."""
    p = METRIC_POWER

    def unit_dev(y, mu):
        return 2 * (y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p))

    y, pred = np.asarray(y, float), np.maximum(np.asarray(pred, float), 1e-9)
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        scores.append(1 - unit_dev(yi, pred[i]).sum() / unit_dev(yi, np.full(len(yi), yi.mean())).sum())
    return float(np.std(scores))


# ----------------------------------------------------------------------------- step 2 driver
def run_models():
    MODELS.mkdir(exist_ok=True)
    meta = json.loads((OUT / "meta.json").read_text())
    comparison, valid_all, test_preds, selections = [], [], [], {}

    for line in LINES:
        t0 = time.time()
        section(f"MODELLING {line.upper()}")
        info = meta["lines"][line]
        feats = [f for f in info["features"] if f not in LEAKY_FEATURES]
        cats = [c for c in info["categorical"] if c in feats]
        loading = info["large_loss_loading"]
        fr = pd.read_parquet(OUT / f"freq_{line}.parquet")
        sv = pd.read_parquet(OUT / f"sev_{line}.parquet")
        te = pd.read_parquet(OUT / f"test_{line}.parquet")
        fr["exposure_adj"] = fr["exposure_adj"].clip(lower=1e-4)   # periods starting on the cut-off day
        tr, va = fr[fr["split"] == "train"], fr[fr["split"] == "valid"]
        sv_tr = sv[sv["split"] == "train"]
        y_va = va["claim_cost"].to_numpy()
        print(f"train rows {len(tr):,} (claims {len(sv_tr):,}) | valid rows {len(va):,} | test rows {len(te):,}")

        # ---------------- 1. GLM with stepwise variable selection
        samp = tr.sample(min(len(tr), SELECTION_SAMPLE), random_state=SEED)
        f_feats, _ = forward_select(samp, feats, samp["exposure_adj"].to_numpy(),
                                    samp["claim_count"].to_numpy(float), 1,
                                    np.log(samp["exposure_adj"].to_numpy()), "frequency")
        s_feats, phi_s = forward_select(sv_tr, feats, np.ones(len(sv_tr)),
                                        sv_tr["net_cost_capped"].to_numpy(float), 2, None, "severity")
        selections[line] = {"frequency": f_feats, "severity": s_feats}
        glm = fit_freq_sev(tr, sv_tr, f_feats, s_feats)
        preds = {"GLM": predict_freq_sev(glm, va, va["exposure_adj"].to_numpy()) * loading}
        print(f"  GLM fitted  (frequency AIC={glm['info']['freq_AIC']:,.0f}, BIC={glm['info']['freq_BIC']:,.0f})")

        # ---------------- 2. GAM on the selected variables
        lam_f = choose_lambda(samp, f_feats, samp["exposure_adj"].to_numpy(), samp["claim_count"].to_numpy(float),
                              1, np.log(samp["exposure_adj"].to_numpy()), 1.0, "GAM frequency")
        lam_s = choose_lambda(sv_tr, s_feats, np.ones(len(sv_tr)), sv_tr["net_cost_capped"].to_numpy(float),
                              2, None, phi_s, "GAM severity")
        lam_f_full = lam_f * len(tr) / len(samp)          # same smoothness on more data
        gam = fit_freq_sev(tr, sv_tr, f_feats, s_feats, smooth=True, lam_f=lam_f_full, lam_s=lam_s)
        preds["GAM"] = predict_freq_sev(gam, va, va["exposure_adj"].to_numpy()) * loading
        print(f"  GAM fitted  (lambda freq={lam_f}, sev={lam_s}; edf freq={gam['info']['freq_edf']:.0f}, "
              f"sev={gam['info']['sev_edf']:.0f})")

        # ---------------- 3. LightGBM Tweedie
        booster = fit_lgbm(tr, feats, cats, va)
        best_iter = booster.best_iteration
        preds["LightGBM"] = predict_lgbm(booster, va, feats, va["exposure_adj"].to_numpy()) * loading
        print(f"  LightGBM fitted ({best_iter} trees)")
        imp = pd.Series(booster.feature_importance("gain"), index=feats).sort_values(ascending=False)
        print("  top LightGBM variables:", ", ".join(imp.index[:10]))

        preds["Blend"] = (preds["GLM"] + preds["GAM"] + preds["LightGBM"]) / 3
        # constant rate x exposure, for reference
        rate = tr["claim_cost_capped"].sum() / tr["exposure_adj"].sum() * loading
        preds["Constant rate"] = rate * va["exposure_adj"].to_numpy()

        # ---------------- compare on 2024 (with and without severity inflation trend)
        trend_va, yearly = severity_trend(sv_tr, VALID_YEAR)
        print(f"  severity trend {yearly:+.1%} per year -> factor for {VALID_YEAR}: {trend_va:.3f}")
        rows = []
        for name, p in preds.items():
            rows.append({"line": line, "model": name, "D2": d2(y_va, p), "D2_with_trend": d2(y_va, p * trend_va),
                         "pred/actual": p.sum() / y_va.sum()})
        res = pd.DataFrame(rows)
        print("\n" + res.round(4).to_string(index=False))
        comparison.append(res)

        # inflation trend kept for a model only if it improved its 2024 score
        res = res.set_index("model")
        trend_on = {name: bool(res.loc[name, "D2_with_trend"] > res.loc[name, "D2"]) for name in preds}
        cand = res.drop(index="Constant rate")
        best = cand[["D2", "D2_with_trend"]].max(axis=1).idxmax()
        print(f"\n  -> best model for {line}: {best} (trend {'on' if trend_on[best] else 'off'})")
        preds["Best"] = preds[best] * (trend_va if trend_on[best] else 1.0)
        valid_all.append(pd.DataFrame({"line": line, "y": y_va, "trend": trend_va, **preds}))

        # ---------------- refit every model on 2021-2024 and predict 2025
        expo = te["exposure"].to_numpy()
        trend_te, _ = severity_trend(sv, 2025)
        final = {"GLM": predict_freq_sev(fit_freq_sev(fr, sv, f_feats, s_feats), te, expo)}
        gam = fit_freq_sev(fr, sv, f_feats, s_feats, smooth=True,
                           lam_f=lam_f * len(fr) / len(samp), lam_s=lam_s * len(sv) / len(sv_tr))
        final["GAM"] = predict_freq_sev(gam, te, expo)
        booster = fit_lgbm(fr, feats, cats, rounds=int(best_iter * 1.1) + 1)
        final["LightGBM"] = predict_lgbm(booster, te, feats, expo)
        final["Blend"] = (final["GLM"] + final["GAM"] + final["LightGBM"]) / 3
        for name in final:
            final[name] = final[name] * loading * (trend_te if trend_on[name] else 1.0)
        final["Best"] = final[best]
        test_preds.append(pd.DataFrame({"row_id": te["row_id"].to_numpy(), **final}))
        print(f"  2025 predicted totals: " + ", ".join(f"{k} {v.sum():,.0f}" for k, v in final.items())
              + f"   (2024 reported actual {y_va.sum():,.0f})   [{(time.time() - t0) / 60:.1f} min]")
        del fr, sv, te, tr, va, sv_tr, samp
        gc.collect()

    # ---------------- overall comparison (all lines together, like the leaderboard)
    section("MODEL COMPARISON - 2024 validation, all lines")
    comp = pd.concat(comparison)
    v = pd.concat(valid_all)
    overall = []
    for name in ["Constant rate", "GLM", "GAM", "LightGBM", "Blend"]:
        overall.append({"line": "ALL", "model": name, "D2": d2(v["y"], v[name]),
                        "D2_with_trend": d2(v["y"], v[name] * v["trend"]),
                        "pred/actual": v[name].sum() / v["y"].sum()})
    # "Best" already contains the trend decision of each line
    overall.append({"line": "ALL", "model": "Best", "D2": d2(v["y"], v["Best"]),
                    "D2_with_trend": d2(v["y"], v["Best"]), "pred/actual": v["Best"].sum() / v["y"].sum()})
    comp = pd.concat([comp, pd.DataFrame(overall)])
    print(comp.pivot(index="model", columns="line", values="D2_with_trend").round(4).to_string())
    sd = d2_bootstrap_sd(v["y"], v["Best"])
    print(f"\nBest (all lines) D2 = {overall[-1]['D2']:.4f}, bootstrap standard deviation {sd:.4f}: "
          f"a leaderboard score within about +/- {2 * sd:.3f} of this is normal sampling noise")
    comp.to_csv(MODELS / "model_comparison.csv", index=False)
    (MODELS / "selected_variables.json").write_text(json.dumps(selections, indent=2))

    # ---------------- one submission file per model, in the sample_submission order
    section("SUBMISSIONS")
    sample = rd("sample_submission")
    preds = pd.concat(test_preds)
    scores = pd.DataFrame(overall).set_index("model")
    for name in ["GLM", "GAM", "LightGBM", "Blend", "Best"]:
        sub = sample[["row_id"]].merge(preds[["row_id", name]].rename(columns={name: "prediction"}),
                                       on="row_id", how="left")
        assert sub["prediction"].notna().all() and len(sub) == len(sample), f"missing predictions ({name})"
        path = SUBMISSION if name == "Best" else HERE / f"submission_{name}.parquet"
        for attempt in range(5):        # the file can be briefly locked (open in a viewer, antivirus, upload)
            try:
                sub.to_parquet(path, index=False)
                break
            except OSError:
                if attempt == 4:
                    raise SystemExit(f"Could not write {path} - close any program that has it open and re-run.")
                time.sleep(3)
        print(f"saved {path.name:28s} mean {sub['prediction'].mean():7.2f}   "
              f"2024 validation D2 {scores.loc[name, ['D2', 'D2_with_trend']].max():.4f}")


def main():
    if FORCE_PREP or not (OUT / "meta.json").exists():
        prepare_data()
    run_models()


if __name__ == "__main__":
    main()
