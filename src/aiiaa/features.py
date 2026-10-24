"""Model feature matrix for one line of business: policy rows joined to customers, addresses and weather.

Run on train and test rows together so categorical levels are identical in both.
"""
import numpy as np
import pandas as pd

# Never used as features: identifiers, dates, the exposure itself (used as weight), and leakage columns
DROP = [
    "row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_raw",
    "surname", "date_of_birth", "year_built", "customer_since_year",
    "renewal_decision",  # Q9: outcome known only after the row ends
    "customer_service_contacts",  # C008: training values include contacts caused by in-period claims
]

# C026: columns dropped per line. Ablation (LightGBM, folds 2021-22 -> 2023 and 2021-23 -> 2024): categorical columns
# without risk signal were overfitted, and the C024 cross-line / household columns carried no signal.
DROP_BY_LINE = {
    "all": ["branch_id", "previous_insurer", "email_domain", "prior_claims_cust", "prior_cost_cust",
            "prior_exposure_cust", "n_policies_customer", "n_customers_address"],
    "motor": ["make_model"],
    "buildings": ["document_preference", "marital_status", "occupation", "self_employed"],  # C035: flag from occupation
    "contents": ["document_preference", "marital_status", "occupation", "self_employed"],
}

# C035: common schema of the two Part 2 markets (harmonise). Ages are banded with the emerging cut points.
AGE_BANDS = ["17-24", "25-34", "35-49", "50-64", "65+"]
AGE_BAND_DTYPE = pd.CategoricalDtype(AGE_BANDS, ordered=True)
AGE_COLUMNS = {"driver_age": "driver_age_band", "policyholder_age": "policyholder_age_band"}
NOT_IN_EMERGING = {"rows": ["annual_mileage", "telematics_opt_in", "telematics_score"],
                   "customers": ["date_of_birth", "credit_band"]}
OCCUPATIONS = ["accountant", "administrator", "builder", "care worker", "chef", "doctor", "driver", "electrician",
               "engineer", "hairdresser", "manager", "mechanic", "nurse", "retail assistant", "retired",
               "sales representative", "software developer", "student", "teacher", "unemployed"]
SELF_EMPLOYED_PREFIX = "self-employed "


def age_band(age):
    """C035: integer ages to the ordered bands 17-24, 25-34, 35-49, 50-64, 65+ (ages under 17 fall in 17-24)."""
    return pd.cut(age, [-np.inf, 24, 34, 49, 64, np.inf], labels=AGE_BANDS).astype(AGE_BAND_DTYPE)


def clean_occupation(occupation):
    """C035 (N1): free-text occupation to (base job, self_employed). Text is trimmed, inner spaces collapsed and
    lower-cased; a leading "self-employed " becomes self_employed=True; jobs outside OCCUPATIONS become "other".
    Missing occupation stays missing in both outputs (self_employed None)."""
    def one(raw):
        if not isinstance(raw, str):
            return None, None
        text = " ".join(raw.split()).lower()
        job = text.removeprefix(SELF_EMPLOYED_PREFIX)
        return (job if job in OCCUPATIONS else "other"), text.startswith(SELF_EMPLOYED_PREFIX)

    pairs = [one(v) for v in occupation]
    return (pd.Series([p[0] for p in pairs], index=occupation.index, dtype=object),
            pd.Series([p[1] for p in pairs], index=occupation.index, dtype=object))


def harmonise(rows, customers, market):
    """C035: policy rows and customers of either market in the common Part 2 schema.

    Ages become ordered bands (developed ages are banded, emerging bands are validated); columns that the emerging
    market lacks are dropped (so customer_age is no longer derived); gender is "unknown" in the developed market;
    occupation is cleaned into a base job plus a self_employed flag; `market` is added. Band and gender columns are
    moved to the end so both markets give the same column order. Returns new (rows, customers) frames.
    """
    r = rows.copy()
    for age, band in AGE_COLUMNS.items():
        if age in r:
            r[band] = age_band(r.pop(age))
        elif band in r:
            values = r.pop(band)
            unknown = sorted(set(values.dropna()) - set(AGE_BANDS))
            if unknown:
                raise ValueError(f"Unknown {band} values: {unknown}")
            r[band] = values.astype(object).astype(AGE_BAND_DTYPE)
    r = r.drop(columns=[c for c in NOT_IN_EMERGING["rows"] if c in r])
    r["market"] = market
    c = customers.drop(columns=[col for col in NOT_IN_EMERGING["customers"] if col in customers])
    c["gender"] = c.pop("gender") if "gender" in c else "unknown"
    if "occupation" in c:
        c["occupation"], c["self_employed"] = clean_occupation(c["occupation"])
    return r, c


def concat_markets(frames):
    """C035: stack feature frames of different markets; category columns get the union of their levels.
    The result has a fresh RangeIndex (row ids collide across markets). Any other column must have the same dtype in
    every frame (a mismatch, e.g. bool vs float, would otherwise become an object column silently)."""
    if any(list(f.columns) != list(frames[0].columns) for f in frames):
        raise ValueError("Feature frames of the markets have different columns")
    for col in frames[0].columns:
        kinds = {"category" if isinstance(f[col].dtype, pd.CategoricalDtype) else str(f[col].dtype) for f in frames}
        if len(kinds) > 1:
            raise ValueError(f"Column {col} has different dtypes across markets: {sorted(kinds)}")
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if any(isinstance(f[col].dtype, pd.CategoricalDtype) for f in frames) and                 not isinstance(out[col].dtype, pd.CategoricalDtype):
            levels = list(dict.fromkeys(v for f in frames for v in f[col].cat.categories))
            out[col] = pd.Categorical(out[col], categories=levels)
    return out


def prior_experience(rows, history, by="policy_id", suffix=""):
    """Claims, cost and exposure held under the same `by` id in years strictly before each row's start year.

    by="policy_id" (suffix "") is the policy's own history; by="customer_id" (suffix "_cust") is the customer's
    history summed over all lines.
    """
    names = {"prior_claims" + suffix: "claim_count", "prior_cost" + suffix: "claim_cost",
             "prior_exposure" + suffix: "exposure"}
    out = pd.DataFrame({"prior_claims" + suffix: 0, "prior_cost" + suffix: 0.0, "prior_exposure" + suffix: 0.0},
                       index=rows.index)
    if history is None or history.empty:
        return out
    key = pd.DataFrame({by: rows[by], "year": pd.to_datetime(rows["period_start"]).dt.year, "_idx": rows.index})
    m = key.merge(history.rename(columns={"year": "h_year"}), on=by)
    m = m[m["h_year"] < m["year"]]
    agg = m.groupby("_idx").agg(**{new: (src, "sum") for new, src in names.items()})
    out.loc[agg.index] = agg.to_numpy()
    return out.astype({"prior_claims" + suffix: int, "prior_cost" + suffix: float, "prior_exposure" + suffix: float})


def household_counts(rows, portfolio):
    """Distinct policies held by the row's customer and distinct customers at the row's address, in the row's
    start year. Both counts include the row's own policy/customer.
    """
    out = pd.DataFrame({"n_policies_customer": 0, "n_customers_address": 0}, index=rows.index)
    if portfolio is None or portfolio.empty:
        return out
    n_pol = portfolio.groupby(["customer_id", "year"])["policy_id"].nunique().rename("n_policies_customer")
    n_cust = portfolio.groupby(["address_id", "year"])["customer_id"].nunique().rename("n_customers_address")
    key = pd.DataFrame({"customer_id": rows["customer_id"], "address_id": rows["address_id"],
                        "year": pd.to_datetime(rows["period_start"]).dt.year})
    key = key.join(n_pol, on=["customer_id", "year"]).join(n_cust, on=["address_id", "year"])
    return key[["n_policies_customer", "n_customers_address"]].fillna(0).astype(int)


def build_features(rows, customers, addresses, weather, history=None, history_all=None, portfolio=None, line=None):
    """history: this line's per-policy yearly claims. history_all: all lines' yearly claims per customer
    (customer_id, year, claim_count, claim_cost, exposure). portfolio: all lines, train and test rows, one row per
    (policy_id, customer_id, address_id, year) occurrence, used for the household counts.
    line: when given, the C026 per-line drop list (DROP_BY_LINE) is removed from the result.
    """
    start = pd.to_datetime(rows["period_start"])
    year = start.dt.year
    x = rows.copy()
    x["year"] = year
    x = x.merge(customers, on="customer_id", how="left")
    x = x.merge(addresses, on="address_id", how="left")
    x = x.merge(weather, on=["address_id", "year"], how="left")
    x.index = rows.index

    if "date_of_birth" in x:  # C035: harmonised Part 2 tables have no date_of_birth
        x["customer_age"] = (start - pd.to_datetime(x["date_of_birth"])).dt.days / 365.25
    x["customer_tenure"] = year - x["customer_since_year"]
    x["property_age"] = year - x["year_built"]
    x = x.join(prior_experience(rows, history))
    x = x.join(prior_experience(rows, history_all, by="customer_id", suffix="_cust"))
    x = x.join(household_counts(rows, portfolio))
    if "campaign_code" in x:
        x["campaign_code"] = x["campaign_code"].fillna("none")  # missing means no campaign, an explicit level
    if "branch_id" in x:
        x["branch_id"] = x["branch_id"].astype("category")  # servicing branch is a code, not a quantity

    x = x.drop(columns=[c for c in DROP + ["year"] if c in x])
    for col in x.columns:
        if x[col].dtype == object and x[col].dropna().map(type).eq(bool).all():
            x[col] = x[col].map({True: 1.0, False: 0.0}).astype(float)  # C012: boolean with missing values
        elif x[col].dtype == object or pd.api.types.is_string_dtype(x[col]):
            x[col] = x[col].astype("category")
    if line is not None:
        x = x.drop(columns=[c for c in DROP_BY_LINE["all"] + DROP_BY_LINE[line] if c in x])
    return x


def target_encode(key_fit, y_fit, w_fit, key_apply, k=2000.0):
    """C031: smoothed log risk relativity of each code, fitted on the fit rows and applied to other rows.

    For code c: log((sum_c y*w + k*ybar) / (sum_c w + k) / ybar), with ybar the w-weighted mean of y over the fit rows
    and k a credibility constant in units of w (exposure-years). Unseen or missing codes get 0 (the global mean).
    Use only for rows that are NOT in the fit set; fit rows take `target_encode_oof`.
    """
    w_fit = np.asarray(w_fit, dtype=float)
    yw = np.asarray(y_fit, dtype=float) * w_fit
    ybar = yw.sum() / w_fit.sum()
    sums = pd.DataFrame({"key": np.asarray(key_fit, dtype=object), "yw": yw, "w": w_fit}).groupby("key").sum()
    rel = np.log((sums["yw"] + k * ybar) / (sums["w"] + k) / ybar)
    return pd.Series(np.asarray(key_apply, dtype=object)).map(rel).fillna(0.0).to_numpy(dtype=float)


def target_encode_oof(key, y, w, k=2000.0, n_folds=5, seed=42):
    """C031: out-of-fold `target_encode` within the given rows: a seeded random split into n_folds folds, and each
    row is encoded from the other folds only, so no row sees its own target."""
    key, y, w = np.asarray(key, dtype=object), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    fold = np.empty(len(key), dtype=int)
    fold[np.random.default_rng(seed).permutation(len(key))] = np.arange(len(key)) % n_folds
    out = np.zeros(len(key))
    for f in range(n_folds):
        rows = fold == f
        out[rows] = target_encode(key[~rows], y[~rows], w[~rows], key[rows], k=k)
    return out
