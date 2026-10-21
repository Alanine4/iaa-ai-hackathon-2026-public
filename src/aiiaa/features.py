"""Model feature matrix for one line of business: policy rows joined to customers, addresses and weather.

Run on train and test rows together so categorical levels are identical in both.
"""
import pandas as pd

# Never used as features: identifiers, dates, the exposure itself (used as weight), and leakage columns
DROP = [
    "row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure", "exposure_raw",
    "surname", "date_of_birth", "year_built", "customer_since_year",
    "renewal_decision",  # Q9: outcome known only after the row ends
    "customer_service_contacts",  # C008: training values include contacts caused by in-period claims
]


def prior_experience(rows, history):
    """Claims, cost and exposure of the same policy in years strictly before each row's start year."""
    out = pd.DataFrame({"prior_claims": 0, "prior_cost": 0.0, "prior_exposure": 0.0}, index=rows.index)
    if history is None or history.empty:
        return out
    key = pd.DataFrame({"policy_id": rows["policy_id"], "year": pd.to_datetime(rows["period_start"]).dt.year,
                        "_idx": rows.index})
    m = key.merge(history.rename(columns={"year": "h_year"}), on="policy_id")
    m = m[m["h_year"] < m["year"]]
    agg = m.groupby("_idx").agg(prior_claims=("claim_count", "sum"), prior_cost=("claim_cost", "sum"),
                                prior_exposure=("exposure", "sum"))
    out.loc[agg.index] = agg.to_numpy()
    return out.astype({"prior_claims": int, "prior_cost": float, "prior_exposure": float})


def build_features(rows, customers, addresses, weather, history=None):
    start = pd.to_datetime(rows["period_start"])
    year = start.dt.year
    x = rows.copy()
    x["year"] = year
    x = x.merge(customers, on="customer_id", how="left")
    x = x.merge(addresses, on="address_id", how="left")
    x = x.merge(weather, on=["address_id", "year"], how="left")
    x.index = rows.index

    x["customer_age"] = (start - pd.to_datetime(x["date_of_birth"])).dt.days / 365.25
    x["customer_tenure"] = year - x["customer_since_year"]
    x["property_age"] = year - x["year_built"]
    x = x.join(prior_experience(rows, history))

    x = x.drop(columns=[c for c in DROP + ["year"] if c in x])
    for col in x.columns:
        if x[col].dtype == object and x[col].dropna().map(type).eq(bool).all():
            x[col] = x[col].map({True: 1.0, False: 0.0}).astype(float)  # C012: boolean with missing values
        elif x[col].dtype == object or pd.api.types.is_string_dtype(x[col]):
            x[col] = x[col].astype("category")
    return x
