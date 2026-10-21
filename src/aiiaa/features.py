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


def build_features(rows, customers, addresses, weather, history=None, history_all=None, portfolio=None):
    """history: this line's per-policy yearly claims. history_all: all lines' yearly claims per customer
    (customer_id, year, claim_count, claim_cost, exposure). portfolio: all lines, train and test rows, one row per
    (policy_id, customer_id, address_id, year) occurrence, used for the household counts.
    """
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
    return x
