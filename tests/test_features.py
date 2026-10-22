import datetime as dt

import pandas as pd
import pytest

from aiiaa import features


def d(s):
    return dt.date.fromisoformat(s)


ROWS = pd.DataFrame({
    "row_id": ["M1", "M2", "M3"],
    "policy_id": ["P1", "P1", "P2"],
    "customer_id": ["00000001", "00000001", "00000002"],
    "address_id": ["A1", "A1", "A2"],
    "period_start": [d("2022-01-01"), d("2023-07-01"), d("2025-01-01")],
    "period_end": [d("2022-12-31"), d("2023-12-31"), d("2025-12-31")],
    "exposure": [1.0, 0.5, 1.0],
    "fuel_type": ["petrol", "diesel", "petrol"],
    "renewal_decision": ["renewed", "renewed", None],
    "customer_service_contacts": [0, 3, 1],
})
CUSTOMERS = pd.DataFrame({
    "customer_id": ["00000001", "00000002"], "date_of_birth": [d("1980-07-01"), d("2000-01-01")],
    "customer_since_year": [2015, 2024], "surname": ["S1", "S2"], "credit_band": ["A", "C"],
    "branch_id": [7, 12], "campaign_code": ["CMP01", None],
    "document_preference": ["email", "post"], "previous_insurer": ["X", "Y"], "email_domain": ["a.com", "b.com"]})
ADDRESSES = pd.DataFrame({"address_id": ["A1", "A2"], "year_built": [1950, 2010], "region": ["R01", "R02"]})
WEATHER = pd.DataFrame({
    "address_id": ["A1", "A1", "A2"], "year": [2022, 2023, 2025], "hail_days": [3.0, 9.0, 5.0]})


def build(rows=ROWS, history=None, history_all=None, portfolio=None, line=None):
    return features.build_features(rows, CUSTOMERS, ADDRESSES, WEATHER, history, history_all, portfolio, line)


def test_weather_joined_on_row_start_year():
    assert build()["hail_days"].tolist() == [3.0, 9.0, 5.0]


def test_customer_age_and_tenure_at_period_start():
    x = build()
    assert x["customer_age"].round(2).tolist() == pytest.approx([41.5, 43.0, 25.0], abs=0.01)
    assert x["customer_tenure"].tolist() == [7, 8, 1]


def test_property_age_from_year_built():
    assert build()["property_age"].tolist() == [72, 73, 15]


def test_leakage_identifier_and_date_columns_removed():
    x = build()
    for col in ["renewal_decision", "customer_service_contacts", "surname", "date_of_birth", "row_id",
                "policy_id", "customer_id", "address_id", "period_start", "period_end", "exposure"]:
        assert col not in x


def test_text_columns_become_categories():
    x = build()
    assert isinstance(x["fuel_type"].dtype, pd.CategoricalDtype)
    assert isinstance(x["region"].dtype, pd.CategoricalDtype)


def test_boolean_column_with_missing_values_becomes_numeric_not_category():
    # C012: smoke_alarms holds True/False/None (object dtype); XGBoost rejects boolean category levels
    rows = ROWS.assign(smoke_alarms=pd.Series([True, None, False], dtype=object))
    x = build(rows)
    assert pd.api.types.is_float_dtype(x["smoke_alarms"])
    assert x["smoke_alarms"].tolist()[0::2] == [1.0, 0.0] and pd.isna(x["smoke_alarms"].iloc[1])


def test_prior_experience_uses_only_earlier_years_of_same_policy():
    history = pd.DataFrame({
        "policy_id": ["P1", "P1", "P1", "P2"], "year": [2021, 2022, 2023, 2024],
        "claim_count": [2, 1, 5, 1], "claim_cost": [100.0, 50.0, 900.0, 10.0], "exposure": [1.0, 1.0, 1.0, 1.0]})
    x = build(history=history)
    # M1 (2022) sees 2021 only; M2 (2023) sees 2021-2022; M3 (P2, 2025) sees 2024
    assert x["prior_claims"].tolist() == [2, 3, 1]
    assert x["prior_exposure"].tolist() == [1.0, 2.0, 1.0]
    assert x["prior_cost"].tolist() == [100.0, 150.0, 10.0]


def test_prior_experience_zero_when_no_history():
    x = build(history=None)
    assert x["prior_claims"].tolist() == [0, 0, 0]
    assert x["prior_exposure"].tolist() == [0.0, 0.0, 0.0]


def test_cross_line_prior_experience_sums_all_lines_of_same_customer_in_earlier_years():
    history_all = pd.DataFrame({
        "customer_id": ["00000001", "00000001", "00000001", "00000001", "00000002"],
        "year": [2021, 2021, 2022, 2023, 2024],
        "claim_count": [1, 2, 4, 9, 1], "claim_cost": [10.0, 5.0, 20.0, 900.0, 7.0],
        "exposure": [1.0, 0.5, 1.0, 1.0, 0.5]})
    x = build(history_all=history_all)
    # M1 (cust 1, 2022) sees 2021 of two lines; M2 (cust 1, 2023) sees 2021-2022; M3 (cust 2, 2025) sees 2024
    assert x["prior_claims_cust"].tolist() == [3, 7, 1]
    assert x["prior_cost_cust"].tolist() == [15.0, 35.0, 7.0]
    assert x["prior_exposure_cust"].tolist() == [1.5, 2.5, 0.5]


def test_cross_line_prior_experience_zero_when_no_history_all():
    x = build(history_all=None)
    assert x["prior_claims_cust"].tolist() == [0, 0, 0]
    assert x["prior_cost_cust"].tolist() == [0.0, 0.0, 0.0]
    assert x["prior_exposure_cust"].tolist() == [0.0, 0.0, 0.0]


PORTFOLIO = pd.DataFrame({
    "policy_id": ["P1", "P1", "P1b", "P3", "P1", "P2", "P2b"],
    "customer_id": ["00000001", "00000001", "00000001", "00000003", "00000001", "00000002", "00000002"],
    "address_id": ["A1", "A1", "A1", "A1", "A1", "A2", "A2"],
    "year": [2022, 2022, 2022, 2022, 2023, 2025, 2025]})


def test_household_counts_use_distinct_policies_and_customers_in_row_start_year():
    x = build(portfolio=PORTFOLIO)
    # 2022: cust 1 holds P1 (listed twice) + P1b; A1 hosts customers 1 and 3. 2023: only P1. 2025: cust 2 holds P2 + P2b at A2
    assert x["n_policies_customer"].tolist() == [2, 1, 2]
    assert x["n_customers_address"].tolist() == [2, 1, 1]


def test_household_counts_zero_when_no_portfolio():
    x = build(portfolio=None)
    assert x["n_policies_customer"].tolist() == [0, 0, 0]
    assert x["n_customers_address"].tolist() == [0, 0, 0]


def test_branch_id_is_category_not_number():
    x = build()
    assert isinstance(x["branch_id"].dtype, pd.CategoricalDtype)
    assert set(x["branch_id"].astype(int)) == {7, 12}


def test_missing_campaign_code_becomes_explicit_none_level():
    x = build()
    assert isinstance(x["campaign_code"].dtype, pd.CategoricalDtype)
    assert x["campaign_code"].astype(str).tolist() == ["CMP01", "CMP01", "none"]
    assert not x["campaign_code"].isna().any()


MOTOR_ROWS = ROWS.assign(make_model=["Ford Fiesta", "Ford Fiesta", "VW Golf"])
CROSS_LINE = ["prior_claims_cust", "prior_cost_cust", "prior_exposure_cust", "n_policies_customer",
              "n_customers_address"]


def test_motor_line_drops_overfitted_columns_and_keeps_risk_columns():
    # C026
    x = build(MOTOR_ROWS, line="motor")
    for col in ["make_model", "branch_id", "previous_insurer", "email_domain"] + CROSS_LINE:
        assert col not in x
    assert "fuel_type" in x and "credit_band" in x and "prior_claims" in x


def test_line_specific_drops_apply_only_to_that_line():
    # C026: document_preference is dropped for buildings and contents, kept for motor (and make_model only dropped for motor)
    assert "document_preference" not in build(line="buildings")
    assert "document_preference" not in build(line="contents")
    assert "document_preference" in build(MOTOR_ROWS, line="motor")
    assert "make_model" in build(MOTOR_ROWS, line="buildings")


def test_no_line_keeps_every_column():
    # C026: line=None is the unchanged behaviour
    x = build(MOTOR_ROWS)
    for col in ["make_model", "branch_id", "previous_insurer", "email_domain", "document_preference"] + CROSS_LINE:
        assert col in x
