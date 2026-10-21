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
    "customer_since_year": [2015, 2024], "surname": ["S1", "S2"], "credit_band": ["A", "C"]})
ADDRESSES = pd.DataFrame({"address_id": ["A1", "A2"], "year_built": [1950, 2010], "region": ["R01", "R02"]})
WEATHER = pd.DataFrame({
    "address_id": ["A1", "A1", "A2"], "year": [2022, 2023, 2025], "hail_days": [3.0, 9.0, 5.0]})


def build(rows=ROWS, history=None):
    return features.build_features(rows, CUSTOMERS, ADDRESSES, WEATHER, history)


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
