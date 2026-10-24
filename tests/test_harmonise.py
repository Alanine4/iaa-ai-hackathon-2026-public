"""C035: one feature schema for the developed and emerging markets (features.harmonise)."""
import datetime as dt

import pandas as pd
import pytest

from aiiaa import features


def d(s):
    return dt.date.fromisoformat(s)


def policy_rows(prefix, extra):
    base = pd.DataFrame({
        "row_id": [f"{prefix}1", f"{prefix}2", f"{prefix}3"], "policy_id": ["P1", "P1", "P2"],
        "customer_id": ["00000001", "00000001", "00000002"], "address_id": ["A1", "A1", "A2"],
        "period_start": [d("2023-01-01"), d("2024-01-01"), d("2025-01-01")],
        "period_end": [d("2023-12-31"), d("2024-12-30"), d("2025-12-31")], "exposure": [1.0, 1.0, 1.0]})
    for k, v in extra.items():
        base[k] = v
    return base.assign(fuel_type=["petrol", "diesel", "petrol"], driver_assist=[True, False, True])


DEV_ROWS = policy_rows("D", {"driver_age": [24, 25, 65], "annual_mileage": [8000.0, 9000.0, 12000.0],
                             "telematics_opt_in": [True, False, True], "telematics_score": [55.0, None, 70.0]})
EM_ROWS = policy_rows("E", {"driver_age_band": ["17-24", "25-34", "65+"]})
DEV_CUSTOMERS = pd.DataFrame({
    "customer_id": ["00000001", "00000002"], "surname": ["S1", "S2"],
    "date_of_birth": [d("1990-01-01"), d("1960-01-01")], "occupation": ["Retired", "Teacher"],
    "credit_band": ["A", "C"], "customer_since_year": [2015, 2020]})
EM_CUSTOMERS = pd.DataFrame({
    "customer_id": ["00000001", "00000002"], "surname": ["S1", "S2"], "gender": ["F", "M"],
    "occupation": ["Retired ", "Self-employed Teacher"], "customer_since_year": [2019, 2023]})
ADDRESSES = {"developed": pd.DataFrame({"address_id": ["A1", "A2"], "year_built": [1950, 2010], "region": ["R01", "R02"]}),
             "emerging": pd.DataFrame({"address_id": ["A1", "A2"], "year_built": [1970, 1990], "region": ["E03", "E05"]})}
WEATHER = pd.DataFrame({"address_id": ["A1", "A1", "A2"], "year": [2023, 2024, 2025], "hail_days": [3.0, 9.0, 5.0]})


def harmonised_features(market, line="motor"):
    rows, cust = (DEV_ROWS, DEV_CUSTOMERS) if market == "developed" else (EM_ROWS, EM_CUSTOMERS)
    r, c = features.harmonise(rows, cust, market)
    return features.build_features(r, c, ADDRESSES[market], WEATHER, line=line)


@pytest.mark.parametrize("age, band", [(17, "17-24"), (24, "17-24"), (25, "25-34"), (34, "25-34"), (35, "35-49"),
                                       (49, "35-49"), (50, "50-64"), (64, "50-64"), (65, "65+"), (92, "65+")])
def test_age_band_edges_follow_the_emerging_cut_points(age, band):
    out = features.age_band(pd.Series([age]))
    assert out.iloc[0] == band
    assert out.dtype == features.AGE_BAND_DTYPE and out.dtype.ordered


@pytest.mark.parametrize("raw, job, self_employed", [
    ("Retired ", "retired", False), ("retired", "retired", False), ("Retired", "retired", False),
    ("Self-employed Teacher", "teacher", True), ("self-employed  teacher ", "teacher", True),
    ("  Software   developer", "software developer", False), ("Self-employed retired", "retired", True),
    ("Astronaut", "other", False)])
def test_occupation_trimmed_lowercased_and_self_employed_split_off(raw, job, self_employed):
    jobs, flags = features.clean_occupation(pd.Series([raw]))
    assert jobs.iloc[0] == job
    assert flags.iloc[0] is self_employed


def test_missing_occupation_stays_missing_with_unknown_self_employed():
    jobs, flags = features.clean_occupation(pd.Series(["Nurse", None], dtype=object))
    assert pd.isna(jobs.iloc[1]) and flags.iloc[1] is None


def test_occupations_are_the_twenty_base_jobs():
    assert len(features.OCCUPATIONS) == 20 and "care worker" in features.OCCUPATIONS


def test_harmonised_developed_and_emerging_have_identical_columns_and_dtypes():
    x_dev, x_em = harmonised_features("developed"), harmonised_features("emerging")
    assert list(x_dev.columns) == list(x_em.columns)
    assert [str(t) for t in x_dev.dtypes] == [str(t) for t in x_em.dtypes]
    assert x_dev["driver_age_band"].dtype == x_em["driver_age_band"].dtype == features.AGE_BAND_DTYPE


def test_harmonised_schema_drops_columns_missing_in_emerging_and_adds_market_gender_self_employed():
    x_dev = harmonised_features("developed")
    for col in ["customer_age", "date_of_birth", "credit_band", "driver_age", "annual_mileage", "telematics_opt_in",
                "telematics_score"]:
        assert col not in x_dev
    assert x_dev["driver_age_band"].astype(str).tolist() == ["17-24", "25-34", "65+"]
    assert x_dev["gender"].astype(str).tolist() == ["unknown"] * 3
    assert x_dev["market"].astype(str).tolist() == ["developed"] * 3
    assert x_dev["occupation"].astype(str).tolist() == ["retired", "retired", "teacher"]
    assert x_dev["self_employed"].tolist() == [0.0, 0.0, 0.0]
    assert "customer_tenure" in x_dev and isinstance(x_dev["region"].dtype, pd.CategoricalDtype)


def test_harmonised_emerging_occupation_and_gender():
    x_em = harmonised_features("emerging")
    assert x_em["occupation"].astype(str).tolist() == ["retired", "retired", "teacher"]
    assert x_em["self_employed"].tolist() == [0.0, 0.0, 1.0]
    assert x_em["gender"].astype(str).tolist() == ["F", "F", "M"]


def test_harmonise_policyholder_age_for_home_lines():
    dev = DEV_ROWS.drop(columns=["driver_age"]).assign(policyholder_age=[30, 50, 70])
    em = EM_ROWS.drop(columns=["driver_age_band"]).assign(policyholder_age_band=["25-34", "50-64", "65+"])
    r_dev, _ = features.harmonise(dev, DEV_CUSTOMERS, "developed")
    r_em, _ = features.harmonise(em, EM_CUSTOMERS, "emerging")
    assert r_dev["policyholder_age_band"].tolist() == r_em["policyholder_age_band"].tolist() == ["25-34", "50-64", "65+"]
    assert "policyholder_age" not in r_dev


def test_harmonise_rejects_unknown_age_band():
    with pytest.raises(ValueError, match="70-79"):
        features.harmonise(EM_ROWS.assign(driver_age_band=["17-24", "70-79", "65+"]), EM_CUSTOMERS, "emerging")


def test_harmonise_does_not_modify_its_inputs():
    before = DEV_ROWS.copy(), DEV_CUSTOMERS.copy()
    features.harmonise(DEV_ROWS, DEV_CUSTOMERS, "developed")
    pd.testing.assert_frame_equal(DEV_ROWS, before[0])
    pd.testing.assert_frame_equal(DEV_CUSTOMERS, before[1])


def test_home_lines_drop_self_employed_with_occupation():
    # C026 drops occupation for buildings/contents; the C035 self_employed flag comes from the same column
    x = harmonised_features("emerging", line="buildings")
    assert "occupation" not in x and "self_employed" not in x


def test_concat_markets_keeps_categories_as_union_of_levels():
    x = features.concat_markets([harmonised_features("developed"), harmonised_features("emerging")])
    assert len(x) == 6 and x.index.is_unique
    assert set(x["region"].cat.categories) == {"R01", "R02", "E03", "E05"}
    assert set(x["gender"].cat.categories) == {"unknown", "F", "M"}
    assert set(x["market"].cat.categories) == {"developed", "emerging"}
    assert x["driver_age_band"].dtype == features.AGE_BAND_DTYPE
    assert not any(t == object for t in x.dtypes)
    assert x["region"].astype(str).tolist() == ["R01", "R01", "R02", "E03", "E03", "E05"]


def test_concat_markets_rejects_columns_whose_dtype_differs_between_markets():
    # C035: e.g. a boolean column with missing values in one market only (float there, bool in the other)
    a = pd.DataFrame({"flag": [True, False], "c": pd.Categorical(["x", "y"])})
    b = pd.DataFrame({"flag": [1.0, float("nan")], "c": pd.Categorical(["z", "x"])})
    with pytest.raises(ValueError, match="flag"):
        features.concat_markets([a, b])
