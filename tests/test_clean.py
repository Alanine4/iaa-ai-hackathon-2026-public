import datetime as dt

import pandas as pd
import pytest

from aiiaa import clean


def d(s):
    return dt.date.fromisoformat(s)


def test_exposure_recomputed_from_period_dates():
    # C001: exposure stored as days (365.0), as 370/365 (1.0139) or on a 360-day basis
    df = pd.DataFrame({
        "period_start": [d("2021-01-01"), d("2022-01-01"), d("2021-01-01"), d("2023-03-01")],
        "period_end": [d("2021-12-31"), d("2022-12-31"), d("2021-06-21"), d("2023-03-31")],
        "exposure": [365.0, 1.0139, 0.4778, 0.0849],
    })
    out = clean.fix_exposure(df)
    assert out["exposure"].round(4).tolist() == [1.0, 1.0, round(172 / 365, 4), round(31 / 365, 4)]


def test_exposure_keeps_original_value_for_audit():
    df = pd.DataFrame({"period_start": [d("2021-01-01")], "period_end": [d("2021-12-31")], "exposure": [365.0]})
    out = clean.fix_exposure(df)
    assert out["exposure_raw"].tolist() == [365.0]


def test_customer_id_left_padded_to_eight_digits():
    # C002: leading zeros lost in motor customer_id
    df = pd.DataFrame({"customer_id": ["5180324", "88505748", "11"]})
    assert clean.fix_customer_id(df)["customer_id"].tolist() == ["05180324", "88505748", "00000011"]


def test_sum_insured_above_threshold_divided_by_1000_and_flagged():
    # C003: some sum_insured values are scaled up by exactly 1000
    df = pd.DataFrame({"sum_insured": [223000.0, 220000000.0, 953000.0, 8200000.0]})
    out = clean.fix_sum_insured(df)
    assert out["sum_insured"].tolist() == [223000.0, 220000.0, 953000.0, 8200.0]
    assert out["sum_insured_scaled"].tolist() == [False, True, False, True]


def test_construction_type_case_normalised_and_codes_mapped_to_timber():
    # C004: mixed case; T (2021-22) and W (2023-24) are the same category, mapped to timber
    df = pd.DataFrame({"construction_type": ["Brick", "BRICK", "brick", "STONE", "NON_STANDARD", "T", "t", "W", "w"]})
    assert clean.fix_construction_type(df)["construction_type"].tolist() == [
        "brick", "brick", "brick", "stone", "non_standard", "timber", "timber", "timber", "timber"]


def test_construction_type_unknown_code_raises():
    df = pd.DataFrame({"construction_type": ["Brick", "X"]})
    with pytest.raises(ValueError, match="X"):
        clean.fix_construction_type(df)


def test_clean_policies_applies_only_relevant_fixes():
    motor = pd.DataFrame({
        "customer_id": ["123"], "period_start": [d("2025-01-01")], "period_end": [d("2025-12-31")], "exposure": [1.0]})
    out = clean.clean_policies(motor)
    assert out["customer_id"].tolist() == ["00000123"]
    assert "sum_insured_scaled" not in out


def test_emerging_construction_codes_map_to_the_developed_categories():
    # C035: emerging local codes (Kaggle Part 2 data page): BRK brick, STN stone, WOD timber, MIX mixed / non-standard
    df = pd.DataFrame({"construction_type": ["BRK", "STN", "WOD", "MIX", "brk"]})
    assert clean.fix_construction_type(df)["construction_type"].tolist() == [
        "brick", "stone", "timber", "non_standard", "brick"]


def test_emerging_construction_unknown_code_still_raises():
    df = pd.DataFrame({"construction_type": ["BRK", "CON"]})
    with pytest.raises(ValueError, match="CON"):
        clean.fix_construction_type(df)
