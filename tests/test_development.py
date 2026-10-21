import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aiiaa import development


def d(s):
    return dt.date.fromisoformat(s)


def test_reporting_cdf_cost_weighted_step_function():
    lags = pd.Series([0, 10, 10, 30])
    weights = pd.Series([1.0, 1.0, 2.0, 4.0])
    cdf = development.reporting_cdf(lags, weights, max_lag=40)
    assert cdf[0] == pytest.approx(1 / 8)
    assert cdf[9] == pytest.approx(1 / 8)
    assert cdf[10] == pytest.approx(4 / 8)
    assert cdf[29] == pytest.approx(4 / 8)
    assert cdf[30] == pytest.approx(1.0)
    assert cdf[40] == pytest.approx(1.0)
    assert len(cdf) == 41


def test_row_completeness_is_one_for_rows_far_before_valuation():
    cdf = np.array([0.5] * 10 + [1.0] * 91)  # fully reported after 10 days
    rows = pd.DataFrame({"period_start": [d("2024-01-01")], "period_end": [d("2024-06-30")]})
    out = development.row_completeness(rows, cdf, valuation=d("2024-12-31"))
    assert out.tolist() == pytest.approx([1.0])


def test_row_completeness_averages_cdf_over_days_in_period():
    cdf = np.array([0.2, 0.4, 0.6, 1.0])
    # days 12-29..12-31 have lags 2, 1, 0 at valuation 12-31 -> mean(0.6, 0.4, 0.2)
    rows = pd.DataFrame({"period_start": [d("2024-12-29")], "period_end": [d("2024-12-31")]})
    out = development.row_completeness(rows, cdf, valuation=d("2024-12-31"))
    assert out.tolist() == pytest.approx([0.4])


def test_row_completeness_uses_last_cdf_value_beyond_max_lag():
    cdf = np.array([0.2, 0.9])
    rows = pd.DataFrame({"period_start": [d("2024-12-01")], "period_end": [d("2024-12-01")]})
    out = development.row_completeness(rows, cdf, valuation=d("2024-12-31"))
    assert out.tolist() == pytest.approx([0.9])


def test_develop_scales_reported_cost_by_completeness():
    t = pd.DataFrame({"claim_cost": [100.0, 0.0], "completeness": [0.8, 0.5], "exposure": [1.0, 0.5]})
    out = development.develop(t)
    assert out["claim_cost_ult"].tolist() == pytest.approx([125.0, 0.0])
    assert out["exposure_eff"].tolist() == pytest.approx([0.8, 0.25])


def make_claims(rows):
    """rows: (peril, lag_days, incurred, status) -> minimal claims frame."""
    return pd.DataFrame({
        "peril": [r[0] for r in rows],
        "loss_date": [d("2021-03-01")] * len(rows),
        "report_date": [d("2021-03-01") + dt.timedelta(days=r[1]) for r in rows],
        "incurred_to_date": [float(r[2]) for r in rows],
        "recoveries": [0.0] * len(rows),
        "status": [r[3] for r in rows],
    })


def test_cdf_by_peril_equals_single_cdf_when_only_one_peril():
    claims = make_claims([("fire", 0, 100, "closed"), ("fire", 10, 100, "closed"), ("fire", 30, 200, "open")])
    out = development.reporting_cdf_by_peril(claims, max_lag=40)
    expected = development.reporting_cdf(pd.Series([0, 10, 30]), pd.Series([100.0, 100.0, 200.0]), max_lag=40)
    assert list(out) == ["fire"]
    assert out["fire"] == pytest.approx(expected)


def test_cdf_by_peril_small_peril_uses_pooled_cdf():
    n = development.MIN_PERIL_CLAIMS
    rows = [("big", 0, 100, "closed")] * n + [("small", 20, 100, "closed")] * (n - 1)
    out = development.reporting_cdf_by_peril(make_claims(rows), max_lag=40)
    pooled = development.reporting_cdf(pd.Series([0] * n + [20] * (n - 1)), pd.Series([100.0] * (2 * n - 1)), max_lag=40)
    assert out["small"] == pytest.approx(pooled)
    assert out["big"][0] == pytest.approx(1.0)  # big has enough claims, keeps its own CDF
    assert out["small"][0] != pytest.approx(out["big"][0])


def test_peril_with_exactly_min_claims_keeps_own_cdf():
    n = development.MIN_PERIL_CLAIMS
    rows = [("a", 0, 100, "closed")] * n + [("b", 20, 100, "closed")] * n
    out = development.reporting_cdf_by_peril(make_claims(rows), max_lag=40)
    assert out["b"][19] == pytest.approx(0.0)
    assert out["b"][20] == pytest.approx(1.0)


def test_peril_cost_shares_sum_to_one_and_exclude_repudiated():
    claims = make_claims([
        ("fire", 0, 300, "closed"), ("flood", 0, 100, "open"), ("theft", 0, 9999, "repudiated"),
    ])
    claims.loc[0, "recoveries"] = 100.0  # fire net cost 200
    shares = development.peril_cost_shares(claims)
    assert sorted(shares) == ["fire", "flood"]
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["fire"] == pytest.approx(200 / 300)
    assert shares["flood"] == pytest.approx(100 / 300)


def test_cdf_by_peril_ignores_repudiated_claims():
    claims = make_claims([("fire", 0, 100, "closed"), ("fire", 30, 5000, "repudiated")])
    out = development.reporting_cdf_by_peril(claims, max_lag=40)
    assert out["fire"][0] == pytest.approx(1.0)


def test_row_completeness_mixed_is_share_weighted_average_of_peril_completeness():
    cdfs = {"a": np.array([0.5, 1.0]), "b": np.array([0.0, 0.5, 1.0])}
    shares = {"a": 0.25, "b": 0.75}
    rows = pd.DataFrame({
        "period_start": [d("2024-12-30"), d("2024-12-30")],
        "period_end": [d("2024-12-30"), d("2024-12-31")],
    })
    out = development.row_completeness_mixed(rows, cdfs, shares, valuation=d("2024-12-31"))
    # row 0: lag 1 -> a = 1.0, b = 0.5 -> 0.25*1.0 + 0.75*0.5
    # row 1: lags 1, 0 -> a = mean(1.0, 0.5) = 0.75, b = mean(0.5, 0.0) = 0.25 -> 0.25*0.75 + 0.75*0.25
    assert out.tolist() == pytest.approx([0.625, 0.375])


def test_row_completeness_mixed_raises_when_peril_missing_from_cdfs():
    rows = pd.DataFrame({"period_start": [d("2024-12-30")], "period_end": [d("2024-12-31")]})
    with pytest.raises(ValueError, match="flood"):
        development.row_completeness_mixed(rows, {"fire": np.array([1.0])}, {"fire": 0.5, "flood": 0.5})
