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
