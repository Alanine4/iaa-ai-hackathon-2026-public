import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aiiaa import target


def d(s):
    return dt.date.fromisoformat(s)


POLICIES = pd.DataFrame({
    "row_id": ["M1", "M2"], "policy_id": ["P1", "P2"],
    "period_start": [d("2021-01-01")] * 2, "period_end": [d("2021-12-31")] * 2})
CLAIMS = pd.DataFrame({
    "claim_id": ["C1", "C2", "C3", "C4"], "policy_id": ["P1", "P1", "P2", "P2"],
    "loss_date": [d("2021-02-01")] * 4, "status": ["closed", "open", "closed", "repudiated"],
    "incurred_to_date": [80_000.0, 30_000.0, 500.0, 90_000.0], "recoveries": [0.0, 0.0, 100.0, 0.0]})


def test_capped_row_cost_caps_each_claim_before_summing():
    # C014: each claim capped at the threshold, then summed per row; repudiated claims excluded
    t = target.row_target(POLICIES, CLAIMS, cap=50_000)
    assert t.set_index("row_id")["claim_cost_capped"].to_dict() == {"M1": 80_000.0, "M2": 400.0}
    assert t.set_index("row_id")["claim_cost"].to_dict() == {"M1": 110_000.0, "M2": 400.0}


def test_capped_cost_equals_uncapped_without_cap():
    t = target.row_target(POLICIES, CLAIMS)
    assert (t["claim_cost_capped"] == t["claim_cost"]).all()


def test_large_loss_cap_is_quantile_of_non_repudiated_net_costs():
    cap = target.large_loss_cap(CLAIMS, q=0.5)
    assert cap == pytest.approx(np.quantile([80_000.0, 30_000.0, 400.0], 0.5))
