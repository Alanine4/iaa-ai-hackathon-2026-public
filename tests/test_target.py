import datetime as dt

import pandas as pd

from aiiaa import target


def d(s):
    return dt.date.fromisoformat(s)


POLICIES = pd.DataFrame({
    "row_id": ["M1", "M2", "M3"],
    "policy_id": ["P1", "P1", "P2"],
    "period_start": [d("2021-01-01"), d("2021-07-01"), d("2021-01-01")],
    "period_end": [d("2021-06-30"), d("2021-12-31"), d("2021-12-31")],
})


def claims(**cols):
    base = {"claim_id": ["C1"], "policy_id": ["P1"], "loss_date": [d("2021-03-01")], "status": ["closed"],
            "incurred_to_date": [1000.0], "recoveries": [0.0]}
    base.update(cols)
    return pd.DataFrame(base)


def test_claim_matched_to_row_covering_loss_date():
    c = claims(claim_id=["C1", "C2"], policy_id=["P1", "P1"], loss_date=[d("2021-03-01"), d("2021-08-01")],
               status=["closed"] * 2, incurred_to_date=[1.0, 2.0], recoveries=[0.0, 0.0])
    m = target.match_claims(POLICIES, c)
    assert dict(zip(m["claim_id"], m["row_id"])) == {"C1": "M1", "C2": "M2"}


def test_claim_on_period_boundary_dates_matched():
    c = claims(claim_id=["C1", "C2"], policy_id=["P1", "P1"], loss_date=[d("2021-06-30"), d("2021-07-01")],
               status=["closed"] * 2, incurred_to_date=[1.0, 2.0], recoveries=[0.0, 0.0])
    m = target.match_claims(POLICIES, c)
    assert dict(zip(m["claim_id"], m["row_id"])) == {"C1": "M1", "C2": "M2"}


def test_unmatched_claim_kept_with_missing_row_id():
    # C005: claims outside any policy row (e.g. 2020 losses) are kept for audit, not silently dropped
    c = claims(loss_date=[d("2020-12-15")])
    m = target.match_claims(POLICIES, c)
    assert len(m) == 1 and m["row_id"].isna().all()


def test_net_cost_is_incurred_minus_recoveries_and_zero_for_repudiated():
    c = claims(claim_id=["C1", "C2"], policy_id=["P1", "P2"], loss_date=[d("2021-03-01")] * 2,
               status=["open", "repudiated"], incurred_to_date=[1000.0, 500.0], recoveries=[150.0, 0.0])
    assert target.net_cost(c).tolist() == [850.0, 0.0]


def test_row_target_sums_claims_and_fills_zero_for_rows_without_claims():
    c = claims(claim_id=["C1", "C2", "C3"], policy_id=["P1", "P1", "P2"],
               loss_date=[d("2021-02-01"), d("2021-03-01"), d("2021-05-01")], status=["closed"] * 3,
               incurred_to_date=[100.0, 200.0, 50.0], recoveries=[0.0, 20.0, 0.0])
    t = target.row_target(POLICIES, c)
    assert t.set_index("row_id")["claim_cost"].to_dict() == {"M1": 280.0, "M2": 0.0, "M3": 50.0}
    assert t.set_index("row_id")["claim_count"].to_dict() == {"M1": 2, "M2": 0, "M3": 1}


def test_repudiated_claims_not_counted():
    c = claims(status=["repudiated"])
    t = target.row_target(POLICIES, c)
    assert t["claim_count"].sum() == 0
