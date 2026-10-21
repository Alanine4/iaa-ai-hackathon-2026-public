import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aiiaa import freqsev


def d(s):
    return dt.date.fromisoformat(s)


POLICIES = pd.DataFrame({
    "row_id": ["M1", "M2", "M3"], "policy_id": ["P1", "P1", "P2"],
    "period_start": [d("2021-01-01"), d("2021-07-01"), d("2021-01-01")],
    "period_end": [d("2021-06-30"), d("2021-12-31"), d("2021-12-31")]}, index=[10, 11, 12])
CLAIMS = pd.DataFrame({
    "claim_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
    "policy_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
    "peril": ["storm", "storm", "fire", "storm", "fire", "flood"],
    "loss_date": [d("2021-02-01"), d("2021-03-01"), d("2021-08-01"), d("2021-05-01"), d("2020-06-01"),
                  d("2021-09-01")],
    "status": ["closed", "open", "closed", "closed", "closed", "repudiated"],
    "incurred_to_date": [80_000.0, 30_000.0, 500.0, 1_000.0, 7_000.0, 90_000.0],
    "recoveries": [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]})


def test_peril_targets_counts_and_caps_each_claim_per_peril():
    # C023: C5 is outside every row (2020 loss) and C6 is repudiated: neither is counted
    pt = freqsev.peril_targets(POLICIES, CLAIMS, cap=50_000)
    assert list(pt.index) == [10, 11, 12]
    assert pt["n_storm"].tolist() == [2, 0, 1]
    assert pt["n_fire"].tolist() == [0, 1, 0]
    assert pt["cost_storm"].tolist() == [80_000.0, 0.0, 1_000.0]
    assert pt["cost_storm_uncapped"].tolist() == [110_000.0, 0.0, 1_000.0]
    assert pt["cost_fire"].tolist() == [0.0, 400.0, 0.0]


def test_peril_targets_has_columns_for_perils_with_only_repudiated_claims():
    pt = freqsev.peril_targets(POLICIES, CLAIMS, cap=50_000)
    assert pt["n_flood"].tolist() == [0, 0, 0]
    assert pt["cost_flood"].tolist() == [0.0, 0.0, 0.0]


def test_peril_targets_default_cap_is_line_quantile():
    pt = freqsev.peril_targets(POLICIES, CLAIMS)
    valid = np.array([80_000.0, 30_000.0, 400.0, 1_000.0, 7_000.0])
    cap = np.quantile(valid, freqsev.CAP_QUANTILE)
    assert pt["cost_storm"].iloc[0] == pytest.approx(min(80_000.0, cap) + 30_000.0)


def synthetic(n=20_000, seed=1, rate_b=0.12):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"freq_a_driver": rng.random(n) < 0.5, "sev_b_driver": rng.random(n) < 0.5,
                      "noise": rng.normal(size=n), "group": pd.Categorical(rng.choice(["u", "v", "w"], n))})
    w = rng.uniform(0.3, 1.0, n)
    n_a = rng.poisson(0.15 * w * np.where(x["freq_a_driver"], 2.0, 1.0))
    n_b = rng.poisson(rate_b * w)
    mean_b = 1_000.0 * np.where(x["sev_b_driver"], 2.0, 1.0)
    cost_a = np.array([rng.gamma(2.0, 400.0, k).sum() for k in n_a])
    cost_b = np.array([rng.gamma(2.0, m / 2.0, k).sum() for k, m in zip(n_b, mean_b)])
    pt = pd.DataFrame({"n_a": n_a, "cost_a": cost_a, "cost_a_uncapped": cost_a * 1.1,
                       "n_b": n_b, "cost_b": cost_b, "cost_b_uncapped": cost_b})
    return x, pt, w


SMALL = dict(n_estimators=100, learning_rate=0.1, freq_params=dict(min_child_samples=500, n_jobs=2),
             sev_params=dict(min_child_samples=100, n_jobs=2))


@pytest.fixture(scope="module")
def fitted():
    x, pt, w = synthetic()
    return x, pt, w, freqsev.FreqSev(**SMALL).fit(x, pt, w)


def test_freqsev_recovers_frequency_and_severity_effects(fitted):
    x, _, _, m = fitted
    r = m.rates(x)
    fa, sb = x["freq_a_driver"].to_numpy(), x["sev_b_driver"].to_numpy()
    assert 1.6 < r["freq_a"][fa].mean() / r["freq_a"][~fa].mean() < 2.5
    assert 1.6 < r["sev_b"][sb].mean() / r["sev_b"][~sb].mean() < 2.5
    # the drivers do not leak into the other components
    assert 0.85 < r["freq_b"][fa].mean() / r["freq_b"][~fa].mean() < 1.15
    assert 0.85 < r["sev_a"][sb].mean() / r["sev_a"][~sb].mean() < 1.15


def test_freqsev_is_balanced_on_training_data(fitted):
    x, pt, w, m = fitted
    r = m.rates(x)
    for p in ["a", "b"]:
        assert (r[f"freq_{p}"] * w).sum() == pytest.approx(pt[f"n_{p}"].sum(), rel=1e-6)
        assert (r[f"freq_{p}"] * w * r[f"sev_{p}"]).sum() == pytest.approx(pt[f"cost_{p}"].sum(), rel=1e-6)
    # predict adds the per-peril large-loss load (peril a: uncapped = 1.1 x capped)
    total = pt["cost_a_uncapped"].sum() + pt["cost_b_uncapped"].sum()
    assert m.predict(x, w).sum() == pytest.approx(total, rel=1e-6)
    assert m.components_["a"]["load"] == pytest.approx(1.1)


def test_freqsev_scales_with_exposure(fitted):
    x, _, w, m = fitted
    assert np.allclose(m.predict(x, 2 * w), 2 * m.predict(x, w))


def test_freqsev_accepts_per_peril_exposure():
    # C023: peril b is only half reported on group "u" rows; its own completeness-weighted exposure removes that
    x, pt, w = synthetic(seed=3)
    rng = np.random.default_rng(3)
    comp_b = np.where(x["group"] == "u", 0.5, 1.0)
    pt["n_b"] = rng.poisson(0.12 * w * comp_b)
    pt["cost_b"] = pt["cost_b_uncapped"] = pt["n_b"] * 1_000.0
    expo = pd.DataFrame({"a": w, "b": w * comp_b})
    m = freqsev.FreqSev(**SMALL).fit(x, pt, expo)
    r = m.rates(x)
    u = (x["group"] == "u").to_numpy()
    assert 0.85 < r["freq_b"][u].mean() / r["freq_b"][~u].mean() < 1.15
    assert (r["freq_b"] * expo["b"]).sum() == pytest.approx(pt["n_b"].sum(), rel=1e-6)
    by_peril = (r["freq_a"] * w * r["sev_a"] * m.components_["a"]["load"]
                + r["freq_b"] * expo["b"] * r["sev_b"] * m.components_["b"]["load"])
    assert np.allclose(m.predict(x, expo), by_peril)


def test_peril_with_few_claims_uses_constant_severity():
    x, pt, w = synthetic(n=20_000, seed=2, rate_b=0.03)  # ~390 claims of peril b
    m = freqsev.FreqSev(**SMALL).fit(x, pt, w)
    assert not isinstance(m.components_["a"]["sev"], float)
    sev_b = m.components_["b"]["sev"]
    assert isinstance(sev_b, float)
    assert sev_b == pytest.approx(pt["cost_b"].sum() / pt["n_b"].sum())
    r = m.rates(x)
    assert np.allclose(r["sev_b"], r["sev_b"][0])
