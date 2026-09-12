"""Size and power of the backtests on synthetic P&L with a KNOWN distribution."""

import numpy as np
import pytest
from scipy.stats import norm

from riskkit import backtest as bt
from riskkit.positions import Book, Market
from riskkit.synth import load_demo

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/demo"
Z99, ES99 = norm.ppf(0.99), norm.pdf(norm.ppf(0.99)) / 0.01   # true 99 % VaR / ES of N(0, 1)


def _iid_exceptions(rng, n, p, m):
    return rng.random((m, n)) < p


def _markov_exceptions(rng, n, p11, p01, m):
    """First-order Markov chain of exception indicators: clustered when p11 >> p01."""
    out = np.zeros((m, n), dtype=bool)
    u = rng.random((m, n))
    for t in range(1, n):
        prob = np.where(out[:, t - 1], p11, p01)
        out[:, t] = u[:, t] < prob
    return out


def test_kupiec_exact_size_and_simulated_rejection_rate_agree():
    """Gaussian P&L with the true 99 % VaR: over 400 samples of 250 days the rejection rate matches the
    exact size of the chi2(1) rule within 3 points. That exact size is 0.095, not 0.05: at n = 250 the
    rule rejects x = 0 (LR = 5.03) and x >= 7, and P(x = 0) alone is 0.081."""
    rng = np.random.default_rng(0)
    exact = bt.kupiec_size(250, 0.01)
    assert exact == pytest.approx(0.0948, abs=5e-4)
    assert set(bt.kupiec_rejection_set(250, 0.01)[:3].tolist()) == {0, 7, 8}
    pnl = rng.standard_normal((400, 250))
    rejects = [bt.kupiec_pof(bt.exceptions(row, Z99), 0.99).reject for row in pnl]
    assert np.mean(rejects) == pytest.approx(exact, abs=0.03)


def test_kupiec_size_is_nominal_at_1000_days():
    """With 1,000 days the discreteness fades: exact size 0.055 and a simulated rate of 5 % +- 3."""
    rng = np.random.default_rng(1)
    assert bt.kupiec_size(1000, 0.01) == pytest.approx(0.055, abs=0.005)
    pnl = rng.standard_normal((400, 1000))
    rate = np.mean([bt.kupiec_pof(bt.exceptions(row, Z99), 0.99).reject for row in pnl])
    assert 0.02 <= rate <= 0.08


def test_kupiec_power_against_a_97_percent_quantile_sold_as_99():
    """VaR set at the 97 % quantile (true exception rate 3 %): exact power 0.625 at 250 days, > 0.9 at 500;
    the simulated rate over 400 samples matches the exact number within 3 points."""
    rng = np.random.default_rng(2)
    p250, p500 = bt.kupiec_power(250, 0.01, 0.03), bt.kupiec_power(500, 0.01, 0.03)
    assert p250 == pytest.approx(0.625, abs=0.01) and p500 > 0.90
    var97 = norm.ppf(0.97)
    pnl = rng.standard_normal((400, 250))
    assert np.mean([bt.kupiec_pof(bt.exceptions(row, var97), 0.99).reject for row in pnl]) == pytest.approx(p250, abs=0.03)
    pnl = rng.standard_normal((400, 500))
    assert np.mean([bt.kupiec_pof(bt.exceptions(row, var97), 0.99).reject for row in pnl]) > 0.80


def test_kupiec_worked_values():
    r = bt.kupiec_pof(np.r_[np.ones(8, bool), np.zeros(242, bool)], 0.99)
    x, n, p = 8, 250, 0.01
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    assert r.statistic == pytest.approx(lr, rel=1e-12) and r.reject and r.n_exceptions == 8 and r.n == 250
    assert bt.kupiec_pof(np.zeros(250, bool), 0.99).statistic == pytest.approx(-2 * 250 * np.log(0.99), rel=1e-12)


def test_christoffersen_independence_rejects_clustered_and_accepts_iid():
    """Markov exceptions with P(exc | exc) = 0.3 at the same 1 % unconditional rate over 2,000 days are
    rejected > 90 % of the time; i.i.d. exceptions are rejected < 10 % (the test's size is below
    nominal here because samples with no consecutive pair give LR = 0)."""
    rng = np.random.default_rng(3)
    p11 = 0.3
    p01 = 0.01 * (1 - p11) / 0.99
    clustered = _markov_exceptions(rng, 2000, p11, p01, 150)
    iid = _iid_exceptions(rng, 2000, 0.01, 150)
    assert np.mean([bt.christoffersen_independence(e).reject for e in clustered]) > 0.90
    assert np.mean([bt.christoffersen_independence(e).reject for e in iid]) < 0.10
    cc = bt.christoffersen_cc(clustered[0], 0.99)
    assert cc.statistic == pytest.approx(cc.details["lr_pof"] + cc.details["lr_ind"], rel=1e-12)


def test_christoffersen_transition_counts():
    e = np.array([0, 1, 1, 0, 0, 1, 0], bool)
    r = bt.christoffersen_independence(e)
    assert (r.details["n00"], r.details["n01"], r.details["n10"], r.details["n11"]) == (1, 2, 2, 1)
    assert bt.christoffersen_independence(np.zeros(50, bool)).statistic == 0.0


def test_traffic_light_zone_boundaries_at_250_days_99_percent():
    """Basel Committee (1996): green 0-4, yellow 5-9, red 10 or more."""
    for x in range(0, 5):
        assert bt.traffic_light(x).verdict == "green"
    for x in range(5, 10):
        assert bt.traffic_light(x).verdict == "yellow"
    for x in (10, 11, 20):
        assert bt.traffic_light(x).verdict == "red" and bt.traffic_light(x).reject
    assert bt.traffic_light(4).details["cumulative_probability"] == pytest.approx(0.8922, abs=1e-3)
    assert bt.traffic_light(5).details["cumulative_probability"] == pytest.approx(0.9588, abs=1e-3)


def test_acerbi_szekely_accepts_true_es_and_rejects_es_30_percent_too_small():
    """N(0,1) P&L. (a) True VaR and ES: reject rate <= 15 % over 40 samples of 1,000 days (nominal 5 %).
    (b) A model 30 % too thin (VaR and ES both 0.7x the truth — an ES 30 % too small with the VaR
    kept right is impossible, ES >= VaR caps the shortfall at 12.7 % here): rejected in >= 90 % of
    40 samples of 250 days, Z2 ~ -4.3. (c) VaR right but the tail heavier than the model's — truth
    Student-t(3) scaled to the same 99 % VaR, so the Gaussian ES is 26 % too small: reject rate >= 70 %
    over 40 samples of 10,000 days (Z2 shifts by ~-0.3 against a null sd of ~10/sqrt(T), so 250 days
    cannot see it; the sample lengths are what the test's own power says they must be)."""
    from scipy.stats import t as student

    rng = np.random.default_rng(4)
    ok = [bt.acerbi_szekely_z2(rng.standard_normal(1000), Z99, ES99, 0.99, n_sim=400, seed=i).reject for i in range(40)]
    assert np.mean(ok) <= 0.15
    thin = [bt.acerbi_szekely_z2(rng.standard_normal(250), 0.7 * Z99, 0.7 * ES99, 0.99, n_sim=400, seed=i) for i in range(40)]
    assert np.mean([r.reject for r in thin]) >= 0.90 and np.mean([r.statistic for r in thin]) < -3.0
    q = student.ppf(0.99, 3)
    scale = Z99 / q
    es_true = scale * (3 + q**2) / 2 * student.pdf(q, 3) / 0.01
    assert ES99 / es_true == pytest.approx(0.743, abs=0.005)
    heavy = [bt.acerbi_szekely_z2(scale * rng.standard_t(3, 10_000), Z99, ES99, 0.99, n_sim=200, seed=i).reject for i in range(40)]
    assert np.mean(heavy) >= 0.70


def test_acerbi_szekely_statistic_is_zero_in_expectation_under_h0():
    rng = np.random.default_rng(5)
    z = [bt.acerbi_szekely_z2(rng.standard_normal(2000), Z99, ES99, 0.99, n_sim=50, seed=i).statistic for i in range(200)]
    assert np.mean(z) == pytest.approx(0.0, abs=0.06)


def test_engle_manganelli_dq_size_and_power():
    """i.i.d. exceptions: reject rate <= 0.12 over 150 samples of 1,000 days (nominal 5 %); Markov-clustered
    exceptions (p11 = 0.3) over 2,000 days: reject rate > 0.9."""
    rng = np.random.default_rng(6)
    pnl = rng.standard_normal((150, 1000))
    size = np.mean([bt.engle_manganelli_dq(row, Z99, 0.99).reject for row in pnl])
    assert size <= 0.12
    p11 = 0.3
    exc = _markov_exceptions(rng, 2000, p11, 0.01 * (1 - p11) / 0.99, 150)
    fake_pnl = np.where(exc, -Z99 - 1.0, 0.0)  # a loss beyond VaR exactly where the chain says
    power = np.mean([bt.engle_manganelli_dq(row, Z99, 0.99).reject for row in fake_pnl])
    assert power > 0.90


def test_suite_returns_six_named_results():
    rng = np.random.default_rng(8)
    res = bt.suite(rng.standard_normal(500), Z99, ES99, 0.99, n_sim=100)
    assert [r.name for r in res] == ["kupiec_pof", "christoffersen_ind", "christoffersen_cc", "basel_traffic_light",
                                     "acerbi_szekely_z2", "engle_manganelli_dq"]
    for r in res:
        assert r.verdict in ("accept", "reject", "green", "yellow", "red") and 0.0 <= r.p_value <= 1.0


def test_rolling_forecasts_shape_and_alignment():
    """Each row's forecast comes from rows strictly before it; the realised P&L is the book's full
    revaluation under that row's factor move (checked against a direct recomputation)."""
    book, mkt, hist = load_demo(DATA)
    small = hist.iloc[-330:].reset_index(drop=True)
    fc = bt.rolling_forecasts(book, mkt, small, 0.99, window=300, n_test=30, methods=("historical", "parametric"), n_scenarios=200)
    assert list(fc.columns) == ["date", "pnl", "var_historical", "es_historical", "var_parametric", "es_parametric"]
    assert len(fc) == 30 and (fc["es_historical"] >= fc["var_historical"]).all()
    # last row: market at the close before the final move is the final market rolled back one move
    from riskkit import pricing as bs
    from riskkit.positions import mark
    t = len(small) - 1
    m_before = Market(mkt.asof, {"SYN": mkt.spot["SYN"] * np.exp(-small["SYN:spot"].iloc[t])},
                      {"SYN": mkt.vol["SYN"] - small["SYN:vol"].iloc[t]})
    ms = mark(book, m_before)
    pnl = bs.revalue(ms, small["SYN:spot"].iloc[t], small["SYN:vol"].iloc[t], 1.0)[0]
    assert fc["pnl"].iloc[-1] == pytest.approx(pnl, rel=1e-9)
    assert isinstance(Book.from_positions([]), Book)
