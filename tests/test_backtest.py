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


def test_christoffersen_cc_simulated_p_value_size_and_power_at_250_days():
    """At 250 days / 99 % the chi2(2) rule is nearly silent: on 400 i.i.d. samples it rejects ~0.5 %
    (nominal 5 %). The default simulated p-value (exact i.i.d. Bernoulli null, 500 sims) has size
    <= 6 % and more power against Markov-clustered exceptions (p11 = 0.3) than the asymptotic rule at
    the same 250 days; at 2,000 days it rejects the clustered chain > 90 %. The statistic and the
    asymptotic p-value are unchanged and carried in `details`."""
    rng = np.random.default_rng(3)
    p11 = 0.3
    p01 = 0.01 * (1 - p11) / 0.99
    iid = _iid_exceptions(rng, 250, 0.01, 400)
    clustered = _markov_exceptions(rng, 250, p11, p01, 400)
    sim_size = np.mean([bt.christoffersen_cc(e, 0.99, n_sim=500, seed=i).reject for i, e in enumerate(iid)])
    asym_size = np.mean([bt.christoffersen_cc(e, 0.99, n_sim=0).reject for e in iid])
    assert asym_size < 0.02 and sim_size <= 0.06
    sim_power = np.mean([bt.christoffersen_cc(e, 0.99, n_sim=500, seed=i).reject for i, e in enumerate(clustered)])
    asym_power = np.mean([bt.christoffersen_cc(e, 0.99, n_sim=0).reject for e in clustered])
    assert sim_power > asym_power and sim_power > 0.30
    long = _markov_exceptions(rng, 2000, p11, p01, 100)
    assert np.mean([bt.christoffersen_cc(e, 0.99, n_sim=300, seed=i).reject for i, e in enumerate(long)]) > 0.90
    r, r0 = bt.christoffersen_cc(clustered[0], 0.99), bt.christoffersen_cc(clustered[0], 0.99, n_sim=0)
    assert r.statistic == r0.statistic and r.details["p_asymptotic"] == r0.p_value and r0.details["n_sim"] == 0


def test_christoffersen_worked_example_240_4_4_2():
    """Transition counts 240/4/4/2 (251 observations, 6 exceptions): LR_ind = 8.152; LR_pof over all
    T = 251 observations = 3.527, so LR_cc = 11.679 (a reference using T-1 = 250 for both gets 11.707)."""
    e = np.zeros(251, bool)
    e[[10, 11, 12, 100, 150, 200]] = True   # 1,1,1 gives n11 = 2 and n01 = 4 with three isolated exceptions
    r = bt.christoffersen_independence(e)
    assert (r.details["n00"], r.details["n01"], r.details["n10"], r.details["n11"]) == (240, 4, 4, 2)
    assert r.statistic == pytest.approx(8.152, abs=1e-3)
    cc = bt.christoffersen_cc(e, 0.99, n_sim=0)
    assert cc.details["lr_pof"] == pytest.approx(3.527, abs=1e-3) and cc.statistic == pytest.approx(11.679, abs=1e-3)
    assert float(bt.kupiec_lr(6, 250, 0.01)) + r.statistic == pytest.approx(11.707, abs=1e-3)


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


def test_traffic_light_exact_size_and_power():
    """P(not green | accurate 99 % model) = P(x >= 5) = 0.108 exactly (Binomial(250, 0.01)); against a
    true 3 % rate it is 0.872. 400 simulated samples of each land within 4 points."""
    from scipy.stats import binom

    rng = np.random.default_rng(7)
    assert binom.sf(4, 250, 0.01) == pytest.approx(0.1078, abs=5e-4) and binom.sf(4, 250, 0.03) == pytest.approx(0.8718, abs=5e-4)
    acc = np.mean([bt.traffic_light(int(e.sum())).verdict != "green" for e in _iid_exceptions(rng, 250, 0.01, 400)])
    bad = np.mean([bt.traffic_light(int(e.sum())).verdict != "green" for e in _iid_exceptions(rng, 250, 0.03, 400)])
    assert acc == pytest.approx(0.108, abs=0.04) and bad == pytest.approx(0.872, abs=0.04)


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


def test_acerbi_szekely_gaussian_sampler_over_rejects_a_correct_fat_tailed_model_and_own_sampler_does_not():
    """Truth and model both Student-t(4) with the true 99 % VaR and ES (ES/VaR = 1.39, Gaussian 1.15).
    The default Gaussian sampler's null has E[Z2] = 1 - ES_gauss/ES_t4 > 0, so it rejects the correct
    model ~10 % of the time at 250 days (and more at longer T); `scenario_sampler` built from the
    model's own scenario set rejects <= 6 %. 200 samples each, paired."""
    from scipy.stats import t as student

    nu = 4
    q = student.ppf(0.99, nu)
    es = (nu + q**2) / (nu - 1) * student.pdf(q, nu) / 0.01
    rng = np.random.default_rng(11)
    scen = rng.standard_t(nu, 20_000)                       # the model's own scenario P&L
    own = bt.scenario_sampler(scen, np.full(250, q), q)
    gauss, mine = [], []
    for i in range(200):
        x = rng.standard_t(nu, 250)
        gauss.append(bt.acerbi_szekely_z2(x, q, es, 0.99, n_sim=200, seed=i).reject)
        mine.append(bt.acerbi_szekely_z2(x, q, es, 0.99, sampler=own, n_sim=200, seed=i).reject)
    assert np.mean(mine) <= 0.06 and np.mean(gauss) >= 0.07 and np.mean(gauss) > np.mean(mine)
    draws = own(np.random.default_rng(0), 5)
    assert draws.shape == (5, 250) and set(np.unique(draws)) <= set(scen)


def test_acerbi_szekely_statistic_is_zero_in_expectation_under_h0():
    rng = np.random.default_rng(5)
    z = [bt.acerbi_szekely_z2(rng.standard_normal(2000), Z99, ES99, 0.99, n_sim=50, seed=i).statistic for i in range(200)]
    assert np.mean(z) == pytest.approx(0.0, abs=0.06)


def test_engle_manganelli_dq_size_and_power():
    """Gaussian P&L with the true 99 % VaR. The chi2 rule rejects a correct model 16 % of the time at
    500 days and 8-11 % elsewhere (measured on 4,000 samples), so the default p-value is simulated
    under the exact i.i.d. Bernoulli(p) null: on 300 samples each at 250 and 500 days (200 sims) the
    rejection rate is 5 % +- 3; the asymptotic rule at 500 days is >= 10 %. Markov-clustered exceptions
    (p11 = 0.3) over 2,000 days: reject rate > 0.9. With a constant VaR its column is the intercept and
    df = 5; with a time-varying VaR df = 6."""
    rng = np.random.default_rng(6)
    for T in (250, 500):
        pnl = rng.standard_normal((300, T))
        size = np.mean([bt.engle_manganelli_dq(row, Z99, 0.99, n_sim=200, seed=i).reject for i, row in enumerate(pnl)])
        assert 0.02 <= size <= 0.08, (T, size)
        if T == 500:
            assert np.mean([bt.engle_manganelli_dq(row, Z99, 0.99, n_sim=0).reject for row in pnl]) >= 0.10
    p11 = 0.3
    exc = _markov_exceptions(rng, 2000, p11, 0.01 * (1 - p11) / 0.99, 100)
    fake_pnl = np.where(exc, -Z99 - 1.0, 0.0)  # a loss beyond VaR exactly where the chain says
    power = np.mean([bt.engle_manganelli_dq(row, Z99, 0.99, n_sim=100, seed=i).reject for i, row in enumerate(fake_pnl)])
    assert power > 0.90
    r = bt.engle_manganelli_dq(pnl[0], Z99, 0.99, n_sim=0)
    assert r.details["df"] == 5 and r.p_value == r.details["p_asymptotic"]
    sig = np.exp(rng.normal(0, 0.3, 500))
    assert bt.engle_manganelli_dq(sig * rng.standard_normal(500), sig * Z99, 0.99, n_sim=0).details["df"] == 6


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
