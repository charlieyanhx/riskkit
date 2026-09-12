"""Identities every method keeps (exact to 1e-9 unless a simulation tolerance is stated)."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from riskkit import var as V
from riskkit.positions import Book, Market
from riskkit.synth import load_demo

SYN = "SYN"
DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/demo"


@pytest.fixture(scope="module")
def demo():
    return load_demo(DATA)


def _stock_book(qty=100.0):
    return Book.from_positions([{"pos_id": "STK", "legs": [{"symbol": SYN, "sec_type": "STK", "side": "BUY", "quantity": qty}]}])


def _history(n=800, seed=3, sigma=0.012, vol_sigma=0.004):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"{SYN}:spot": rng.normal(0, sigma, n), f"{SYN}:vol": rng.normal(0, vol_sigma, n)})


MKT = Market(asof=pd.Timestamp("2026-06-12T20:00", tz="UTC"), spot={SYN: 500.0}, vol={SYN: 0.2})


def test_parametric_delta_only_equals_closed_form_for_a_single_stock():
    """VaR = q * S * sigma_hat * z_c * sqrt(h), ES = q * S * sigma_hat * phi(z_p)/p * sqrt(h), sigma_hat the
    sample std (ddof=1) of the spot log returns; exact because k3 = k4 = 0 for a linear book."""
    hist = _history()
    sig = hist[f"{SYN}:spot"].std(ddof=1)
    for c, h in ((0.99, 1), (0.975, 10), (0.95, 5)):
        r = V.parametric_var(_stock_book(100.0), MKT, hist, c, h, order="delta")
        assert r.var == pytest.approx(100.0 * 500.0 * sig * norm.ppf(c) * np.sqrt(h), rel=1e-10)
        assert r.es == pytest.approx(100.0 * 500.0 * sig * norm.pdf(norm.ppf(1 - c)) / (1 - c) * np.sqrt(h), rel=1e-10)
        assert r.method == "parametric-delta" and r.confidence == c and r.horizon == h


def test_parametric_scales_with_sqrt_horizon():
    hist = _history()
    v1 = V.parametric_var(_stock_book(), MKT, hist, 0.99, 1, order="delta").var
    for h in (2, 5, 10, 20):
        assert V.parametric_var(_stock_book(), MKT, hist, 0.99, h, order="delta").var == pytest.approx(v1 * np.sqrt(h), rel=1e-10)


def test_historical_var_at_100_percent_confidence_is_the_worst_scenario(demo):
    book, mkt, hist = demo
    r = V.historical_var(book, mkt, hist, confidence=1.0)
    assert r.var == pytest.approx(-r.pnl.min(), abs=1e-9)
    assert r.es == pytest.approx(r.var, abs=1e-9)
    assert r.n_scenarios == len(hist) == len(r.pnl)


def test_var_es_from_pnl_worked_example_and_es_ge_var_property():
    """n=1000 losses 1..1000: c=0.99 -> VaR is the linear 99 % quantile (990.01), ES the mean of the
    ten worst (995.5); ES >= VaR on random samples for every c in [0.5, 1]."""
    pnl = -np.arange(1.0, 1001.0)
    v, e = V.var_es_from_pnl(pnl, 0.99)
    assert v == pytest.approx(990.01, abs=1e-9) and e == pytest.approx(995.5, abs=1e-9)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = rng.standard_t(3, rng.integers(5, 400))
        c = rng.uniform(0.5, 1.0)
        v, e = V.var_es_from_pnl(x, c)
        assert e >= v - 1e-12


def test_es_ge_var_for_every_method(demo):
    book, mkt, hist = demo
    for h in (1, 10):
        for r in V.all_methods(book, mkt, hist, 0.99, h, n_scenarios=4000):
            assert r.es >= r.var, r
            assert r.var > 0 and r.confidence == 0.99 and r.horizon == h


def test_var_is_positively_homogeneous_in_position_size(demo):
    """Doubling every quantity doubles VaR and ES for all four methods (same seed -> same scenarios)."""
    book, mkt, hist = demo
    base = V.all_methods(book, mkt, hist, 0.99, 1, n_scenarios=4000, seed=5)
    twice = V.all_methods(book.scaled(2.0), mkt, hist, 0.99, 1, n_scenarios=4000, seed=5)
    for a, b in zip(base, twice, strict=True):
        assert b.method == a.method
        assert b.var == pytest.approx(2.0 * a.var, rel=1e-9)
        assert b.es == pytest.approx(2.0 * a.es, rel=1e-9)


def test_monte_carlo_matches_closed_form_for_a_linear_book():
    """Gaussian MC on a stock is the parametric-delta closed form up to simulation error (n=200k)."""
    hist = _history()
    exact = V.parametric_var(_stock_book(), MKT, hist, 0.99, 1, order="delta")
    mc = V.monte_carlo_var(_stock_book(), MKT, hist, 0.99, 1, n_scenarios=200_000, seed=2)
    # a stock's P&L is S(e^x - 1), whose 1 % quantile sits 0.5 x^2 ~ 0.03 % below the linear one
    assert mc.var == pytest.approx(exact.var, rel=0.02)
    assert mc.es == pytest.approx(exact.es, rel=0.02)


def test_quadratic_form_cumulants_chi_square_identity():
    """b=0, L=1, S=s^2: Q = s^2/2 * chi2(1) with cumulants s^2/2, s^4/2, s^6, 3 s^8."""
    s2 = 0.7
    k = V.quadratic_form_cumulants(np.zeros(1), np.eye(1), np.array([[s2]]))
    assert k == pytest.approx((s2 / 2, s2**2 / 2, s2**3, 3 * s2**4), rel=1e-12)
    # b != 0, L = 0: pure Gaussian -> k1 = k3 = k4 = 0, k2 = b'Sb
    b, S = np.array([1.0, 2.0]), np.array([[1.0, 0.3], [0.3, 2.0]])
    assert V.quadratic_form_cumulants(b, np.zeros((2, 2)), S) == pytest.approx((0.0, float(b @ S @ b), 0.0, 0.0), abs=1e-14)


def test_cornish_fisher_tail_mean_equals_numerical_integral():
    for p, s, k in ((0.01, 0.0, 0.0), (0.01, -0.4, 1.2), (0.05, 0.3, 0.5)):
        num, _ = quad(lambda u, s=s, k=k: V.cornish_fisher_z(norm.ppf(u), s, k), 1e-12, p, limit=200)
        assert V.cornish_fisher_tail_mean(p, s, k) == pytest.approx(num / p, rel=1e-6)
    assert V.cornish_fisher_tail_mean(0.01, 0.0, 0.0) == pytest.approx(-norm.pdf(norm.ppf(0.01)) / 0.01, rel=1e-12)
    assert V.cornish_fisher_is_monotone(0.0, 0.0) and V.cornish_fisher_is_monotone(-0.3, 0.8)
    assert not V.cornish_fisher_is_monotone(2.5, 0.0)


def _straddles():
    legs = [{"symbol": SYN, "sec_type": "OPT", "expiration": "20260727", "strike": 500.0, "right": r, "side": "BUY", "quantity": 10}
            for r in ("C", "P")]
    long_ = Book.from_positions([{"pos_id": "L", "legs": legs}])
    short = Book.from_positions([{"pos_id": "S", "legs": [{**leg, "side": "SELL"} for leg in legs]}])
    return long_, short


def test_parametric_gamma_lowers_var_for_long_gamma_and_raises_it_for_short(demo):
    """A long straddle has positive gamma: the delta-gamma VaR is below delta-only; short is above.
    Both straddles' delta-gamma P&L is 1/2 sigma^2 chi2(1)-shaped (skew +-2.83, excess kurtosis 12),
    outside the Cornish-Fisher domain, so both results come from the exact quadratic form."""
    _, mkt, hist = demo
    long_, short = _straddles()
    for book, sign in ((long_, -1), (short, +1)):
        d = V.parametric_var(book, mkt, hist, 0.99, 1, order="delta").var
        r = V.parametric_var(book, mkt, hist, 0.99, 1, order="delta-gamma")
        assert sign * (r.var - d) > 0
        assert "quadratic_form_simulated" in r.notes and r.n_scenarios == V.EXACT_N_DRAWS


def test_long_gamma_parametric_var_is_a_positive_loss_bounded_by_theta_and_es_ge_var(demo):
    """A long straddle's delta-gamma P&L is theta*h + delta$ x + 1/2 gamma$ x^2, bounded below by
    theta*h - delta$^2 / (2 gamma$) (here -$196/day): the 99 % loss is a positive number at most that
    floor, and ES >= VaR. The raw Cornish-Fisher polynomial for this book is non-monotone and returns
    a NEGATIVE VaR (-260) and ES < VaR; the result must carry the note and the exact numbers instead."""
    _, mkt, hist = demo
    long_, _ = _straddles()
    theta_h, b, L, S = V.parametric_moments(long_, mkt, hist, 1, "delta-gamma")
    assert theta_h < 0 and L[0, 0] > 0
    raw_var, raw_es, _, skew, exk = V.cornish_fisher_var_es(theta_h, b, L, S, 0.99)
    assert not V.cornish_fisher_is_monotone(skew, exk) and raw_var < 0 and raw_es < raw_var
    r = V.parametric_var(long_, mkt, hist, 0.99, 1, order="delta-gamma")
    assert "cornish_fisher_nonmonotone" in r.notes
    floor = -theta_h + b[0] ** 2 / (2 * L[0, 0])
    assert 0 < r.var <= r.es <= floor + 1e-9
    assert r.var == pytest.approx(floor, rel=0.01)          # the 1 % quantile of the quadratic sits at the floor
    exact = V.quadratic_form_var_es(theta_h, b, L, S, 0.99, n=V.EXACT_N_DRAWS, seed=0)
    assert (r.var, r.es) == exact


def test_cornish_fisher_matches_the_exact_quadratic_form_inside_its_domain_and_defers_outside(demo):
    """Demo book, delta-gamma-vega. h = 1 (skew -0.4, excess kurtosis 0.7): the CF quantile and tail mean
    are within 3 % of the exact quadratic form (400k draws). h = 10 (skew -1.6, kurtosis 5.4): the raw CF
    is 10-14 % above the exact numbers, so it is out of domain and `parametric_var` returns the
    exact simulation with a note, identical to quantile="exact" at the same seed. h = 60 (skew -2.5,
    kurtosis 10) defers the same way. The global Maillard test is stricter than the guard: with
    kurtosis ~0 and skew -0.4 (most days of the rolling backtest) the cubic turns over at z ~ +4 and
    -9, outside the loss tail, and CF is within 1 % of the exact numbers there."""
    book, mkt, hist = demo
    th, b, L, S = V.parametric_moments(book, mkt, hist, 1)
    cf_var, cf_es, _, skew, exk = V.cornish_fisher_var_es(th, b, L, S, 0.99)
    assert V.cornish_fisher_in_domain(skew, exk) and abs(skew) < 0.5 and exk < 1.0
    ex_var, ex_es = V.quadratic_form_var_es(th, b, L, S, 0.99, n=400_000, seed=1)
    assert cf_var == pytest.approx(ex_var, rel=0.03) and cf_es == pytest.approx(ex_es, rel=0.03)
    r1 = V.parametric_var(book, mkt, hist, 0.99, 1)
    assert r1.notes == () and r1.n_scenarios == 0 and (r1.var, r1.es) == (cf_var, cf_es)
    th, b, L, S = V.parametric_moments(book, mkt, hist, 10)
    cf_var, cf_es, _, skew, exk = V.cornish_fisher_var_es(th, b, L, S, 0.99)
    ex_var, ex_es = V.quadratic_form_var_es(th, b, L, S, 0.99, n=400_000, seed=1)
    assert not V.cornish_fisher_in_domain(skew, exk) and V.cornish_fisher_is_monotone(skew, exk)
    assert 1.05 < cf_var / ex_var < 1.15 and 1.08 < cf_es / ex_es < 1.18
    for h in (10, 60):
        r = V.parametric_var(book, mkt, hist, 0.99, h, seed=3)
        e = V.parametric_var(book, mkt, hist, 0.99, h, quantile="exact", seed=3)
        assert r.notes == ("cornish_fisher_out_of_domain", "quadratic_form_simulated") and e.notes == ("quadratic_form_simulated",)
        assert (r.var, r.es) == (e.var, e.es) and r.es >= r.var > 0
    assert not V.cornish_fisher_is_monotone(-0.4, 0.0) and V.cornish_fisher_in_domain(-0.4, 0.0)
    assert V.cornish_fisher_is_monotone_on(-0.4, 0.0, -6.0, norm.ppf(0.01)) and not V.cornish_fisher_is_monotone_on(-0.4, 0.0, -6.0, 6.0)
    from riskkit.backtest import _market_at
    from riskkit.positions import mark
    for t in (len(hist) - 250, len(hist) - 125):    # two rolling-backtest days: the market then, the 1000 days before
        m = _market_at(book, mkt, hist, t)
        th, b, L, S = V.parametric_moments(book, m, hist.iloc[t - 1000:t], 1, marks=mark(book, m))
        v, e, _, skew, exk = V.cornish_fisher_var_es(th, b, L, S, 0.99)
        assert V.cornish_fisher_in_domain(skew, exk) and abs(skew) < 0.6 and exk < 0.5
        ev, ee = V.quadratic_form_var_es(th, b, L, S, 0.99, n=400_000, seed=1)
        assert v == pytest.approx(ev, rel=0.01) and e == pytest.approx(ee, rel=0.01)
    with pytest.raises(ValueError, match="quantile"):
        V.parametric_var(book, mkt, hist, 0.99, 1, quantile="magic")


def test_fhs_garch_fit_recovers_known_parameters():
    """arch's GARCH(1,1) MLE on 6,000 simulated days of omega=2e-6, alpha=0.08, beta=0.90 lands
    within 0.03 on alpha and beta (the usual sampling error at this length)."""
    rng = np.random.default_rng(9)
    n, omega, alpha, beta = 6000, 2e-6, 0.08, 0.90
    s2, x = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        x[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * x[t] ** 2 + beta * s2
    g = V.fit_garch11(x)
    assert g.alpha == pytest.approx(alpha, abs=0.03) and g.beta == pytest.approx(beta, abs=0.03)
    sig, s_next = g.filter(x)
    assert sig.shape == (n,) and s_next > 0
    paths = g.simulate(s_next, rng.standard_normal((1000, 5)))
    assert paths.shape == (1000, 5)


def test_empty_book_every_method_is_zero_and_flagged(demo):
    _, mkt, hist = demo
    empty = Book.from_positions([])
    for r in V.all_methods(empty, mkt, hist, 0.99, 1, n_scenarios=100):
        assert r.var == 0.0 and r.es == 0.0 and r.n_scenarios == 0 and "EMPTY" in r.notes
