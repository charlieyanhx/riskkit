import math

import numpy as np
import pytest

from riskkit import pricing as bs
from riskkit.positions import Book, LegMark, Market, mark

SYN = "SYN"


def test_canonical_atm_case_call_and_put():
    """S=K=100, r=5%, sigma=20%, T=1: call 10.4505835722, put 5.5735260223 — closed form recomputed
    with scipy and QuantLib 1.43 AnalyticEuropeanEngine (_plan4_inputs/oracle_values.json), tol 1e-8."""
    assert bs.price(100, 100, 1.0, 0.2, "C", r=0.05) == pytest.approx(10.4505835722, abs=1e-8)
    assert bs.price(100, 100, 1.0, 0.2, "P", r=0.05) == pytest.approx(5.5735260223, abs=1e-8)


def test_hull_textbook_example():
    """Hull, Options, Futures and Other Derivatives, BSM worked example: S=42, K=40, r=10%, sigma=20%,
    T=0.5 -> call 4.76, put 0.81 (exact 4.7594223929 / 0.8085993729), tol 1e-8."""
    assert bs.price(42, 40, 0.5, 0.2, "C", r=0.10) == pytest.approx(4.7594223929, abs=1e-8)
    assert bs.price(42, 40, 0.5, 0.2, "P", r=0.10) == pytest.approx(0.8085993729, abs=1e-8)


def test_put_call_parity_and_implied_vol_roundtrip():
    S, K, T, r, q = 480.0, 500.0, 0.25, 0.03, 0.01
    c, p = bs.price(S, K, T, 0.22, "C", r, q), bs.price(S, K, T, 0.22, "P", r, q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)
    assert bs.implied_vol(c, S, K, T, "C", r, q) == pytest.approx(0.22, abs=1e-8)
    assert math.isnan(bs.implied_vol(-1.0, S, K, T, "C"))


def test_price_vec_equals_scalar_price_elementwise():
    rng = np.random.default_rng(0)
    S = 500.0 * np.exp(rng.normal(0, 0.05, 50))
    sig = 0.2 + rng.normal(0, 0.03, 50)
    for right in ("C", "P"):
        v = bs.price_vec(S, 490.0, 0.1, sig, right, 0.02, 0.01)
        for i in range(50):
            assert v[i] == pytest.approx(bs.price(S[i], 490.0, 0.1, sig[i], right, 0.02, 0.01), abs=1e-12)
    assert bs.price_vec(np.array([510.0, 490.0]), 500.0, 0.0, np.array([0.2, 0.2]), "C").tolist() == [10.0, 0.0]


def _marks():
    mkt = Market(asof=__import__("pandas").Timestamp("2026-06-12T20:00", tz="UTC"), spot={SYN: 500.0}, vol={SYN: 0.2})
    book = Book.from_positions([
        {"pos_id": "P1", "sleeve": "A", "legs": [
            {"symbol": SYN, "sec_type": "OPT", "expiration": "20260801", "strike": 480.0, "right": "P", "side": "SELL", "quantity": 3},
            {"symbol": SYN, "sec_type": "OPT", "expiration": "20260801", "strike": 520.0, "right": "C", "side": "BUY", "quantity": 2}],
         "hedge_shares": 50},
    ])
    return book, mkt, mark(book, mkt)


def test_revalue_zero_shock_is_zero_and_matches_scalar_repricing():
    """P&L at (0, 0, 0) is exactly 0; at a shock it equals leg-by-leg scalar repricing times signed qty x mult."""
    _, mkt, ms = _marks()
    assert bs.revalue(ms, 0.0, 0.0, 0.0)[0] == 0.0
    x, y, d = -0.03, 0.02, 5.0
    expected = 0.0
    for m in ms:
        if m.sec_type == "OPT":
            v1 = bs.price(m.spot * math.exp(x), m.strike, m.T - d / 365.0, m.iv + y, m.right)
        else:
            v1 = m.spot * math.exp(x)
        expected += (v1 - m.value) * m.sq * m.mult
    assert bs.revalue(ms, x, y, d)[0] == pytest.approx(expected, abs=1e-9)


def test_revalue_broadcasts_over_scenarios():
    _, mkt, ms = _marks()
    xs = np.linspace(-0.1, 0.1, 21)
    pnl = bs.revalue(ms, xs, 0.0)
    assert pnl.shape == (21,)
    assert pnl[10] == pytest.approx(0.0, abs=1e-12)
    for i, x in enumerate(xs):
        assert pnl[i] == pytest.approx(bs.revalue(ms, x, 0.0)[0], abs=1e-9)


def test_revalue_empty_book_is_zero_vector():
    assert bs.revalue([], np.zeros(3), np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    assert isinstance(LegMark("p", "k", SYN, "STK", 1.0, 1.0, 1.0, 1.0), LegMark)
