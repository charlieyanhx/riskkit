"""The screening question — "the market is limit-down, the book is empty, VaR says risk fell" —
as three named contracts."""

import math

import numpy as np
import pandas as pd
import pytest

from riskkit import edge_cases as ec
from riskkit import pricing as bs
from riskkit.positions import Book, Market
from riskkit.synth import load_demo
from riskkit.var import historical_var

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/demo"


@pytest.fixture(scope="module")
def demo():
    return load_demo(DATA)


def test_empty_book_reports_empty_not_ok(demo):
    _, mkt, hist = demo
    a = ec.assess(Book.from_positions([]), mkt, hist)
    assert a.state == "EMPTY" and a.state != "OK"
    assert a.var.var == 0.0 and a.var.es == 0.0 and a.var.n_scenarios == 0
    assert [f.kind for f in a.flags] == ["EMPTY"] and "not a result" in a.flags[0].reason
    assert a.marks == ()


def test_limit_down_does_not_reduce_var(demo):
    """The underlying locks limit-down (-7 %). The feed keeps printing the locked price, so the
    rolling window gains a run of zero-return days (and has rolled off the last stress episode):
    the raw recomputation of VaR *falls*. Under the lock the book is marked at the last valid mid,
    the scenario set gains the lock move continuing (x1, x2, x3 with the vol factor at its
    historical beta), and VaR / ES may not fall below the last unlocked values — if the widened
    set alone is still lower they are frozen there, with a VAR_FROZEN flag saying why."""
    book, mkt, hist = demo
    unlocked = ec.assess(book, mkt, hist)
    assert unlocked.state == "OK" and not unlocked.flags
    # the locked feed: 40 days of zero returns and zero vol changes, in a 340-day rolling window
    window = pd.concat([hist.iloc[-300:], pd.DataFrame({"date": ["lock"] * 40, "SYN:spot": 0.0, "SYN:vol": 0.0})], ignore_index=True)
    raw = historical_var(book, mkt, window, 0.99, 1)
    assert raw.var < unlocked.var.var                                     # "VaR says risk fell"
    lock = ec.Lock("SYN", last_valid_spot=mkt.spot["SYN"], lock_move=math.log(0.93), since=mkt.asof - pd.Timedelta(minutes=30))
    locked = ec.assess(book, mkt, window, 0.99, 1, lock=lock, previous=unlocked.var)
    assert locked.state == "LIMIT_DOWN"
    assert locked.var.var >= unlocked.var.var and locked.var.es >= unlocked.var.es
    kinds = [f.kind for f in locked.flags]
    assert "LIMIT_DOWN" in kinds and "lock move continuing" in next(f.reason for f in locked.flags if f.kind == "LIMIT_DOWN")
    # the widened scenario set contains the lock move continuing, and its P&L sits in the loss tail
    extra = ec.lock_scenarios(book, window, lock)
    assert extra.shape == (3, 2) and extra[:, 0].tolist() == pytest.approx([lock.lock_move, 2 * lock.lock_move, 3 * lock.lock_move])
    assert extra[0, 1] == pytest.approx(ec.spot_vol_beta(window, "SYN") * lock.lock_move)
    assert locked.var.n_scenarios == len(window) + 3
    assert locked.var.pnl[-3:].max() < -raw.var                           # every continuation loses more than the raw VaR
    assert all(m.spot == mkt.spot["SYN"] for m in locked.marks)           # marked at the last valid mid
    # the same lock without the widening (previous only) is held by the freeze alone
    frozen = ec.assess(book, mkt, window, 0.99, 1, lock=ec.Lock("SYN", mkt.spot["SYN"], -1e-9, mkt.asof), previous=unlocked.var)
    assert frozen.var.var == unlocked.var.var and "VAR_FROZEN" in [f.kind for f in frozen.flags]
    assert "frozen" in next(f.reason for f in frozen.flags if f.kind == "VAR_FROZEN") and "VAR_FROZEN" in frozen.var.notes


def test_limit_down_freeze_triggers_when_recomputation_is_lower(demo):
    """Direct check of the freeze rule with a `previous` above anything the data produces."""
    book, mkt, hist = demo
    lock = ec.Lock("SYN", mkt.spot["SYN"], math.log(0.93), mkt.asof)
    big = historical_var(book, mkt, hist, 0.99, 1)
    previous = big.__class__(big.method, big.confidence, big.horizon, big.var * 5, big.es * 5, big.n_scenarios)
    a = ec.assess(book, mkt, hist, lock=lock, previous=previous)
    assert a.var.var == previous.var and a.var.es == previous.es and "VAR_FROZEN" in [f.kind for f in a.flags]
    assert "VAR_FROZEN" in a.var.notes and a.state == "LIMIT_DOWN"


def test_stale_quote_marks_to_last_valid_mid_and_flags(demo):
    """Quotes for one leg: a valid one 40 min old, then a crossed (invalid) print 1 min old. The leg is
    marked at the 40-minute-old mid (the last VALID one), flagged STALE_QUOTE with its age; a fresh
    valid quote on another leg is used silently; a mid with no implied vol is flagged NO_IV."""
    book, mkt, hist = demo
    key_stale, key_fresh, key_noiv = "SYN:20260727:480:P", "SYN:20260727:440:P", "SYN:20260811:490:P"
    asof = pd.Timestamp(mkt.asof)
    quotes = {
        key_stale: [ec.Quote(asof - pd.Timedelta(minutes=40), 2.40, 2.60), ec.Quote(asof - pd.Timedelta(minutes=1), 2.70, 2.50)],
        key_fresh: [ec.Quote(asof - pd.Timedelta(minutes=2), 0.05, 0.07)],
        key_noiv: [ec.Quote(asof - pd.Timedelta(minutes=3), 494.0, 496.0)],   # a put above its strike: no IV solves
    }
    a = ec.assess(book, mkt, hist, quotes=quotes, stale_after=pd.Timedelta(minutes=15))
    by_key = {m.key: m for m in a.marks}
    assert by_key[key_stale].value == pytest.approx(2.50) and by_key[key_fresh].value == pytest.approx(0.06)
    assert by_key[key_stale].iv == pytest.approx(bs.implied_vol(2.50, 500.0, 480.0, by_key[key_stale].T, "P"), abs=1e-10)
    kinds = {f.kind: f for f in a.flags}
    assert a.state == "STALE" and "STALE_QUOTE" in kinds and kinds["STALE_QUOTE"].subject == key_stale
    assert "0 days 00:40:00" in kinds["STALE_QUOTE"].reason and "2.5000" in kinds["STALE_QUOTE"].reason
    assert "NO_IV" in kinds and kinds["NO_IV"].subject == key_noiv and by_key[key_noiv].iv == mkt.vol["SYN"]
    assert by_key[key_noiv].flags == ("NO_IV",)
    assert not ec.Quote(asof, 2.70, 2.50).valid and not ec.Quote(asof, float("nan"), 1.0).valid and ec.Quote(asof, 0.0, 0.02).valid


def test_no_iv_quote_changes_book_value_but_not_risk(demo):
    """A quote mid with no implied vol (a put quoted above its strike) marks the leg at the mid but the
    scenario P&L is measured from the model price at the fallback vol: zero shock is exactly zero for
    every leg and VaR / ES are unchanged to 1e-9 (a 0.1 % quote-vs-model gap on one leg used to enter
    every scenario as a constant $-97,866 and move VaR from 6,277 to 104,143)."""
    book, mkt, hist = demo
    asof = pd.Timestamp(mkt.asof)
    key = "SYN:20260811:490:P"
    base = ec.assess(book, mkt, hist)
    a = ec.assess(book, mkt, hist, quotes={key: [ec.Quote(asof - pd.Timedelta(minutes=3), 494.0, 496.0)]})
    m = next(m for m in a.marks if m.key == key)
    assert m.value == 495.0 and m.flags == ("NO_IV",) and a.state == "OK"
    from riskkit.positions import book_value
    assert book_value(list(a.marks)) != book_value(list(base.marks))
    zero = bs.revalue_factors(a.marks, {"SYN": np.zeros(1)}, {"SYN": np.zeros(1)}, 0.0)
    assert zero[0] == 0.0
    assert a.var.var == pytest.approx(base.var.var, abs=1e-9) and a.var.es == pytest.approx(base.var.es, abs=1e-9)
    assert np.array_equal(a.var.pnl, base.var.pnl)
    # the same holds for a stale quote (marked at its mid, implied vol inverted) and a fresh one
    quotes = {"SYN:20260727:480:P": [ec.Quote(asof - pd.Timedelta(minutes=40), 2.40, 2.60)],
              "SYN:20260727:440:P": [ec.Quote(asof - pd.Timedelta(minutes=2), 0.05, 0.07)]}
    b = ec.assess(book, mkt, hist, quotes=quotes)
    assert b.state == "STALE"
    for leg in b.marks:
        assert bs.revalue_factors([leg], {"SYN": np.zeros(2)}, {"SYN": np.zeros(2)}, 0.0).tolist() == [0.0, 0.0]


def test_limit_up_lock_reports_limit_up(demo):
    """A positive lock move is the same contract with the opposite sign: state LIMIT_UP, flag LIMIT_UP,
    continuations up, marks at the last valid mid, and the assessment carries the marked market."""
    book, mkt, hist = demo
    unlocked = ec.assess(book, mkt, hist)
    up = ec.assess(book, mkt, hist, lock=ec.Lock("SYN", 500.0, math.log(1.07), mkt.asof), previous=unlocked.var)
    assert up.state == "LIMIT_UP" and "LIMIT_UP" in ec.STATES and [f.kind for f in up.flags][0] == "LIMIT_UP"
    assert up.var.var >= unlocked.var.var and up.var.es >= unlocked.var.es
    assert ec.lock_scenarios(book, hist, ec.Lock("SYN", 500.0, math.log(1.07), mkt.asof))[:, 0].min() > 0
    assert up.market.spot["SYN"] == 500.0 and unlocked.market == mkt
    locked_print = Market(mkt.asof, {"SYN": 535.0}, mkt.vol)
    up2 = ec.assess(book, locked_print, hist, lock=ec.Lock("SYN", 500.0, math.log(1.07), mkt.asof))
    assert up2.market.spot["SYN"] == 500.0 and all(m.spot == 500.0 for m in up2.marks)


def test_state_precedence_empty_over_lock_over_stale(demo):
    _, mkt, hist = demo
    lock = ec.Lock("SYN", 500.0, math.log(0.93), mkt.asof)
    assert ec.assess(Book.from_positions([]), mkt, hist, lock=lock).state == "EMPTY"
    book, _, _ = demo
    quotes = {"SYN:20260727:480:P": [ec.Quote(mkt.asof - pd.Timedelta(hours=2), 2.4, 2.6)]}
    assert ec.assess(book, mkt, hist, quotes=quotes, lock=lock).state == "LIMIT_DOWN"
    assert ec.assess(book, mkt, hist, quotes=quotes).state == "STALE"
    assert ec.assess(book, mkt, hist).state == "OK"
    assert isinstance(np.zeros(1), np.ndarray) and isinstance(Market, type)
