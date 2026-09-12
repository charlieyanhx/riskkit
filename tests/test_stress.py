import math

import numpy as np
import pytest

from riskkit import stress as st
from riskkit.positions import Book, Market, mark
from riskkit.synth import load_demo

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/demo"


@pytest.fixture(scope="module")
def marked():
    book, mkt, _ = load_demo(DATA)
    return mark(book, mkt), mkt


def test_ladder_shape_zero_cell_and_scalar_agreement(marked):
    ms, mkt = marked
    lad = st.ladder(ms, mkt)
    assert lad.shape == (8, 5) and lad.index.name == "spot_move" and lad.columns.name == "vol_move"
    assert lad.loc[0.0, 0.0] == 0.0
    assert lad.loc[-0.10, 0.05] == pytest.approx(st.scenario_pnl(ms, mkt, -0.10, 0.05), abs=1e-9)


def test_named_scenarios_are_data_with_sources(marked):
    ms, mkt = marked
    names = [s.name for s in st.HISTORICAL_SCENARIOS]
    assert {"Oct-2008", "Mar-2020", "Aug-2024 vol spike"} <= set(names)
    for s in st.HISTORICAL_SCENARIOS:
        assert s.spot_move < 0 and s.vol_move > 0 and s.source
    oct08 = next(s for s in st.HISTORICAL_SCENARIOS if s.name == "Oct-2008")
    assert math.exp(oct08.spot_move) - 1 == pytest.approx(968.75 / 1166.36 - 1, abs=1e-12)
    assert oct08.vol_move == pytest.approx(0.205, abs=1e-12)
    df = st.historical_scenarios(ms, mkt)
    assert list(df.columns) == ["scenario", "spot_move", "vol_move", "days", "pnl", "source"] and len(df) == len(names)


def test_reverse_stress_spot_reproduces_the_limit_loss_and_a_bigger_shock_loses_more(marked):
    ms, mkt = marked
    limit = 25_000.0
    rs = st.reverse_stress_spot(ms, mkt, limit)
    assert rs.found and rs.axis == "spot" and rs.direction == "down" and rs.shock < 0
    assert rs.pnl == pytest.approx(-limit, abs=1e-6)
    assert st.scenario_pnl(ms, mkt, rs.shock, rs.response) == pytest.approx(-limit, abs=1e-6)
    assert rs.response == pytest.approx(-rs.shock, abs=1e-12)        # default vol response: -x
    bigger = st.scenario_pnl(ms, mkt, rs.shock * 1.5, -rs.shock * 1.5)
    assert bigger < -limit
    smaller = st.scenario_pnl(ms, mkt, rs.shock * 0.5, -rs.shock * 0.5)
    assert smaller > -limit


def test_reverse_stress_vol_reproduces_the_limit_loss(marked):
    ms, mkt = marked
    limit = 10_000.0
    rs = st.reverse_stress_vol(ms, mkt, limit)
    assert rs.found and rs.axis == "vol" and rs.shock > 0 and rs.response == 0.0
    assert rs.pnl == pytest.approx(-limit, abs=1e-6)
    assert st.scenario_pnl(ms, mkt, 0.0, rs.shock * 1.5) < -limit


def test_reverse_stress_reports_no_breach_when_the_book_cannot_lose_that_much():
    """A long put alone never loses more than its premium: reverse stress on spot up says so."""
    mkt = Market(asof=__import__("pandas").Timestamp("2026-06-12T20:00", tz="UTC"), spot={"SYN": 500.0}, vol={"SYN": 0.2})
    book = Book.from_positions([{"pos_id": "P", "legs": [
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260801", "strike": 490.0, "right": "P", "side": "BUY", "quantity": 1}]}])
    ms = mark(book, mkt)
    premium = ms[0].value * 100
    rs = st.reverse_stress_spot(ms, mkt, premium * 2, direction="up")
    assert not rs.found and math.isnan(rs.shock) and math.isnan(rs.pnl)
    rs2 = st.reverse_stress_spot(ms, mkt, premium * 0.5, vol_response=lambda x: 0.0, direction="up")
    assert rs2.found and rs2.shock > 0 and rs2.pnl == pytest.approx(-premium * 0.5, abs=1e-6)


def test_reverse_stress_finds_the_first_breach_not_an_arbitrary_root():
    """A short strangle loses on both sides; bracketing outward from zero finds the nearest breach."""
    mkt = Market(asof=__import__("pandas").Timestamp("2026-06-12T20:00", tz="UTC"), spot={"SYN": 500.0}, vol={"SYN": 0.2})
    legs = [{"symbol": "SYN", "sec_type": "OPT", "expiration": "20260801", "strike": k, "right": r, "side": "SELL", "quantity": 5}
            for k, r in ((470.0, "P"), (530.0, "C"))]
    ms = mark(Book.from_positions([{"pos_id": "S", "legs": legs}]), mkt)
    down, up = st.reverse_stress_spot(ms, mkt, 5000.0, lambda x: 0.0, direction="down"), st.reverse_stress_spot(ms, mkt, 5000.0, lambda x: 0.0, direction="up")
    assert down.found and up.found and down.shock < 0 < up.shock
    for s in np.linspace(0, abs(down.shock) * 0.99, 20):
        assert st.scenario_pnl(ms, mkt, -s, 0.0) > -5000.0
