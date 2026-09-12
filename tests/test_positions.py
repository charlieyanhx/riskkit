import json

import pandas as pd
import pytest

from riskkit.positions import (
    Book,
    Market,
    book_value,
    contract_key,
    dollar_greeks,
    mark,
    multiplier,
    risk_factors,
    signed_qty,
)

# deskboard's committed demo blotter, verbatim (data/demo/blotter.json shape from feeds/blotter.py)
DESKBOARD_BLOTTER = [
    {"pos_id": "A-0601", "sleeve": "A", "legs": [
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260730", "strike": 480.0, "right": "P", "side": "SELL", "quantity": 4.0, "avg_fill_price": None},
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260730", "strike": 440.0, "right": "P", "side": "BUY", "quantity": 4.0, "avg_fill_price": None}]},
    {"pos_id": "C-0603", "sleeve": "C", "legs": [
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260715", "strike": 470.0, "right": "P", "side": "SELL", "quantity": 2.0, "avg_fill_price": None},
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260715", "strike": 530.0, "right": "C", "side": "SELL", "quantity": 2.0, "avg_fill_price": None}],
     "hedge_shares": -40},
    {"pos_id": "S-0610", "sleeve": "S", "legs": [
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260625", "strike": 490.0, "right": "P", "side": "BUY", "quantity": 3.0, "avg_fill_price": None},
        {"symbol": "SYN", "sec_type": "OPT", "expiration": "20260625", "strike": 510.0, "right": "C", "side": "SELL", "quantity": 3.0, "avg_fill_price": None}]},
]
MKT = Market(asof=pd.Timestamp("2026-06-15T20:00", tz="UTC"), spot={"SYN": 500.0}, vol={"SYN": 0.18})


def test_deskboard_blotter_shape_loads_verbatim(tmp_path):
    """Both file shapes deskboard accepts: a bare list and {"positions": [...]} (a live bot state file)."""
    (tmp_path / "list.json").write_text(json.dumps(DESKBOARD_BLOTTER))
    (tmp_path / "state.json").write_text(json.dumps({"positions": DESKBOARD_BLOTTER, "other": 1}))
    b1, b2 = Book.load(tmp_path / "list.json"), Book.load(tmp_path / "state.json")
    assert b1 == b2
    assert [p["pos_id"] for p in b1.positions] == ["A-0601", "C-0603", "S-0610"]
    assert len(b1.legs()) == 7  # six option legs + the hedge-share STK leg
    hedge = [leg for pid, leg in b1.legs() if pid == "C-0603" and leg["sec_type"] == "STK"][0]
    assert hedge == {"symbol": "SYN", "sec_type": "STK", "side": "SELL", "quantity": 40.0}


def test_contract_key_signed_qty_multiplier_match_deskboard_conventions():
    leg = DESKBOARD_BLOTTER[0]["legs"][0]
    assert contract_key(leg) == "SYN:20260730:480:P"
    assert signed_qty(leg) == -4.0 and multiplier(leg) == 100.0
    assert contract_key({"symbol": "SYN", "sec_type": "STK"}) == "SYN" and multiplier({"symbol": "SYN", "sec_type": "STK"}) == 1.0
    fut = {"symbol": "ES", "sec_type": "FUT", "expiration": "20260918", "side": "BUY", "quantity": 2, "multiplier": 50}
    assert multiplier(fut) == 50.0 and contract_key(fut) == "ES:20260918:FUT"
    with pytest.raises(ValueError, match="multiplier"):
        multiplier({"symbol": "ES", "sec_type": "FUT", "side": "BUY", "quantity": 1})


def test_risk_factor_mapping():
    assert risk_factors(DESKBOARD_BLOTTER[0]["legs"][0]) == {"spot": "SYN", "vol": "SYN:vol"}
    assert risk_factors({"symbol": "SYN", "sec_type": "STK"}) == {"spot": "SYN"}
    assert risk_factors({"symbol": "ES", "sec_type": "FUT", "multiplier": 50}) == {"spot": "ES"}
    assert Book.from_positions(DESKBOARD_BLOTTER).underlyings() == ["SYN"]


def test_validation_errors():
    with pytest.raises(ValueError, match="missing"):
        Book.from_positions([{"pos_id": "X", "legs": [{"symbol": "SYN", "sec_type": "OPT", "strike": 1}]}])
    with pytest.raises(ValueError, match="right"):
        Book.from_positions([{"pos_id": "X", "legs": [{**DESKBOARD_BLOTTER[0]["legs"][0], "right": "X"}]}])
    with pytest.raises(ValueError, match="sec_type"):
        Book.from_positions([{"pos_id": "X", "legs": [{"symbol": "SYN", "sec_type": "BOND", "quantity": 1}]}])


def test_mark_book_value_identity_and_scaling():
    """book_value == sum(value * signed_qty * mult); scaling the book by k scales value and Greeks by k."""
    book = Book.from_positions(DESKBOARD_BLOTTER)
    ms = mark(book, MKT)
    assert book_value(ms) == pytest.approx(sum(m.value * m.sq * m.mult for m in ms), abs=1e-9)
    g1 = dollar_greeks(ms, MKT)["SYN"]
    ms3 = mark(book.scaled(3.0), MKT)
    assert book_value(ms3) == pytest.approx(3.0 * book_value(ms), rel=1e-12)
    g3 = dollar_greeks(ms3, MKT)["SYN"]
    for k in g1:
        assert g3[k] == pytest.approx(3.0 * g1[k], rel=1e-12)


def test_mark_uses_quote_mid_and_inverts_iv_when_given():
    book = Book.from_positions(DESKBOARD_BLOTTER[:1])
    key = "SYN:20260730:480:P"
    ms = mark(book, MKT, quotes={key: (7.25, 0.21)})
    m = [x for x in ms if x.key == key][0]
    assert m.value == 7.25 and m.iv == 0.21
    m2 = [x for x in mark(book, MKT, quotes={key: (7.25, float("nan"))}) if x.key == key][0]
    assert m2.value == 7.25 and m2.iv == 0.18 and m2.flags == ("NO_IV",)
