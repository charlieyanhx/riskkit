"""Book schema, risk-factor mapping and marking.

The leg / position shape is deskboard's (`engine/book.py`, `feeds/blotter.py`) — OPT / STK
legs and `hedge_shares` load unchanged — plus a FUT leg with an explicit per-contract
multiplier, which deskboard does not have:

    leg      = {symbol, sec_type "OPT"|"STK"|"FUT", expiration "YYYYMMDD", strike, right "C"|"P",
                side "BUY"|"SELL", quantity, multiplier?}
    position = {pos_id, sleeve, legs: [leg, ...], hedge_shares?}

`multiplier` defaults to 100 for OPT and 1 for STK; a FUT leg must carry its own
(per-contract) multiplier. `hedge_shares` becomes a STK leg on the first leg's symbol, as
in deskboard. A blotter file is either a list of positions or `{"positions": [...]}`.

Risk factors. Every leg maps to a spot factor named by its symbol; an option leg also maps
to a vol factor `<symbol>:vol` (one ATM vol level per underlying in v0.1 — a vol shock is a
parallel shift of every implied vol on that underlying). Factor histories are DataFrames
with columns `<symbol>:spot` (daily log returns) and `<symbol>:vol` (daily changes of the
vol level, in vol units: 0.01 = one vol point).

Marks. `mark(book, market)` values every leg at the valuation time: options by
Black-Scholes at the underlying's vol level unless a quote is supplied for the contract, in
which case the mid is the value and its implied vol is inverted (deskboard's convention);
STK and FUT legs at the spot factor level. Units: per-share value; dollar exposure is
value x signed quantity x multiplier. Invariant kept here: `book_value(marks)` equals the
sum of leg values times signed quantity times multiplier, and `revalue` at zero shock is
exactly zero P&L for every leg — scenario P&L is measured from the model price at the leg's
marked (spot, iv, T), so a quote mid with no implied vol (NO_IV) changes the book value
but not the risk numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import pricing as bs

YEAR = 365.0
EXPIRY_HOUR_UTC = 20  # 16:00 New York ~ 20:00 UTC, as in deskboard
REQUIRED_LEG = {"symbol", "expiration", "strike", "right", "side", "quantity"}
DEFAULT_MULT = {"OPT": 100.0, "STK": 1.0}


def contract_key(leg: dict) -> str:
    st = leg.get("sec_type", "OPT")
    if st == "STK":
        return leg["symbol"]
    if st == "FUT":
        return f"{leg['symbol']}:{leg.get('expiration', '')}:FUT"
    return f"{leg['symbol']}:{leg['expiration']}:{float(leg['strike']):g}:{leg['right']}"


def signed_qty(leg: dict) -> float:
    q = float(leg["quantity"])
    return q if str(leg.get("side", "BUY")).upper() in ("BUY", "LONG") else -q


def multiplier(leg: dict) -> float:
    st = leg.get("sec_type", "OPT")
    if "multiplier" in leg and leg["multiplier"] is not None:
        return float(leg["multiplier"])
    if st not in DEFAULT_MULT:
        raise ValueError(f"{contract_key(leg)}: a {st} leg needs an explicit per-contract multiplier")
    return DEFAULT_MULT[st]


def risk_factors(leg: dict) -> dict[str, str]:
    """Factor names a leg is exposed to: always a spot factor; options also a vol factor."""
    st = leg.get("sec_type", "OPT")
    f = {"spot": leg["symbol"]}
    if st == "OPT":
        f["vol"] = f"{leg['symbol']}:vol"
    return f


def years_to_expiry(expiration: str, ts: float) -> float:
    exp = datetime.strptime(expiration, "%Y%m%d").replace(hour=EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    return max(0.0, (exp.timestamp() - ts) / 86400.0 / YEAR)


# ---- book ---------------------------------------------------------------------------------

def normalize_position(p: dict) -> dict:
    """Validate one position and return it with numeric strike/quantity (deskboard's rule)."""
    legs = []
    for leg in p["legs"]:
        st = leg.get("sec_type", "OPT")
        if st == "OPT" and not REQUIRED_LEG <= set(leg):
            raise ValueError(f"{p.get('pos_id')}: leg missing {sorted(REQUIRED_LEG - set(leg))}")
        if st not in ("OPT", "STK", "FUT"):
            raise ValueError(f"{p.get('pos_id')}: unknown sec_type {st!r}")
        out = {**leg, "sec_type": st, "quantity": float(leg["quantity"])}
        if st == "OPT":
            out["strike"] = float(leg["strike"])
            if str(leg["right"]) not in ("C", "P"):
                raise ValueError(f"{p.get('pos_id')}: right must be 'C' or 'P'")
        multiplier(out)  # raises for a FUT without one
        legs.append(out)
    return {"pos_id": str(p["pos_id"]), "sleeve": p.get("sleeve", ""), "legs": legs,
            "hedge_shares": float(p.get("hedge_shares", 0) or 0)}


@dataclass(frozen=True)
class Book:
    positions: tuple[dict, ...]

    @classmethod
    def from_positions(cls, positions: list[dict]) -> Book:
        return cls(tuple(normalize_position(p) for p in positions))

    @classmethod
    def load(cls, path: str | Path) -> Book:
        raw = json.loads(Path(path).read_text())
        return cls.from_positions(raw["positions"] if isinstance(raw, dict) else raw)

    def legs(self) -> list[tuple[str, dict]]:
        """(pos_id, leg) for every leg including the hedge-share STK leg of each position."""
        out = []
        for p in self.positions:
            for leg in p["legs"]:
                out.append((p["pos_id"], leg))
            h = p.get("hedge_shares", 0.0)
            if h:
                out.append((p["pos_id"], {"symbol": p["legs"][0]["symbol"], "sec_type": "STK",
                                          "side": "BUY" if h > 0 else "SELL", "quantity": abs(h)}))
        return out

    def underlyings(self) -> list[str]:
        return sorted({leg["symbol"] for _, leg in self.legs()})

    def is_empty(self) -> bool:
        return len(self.legs()) == 0

    def scaled(self, k: float) -> Book:
        """Same book with every quantity (and hedge) multiplied by k; used by the homogeneity test."""
        ps = []
        for p in self.positions:
            ps.append({**p, "legs": [{**leg, "quantity": leg["quantity"] * k} for leg in p["legs"]],
                       "hedge_shares": p.get("hedge_shares", 0.0) * k})
        return Book(tuple(ps))


# ---- market and marks ---------------------------------------------------------------------

@dataclass(frozen=True)
class Market:
    """Valuation inputs: `asof` (tz-aware timestamp), spot and ATM vol level per underlying."""
    asof: pd.Timestamp
    spot: dict[str, float]
    vol: dict[str, float]
    r: float = 0.0
    q: float = 0.0

    @property
    def ts(self) -> float:
        return float(pd.Timestamp(self.asof).timestamp())


@dataclass(frozen=True)
class LegMark:
    pos_id: str
    key: str
    symbol: str
    sec_type: str
    sq: float          # signed quantity
    mult: float
    value: float       # per-share value at the mark
    spot: float
    strike: float = float("nan")
    right: str = ""
    T: float = 0.0
    iv: float = float("nan")
    flags: tuple[str, ...] = field(default=())

    @property
    def dollar_value(self) -> float:
        return self.value * self.sq * self.mult


def mark(book: Book, market: Market, quotes: dict[str, tuple[float, float]] | None = None) -> list[LegMark]:
    """Mark every leg. `quotes` optionally maps contract key -> (mid, iv_or_nan) for option
    legs (see `edge_cases.mark_with_quotes` for stale handling); without a quote an option is
    valued at the underlying's vol level."""
    out = []
    for pid, leg in book.legs():
        sym, st = leg["symbol"], leg.get("sec_type", "OPT")
        if sym not in market.spot:
            raise KeyError(f"no spot for {sym}")
        S = float(market.spot[sym])
        key = contract_key(leg)
        sq, mult = signed_qty(leg), multiplier(leg)
        if st != "OPT":
            out.append(LegMark(pid, key, sym, st, sq, mult, S, S))
            continue
        K, right = float(leg["strike"]), leg["right"]
        T = years_to_expiry(leg["expiration"], market.ts)
        flags: tuple[str, ...] = ()
        if quotes is not None and key in quotes:
            mid, iv = quotes[key]
            if not np.isfinite(iv):
                iv, flags = float(market.vol[sym]), ("NO_IV",)
            value = float(mid)
        else:
            iv = float(market.vol[sym])
            value = float(bs.price(S, K, T, iv, right, market.r, market.q))
        out.append(LegMark(pid, key, sym, st, sq, mult, value, S, K, right, T, iv, flags))
    return out


def book_value(marks: list[LegMark]) -> float:
    return float(sum(m.dollar_value for m in marks))


def dollar_greeks(marks: list[LegMark], market: Market) -> dict[str, dict[str, float]]:
    """Per underlying: delta$ (per 1.00 log-return, = Delta*S*q*mult), gamma$ (Gamma*S^2*q*mult, so
    that 1/2*gamma$*x^2 is the second-order P&L in the log-return x), vega$ per 1.00 vol,
    theta$ per calendar day. STK/FUT legs contribute delta$ only."""
    out: dict[str, dict[str, float]] = {}
    for m in marks:
        g = out.setdefault(m.symbol, {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0})
        scale = m.sq * m.mult
        if m.sec_type != "OPT":
            g["delta"] += m.spot * scale
            continue
        gr = bs.greeks(m.spot, m.strike, m.T, m.iv, m.right, market.r, market.q)
        g["delta"] += gr.delta * m.spot * scale
        g["gamma"] += gr.gamma * m.spot**2 * scale
        g["vega"] += gr.vega * scale
        g["theta"] += gr.theta * scale
    return out
