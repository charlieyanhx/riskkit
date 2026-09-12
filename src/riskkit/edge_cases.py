"""Limit-down, empty-book and stale-quote semantics — the risk-desk screening question
("the market is limit-down, the book is empty, VaR says risk fell — what do you do?")
answered as code contracts, each with a named test in tests/test_edge_cases.py:

1. Empty book (`test_empty_book_reports_empty_not_ok`): every risk number is zero and the
   report state is "EMPTY", never "OK". Zero risk is a state to explain (positions feed down?
   book flattened?), not a result to file.
2. Limit-down / locked underlying (`test_limit_down_does_not_reduce_var`): the book is marked
   at the last VALID mid (a locked price is not a two-sided market), the scenario set is
   widened with the lock move continuing (1x, 2x, 3x the lock move with the vol factor
   responding by its historical beta), and VaR / ES are never allowed to fall below the last
   unlocked values — if the recomputation is lower it is frozen at the previous number with a
   "VAR_FROZEN" flag saying why. The state is "LIMIT_DOWN" (or "LIMIT_UP" for a positive lock
   move; the contract is the same).
3. Stale quote (`test_stale_quote_marks_to_last_valid_mid_and_flags`): a leg whose latest
   valid quote is older than `stale_after` is marked at that last valid mid and flagged
   "STALE_QUOTE" with its age; a quote whose mid has no implied vol is flagged "NO_IV" and
   the leg's vol falls back to the underlying's level. The state is "STALE".

State precedence: EMPTY > LIMIT_DOWN / LIMIT_UP > STALE > OK. Flags carry (kind, subject,
reason) so the report can print the sentence a trader acts on. `RiskAssessment.market` is the
market the marks and the numbers were computed on (the last valid mid under a lock), so a
report prints the spot the numbers refer to, not the locked print.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing as bs
from .positions import Book, LegMark, Market, contract_key, mark
from .var import VaRResult, historical_var

STATES = ("OK", "EMPTY", "LIMIT_DOWN", "LIMIT_UP", "STALE")
DEFAULT_STALE_AFTER = pd.Timedelta(minutes=15)
LOCK_CONTINUATIONS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Quote:
    ts: pd.Timestamp
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def valid(self) -> bool:
        """Two-sided and finite: 0 <= bid <= ask, ask > 0. A locked or crossed print is not valid."""
        return bool(np.isfinite(self.bid) and np.isfinite(self.ask) and 0.0 <= self.bid <= self.ask and self.ask > 0.0)


@dataclass(frozen=True)
class Flag:
    kind: str       # EMPTY | LIMIT_DOWN | LIMIT_UP | VAR_FROZEN | STALE_QUOTE | NO_IV
    subject: str    # contract key, symbol or "book"
    reason: str


@dataclass(frozen=True)
class Lock:
    """An underlying that is locked (limit-down or limit-up) or whose print is stale."""
    symbol: str
    last_valid_spot: float   # last two-sided mid before the lock
    lock_move: float         # log move from last valid mid to the locked price (negative = limit-down)
    since: pd.Timestamp


@dataclass(frozen=True)
class RiskAssessment:
    state: str
    var: VaRResult
    flags: tuple[Flag, ...]
    marks: tuple[LegMark, ...]
    market: Market           # the market the marks and `var` were computed on (last valid mid under a lock)


def latest_valid_quote(quotes: list[Quote]) -> Quote | None:
    valid = [q for q in quotes if q.valid]
    return max(valid, key=lambda q: q.ts) if valid else None


def mark_with_quotes(book: Book, market: Market, quotes: dict[str, list[Quote]] | None,
                     stale_after: pd.Timedelta = DEFAULT_STALE_AFTER) -> tuple[list[LegMark], list[Flag]]:
    """Mark option legs at their latest valid quote mid (implied vol inverted), flagging stale
    quotes and mids without an implied vol; legs without quotes are model-marked."""
    flags: list[Flag] = []
    by_key: dict[str, tuple[float, float]] = {}
    asof = pd.Timestamp(market.asof)
    for _, leg in book.legs():
        if leg.get("sec_type", "OPT") != "OPT" or not quotes:
            continue
        key = contract_key(leg)
        q = latest_valid_quote(quotes.get(key, []))
        if q is None:
            continue
        age = asof - pd.Timestamp(q.ts)
        if age > stale_after:
            flags.append(Flag("STALE_QUOTE", key, f"last valid quote is {age} old (> {stale_after}); marked at its mid {q.mid:.4f}"))
        T = max(0.0, (pd.Timestamp(leg["expiration"] + "T20:00", tz="UTC") - asof).total_seconds() / 86400.0 / 365.0)
        iv = bs.implied_vol(q.mid, market.spot[leg["symbol"]], float(leg["strike"]), T, leg["right"], market.r, market.q)
        if not np.isfinite(iv):
            flags.append(Flag("NO_IV", key, f"mid {q.mid:.4f} has no implied vol; vol falls back to {market.vol[leg['symbol']]:.4f}"))
        by_key[key] = (q.mid, iv)
    return mark(book, market, by_key), flags


def spot_vol_beta(history: pd.DataFrame, symbol: str) -> float:
    """Historical response of the vol factor to the spot factor (vol units per unit log return)."""
    x, y = history[symbol + ":spot"].to_numpy(float), history[symbol + ":vol"].to_numpy(float)
    vx = float(np.var(x))
    return float(np.cov(x, y, ddof=0)[0, 1] / vx) if vx > 0 else 0.0


def lock_scenarios(book: Book, history: pd.DataFrame, lock: Lock, continuations=LOCK_CONTINUATIONS) -> np.ndarray:
    """Extra h-day scenarios [spot..., vol...]: the lock move continuing k times, the locked
    symbol's vol responding by its historical beta, every other factor at zero."""
    us = book.underlyings()
    beta = spot_vol_beta(history, lock.symbol)
    rows = []
    for k in continuations:
        row = np.zeros(2 * len(us))
        i = us.index(lock.symbol)
        row[i], row[len(us) + i] = k * lock.lock_move, beta * k * lock.lock_move
        rows.append(row)
    return np.vstack(rows)


def assess(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99, horizon_days: int = 1,
           quotes: dict[str, list[Quote]] | None = None, stale_after: pd.Timedelta = DEFAULT_STALE_AFTER,
           lock: Lock | None = None, previous: VaRResult | None = None) -> RiskAssessment:
    """Historical-simulation VaR/ES under the three contracts above. `previous` is the last
    result computed while the market was unlocked; under a lock the new numbers may not fall
    below it."""
    if book.is_empty():
        zero = VaRResult("historical", confidence, horizon_days, 0.0, 0.0, 0, ("EMPTY",), np.zeros(0))
        return RiskAssessment("EMPTY", zero, (Flag("EMPTY", "book", "book has no legs: zero risk is a state to explain, not a result"),), (), market)
    flags: list[Flag] = []
    mkt = market
    extra = None
    if lock is not None and lock.symbol in book.underlyings():
        mkt = Market(market.asof, {**market.spot, lock.symbol: lock.last_valid_spot}, market.vol, market.r, market.q)
        extra = lock_scenarios(book, history, lock)
        flags.append(Flag("LIMIT_DOWN" if lock.lock_move < 0 else "LIMIT_UP", lock.symbol,
                          f"locked since {lock.since} at {lock.lock_move:+.2%} from the last valid mid {lock.last_valid_spot:.4f}; "
                          f"marked at that mid; scenario set widened with the lock move continuing x{', x'.join(f'{k:g}' for k in LOCK_CONTINUATIONS)}"))
    marks, qflags = mark_with_quotes(book, mkt, quotes, stale_after)
    flags.extend(qflags)
    res = historical_var(book, mkt, history, confidence, horizon_days, marks, extra)
    if lock is not None and previous is not None and (res.var < previous.var or res.es < previous.es):
        frozen_var, frozen_es = max(res.var, previous.var), max(res.es, previous.es)
        flags.append(Flag("VAR_FROZEN", lock.symbol,
                          f"recomputed VaR {res.var:,.2f} / ES {res.es:,.2f} below the last unlocked {previous.var:,.2f} / {previous.es:,.2f}; "
                          f"frozen at the unlocked values — a locked market prints no information, not less risk"))
        res = VaRResult(res.method, res.confidence, res.horizon, frozen_var, frozen_es, res.n_scenarios,
                        res.notes + ("VAR_FROZEN",), res.pnl)
    kinds = {f.kind for f in flags}
    if "LIMIT_DOWN" in kinds or "LIMIT_UP" in kinds:
        state = "LIMIT_DOWN" if "LIMIT_DOWN" in kinds else "LIMIT_UP"
    else:
        state = "STALE" if "STALE_QUOTE" in kinds else "OK"
    return RiskAssessment(state, res, tuple(flags), tuple(marks), mkt)
