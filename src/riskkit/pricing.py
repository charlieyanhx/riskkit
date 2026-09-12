"""Black-Scholes(-Merton) price, Greeks and implied vol, plus full revaluation of a marked
book under a (spot shock, vol shock, time step) scenario.

Units and conventions
- Prices are per share; the position layer applies signed quantity x multiplier.
- `vega` is per 1.00 change in vol (100 vol points); `theta` per calendar day; `T` in years
  on a 365-day calendar, the same conventions as deskboard's `engine/greeks.py`.
- Spot shocks are log returns (S' = S * exp(x)); vol shocks are absolute changes in the
  option's implied vol (0.01 = one vol point); a time step of `days` moves T by days/365.
- P&L is value(after) - value(before), positive = gain.

`price`, `greeks`, `implied_vol` have the same signatures as deskboard's pricer, so the
`pricers` sibling repo (or a surface pricer) is a drop-in: only these three names are used
outside this module, plus `price_vec`, the array form of `price` used by the scenario
engines. The scalar three are copied verbatim from deskboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

YEAR = 365.0
VOL_FLOOR = 1e-4  # a vol shock can never take an implied vol at or below zero


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float      # per 1.00 change in vol (i.e. per 100 vol points)
    theta: float     # per calendar day
    rho: float


def _d1d2(S, K, T, sigma, r, q):
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / v
    return d1, d1 - v


def price(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    if right == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def greeks(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.0, q: float = 0.0) -> Greeks:
    if T <= 0:
        itm = (S > K) if right == "C" else (S < K)
        d = (1.0 if right == "C" else -1.0) * (1.0 if itm else 0.0)
        return Greeks(price(S, K, T, sigma, right, r, q), d, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    sq = np.sqrt(T)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * sq)
    vega = S * disc_q * pdf * sq
    if right == "C":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)) / YEAR
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)) / YEAR
        rho = -K * T * disc_r * norm.cdf(-d2)
    return Greeks(price(S, K, T, sigma, right, r, q), float(delta), float(gamma), float(vega), float(theta), float(rho))


def implied_vol(target: float, S: float, K: float, T: float, right: str, r: float = 0.0, q: float = 0.0,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Brent inversion; NaN when the price is outside no-arbitrage bounds or T <= 0."""
    if T <= 0 or not np.isfinite(target):
        return float("nan")
    intrinsic = price(S, K, T, lo, right, r, q)
    if target < intrinsic - 1e-12 or target > price(S, K, T, hi, right, r, q) + 1e-12:
        return float("nan")
    try:
        return float(brentq(lambda s: price(S, K, T, s, right, r, q) - target, lo, hi, xtol=1e-10, maxiter=200))
    except ValueError:
        return float("nan")


# ---- array form and revaluation ---------------------------------------------------------

def price_vec(S: np.ndarray, K: float, T: float, sigma: np.ndarray, right: str, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """`price` over arrays of spot and vol (one T per call). Equals `price` element-wise."""
    S = np.asarray(S, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), VOL_FLOOR)
    if T <= 0:
        return np.maximum(0.0, (S - K) if right == "C" else (K - S))
    d1, d2 = _d1d2(S, K, T, sigma, r, q)
    if right == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def revalue(marks, spot_shock, vol_shock, days: float = 0.0, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """Full revaluation. `marks` is a sequence of `positions.LegMark`; `spot_shock` and
    `vol_shock` are scalars or arrays of the same shape, per scenario, applied to every
    underlying (use `revalue_factors` for per-underlying shocks). Returns book P&L per
    scenario in dollars: sum over legs of (value' - value0) * signed_qty * multiplier."""
    x = np.atleast_1d(np.asarray(spot_shock, dtype=float))
    y = np.atleast_1d(np.asarray(vol_shock, dtype=float))
    x, y = np.broadcast_arrays(x, y)
    if not marks:
        return np.zeros(x.shape)
    return revalue_factors(marks, {m.symbol: x for m in marks}, {m.symbol: y for m in marks}, days, r, q)


def revalue_factors(marks, spot_shocks: dict[str, np.ndarray], vol_shocks: dict[str, np.ndarray],
                    days: float = 0.0, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    """Full revaluation with one (spot, vol) shock series per underlying symbol. Every array
    in `spot_shocks` / `vol_shocks` has the same length n; returns book P&L of shape (n,)."""
    pnl = None
    for m in marks:
        x = np.asarray(spot_shocks[m.symbol], dtype=float)
        S1 = m.spot * np.exp(x)
        if m.sec_type == "OPT":
            y = np.asarray(vol_shocks[m.symbol], dtype=float)
            T1 = max(0.0, m.T - days / YEAR)
            v1 = price_vec(S1, m.strike, T1, m.iv + y, m.right, r, q)
        else:  # STK / FUT: linear in the spot factor
            v1 = S1
        leg = (v1 - m.value) * m.sq * m.mult
        pnl = leg if pnl is None else pnl + leg
    if pnl is None:
        n = len(next(iter(spot_shocks.values()))) if spot_shocks else 1
        return np.zeros(n)
    return pnl
