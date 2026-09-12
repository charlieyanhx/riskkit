"""Stress ladders, named historical scenarios, and reverse stress.

Conventions: a scenario is (spot log return, vol change in vol units, days); P&L is
positive = gain, reported in dollars from full revaluation of the marked book. A loss limit
is a positive number of dollars. Reverse stress finds the *smallest* shock on one axis that
loses at least the limit, given a response on the other axis (default: vol rises one vol
point per 1 % of spot fall, i.e. dvol = -x; the spot response to a vol shock defaults to
zero). It brackets outward from zero on a grid of `n_grid` steps (default 512: 0.1 % of
spot over a 50 % range, 0.2 vol points over 100 points) and then uses Brent's method on the
first cell that breaches, so the found shock reproduces the limit loss to the solver
tolerance (tested at 1e-6 dollars). A loss region narrower than one grid step can be
stepped over; `ReverseStress.grid_step` records the resolution and `n_grid` is an argument.

`HISTORICAL_SCENARIOS` are close-to-close index moves from public data, applied as a log
spot move and an absolute ATM-vol change (VIX points stand in for 30-day ATM vol points);
`days` is the calendar length of the window, which is what `pricing.revalue` moves T by:
- Oct-2008: S&P 500 1166.36 (30 Sep) -> 968.75 (31 Oct) = -16.9 %; VIX 39.39 -> 59.89 (+20.5);
  31 calendar days. Source: S&P Dow Jones Indices closes via FRED series SP500; Cboe VIX
  historical data.
- Mar-2020: S&P 500 3386.15 (19 Feb) -> 2237.40 (23 Mar) = -33.9 %; VIX 14.38 (19 Feb) -> 82.69
  (16 Mar) = +68.3; 33 calendar days. Same sources.
- 16-Mar-2020 (one day): S&P 500 2711.02 -> 2386.13 = -12.0 %; VIX 57.83 -> 82.69 (+24.9).
- Feb-2018 (5 Feb, one day): S&P 500 2762.13 -> 2648.94 = -4.1 %; VIX 17.31 -> 37.32 (+20.0).
- Aug-2024 vol spike (2 Aug -> 5 Aug): S&P 500 5346.56 -> 5186.33 = -3.0 %; VIX 23.39 -> 38.57
  (+15.2; intraday high 65.73 on 5 Aug). Same sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import pricing as bs
from .positions import LegMark, Market


@dataclass(frozen=True)
class Scenario:
    name: str
    spot_move: float      # log return
    vol_move: float       # vol units (0.01 = one vol point)
    days: float = 0.0     # calendar days the scenario spans (theta runs for this long)
    source: str = ""


HISTORICAL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Oct-2008", float(np.log(968.75 / 1166.36)), (59.89 - 39.39) / 100, 31, "FRED SP500; Cboe VIX history"),
    Scenario("Mar-2020", float(np.log(2237.40 / 3386.15)), (82.69 - 14.38) / 100, 33, "FRED SP500; Cboe VIX history"),
    Scenario("16-Mar-2020", float(np.log(2386.13 / 2711.02)), (82.69 - 57.83) / 100, 1, "FRED SP500; Cboe VIX history"),
    Scenario("Feb-2018", float(np.log(2648.94 / 2762.13)), (37.32 - 17.31) / 100, 1, "FRED SP500; Cboe VIX history"),
    Scenario("Aug-2024 vol spike", float(np.log(5186.33 / 5346.56)), (38.57 - 23.39) / 100, 1, "FRED SP500; Cboe VIX history"),
)


def scenario_pnl(marks: list[LegMark], market: Market, spot_move: float, vol_move: float, days: float = 0.0) -> float:
    return float(bs.revalue(marks, spot_move, vol_move, days, market.r, market.q)[0])


def ladder(marks: list[LegMark], market: Market, spot_moves=(-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
           vol_moves=(-0.05, 0.0, 0.05, 0.10, 0.20), days: float = 0.0) -> pd.DataFrame:
    """P&L grid, rows = spot log moves, columns = vol changes (vol units). The (0, 0) cell is 0."""
    out = np.empty((len(spot_moves), len(vol_moves)))
    for i, x in enumerate(spot_moves):
        for j, y in enumerate(vol_moves):
            out[i, j] = scenario_pnl(marks, market, x, y, days)
    return pd.DataFrame(out, index=pd.Index(list(spot_moves), name="spot_move"), columns=pd.Index(list(vol_moves), name="vol_move"))


def historical_scenarios(marks: list[LegMark], market: Market, scenarios=HISTORICAL_SCENARIOS) -> pd.DataFrame:
    rows = [{"scenario": s.name, "spot_move": s.spot_move, "vol_move": s.vol_move, "days": s.days,
             "pnl": scenario_pnl(marks, market, s.spot_move, s.vol_move, s.days), "source": s.source} for s in scenarios]
    return pd.DataFrame(rows)


# ---- reverse stress ------------------------------------------------------------------------

@dataclass(frozen=True)
class ReverseStress:
    axis: str            # "spot" | "vol"
    direction: str       # "down" | "up"
    shock: float         # the smallest move on the axis that loses `limit` (NaN if none within max_shock)
    response: float      # the other axis at that shock
    pnl: float           # P&L at the found shock (= -limit to solver tolerance when found)
    limit: float
    found: bool
    max_shock: float
    grid_step: float = float("nan")   # bracketing resolution on the axis; a narrower loss region can be missed


N_GRID = 512


def _search(f: Callable[[float], float], limit: float, max_shock: float, sign: float, n_grid: int = N_GRID) -> tuple[float, bool]:
    """Smallest |s| with f(sign*s) <= -limit: scan outward on a grid of `n_grid` steps to
    bracket the first crossing, then Brent on that bracket (so a non-monotone P&L still
    yields the *first* breach, not an arbitrary root). A loss region narrower than
    max_shock / n_grid can be stepped over."""
    grid = np.linspace(0.0, max_shock, n_grid + 1)[1:]
    prev = 0.0
    for s in grid:
        if f(sign * s) + limit <= 0.0:
            root = brentq(lambda u: f(sign * u) + limit, prev, s, xtol=1e-14, rtol=1e-14, maxiter=500)
            return float(root), True
        prev = s
    return float("nan"), False


def reverse_stress_spot(marks: list[LegMark], market: Market, limit: float, vol_response: Callable[[float], float] | None = None,
                        days: float = 0.0, max_shock: float = 0.5, direction: str = "down", n_grid: int = N_GRID) -> ReverseStress:
    """Smallest spot log move (down by default) that loses `limit`, with vol responding by
    `vol_response(x)` (default -x: one vol point per 1 % of spot, rising when spot falls).
    Bracketing resolution max_shock / n_grid (default 0.1 % of spot)."""
    resp = vol_response or (lambda x: -x)
    sign = -1.0 if direction == "down" else 1.0

    def f(x: float) -> float:
        return scenario_pnl(marks, market, x, resp(x), days)

    s, ok = _search(f, limit, max_shock, sign, n_grid)
    x = sign * s if ok else float("nan")
    return ReverseStress("spot", direction, x, resp(x) if ok else float("nan"), f(x) if ok else float("nan"), limit, ok, max_shock,
                         max_shock / n_grid)


def reverse_stress_vol(marks: list[LegMark], market: Market, limit: float, spot_response: Callable[[float], float] | None = None,
                       days: float = 0.0, max_shock: float = 1.0, direction: str = "up", n_grid: int = N_GRID) -> ReverseStress:
    """Smallest vol change (up by default) that loses `limit`, with spot responding by
    `spot_response(y)` (default 0). Bracketing resolution max_shock / n_grid (default 0.2 vol
    points)."""
    resp = spot_response or (lambda y: 0.0)
    sign = 1.0 if direction == "up" else -1.0

    def f(y: float) -> float:
        return scenario_pnl(marks, market, resp(y), y, days)

    s, ok = _search(f, limit, max_shock, sign, n_grid)
    y = sign * s if ok else float("nan")
    return ReverseStress("vol", direction, y, resp(y) if ok else float("nan"), f(y) if ok else float("nan"), limit, ok, max_shock,
                         max_shock / n_grid)
