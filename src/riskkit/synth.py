"""Seeded synthetic demo: a four-position book on one underlying and a five-year daily
factor history with volatility clustering. `riskkit demo` regenerates both byte for byte.

History model (all seeded, `numpy.random.default_rng`):
- spot log returns r_t = sigma_t * eps_t, eps_t ~ Student-t(6) scaled to unit variance;
  sigma_t^2 follows GJR-GARCH(1,1): omega + (alpha + gamma*1{r<0}) r_{t-1}^2 + beta sigma_{t-1}^2,
  so vol clusters and rises after down days;
- the ATM vol level v_t = 1.15 * sqrt(252) * sigma_{t+1} + u_t with u_t an AR(1) premium,
  i.e. end-of-day implied vol reacts to the day's return (a negative spot-vol correlation);
- the vol factor is dv_t = v_t - v_{t-1} in vol units (0.01 = one vol point).

Columns written: `date, SYN:spot, SYN:vol, SYN:spot_level, SYN:vol_level`. Values are rounded
to 10 decimals before writing so the CSV is identical across platforms (libm differences in
`exp` / `sqrt` sit far below that).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .positions import Book, Market

SYMBOL = "SYN"
N_DAYS = 1260          # ~5 years of business days
ASOF = "2026-06-12"    # last history date; the book is valued at the close of this day
SPOT0 = 500.0
FUT_MULT = 50.0


def demo_book(asof: str = ASOF) -> list[dict]:
    """Four positions on SYN: a put spread (sleeve A), a strangle with a stock hedge (C),
    a long put (S) and a short future (F, a macro hedge with a per-contract multiplier).
    Expirations are relative to `asof` so T is stable when the demo is regenerated."""
    d = pd.Timestamp(asof)
    e45 = (d + timedelta(days=45)).strftime("%Y%m%d")
    e30 = (d + timedelta(days=30)).strftime("%Y%m%d")
    e60 = (d + timedelta(days=60)).strftime("%Y%m%d")
    e90 = (d + timedelta(days=90)).strftime("%Y%m%d")

    def opt(strike, right, side, qty, exp):
        return {"symbol": SYMBOL, "sec_type": "OPT", "expiration": exp, "strike": float(strike),
                "right": right, "side": side, "quantity": float(qty)}

    return [
        {"pos_id": "A-0601", "sleeve": "A", "legs": [opt(480, "P", "SELL", 10, e45), opt(440, "P", "BUY", 10, e45)]},
        {"pos_id": "C-0603", "sleeve": "C", "legs": [opt(470, "P", "SELL", 6, e30), opt(530, "C", "SELL", 6, e30)],
         "hedge_shares": -30},
        {"pos_id": "S-0610", "sleeve": "S", "legs": [opt(490, "P", "BUY", 2, e60)]},
        {"pos_id": "F-0611", "sleeve": "F", "legs": [
            {"symbol": SYMBOL, "sec_type": "FUT", "expiration": e90, "side": "SELL", "quantity": 1.0, "multiplier": FUT_MULT}]},
    ]


def demo_history(seed: int = 11, n_days: int = N_DAYS, asof: str = ASOF) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = 0.011                                   # unconditional daily vol ~17.5% annualised
    alpha, gamma, beta = 0.05, 0.08, 0.88
    omega = daily**2 * (1 - alpha - beta - gamma / 2)
    nu = 6.0
    eps = rng.standard_t(nu, n_days) / np.sqrt(nu / (nu - 2))
    eta = rng.normal(0.0, 1.0, n_days)
    sig2 = np.empty(n_days + 1)
    r = np.empty(n_days)
    sig2[0] = daily**2
    for t in range(n_days):
        r[t] = np.sqrt(sig2[t]) * eps[t]
        sig2[t + 1] = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * sig2[t]
    u = np.empty(n_days)
    u[0] = 0.0
    for t in range(1, n_days):
        u[t] = 0.9 * u[t - 1] + 0.004 * eta[t]
    vol_level = 1.15 * np.sqrt(252.0) * np.sqrt(sig2[1:]) + u
    spot_level = SPOT0 * np.exp(np.cumsum(r) - np.sum(r))  # ends exactly at SPOT0
    dv = np.diff(vol_level, prepend=vol_level[0])
    dates = pd.bdate_range(end=pd.Timestamp(asof), periods=n_days)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       f"{SYMBOL}:spot": np.round(r, 10), f"{SYMBOL}:vol": np.round(dv, 10),
                       f"{SYMBOL}:spot_level": np.round(spot_level, 10), f"{SYMBOL}:vol_level": np.round(vol_level, 10)})
    return df.iloc[1:].reset_index(drop=True)  # drop the first row: its dv is zero by construction


def demo_market(history: pd.DataFrame, r: float = 0.0, q: float = 0.0) -> Market:
    last = history.iloc[-1]
    asof = pd.Timestamp(last["date"], tz="UTC") + pd.Timedelta(hours=20)  # 16:00 New York close
    return Market(asof=asof, spot={SYMBOL: float(last[f"{SYMBOL}:spot_level"])},
                  vol={SYMBOL: float(last[f"{SYMBOL}:vol_level"])}, r=r, q=q)


def write_demo(out_dir: str | Path, seed: int = 11) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist = demo_history(seed)
    hp, bp = out / "history.csv", out / "book.json"
    hist.to_csv(hp, index=False, float_format="%.10f", lineterminator="\n")
    bp.write_text(json.dumps(demo_book(), indent=1) + "\n")
    return bp, hp


def load_demo(data_dir: str | Path) -> tuple[Book, Market, pd.DataFrame]:
    d = Path(data_dir)
    hist = pd.read_csv(d / "history.csv")
    return Book.load(d / "book.json"), demo_market(hist), hist
