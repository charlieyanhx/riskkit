"""VaR and ES backtests with known size and power, and the rolling forecaster that feeds them.

Inputs are aligned daily series: realised P&L `pnl` (positive = gain), the VaR forecast
`var` made for that day (positive = loss) and, for the ES test, the ES forecast `es`. An
exception is `pnl < -var`. `confidence` c gives the nominal exception rate p = 1 - c.
Every test returns a `TestResult(statistic, p_value, reject, verdict)` at size `alpha`.

Nulls and statistics (details and citations in docs/DESIGN.md):
- Kupiec (1995) POF: H0 exception rate = p; LR = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)] ~ chi2(1).
  The asymptotic test is size-distorted at n = 250, p = 0.01; `kupiec_size` and `kupiec_power`
  compute the exact binomial rejection probabilities of that rule so the distortion is a number.
- Christoffersen (1998): independence LR_ind (first-order Markov vs i.i.d.) ~ chi2(1) and
  conditional coverage LR_cc = LR_pof + LR_ind ~ chi2(2). The CC p-value is by default
  simulated under the exact null (i.i.d. Bernoulli(p) exceptions of the same length), because
  at 250 days / 99 % the chi2(2) rule has size ~0.6 % and power ~25 %; `n_sim=0` gives the
  asymptotic p-value.
- Basel Committee (1996) traffic light: zones from the binomial cdf at 250 days / 99 %:
  green 0-4, yellow 5-9, red 10+ exceptions.
- Acerbi & Szekely (2014) Z2 = sum_t X_t I_t / (T p ES_t) + 1, E[Z2] = 0 under H0; the p-value
  is simulated from the model's own predictive distribution: a sampler you pass —
  `scenario_sampler` builds one from a method's own scenario P&L, scaled to each day's VaR —
  or, by default, Gaussian scaled by the VaR, which is exact only for a Gaussian model and
  over-rejects a correct fat-tailed one (tested). Reject when Z2 is too negative.
- Engle & Manganelli (2004) DQ: regress Hit_t = I_t - p on a constant, k lagged hits and VaR_t;
  DQ = b'X'Xb / (p(1-p)). The p-value is by default simulated under the exact null (i.i.d.
  Bernoulli(p) hits against the same VaR series), because the chi2 rule rejects a correct
  model 8-16 % of the time at 250-2,000 days; `n_sim=0` gives the asymptotic p-value with
  df = number of regressors (5 when the VaR is constant and its column is the intercept, 6
  otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2, norm

from . import pricing as bs
from . import var as V
from .positions import Book, Market, mark


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    reject: bool
    verdict: str
    n: int
    n_exceptions: int
    details: dict = field(default_factory=dict)


def exceptions(pnl: np.ndarray, var: np.ndarray) -> np.ndarray:
    pnl, var = np.asarray(pnl, dtype=float), np.broadcast_to(np.asarray(var, dtype=float), np.shape(pnl))
    return pnl < -var


def _verdict(reject: bool) -> str:
    return "reject" if reject else "accept"


# ---- Kupiec -------------------------------------------------------------------------------

def kupiec_lr(x: int | np.ndarray, n: int, p: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - x / n) + xlogy(x, x / n)
    return -2.0 * (ll0 - ll1)


def kupiec_pof(exc: np.ndarray, confidence: float = 0.99, alpha: float = 0.05) -> TestResult:
    exc = np.asarray(exc, dtype=bool)
    n, x, p = exc.size, int(exc.sum()), 1.0 - confidence
    lr = float(kupiec_lr(x, n, p))
    pv = float(chi2.sf(lr, 1))
    return TestResult("kupiec_pof", lr, pv, pv < alpha, _verdict(pv < alpha), n, x,
                      {"expected_exceptions": n * p, "observed_rate": x / n if n else float("nan")})


def kupiec_rejection_set(n: int, p: float, alpha: float = 0.05) -> np.ndarray:
    """Exception counts x at which the asymptotic chi2(1) rule rejects."""
    xs = np.arange(n + 1)
    return xs[kupiec_lr(xs, n, p) > chi2.ppf(1 - alpha, 1)]


def kupiec_size(n: int, p: float, alpha: float = 0.05) -> float:
    """Exact P(reject | true rate p) of the asymptotic rule: the sum of Binomial(n, p) mass over
    the rejection set. At n = 250, p = 0.01 this is ~0.090, not 0.05 (x = 0 rejects)."""
    return float(binom.pmf(kupiec_rejection_set(n, p, alpha), n, p).sum())


def kupiec_power(n: int, p_nominal: float, p_true: float, alpha: float = 0.05) -> float:
    """Exact P(reject | true rate p_true) when the model claims p_nominal."""
    return float(binom.pmf(kupiec_rejection_set(n, p_nominal, alpha), n, p_true).sum())


# ---- Christoffersen ----------------------------------------------------------------------

def _transitions(exc: np.ndarray) -> tuple[int, int, int, int]:
    a, b = exc[:-1].astype(int), exc[1:].astype(int)
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    return n00, n01, n10, n11


def christoffersen_independence(exc: np.ndarray, alpha: float = 0.05) -> TestResult:
    exc = np.asarray(exc, dtype=bool)
    n00, n01, n10, n11 = _transitions(exc)
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    pv = float(chi2.sf(lr, 1))
    return TestResult("christoffersen_ind", lr, pv, pv < alpha, _verdict(pv < alpha), exc.size, int(exc.sum()),
                      {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "p01": p01, "p11": p11})


def _lr_ind_batch(exc: np.ndarray) -> np.ndarray:
    """LR_ind for every row of a 2-d boolean array (vectorised `christoffersen_independence`)."""
    a, b = exc[:, :-1], exc[:, 1:]
    n00 = np.sum(~a & ~b, axis=1)
    n01 = np.sum(~a & b, axis=1)
    n10 = np.sum(a & ~b, axis=1)
    n11 = np.sum(a & b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p01 = np.where(n00 + n01 > 0, n01 / np.maximum(n00 + n01, 1), 0.0)
        p11 = np.where(n10 + n11 > 0, n11 / np.maximum(n10 + n11, 1), 0.0)
        pi = (n01 + n11) / np.maximum(n00 + n01 + n10 + n11, 1)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    return np.maximum(-2.0 * (ll0 - ll1), 0.0)


def _simulated_p(obs: float, sims: np.ndarray) -> float:
    """One-sided P(statistic >= observed) under the simulated null, with the +1 correction."""
    return float((np.sum(sims >= obs) + 1) / (sims.size + 1))


def christoffersen_cc(exc: np.ndarray, confidence: float = 0.99, alpha: float = 0.05, n_sim: int = 2000,
                      seed: int = 0) -> TestResult:
    """LR_cc = LR_pof + LR_ind. p-value simulated from `n_sim` i.i.d. Bernoulli(p) exception
    series of the same length (the exact null of the CC test); `n_sim=0` uses chi2(2). The
    asymptotic p-value is always in `details["p_asymptotic"]`."""
    exc = np.asarray(exc, dtype=bool)
    pof, ind = kupiec_pof(exc, confidence, alpha), christoffersen_independence(exc, alpha)
    lr = pof.statistic + ind.statistic
    p_asym = float(chi2.sf(lr, 2))
    if n_sim > 0:
        p = 1.0 - confidence
        sim = np.random.default_rng(seed).random((n_sim, exc.size)) < p
        lr_sim = kupiec_lr(sim.sum(axis=1), exc.size, p) + _lr_ind_batch(sim)
        pv = _simulated_p(lr, lr_sim)
    else:
        pv = p_asym
    return TestResult("christoffersen_cc", lr, pv, pv < alpha, _verdict(pv < alpha), pof.n, pof.n_exceptions,
                      {"lr_pof": pof.statistic, "lr_ind": ind.statistic, "p_asymptotic": p_asym, "n_sim": n_sim})


# ---- Basel traffic light -----------------------------------------------------------------

def traffic_light(n_exceptions: int, n_days: int = 250, confidence: float = 0.99) -> TestResult:
    """Zone by the cumulative binomial probability of `n_exceptions` or fewer under an accurate
    model: green below 95 %, yellow from 95 % to 99.99 %, red at or above 99.99 % (Basel
    Committee 1996, table 1). At 250 / 99 %: green 0-4, yellow 5-9, red 10+."""
    p = 1.0 - confidence
    cdf = float(binom.cdf(n_exceptions, n_days, p))
    zone = "green" if cdf < 0.95 else ("yellow" if cdf < 0.9999 else "red")
    tail = float(binom.sf(n_exceptions - 1, n_days, p))
    return TestResult("basel_traffic_light", float(n_exceptions), tail, zone == "red", zone, n_days, int(n_exceptions),
                      {"cumulative_probability": cdf})


# ---- Acerbi-Szekely ES test --------------------------------------------------------------

Sampler = Callable[[np.random.Generator, int], np.ndarray]


def gaussian_sampler(var: np.ndarray, confidence: float) -> Sampler:
    """P&L draws from N(0, sigma_t) with sigma_t = VaR_t / z_c: the model implied by a Gaussian VaR."""
    sigma = np.asarray(var, dtype=float) / norm.ppf(confidence)

    def draw(rng: np.random.Generator, n_sim: int) -> np.ndarray:
        return rng.standard_normal((n_sim, sigma.size)) * sigma[None, :]

    return draw


def scenario_sampler(pnl_scenarios: np.ndarray, var: np.ndarray, var_ref: float) -> Sampler:
    """P&L draws resampled from a method's own scenario P&L set (e.g. `VaRResult.pnl`), scaled
    on day t by VaR_t / var_ref so the shape is the model's and the scale is that day's
    forecast. This is the model's own predictive distribution for a historical, Monte Carlo
    or FHS method, which is what the Z2 p-value assumes."""
    scen = np.asarray(pnl_scenarios, dtype=float)
    scale = np.asarray(var, dtype=float) / var_ref

    def draw(rng: np.random.Generator, n_sim: int) -> np.ndarray:
        idx = rng.integers(0, scen.size, (n_sim, scale.size))
        return scen[idx] * scale[None, :]

    return draw


def acerbi_szekely_z2(pnl: np.ndarray, var: np.ndarray, es: np.ndarray, confidence: float = 0.99,
                      sampler: Sampler | None = None, n_sim: int = 2000, seed: int = 0, alpha: float = 0.05) -> TestResult:
    """Z2 with a simulated one-sided p-value P(Z2_sim <= Z2_obs) under the model's distribution."""
    pnl = np.asarray(pnl, dtype=float)
    T = pnl.size
    var = np.broadcast_to(np.asarray(var, dtype=float), (T,))
    es = np.broadcast_to(np.asarray(es, dtype=float), (T,))
    p = 1.0 - confidence

    def z2(x: np.ndarray) -> np.ndarray:
        hit = x < -var
        return np.sum(x * hit / es, axis=-1) / (T * p) + 1.0

    obs = float(z2(pnl))
    sampler = sampler or gaussian_sampler(var, confidence)
    rng = np.random.default_rng(seed)
    sims = z2(sampler(rng, n_sim))
    pv = float((np.sum(sims <= obs) + 1) / (n_sim + 1))
    return TestResult("acerbi_szekely_z2", obs, pv, pv < alpha, _verdict(pv < alpha), T, int(np.sum(pnl < -var)),
                      {"n_sim": n_sim, "z2_null_sd": float(np.std(sims))})


# ---- Engle-Manganelli DQ ----------------------------------------------------------------

def _dq_statistic(hit: np.ndarray, var: np.ndarray, p: float, lags: int) -> tuple[np.ndarray, int]:
    """DQ = y'X (X'X)^-1 X'y / (p(1-p)) for every row of `hit` (m, T) (centred hits I_t - p),
    and the number of regressors. `var` is (T,) and shared across rows; when it is constant
    its column is the intercept and is dropped (so X has full column rank, df = lags + 1)."""
    m, T = hit.shape
    y = hit[:, lags:]
    cols = [np.ones((m, T - lags))] + [hit[:, lags - i:T - i] for i in range(1, lags + 1)]
    if np.ptp(var[lags:]) > 0.0:
        cols.append(np.broadcast_to(var[lags:], (m, T - lags)))
    X = np.stack(cols, axis=-1)
    Xt = X.transpose(0, 2, 1)
    xty = (Xt @ y[..., None])[..., 0]
    beta = (np.linalg.pinv(Xt @ X, rcond=1e-10) @ xty[..., None])[..., 0]
    return np.sum(beta * xty, axis=1) / (p * (1 - p)), X.shape[-1]


def engle_manganelli_dq(pnl: np.ndarray, var: np.ndarray, confidence: float = 0.99, lags: int = 4,
                        alpha: float = 0.05, n_sim: int = 2000, seed: int = 0) -> TestResult:
    """p-value simulated from `n_sim` i.i.d. Bernoulli(p) hit series regressed on the same VaR
    series (the exact null of the DQ test); `n_sim=0` uses chi2(df) with df the number of
    regressors (lags + 1 for a constant VaR, lags + 2 otherwise). The asymptotic p-value is
    always in `details["p_asymptotic"]`."""
    pnl = np.asarray(pnl, dtype=float)
    T = pnl.size
    var = np.broadcast_to(np.asarray(var, dtype=float), (T,))
    p = 1.0 - confidence
    hit = (pnl < -var).astype(float) - p
    dq, df = _dq_statistic(hit[None, :], var, p, lags)
    dq = float(dq[0])
    p_asym = float(chi2.sf(dq, df))
    if n_sim > 0:
        sim = (np.random.default_rng(seed).random((n_sim, T)) < p).astype(float) - p
        pv = _simulated_p(dq, _dq_statistic(sim, var, p, lags)[0])
    else:
        pv = p_asym
    return TestResult("engle_manganelli_dq", dq, pv, pv < alpha, _verdict(pv < alpha), T, int(np.sum(hit > 0)),
                      {"lags": lags, "df": df, "p_asymptotic": p_asym, "n_sim": n_sim})


# ---- suite --------------------------------------------------------------------------------

def suite(pnl: np.ndarray, var: np.ndarray, es: np.ndarray, confidence: float = 0.99, alpha: float = 0.05,
          sampler: Sampler | None = None, n_sim: int = 2000, seed: int = 0) -> list[TestResult]:
    """All six tests; `n_sim` and `seed` drive the simulated p-values of Z2, CC and DQ, and
    `sampler` the Z2 null (see `scenario_sampler`)."""
    exc = exceptions(pnl, var)
    return [kupiec_pof(exc, confidence, alpha), christoffersen_independence(exc, alpha),
            christoffersen_cc(exc, confidence, alpha, n_sim, seed), traffic_light(int(exc.sum()), exc.size, confidence),
            acerbi_szekely_z2(pnl, var, es, confidence, sampler, n_sim, seed, alpha),
            engle_manganelli_dq(pnl, var, confidence, alpha=alpha, n_sim=n_sim, seed=seed)]


# ---- rolling out-of-sample forecasts ----------------------------------------------------

METHODS = ("historical", "parametric", "monte-carlo", "fhs")


def _market_at(book: Book, market: Market, history: pd.DataFrame, t: int) -> Market:
    """Spot and vol levels at the close before row t, reconstructed from the final market and
    the factor moves from t onwards."""
    spot, vol = {}, {}
    for u in book.underlyings():
        spot[u] = market.spot[u] * float(np.exp(-history[u + ":spot"].iloc[t:].sum()))
        vol[u] = max(market.vol[u] - float(history[u + ":vol"].iloc[t:].sum()), 0.02)
    return Market(market.asof, spot, vol, market.r, market.q)


def rolling_forecasts(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99,
                      window: int = 1000, n_test: int = 250, methods: tuple[str, ...] = METHODS,
                      n_scenarios: int = 5000, seed: int = 0, refit_every: int = 250) -> pd.DataFrame:
    """One-day-ahead VaR/ES for the last `n_test` rows of `history`, each from the preceding
    `window` rows, against the realised P&L of the book under that row's factor move (full
    revaluation). The book is held at constant maturity (T as of `market.asof`) and static
    quantities, so this backtests the *risk models*, not the trading. FHS refits its GARCH
    every `refit_every` days and filters daily in between."""
    N = len(history)
    if N < window + n_test:
        raise ValueError(f"need {window + n_test} rows, have {N}")
    us = book.underlyings()
    rows, models = [], {}
    for j, t in enumerate(range(N - n_test, N)):
        est = history.iloc[t - window:t]
        mkt = _market_at(book, market, history, t)
        ms = mark(book, mkt)
        row = {"date": history["date"].iloc[t] if "date" in history else t}
        move = history.iloc[t]
        row["pnl"] = float(bs.revalue_factors(ms, {u: np.array([move[u + ":spot"]]) for u in us},
                                              {u: np.array([move[u + ":vol"]]) for u in us}, 1, mkt.r, mkt.q)[0])
        for m in methods:
            if m == "historical":
                r = V.historical_var(book, mkt, est, confidence, 1, ms)
            elif m == "parametric":
                r = V.parametric_var(book, mkt, est, confidence, 1, "delta-gamma-vega", ms)
            elif m == "monte-carlo":
                r = V.monte_carlo_var(book, mkt, est, confidence, 1, n_scenarios, seed + t, ms)
            elif m == "fhs":
                if j % refit_every == 0:
                    models = {}
                r, models = V.fhs_var(book, mkt, est, confidence, 1, n_scenarios, seed + t, ms, models)
            else:
                raise ValueError(f"unknown method {m!r}")
            row[f"var_{m}"], row[f"es_{m}"] = r.var, r.es
        rows.append(row)
    return pd.DataFrame(rows)
