"""VaR and ES by four methods on a marked book, all on the same conventions.

Conventions (kept by every method, tested in tests/test_var_identities.py)
- P&L positive = gain; `var` and `es` are positive numbers of dollars = losses.
- `var` at confidence c is the c-quantile of the loss distribution; `es` is the mean loss
  beyond it (Acerbi-Tasche tail average for scenario sets), so `es >= var` always.
- Horizon h days: simulation methods use h-day factor moves (overlapping windows for
  historical, path simulation for FHS, sqrt-time covariance for Monte Carlo) and a time step
  of h calendar days through theta; the parametric method scales the factor covariance by h.
- Full revaluation: historical, Monte Carlo and FHS reprice every leg with Black-Scholes
  under each scenario. The parametric method is the one Greek-based approximation and says so
  in its `method` name.
- Factor histories are DataFrames with `<sym>:spot` (daily log returns) and `<sym>:vol`
  (daily vol-level changes, 0.01 = one vol point) per underlying.

Parametric delta-gamma-vega (Zangari 1996 / Jorion ch. 10 with a Cornish-Fisher quantile):
P&L ~ theta*h + b'z + 1/2 z'Lz with z ~ N(0, h*Sigma); the cumulants of a Gaussian quadratic
form are k1 = 1/2 tr(LS), k2 = b'Sb + 1/2 tr((LS)^2), k3 = 3 b'SLSb + tr((LS)^3),
k4 = 12 b'SLSLSb + 3 tr((LS)^4) (S = h*Sigma). The CF quantile uses skew and excess kurtosis
from those; ES is the exact tail mean of the CF quantile function. The CF expansion is a
four-cumulant approximation that is only accurate for moderate departures from normality
(Jaschke 2002): it is used when |skew| <= CF_MAX_SKEW, excess kurtosis <= CF_MAX_EXKURT AND
the polynomial is monotone over the loss tail VaR and ES read, [-CF_Z_MAX, z_p] (Maillard
2012's condition restricted to that interval: with kurtosis near zero the cubic always turns
over somewhere, but at |z| ~ 5-10 where the Gaussian weight is nil); otherwise, or with
quantile="exact", the quantile and tail mean are taken from the exact distribution of the
quadratic form, simulated with a fixed seed (theta*h + b'z + 1/2 z'Lz over n draws of
z ~ N(0, S)) and the result says so in `notes`. Measured on the demo book against 400k
draws, CF is within 2 % at h = 1 (skew -0.4, kurtosis 0.7) and 10-14 % off at h = 10
(skew -1.6, kurtosis 5.4), which is why the domain guard exists (tests/test_var_identities.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import pricing as bs
from .positions import Book, LegMark, Market, dollar_greeks, mark


@dataclass(frozen=True)
class VaRResult:
    method: str
    confidence: float
    horizon: int          # days
    var: float            # dollars, positive = loss
    es: float             # dollars, positive = loss, >= var
    n_scenarios: int      # 0 for the parametric method
    notes: tuple[str, ...] = ()
    pnl: np.ndarray | None = field(default=None, repr=False, compare=False)  # scenario P&L, when simulated


CF_MAX_SKEW = 1.0        # |skew| beyond which the Cornish-Fisher quantile is not trusted
CF_MAX_EXKURT = 3.0      # excess kurtosis beyond which it is not trusted
CF_Z_MAX = 6.0           # the CF polynomial must be monotone on [-CF_Z_MAX, z_p], the loss tail used
EXACT_N_DRAWS = 200_000  # draws for the exact quadratic-form fallback


# ---- scenario set -> (VaR, ES) ------------------------------------------------------------

def var_es_from_pnl(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """VaR = linear-interpolated c-quantile of losses (= the worst scenario at c = 1);
    ES = tail average of the worst n(1-c) losses with fractional weight on the boundary
    scenario (Acerbi & Tasche 2002). ES >= VaR for c >= 0.5."""
    loss = -np.asarray(pnl, dtype=float)
    n = loss.size
    if n == 0:
        return 0.0, 0.0
    var = float(np.quantile(loss, confidence))
    tail = n * (1.0 - confidence)
    if tail <= 0.0:
        return var, var
    srt = np.sort(loss)[::-1]
    k = int(np.floor(tail))
    total = float(srt[:k].sum())
    if k < n and tail - k > 0:
        total += (tail - k) * float(srt[k])
    return var, total / tail


def factor_names(book: Book) -> tuple[list[str], list[str]]:
    """(spot factor columns, vol factor columns) in underlying order."""
    us = book.underlyings()
    return [u + ":spot" for u in us], [u + ":vol" for u in us]


def _check_history(history: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in history.columns]
    if missing:
        raise KeyError(f"history lacks factor columns {missing}")
    if len(history) < 30:
        raise ValueError(f"history has {len(history)} rows; need at least 30")


def _marks(book: Book, market: Market, marks: list[LegMark] | None) -> list[LegMark]:
    return marks if marks is not None else mark(book, market)


def _reval(marks, market, spot_by_sym: dict[str, np.ndarray], vol_by_sym: dict[str, np.ndarray], h: int) -> np.ndarray:
    return bs.revalue_factors(marks, spot_by_sym, vol_by_sym, days=h, r=market.r, q=market.q)


def _split(book: Book, arr: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Columns of `arr` (n, 2u) ordered [spot_1..spot_u, vol_1..vol_u] -> per-symbol dicts."""
    us = book.underlyings()
    u = len(us)
    return {s: arr[:, i] for i, s in enumerate(us)}, {s: arr[:, u + i] for i, s in enumerate(us)}


# ---- historical simulation ----------------------------------------------------------------

def historical_var(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99,
                   horizon_days: int = 1, marks: list[LegMark] | None = None,
                   extra_scenarios: np.ndarray | None = None) -> VaRResult:
    """Every h-day window of the factor history is a scenario, fully revalued. `extra_scenarios`
    (rows of [spot..., vol...] h-day shocks) are appended, which is how the limit-down rule
    widens the set."""
    spot_cols, vol_cols = factor_names(book)
    ms = _marks(book, market, marks)
    if not ms:
        return VaRResult("historical", confidence, horizon_days, 0.0, 0.0, 0, ("EMPTY",), np.zeros(0))
    _check_history(history, spot_cols + vol_cols)
    arr = history[spot_cols + vol_cols].to_numpy(dtype=float)
    if horizon_days > 1:
        arr = np.lib.stride_tricks.sliding_window_view(arr, horizon_days, axis=0).sum(axis=-1)
    if extra_scenarios is not None and len(extra_scenarios):
        arr = np.vstack([arr, np.asarray(extra_scenarios, dtype=float).reshape(-1, arr.shape[1])])
    spot, vol = _split(book, arr)
    pnl = _reval(ms, market, spot, vol, horizon_days)
    v, e = var_es_from_pnl(pnl, confidence)
    return VaRResult("historical", confidence, horizon_days, v, e, len(pnl), (), pnl)


# ---- parametric delta-gamma-vega with Cornish-Fisher --------------------------------------

def cornish_fisher_z(z: float, skew: float, exkurt: float) -> float:
    return z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * exkurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36


def _cf_coefficients(skew: float, exkurt: float) -> tuple[float, float, float]:
    """(a, b, c) of the CF polynomial a z^3 + b z^2 + c z + d."""
    return exkurt / 24 - skew**2 / 18, skew / 6, 1 - exkurt / 8 + 5 * skew**2 / 36


def cornish_fisher_is_monotone(skew: float, exkurt: float) -> bool:
    """The CF polynomial a z^3 + b z^2 + c z + d is a valid quantile function iff its
    derivative 3a z^2 + 2b z + c is non-negative everywhere (Maillard 2012)."""
    a, b, c = _cf_coefficients(skew, exkurt)
    if abs(a) < 1e-14:
        return abs(b) < 1e-14 and c >= 0
    return a > 0 and 4 * b**2 - 12 * a * c <= 1e-12


def cornish_fisher_is_monotone_on(skew: float, exkurt: float, z_lo: float, z_hi: float) -> bool:
    """Derivative 3a z^2 + 2b z + c >= 0 on [z_lo, z_hi]: checked at both ends and at the
    vertex of the quadratic when it lies inside."""
    a, b, c = _cf_coefficients(skew, exkurt)

    def d(z: float) -> float:
        return 3 * a * z**2 + 2 * b * z + c

    pts = [z_lo, z_hi]
    if abs(a) > 1e-14 and z_lo < -b / (3 * a) < z_hi:
        pts.append(-b / (3 * a))
    return min(d(z) for z in pts) >= -1e-12


def cornish_fisher_tail_mean(p: float, skew: float, exkurt: float) -> float:
    """E[q_cf(Z) | Z <= z_p] for Z ~ N(0,1): the exact tail mean of the CF quantile polynomial,
    using int t^n phi over (-inf, z]: 1 -> p, t -> -phi, t^2 -> p - z phi, t^3 -> -(z^2+2) phi."""
    z, ph = norm.ppf(p), norm.pdf(norm.ppf(p))
    a, b, c, d = exkurt / 24 - skew**2 / 18, skew / 6, 1 - exkurt / 8 + 5 * skew**2 / 36, -skew / 6
    integral = a * (-(z**2 + 2) * ph) + b * (p - z * ph) + c * (-ph) + d * p
    return integral / p


def quadratic_form_cumulants(b: np.ndarray, L: np.ndarray, S: np.ndarray) -> tuple[float, float, float, float]:
    """Cumulants k1..k4 of b'z + 1/2 z'Lz with z ~ N(0, S)."""
    LS = L @ S
    LS2 = LS @ LS
    LS3 = LS2 @ LS
    LS4 = LS3 @ LS
    Sb = S @ b
    k1 = 0.5 * np.trace(LS)
    k2 = float(b @ Sb) + 0.5 * np.trace(LS2)
    k3 = 3.0 * float(b @ S @ L @ Sb) + np.trace(LS3)
    k4 = 12.0 * float(b @ S @ L @ S @ L @ Sb) + 3.0 * np.trace(LS4)
    return float(k1), float(k2), float(k3), float(k4)


def factor_cov(history: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Sample covariance (ddof=1) of the daily factor changes; factor means are set to zero."""
    _check_history(history, cols)
    return np.atleast_2d(np.cov(history[cols].to_numpy(dtype=float), rowvar=False, ddof=1))


def cornish_fisher_in_domain(skew: float, exkurt: float, confidence: float = 0.99) -> bool:
    """CF is used only where it is a valid quantile function over the loss tail it is read on
    ([-CF_Z_MAX, z_p], p = 1 - confidence) and the cumulants are moderate."""
    z_p = float(norm.ppf(1.0 - confidence))
    return (abs(skew) <= CF_MAX_SKEW and exkurt <= CF_MAX_EXKURT
            and cornish_fisher_is_monotone_on(skew, exkurt, -CF_Z_MAX, max(z_p, -CF_Z_MAX)))


def cornish_fisher_var_es(theta_h: float, b: np.ndarray, L: np.ndarray, S: np.ndarray, confidence: float
                          ) -> tuple[float, float, float, float, float]:
    """(VaR, ES, sd, skew, excess kurtosis) of theta_h + b'z + 1/2 z'Lz, z ~ N(0, S), by the CF
    quantile and its exact tail mean; sd = 0 gives (-mu, -mu, 0, 0, 0)."""
    k1, k2, k3, k4 = quadratic_form_cumulants(b, L, S)
    mu, sd = theta_h + k1, float(np.sqrt(max(k2, 0.0)))
    if sd == 0.0:
        return -mu, -mu, 0.0, 0.0, 0.0
    skew, exk = k3 / sd**3, k4 / sd**4
    p = 1.0 - confidence
    var = -(mu + sd * cornish_fisher_z(norm.ppf(p), skew, exk))
    es = -(mu + sd * cornish_fisher_tail_mean(p, skew, exk))
    return float(var), float(es), sd, float(skew), float(exk)


def quadratic_form_var_es(theta_h: float, b: np.ndarray, L: np.ndarray, S: np.ndarray, confidence: float,
                          n: int = EXACT_N_DRAWS, seed: int = 0) -> tuple[float, float]:
    """(VaR, ES) of theta_h + b'z + 1/2 z'Lz, z ~ N(0, S), from `n` seeded draws: the exact
    distribution of the delta-gamma(-vega) P&L up to simulation error, and homogeneous in the
    book because the draws are fixed by the seed."""
    nf = len(b)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, nf)) @ np.linalg.cholesky(S + 1e-18 * np.eye(nf)).T
    pnl = theta_h + z @ b + 0.5 * np.einsum("ij,ij->i", z @ L, z)
    return var_es_from_pnl(pnl, confidence)


def parametric_moments(book: Book, market: Market, history: pd.DataFrame, horizon_days: int = 1,
                       order: str = "delta-gamma-vega", marks: list[LegMark] | None = None
                       ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """(theta$*h, b, L, S) of the Greek expansion P&L ~ theta*h + b'z + 1/2 z'Lz, z ~ N(0, S):
    b = delta$ (and vega$ for order "delta-gamma-vega"), L = diag(gamma$) unless order "delta",
    S = h * sample covariance of the factors used. Empty book -> zero-length arrays."""
    if order not in ("delta", "delta-gamma", "delta-gamma-vega"):
        raise ValueError(f"unknown order {order!r}")
    spot_cols, vol_cols = factor_names(book)
    ms = _marks(book, market, marks)
    us = book.underlyings()
    cols = spot_cols + (vol_cols if order == "delta-gamma-vega" else [])
    if not ms:
        return 0.0, np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0))
    S = horizon_days * factor_cov(history, cols)
    g = dollar_greeks(ms, market)
    nf = len(cols)
    b, L = np.zeros(nf), np.zeros((nf, nf))
    theta = 0.0
    for i, u in enumerate(us):
        b[i] = g[u]["delta"]
        if order != "delta":
            L[i, i] = g[u]["gamma"]
        if order == "delta-gamma-vega":
            b[len(us) + i] = g[u]["vega"]
        theta += g[u]["theta"]
    return theta * horizon_days, b, L, S


def parametric_var(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99,
                   horizon_days: int = 1, order: str = "delta-gamma-vega",
                   marks: list[LegMark] | None = None, quantile: str = "cornish-fisher",
                   n_draws: int = EXACT_N_DRAWS, seed: int = 0) -> VaRResult:
    """`order` in {"delta", "delta-gamma", "delta-gamma-vega"}. Delta and gamma are in the
    spot log-return x (delta$ = Delta*S*q*mult, gamma$ = Gamma*S^2*q*mult); vega$ per 1.00
    vol; theta$*h enters the mean. With order="delta" the result is the Gaussian closed form
    VaR = z_c * sqrt(h * b'Sigma b) - theta*h exactly (k3 = k4 = 0).

    `quantile`: "cornish-fisher" uses the CF quantile inside its validity domain
    (`cornish_fisher_in_domain`) and the exact simulated quadratic form outside it, with the
    notes "cornish_fisher_nonmonotone" (not monotone on the loss tail) and / or
    "cornish_fisher_out_of_domain" (cumulants too large), plus "quadratic_form_simulated";
    "exact" always simulates (`n_draws`, `seed`)."""
    if quantile not in ("cornish-fisher", "exact"):
        raise ValueError(f"unknown quantile {quantile!r}")
    name = "parametric-" + order
    theta_h, b, L, S = parametric_moments(book, market, history, horizon_days, order, marks)
    if len(b) == 0:
        return VaRResult(name, confidence, horizon_days, 0.0, 0.0, 0, ("EMPTY",))
    var, es, sd, skew, exk = cornish_fisher_var_es(theta_h, b, L, S, confidence)
    if sd == 0.0:
        return VaRResult(name, confidence, horizon_days, max(var, 0.0), max(es, 0.0), 0, ("ZERO_VARIANCE",))
    notes: tuple[str, ...] = ()
    if quantile == "cornish-fisher" and cornish_fisher_in_domain(skew, exk, confidence):
        return VaRResult(name, confidence, horizon_days, var, es, 0, notes)
    if quantile == "cornish-fisher":
        z_p = max(float(norm.ppf(1.0 - confidence)), -CF_Z_MAX)
        if not cornish_fisher_is_monotone_on(skew, exk, -CF_Z_MAX, z_p):
            notes += ("cornish_fisher_nonmonotone",)
        if abs(skew) > CF_MAX_SKEW or exk > CF_MAX_EXKURT:
            notes += ("cornish_fisher_out_of_domain",)
    var, es = quadratic_form_var_es(theta_h, b, L, S, confidence, n_draws, seed)
    return VaRResult(name, confidence, horizon_days, var, es, n_draws, notes + ("quadratic_form_simulated",))


# ---- Monte Carlo from a fitted covariance ------------------------------------------------

def monte_carlo_var(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99,
                    horizon_days: int = 1, n_scenarios: int = 20_000, seed: int = 0,
                    marks: list[LegMark] | None = None) -> VaRResult:
    """Gaussian draws z ~ N(0, h*Sigma) of (log-spot, vol) changes from the sample covariance,
    fully revalued. Same seed -> same draws, so the result is exactly homogeneous in size."""
    spot_cols, vol_cols = factor_names(book)
    ms = _marks(book, market, marks)
    if not ms:
        return VaRResult("monte-carlo", confidence, horizon_days, 0.0, 0.0, 0, ("EMPTY",), np.zeros(0))
    S = horizon_days * factor_cov(history, spot_cols + vol_cols)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(S + 1e-18 * np.eye(len(S)))
    z = rng.standard_normal((n_scenarios, len(S))) @ chol.T
    spot, vol = _split(book, z)
    pnl = _reval(ms, market, spot, vol, horizon_days)
    v, e = var_es_from_pnl(pnl, confidence)
    return VaRResult("monte-carlo", confidence, horizon_days, v, e, n_scenarios, (), pnl)


# ---- filtered historical simulation (GARCH via arch) ---------------------------------------

@dataclass(frozen=True)
class Garch11:
    """GARCH(1,1) parameters in the data's own units (the fit scales by 100 internally,
    which `arch` expects for numerical stability, and unscales omega here)."""
    omega: float
    alpha: float
    beta: float

    def filter(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Conditional sigma_t for each observation and the one-step-ahead sigma_{T+1}.
        Initialised at the sample variance; after a few hundred observations that choice is
        immaterial."""
        x = np.asarray(x, dtype=float)
        s2 = np.empty(len(x) + 1)
        s2[0] = float(np.var(x))
        for t in range(len(x)):
            s2[t + 1] = self.omega + self.alpha * x[t] ** 2 + self.beta * s2[t]
        return np.sqrt(s2[:-1]), float(np.sqrt(s2[-1]))

    def simulate(self, sigma_next: float, eps: np.ndarray) -> np.ndarray:
        """Paths from resampled standardised residuals `eps` (n, h): returns (n, h) daily moves."""
        n, h = eps.shape
        out = np.empty((n, h))
        s2 = np.full(n, sigma_next**2)
        for t in range(h):
            out[:, t] = np.sqrt(s2) * eps[:, t]
            s2 = self.omega + self.alpha * out[:, t] ** 2 + self.beta * s2
        return out


def fit_garch11(x: np.ndarray, scale: float = 100.0) -> Garch11:
    """Zero-mean GARCH(1,1) with normal innovations, fitted by `arch` (a dependency, not a
    reimplementation). The vol filter and simulation above use the fitted parameters."""
    from arch import arch_model

    am = arch_model(np.asarray(x, dtype=float) * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    return Garch11(float(p["omega"]) / scale**2, float(p["alpha[1]"]), float(p["beta[1]"]))


def fhs_var(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99, horizon_days: int = 1,
            n_scenarios: int = 20_000, seed: int = 0, marks: list[LegMark] | None = None,
            models: dict[str, Garch11] | None = None) -> tuple[VaRResult, dict[str, Garch11]]:
    """Barone-Adesi, Giannopoulos & Vosper (1999): fit GARCH(1,1) to each factor, standardise,
    resample residual *dates* (so the spot/vol dependence is kept), rescale by the forecast
    vol along h-day paths, fully revalue. Returns the result and the fitted models so a
    rolling backtest can refit on a schedule and filter daily in between."""
    spot_cols, vol_cols = factor_names(book)
    ms = _marks(book, market, marks)
    cols = spot_cols + vol_cols
    if not ms:
        return VaRResult("fhs-garch", confidence, horizon_days, 0.0, 0.0, 0, ("EMPTY",), np.zeros(0)), models or {}
    _check_history(history, cols)
    fitted = dict(models or {})
    z_by_col, sig_next = {}, {}
    for c in cols:
        x = history[c].to_numpy(dtype=float)
        if c not in fitted:
            fitted[c] = fit_garch11(x)
        sig, s_next = fitted[c].filter(x)
        z_by_col[c], sig_next[c] = x / sig, s_next
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(history), size=(n_scenarios, horizon_days))
    shocks = {c: fitted[c].simulate(sig_next[c], z_by_col[c][idx]).sum(axis=1) for c in cols}
    us = book.underlyings()
    pnl = _reval(ms, market, {u: shocks[u + ":spot"] for u in us}, {u: shocks[u + ":vol"] for u in us}, horizon_days)
    v, e = var_es_from_pnl(pnl, confidence)
    return VaRResult("fhs-garch", confidence, horizon_days, v, e, n_scenarios, (), pnl), fitted


def all_methods(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99, horizon_days: int = 1,
                n_scenarios: int = 20_000, seed: int = 0, marks: list[LegMark] | None = None) -> list[VaRResult]:
    ms = _marks(book, market, marks)
    fhs, _ = fhs_var(book, market, history, confidence, horizon_days, n_scenarios, seed, ms)
    return [historical_var(book, market, history, confidence, horizon_days, ms),
            parametric_var(book, market, history, confidence, horizon_days, "delta-gamma-vega", ms, seed=seed),
            monte_carlo_var(book, market, history, confidence, horizon_days, n_scenarios, seed, ms), fhs]
