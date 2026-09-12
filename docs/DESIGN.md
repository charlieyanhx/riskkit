# riskkit — design

## One rule

A risk number never travels without its state. `assess()` returns `(state, VaR, flags)` and
the report prints `state:` before any number: an empty book is `EMPTY`, not `OK` with zero
risk; a locked market can widen the scenario set and freeze VaR but never lower it; a stale
quote is marked at its last valid mid and named. The three contracts are tests, listed under
"Edge cases" below.

## Conventions

- **P&L sign.** P&L is value(after) − value(before) in dollars, positive = gain. Every scenario
  engine returns P&L; `var_es_from_pnl` turns it into losses.
- **VaR positive = loss.** VaR at confidence c is the c-quantile of the loss distribution
  (linear interpolation on scenario sets, so at c = 1 it is exactly the worst scenario). ES is
  the tail mean of the worst n(1 − c) scenarios with a fractional weight on the boundary one
  (Acerbi & Tasche 2002), which makes `ES ≥ VaR` an identity for c ≥ 0.5, tested on random
  samples and on every method.
- **Horizon.** h days. Historical simulation uses overlapping h-day sums of the factor moves;
  Monte Carlo draws from h × the daily covariance; FHS simulates h GARCH steps per path; the
  parametric method scales the covariance by h. All four also move time by h calendar days
  through theta (T − h/365). For a linear book with zero drift the parametric VaR scales
  exactly as √h (tested).
- **Full revaluation vs Greeks.** Historical, Monte Carlo and FHS reprice every leg with
  Black-Scholes under each scenario (`pricing.revalue_factors`). The parametric method is the
  one Greek-based approximation: P&L ≈ θh + Σ δ$ x + ½ Σ γ$ x² + Σ ν$ y with x the spot log
  return and y the vol change, δ$ = Δ·S·q·mult, γ$ = Γ·S²·q·mult, ν$ = vega·q·mult per 1.00
  vol. Cross-gamma, vanna and volga are omitted; the exp convexity ½ΔS x² is dropped, as in
  the standard delta-gamma-normal setup. Its `method` name says "parametric" so the report
  reader knows which line is an approximation.
- **Risk factors.** One spot factor per underlying (`<sym>:spot`, daily log returns) and one
  vol factor per underlying (`<sym>:vol`, daily changes of an ATM vol level in vol units;
  0.01 = one vol point). A vol shock is a parallel shift of every implied vol on that
  underlying. STK and FUT legs load on the spot factor only; a FUT is marked at the spot
  factor level (no basis model in v0.1) with its own per-contract multiplier.
- **Marks.** An option leg is valued at Black-Scholes on the underlying's vol level unless a
  quote is supplied, in which case the mid is the value and its implied vol is inverted
  (deskboard's convention). The book value is Σ value × signed quantity × multiplier, and
  `revalue` at zero shock is exactly zero.
- **Factor means are zero**, covariance is the sample covariance (ddof = 1) of the daily
  changes.

## Parametric delta-gamma-vega with Cornish-Fisher

With z ~ N(0, S), S = h·Σ, and P&L = θh + b'z + ½ z'Lz, the cumulants of a Gaussian quadratic
form are (from the cumulant generating function −½ log det(I − tA) + ½ t² c'(I − tA)⁻¹c with
A = S^½ L S^½, c = S^½ b):

    k1 = ½ tr(LS)
    k2 = b'Sb + ½ tr((LS)²)
    k3 = 3 b'SLSb + tr((LS)³)
    k4 = 12 b'SLSLSb + 3 tr((LS)⁴)

`quadratic_form_cumulants` is tested against the χ² identity (b = 0, L = 1: ½σ²χ²₁ has
cumulants σ²/2, σ⁴/2, σ⁶, 3σ⁸) and against the pure-Gaussian case. The Cornish-Fisher
quantile z_cf = z + (z² − 1)s/6 + (z³ − 3z)k/24 − (2z³ − 5z)s²/36 uses skew s = k3/k2^1.5 and
excess kurtosis k = k4/k2². ES is the exact tail mean of the CF polynomial under the Gaussian
(∫ tⁿφ over (−∞, z] has closed forms), tested against numerical quadrature; for s = k = 0 it
reduces to φ(z_p)/p. When the CF polynomial is not monotone (Maillard 2012's validity domain:
3a z² + 2b z + c ≥ 0 for all z) the result carries the note `cornish_fisher_nonmonotone`
rather than a silently wrong quantile.

## Filtered historical simulation

Barone-Adesi, Giannopoulos & Vosper (1999). Each factor gets a zero-mean GARCH(1,1) with
normal innovations fitted by the `arch` package (a dependency, not a reimplementation; the
data is scaled by 100 for the optimiser, omega unscaled afterwards). `Garch11.filter`
recomputes the conditional vol path with the fitted parameters (initialised at the sample
variance), residuals are standardised, and scenario paths resample residual *dates* — the
same date for every factor — so the spot/vol dependence survives, then rescale by the
forecast vol along h simulated GARCH steps. The fit is tested for parameter recovery on
6,000 simulated days. In the rolling backtest the GARCH is refit every 250 days and filtered
daily in between, the usual desk practice.

## Backtests: null, statistic, citation

| Test | H0 | Statistic | Reference distribution | Source |
|---|---|---|---|---|
| Kupiec POF | exception rate = p | LR = −2 ln[(1−p)^(n−x) p^x / ((1−x/n)^(n−x) (x/n)^x)] | χ²(1) | Kupiec, P. (1995), "Techniques for verifying the accuracy of risk measurement models", *J. Derivatives* 3(2) |
| Christoffersen independence | exceptions are i.i.d. (against a first-order Markov chain) | LR_ind from the 2×2 transition counts | χ²(1) | Christoffersen, P. (1998), "Evaluating interval forecasts", *Int. Econ. Review* 39(4) |
| Christoffersen conditional coverage | correct rate and i.i.d. | LR_cc = LR_pof + LR_ind | χ²(2) | same |

Convention: LR_pof inside LR_cc is computed over all T indicator observations (Christoffersen 1998), not the T−1 transitions used by LR_ind; a reference that uses T−1 for both differs in the third decimal (e.g. 11.679 vs 11.707 for counts 240/4/4/2).
| Basel traffic light | accurate 99 % model over 250 days | number of exceptions | Binomial(250, 0.01) cumulative probability: green < 95 %, yellow < 99.99 %, red otherwise → 0–4 / 5–9 / 10+ | Basel Committee on Banking Supervision (1996), "Supervisory framework for the use of 'backtesting' in conjunction with the internal models approach" |
| Acerbi-Szekely Z2 | the model's predictive distribution is the true one (so VaR and ES are right) | Z2 = Σ X_t I_t / (T p ES_t) + 1, E[Z2] = 0 | simulated from the model's distribution (default: Gaussian with σ_t = VaR_t / z_c), one-sided | Acerbi, C. & Szekely, B. (2014), "Backtesting Expected Shortfall", *Risk* December |
| Engle-Manganelli DQ | Hit_t = I_t − p is orthogonal to its past and to VaR_t | DQ = β̂'X'Xβ̂ / (p(1−p)) from the regression of Hit on a constant, 4 lagged hits and VaR_t | χ²(6) | Engle, R. & Manganelli, S. (2004), "CAViaR", *J. Business & Economic Statistics* 22(4) |

**Known size and power.** The asymptotic Kupiec rule is size-distorted at 250 days and 99 %:
it rejects x = 0 (LR = 5.03 > 3.84) and x ≥ 7, and P(x = 0) alone is 0.081, so the exact size
is 0.095. `kupiec_size(n, p)` and `kupiec_power(n, p_nominal, p_true)` sum the binomial mass
over that rejection set, and the tests check that 400 simulated samples land within 3 points
of those exact numbers: size 0.095 at 250 days, 0.055 at 1,000; power against a VaR that is
really the 97 % quantile 0.625 at 250 days and 0.933 at 500. A "5 % size, 80 % power" story
at 250 days is not available from this test and the code does not pretend otherwise — the
demo report's own backtest shows it: zero exceptions in a calm year is a Kupiec rejection
(p = 0.025) and a green traffic light on the same row.

For the ES test, "an ES 30 % too small with the VaR right" is impossible (ES ≥ VaR caps the
shortfall at 12.7 % for a Gaussian at 99 %). The alternatives tested are the ones that exist:
a model whose whole distribution is 30 % too thin (VaR and ES both 0.7×) — rejected in
≥ 90 % of 250-day samples with Z2 ≈ −4.3 — and a heavier tail than the model's (truth
Student-t(3) scaled to the same 99 % VaR, so the Gaussian ES is 26 % too small) — rejected
in ≥ 70 % of 10,000-day samples. Z2 shifts by about −0.3 in that second case against a null
standard deviation of ~10/√T, which is why 250 days cannot see it; the sample lengths in the
tests are what the test's own power says they must be. The p-value is exact only when the
model's predictive distribution is the sampler's; pass your own `sampler` (e.g. the
historical scenario set) for other models.

The rolling backtest (`backtest.rolling_forecasts`) holds the book at constant maturity and
static quantities and revalues it under each realised factor move, so it backtests the risk
models, not the trading.

## Edge cases — the contracts

| Situation | Contract | Test |
|---|---|---|
| Empty book | every metric 0; `state = "EMPTY"`, never `"OK"`; flag `EMPTY` with the sentence "zero risk is a state to explain, not a result" | `test_empty_book_reports_empty_not_ok` |
| Underlying locked (limit-down) or its print stale | mark at the last VALID mid; append the lock move continuing ×1, ×2, ×3 (vol factor at its historical beta) to the scenario set; VaR and ES may not fall below the last unlocked values — if the recomputation is lower they are frozen there with flag `VAR_FROZEN` and its reason; `state = "LIMIT_DOWN"` | `test_limit_down_does_not_reduce_var`, `test_limit_down_freeze_triggers_when_recomputation_is_lower` |
| Stale quote on a leg | the latest *valid* (0 ≤ bid ≤ ask, ask > 0, finite) quote is used; if older than `stale_after` (default 15 min) the leg is flagged `STALE_QUOTE` with its age and the mid used; a mid with no implied vol is flagged `NO_IV` and the vol falls back to the underlying's level; `state = "STALE"` | `test_stale_quote_marks_to_last_valid_mid_and_flags` |

Precedence: EMPTY > LIMIT_DOWN > STALE > OK. The screening answer, then: a fall in VaR while
the market is locked is not information; the report says LIMIT_DOWN with a frozen number and
the reason, and an empty book says EMPTY, so the question a desk asks — "where did the
positions go, and what does the lock cost if it continues?" — is on the page rather than
hidden behind a green OK.

## Stress and reverse stress

`stress.ladder` is a spot × vol P&L grid with the (0, 0) cell exactly 0. Named scenarios are
data (`Scenario(name, spot_move, vol_move, days, source)`), close-to-close index moves from
public sources listed in the module docstring, applied as a log spot move and an ATM vol
change with VIX points standing in for 30-day ATM vol points. Reverse stress brackets
outward from zero on a grid and then runs Brent on the first crossing, so on a non-monotone
P&L (a short strangle loses both ways) it returns the *nearest* breach; the found shock
reproduces the limit loss to 1e-6 dollars and a 1.5× shock loses more (tested). The default
spot→vol response is −x (one vol point per 1 % of spot, rising when spot falls); the
vol→spot response is 0. Both are arguments.

## Synthetic data

`synth.demo_history` is a GJR-GARCH(1,1) spot process with Student-t(6) innovations and an
ATM vol level that reacts to the day's return, so the factor history has clustering and a
negative spot–vol correlation (both tested). Values are rounded to 10 decimals before
writing so the CSV does not depend on the platform's libm; the CI step regenerates it on
Linux and diffs against the committed file. The demo book is four positions on one
underlying: a put spread, a strangle with a stock hedge, a long put, and a short future with
its own multiplier — net slightly long delta, short gamma, short vega, which is why the
limit-down scenarios sit in its loss tail.

## Not yet

- **v0.2** — per-strike vol factors (smile shocks instead of a parallel shift), a futures
  basis factor, multi-underlying demo, Student-t Monte Carlo, an `arch` GJR option for FHS,
  a `sampler` for the ES test built from each method's own scenario set.
- **v0.3 — SPAN margin replication.** Research on public exchange files established that
  only legacy-SPAN products such as GC reconcile exactly with the published parameter files;
  ES and CL are SPAN 2 products whose parameter files are not public, so an exact
  replication cannot be validated for them. v0.3 therefore ships a legacy-SPAN engine tested
  against GC files, and a SPAN 2 approximation clearly labelled as such — not before.
- **v0.4** — live marks from the deskboard bus and a `pricers` surface pricer as a drop-in
  (the pricing API is already deskboard's `price`, `greeks`, `implied_vol`).
