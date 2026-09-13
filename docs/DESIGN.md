# riskkit — design

## One rule

A risk number never travels without its state. `assess()` returns a
`RiskAssessment(state, var, flags, marks, market)` and the report prints `state:` before any
number: an empty book is `EMPTY`, not `OK` with zero risk; a locked market can widen the
scenario set and freeze VaR but never lower it; a stale quote is marked at its last valid
mid and named. `market` is the market the marks and `var` were computed on (the last valid
mid under a lock), and the report's governing number is `var` — it is the `historical` row,
printed again by name, and the other methods run on that same market. The contracts are
tests, listed under "Edge cases" below.

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
  (deskboard's convention). The book value is Σ value × signed quantity × multiplier. Scenario
  P&L is measured from the model price at each leg's marked (spot, iv, T) — not from the mid —
  so `revalue` at zero shock is exactly zero for every leg, and a mid with no implied vol
  (NO_IV, vol falls back to the underlying's level) changes the book value but not the risk
  numbers (tested to 1e-9; before this rule the quote-vs-model gap entered every scenario as a
  constant).
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
(∫ tⁿφ over (−∞, z] has closed forms), tested against numerical quadrature to 1e-6; for
s = k = 0 it reduces to φ(z_p)/p.

CF is a four-cumulant expansion and is only accurate for moderate departures from normality
(Jaschke 2002). It is used when |s| ≤ 1, k ≤ 3 **and** the polynomial is monotone over the
loss tail it is read on — 3a z² + 2b z + c ≥ 0 on [−6, z_p] (Maillard 2012's validity
condition restricted to that interval: with k ≈ 0 and any s ≠ 0 the cubic always turns over
somewhere, but on most days of the demo's rolling backtest that is at z ≈ +4 and −9, where the
Gaussian weight is nil, and CF is within 1 % of the exact numbers there); otherwise
`parametric_var` returns the exact distribution of the quadratic form — θh + b'z + ½ z'Lz over
200,000 seeded draws of z ~ N(0, S), homogeneous in the book because the draws are fixed —
with the notes `cornish_fisher_nonmonotone` and / or `cornish_fisher_out_of_domain` plus
`quadratic_form_simulated`; `quantile="exact"` always does this and `cornish_fisher_is_monotone`
remains the global test. Measured against 400,000 draws on the demo book, CF is within 2 % at
h = 1 (s = −0.4, k = 0.7) and 10–14 % high at h = 10 (s = −1.6, k = 5.4), which sets the
cumulant bounds; a long straddle's delta-gamma P&L (½σ²χ²₁-shaped, s = 2.8, k = 12) is
non-monotone at z_p itself and its raw CF quantile is a *negative* VaR with ES < VaR, while
the exact number is the theta floor (+196) with ES ≥ VaR (both tested). A monotone polynomial
is not a sufficient condition: a long straddle at h = 60 is globally monotone with s = 2.3,
k = 9.2 and its raw CF VaR is 61 % below the exact one, which is why the cumulant bounds
exist.

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
| Christoffersen conditional coverage | correct rate and i.i.d. | LR_cc = LR_pof + LR_ind | χ²(2) asymptotically; the p-value is simulated from i.i.d. Bernoulli(p) exception series of the same length by default (`n_sim=0` for χ²) | same |
| Basel traffic light | accurate 99 % model over 250 days | number of exceptions | Binomial(250, 0.01) cumulative probability: green < 95 %, yellow < 99.99 %, red otherwise → 0–4 / 5–9 / 10+ | Basel Committee on Banking Supervision (1996), "Supervisory framework for the use of 'backtesting' in conjunction with the internal models approach" |
| Acerbi-Szekely Z2 | the model's predictive distribution is the true one (so VaR and ES are right) | Z2 = Σ X_t I_t / (T p ES_t) + 1, E[Z2] = 0 | simulated from the model's distribution, one-sided: `scenario_sampler` resamples the method's own scenario P&L scaled to VaR_t (what the report uses for historical, Monte Carlo and FHS); the default Gaussian with σ_t = VaR_t / z_c is exact only for a Gaussian model (the parametric line) | Acerbi, C. & Szekely, B. (2014), "Backtesting Expected Shortfall", *Risk* December |
| Engle-Manganelli DQ | Hit_t = I_t − p is orthogonal to its past and to VaR_t | DQ = β̂'X'Xβ̂ / (p(1−p)) from the regression of Hit on a constant, 4 lagged hits and VaR_t (the VaR column is dropped when constant: it is the intercept) | χ²(df), df = number of regressors (5 or 6), asymptotically; the p-value is simulated from i.i.d. Bernoulli(p) hits against the same VaR series by default | Engle, R. & Manganelli, S. (2004), "CAViaR", *J. Business & Economic Statistics* 22(4) |

Convention: LR_pof inside LR_cc is computed over all T indicator observations (Christoffersen 1998), not the T−1 transitions used by LR_ind; a reference that uses T−1 for both differs in the second decimal, by ~0.03 (11.679 vs 11.707 for counts 240/4/4/2; tested).

**Known size and power.** The asymptotic Kupiec rule is size-distorted at 250 days and 99 %:
it rejects x = 0 (LR = 5.03 > 3.84) and x ≥ 7, and P(x = 0) alone is 0.081, so the exact size
is 0.095. `kupiec_size(n, p)` and `kupiec_power(n, p_nominal, p_true)` sum the binomial mass
over that rejection set: size 0.095 at 250 days and 0.055 at 1,000; power against a VaR that
is really the 97 % quantile 0.625 at 250 days and 0.933 at 500. The tests check 400 simulated
samples against them: within 3 points at 250 days (size and power); at 1,000 days a simulated
size of 5 % ± 3; at 500 days a simulated power > 0.8. A "5 % size, 80 % power" story at 250
days is not available from this test and the code does not pretend otherwise — the demo
report's own backtest shows it: zero exceptions in a calm year is a Kupiec rejection
(p = 0.025) and a green traffic light on the same row.

The χ² rules for the other two tests are also off at the report's own length, in opposite
directions, so their p-values are simulated under the exact null (i.i.d. Bernoulli(p)
exceptions, or hits against the same VaR series), the same device the ES test already used.
Christoffersen CC with χ²(2) at 250 days / 99 % has size 0.5 % and power 28 % against a
Markov chain with P(exception | exception) = 0.3; the simulated p-value has size 2.3 % and
power 35 % at 250 days and > 90 % at 2,000 (measured on 400 samples; nominal 5 % is not
reached because the null distribution is a few atoms). Engle-Manganelli DQ with χ² rejects a
correct model 16 % of the time at 500 days and 8–11 % at 250–2,000 (measured on 4,000
samples per length); the simulated p-value has size 3.8–4.5 % at 250 and 500 days and power
95 % against the same chain. The traffic light is not a size-α test but its exact
probabilities are stated: P(not green | accurate) = P(x ≥ 5) = 0.108, and 0.872 against a
3 % rate. All of these are in tests/test_backtest.py.

For the ES test, "an ES 30 % too small with the VaR right" is impossible (ES ≥ VaR caps the
shortfall at 12.7 % for a Gaussian at 99 %). The alternatives tested are the ones that exist:
a model whose whole distribution is 30 % too thin (VaR and ES both 0.7×) — rejected in
≥ 90 % of 250-day samples with Z2 ≈ −4.3 — and a heavier tail than the model's (truth
Student-t(3) scaled to the same 99 % VaR, so the Gaussian ES is 26 % too small) — rejected
in ≥ 70 % of 10,000-day samples. Z2 shifts by about −0.3 in that second case against a null
standard deviation of ~10/√T, which is why 250 days cannot see it; the sample lengths in the
tests are what the test's own power says they must be. The p-value is exact only when the
model's predictive distribution is the sampler's: with the default Gaussian sampler a
*correct* Student-t(4) model (ES/VaR 1.39 against the Gaussian 1.15) is rejected 10 % of the
time at 250 days and more at longer T, because E[Z2] under that null is 1 − ES_gauss/ES_model
> 0. `scenario_sampler(pnl_scenarios, var_t, var_ref)` resamples a method's own scenario set
scaled to each day's VaR, which brings the size back to 3 % on the same data (tested), and
the report builds one from each simulated method's `VaRResult.pnl`; only the parametric line,
which has no scenario set, keeps the Gaussian null.

The rolling backtest (`backtest.rolling_forecasts`) holds the book at constant maturity and
static quantities and revalues it under each realised factor move, so it backtests the risk
models, not the trading.

## Edge cases — the contracts

| Situation | Contract | Test |
|---|---|---|
| Empty book | every metric 0; `state = "EMPTY"`, never `"OK"`; flag `EMPTY` with the sentence "zero risk is a state to explain, not a result" | `test_empty_book_reports_empty_not_ok` |
| Underlying locked (limit-down or limit-up) or its print stale | mark at the last VALID mid; append the lock move continuing ×1, ×2, ×3 (vol factor at its historical beta) to the scenario set; VaR and ES may not fall below the last unlocked values — if the recomputation is lower they are frozen there with flag `VAR_FROZEN` and its reason; `state = "LIMIT_DOWN"` for a negative lock move, `"LIMIT_UP"` for a positive one; the assessment's `market` carries the last valid mid and the report's historical row and header use it | `test_limit_down_does_not_reduce_var`, `test_limit_down_freeze_triggers_when_recomputation_is_lower`, `test_limit_up_lock_reports_limit_up`, `test_report_under_a_lock_prints_the_governing_number_and_the_marked_spot` |
| Stale quote on a leg | the latest *valid* (0 ≤ bid ≤ ask, ask > 0, finite) quote is used; if older than `stale_after` (default 15 min) the leg is flagged `STALE_QUOTE` with its age and the mid used; a mid with no implied vol is flagged `NO_IV` and the vol falls back to the underlying's level; `state = "STALE"` | `test_stale_quote_marks_to_last_valid_mid_and_flags` |

Precedence: EMPTY > LIMIT_DOWN / LIMIT_UP > STALE > OK. The screening answer, then: a fall in VaR while
the market is locked is not information; the report says LIMIT_DOWN with a frozen number and
the reason, and an empty book says EMPTY, so the question a desk asks — "where did the
positions go, and what does the lock cost if it continues?" — is on the page rather than
hidden behind a green OK.

## Stress and reverse stress

`stress.ladder` is a spot × vol P&L grid with the (0, 0) cell exactly 0. Named scenarios are
data (`Scenario(name, spot_move, vol_move, days, source)`), close-to-close index moves from
public sources listed in the module docstring, applied as a log spot move and an ATM vol
change with VIX points standing in for 30-day ATM vol points; `days` is the calendar length
of the window (31 for Oct-2008, 33 for Mar-2020), the unit `revalue` moves T by. Reverse
stress brackets outward from zero on a 512-point grid (0.1 % of spot over the 50 % range,
0.2 vol points over 100) and then runs Brent on the first crossing, so on a non-monotone P&L
(a short strangle loses both ways) it returns the *nearest* breach; the found shock
reproduces the limit loss to 1e-6 dollars and a 1.5× shock loses more (tested). A loss region
narrower than one grid step can still be stepped over — the resolution is reported in
`ReverseStress.grid_step` and `n_grid` is an argument; the test with a pinned put fly shows
a 64-point grid missing a 0.3 %-wide region at −0.5 % and reporting −5.8 %. The default
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
  vanna / volga cross terms in the parametric quadratic form (the cumulant machinery takes
  them as off-diagonal entries of L; they close most of the 24 % parametric-vs-Monte Carlo
  gap on the demo book), a simulated-null p-value for the independence test.
- **v0.4** — inter-commodity spread credits and the table-driven delivery charge in the
  SPAN engine (both parsed today, both zero with a note); live marks from the deskboard bus;
  a `pricers` surface pricer as a drop-in (the pricing API is deskboard's `price`, `greeks`,
  `implied_vol` plus `price_vec`).

The SPAN engine shipped in v0.3 (next section). SPAN 2 products get no approximation:
`span2_target()` returns the published margin and the model label, nothing else.

## SPAN (v0.3)

**Files.** `span_files.py` reads CME's expanded-unpacked `.pa2` (or its `.zip`) as a stream
and keeps one combined commodity: Type 0 header; P (product, decimal locators, contract
value factor); 2 (combined-commodity membership); S (scanning method); 3 (intra tiers);
E / C (series and tier intra-commodity spreads with priority and charge rate); 4 (delivery
method and short-option minimum); B (scan range, vol scan, extreme multiplier and cover,
delta scaling); 81/82 (the 16-value risk array, composite delta, implied vol, settlement,
current delta); 6 (inter-commodity spreads, parsed and counted, not applied). Column
offsets live in one `LAYOUT` table. Facts pinned on the real file: values are a 5-digit
magnitude followed by the sign (`05333-`), loss-positive for a long; the 81 line carries
scenarios 1–9 from column 55, the 82 line 10–16 then composite delta (4 decimals),
implied vol (6 decimals), settlement (locator from the P record); the GC September 2025
array is (0, 0, −5333, −5333, 5333, 5333, −10667, −10667, 10667, 10667, −16000, −16000,
16000, 16000, −15840, 15840) and the OG 3700 December put has delta −0.4752, vol 0.152551,
settlement 107.40 (each an exact equality in `test_span_private.py`).

Four things the layout page does not say, found by an independent re-read of the whole
2025-09-12 file and each now a test on hand-typed lines: (1) a settlement field of all
nines (`9999999`, with the same in the 81 high-precision field) means *no price* —
`settlement_price` is NaN and `option_values` leaves the contract out of the net option
value with a note, instead of pricing it at 9,999.999 × the contract value factor; (2) when
the 81 high-precision flag (col 123) is `Y` the regular field is zero and the price is read
from the 14-digit field at cols 109–122, otherwise the two fields agree; (3) the P record's
alignment codes (cols 40–41) are `C` = points and 32nds (CBT Treasury futures), `K` =
points and 64ths (their options), `0` = decimal with an eighths last digit (CBT grains),
the final digit in every case a truncated eighth (0 1 2 3 5 6 7 8 for 0/8 … 7/8) —
established by put-call parity on the ZB Nov-25, ZN Oct-25 and corn Mar-26 chains, exact
under the decode and off by up to half a point in decimal; Treasury option strikes carry
the same eighths digit with a blank strike code (ZN 112.25 is `1122`), grain strikes are
whole cents; an unknown code or a sub-tick digit of 4 or 9 raises rather than misprices;
(4) the identity of a contract includes the futures and option day/week codes — 604,160
arrays are distinct on it and 39,888 collide without the codes — so `RiskArray.key`,
`SpanData.find` and `Position` carry them and a duplicate identity fails the load rather
than overwriting.

**Engine.** CME's 2019 methodology deck: scenarios (price {0, ±⅓, ±⅔, ±1} × vol {up, down}
= 1–14; ±3× scan at 33 % cover = 15–16); scan risk = max over scenarios of Σ quantity ×
array value; composite delta = weighted average over the seven price points (0.27,
0.217 × 2, 0.11 × 2, 0.037 × 2 — read from the file, not recomputed); intra-commodity
charge by delta consumption, series spreads (Type E) before tier spreads (Type C), each
in priority order; SOM = short options × rate, a floor; requirement = max(scan + intra +
delivery − inter, SOM); total = requirement − net option value (long value − short value,
premium-style options). `compute()` returns every component, the active scenario's
label, and the notes for what was set to zero or left out and why (an option with no
settlement price in the file is named there, not priced).

**Reconciliation.** `riskkit span` prints scan risk for long 1 / short 1 per contract
month against `data/span/published_2025-09-12.csv` (GC 16,000 / 16,000; ES 20,936 /
20,051 — SPAN 2; CL — SPAN 2, value not transcribed) with the absolute and percentage
error as the result. GC is exact. ES and CL are SPAN 2 products whose parameter files are
on the clearing-firm SFTP, not public; `span2_target()` returns the published number and
the model label so a report never presents a legacy array as their margin.

**Fixture policy.** CME's file is not redistributed. `span_synth` writes a synthetic file in
the identical layout (generator and parser written from the same layout pages
independently, so a round-trip proves the column; it carries two day-coded weeklies so the
identity rule is exercised through the loader), the no-settlement sentinel, the
high-precision flag and the alignment codes are tested on hand-typed lines in the same
layout with fictional product codes, and the private GC slice is gitignored, its seven
tests skipped when absent; `docs/validation-private.md` (gitignored) records the real run.

**Not implemented.** Inter-commodity spread credits and the table-driven delivery charge
(both parsed, both zero with a note); combination products; SPAN 2.
