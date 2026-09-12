# riskkit

[![ci](https://github.com/charlieyanhx/riskkit/actions/workflows/ci.yml/badge.svg)](https://github.com/charlieyanhx/riskkit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Portfolio risk for a futures-and-options book, with the parts open source leaves out made
first-class: a VaR **and ES** backtest suite whose size and power are computed rather than
assumed, delta-gamma-vega parametric VaR with a Cornish-Fisher quantile, historical / Monte
Carlo / filtered-historical (GARCH via `arch`) VaR and ES with full revaluation, stress and
**reverse** stress, and explicit limit-down / empty-book / stale-quote semantics. Every
identity is a test; every backtest's size and power is measured on a known distribution, and
the Kupiec rule's exact size at 250 days is a function, not an assumption. The one design
rule: a risk number never travels without its state — an empty book is `EMPTY`, not
`OK`; a locked market can widen the scenario set and freeze VaR but never lower it; a stale
quote is marked at its last valid mid and named.

**The screening question.** *The market is limit-down, the book is empty, VaR says risk
fell — what do you do?* The code's answer, in `edge_cases.assess`:

```python
a = assess(book, market, history, lock=Lock("SYN", last_valid_spot=500.0, lock_move=log(0.93), since=t0),
           previous=last_unlocked_var)
a.state   # "LIMIT_DOWN" — or "EMPTY" if the book has no legs, never "OK"
a.var     # >= last_unlocked_var: marked at the last valid mid, scenario set widened with the
          #    lock move continuing x1, x2, x3, and frozen at the unlocked value if the
          #    recomputation came in lower — with a VAR_FROZEN flag saying so
a.flags   # LIMIT_DOWN [SYN]: locked since ... at -7.26% from the last valid mid 500.0000; ...
```

You do not believe the drop. A locked print is not a two-sided market and contributes no
information; a book that is suddenly empty is a state to explain (positions feed down?
flattened?), not a result to file. The report says `state: LIMIT_DOWN` or `state: EMPTY`
before any number, and the three contracts are named tests:
`test_limit_down_does_not_reduce_var`, `test_empty_book_reports_empty_not_ok`,
`test_stale_quote_marks_to_last_valid_mid_and_flags`.

`riskkit report` on the committed demo book (a put spread, a strangle with a stock hedge, a
long put and a short future on one synthetic underlying; Apple M1 laptop, macOS, Python
3.13; the Monte Carlo and FHS lines move 2–4 % across seeds, nothing else varies) prints this
headline:

```
state: OK
asof 2026-06-12 20:00 UTC · SYN spot 500.00 · ATM vol 12.3% · 4 positions · 7 legs · book value $-41,221
Units: US dollars. VaR and ES are losses (positive numbers); P&L is positive = gain. Confidence 99%, horizon 1 day(s)
```

| method | VaR | ES | n scenarios | exceptions in the last 250 days (expected 2.5) | Kupiec p | Christoffersen CC p | Basel zone | Acerbi-Szekely Z2 (p) | DQ p |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| historical | 6,277 | 9,002 | 1259 | 0 | 0.025 | 0.081 | green | +1.00 (1.000) | 0.870 |
| parametric-delta-gamma-vega | 2,137 | 2,567 | — | 1 | 0.278 | 0.553 | green | +0.56 (0.748) | 0.985 |
| monte-carlo | 2,643 | 3,275 | 20000 | 1 | 0.278 | 0.553 | green | +0.53 (0.736) | 0.983 |
| fhs-garch | 5,038 | 6,729 | 20000 | 1 | 0.278 | 0.553 | green | +0.61 (0.828) | 0.972 |

Read the backtest row for `historical` as a risk manager would: zero exceptions in a calm
year is a Kupiec *rejection* (at 250 days the χ² rule rejects x = 0, LR = 5.03) and a green
traffic light on the same row. That is the size distortion of the asymptotic test, and
`kupiec_size(250, 0.01)` prints it: 0.095, not 0.05. The historical and FHS lines are 2–3×
the Gaussian ones because the five-year history has fat tails and clustering and the last
year was calm; the parametric and Monte Carlo lines agree with each other because both are
Gaussian in the factors.

## Run it

```bash
pip install -e ".[dev]"
pytest -q          # 55 tests: pricing oracles, book schema, VaR/ES identities (closed form, worst
                   #   scenario, ES >= VaR, homogeneity, sqrt-horizon, CF tail mean), backtest size
                   #   and power on known distributions, stress and reverse stress, the three
                   #   edge-case contracts, report and CLI; ~5 s on a laptop
riskkit demo       # regenerate data/demo (book.json + 5-year factor history, seeded, byte-identical)
riskkit report     # the markdown report above: VaR/ES by method, 250-day backtest, stress ladder,
                   #   named scenarios, reverse stress, flags (~4 s)
```

Your own book: a JSON file in deskboard's blotter shape (`[{pos_id, sleeve, legs, hedge_shares}]`
or `{"positions": [...]}`), a `Market(asof, spot, vol)` and a factor-history DataFrame with
`<sym>:spot` (daily log returns) and `<sym>:vol` (daily ATM-vol changes) columns:

```python
from riskkit import Book, Market, all_methods, suite, assess
book = Book.load("book.json")
for r in all_methods(book, market, history, confidence=0.99, horizon_days=1):
    print(r.method, r.var, r.es, r.n_scenarios)     # VaRResult, losses in dollars
```

## What it computes

| Module | What |
|---|---|
| `positions` | deskboard's leg / position schema verbatim (`OPT` / `STK` / `FUT`, `hedge_shares`), risk-factor mapping (spot per symbol, vol per symbol for options), marks, dollar Greeks |
| `pricing` | Black-Scholes price / Greeks / implied vol with deskboard's API (the `pricers` sibling is a drop-in), array pricing, full revaluation under (spot shock, vol shock, time step) |
| `var` | historical simulation, parametric delta-gamma-vega with Cornish-Fisher (exact cumulants of the Gaussian quadratic form, exact CF tail mean for ES), Gaussian Monte Carlo from the fitted factor covariance, FHS with GARCH(1,1) per factor via `arch` and date-resampled residuals; each a `VaRResult(method, confidence, horizon, var, es, n_scenarios)` |
| `backtest` | Kupiec POF, Christoffersen independence and conditional coverage, Basel traffic light, Acerbi-Szekely Z2 with a simulated p-value, Engle-Manganelli DQ; `kupiec_size` / `kupiec_power` exact; a rolling out-of-sample forecaster |
| `stress` | spot × vol ladder, named historical scenarios as data with sources, reverse stress on either axis by bracketing + Brent |
| `edge_cases` | `Quote`, `Lock`, `assess()` — the three contracts |
| `report`, `cli` | the markdown report; `riskkit demo`, `riskkit report` |

## Validation

What the tests establish (identities exact to 1e-9 or 1e-10 unless a tolerance is stated):

| Claim | Test |
|---|---|
| Parametric delta-only VaR and ES on a stock equal the Gaussian closed forms q·S·σ̂·z_c·√h and q·S·σ̂·φ(z_p)/p·√h | `test_parametric_delta_only_equals_closed_form_for_a_single_stock`, `..._scales_with_sqrt_horizon` |
| Historical VaR at 100 % confidence is the worst scenario; ES = VaR there | `test_historical_var_at_100_percent_confidence_is_the_worst_scenario` |
| ES ≥ VaR for every method, horizon 1 and 10, and on 200 random samples for c ∈ [0.5, 1] | `test_es_ge_var_for_every_method`, `test_var_es_from_pnl_worked_example_and_es_ge_var_property` |
| VaR and ES are positively homogeneous in position size, all four methods | `test_var_is_positively_homogeneous_in_position_size` |
| Quadratic-form cumulants match the χ² identity; CF tail mean matches quadrature | `test_quadratic_form_cumulants_chi_square_identity`, `test_cornish_fisher_tail_mean_equals_numerical_integral` |
| `arch` GARCH(1,1) recovers α = 0.08, β = 0.90 within 0.03 on 6,000 days | `test_fhs_garch_fit_recovers_known_parameters` |
| Kupiec: 400 simulated 250-day samples reject at the exact size 0.095 ± 0.03; at 1,000 days 5 % ± 3; power against a 97 % quantile sold as 99 % is 0.625 at 250 days (simulated ± 0.03) and > 0.8 at 500 | `test_kupiec_*` |
| Christoffersen independence rejects Markov-clustered exceptions > 90 %, accepts i.i.d. < 10 % | `test_christoffersen_independence_rejects_clustered_and_accepts_iid` |
| Traffic light: 0–4 green, 5–9 yellow, 10+ red at 250 / 99 % | `test_traffic_light_zone_boundaries_at_250_days_99_percent` |
| Acerbi-Szekely accepts the true ES (≤ 15 % rejections), rejects a model 30 % too thin (≥ 90 % at 250 days) and a t(3) tail with the Gaussian VaR right (≥ 70 % at 10,000 days) | `test_acerbi_szekely_*` |
| DQ size ≤ 12 %, power > 90 % on clustered exceptions | `test_engle_manganelli_dq_size_and_power` |
| Reverse stress reproduces the limit loss to 1e-6 $ and a 1.5× shock loses more; a non-monotone P&L yields the first breach | `test_reverse_stress_*` |
| The three edge-case contracts | `test_limit_down_does_not_reduce_var`, `test_empty_book_reports_empty_not_ok`, `test_stale_quote_marks_to_last_valid_mid_and_flags` |
| The report states units, confidence, horizon and n; `riskkit demo` is byte-identical | `test_report_states_units_confidence_horizon_and_n`, `test_demo_regenerates_byte_identically` |

## Design rules

Tested:

- **Sign and units** — P&L positive = gain, VaR and ES positive = loss, dollars; the report prints them with the confidence, horizon and scenario count on every table.
- **Full revaluation** for historical, Monte Carlo and FHS; the parametric line is the one Greek approximation and is named so.
- **Known size and power** — the Kupiec rejection set and its exact binomial size/power are functions, and the simulations are checked against them, including the 0.095 size at 250 days that the χ² approximation hides.
- **Edge-case contracts** — EMPTY not OK; a lock never lowers VaR; stale quotes are marked at the last valid mid and flagged.
- **Determinism** — seeded synthetic data regenerates byte-identically; Monte Carlo and FHS with the same seed give the same scenarios, which is what makes the homogeneity test exact.

By construction (not a test): the pricer is deskboard's three-function interface, so a
surface pricer replaces it without touching the engines; named scenarios are data with a
source string, not code; the factor model is one spot and one vol factor per underlying
(a parallel smile shift), which is what v0.1's demo needs and what v0.2 refines; nothing
here reads live market data or submits orders.

## What is where

```
src/riskkit/
  positions.py     book schema (deskboard's), risk-factor mapping, Market, marks, dollar Greeks
  pricing.py       BSM price / Greeks / implied vol (deskboard API), price_vec, revalue
  synth.py         seeded demo book + GJR-GARCH factor history; write_demo / load_demo
  var.py           VaRResult; historical, parametric-CF, Monte Carlo, FHS (arch); var_es_from_pnl
  backtest.py      Kupiec, Christoffersen, traffic light, Acerbi-Szekely Z2, DQ; exact size/power; rolling forecasts
  stress.py        ladder, HISTORICAL_SCENARIOS, reverse_stress_spot / _vol
  edge_cases.py    Quote, Lock, Flag, assess(): EMPTY / LIMIT_DOWN / STALE contracts
  report.py        build() + render() markdown
  cli.py           demo · report
data/demo/         book.json, history.csv (regenerated by `riskkit demo`)
docs/DESIGN.md     conventions, each backtest's null/statistic/citation, the contracts, the SPAN plan
tests/             55 tests
```

## Roadmap

v0.2: per-strike vol factors (smile shocks), a futures basis factor, a multi-underlying
demo, Student-t Monte Carlo, GJR for FHS, an ES-test sampler built from each method's own
scenario set. **v0.3: SPAN margin replication** — research on public exchange files
established that only legacy-SPAN products such as GC reconcile exactly with the published
parameter files, while ES and CL are SPAN 2 products whose parameter files are not public;
v0.3 ships a legacy-SPAN engine validated against GC files and a labelled SPAN 2
approximation, and nothing is built until then. v0.4: live marks from the deskboard bus,
`pricers` as the surface pricer.

## Data and privacy

Everything in this repo is synthetic: the demo book is fictional and the factor history is
generated by `riskkit demo` from a seeded GJR-GARCH process. The named stress scenarios use
public index and VIX closes cited in `stress.py`. The author's live positions never enter
the repo; a live book runs through the same blotter schema from a local, gitignored file.

## Companion repos

[tcakit](https://github.com/charlieyanhx/tcakit) — transaction cost analysis and market
impact ·
[quant-research-agent](https://github.com/charlieyanhx/quant-research-agent) — a backtest
review agent and the evals that measure it ·
[deskboard](https://github.com/charlieyanhx/deskboard) — real-time options risk / P&L
dashboard whose book schema and pricer API this repo shares ·
[pricers](https://github.com/charlieyanhx/pricers) — option pricers, a drop-in for
`riskkit.pricing`.

MIT © Hanxiong (Charlie) Yan
