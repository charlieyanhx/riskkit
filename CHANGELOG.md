# Changelog

## 0.3.1 — 2026-09-12
Fixes from an independent numeric and claims review of the SPAN module (the reviewer re-read the whole 2025-09-12 CME file with a second parser); 98 tests (7 skip without the private CME slice). GC still reconciles exactly (the four defects never touch the GC/OG slice).
- `span_files.parse_risk_array_pair` — a settlement field of all nines (`9999999`, CME's no-price sentinel; 24,349 option records in the file) reads as NaN instead of 9,999.999 × the contract value factor: a long 1 CBT ZB Oct-25 80 put was carrying $9,999,999 of long option value and a total requirement of −$9,999,999. `RiskArray.has_settlement`; `span.option_values` leaves such a contract out of the net option value and returns it in a third element, and `compute()` names it in the notes
- `span_files.parse_risk_array_pair` — with the 81 high-precision flag (col 123) `Y` the price is read from the 14-digit field at cols 109–122 (2,548 flag-Y option records had settlement 0 and no option value: a short BTC Sep-25 220,000 put's total was $140,720 instead of $653,570)
- `span_files.aligned_price` and `PriceParams.settlement_alignment` / `strike_alignment` / `strike_format` — the P record's price alignment codes (cols 40–41, not in `LAYOUT` before) decode CBT Treasuries in points-and-32nds (`C`, futures) and 64ths (`K`, options) and CBT grains in eighths of a cent (`0`), each with a truncated-eighths last digit; Treasury option strikes carry the same digit (ZN 112.25 was 112.2). ZB Dec-25 future 117.13 → 117.40625; the Nov-25 114 call $3,540 → $3,843.75. Pinned by put-call parity on the file's chains; an unknown code or an impossible sub-tick digit raises
- `RiskArray.key`, `SpanData.find`, `span.Position` — the futures and option day/week codes (81 cols 36–37 / 45–46) are part of a contract's identity; `SpanData` raises on a duplicate identity instead of keeping the last line (CBT ECYM: 50 of 236 event-contract arrays were unreachable; 39,888 of the file's 604,160 arrays collide without the codes). `outright_table` carries `contract_day`
- `span.SpanResult.active_scenario_label`; the unused composite-delta weights constant removed (the deck's weights are cited in DESIGN)
- `parse_risk_array_pair(l81, l82, price_params, risk_scale)` takes the family's `PriceParams` instead of the two locators
- synthetic fixture gains two day-coded weekly calls; tests on hand-typed CME-layout lines for the sentinel, the flag and the alignment codes; real-slice tests now pin the OG put's delta / vol exactly, its long and short total requirement (−3,143 / 23,618), the header, Type 4 and Type B fields
- README / DESIGN / CHANGELOG — the offset claim narrowed to the columns the real-slice tests pin; Data and privacy states what is synthetic, public (the hand-transcribed margin CSV) and private-and-gitignored; What is where, the module table, Run it and the Roadmap list the SPAN modules, commands and test count; the stale "v0.3 not yet / SPAN 2 approximation" bullet removed from DESIGN; CL noted as untranscribed; the multi-commodity upper bound stated with its delivery-method condition

## 0.3.0 — 2026-09-12
- `span_files`: streaming parser for CME expanded-unpacked `.pa2` files (Type 0/P/2/S/3/E/C/4/B/81/82/6), offsets pinned on the real 2025-09-12 file; `extract_slice`; synthetic fixture generator `span_synth`.
- `span`: legacy SPAN engine (scan risk, composite delta, series and tier spread charges, SOM, NOV, every component reported); published-margin data `data/span/published_2025-09-12.csv` (CME's public margins, hand-transcribed with sources); reconciliation CLI `riskkit span` — GC exact (16,000 = 16,000 every month), ES/CL labelled SPAN 2; `riskkit span-fixture` (regenerate the synthetic file) and `riskkit span-slice` (cut one combined commodity out of a full file).
- data policy: CME's file is not redistributed; `tests/fixtures/private_*` and `docs/validation-private.md` are gitignored and the real-slice tests skip without them. 89 tests (5 skipped without the private slice).
- fix: implied vol in the 82 record has 6 implied decimals (was read with 8).

## 0.1.1 — 2026-09-12
Fixes from an independent numeric and claims review; 67 tests.
- `var.parametric_var` — the Cornish-Fisher quantile is used only inside its validity domain (monotone, |skew| ≤ 1, excess kurtosis ≤ 3); outside it, or with `quantile="exact"`, the exact quadratic form is simulated (seeded, homogeneous) and the result carries `cornish_fisher_nonmonotone` / `cornish_fisher_out_of_domain` + `quadratic_form_simulated`. A long straddle's delta-gamma VaR was −260 (a gain) with ES < VaR; it is now +196 with ES ≥ VaR. New `parametric_moments`, `cornish_fisher_var_es`, `quadratic_form_var_es`, `cornish_fisher_in_domain`
- `pricing.revalue_factors` — scenario P&L is measured from the model price at each leg's marked inputs, so zero shock is exactly zero for every leg; a NO_IV quote mid no longer enters every scenario as a constant (it moved the demo VaR from 6,277 to 104,143)
- `report` — under a lock the VaR table's historical row is the assessment's governing number (widened, frozen), the other methods run on the marked market, the header prints the last valid mid, and the governing number is printed by name; `RiskAssessment` gains `market`
- `backtest` — Christoffersen CC and Engle-Manganelli DQ p-values simulated under the exact i.i.d. null by default (`n_sim=0` for χ²; asymptotic p kept in `details`); the DQ regressor matrix drops the constant-VaR column (df 5, not 6); `scenario_sampler` builds the Acerbi-Szekely null from a method's own scenario set and the report uses it for historical / Monte Carlo / FHS
- `edge_cases` — a positive lock move reports state `LIMIT_UP` (was `LIMIT_DOWN` with a `LIMIT_UP` flag)
- `stress` — reverse-stress bracketing on a 512-point grid (was 64) with the resolution in `ReverseStress.grid_step`; `HISTORICAL_SCENARIOS` days are calendar days (Oct-2008: 31, Mar-2020: 33)
- `riskkit report --seed`
- README / DESIGN — the DESIGN backtest table repaired; seed movement, ratios, the schema and pricer-interface claims, the report's table shape, and the tolerance of the CF tail-mean test stated as measured

## 0.1.0 — 2026-09-11
- `positions` — deskboard's leg / position schema (OPT / STK, `hedge_shares`) plus a FUT leg with an explicit multiplier, risk-factor mapping, marks, dollar Greeks
- `pricing` — Black-Scholes price / Greeks / implied vol with deskboard's API; array pricing; full revaluation under (spot, vol, time) scenarios
- `var` — historical simulation, parametric delta-gamma-vega with a Cornish-Fisher quantile and exact CF tail-mean ES, Gaussian Monte Carlo, filtered historical simulation with GARCH(1,1) via `arch`; one `VaRResult` shape
- `backtest` — Kupiec POF, Christoffersen independence / conditional coverage, Basel traffic light, Acerbi-Szekely Z2 with simulated p-value, Engle-Manganelli DQ; exact `kupiec_size` / `kupiec_power`; rolling out-of-sample forecaster
- `stress` — spot × vol ladder, named historical scenarios with sources, reverse stress on either axis
- `edge_cases` — EMPTY / LIMIT_DOWN / STALE contracts with named tests
- `report`, `cli` — markdown report; `riskkit demo`, `riskkit report`
- seeded synthetic demo book and five-year GJR-GARCH factor history, regenerated byte-identically
- CI (3.11 / 3.12, ruff, pytest, demo regeneration diff), MIT licence; 55 tests
