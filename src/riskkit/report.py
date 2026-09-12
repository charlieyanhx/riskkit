"""Markdown risk report. States units (US dollars), confidence, horizon and scenario counts
inline; VaR and ES are losses (positive), P&L is positive = gain. `build()` computes every
section for a book, market and factor history; `render()` only formats.

The governing number is `assessment.var` — the historical line computed under the edge-case
contracts (marked at the last valid mid under a lock, scenario set widened, frozen at the
last unlocked value if lower). It is the `historical` row of the VaR table and is printed
again by name under it; the other three methods are computed on the same marked market.
The header prints the spot the marks use (`assessment.market`), not a locked print.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass

import pandas as pd

from . import backtest as bt
from . import stress as st
from . import var as V
from .edge_cases import RiskAssessment, assess
from .positions import Book, Market, book_value


@dataclass(frozen=True)
class Report:
    book: Book
    market: Market
    confidence: float
    horizon_days: int
    results: list[V.VaRResult]
    forecasts: pd.DataFrame | None
    tests: dict[str, list[bt.TestResult]]
    ladder: pd.DataFrame
    scenarios: pd.DataFrame
    reverse: list[st.ReverseStress]
    assessment: RiskAssessment
    loss_limit: float
    window: int


def build(book: Book, market: Market, history: pd.DataFrame, confidence: float = 0.99, horizon_days: int = 1,
          n_scenarios: int = 20_000, n_test: int = 250, window: int = 1000, loss_limit: float = 25_000.0,
          seed: int = 0, run_backtest: bool = True, **assess_kw) -> Report:
    assessment = assess(book, market, history, confidence, horizon_days, **assess_kw)
    market = assessment.market          # the marked market: last valid mid under a lock
    marks = list(assessment.marks)
    results = V.all_methods(book, market, history, confidence, horizon_days, n_scenarios, seed, marks) if marks else []
    results = [assessment.var if r.method == "historical" else r for r in results]   # the governing number
    forecasts, tests = None, {}
    if run_backtest and marks and len(history) >= window + n_test:
        forecasts = bt.rolling_forecasts(book, market, history, confidence, window, n_test, seed=seed)
        for m in bt.METHODS:
            var_t = forecasts[f"var_{m}"].to_numpy()
            tests[m] = bt.suite(forecasts["pnl"].to_numpy(), var_t, forecasts[f"es_{m}"].to_numpy(), confidence,
                                sampler=_own_sampler(results, m, var_t), seed=seed)
    ladder = st.ladder(marks, market) if marks else pd.DataFrame()
    scen = st.historical_scenarios(marks, market) if marks else pd.DataFrame()
    reverse = [st.reverse_stress_spot(marks, market, loss_limit), st.reverse_stress_vol(marks, market, loss_limit)] if marks else []
    return Report(book, market, confidence, horizon_days, results, forecasts, tests, ladder, scen, reverse, assessment, loss_limit, window)


def _own_sampler(results: list[V.VaRResult], method: str, var_t) -> bt.Sampler | None:
    """The Z2 null for `method`: its own scenario P&L set at the final market, scaled to each
    day's VaR forecast; None (Gaussian) for the parametric line, which has no scenario set."""
    for r in results:
        if r.method.startswith(method) and r.pnl is not None and r.pnl.size and r.var > 0:
            return bt.scenario_sampler(r.pnl, var_t, r.var)
    return None


def _money(x: float) -> str:
    return f"{x:,.0f}"


def _by_name(tests: list[bt.TestResult]) -> dict[str, bt.TestResult]:
    return {t.name: t for t in tests}


def render(rep: Report) -> str:
    b, m = rep.book, rep.market
    legs = b.legs()
    lines = ["# riskkit report", "", f"state: {rep.assessment.state}", ""]
    spot = " · ".join(f"{u} spot {m.spot[u]:,.2f} · ATM vol {m.vol[u]:.1%}" for u in b.underlyings())
    lines += [f"asof {pd.Timestamp(m.asof):%Y-%m-%d %H:%M %Z} · {spot} · {len(b.positions)} positions · {len(legs)} legs"
              + (f" · book value ${_money(book_value(list(rep.assessment.marks)))}" if rep.assessment.marks else ""),
              "", "Units: US dollars. VaR and ES are losses (positive numbers); P&L is positive = gain. "
              f"Confidence {rep.confidence:.0%}, horizon {rep.horizon_days} day(s), full revaluation unless the method says parametric.", ""]
    lines += [f"## VaR / ES, {rep.horizon_days}-day, {rep.confidence:.0%}", "", "| method | VaR | ES | n scenarios | notes |", "|---|---:|---:|---:|---|"]
    for r in rep.results:
        lines.append(f"| {r.method} | {_money(r.var)} | {_money(r.es)} | {r.n_scenarios or '—'} | {', '.join(r.notes) or ''} |")
    if not rep.results:
        lines.append("| — | 0 | 0 | 0 | EMPTY |")
    g = rep.assessment.var
    lines += ["", f"Governing number (state {rep.assessment.state}): {g.method} VaR {_money(g.var)} / ES {_money(g.es)}"
              + (f" — {', '.join(g.notes)}" if g.notes else "") + "."]
    if rep.forecasts is not None:
        n = len(rep.forecasts)
        lines += ["", f"## Backtest, last {n} days, one-day {rep.confidence:.0%} VaR/ES out of sample (each forecast from the preceding {rep.window} days)", "",
                  f"| method | exceptions (expected {n * (1 - rep.confidence):.1f}) | Kupiec p | Christoffersen CC p | Basel zone | Acerbi-Szekely Z2 (p) | DQ p |",
                  "|---|---:|---:|---:|---|---:|---:|"]
        for meth, ts in rep.tests.items():
            t = _by_name(ts)
            lines.append(f"| {meth} | {t['kupiec_pof'].n_exceptions} | {t['kupiec_pof'].p_value:.3f} | {t['christoffersen_cc'].p_value:.3f} | "
                         f"{t['basel_traffic_light'].verdict} | {t['acerbi_szekely_z2'].statistic:+.2f} ({t['acerbi_szekely_z2'].p_value:.3f}) | "
                         f"{t['engle_manganelli_dq'].p_value:.3f} |")
        lines.append("")
        lines.append("Verdicts at 5 %: p < 0.05 rejects. Kupiec's exact size at n = 250, p = 0.01 is "
                     f"{bt.kupiec_size(250, 0.01):.3f}, not 0.05 (see docs/DESIGN.md). Christoffersen CC and DQ p-values "
                     "are simulated under the exact i.i.d. null at this n; the Z2 p-value is simulated from each method's "
                     "own scenario set scaled to the day's VaR (Gaussian for the parametric line).")
    if len(rep.ladder):
        lines += ["", "## Stress ladder (P&L, $; rows = spot log move, columns = vol change in points)", ""]
        lines += _table(rep.ladder)
    if len(rep.scenarios):
        lines += ["", "## Historical scenarios (P&L, $; spot move as a simple return, vol in points)", "",
                  "| scenario | spot move | vol move | days | P&L |", "|---|---:|---:|---:|---:|"]
        for _, r in rep.scenarios.iterrows():
            lines.append(f"| {r['scenario']} | {math.exp(r['spot_move']) - 1:+.1%} | {r['vol_move'] * 100:+.1f} pts | {r['days']:.0f} | {_money(r['pnl'])} |")
    if rep.reverse:
        lines += ["", f"## Reverse stress (loss limit ${_money(rep.loss_limit)})", ""]
        for rs in rep.reverse:
            if rs.found:
                other = "vol" if rs.axis == "spot" else "spot"
                shock = f"{rs.shock:+.2%}" if rs.axis == "spot" else f"{rs.shock * 100:+.1f} vol pts"
                resp = f"{rs.response * 100:+.1f} vol pts" if rs.axis == "spot" else f"{rs.response:+.2%}"
                lines.append(f"- smallest {rs.axis} move {rs.direction} that loses the limit: {shock} (with {other} {resp}); P&L {_money(rs.pnl)}")
            else:
                lines.append(f"- {rs.axis} {rs.direction}: no breach within {rs.max_shock:g} — the book does not lose ${_money(rs.limit)} on this axis")
    lines += ["", "## Flags", ""]
    if rep.assessment.flags:
        lines += [f"- {f.kind} [{f.subject}]: {f.reason}" for f in rep.assessment.flags]
    else:
        lines.append("- none")
    lines += ["", f"generated by `riskkit report` · python {platform.python_version()} · {platform.system()} {platform.machine()}", ""]
    return "\n".join(lines)


def _table(df: pd.DataFrame) -> list[str]:
    cols = [f"{c * 100:+.0f} pts" for c in df.columns]
    out = [f"| {df.index.name} \\ {df.columns.name} | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for idx, row in df.iterrows():
        out.append(f"| {idx:+.0%} | " + " | ".join(_money(v) for v in row) + " |")
    return out
