"""`riskkit demo | report | span | span-fixture | span-slice`."""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEMO_DIR = str(_REPO / "data/demo") if (_REPO / "data").exists() else "data/demo"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="riskkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="write the synthetic demo book and factor history (seeded, byte-identical)")
    d.add_argument("--out", default=DEMO_DIR)
    d.add_argument("--seed", type=int, default=11)
    r = sub.add_parser("report", help="print the markdown risk report for the demo book")
    r.add_argument("--data", default=DEMO_DIR, help="directory with book.json and history.csv")
    r.add_argument("--confidence", type=float, default=0.99)
    r.add_argument("--horizon", type=int, default=1)
    r.add_argument("--scenarios", type=int, default=20_000, help="Monte Carlo / FHS scenario count")
    r.add_argument("--n-test", type=int, default=250, help="out-of-sample days for the backtest")
    r.add_argument("--window", type=int, default=1000, help="estimation window (days) for each forecast")
    r.add_argument("--limit", type=float, default=25_000.0, help="loss limit for reverse stress, $")
    r.add_argument("--seed", type=int, default=0, help="seed for the Monte Carlo / FHS draws and the simulated p-values")
    r.add_argument("--no-backtest", action="store_true")
    sp = sub.add_parser("span", help="legacy-SPAN outright reconciliation of a CME .pa2 (or .zip) file against the published margins")
    sp.add_argument("--file", required=True, help="expanded unpacked SPAN file, .pa2 or a .zip containing one")
    sp.add_argument("--commodity", default="GC", help="comma-separated product codes, e.g. GC,ES,CL")
    sp.add_argument("--published", default=str(_REPO / "data/span/published_2025-09-12.csv"), help="published-margin CSV")
    sp.add_argument("--months", type=int, default=0, help="show only the first N futures months per product (0 = all)")
    sf = sub.add_parser("span-fixture", help="write the synthetic SPAN file (tests/fixtures/span_synthetic.pa2, byte-identical)")
    sf.add_argument("--out", default=str(_REPO / "tests/fixtures/span_synthetic.pa2"))
    ss = sub.add_parser("span-slice", help="cut one combined commodity out of a full SPAN file (private slice, gitignored)")
    ss.add_argument("--file", required=True)
    ss.add_argument("--commodity", required=True, help="a product code in the combined commodity, e.g. GC")
    ss.add_argument("--families", default=None, help="comma-separated product codes to keep arrays for, e.g. GC,OG (default: all)")
    ss.add_argument("--exchange", default=None)
    ss.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        from .synth import write_demo

        bp, hp = write_demo(a.out, a.seed)
        print(f"wrote {bp} and {hp}")
    elif a.cmd == "report":
        from .report import build, render
        from .synth import load_demo

        book, market, history = load_demo(a.data)
        rep = build(book, market, history, a.confidence, a.horizon, a.scenarios, a.n_test, a.window, a.limit,
                    seed=a.seed, run_backtest=not a.no_backtest)
        print(render(rep))
    elif a.cmd == "span":
        from .span import reconcile

        text, _ = reconcile(a.file, [c.strip() for c in a.commodity.split(",") if c.strip()], a.published, a.months)
        print(text, end="")
    elif a.cmd == "span-fixture":
        from .span_synth import write_synthetic

        print(f"wrote {write_synthetic(a.out)}")
    elif a.cmd == "span-slice":
        from .span_files import extract_slice

        fams = [f.strip() for f in a.families.split(",")] if a.families else None
        n = extract_slice(a.file, a.out, a.commodity, a.exchange, fams)
        print(f"wrote {n} lines to {a.out}")


if __name__ == "__main__":
    main()
