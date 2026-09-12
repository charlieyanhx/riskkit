"""`riskkit demo | report`."""

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
    r.add_argument("--no-backtest", action="store_true")
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
                    run_backtest=not a.no_backtest)
        print(render(rep))


if __name__ == "__main__":
    main()
