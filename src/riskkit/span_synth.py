"""A small synthetic SPAN parameter file in CME's expanded unpacked (.pa2) layout, regenerable.

The repo may not redistribute CME's file, so the committed fixture `tests/fixtures/span_synthetic.pa2` is
generated here for a fictional exchange complex "SYN" / exchange "XYZ": a futures family "ZZ", its options
"OZ" (combined commodity "CC-ZZ") and an unrelated "YY" (combined commodity "CC-YY") so that selection is
exercised. The writer places every field at its 1-based CME column with the widths and implied decimals of
the layout pages (see `span_files`), written out by hand here rather than derived from the parser's table,
so a parse-then-compare round trip checks the two against each other. Array values follow the convention
of the real file: 5-digit magnitude then sign, loss-positive for long 1; the futures arrays are the
standard SPAN shape (0, 0, -/+ 1/3, +/- 2/3, 3/3 and the 0.33 x 3 x scan extremes) at a 12,000 scan range.
Two weekly calls share a month and strike and differ only in the option day/week code, so the fixture
exercises the day-coded identity. The no-settlement sentinel, the high-precision flag and the CBT price
alignment codes are covered by hand-typed lines in the tests, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .span_files import (
    ArrayParams,
    CombinedCommodity,
    DeliverySom,
    Header,
    InterLeg,
    InterSpread,
    IntraSpread,
    IntraTiers,
    PriceParams,
    ProductFamily,
    RiskArray,
    ScanningMethod,
    SpreadLeg,
    Tier,
)

EXCHANGE = "XYZ"
COMPLEX = "SYN"
FUT, OPT, OTHER = "ZZ", "OZ", "YY"
CC, CC_OTHER = "CC-ZZ", "CC-YY"
SCAN = 12_000.0
EXTREME_MULT, COVER = 3.0, 0.33
SOM_RATE = 50.0
STRIKE_DECIMALS, SETTLE_DECIMALS = 0, 2
CVF = 100.0


@dataclass(frozen=True)
class SyntheticSpan:
    header: Header
    price_params: tuple[PriceParams, ...]
    combined: CombinedCommodity
    combined_other: CombinedCommodity
    scanning: ScanningMethod
    intra_tiers: IntraTiers
    intra_spreads: tuple[IntraSpread, ...]
    delivery_som: DeliverySom
    array_params: tuple[ArrayParams, ...]
    arrays: tuple[RiskArray, ...]
    arrays_other: tuple[RiskArray, ...]
    inter_spreads: tuple[InterSpread, ...]


def futures_array(scan: float) -> tuple[float, ...]:
    """The standard SPAN futures array for a long 1: no vol effect, +/- thirds of the scan, extremes covered."""
    e = round(COVER * EXTREME_MULT * scan)
    t = scan / 3
    return (0.0, 0.0, -round(t), -round(t), round(t), round(t), -round(2 * t), -round(2 * t), round(2 * t), round(2 * t),
            -scan, -scan, scan, scan, -e, e)


def _fut(month: int, settle: float) -> RiskArray:
    return RiskArray(EXCHANGE, FUT, FUT, "FUT", "", month, "", 0, "", 0.0, futures_array(SCAN), 1.0, 0.0, settle, settle, 1.0, "C")


def _opt(right: str, fut_month: int, opt_month: int, strike: float, values, delta: float, iv: float, settle: float,
         option_day: str = "") -> RiskArray:
    return RiskArray(EXCHANGE, OPT, FUT, "OOF", right, fut_month, "", opt_month, option_day, strike, tuple(float(v) for v in values),
                     delta, iv, settle, settle, delta, "C")


def _bparams(commodity: str, ptype: str, fut_month: int, opt_month: int, base_vol: float, tte: float, expiry: str) -> ArrayParams:
    return ArrayParams(EXCHANGE, commodity, ptype, fut_month, "", opt_month, "", base_vol, 40.0 if ptype == "OOF" else 0.0, SCAN,
                       EXTREME_MULT, COVER, 0.04, tte, 0.0, 1.0, expiry, FUT, "W" if ptype == "OOF" else "", CVF,
                       "P" if ptype == "OOF" else "", "", 0)


def synthetic() -> SyntheticSpan:
    header = Header(COMPLEX, "20250912", "S", "C", "1836", "U2", "H", "1", "M")
    pp = (
        PriceParams(EXCHANGE, FUT, "FUT", "SYN 100 GOLD", SETTLE_DECIMALS, 0, CVF, 0.0, "USD", "STD", "AMER", "SYNTHETIC GOLD FUTURES",
                    "FUT", "DELIV"),
        PriceParams(EXCHANGE, OPT, "OOF", "SYN GOLD OPT", SETTLE_DECIMALS, STRIKE_DECIMALS, CVF, 0.0, "USD", "STD", "AMER",
                    "SYNTHETIC GOLD OPTIONS", "EQTY", "DELIV"),
        PriceParams(EXCHANGE, OTHER, "FUT", "SYN OTHER", 2, 0, 50.0, 0.0, "USD", "STD", "AMER", "SYNTHETIC OTHER FUTURES", "FUT",
                    "CASH"),
    )
    cc = CombinedCommodity(EXCHANGE, CC, 0, "USD", "P", "N", (ProductFamily(EXCHANGE, FUT, "FUT"), ProductFamily(EXCHANGE, OPT, "OOF")))
    cc_other = CombinedCommodity(EXCHANGE, CC_OTHER, 0, "USD", "P", "N", (ProductFamily(EXCHANGE, OTHER, "FUT"),))
    scanning = ScanningMethod(CC, "01", (), "2", (0, 0, 0, 0, 0))
    tiers = IntraTiers(CC, "10", (Tier(1, 202509, 202509), Tier(2, 202510, 202512), Tier(3, 202601, 202612)), 1.0, 1.0, 1.1)
    spreads = (
        IntraSpread(CC, "series", 1, 250.0, (SpreadLeg(202509, 1.0, "A"), SpreadLeg(202510, 1.0, "B"))),
        IntraSpread(CC, "series", 2, 300.0, (SpreadLeg(202510, 1.0, "A"), SpreadLeg(202512, 1.0, "B"))),
        IntraSpread(CC, "tier", 1, 400.0, (SpreadLeg(1, 1.0, "A"), SpreadLeg(2, 1.0, "B"))),
        IntraSpread(CC, "tier", 2, 350.0, (SpreadLeg(2, 1.0, "B"), SpreadLeg(2, 1.0, "A"))),
        IntraSpread(CC, "tier", 3, 600.0, (SpreadLeg(2, 1.0, "A"), SpreadLeg(3, 1.0, "B"))),
        IntraSpread(CC, "tier", 4, 700.0, (SpreadLeg(1, 1.0, "A"), SpreadLeg(3, 1.0, "B"))),
    )
    som = DeliverySom(CC, "01", SOM_RATE, 1.0, 1.0, 1.0, "2", "")
    bparams = (
        _bparams(FUT, "FUT", 202509, 0, 0.0, 0.038356, "20250926"),
        _bparams(FUT, "FUT", 202510, 0, 0.0, 0.128767, "20251029"),
        _bparams(FUT, "FUT", 202512, 0, 0.0, 0.29589, "20251229"),
        _bparams(FUT, "FUT", 202602, 0, 0.0, 0.454795, "20260225"),
        _bparams(OPT, "OOF", 202512, 202511, 0.151105, 0.126027, "20251028"),
        _bparams(OPT, "OOF", 202512, 202512, 0.150842, 0.2, "20251124"),
    )
    arrays = (
        _fut(202509, 3600.0), _fut(202510, 3610.5), _fut(202512, 3640.0), _fut(202602, 3670.25),
        # long-1 arrays, loss-positive: a call gains (negative) when the future rises, a put when it falls
        _opt("C", 202512, 202512, 3600.0, (-1500, 1400, -4200, -1600, 1100, 3900, -7300, -4900, 3100, 5900, -10500, -8400, 4900,
                                            7600, -11300, 3100), 0.55, 0.150842, 120.0),
        _opt("C", 202512, 202512, 3800.0, (-1100, 900, -2600, -400, 700, 1900, -5200, -2800, 1600, 3100, -8400, -5800, 2300, 3900,
                                            -10200, 1400), 0.30, 0.155, 45.0),
        _opt("P", 202512, 202512, 3600.0, (-1400, 1300, 1000, 3300, -3900, -1300, 2700, 5100, -7000, -4400, 4300, 6900, -10200,
                                            -8100, 2900, -11000), -0.45, 0.150842, 95.0),
        _opt("P", 202512, 202512, 3200.0, (-8, 6, 4, 9, -11, 2, 7, 12, -18, -3, 10, 14, -27, -9, 10, -30), -0.02, 0.21, 1.5),
        _opt("C", 202512, 202511, 3700.0, (-1200, 1000, -3300, -900, 900, 2600, -6100, -3600, 2200, 4200, -9300, -6800, 3300, 5300,
                                            -10800, 1900), 0.40, 0.151105, 60.0),
        # two weekly calls on the same month and strike, told apart only by the option day/week code (81 cols 45-46)
        _opt("C", 202512, 202511, 3700.0, (-900, 700, -2800, -600, 700, 2000, -5400, -2900, 1700, 3500, -8600, -6000, 2600, 4400,
                                            -10300, 1500), 0.38, 0.152, 41.0, option_day="12"),
        _opt("C", 202512, 202511, 3700.0, (-1100, 900, -3100, -800, 800, 2400, -5800, -3300, 2000, 3900, -9000, -6500, 3000, 4900,
                                            -10600, 1700), 0.39, 0.1515, 52.5, option_day="19"),
    )
    arrays_other = (RiskArray(EXCHANGE, OTHER, OTHER, "FUT", "", 202512, "", 0, "", 0.0, futures_array(3000.0), 1.0, 0.0, 95.5, 95.5,
                              1.0, "C"),)
    inter = (InterSpread("SYN", 1, 50.0, "01", "", "", (InterLeg(EXCHANGE, "", CC, 1.0, "A", 0), InterLeg(EXCHANGE, "", CC_OTHER, 1.0, "B", 0))),)
    return SyntheticSpan(header, pp, cc, cc_other, scanning, tiers, spreads, som, bparams, arrays, arrays_other, inter)


# --- writer: (1-based column, text) pieces on a blank line ----------------------------------------------------
def _line(pieces: list[tuple[int, str]]) -> str:
    width = max(c + len(t) - 1 for c, t in pieces)
    buf = [" "] * width
    for col, text in pieces:
        buf[col - 1:col - 1 + len(text)] = list(text)
    return "".join(buf).rstrip()


def _n(v: float, width: int, decimals: int = 0) -> str:
    """Unsigned implied-decimal magnitude, zero-padded."""
    return f"{round(abs(v) * 10**decimals):0{width}d}"


def _sv(v: float, width: int, decimals: int = 0) -> str:
    """Magnitude then sign character (array values, deltas, settlement prices)."""
    return _n(v, width, decimals) + ("-" if v < 0 else "+")


def _header(h: Header) -> str:
    return _line([(1, "0 "), (3, f"{h.exchange_complex:<6}"), (9, h.business_date), (17, h.settlement_flag), (18, f"{h.file_identifier:<2}"),
                  (20, f"{h.business_time:<4}"), (24, h.business_date), (32, "1900"), (36, h.file_format), (38, "Y"), (39, "N"), (40, "CLR  "),
                  (51, "C"), (53, "CUST "), (59, h.account_type), (61, "HEDGE"), (67, h.pb_class), (69, "CORE "), (75, h.maint_or_init),
                  (77, "MAINT")])


def _price(p: PriceParams) -> str:
    return _line([(1, "P "), (3, p.exchange), (6, f"{p.commodity:<10}"), (16, p.product_type), (19, f"{p.short_name:<15}"),
                  (34, f"{p.settlement_decimals:03d}"), (37, f"{p.strike_decimals:03d}"), (42, _n(p.contract_value_factor, 14, 7)),
                  (56, _n(p.cabinet_value, 8, 2)), (64, "01"), (66, p.settlement_currency), (69, "$"), (70, p.price_quotation),
                  (74, "00"), (76, p.exercise_style), (80, f"{p.long_name:<35}"), (115, "Y"), (116, "F"), (117, f"{p.valuation_method:<5}"),
                  (122, f"{p.settlement_method:<5}")])


def _combined(c: CombinedCommodity) -> str:
    pieces = [(1, "2 "), (3, c.exchange), (7, f"{c.code:<6}"), (13, str(c.risk_exponent)), (14, c.pb_currency), (17, "$"),
              (18, c.option_margin_style), (19, c.limit_option_value)]
    for k, f in enumerate(c.families):
        pieces += [(23 + 16 * k, f"{f.commodity:<10}"), (33 + 16 * k, f.product_type)]
    return _line(pieces)


def _scanning(s: ScanningMethod) -> str:
    return _line([(1, "S "), (3, f"{s.code:<6}"), (9, s.method), (11, f"{len(s.tiers):02d}"), (83, s.weighted_futures_method)])


def _tiers(t: IntraTiers) -> str:
    pieces = [(1, "3 "), (3, f"{t.code:<6}"), (9, t.method)]
    for k, tier in enumerate(t.tiers):
        pieces += [(11 + 14 * k, f"{tier.number:02d}"), (13 + 14 * k, str(tier.start)), (19 + 14 * k, str(tier.end))]
    pieces += [(69, _n(t.im_member, 4, 3)), (73, _n(t.im_hedger, 4, 3)), (77, _n(t.im_speculator, 4, 3))]
    return _line(pieces)


def _series(sp: IntraSpread) -> str:
    pieces = [(1, "E "), (3, f"{sp.code:<6}"), (9, f"{sp.priority:05d}"), (14, _n(sp.charge_rate, 7))]
    for k, leg in enumerate(sp.legs):
        pieces += [(21 + 14 * k, f"{leg.period - 200000:04d}"), (28 + 14 * k, _n(leg.ratio, 6, 4)), (34 + 14 * k, leg.side)]
    return _line(pieces)


def _tier_spread(sp: IntraSpread) -> str:
    pieces = [(1, "C "), (3, f"{sp.code:<6}"), (9, "10"), (11, f"{sp.priority:02d}"), (13, f"{len(sp.legs):02d}"), (15, _n(sp.charge_rate, 7))]
    for k, leg in enumerate(sp.legs):
        pieces += [(22 + 7 * k, f"{k + 1:02d}"), (24 + 7 * k, f"{leg.period:02d}"), (26 + 7 * k, _n(leg.ratio, 2)), (28 + 7 * k, leg.side)]
    return _line(pieces)


def _delivery(d: DeliverySom) -> str:
    return _line([(1, "4 "), (3, f"{d.code:<6}"), (9, d.delivery_method), (63, _n(d.som_rate, 7)), (70, _n(d.adj_member, 3, 2)),
                  (73, _n(d.adj_hedger, 3, 2)), (76, _n(d.adj_speculator, 3, 2)), (79, d.som_method or " "), (80, d.som_aggregation or " ")])


def _bline(b: ArrayParams) -> str:
    return _line([(1, "B "), (3, b.exchange), (6, f"{b.commodity:<10}"), (16, b.product_type), (19, str(b.contract_month)),
                  (28, f"{b.option_month:06d}" if b.option_month else "      "), (37, _n(b.base_vol, 8, 6)), (45, _n(b.vol_scan_range, 8, 6)),
                  (53, _n(b.price_scan_range, 5)), (58, _n(b.extreme_move_multiplier, 5, 3)), (63, _n(b.extreme_move_covered_fraction, 5, 4)),
                  (68, _n(b.interest_rate, 5, 4)), (73, _n(b.time_to_expiry, 7, 6)), (80, _n(b.lookahead, 6, 6)), (86, _n(b.delta_scaling, 6, 4)),
                  (92, b.expiration_date), (100, f"{b.underlying:<10}"), (110, f"{b.pricing_model:<2}"), (112, "00000000"),
                  (129, _n(b.contract_value_factor, 14, 7)), (143, "00"), (146, "00"), (149, "00"), (152, "010000000000"),
                  (164, b.vol_scan_quotation or " "), (165, b.price_scan_quotation or " "), (166, f"{abs(b.price_scan_exponent):02d}"),
                  (168, "-" if b.price_scan_exponent < 0 else " ")])


def _identity(a: RiskArray, rec: str) -> list[tuple[int, str]]:
    return [(1, rec), (3, a.exchange), (6, f"{a.commodity:<10}"), (16, f"{a.underlying:<10}"), (26, a.product_type), (29, a.right or " "),
            (30, str(a.contract_month)), (36, f"{a.contract_day:<2}"), (39, f"{a.option_month:06d}" if a.option_month else "      "),
            (45, f"{a.option_day:<2}"), (48, _n(a.strike, 7, STRIKE_DECIMALS))]


def _lines_81_82(a: RiskArray) -> list[str]:
    l81 = _identity(a, "81") + [(55 + 6 * i, _sv(a.values[i], 5)) for i in range(9)] + [
        (109, _n(a.hp_settlement_price, 14, SETTLE_DECIMALS)), (123, "N")]
    l82 = _identity(a, "82") + [(55 + 6 * (i - 9), _sv(a.values[i], 5)) for i in range(9, 16)] + [
        (97, _sv(a.composite_delta, 5, 4)), (103, _n(a.implied_vol, 8, 6)), (111, _sv(a.settlement_price, 7, SETTLE_DECIMALS)),
        (119, "+"), (120, _sv(a.current_delta, 5, 4)), (126, a.delta_flag)]
    return [_line(l81), _line(l82)]


def _inter(sp: InterSpread) -> str:
    pieces = [(1, "6 "), (3, sp.group), (6, f"{sp.priority:04d}"), (10, _n(sp.credit_rate, 7, 4))]
    for k, leg in enumerate(sp.legs):
        pieces += [(17 + 18 * k, leg.exchange), (20 + 18 * k, leg.required or " "), (21 + 18 * k, f"{leg.code:<6}"),
                   (27 + 18 * k, _n(leg.ratio, 7, 4)), (34 + 18 * k, leg.side)]
    pieces += [(89, sp.method), (101, sp.credit_method or " "), (102, "".join(f"{leg.tier:02d}" for leg in sp.legs)), (110, sp.spread_group or " ")]
    return _line(pieces)


def render(spec: SyntheticSpan) -> str:
    """The whole synthetic file, in CME's record order: header, currency, exchange, P, per-combined-commodity
    blocks (2, S, 3, E, C, 4, B), risk arrays (81/82), inter-commodity spreads (6)."""
    out = [_header(spec.header), "T USD$USD$0001000000", f"1 {EXCHANGE}  01"]
    out += [_price(p) for p in spec.price_params]
    out += [_combined(spec.combined), _scanning(spec.scanning), _tiers(spec.intra_tiers)]
    out += [_series(s) for s in spec.intra_spreads if s.kind == "series"]
    out += [_tier_spread(s) for s in spec.intra_spreads if s.kind == "tier"]
    out += [_delivery(spec.delivery_som)]
    out += [_bline(b) for b in spec.array_params]
    other = spec.combined_other
    out += [_combined(other), _scanning(ScanningMethod(other.code, "01", (), "2", (0,) * 5)),
            _tiers(IntraTiers(other.code, "01", (Tier(1, 202509, 203012),), 1.0, 1.0, 1.1)), _delivery(DeliverySom(other.code, "01", 25.0, 1.0, 1.0, 1.0, "2", ""))]
    for a in spec.arrays + spec.arrays_other:
        out += _lines_81_82(a)
    out += [_inter(s) for s in spec.inter_spreads]
    return "\n".join(out) + "\n"


def write_synthetic(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(synthetic()), encoding="latin-1")
    return p
