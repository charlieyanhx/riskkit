"""CME SPAN risk-parameter files: a streaming parser for the expanded unpacked ("U2", `.pa2`) positional format.

Units and conventions
- Every field is a fixed column slice. `LAYOUT` holds the offsets for all record types as 1-based inclusive
  `(from, to)` pairs exactly as printed on CME's layout pages, converted to Python slices in one place
  (`_s`). Nothing is regex-guessed. Source: CME Group Client Systems Wiki (pubsub space), "Risk Parameter
  File Layouts for the Positional Formats", https://cmegroupclientsite.atlassian.net/wiki/spaces/pubsub/
  pages/457083445, expanded-format pages for record types 0, 1, 2, 3, S, C, E, 4, B, P, 6 and 8 (81/82);
  cross-checked against the Apache-2.0 jburgy/span project (offsets read, not copied).
- Risk-array values: Type 81 carries values 1-9 (cols 55-108), Type 82 values 10-16 (cols 55-96), each a
  5-digit magnitude FOLLOWED by its sign character ("+" or "-"). A positive value is a LOSS for a single
  LONG position ("long the instrument": a long put is long); negative is a gain. Values are in the combined
  commodity's performance-bond currency, times 10**risk_exponent from the Type 2 record (`risk_scale`).
  Scenario order (CME Type 8 page): 1 unch/vol up, 2 unch/vol down, 3 up 1/3/vol up, 4 up 1/3/vol down,
  5 down 1/3/vol up, 6 down 1/3/vol down, 7 up 2/3/vol up, 8 up 2/3/vol down, 9 down 2/3/vol up,
  10 down 2/3/vol down, 11 up 3/3/vol up, 12 up 3/3/vol down, 13 down 3/3/vol up, 14 down 3/3/vol down,
  15 up extreme x cover fraction, 16 down extreme x cover fraction.
- Composite delta (82 cols 97-101 + sign col 102): 9V9(4), a future is +1.0000, puts are negative.
- Strike (cols 48-54) and settlement price (82 cols 111-117 + sign col 118) are integers with the decimal
  locators of the product family's Type P record (cols 34-36 settlement, 37-39 strike).
- Lines may be shorter than the layout: CME truncates trailing blanks. A missing numeric field reads as 0,
  a missing sign as "+", a missing text field as "".
- Months are ints CCYYMM; Type E months are YYMM in the file and are expanded to 20YYMM here.

The file is streamed line by line; `load_commodity` keeps the header, all Type P records (a few thousand
small records), one combined commodity's parameter block, that block's risk arrays and the Type 6 spreads
naming it. It never holds the 1.2 million risk-array lines of a full CME file at once.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# --- column table --------------------------------------------------------------------------------------------
# 1-based inclusive (from, to) as printed by CME (see module docstring for the source pages).
_ARRAY_81 = {f"v{i}": (55 + 6 * (i - 1), 59 + 6 * (i - 1)) for i in range(1, 10)}          # values 1-9
_ARRAY_82 = {f"v{i}": (55 + 6 * (i - 10), 59 + 6 * (i - 10)) for i in range(10, 17)}       # values 10-16
_IDENTITY_8 = {
    "exchange": (3, 5), "commodity": (6, 15), "underlying": (16, 25), "product_type": (26, 28), "right": (29, 29),
    "contract_month": (30, 35), "contract_day": (36, 37), "option_month": (39, 44), "option_day": (45, 46),
    "strike": (48, 54),
}
LAYOUT: dict[str, dict[str, tuple[int, int]]] = {
    "0 ": {  # exchange complex header
        "exchange_complex": (3, 8), "business_date": (9, 16), "settlement_flag": (17, 17), "file_identifier": (18, 19),
        "business_time": (20, 23), "creation_date": (24, 31), "creation_time": (32, 35), "file_format": (36, 37),
        "gross_net": (38, 38), "limit_option_value": (39, 39), "business_function": (40, 44),
        "clearing_or_customer": (51, 51), "account_type": (59, 59), "pb_class": (67, 67), "maint_or_init": (75, 75),
    },
    "1 ": {"exchange": (3, 5), "exchange_code": (8, 9)},
    "2 ": {  # combined commodity definition; six (commodity, product type, decimal locator) slots
        "exchange": (3, 5), "code": (7, 12), "risk_exponent": (13, 13), "pb_currency": (14, 16), "pb_currency_code": (17, 17),
        "option_margin_style": (18, 18), "limit_option_value": (19, 19), "combination_method": (20, 20),
        **{f"commodity_{k}": (23 + 16 * (k - 1), 32 + 16 * (k - 1)) for k in range(1, 7)},
        **{f"product_type_{k}": (33 + 16 * (k - 1), 35 + 16 * (k - 1)) for k in range(1, 7)},
        **{f"decimal_locator_{k}": (36 + 16 * (k - 1), 36 + 16 * (k - 1)) for k in range(1, 7)},
    },
    "3 ": {  # intracommodity spread tiers (up to four per record) + initial/maintenance ratios
        "code": (3, 8), "method": (9, 10),
        **{f"tier_{k}": (11 + 14 * (k - 1), 12 + 14 * (k - 1)) for k in range(1, 5)},
        **{f"start_{k}": (13 + 14 * (k - 1), 18 + 14 * (k - 1)) for k in range(1, 5)},
        **{f"end_{k}": (19 + 14 * (k - 1), 24 + 14 * (k - 1)) for k in range(1, 5)},
        "im_member": (69, 72), "im_hedger": (73, 76), "im_speculator": (77, 80),
    },
    "S ": {  # scanning / intercommodity tiers (up to five) + weighted futures price risk method + SOM tier rates
        "code": (3, 8), "method": (9, 10), "n_tiers": (11, 12),
        **{f"tier_{k}": (13 + 14 * (k - 1), 14 + 14 * (k - 1)) for k in range(1, 6)},
        **{f"start_{k}": (15 + 14 * (k - 1), 20 + 14 * (k - 1)) for k in range(1, 6)},
        **{f"end_{k}": (21 + 14 * (k - 1), 26 + 14 * (k - 1)) for k in range(1, 6)},
        "weighted_futures_method": (83, 83),
        **{f"som_rate_{k}": (104 + 7 * (k - 1), 110 + 7 * (k - 1)) for k in range(1, 6)},
    },
    "C ": {  # tier-to-tier intracommodity spread, up to eight legs
        "code": (3, 8), "method": (9, 10), "priority": (11, 12), "n_legs": (13, 14), "charge_rate": (15, 21),
        **{f"leg_{k}": (22 + 7 * (k - 1), 23 + 7 * (k - 1)) for k in range(1, 9)},
        **{f"tier_{k}": (24 + 7 * (k - 1), 25 + 7 * (k - 1)) for k in range(1, 9)},
        **{f"ratio_{k}": (26 + 7 * (k - 1), 27 + 7 * (k - 1)) for k in range(1, 9)},
        **{f"side_{k}": (28 + 7 * (k - 1), 28 + 7 * (k - 1)) for k in range(1, 9)},
    },
    "E ": {  # series-to-series intracommodity spread, up to four legs; months YYMM; ratio 9(2)V9(4)
        "code": (3, 8), "priority": (9, 13), "charge_rate": (14, 20),
        **{f"month_{k}": (21 + 14 * (k - 1), 24 + 14 * (k - 1)) for k in range(1, 5)},
        **{f"ratio_{k}": (28 + 14 * (k - 1), 33 + 14 * (k - 1)) for k in range(1, 5)},
        **{f"side_{k}": (34 + 14 * (k - 1), 34 + 14 * (k - 1)) for k in range(1, 5)},
    },
    "4 ": {  # delivery (spot) charge method + short option minimum
        "code": (3, 8), "delivery_method": (9, 10), "som_rate": (63, 69), "adj_member": (70, 72), "adj_hedger": (73, 75),
        "adj_speculator": (76, 78), "som_method": (79, 79), "som_aggregation": (80, 80),
    },
    "B ": {  # array calculation parameters per future / option series
        "exchange": (3, 5), "commodity": (6, 15), "product_type": (16, 18), "contract_month": (19, 24), "contract_day": (25, 26),
        "option_month": (28, 33), "option_day": (34, 35), "base_vol": (37, 44), "vol_scan_range": (45, 52),
        "price_scan_range": (53, 57), "extreme_move_multiplier": (58, 62), "extreme_move_covered_fraction": (63, 67),
        "interest_rate": (68, 72), "time_to_expiry": (73, 79), "lookahead": (80, 85), "delta_scaling": (86, 91),
        "expiration_date": (92, 99), "underlying": (100, 109), "pricing_model": (110, 111),
        "contract_value_factor": (129, 142), "vol_scan_quotation": (164, 164), "price_scan_quotation": (165, 165),
        "price_scan_exponent": (166, 167), "price_scan_exponent_sign": (168, 168),
    },
    "P ": {  # price conversion parameters per product family
        "exchange": (3, 5), "commodity": (6, 15), "product_type": (16, 18), "short_name": (19, 33),
        "settlement_decimals": (34, 36), "strike_decimals": (37, 39), "contract_value_factor": (42, 55),
        "cabinet_value": (56, 63), "settlement_currency": (66, 68), "price_quotation": (70, 72),
        "cvf_exponent_sign": (73, 73), "cvf_exponent": (74, 75), "exercise_style": (76, 79), "long_name": (80, 114),
        "valuation_method": (117, 121), "settlement_method": (122, 126),
    },
    "6 ": {  # intercommodity spread, up to four legs per record
        "group": (3, 5), "priority": (6, 9), "credit_rate": (10, 16),
        **{f"exchange_{k}": (17 + 18 * (k - 1), 19 + 18 * (k - 1)) for k in range(1, 5)},
        **{f"required_{k}": (20 + 18 * (k - 1), 20 + 18 * (k - 1)) for k in range(1, 5)},
        **{f"code_{k}": (21 + 18 * (k - 1), 26 + 18 * (k - 1)) for k in range(1, 5)},
        **{f"ratio_{k}": (27 + 18 * (k - 1), 33 + 18 * (k - 1)) for k in range(1, 5)},
        **{f"side_{k}": (34 + 18 * (k - 1), 34 + 18 * (k - 1)) for k in range(1, 5)},
        "method": (89, 90), "credit_method": (101, 101),
        **{f"tier_{k}": (102 + 2 * (k - 1), 103 + 2 * (k - 1)) for k in range(1, 5)},
        "spread_group": (110, 110),
    },
    "81": {**_IDENTITY_8, **_ARRAY_81, "hp_settlement": (109, 122), "hp_settlement_flag": (123, 123)},
    "82": {
        **_IDENTITY_8, **_ARRAY_82, "composite_delta": (97, 101), "composite_delta_sign": (102, 102),
        "implied_vol": (103, 110), "settlement": (111, 117), "settlement_sign": (118, 118), "strike_sign": (119, 119),
        "current_delta": (120, 124), "current_delta_sign": (125, 125), "delta_flag": (126, 126),
    },
}
SIGN_COL_AFTER_VALUE = 1   # each array value's sign is the character immediately after its 5 digits
N_SCENARIOS = 16
PRODUCT_TYPES_OPTION = ("OOF", "OOP", "OOC")


# --- slicing helpers -------------------------------------------------------------------------------------------
def _s(line: str, rec: str, name: str) -> str:
    a, b = LAYOUT[rec][name]
    return line[a - 1:b]


def _text(line: str, rec: str, name: str) -> str:
    return _s(line, rec, name).strip()


def _int(line: str, rec: str, name: str, default: int = 0) -> int:
    t = _s(line, rec, name).strip()
    return int(t) if t else default


def _dec(line: str, rec: str, name: str, decimals: int) -> float:
    """Unsigned implied-decimal number 9(n)V9(decimals); blank reads as 0."""
    return _int(line, rec, name) / 10**decimals


def _signed_after(line: str, rec: str, name: str, decimals: int = 0) -> float:
    """Magnitude field followed by its sign character (the SPAN convention for array values and deltas)."""
    a, b = LAYOUT[rec][name]
    mag = line[a - 1:b].strip()
    sign = line[b:b + SIGN_COL_AFTER_VALUE]
    v = (int(mag) if mag else 0) / 10**decimals
    return -v if sign == "-" else v


def _locator(line: str, rec: str, name: str) -> int:
    """Decimal locator 9(3) with an optional leading '-' ("-02" means -2)."""
    t = _s(line, rec, name).strip()
    return int(t) if t else 0


# --- records ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Header:
    exchange_complex: str
    business_date: str          # CCYYMMDD
    settlement_flag: str        # S settlement / I intraday
    file_identifier: str        # E early, F final, C complete
    business_time: str
    file_format: str            # U2 expanded unpacked
    account_type: str           # blank/H hedger, M member, S speculator
    pb_class: str               # blank/1 core, 2 reserve
    maint_or_init: str          # M maintenance / I initial


@dataclass(frozen=True)
class ProductFamily:
    exchange: str
    commodity: str
    product_type: str
    decimal_locator: int = 0    # non-zero means Type 83/84 records (not supported here)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.exchange, self.commodity, self.product_type)


@dataclass(frozen=True)
class CombinedCommodity:
    exchange: str
    code: str
    risk_exponent: int
    pb_currency: str
    option_margin_style: str    # P premium-style (NOV applies), F futures-style
    limit_option_value: str
    families: tuple[ProductFamily, ...]

    @property
    def risk_scale(self) -> float:
        return 10.0**self.risk_exponent


@dataclass(frozen=True)
class Tier:
    number: int
    start: int                  # CCYYMM
    end: int                    # CCYYMM inclusive

    def contains(self, month: int) -> bool:
        return self.start <= month <= self.end


@dataclass(frozen=True)
class IntraTiers:
    code: str
    method: str                 # "01" no charge, "10" table-driven
    tiers: tuple[Tier, ...]
    im_member: float
    im_hedger: float
    im_speculator: float


@dataclass(frozen=True)
class ScanningMethod:
    code: str
    method: str
    tiers: tuple[Tier, ...]
    weighted_futures_method: str
    som_rates: tuple[float, ...]


@dataclass(frozen=True)
class SpreadLeg:
    period: int                 # tier number (Type C) or contract month CCYYMM (Type E)
    ratio: float                # delta per spread
    side: str                   # "A" or "B"


@dataclass(frozen=True)
class IntraSpread:
    code: str
    kind: str                   # "tier" (Type C) or "series" (Type E)
    priority: int
    charge_rate: float          # performance-bond currency per spread, before risk_scale
    legs: tuple[SpreadLeg, ...]


@dataclass(frozen=True)
class DeliverySom:
    code: str
    delivery_method: str        # "01" no spot charge, "10" table-driven, "11" basis risk
    som_rate: float             # per short option, before risk_scale
    adj_member: float
    adj_hedger: float
    adj_speculator: float
    som_method: str             # blank/"2" sum of short calls and puts, "1" the greater of the two
    som_aggregation: str


@dataclass(frozen=True)
class ArrayParams:
    exchange: str
    commodity: str
    product_type: str
    contract_month: int
    contract_day: str
    option_month: int
    option_day: str
    base_vol: float
    vol_scan_range: float
    price_scan_range: float     # performance-bond currency, before risk_scale and the PSR exponent
    extreme_move_multiplier: float
    extreme_move_covered_fraction: float
    interest_rate: float
    time_to_expiry: float
    lookahead: float
    delta_scaling: float
    expiration_date: str
    underlying: str
    pricing_model: str
    contract_value_factor: float
    vol_scan_quotation: str
    price_scan_quotation: str
    price_scan_exponent: int

    @property
    def series_key(self) -> tuple[str, str, str, int, str, int, str]:
        return (self.exchange, self.commodity, self.product_type, self.contract_month, self.contract_day,
                self.option_month, self.option_day)


@dataclass(frozen=True)
class PriceParams:
    exchange: str
    commodity: str
    product_type: str
    short_name: str
    settlement_decimals: int
    strike_decimals: int
    contract_value_factor: float
    cabinet_value: float
    settlement_currency: str
    price_quotation: str
    exercise_style: str
    long_name: str
    valuation_method: str
    settlement_method: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.exchange, self.commodity, self.product_type)


@dataclass(frozen=True)
class InterLeg:
    exchange: str
    required: str
    code: str
    ratio: float
    side: str
    tier: int


@dataclass(frozen=True)
class InterSpread:
    group: str
    priority: int
    credit_rate: float          # percent
    method: str
    credit_method: str
    spread_group: str
    legs: tuple[InterLeg, ...]


@dataclass(frozen=True)
class RiskArray:
    exchange: str
    commodity: str
    underlying: str
    product_type: str           # FUT, OOF, OOP, PHY, CMB, OOC
    right: str                  # "C" / "P" / ""
    contract_month: int         # CCYYMM of the future (for an option: its underlying future's month)
    contract_day: str
    option_month: int           # CCYYMM, 0 for a future
    option_day: str
    strike: float
    values: tuple[float, ...]   # 16 scenario values, loss-positive for long 1, in PB currency (risk_scale applied)
    composite_delta: float
    implied_vol: float          # decimal fraction (0.15 = 15 %)
    settlement_price: float     # price units (decimal locator applied)
    hp_settlement_price: float
    current_delta: float
    delta_flag: str

    @property
    def key(self) -> tuple[str, str, int, int, str, float]:
        return (self.commodity, self.product_type, self.contract_month, self.option_month, self.right, self.strike)

    @property
    def is_option(self) -> bool:
        return self.product_type in PRODUCT_TYPES_OPTION

    @property
    def series_key(self) -> tuple[str, str, str, int, str, int, str]:
        return (self.exchange, self.commodity, self.product_type, self.contract_month, self.contract_day,
                self.option_month, self.option_day)


# --- per-record parsers ---------------------------------------------------------------------------------------
def parse_header(line: str) -> Header:
    r = "0 "
    return Header(_text(line, r, "exchange_complex"), _text(line, r, "business_date"), _text(line, r, "settlement_flag"),
                  _text(line, r, "file_identifier"), _text(line, r, "business_time"), _text(line, r, "file_format"),
                  _text(line, r, "account_type"), _text(line, r, "pb_class"), _text(line, r, "maint_or_init"))


def parse_combined_commodity(line: str) -> CombinedCommodity:
    """One Type 2 line (a combined commodity may continue over several; `load_commodity` merges them)."""
    r = "2 "
    exchange = _text(line, r, "exchange")
    fams = []
    for k in range(1, 7):
        c = _text(line, r, f"commodity_{k}")
        if c:
            fams.append(ProductFamily(exchange, c, _text(line, r, f"product_type_{k}"), _int(line, r, f"decimal_locator_{k}")))
    return CombinedCommodity(exchange, _text(line, r, "code"), _int(line, r, "risk_exponent"), _text(line, r, "pb_currency"),
                             _text(line, r, "option_margin_style") or "P", _text(line, r, "limit_option_value"), tuple(fams))


def _tiers(line: str, rec: str, n: int) -> tuple[Tier, ...]:
    out = []
    for k in range(1, n + 1):
        num = _int(line, rec, f"tier_{k}")
        if num:
            out.append(Tier(num, _int(line, rec, f"start_{k}"), _int(line, rec, f"end_{k}")))
    return tuple(out)


def parse_intra_tiers(line: str) -> IntraTiers:
    r = "3 "
    return IntraTiers(_text(line, r, "code"), _text(line, r, "method"), _tiers(line, r, 4),
                      _dec(line, r, "im_member", 3), _dec(line, r, "im_hedger", 3), _dec(line, r, "im_speculator", 3))


def parse_scanning(line: str) -> ScanningMethod:
    r = "S "
    return ScanningMethod(_text(line, r, "code"), _text(line, r, "method"), _tiers(line, r, 5),
                          _text(line, r, "weighted_futures_method"), tuple(_int(line, r, f"som_rate_{k}") for k in range(1, 6)))


def parse_tier_spread(line: str) -> IntraSpread:
    r = "C "
    legs = tuple(SpreadLeg(_int(line, r, f"tier_{k}"), float(_int(line, r, f"ratio_{k}")), _text(line, r, f"side_{k}"))
                 for k in range(1, 9) if _int(line, r, f"leg_{k}"))
    return IntraSpread(_text(line, r, "code"), "tier", _int(line, r, "priority"), float(_int(line, r, "charge_rate")), legs)


def parse_series_spread(line: str) -> IntraSpread:
    r = "E "
    legs = tuple(SpreadLeg(200000 + _int(line, r, f"month_{k}"), _dec(line, r, f"ratio_{k}", 4), _text(line, r, f"side_{k}"))
                 for k in range(1, 5) if _int(line, r, f"month_{k}"))
    return IntraSpread(_text(line, r, "code"), "series", _int(line, r, "priority"), float(_int(line, r, "charge_rate")), legs)


def parse_delivery_som(line: str) -> DeliverySom:
    r = "4 "
    return DeliverySom(_text(line, r, "code"), _text(line, r, "delivery_method"), float(_int(line, r, "som_rate")),
                       _dec(line, r, "adj_member", 2) or 1.0, _dec(line, r, "adj_hedger", 2) or 1.0,
                       _dec(line, r, "adj_speculator", 2) or 1.0, _text(line, r, "som_method"), _text(line, r, "som_aggregation"))


def parse_array_params(line: str) -> ArrayParams:
    r = "B "
    exp = _int(line, r, "price_scan_exponent")
    if _text(line, r, "price_scan_exponent_sign") == "-":
        exp = -exp
    return ArrayParams(
        _text(line, r, "exchange"), _text(line, r, "commodity"), _text(line, r, "product_type"), _int(line, r, "contract_month"),
        _text(line, r, "contract_day"), _int(line, r, "option_month"), _text(line, r, "option_day"),
        _dec(line, r, "base_vol", 6), _dec(line, r, "vol_scan_range", 6), float(_int(line, r, "price_scan_range")),
        _dec(line, r, "extreme_move_multiplier", 3), _dec(line, r, "extreme_move_covered_fraction", 4),
        _dec(line, r, "interest_rate", 4), _dec(line, r, "time_to_expiry", 6), _dec(line, r, "lookahead", 6),
        _dec(line, r, "delta_scaling", 4) or 1.0, _text(line, r, "expiration_date"), _text(line, r, "underlying"),
        _text(line, r, "pricing_model"), _dec(line, r, "contract_value_factor", 7), _text(line, r, "vol_scan_quotation"),
        _text(line, r, "price_scan_quotation"), exp,
    )


def parse_price_params(line: str) -> PriceParams:
    r = "P "
    cvf = _dec(line, r, "contract_value_factor", 7)
    exp = _int(line, r, "cvf_exponent")
    if exp:
        cvf *= 10.0 ** (-exp if _text(line, r, "cvf_exponent_sign") == "-" else exp)
    return PriceParams(_text(line, r, "exchange"), _text(line, r, "commodity"), _text(line, r, "product_type"),
                       _text(line, r, "short_name"), _locator(line, r, "settlement_decimals"), _locator(line, r, "strike_decimals"),
                       cvf, _dec(line, r, "cabinet_value", 2), _text(line, r, "settlement_currency"), _text(line, r, "price_quotation"),
                       _text(line, r, "exercise_style") or "AMER", _text(line, r, "long_name"), _text(line, r, "valuation_method"),
                       _text(line, r, "settlement_method"))


def parse_inter_spread(line: str) -> InterSpread:
    r = "6 "
    legs = tuple(InterLeg(_text(line, r, f"exchange_{k}"), _text(line, r, f"required_{k}"), _text(line, r, f"code_{k}"),
                          _dec(line, r, f"ratio_{k}", 4), _text(line, r, f"side_{k}"), _int(line, r, f"tier_{k}"))
                 for k in range(1, 5) if _text(line, r, f"code_{k}"))
    return InterSpread(_text(line, r, "group"), _int(line, r, "priority"), _dec(line, r, "credit_rate", 4), _text(line, r, "method") or "01",
                       _text(line, r, "credit_method"), _text(line, r, "spread_group"), legs)


def identity_8(line: str) -> tuple:
    """The contract identity shared by the 81 and 82 lines (used to pair them)."""
    r = line[:2]
    return tuple(_text(line, r, k) for k in _IDENTITY_8)


def parse_risk_array_pair(l81: str, l82: str, strike_decimals: int = 0, settlement_decimals: int = 0,
                          risk_scale: float = 1.0) -> RiskArray:
    """An 81 line and its 82 line -> one contract's `RiskArray`. Raises if the two identities differ."""
    if l81[:2] != "81" or l82[:2] != "82":
        raise ValueError(f"expected an 81 line then an 82 line, got {l81[:2]!r} then {l82[:2]!r}")
    if identity_8(l81) != identity_8(l82):
        raise ValueError(f"81/82 identity mismatch: {identity_8(l81)} vs {identity_8(l82)}")
    vals = [_signed_after(l81, "81", f"v{i}") for i in range(1, 10)] + [_signed_after(l82, "82", f"v{i}") for i in range(10, 17)]
    strike = _int(l81, "81", "strike") / 10**strike_decimals
    if _text(l82, "82", "strike_sign") == "-":
        strike = -strike
    return RiskArray(
        _text(l81, "81", "exchange"), _text(l81, "81", "commodity"), _text(l81, "81", "underlying"), _text(l81, "81", "product_type"),
        _text(l81, "81", "right"), _int(l81, "81", "contract_month"), _text(l81, "81", "contract_day"), _int(l81, "81", "option_month"),
        _text(l81, "81", "option_day"), strike, tuple(v * risk_scale for v in vals),
        _signed_after(l82, "82", "composite_delta", 4), _dec(l82, "82", "implied_vol", 6),
        _signed_after(l82, "82", "settlement", settlement_decimals), _int(l81, "81", "hp_settlement") / 10**settlement_decimals,
        _signed_after(l82, "82", "current_delta", 4), _text(l82, "82", "delta_flag"),
    )


# --- streaming --------------------------------------------------------------------------------------------------
def iter_lines(path: str | Path) -> Iterator[str]:
    """Lines of a .pa2 file, or of the .pa2 member inside a .zip, without the line terminator."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            member = next((n for n in names if n.lower().endswith(".pa2")), names[0])
            with zf.open(member) as raw, io.TextIOWrapper(raw, encoding="latin-1", newline="") as fh:
                for line in fh:
                    yield line.rstrip("\r\n")
    else:
        with open(p, encoding="latin-1", newline="") as fh:
            for line in fh:
                yield line.rstrip("\r\n")


@dataclass
class SpanData:
    """One combined commodity's parameters and risk arrays from one SPAN file."""

    header: Header
    combined: CombinedCommodity
    price_params: dict[tuple[str, str, str], PriceParams]
    intra_tiers: IntraTiers | None
    scanning: ScanningMethod | None
    intra_spreads: tuple[IntraSpread, ...]          # series (Type E) then tier (Type C), each by priority
    delivery_som: DeliverySom | None
    array_params: dict[tuple, ArrayParams]
    arrays: list[RiskArray]
    inter_spreads: tuple[InterSpread, ...]
    notes: tuple[str, ...] = ()
    _index: dict[tuple, RiskArray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {a.key: a for a in self.arrays}

    def find(self, commodity: str, product_type: str, contract_month: int, right: str = "", strike: float = 0.0,
             option_month: int | None = None) -> RiskArray:
        om = 0 if product_type == "FUT" else (contract_month if option_month is None else option_month)
        key = (commodity, product_type, contract_month, om, right, float(strike))
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(f"no risk array for {key} in {self.combined.code} ({len(self.arrays)} contracts loaded)") from None

    def price_params_for(self, array: RiskArray) -> PriceParams | None:
        return self.price_params.get((array.exchange, array.commodity, array.product_type))

    def array_params_for(self, array: RiskArray) -> ArrayParams | None:
        return self.array_params.get(array.series_key)

    def futures(self, commodity: str | None = None) -> list[RiskArray]:
        return [a for a in self.arrays if a.product_type == "FUT" and (commodity is None or a.commodity == commodity)]


def _wanted_families(cc: CombinedCommodity, families: Iterable[str] | None) -> set[tuple[str, str, str]]:
    keep = None if families is None else {f.strip() for f in families}
    return {f.key for f in cc.families if keep is None or f.commodity in keep}


def load_commodity(path: str | Path, commodity: str, exchange: str | None = None,
                   families: Iterable[str] | None = None) -> SpanData:
    """Stream `path` once and return the combined commodity that lists product `commodity` (any product type).

    `exchange` disambiguates a code listed on more than one exchange; `families` restricts the risk arrays
    kept to those product codes (e.g. ("GC", "OG")); the parameter block is always complete. Type 6 spreads
    are kept when any leg names the combined commodity.
    """
    header = None
    price_params: dict[tuple[str, str, str], PriceParams] = {}
    cc: CombinedCommodity | None = None
    in_block = False
    intra_tiers: IntraTiers | None = None
    scanning: ScanningMethod | None = None
    series: list[IntraSpread] = []
    tiers_spreads: list[IntraSpread] = []
    delivery: DeliverySom | None = None
    bparams: dict[tuple, ArrayParams] = {}
    arrays: list[RiskArray] = []
    inter: list[InterSpread] = []
    wanted: set[tuple[str, str, str]] = set()
    pending81: str | None = None
    notes: list[str] = []

    for line in iter_lines(path):
        rec = line[:2]
        if rec == "0 ":
            header = parse_header(line)
        elif rec == "P ":
            pp = parse_price_params(line)
            price_params[pp.key] = pp
        elif rec == "2 ":
            part = parse_combined_commodity(line)
            if cc is not None and part.code == cc.code and part.exchange == cc.exchange:
                cc = CombinedCommodity(cc.exchange, cc.code, cc.risk_exponent, cc.pb_currency, cc.option_margin_style,
                                       cc.limit_option_value, cc.families + part.families)
                wanted = _wanted_families(cc, families)
            elif cc is None and any(f.commodity == commodity for f in part.families) and (exchange is None or part.exchange == exchange):
                cc, in_block = part, True
                wanted = _wanted_families(cc, families)
            else:
                in_block = False
        elif not in_block and rec in ("S ", "3 ", "E ", "C ", "4 ", "B "):
            continue
        elif rec == "S ":
            scanning = parse_scanning(line) if scanning is None else scanning
        elif rec == "3 ":
            t = parse_intra_tiers(line)
            intra_tiers = t if intra_tiers is None else IntraTiers(t.code, t.method, intra_tiers.tiers + t.tiers, t.im_member,
                                                                    t.im_hedger, t.im_speculator)
        elif rec == "E ":
            series.append(parse_series_spread(line))
        elif rec == "C ":
            tiers_spreads.append(parse_tier_spread(line))
        elif rec == "4 ":
            delivery = parse_delivery_som(line) if delivery is None else delivery
        elif rec == "B ":
            bp = parse_array_params(line)
            if (bp.exchange, bp.commodity, bp.product_type) in wanted:
                bparams[bp.series_key] = bp
        elif rec == "81":
            pending81 = line
        elif rec == "82":
            if pending81 is None or cc is None:
                continue
            key = (_text(line, "82", "exchange"), _text(line, "82", "commodity"), _text(line, "82", "product_type"))
            if key in wanted:
                pp = price_params.get(key)
                arrays.append(parse_risk_array_pair(pending81, line, pp.strike_decimals if pp else 0,
                                                    pp.settlement_decimals if pp else 0, cc.risk_scale))
            pending81 = None
        elif rec == "6 " and cc is not None:
            sp = parse_inter_spread(line)
            if any(leg.code == cc.code for leg in sp.legs):
                inter.append(sp)

    if header is None:
        raise ValueError(f"{path}: no Type 0 header record")
    if cc is None:
        raise ValueError(f"{path}: no combined commodity lists product {commodity!r}" + (f" on {exchange}" if exchange else ""))
    if any(f.decimal_locator for f in cc.families):
        notes.append("a product family in this combined commodity uses Type 83/84 (float) arrays, which are not parsed")
    spreads = tuple(sorted(series, key=lambda s: s.priority)) + tuple(sorted(tiers_spreads, key=lambda s: s.priority))
    return SpanData(header, cc, {k: v for k, v in price_params.items() if k in wanted}, intra_tiers, scanning, spreads, delivery,
                    bparams, arrays, tuple(inter), tuple(notes))


def extract_slice(src: str | Path, dst: str | Path, commodity: str, exchange: str | None = None,
                  families: Iterable[str] | None = None) -> int:
    """Copy the lines `load_commodity` would use (header, Type 1, the families' P records, the combined
    commodity's parameter block, its 81/82 pairs and the Type 6 spreads naming it) to `dst`. Returns the line
    count. Used to cut a private test slice out of a full exchange file; the slice keeps CME's byte layout."""
    keep_fams = None if families is None else {f.strip() for f in families}
    cc: CombinedCommodity | None = None
    wanted: set[tuple[str, str, str]] = set()
    in_block = False
    out: list[str] = []
    pending81: str | None = None
    plines: dict[tuple[str, str, str], str] = {}

    for line in iter_lines(src):
        rec = line[:2]
        if rec in ("0 ", "1 "):
            out.append(line)
        elif rec == "P ":
            plines[(_text(line, "P ", "exchange"), _text(line, "P ", "commodity"), _text(line, "P ", "product_type"))] = line
        elif rec == "2 ":
            part = parse_combined_commodity(line)
            if cc is not None and part.code == cc.code and part.exchange == cc.exchange:
                cc = CombinedCommodity(cc.exchange, cc.code, cc.risk_exponent, cc.pb_currency, cc.option_margin_style,
                                       cc.limit_option_value, cc.families + part.families)
                out.append(line)
            elif cc is None and any(f.commodity == commodity for f in part.families) and (exchange is None or part.exchange == exchange):
                cc, in_block = part, True
                out.append(line)
            else:
                in_block = False
            if cc is not None:
                wanted = {f.key for f in cc.families if keep_fams is None or f.commodity in keep_fams}
        elif rec in ("S ", "3 ", "E ", "C ", "4 "):
            if in_block:
                out.append(line)
        elif rec == "B ":
            if in_block and (_text(line, "B ", "exchange"), _text(line, "B ", "commodity"), _text(line, "B ", "product_type")) in wanted:
                out.append(line)
        elif rec == "81":
            pending81 = line
        elif rec == "82":
            key = (_text(line, "82", "exchange"), _text(line, "82", "commodity"), _text(line, "82", "product_type"))
            if pending81 is not None and key in wanted:
                out.extend([pending81, line])
            pending81 = None
        elif rec == "6 " and cc is not None:
            if any(leg.code == cc.code for leg in parse_inter_spread(line).legs):
                out.append(line)
    if cc is None:
        raise ValueError(f"{src}: no combined commodity lists product {commodity!r}")
    # P records precede the Type 2 block in CME files; put the wanted ones right after the header/exchange lines
    head = [x for x in out if x[:2] in ("0 ", "1 ")]
    rest = [x for x in out if x[:2] not in ("0 ", "1 ")]
    lines = head + [plines[k] for k in sorted(wanted) if k in plines] + rest
    Path(dst).write_text("\n".join(lines) + "\n", encoding="latin-1")
    return len(lines)
