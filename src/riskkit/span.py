"""Legacy CME SPAN performance-bond engine for one combined commodity, and the published-margin reconciliation.

Method (CME Group, "CME SPAN: Standard Portfolio Analysis of Risk", methodology deck,
https://www.cmegroup.com/clearing/files/span-methodology.pdf, 2019 edition; the 2010 edition mirrored at
https://www.sfu.ca/~poitras/span-methodology.pdf has the same slides — scan risk slides 6-9, extreme scenarios
slide 10, composite delta slide 11, intra-commodity spreads slides 12-13, short option minimum slides 21-22,
the requirement formula slide 23, net option value slides 24-26):

    scan risk            = max over the 16 scenarios of  sum_positions  quantity x array value
                           (array values are loss-positive for LONG 1, so a short position flips the sign)
    intra-commodity      = sum over spreads formed of  n_spreads x charge rate
                           (delta by contract month; series-to-series (Type E) spreads first, then
                           tier-to-tier (Type C) spreads, each in priority order)
    delivery charge      = 0 here (Type 4 method "01" = no spot charge; method "10" is not implemented)
    inter-commodity      = 0 here (a single combined commodity cannot form one; the Type 6 records are
                           parsed and counted)
    SPAN requirement     = max(scan + intra + delivery - inter, short option minimum)
    net option value     = long option value - short option value   (premium-style options only)
    total requirement    = SPAN requirement - net option value

Units: the combined commodity's performance-bond currency (USD for the products here), per the file's
account type and maintenance/initial flag (the header says which; CME's public .c file is the maintenance
file). Quantities are signed contracts (+ long, - short). Delta is composite delta x quantity x the delta
scaling factor of the contract's Type B record (1.0 when absent). The short option minimum counts short
option contracts per the Type 4 method (blank/"2": short calls + short puts; "1": the greater of the two)
times the SOM charge rate. Extreme scenarios 15/16 are the file's own values (CME: 3 x the price scan
range, 33 % covered), used as-is. Every dollar figure is scaled by 10**risk_exponent of the Type 2 record.
An option whose settlement price the file does not carry (`RiskArray.settlement_price` NaN — CME's all-nines
field) is left out of the net option value and named in the result's notes; it is never priced at zero or
at the sentinel.

Not implemented (see docs/DESIGN.md): inter-commodity spread credits, the table-driven delivery charge,
intercommodity/scanning tiers other than "01", combination products, and SPAN 2 — whose parameter files are
not public; for SPAN 2 products the only public target is the published outright margin (`span2_target`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .span_files import PRODUCT_TYPES_OPTION, RiskArray, SpanData, load_commodity

SCENARIO_LABELS = (   # 1-16, the order of the risk array (CME Type 8 page)
    "unch / vol up", "unch / vol down", "up 1/3 / vol up", "up 1/3 / vol down", "down 1/3 / vol up", "down 1/3 / vol down",
    "up 2/3 / vol up", "up 2/3 / vol down", "down 2/3 / vol up", "down 2/3 / vol down", "up 3/3 / vol up", "up 3/3 / vol down",
    "down 3/3 / vol up", "down 3/3 / vol down", "up extreme x cover", "down extreme x cover",
)
_REPO = Path(__file__).resolve().parents[2]
PUBLISHED_CSV = _REPO / "data/span/published_2025-09-12.csv"


@dataclass(frozen=True)
class Position:
    """A signed position in one contract of the loaded combined commodity."""

    commodity: str
    product_type: str           # FUT / OOF / OOP
    contract_month: int         # CCYYMM of the future (for an option: its underlying future's month)
    quantity: float             # + long / - short, contracts
    right: str = ""             # "C" / "P" for options
    strike: float = 0.0
    option_month: int | None = None   # defaults to contract_month for options
    contract_day: str = ""      # futures day/week code, "" for a monthly contract
    option_day: str = ""        # option day/week code (dailies, weeklies, flex), "" for a monthly option


@dataclass(frozen=True)
class SpreadFormed:
    kind: str                   # "series" / "tier"
    priority: int
    legs: tuple[tuple[int, str], ...]   # (month or tier, side)
    n_spreads: float
    charge_rate: float
    charge: float


@dataclass(frozen=True)
class SpanResult:
    scan_risk: float
    active_scenario: int                   # 1-16
    scenario_losses: tuple[float, ...]     # 16 portfolio values, loss-positive
    composite_delta: float                 # net, all months
    delta_by_month: dict[int, float]
    intra_spread_charge: float
    spreads_formed: tuple[SpreadFormed, ...]
    delivery_charge: float
    inter_spread_credit: float
    short_option_minimum: float
    n_short_options: float
    span_requirement: float                # max(scan + intra + delivery - inter, SOM)
    long_option_value: float
    short_option_value: float
    net_option_value: float
    total_requirement: float               # span_requirement - net_option_value
    initial_to_maintenance: tuple[float, float, float]   # member, hedger, speculator
    currency: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def active_scenario_label(self) -> str:
        return SCENARIO_LABELS[self.active_scenario - 1]


def _resolve(data: SpanData, p: Position) -> RiskArray:
    return data.find(p.commodity, p.product_type, p.contract_month, p.right, p.strike, p.option_month, p.contract_day, p.option_day)


def _describe(p: Position) -> str:
    day = f"/{p.contract_day}" if p.contract_day else ""
    if p.product_type not in PRODUCT_TYPES_OPTION:
        return f"{p.commodity} {p.product_type} {p.contract_month}{day}"
    om = p.contract_month if p.option_month is None else p.option_month
    oday = f"/{p.option_day}" if p.option_day else ""
    return f"{p.commodity} {p.product_type} {p.contract_month}{day} {om}{oday} {p.right} {p.strike:g}"


def scenario_losses(data: SpanData, positions: list[Position]) -> tuple[float, ...]:
    """Portfolio loss under each of the 16 scenarios: sum quantity x array value (loss-positive for long 1)."""
    out = [0.0] * 16
    for p in positions:
        a = _resolve(data, p)
        for i, v in enumerate(a.values):
            out[i] += p.quantity * v
    return tuple(out)


def scan_risk(data: SpanData, positions: list[Position]) -> tuple[float, int]:
    """(largest scenario loss, its 1-based scenario number). Not floored: the requirement formula's max with
    the short option minimum (>= 0) does that."""
    losses = scenario_losses(data, positions)
    i = max(range(16), key=lambda k: losses[k])
    return losses[i], i + 1


def delta_by_month(data: SpanData, positions: list[Position]) -> dict[int, float]:
    """Net delta per futures contract month: quantity x composite delta x delta scaling factor."""
    out: dict[int, float] = {}
    for p in positions:
        a = _resolve(data, p)
        bp = data.array_params_for(a)
        scale = bp.delta_scaling if bp is not None else 1.0
        out[a.contract_month] = out.get(a.contract_month, 0.0) + p.quantity * a.composite_delta * scale
    return dict(sorted(out.items()))


def _tier_of(data: SpanData, month: int) -> int | None:
    if data.intra_tiers is None:
        return None
    for t in data.intra_tiers.tiers:
        if t.contains(month):
            return t.number
    return None


def _form_series(rem: dict[int, float], legs, charge_rate: float) -> tuple[float, float]:
    """Series-to-series spread: legs are contract months. Returns (n_spreads, charge)."""
    for orient in (1.0, -1.0):
        n = None
        for leg in legs:
            s = orient if leg.side == "A" else -orient
            d = rem.get(leg.period, 0.0) * s
            avail = d / leg.ratio if leg.ratio > 0 else 0.0
            n = avail if n is None else min(n, avail)
        if n is not None and n > 0:
            for leg in legs:
                s = orient if leg.side == "A" else -orient
                rem[leg.period] = rem.get(leg.period, 0.0) - s * n * leg.ratio
            return n, n * charge_rate
    return 0.0, 0.0


def _form_tier(rem: dict[int, float], tier_months: dict[int, list[int]], legs, charge_rate: float) -> tuple[float, float]:
    """Tier-to-tier spread: an "A" leg draws on the tier's long (positive remaining) delta pool and a "B" leg on its
    short pool, or the reverse orientation; consumed delta is taken from the tier's months front to back."""
    for orient in (1.0, -1.0):
        n = None
        for leg in legs:
            s = orient if leg.side == "A" else -orient
            pool = sum(max(rem.get(m, 0.0) * s, 0.0) for m in tier_months.get(leg.period, []))
            avail = pool / leg.ratio if leg.ratio > 0 else 0.0
            n = avail if n is None else min(n, avail)
        if n is not None and n > 0:
            for leg in legs:
                s = orient if leg.side == "A" else -orient
                need = n * leg.ratio
                for m in tier_months.get(leg.period, []):
                    have = max(rem.get(m, 0.0) * s, 0.0)
                    take = min(have, need)
                    rem[m] = rem.get(m, 0.0) - s * take
                    need -= take
                    if need <= 1e-12:
                        break
            return n, n * charge_rate
    return 0.0, 0.0


def intra_spread_charge(data: SpanData, positions: list[Position]) -> tuple[float, tuple[SpreadFormed, ...], dict[int, float]]:
    """Table-driven intracommodity spreading (Type 3 method "10"): Type E spreads then Type C spreads, each in
    priority order, on the remaining delta by month. Returns (charge, spreads formed, remaining delta)."""
    rem = delta_by_month(data, positions)
    formed: list[SpreadFormed] = []
    if data.intra_tiers is None or data.intra_tiers.method != "10":
        return 0.0, (), rem
    scale = data.combined.risk_scale
    tier_months: dict[int, list[int]] = {}
    for m in rem:
        t = _tier_of(data, m)
        if t is not None:
            tier_months.setdefault(t, []).append(m)
    total = 0.0
    for sp in data.intra_spreads:
        rate = sp.charge_rate * scale
        if sp.kind == "series":
            n, charge = _form_series(rem, sp.legs, rate)
        else:
            n, charge = _form_tier(rem, tier_months, sp.legs, rate)
        if n > 0:
            formed.append(SpreadFormed(sp.kind, sp.priority, tuple((leg.period, leg.side) for leg in sp.legs), n, rate, charge))
            total += charge
    return total, tuple(formed), rem


def short_option_minimum(data: SpanData, positions: list[Position]) -> tuple[float, float]:
    """(SOM charge, number of short options charged) per the Type 4 record's method."""
    if data.delivery_som is None:
        return 0.0, 0.0
    calls = sum(-p.quantity for p in positions if p.product_type in PRODUCT_TYPES_OPTION and p.right == "C" and p.quantity < 0)
    puts = sum(-p.quantity for p in positions if p.product_type in PRODUCT_TYPES_OPTION and p.right == "P" and p.quantity < 0)
    n = max(calls, puts) if data.delivery_som.som_method == "1" else calls + puts
    return n * data.delivery_som.som_rate * data.combined.risk_scale, n


def option_values(data: SpanData, positions: list[Position]) -> tuple[float, float, tuple[str, ...]]:
    """(long option value, short option value, unpriced) — value = settlement price x contract value factor x
    |quantity| for premium-style options; futures-style options (Type 2 style "F") carry no option value. An
    option whose array has no settlement price (NaN, CME's all-nines field) is excluded from both sums and named
    in `unpriced` with its quantity, so the caller sees what the net option value is missing."""
    if data.combined.option_margin_style == "F":
        return 0.0, 0.0, ()
    lov = sov = 0.0
    unpriced: list[str] = []
    for p in positions:
        if p.product_type not in PRODUCT_TYPES_OPTION:
            continue
        a = _resolve(data, p)
        if not a.has_settlement:
            unpriced.append(f"{p.quantity:+g} x {_describe(p)}")
            continue
        pp = data.price_params_for(a)
        cvf = pp.contract_value_factor if pp is not None else 1.0
        v = a.settlement_price * cvf * abs(p.quantity)
        if p.quantity > 0:
            lov += v
        else:
            sov += v
    return lov, sov, tuple(unpriced)


def compute(data: SpanData, positions: list[Position]) -> SpanResult:
    """Total performance bond for `positions` (all in `data.combined`), every component reported."""
    notes = list(data.notes)
    losses = scenario_losses(data, positions)
    scan, active = scan_risk(data, positions)
    intra, formed, _rem = intra_spread_charge(data, positions)
    dbm = delta_by_month(data, positions)
    delivery = 0.0
    if data.delivery_som is not None and data.delivery_som.delivery_method not in ("01", ""):
        notes.append(f"delivery charge method {data.delivery_som.delivery_method} not implemented; delivery charge set to 0")
    inter = 0.0
    notes.append(f"inter-commodity credit 0: single combined commodity ({len(data.inter_spreads)} Type 6 spreads name "
                 f"{data.combined.code} but need another combined commodity in the portfolio)")
    som, n_short = short_option_minimum(data, positions)
    if data.delivery_som is None:
        notes.append("no Type 4 record: short option minimum 0")
    req = max(scan + intra + delivery - inter, som)
    lov, sov, unpriced = option_values(data, positions)
    if unpriced:
        notes.append(f"net option value excludes {len(unpriced)} position(s) whose contract has no settlement price in the "
                     f"file (CME all-nines field): {'; '.join(unpriced)}")
    nov = lov - sov
    im = ((data.intra_tiers.im_member, data.intra_tiers.im_hedger, data.intra_tiers.im_speculator)
          if data.intra_tiers is not None else (1.0, 1.0, 1.0))
    return SpanResult(scan, active, losses, sum(dbm.values()), dbm, intra, formed, delivery, inter, som, n_short, req, lov, sov,
                      nov, req - nov, im, data.combined.pb_currency, tuple(notes))


# --- published margins (data) and reconciliation --------------------------------------------------------------
@dataclass(frozen=True)
class PublishedMargin:
    product: str
    exchange: str
    combined_commodity: str
    commodity: str
    tier: str
    date: str
    long_margin: float | None      # maintenance, outright, per contract, USD
    short_margin: float | None
    margin_model: str              # "SPAN" or "SPAN 2"
    model_since: str               # date the product moved to SPAN 2 ("" for legacy SPAN)
    source: str
    note: str


def load_published(path: str | Path = PUBLISHED_CSV) -> list[PublishedMargin]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(PublishedMargin(r["product"], r["exchange"], r["combined_commodity"], r["commodity"], r["tier"], r["date"],
                                        float(r["long_margin_usd"]) if r["long_margin_usd"] else None,
                                        float(r["short_margin_usd"]) if r["short_margin_usd"] else None,
                                        r["margin_model"], r["model_since"], r["source"], r["note"]))
    return rows


def span2_target(product: str, path: str | Path = PUBLISHED_CSV) -> PublishedMargin:
    """The published outright long/short margin for a product with its date and source — the only public
    target for a SPAN 2 product. Raises KeyError for a product not in the table."""
    for r in load_published(path):
        if r.product == product:
            return r
    raise KeyError(f"{product!r} is not in {path}")


def outright_table(data: SpanData, commodity: str, published: PublishedMargin | None = None,
                   months: int = 0) -> pd.DataFrame:
    """Scan risk of long 1 and short 1 for each futures contract month of `commodity` against the published
    margin. `error` is legacy minus published; `months` > 0 keeps only the first that many months."""
    futs = sorted(data.futures(commodity), key=lambda a: a.contract_month)
    if months > 0:
        futs = futs[:months]
    rows = []
    for a in futs:
        long_scan, i_long = scan_risk(data, [Position(commodity, "FUT", a.contract_month, 1.0, contract_day=a.contract_day)])
        short_scan, i_short = scan_risk(data, [Position(commodity, "FUT", a.contract_month, -1.0, contract_day=a.contract_day)])
        pl = published.long_margin if published else None
        ps = published.short_margin if published else None
        rows.append({
            "product": commodity, "combined_commodity": data.combined.code, "contract_month": a.contract_month,
            "contract_day": a.contract_day,
            "settlement": a.settlement_price, "scan_long_1": long_scan, "active_long": i_long, "scan_short_1": short_scan,
            "active_short": i_short, "published_long": pl, "published_short": ps,
            "error_long": None if pl is None else long_scan - pl, "error_pct_long": None if not pl else 100.0 * (long_scan - pl) / pl,
            "error_short": None if ps is None else short_scan - ps, "error_pct_short": None if not ps else 100.0 * (short_scan - ps) / ps,
            "margin_model": published.margin_model if published else "unknown",
        })
    return pd.DataFrame(rows)


def _money(x: float | None) -> str:
    return "unknown" if x is None else f"{x:,.0f}"


def _err(x: float | None, pct: float | None) -> str:
    return "unknown" if x is None else f"{x:+,.0f} ({pct:+.1f} %)"


def render_reconciliation(tables: list[tuple[SpanData, PublishedMargin | None, pd.DataFrame]]) -> str:
    """Markdown: header line, one table (all products), then per-product labels and sources."""
    if not tables:
        return "(no products)\n"
    h = tables[0][0].header
    lines = [f"SPAN file: exchange complex {h.exchange_complex}, business date {h.business_date}, cycle {h.file_identifier.strip() or '?'} "
             f"({'settlement' if h.settlement_flag == 'S' else 'intraday'}), format {h.file_format}, "
             f"{'maintenance' if h.maint_or_init != 'I' else 'initial'} rates, account type {h.account_type or 'default'}.", "",
             "| product | model | contract month | settle | scan long 1 | scan short 1 | published long | published short | error long | error short |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _data, pub, df in tables:
        model = pub.margin_model if pub else "unknown"
        for r in df.itertuples():
            lines.append(f"| {r.product} | {model} | {r.contract_month} | {r.settlement:,.2f} | {_money(r.scan_long_1)} | "
                         f"{_money(r.scan_short_1)} | {_money(r.published_long)} | {_money(r.published_short)} | "
                         f"{_err(r.error_long, r.error_pct_long)} | {_err(r.error_short, r.error_pct_short)} |")
    lines.append("")
    for data, pub, df in tables:
        n_all = len(data.futures(df["product"].iloc[0])) if len(df) else 0
        shown = len(df)
        prod = df["product"].iloc[0] if len(df) else "?"
        if pub is None:
            lines.append(f"- {prod} ({data.combined.code}): no published margin in the table; {shown} of {n_all} futures months shown.")
            continue
        tag = (f"SPAN 2 since {pub.model_since} — legacy arrays not authoritative" if pub.margin_model == "SPAN 2"
               else "legacy SPAN — arrays are the exchange's own")
        lines.append(f"- {prod} ({data.combined.code}): {tag}; published {pub.tier} {_money(pub.long_margin)} long / "
                     f"{_money(pub.short_margin)} short on {pub.date}, source: {pub.source}"
                     + (f"; {pub.note}" if pub.note else "") + f". {shown} of {n_all} futures months shown.")
    return "\n".join(lines) + "\n"


def reconcile(file: str | Path, commodities: list[str], published_path: str | Path = PUBLISHED_CSV,
              months: int = 0) -> tuple[str, list[tuple[SpanData, PublishedMargin | None, pd.DataFrame]]]:
    """Load each commodity from `file` (one streaming pass per commodity) and reconcile against the published table."""
    pubs = {p.product: p for p in load_published(published_path)}
    tables = []
    for c in commodities:
        pub = pubs.get(c)
        data = load_commodity(file, c, exchange=pub.exchange if pub else None, families=[c])
        tables.append((data, pub, outright_table(data, c, pub, months)))
    return render_reconciliation(tables), tables
