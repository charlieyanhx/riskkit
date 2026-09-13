"""On the real CME file slice (gitignored; skipped when absent): the column offsets pinned to CME's bytes,
and the reconciliation that the README quotes. Source file: cme.20250912.c.pa2 (EOD cycle, anonymous FTP
archive), sliced to the CX-GC combined commodity with `riskkit span-slice`."""

from pathlib import Path

import pytest

from riskkit import span as sp
from riskkit import span_files as sf

SLICE = Path(__file__).resolve().parents[1] / "tests/fixtures/private_span_gc_20250912.pa2"
pytestmark = pytest.mark.skipif(not SLICE.exists(), reason="private CME slice not present")

GC_SEP_2025 = (0, 0, -5333, -5333, 5333, 5333, -10667, -10667, 10667, 10667, -16000, -16000, 16000, 16000, -15840, 15840)


@pytest.fixture(scope="module")
def data():
    return sf.load_commodity(SLICE, "GC")


def test_gc_september_2025_array_is_the_published_one(data):
    a = data.find("GC", "FUT", 202509)
    assert a.values == GC_SEP_2025 and a.composite_delta == 1.0 and a.settlement_price == 3649.4


def test_gc_outright_scan_risk_equals_the_published_margin(data):
    pub = sp.span2_target("GC")
    for m in sorted({a.contract_month for a in data.arrays if a.commodity == "GC"})[:6]:
        for q in (1, -1):
            r = sp.compute(data, [sp.Position("GC", "FUT", m, q)])
            assert r.scan_risk == pub.long_margin == 16000, (m, q, r.scan_risk)


def test_extreme_scenarios_are_a_third_of_three_times_scan(data):
    a = data.find("GC", "FUT", 202509)
    assert a.values[14] == -15840 and a.values[15] == 15840 and 15840 == round(0.33 * 3 * 16000)


def test_og_put_signs_delta_and_vol(data):
    put = data.find("OG", "OOF", 202512, "P", 3700.0)
    assert put.values[0] < 0 < put.values[1]            # vol up helps a long put, vol down hurts
    assert put.values[2] > 0 > put.values[4]            # price up hurts, price down helps
    assert put.composite_delta == -0.4752 and put.implied_vol == 0.152551      # 82 cols 97-102 (4 dp) and 103-110 (6 dp)
    assert put.settlement_price == put.hp_settlement_price == 107.4 and put.has_settlement
    assert data.find("OG", "OOF", 202512, "C", 3700.0).composite_delta == 0.5129


def test_og_put_requirement_is_scan_plus_premium_owed_short_and_excess_long(data):
    """Short 1 OG Dec-25 3700 put: scan 12,878 (scenario 16) > SOM 35, plus the 10,740 premium owed (107.40 x 100)
    = 23,618. Long 1: scan 7,597 (scenario 12) minus the 10,740 premium held = -3,143, an excess."""
    short = sp.compute(data, [sp.Position("OG", "OOF", 202512, -1, "P", 3700.0)])
    assert (short.scan_risk, short.active_scenario, short.short_option_minimum) == (12878.0, 16, 35.0)
    assert short.span_requirement == 12878.0 and short.short_option_value == 10740.0 and short.long_option_value == 0.0
    assert short.total_requirement == 12878 + 10740 == 23618
    long = sp.compute(data, [sp.Position("OG", "OOF", 202512, 1, "P", 3700.0)])
    assert (long.scan_risk, long.active_scenario, long.long_option_value) == (7597.0, 12, 10740.0)
    assert long.total_requirement == 7597 - 10740 == -3143 < 0
    assert not any("no settlement price" in n for n in short.notes + long.notes)


def test_header_type_4_type_b_and_p_fields_are_pinned_on_the_real_bytes(data):
    """Fields no arithmetic exercises, checked against the file's own values: the Type 0 header, the Type 4 short
    option minimum ($35 per short option, summed over calls and puts), the Type B parameters of GC Sep-25 and
    the two P records (2 / 0 decimals, cvf 100, no alignment codes). Every contract in the slice has a settlement
    price, no day/week codes, and a unique identity."""
    h = data.header
    assert (h.exchange_complex, h.business_date, h.settlement_flag, h.file_identifier) == ("CME", "20250912", "S", "F")
    assert (h.file_format, h.account_type, h.pb_class, h.maint_or_init) == ("U2", "H", "1", "M")
    som = data.delivery_som
    assert (som.code, som.delivery_method, som.som_rate, som.som_method) == ("CX-GC", "01", 35.0, "2")
    assert (som.adj_member, som.adj_hedger, som.adj_speculator) == (1.0, 1.0, 1.0)
    b = data.array_params_for(data.find("GC", "FUT", 202509))
    assert (b.price_scan_range, b.extreme_move_multiplier, b.extreme_move_covered_fraction) == (16000.0, 3.0, 0.33)
    assert (b.expiration_date, b.delta_scaling, b.contract_value_factor, b.underlying, b.time_to_expiry) == ("20250926", 1.0, 100.0, "GC", 0.038356)
    assert data.combined.code == "CX-GC" and data.combined.risk_exponent == 0 and data.combined.option_margin_style == "P"
    assert data.intra_tiers.method == "10" and [t.number for t in data.intra_tiers.tiers] == [1, 2, 3, 4, 5]
    assert (data.intra_tiers.im_member, data.intra_tiers.im_hedger, data.intra_tiers.im_speculator) == (1.0, 1.0, 1.1)
    for key, dec in ((("CMX", "GC", "FUT"), (2, 0)), (("CMX", "OG", "OOF"), (2, 0))):
        pp = data.price_params[key]
        assert (pp.settlement_decimals, pp.strike_decimals, pp.contract_value_factor) == (*dec, 100.0)
        assert (pp.settlement_alignment, pp.strike_alignment, pp.strike_format) == ("", "", "")
    assert len(data.arrays) == len(data._index) == 22966
    assert all(a.has_settlement and a.contract_day == "" and a.option_day == "" for a in data.arrays)


def test_calendar_spread_charges_come_from_the_file(data):
    r = sp.compute(data, [sp.Position("GC", "FUT", 202509, 1), sp.Position("GC", "FUT", 202510, -1)])
    assert r.scan_risk == 0.0 and r.intra_spread_charge == 300.0
    r4 = sp.compute(data, [sp.Position("GC", "FUT", 202509, 1), sp.Position("GC", "FUT", 202606, -1)])
    assert r4.intra_spread_charge == 1000.0
