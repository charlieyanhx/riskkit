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
    assert -1.0 < put.composite_delta < 0 and 0.05 < put.implied_vol < 1.0 and put.settlement_price == 107.4


def test_calendar_spread_charges_come_from_the_file(data):
    r = sp.compute(data, [sp.Position("GC", "FUT", 202509, 1), sp.Position("GC", "FUT", 202510, -1)])
    assert r.scan_risk == 0.0 and r.intra_spread_charge == 300.0
    r4 = sp.compute(data, [sp.Position("GC", "FUT", 202509, 1), sp.Position("GC", "FUT", 202606, -1)])
    assert r4.intra_spread_charge == 1000.0
