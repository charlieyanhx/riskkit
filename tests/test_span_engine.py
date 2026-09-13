"""SPAN engine on the synthetic fixture (always runs): scan-risk identities, spread charges, SOM, NOV signs."""

from pathlib import Path

import pytest

from riskkit import span as sp
from riskkit import span_files as sf
from riskkit.span_synth import COVER, EXTREME_MULT, FUT, OPT, SCAN

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/span_synthetic.pa2"


@pytest.fixture(scope="module")
def data():
    return sf.load_commodity(FIXTURE, FUT)


def _m(data):
    return sorted({a.contract_month for a in data.arrays if a.commodity == FUT})


def test_long_and_short_outright_scan_to_the_scan_range_on_opposite_scenarios(data):
    m = _m(data)[0]
    long = sp.compute(data, [sp.Position(FUT, "FUT", m, 1)])
    short = sp.compute(data, [sp.Position(FUT, "FUT", m, -1)])
    assert long.scan_risk == short.scan_risk == SCAN
    assert {long.active_scenario, short.active_scenario} == {11, 13}      # up 3/3 hurts the short, down 3/3 the long
    assert long.scenario_losses == tuple(-x for x in short.scenario_losses)   # arrays are negated for a short
    assert long.total_requirement == SCAN and long.net_option_value == 0.0


def test_extreme_scenarios_are_cover_times_multiplier_times_scan(data):
    m = _m(data)[0]
    r = sp.compute(data, [sp.Position(FUT, "FUT", m, 1)])
    assert r.scenario_losses[15] == pytest.approx(COVER * EXTREME_MULT * SCAN)   # extreme down, long loses
    assert r.scenario_losses[14] == pytest.approx(-COVER * EXTREME_MULT * SCAN)


def test_scan_risk_is_positively_homogeneous_and_flat_book_scans_to_zero(data):
    m = _m(data)[0]
    one = sp.compute(data, [sp.Position(FUT, "FUT", m, 1)])
    five = sp.compute(data, [sp.Position(FUT, "FUT", m, 5)])
    assert five.scan_risk == pytest.approx(5 * one.scan_risk)
    flat = sp.compute(data, [sp.Position(FUT, "FUT", m, 3), sp.Position(FUT, "FUT", m, -3)])
    assert flat.scan_risk == 0.0 and flat.total_requirement == 0.0


def test_calendar_spread_scans_to_zero_and_pays_the_spread_charge(data):
    m1, m2 = _m(data)[:2]
    r = sp.compute(data, [sp.Position(FUT, "FUT", m1, 1), sp.Position(FUT, "FUT", m2, -1)])
    assert r.scan_risk == 0.0
    assert r.intra_spread_charge > 0 and len(r.spreads_formed) >= 1
    assert r.span_requirement == pytest.approx(r.intra_spread_charge)
    rates = {s.charge_rate for s in data.intra_spreads}
    assert r.intra_spread_charge in rates or r.intra_spread_charge == pytest.approx(sum(f.charge for f in r.spreads_formed))


def test_short_option_minimum_floors_a_far_out_of_the_money_short(data):
    opt = [a for a in data.arrays if a.commodity == OPT and a.right == "P"]
    far = min(opt, key=lambda a: a.strike)
    r = sp.compute(data, [sp.Position(OPT, "OOF", far.contract_month, -1, "P", far.strike, far.option_month)])
    assert r.short_option_minimum > 0 and r.n_short_options == 1
    assert r.span_requirement >= r.short_option_minimum


def test_net_option_value_sign(data):
    opt = [a for a in data.arrays if a.commodity == OPT and a.right == "C"]
    a = max(opt, key=lambda a: a.settlement_price)
    long = sp.compute(data, [sp.Position(OPT, "OOF", a.contract_month, 1, "C", a.strike, a.option_month)])
    short = sp.compute(data, [sp.Position(OPT, "OOF", a.contract_month, -1, "C", a.strike, a.option_month)])
    assert long.net_option_value > 0 > short.net_option_value
    assert long.net_option_value == pytest.approx(-short.net_option_value)
    assert long.total_requirement == pytest.approx(long.span_requirement - long.net_option_value)   # a long option's value is a credit
    assert short.total_requirement > short.span_requirement                                       # a short option's premium is owed


def test_published_table_and_span2_target():
    pub = sp.load_published()
    gc = sp.span2_target("GC")
    assert gc.long_margin == gc.short_margin == 16000 and gc.margin_model == "SPAN"
    es = sp.span2_target("ES")
    assert es.margin_model == "SPAN 2" and es.long_margin == 20936 and es.short_margin == 20051
    assert {p.product for p in pub} >= {"GC", "ES", "CL"}
