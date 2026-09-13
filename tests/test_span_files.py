"""Parser tests on the synthetic .pa2 fixture (always run) — offsets, signs, selection, streaming.

The fixture is written by `span_synth` with hand-placed CME columns and read back by `span_files`' offset
table; the two were written from the same CME layout pages but independently, so a field that round-trips
here is at the column both agree on. The real-file check that pins the columns to CME's own bytes is in
test_span_private.py (skipped when the private slice is absent).
"""

import filecmp
import zipfile
from pathlib import Path

import pytest

from riskkit import span_files as sf
from riskkit.cli import main
from riskkit.span_synth import (
    CC,
    CC_OTHER,
    EXCHANGE,
    FUT,
    OPT,
    OTHER,
    SETTLE_DECIMALS,
    STRIKE_DECIMALS,
    render,
    synthetic,
    write_synthetic,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/span_synthetic.pa2"


@pytest.fixture(scope="module")
def data() -> sf.SpanData:
    return sf.load_commodity(FIXTURE, FUT)


def test_synthetic_fixture_regenerates_byte_identically(tmp_path):
    a = write_synthetic(tmp_path / "a.pa2")
    main(["span-fixture", "--out", str(tmp_path / "b.pa2")])
    assert filecmp.cmp(a, tmp_path / "b.pa2", shallow=False)
    assert filecmp.cmp(a, FIXTURE, shallow=False), "tests/fixtures/span_synthetic.pa2 differs from the generator; run `riskkit span-fixture`"
    assert render(synthetic()) == FIXTURE.read_text(encoding="latin-1")


def test_every_record_round_trips_from_the_generator(data):
    """Header, P, 2, S, 3, E, C, 4, B, 81/82 and 6: the parsed records equal the generator's spec field for field."""
    spec = synthetic()
    assert data.header == spec.header
    assert data.combined == spec.combined
    assert data.scanning == spec.scanning
    assert data.intra_tiers == spec.intra_tiers
    assert data.intra_spreads == spec.intra_spreads              # series first, then tier, each by priority
    assert data.delivery_som == spec.delivery_som
    assert set(data.price_params.values()) == {p for p in spec.price_params if p.commodity in (FUT, OPT)}
    assert set(data.array_params.values()) == set(spec.array_params)
    assert data.arrays == list(spec.arrays)
    assert data.inter_spreads == spec.inter_spreads
    assert data.notes == ()


def test_selection_keeps_one_combined_commodity_and_streams_past_the_rest(data):
    """Selecting ZZ loads CC-ZZ (its options included) and none of CC-YY; selecting YY loads the other."""
    assert data.combined.code == CC and {a.commodity for a in data.arrays} == {FUT, OPT}
    other = sf.load_commodity(FIXTURE, OTHER)
    assert other.combined.code == CC_OTHER and [a.commodity for a in other.arrays] == [OTHER]
    assert other.intra_tiers.method == "01" and other.intra_spreads == () and other.delivery_som.som_rate == 25.0
    only_fut = sf.load_commodity(FIXTURE, FUT, families=[FUT])
    assert {a.commodity for a in only_fut.arrays} == {FUT} and only_fut.combined == data.combined
    with pytest.raises(ValueError, match="no combined commodity"):
        sf.load_commodity(FIXTURE, "QQ")
    with pytest.raises(ValueError, match="no combined commodity"):
        sf.load_commodity(FIXTURE, FUT, exchange="NOPE")


def test_zip_input_reads_the_pa2_member(tmp_path, data):
    z = tmp_path / "syn.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(FIXTURE, "cme.synthetic.c.pa2")
    from_zip = sf.load_commodity(z, FUT)
    assert from_zip.arrays == data.arrays and from_zip.header == data.header and from_zip.intra_spreads == data.intra_spreads


def test_array_value_columns_and_trailing_sign():
    """Value k of an 81 line sits at columns 55+6(k-1)..59+6(k-1) with its sign in the next column (CME Type 8
    page); a trailing '-' parses negative, '+' positive. Checked on a hand-typed pair, not on generator output."""
    ident = "81XYZZZ        ZZ        FUT 202512            0000000"
    assert len(ident) == 54
    l81 = ident + "00000+00000+04000-04000-04000+04000+08000-08000-08000+" + "00000000364000" + "N"
    l82 = "82" + ident[2:] + "08000+12000-12000-12000+12000+11880-11880+" + "10000+" + "00000000" + "0364000" + "+" + "+" + "10000+" + "C"
    a = sf.parse_risk_array_pair(l81, l82, strike_decimals=0, settlement_decimals=2)
    assert a.values == (0, 0, -4000, -4000, 4000, 4000, -8000, -8000, 8000, 8000, -12000, -12000, 12000, 12000, -11880, 11880)
    assert l81[54:59] == "00000" and l81[59] == "+" and l81[66:71] == "04000" and l81[71] == "-"   # cols 55-59 / 60, 67-71 / 72
    assert a.composite_delta == 1.0 and a.settlement_price == 3640.0 and a.hp_settlement_price == 3640.0 and a.delta_flag == "C"
    assert sf.LAYOUT["81"]["v1"] == (55, 59) and sf.LAYOUT["81"]["v9"] == (103, 107)
    assert sf.LAYOUT["82"]["v10"] == (55, 59) and sf.LAYOUT["82"]["v16"] == (91, 95) and sf.LAYOUT["82"]["composite_delta"] == (97, 101)


def test_sign_convention_is_loss_positive_for_long_and_puts_have_negative_delta(data):
    """A long call's 'up 3/3 / vol up' value (11) is negative (a gain), its 'down 3/3 / vol down' (14) positive; a
    put's composite delta is negative; a future's is +1."""
    call = data.find(OPT, "OOF", 202512, "C", 3600.0)
    put = data.find(OPT, "OOF", 202512, "P", 3600.0)
    assert call.values[10] < 0 < call.values[13] and put.values[12] < 0 < put.values[11]
    assert call.composite_delta > 0 > put.composite_delta and data.find(FUT, "FUT", 202512).composite_delta == 1.0


def test_pair_mismatch_and_wrong_order_raise():
    spec = synthetic()
    lines = render(spec).splitlines()
    l81 = [x for x in lines if x.startswith("81")]
    l82 = [x for x in lines if x.startswith("82")]
    with pytest.raises(ValueError, match="identity mismatch"):
        sf.parse_risk_array_pair(l81[0], l82[1])
    with pytest.raises(ValueError, match="expected an 81"):
        sf.parse_risk_array_pair(l82[0], l81[0])


def test_short_lines_and_unknown_record_types_are_tolerated(tmp_path):
    """CME truncates trailing blanks: a header cut after the business date still parses (missing fields blank),
    and record types the loader does not know (T, X, Y, Z, 5) are skipped."""
    text = FIXTURE.read_text(encoding="latin-1").splitlines()
    text[0] = text[0][:16]
    text.insert(3, "X M31        0000000")
    text.insert(3, "5 ABC      ZZ    OZ")
    p = tmp_path / "cut.pa2"
    p.write_text("\n".join(text) + "\n", encoding="latin-1")
    d = sf.load_commodity(p, FUT)
    assert d.header.business_date == "20250912" and d.header.maint_or_init == "" and d.header.account_type == ""
    assert len(d.arrays) == len(synthetic().arrays)


def test_decimal_locators_and_option_month_keying(data):
    """Strike and settlement use the P record's locators (0 and 2 here); an option keyed by its underlying
    future's month and its own option month resolves, and `find` defaults option_month to the futures month."""
    nov_call = data.find(OPT, "OOF", 202512, "C", 3700.0, option_month=202511)
    assert nov_call.option_month == 202511 and nov_call.settlement_price == 60.0 and nov_call.strike == 3700.0
    dec_call = data.find(OPT, "OOF", 202512, "C", 3600.0)
    assert dec_call.option_month == 202512 and dec_call.settlement_price == 120.0
    with pytest.raises(KeyError, match="no risk array"):
        data.find(OPT, "OOF", 202512, "C", 3700.0)
    pp = data.price_params_for(dec_call)
    assert (pp.strike_decimals, pp.settlement_decimals, pp.contract_value_factor) == (STRIKE_DECIMALS, SETTLE_DECIMALS, 100.0)
    assert data.price_params[(EXCHANGE, FUT, "FUT")].settlement_decimals == 2


def test_extract_slice_reproduces_the_loadable_subset(tmp_path):
    out = tmp_path / "slice.pa2"
    n = sf.extract_slice(FIXTURE, out, FUT, families=[FUT, OPT])
    assert n > 0
    sliced = sf.load_commodity(out, FUT)
    full = sf.load_commodity(FIXTURE, FUT)
    assert sliced.arrays == full.arrays and sliced.intra_spreads == full.intra_spreads and sliced.inter_spreads == full.inter_spreads
    assert sliced.combined == full.combined and sliced.header == full.header and sliced.price_params == full.price_params
    lines = out.read_text(encoding="latin-1").splitlines()
    assert not any(x.startswith("81XYZYY") for x in lines) and not any(x.startswith("2 XYZ CC-YY") for x in lines)
