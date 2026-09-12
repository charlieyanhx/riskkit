"""riskkit — portfolio risk for a futures-and-options book.

VaR / ES by historical simulation, parametric delta-gamma-vega (Cornish-Fisher), Monte Carlo
and filtered historical simulation (GARCH via `arch`), all with full revaluation where the
method allows it; a backtest suite with known size and power; stress and reverse stress;
explicit limit-down / empty-book / stale-quote semantics.
"""

from .backtest import (
    TestResult,
    acerbi_szekely_z2,
    christoffersen_cc,
    christoffersen_independence,
    engle_manganelli_dq,
    kupiec_pof,
    kupiec_power,
    kupiec_size,
    suite,
    traffic_light,
)
from .edge_cases import Flag, Lock, Quote, RiskAssessment, assess
from .positions import Book, LegMark, Market, mark
from .stress import HISTORICAL_SCENARIOS, ladder, reverse_stress_spot, reverse_stress_vol
from .var import VaRResult, all_methods, fhs_var, historical_var, monte_carlo_var, parametric_var

__version__ = "0.1.1"
__all__ = [
    "Book", "LegMark", "Market", "mark", "VaRResult", "historical_var", "parametric_var", "monte_carlo_var", "fhs_var",
    "all_methods", "TestResult", "kupiec_pof", "kupiec_size", "kupiec_power", "christoffersen_independence",
    "christoffersen_cc", "traffic_light", "acerbi_szekely_z2", "engle_manganelli_dq", "suite", "HISTORICAL_SCENARIOS",
    "ladder", "reverse_stress_spot", "reverse_stress_vol", "Quote", "Lock", "Flag", "RiskAssessment", "assess",
]
