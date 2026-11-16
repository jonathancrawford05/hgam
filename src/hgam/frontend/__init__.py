"""The formula front end moved from polaris-re: formula parsing, ``gam()``, ``bam``'s
performance iteration and the fit summary.

``hgam.frontend.gam`` takes a mapping of NumPy columns. The top-level ``hgam.gam`` and
``hgam.bam`` stay unimplemented until the port reaches the exit test (docs/PLAN.md).
"""

from hgam.frontend.api import (
    ORACLE_COLLATION,
    Factor,
    GamFit,
    gam,
    r_factor_levels,
    structural_rank_deficiency,
)
from hgam.frontend.formula import ParsedFormula, parse_formula
from hgam.frontend.summary import GamSummary, SmoothSummary

__all__ = [
    "ORACLE_COLLATION",
    "Factor",
    "GamFit",
    "GamSummary",
    "ParsedFormula",
    "SmoothSummary",
    "gam",
    "parse_formula",
    "r_factor_levels",
    "structural_rank_deficiency",
]
