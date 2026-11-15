"""Public entry points. mgcv counterparts: ``gam`` (mgcv.r) and ``bam`` (bam.r).

Nothing is fitted yet: milestone M0 moves the verified engine across, and the
functions below start working term by term from M1. See docs/PLAN.md.
"""

from typing import Any

_NOT_YET = "hgam.{name} is not implemented yet: see docs/PLAN.md for the milestone that delivers it."


def gam(formula: str, data: Any, **kwargs: Any) -> Any:
    """Fit a GAM by REML. mgcv counterpart: ``mgcv::gam``."""
    raise NotImplementedError(_NOT_YET.format(name="gam"))


def bam(formula: str, data: Any, **kwargs: Any) -> Any:
    """Fit a large GAM by fREML. mgcv counterpart: ``mgcv::bam``."""
    raise NotImplementedError(_NOT_YET.format(name="bam"))
