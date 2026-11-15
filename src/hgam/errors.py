"""Exceptions and warnings raised by hgam.

These replace polaris-re's ``PolarisValidationError``, ``PolarisComputationError`` and
``PolarisRankDeficiencyWarning``, which the engine used when it lived there.
"""

__all__ = ["HgamComputationError", "HgamError", "HgamRankDeficiencyWarning", "HgamValidationError"]


class HgamError(Exception):
    """Base class for hgam errors."""


class HgamValidationError(HgamError, ValueError):
    """An input or specification is invalid: bad arguments, inconsistent shapes, an impossible model."""


class HgamComputationError(HgamError, ArithmeticError):
    """A numerical computation failed: no convergence, a singular system, a non-finite result."""


class HgamRankDeficiencyWarning(UserWarning):
    """A model contains coefficients the data cannot identify.

    The fit is valid and its fitted values, total edf and ``Vp``-based standard errors are well defined,
    but quantities that depend on which redundant coefficient is eliminated (per-term edf, the
    unconditional covariance) are not identified; mgcv itself does not reproduce them between runs.
    """
