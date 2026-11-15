"""Errors and warnings raised by hgam.

They replace the ``PolarisValidationError`` / ``PolarisComputationError`` /
``PolarisRankDeficiencyWarning`` of the polaris-re engine this package started from.
``ValidationError`` subclasses ``ValueError`` so callers can catch it as bad input.
"""

__all__ = ["ComputationError", "HgamError", "RankDeficiencyWarning", "ValidationError"]


class HgamError(Exception):
    """Base class for hgam errors."""


class ValidationError(HgamError, ValueError):
    """An input, a model specification or an option is invalid or not supported."""


class ComputationError(HgamError, ArithmeticError):
    """A numerical step failed: a solver did not converge, a matrix was not positive definite."""


class RankDeficiencyWarning(UserWarning):
    """A model has coefficients the data cannot identify.

    The fit is valid and its fitted values, total edf and ``Vp``-based standard errors are
    well defined, but quantities that depend on which redundant coefficient is eliminated
    (per-term edf, the unconditional covariance) are not identified; mgcv does not
    reproduce them between runs either.
    """
