"""Public entry points. mgcv counterparts: ``gam`` (mgcv.r) and ``bam`` (bam.r), mgcv 1.9-4.

Both forward to :func:`hgam.frontend.api.gam`, which holds the verified subset: ``gam`` fits by exact
Laplace REML (``method="REML"``), ``bam`` by ``bgam.fitd``'s performance iteration on discretised
covariates (``method="fREML", discrete=TRUE``). Anything outside that subset is refused by name there.
"""

from collections.abc import Mapping, Sequence

from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.api import GamFit
from hgam.frontend.api import gam as _frontend_gam
from hgam.frontend.threads import blas_thread_limit, resolve_nthreads

__all__ = ["bam", "gam"]


def _family_with_link(family: str, link: str | None) -> str:
    """``family`` as the front end reads it: ``binomial`` with ``link="cloglog"`` becomes
    ``'binomial(link="cloglog")'``, as mgcv writes ``binomial(link = "cloglog")``."""
    if link is None:
        return family
    if "(" in family:
        raise HgamValidationError(
            f"link={link!r} was given beside a family that already names its arguments ({family!r}); use one."
        )
    return f'{family}(link="{link}")'


def gam(
    formula: str,
    data: Mapping[str, object],
    family: str = "gaussian",
    *,
    link: str | None = None,
    weights: str | None = None,
    offset: str | None = None,
    select: bool = False,
    knots: Mapping[str, Sequence[float]] | None = None,
    method: str = "GCV.Cp",
    control: Mapping[str, object] | None = None,
) -> GamFit:
    """Fit a GAM by REML. mgcv counterpart: ``gam`` (mgcv.r), mgcv 1.9-4.

    Arguments are mgcv's where they exist (``family`` as a string such as ``'binomial(link="cloglog")'``,
    or ``family="binomial", link="cloglog"``; ``weights`` and ``offset`` name columns of ``data``). Only
    ``method="REML"`` is ported for ``gam``, so it must be written out: mgcv's default, ``method="GCV.Cp"``,
    is refused by name rather than fitted by REML.

    ``control`` is mgcv's ``control = gam.control(...)`` as a mapping; only
    ``{"newton": {"conv.tol": value}}`` is ported (default ``1e-6``, as ``gam.control``), and every other
    entry is refused by name.
    """
    if method != "REML":
        raise HgamNotImplementedError(
            f"hgam.gam(): method={method!r} is not ported; only method='REML' is, and it must be given "
            "explicitly (mgcv's default is method='GCV.Cp'). Use hgam.bam for fREML."
        )
    return _frontend_gam(
        formula,
        data,
        _family_with_link(family, link),
        weights=weights,
        offset=offset,
        select=select,
        knots=knots,
        method="REML",
        discrete=False,
        control=control,
    )


def bam(
    formula: str,
    data: Mapping[str, object],
    family: str = "gaussian",
    *,
    link: str | None = None,
    weights: str | None = None,
    offset: str | None = None,
    select: bool = False,
    knots: Mapping[str, Sequence[float]] | None = None,
    method: str = "fREML",
    discrete: bool = False,
    nthreads: int = 1,
) -> GamFit:
    """Fit a large GAM by fREML. mgcv counterpart: ``bam`` (bam.r), mgcv 1.9-4.

    Arguments as :func:`gam`. Only ``method="fREML", discrete=True`` is ported (``bgam.fitd`` on
    covariates binned as ``discrete.mf`` bins them); mgcv's default ``discrete=FALSE`` path
    (``bgam.fit``) is refused by name.

    ``nthreads`` (default 1, as in ``bam``) caps the BLAS thread pool for the length of the fit, through
    the optional extra ``hgam[threads]`` (:mod:`hgam.frontend.threads`). It maps mgcv's argument name onto
    BLAS: mgcv's ``nthreads`` drives mgcv's own OpenMP code, which hgam does not have. The cap is
    process-wide while the fit runs and is restored afterwards. Without the extra, ``nthreads=1`` leaves
    BLAS as the process set it and other values are refused; the two-element form is refused.
    """
    if (method, discrete) != ("fREML", True):
        raise HgamNotImplementedError(
            f"hgam.bam(): method={method!r}, discrete={discrete!r} is not ported; only method='fREML' with "
            "discrete=True is (bgam.fitd). bam's discrete=FALSE path (bgam.fit) is not ported."
        )
    n_threads = resolve_nthreads(nthreads)
    with blas_thread_limit(n_threads):
        return _frontend_gam(
            formula,
            data,
            _family_with_link(family, link),
            weights=weights,
            offset=offset,
            select=select,
            knots=knots,
            method="fREML",
            discrete=True,
        )
