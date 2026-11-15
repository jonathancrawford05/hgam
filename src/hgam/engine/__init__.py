"""The GAM engine moved from polaris-re (milestone M0).

Most of it was re-derived and verified against mgcv in polaris-re; six functions were ported
from mgcv source and name their counterparts (``bam_fitd``, ``_fit_chol``,
``initial_log10_lambda_from_xx_diag``, ``compress_covariates``, ``default_m``, ``_tero``).
Later milestones replace the re-derived pieces with direct ports where they leave mgcv's path
(docs/PLAN.md). The public entry points ``hgam.gam`` and ``hgam.bam`` do not call it yet.

The code moved as it was: ``ADR-nnn`` numbers, ``docs/...`` paths and "Polaris" in docstrings
refer to polaris-re's records, not this repository's. ``api.gam`` takes a mapping of column
name to array instead of a Polars DataFrame, so the core dependencies stay NumPy and SciPy.
"""
