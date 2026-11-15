# Continuation

Last updated: 2026-11-15, by the setup session.

## State

Scaffold only. `hgam.gam` and `hgam.bam` raise `NotImplementedError`. Tests: 5 pass, 1 strict xfail
(the exit test). Fixtures are tier 1 (local R 4.3.3, mgcv 1.9-1): provisional.

## Next slice: M0, move the engine

See `docs/PLAN.md`, "M0, second slice". Take it in steps of one pull request each:

1. Bases and families: `gam_basis_cr`, `gam_basis_tp`, `gam_basis_re`, `gam_basis_parametric`, `gam_family`,
   `gam_term_spec`, with their tests. No import from `polaris_re`.
2. Fitting core: `gam_fit`, `gam_reml*`, `gam_derivatives`, `gam_initial_sp`, `gam_dispersion`, `gam_model`.
3. Inference and front end: `gam_predict`, `gam_vcov`, `gam_uncertainty`, `gam_select_penalty`,
   `gam_rank_pivot`, and `gam/api.py`, `formula.py`, `summary.py`, `bam_fit.py`, `bam_fitd.py`.

polaris-re is public: clone it read-only into a scratch directory. Replace its `PolarisValidationError` and
`PolarisComputationError` with `hgam.errors` equivalents; drop the `core.verification` claim objects. The six
functions ported from mgcv source gain their counterpart lines. Each step is done when the moved tests pass here.

## Routines

- Daily development: `docs/routines/DAILY_DEV.md`. Review: `docs/routines/PR_REVIEW.md`. Merging:
  `docs/routines/SETTINGS.md` (`auto_merge` is off until Jonathan turns it on).
- Both run as Claude Code scheduled tasks bound to sessions with this repository attached.

## Open items for Jonathan

1. Run the `oracle-fixtures` workflow once and commit its tier 3 fixtures in place of the tier 1 ones.
   The image is pinned to R-Gam-base build 10 (R 4.6.1, mgcv 1.9.4).
2. Decide `auto_merge` in `docs/routines/SETTINGS.md`.

## Open questions

None.
