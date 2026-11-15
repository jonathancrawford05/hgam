# Continuation

Last updated: 2026-11-15, by the development run that moved the engine. Read `docs/PRIORITIES.md` before
choosing work.

## State

M0 is done: the polaris-re GAM engine is in `src/hgam/engine/` (bases, families, fit, REML and its
derivatives, smoothing-parameter search, covariance, prediction, `gam_term_build.py` with the term
builders taken out of polaris-re's `gam_stage_a.py`) and `src/hgam/frontend/` (`api.py`, `formula.py`,
`summary.py`, `bam_fit.py`, `bam_fitd.py`, `discrete.py` with `compress.df`). Errors are in `hgam.errors`.
Nothing imports `polaris_re`. The six mgcv-ported functions carry `mgcv counterpart:` lines naming mgcv
1.9-1, the version they were read from: `bam_fitd`, `_fit_chol`, `initial_log10_lambda_from_xx_diag`,
`compress_covariates`, `default_m`, `_tero`.

The front end takes a mapping of NumPy columns plus `hgam.frontend.Factor`. polars was dropped, not
added as a dependency. The top-level `hgam.gam` / `hgam.bam` are still stubs, so the exit test stays a
strict xfail. Tests: 564 pass, 1 strict xfail. What did not move is listed in `docs/DEFERRED.md`.

The engine has no hgam fixture tests yet: its moved tests are closed-form and self-consistency checks.
Each M1 slice that re-ports a piece adds its fixture to `oracle/make_fixtures.R`.

## Next slice: M1, first slice

Start "After the move" in `docs/PLAN.md`: the outer smoothing-parameter search. Port mgcv's `newton`
(gam.fit3.r, at the pinned version 1.9-4) in place of `hgam.engine.gam_reml_newton.newton_select_lambdas`,
keeping its function boundary. Add a `gam(..., method="REML")` fixture to `oracle/make_fixtures.R` for a
small cr + ti binomial-cloglog model with `sp` and `edf` per term, then test the linear predictor and edf
against the gates. `gam_poisson_offset_reml.json` is already a REML fixture you can reuse.

## Open items for Jonathan

- The `fixtures-tier3` artifact (CI run 43727740202 on `main`) could not be downloaded here: the
  network policy blocks the artifact host (`productionresultssa32.blob.core.windows.net`). The
  committed fixtures are still tier 1. Either allow that host in the environment, or commit the
  artifact by hand.

## Open questions

None.
