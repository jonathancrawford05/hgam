# Plan

About 40 pull-request-sized slices. The slice counts are rough estimates.

| Milestone | What | Slices |
|---|---|---|
| M0 Scaffold | This repository; then move the verified engine across from polaris-re | 2 |
| M1 gam core | `gam.setup`, `gam.fit3`, `newton`, REML, the derivative code (`gdi.c`) | 12 |
| M2 Bases | The remaining bases and constructions, user-supplied knots | 10 |
| M3 bam | `bgam.fitd` in full, `discrete.mf` with R's random number stream, `XWXd` | 8 |
| M4 Inference | `predict`, `vcov`, `summary`, `gam.check` | 5 |
| Packaging | Documentation, first release | 3 |

## M0, second slice: move the engine (done 2026-11-15)

The starting point is the engine in `jonathancrawford05/polaris-re`: about 9,700 lines in 27 files
under `src/polaris_re/analytics/gam_*.py` and `src/polaris_re/gam/`. It depends on polaris-re only
through `core.exceptions` and `core.verification`. Leave behind the conformance harness modules
(`*_conformance.py`, `gam_stage_a.py`, `real_data_trial.py`), the older `experience_gam*.py` model,
and the dashboard helpers. Bring the tests that exercise what moves.

Six functions there were ported from mgcv source and move as they are, each gaining its
counterpart line: `bam_fitd`, `_fit_chol`, `initial_log10_lambda_from_xx_diag`,
`compress_covariates`, `default_m`, `_tero`.

## After the move

Replace re-derived pieces with direct ports where they differ from mgcv's own path, starting with
the outer smoothing-parameter search, where the old engine recorded disagreements on flat criteria.
