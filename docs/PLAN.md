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

- Outer smoothing-parameter search (done 2026-11-16): the moved Newton search meets the gates on
  `gam_binomial_cloglog_ti_reml` and `gam_poisson_offset_reml` (`tests/test_gam_reml_fixtures.py`), so
  mgcv's `newton` is not ported. Re-port it only if a later fixture fails a gate on the search.
- User-supplied knots for cr smooths and ti margins (done 2026-11-16, an M2 item the exit test needs):
  `tests/test_cr_knots.py`, `gam_binomial_cloglog_knots_reml`.
- The sz basis with a cr base, `s(f, x, bs="sz", xt=list(bs="cr"))` (done 2026-11-16, M2, the exit test needs it):
  `tests/test_sz_basis.py`, `gam_binomial_cloglog_sz_reml`.
- A factor main effect beside sz smooths of that factor, the exit formula's rank-deficient shape (fixtures and gates
  done 2026-11-16): `tests/test_rank_deficient_sz.py`,
  `gam_binomial_cloglog_f_sz_reml`, `gam_binomial_cloglog_two_sz_reml`. Structure, eta, se, uninvolved edf and the
  summed edf of the involved terms meet their gates. The per-term split inside the involved terms is not
  identified and not gated (Jonathan's decision on PR #8; `docs/DEFERRED.md`).
- `select=TRUE` with sz smooths (done 2026-11-16): `tests/test_sz_basis.py` (null-space penalty vs `smoothCon(...,
  null.space.penalty=TRUE)`), `gam_binomial_cloglog_sz_select_reml`.
