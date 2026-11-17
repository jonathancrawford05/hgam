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

## Order of work from here (Jonathan, 2026-11-17)

1. **Finish the initial scope.** That is the exit model end to end: the exit test (done) and M4 on that fit,
   namely `predict` (done), `summary` (parametric table done; smooth-term table and its p-values), `vcov` and
   `gam.check`.
2. **Confirm it.** When the last M4 item merges, the next development run starts no new work. It opens a pull
   request that changes only the routine files and lists, under "Questions for Jonathan", each item of step 1 with
   its test, fixture and measured numbers against its gate in the pinned image, what is still in
   `docs/DEFERRED.md`, and any strict xfail or refusal the exit model can reach. Widening starts only after
   Jonathan confirms on that pull request.
3. **Widen to the rest of `docs/SCOPE.md`**, one construct per slice, each held to an mgcv fixture as before.
   Default order, by what hierarchical models reach for first: factor `by`, `fs`, `re`, `te` and `t2`; then the
   bases `tp`, `ts`, `cs`, `ps`, `cc`; then the families gaussian, poisson, Gamma and the quasi families on `gam`
   and on discrete `bam`. For each, first measure what the moved engine already does against a fixture; port only
   what misses a gate.
4. **Packaging** (documentation, first release) comes after the widening, unless Jonathan asks for it sooner.

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
- A numeric `by` on a cr smooth, the exit formula's `s(AttdAge, by = StudyYear_C)` (done 2026-11-16):
  `tests/test_cr_by_numeric.py`, `gam_binomial_cloglog_by_numeric_reml` (eta gated against mgcv at a tight newton
  `conv.tol`, Jonathan's decision on PR #12), `gam_binomial_cloglog_by_numeric_select_reml`.
- `select=TRUE` on the factor + sz rank-deficient shapes (done 2026-11-16): `tests/test_rank_deficient_sz.py`,
  `gam_binomial_cloglog_f_sz_select_reml`, `gam_binomial_cloglog_two_sz_select_reml`. The null-space penalties make
  the fit full rank in mgcv and hgam, so every term meets the per-term edf gate.
- `bam(method="fREML", discrete=TRUE)` on cr + ti and on sz, with and without `select=TRUE` (done 2026-11-16, the
  first `discrete.mf` step of M3): `tests/test_bam_discrete_fixtures.py`, section 12. Covariates are binned per
  variable by `compress.df` before the fit (`discretise_frame`); without it hgam matched mgcv's `discrete = FALSE` bam
  (8e-9) and missed its discrete bam (eta 4.0e-3).
- `bam(discrete=TRUE, select=TRUE)` on a factor beside its sz smooth and on a numeric `by` (done 2026-11-16):
  `tests/test_bam_discrete_fixtures.py`, section 13. Full rank with select in mgcv and hgam; every term, the factor
  included, meets the per-term edf gate.
- The exit formula on `bam(discrete=TRUE, select=TRUE)` (done 2026-11-16): `tests/test_bam_discrete_fixtures.py`,
  section 14, section 3's frame (n = 4000) and knots. Full rank in mgcv and hgam; eta 1.1e-4, se 2.6e-4 relative,
  per-term edf (the three parametric terms included) 2.2e-3. What remains for the exit test is the entry points.
- The exit test passes (done 2026-11-16): `hgam.bam` / `hgam.gam` forward to the front end, the response
  `DthCnt / ExposCnt` is evaluated as `model.frame` does, and `tests/test_exit.py` fits `hgam_formula` as written on
  section 14's data and meets every gate (strict xfail lifted).
- M4, `predict` on new data from a discrete fit (done 2026-11-16): `GamFit.predict` re-bins new data as `predict.bamd`
  does; `tests/test_predict_bam.py`, section 15. Next: `summary`.
- M4, `summary`, first step (done 2026-11-16): the parametric table of `summary.gam` on a discrete fit;
  `tests/test_summary_bam.py`, section 16.
- M4, discrete bam's constraint rows (done 2026-11-17, Q1 option A of PR #21): R's RNG and `discrete.mf`'s padded
  frame ported (`hgam.frontend.r_rng`, `discrete_model_frame`, ADR 0002, section 17); the intercept meets its gate
  (4.8e-7) and the strict xfail is lifted. Next: the smooth-term table of `summary.gam`.
