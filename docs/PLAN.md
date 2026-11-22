# Plan

About 40 pull-request-sized slices. The slice counts are rough estimates.

| Milestone | What | Slices |
|---|---|---|
| M0 Scaffold | This repository; then move the verified engine across from polaris-re | 2 |
| M1 gam core | `gam.setup`, `gam.fit3`, `newton`, REML, the derivative code (`gdi.c`) | 12 |
| M2 Bases | The remaining bases and constructions, user-supplied knots | 10 |
| M3 bam | `bgam.fitd` in full, `discrete.mf` with R's random number stream, `XWXd` | 8 |
| M4 Inference | `predict`, `vcov`, `summary`, `gam.check` (done for the exit fit, 2026-11-17) | 5 |
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
   Before the bases, two follow-ups from PR #48 (Jonathan's decision, 2026-11-21), in this order:
   - **(c) `gam.control(newton = list(conv.tol = ))`** as a user-facing option on `gam(method="REML")`, default
     unchanged at mgcv's 1e-6. Held to section 27d3's tight refit (`conv.tol = 1e-12`). One slice.
   - **(b) Port mgcv's `newton` (gam.fit3.r).** The re-port trigger below now applies: 27d3 misses the predict gate
     on the search. Split into slices. Done when 27d3 meets every gate against mgcv's default fit and the
     `select=TRUE` refusal for t2 in a REML gam is lifted.
     First slice (2026-11-21): the port, opt in as `newton_select_lambdas(iteration="mgcv")`, held to section 27e;
     27d3 meets every gate against mgcv's default fit through it. Second slice (2026-11-22, Jonathan chose PR #51
     Q1 option (a)): the port is `fit_engine_gam`'s default and the refusal is lifted; done. The ridge cases are
     gated like for like. Next: the bases (`tp` first).
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
- M4, `summary` smooth-term table, first step (done 2026-11-17, PR #23): `reTest` / `recov` / `mroot` give Chi.sq (all
  eight exit smooths, 1.7e-6) and Ref.df (exact on five; three sz terms reported, not gated: option (a)).
- M4, `summary` smooth-term p-value, first step (done 2026-11-17, PR #25): `psum.chisq`, Davies' `davies.c` and `liu2`
  ported (`hgam.frontend.davies`, `tests/test_psum_chisq.py`, section 19).
- M4, `summary` smooth-term p-value wired (done 2026-11-17, PR #26): `re_test` returns `reTest`'s p-value and
  `summary()` reports it; exit fit 3.2e-7 at most (`tests/test_summary_bam.py`, section 16). `summary` is done for the
  exit fit.
- M4, `vcov` (done 2026-11-17, PR #27): `GamFit.vcov(freq=, dispersion=)` and `GamFit.coefficient_names` port `vcov.gam`;
  exit fit Vp diagonal 8.3e-6, Ve 8.7e-5 relative, names exact (`tests/test_vcov_bam.py`, sections 20 and 20b). `vcov` is done
  for the exit fit (Vc refused for a discrete fit).
- M4, `gam.check`, first step (2026-11-17, open): `k.check` (plots.r) as `GamFit.k_check(data, seed=)`, one-covariate
  smooths and the factor NA; exit fit k', edf and NA exact or in gate, k-index 6.8e-10, p exact under R's draws
  (`tests/test_k_check_bam.py`, section 21). Second step (2026-11-17): the tensor branch, so the exit fit's ti row has
  k-index (1.3e-10 relative) and p (exact); `k.check` is done for the exit fit.
- M4, `gam.check` report (done 2026-11-17): `GamFit.gam_check(data, seed=)` gives what `gam.check` prints for the discrete
  fit (method, optimizer, `outer.info` grad and Hessian, model rank, the k table after `qq.gam`'s ten draws), plots not
  ported; rank 132 / 132 exact, k table exact (`tests/test_gam_check_bam.py`, section 22). **M4 is done for the exit fit.**
- Widening, factor `by` on cr (done 2026-11-17): `s(x, by=f, bs="cr")` measured against mgcv with no engine change:
  basis vs `smoothCon` 1.7e-14, structure and coefficient names exact, eta 5.7e-7 (REML gam) and 1e-13 (discrete bam,
  with and without `select=TRUE`), se 5.6e-4, edf within 2.3e-4 of the gate's width, predict on new data 1.2e-13
  (`tests/test_cr_by_factor.py`, section 23). An ordered factor is refused. Next: `fs`.
- Widening, `fs` on cr, first step (2026-11-17): the basis. `nat.param` (type 1) and `smooth.construct.fs.smooth.spec`
  ported (`hgam.engine.gam_basis_fs`), held to `smoothCon` and `PredictMat` at 2.4e-14 up to column signs and the order
  of the two null-space columns (`tests/test_fs_basis.py`, section 24). `gam()` still refuses `bs="fs"`. Next: the fit.
- Widening, `fs` on cr, second step (2026-11-17): the REML gam fit. `s(x, f, bs="fs", xt="cr", k=)` (either order,
  factor found by type) through `gam(method="REML")`: structure, names and full rank exact, eta 1.1e-4, se 4.3e-4
  relative, edf 1.3e-3 of the gate width, predict on new data 1.1e-4 (`tests/test_fs_fit.py`, section 24b); beside a cr
  smooth of x or a factor main effect, every gate met (section 24c). Refused:
  `discrete=True` (bam's tensor-like fs margins), `select=True`, `by=`, `m=`, the default tp base. Next: fs in
  discrete bam.
- Widening, `fs` on cr, third step (2026-11-17): discrete bam. The term is built on `discrete.mf`'s `mf0` (`P` and
  penalty scaling over its rows, margins compressed in written order): eta 6.1e-14, se 3.7e-14, `P` 1.3e-14,
  predict on new data 6.3e-14, summary's Ref.df exact and Chi.sq 3e-14, both written orders (`tests/test_fs_fit.py`,
  section 24d). Next: `select=TRUE` with fs.
- Widening, `fs` on cr, fourth step (2026-11-20): `select=TRUE`. smoothCon's `null.space.penalty` `need.full` branch
  adds nothing to fs (its penalties sum to full rank; mgcv's fit equals the `select=FALSE` one), and neither does
  `null_space_penalty`; the refusal is dropped. REML gam eta 1.1e-4, se 4.3e-4; discrete bam eta 6.1e-14; beside
  `s(x, bs="cr")` (whose null-space penalty is added) eta 2.9e-5, se 4.4e-4, edf 1.6e-2 of the gate width
  (`tests/test_fs_fit.py`, section 24e). `fs` on cr is done. Next: `re` beside fs and in discrete bam.
- Widening, `re` (2026-11-20): `s(g, bs="re")` measured against mgcv with no code change, beside a cr smooth and
  beside fs, as a REML gam and a discrete bam (section 25, `tests/test_re_fit.py`): structure, rank and names exact,
  eta at most 1.6e-5, se 7.2e-4 relative, edf 5.2e-3 of the gate width, predict on new data eta 2.7e-5, se 3.5e-4.
  `summary`'s smooth-term row beside an re term stays `None` (`recov`'s re branch, DEFERRED). Next: `te` on cr.
- Widening, `te` on cr, first step (2026-11-20): the basis. `hgam.engine.gam_basis_te` ports
  `smooth.construct.tensor.smooth.spec`'s te branch (margins by `smooth.construct`; cr's `noterp` skips `np`),
  `tensor.prod.model.matrix`, `tensor.prod.penalties`, smoothCon's scaling and whole-term constraint, and
  `Predict.matrix.tensor.smooth`. Two and three margins (section 26, `tests/test_te_basis.py`): structure exact, X, S
  and PredictMat at most 1.6e-14 relative. `gam()` still refuses te. Next: the te REML fit (section 26b).
- Widening, `te` on cr, second step (2026-11-20): the REML fit. `te(x, z, k=)` parsed and built as
  `TermSpec(basis="te")` (section 26b, `tests/test_te_fit.py`): `te(x, z, k=c(5, 4))` eta 3.0e-4, se 2.0e-3, edf
  1.4e-2 of the gate width; `f + te(x, z)` eta 4.9e-6, se 2.5e-3; predict on new data within gate; structure, rank and
  names exact. Refused by name: select=TRUE, discrete bam, te beside another smooth of its covariates (gam.side). Next:
  te in discrete bam (section 26c).
- Widening, `te` on cr, third step (2026-11-20): discrete bam. `tero` reorders te as ti (keeping the kind),
  `discrete.mf` compresses each margin covariate, and the term is built on `mf0` (`te_basis(frame=)`: knots, penalty
  scaling and constraint over its rows, design by `Predict.matrix.tensor.smooth` at the binned data). Section 26c,
  `tests/test_te_fit.py`: eta 1.5e-13, se 7.5e-14, edf 3.9e-6 of the gate width, predict on new data 6.2e-13, the
  constraint `C Z` 1.1e-15 (over the data rows it misses by 0.23); structure, rank and names exact (`te(z,x)` after
  `tero`). Next: `select=TRUE` with te.
- Widening, `te` on cr, fourth step (2026-11-20): `select=TRUE`. smoothCon's `null.space.penalty` `need.full` branch
  appends `U U'` over the null space of the summed te penalties (rank 3); `null_space_penalty` gives it at 1.6e-14
  (section 26d, `tests/test_te_basis.py`) with no engine change, and the refusal is dropped. REML gam eta 3.3e-5, se
  2.7e-3; discrete bam eta 3.6e-13, se 4.7e-13; three sp, structure, rank and names exact (`tests/test_te_fit.py`).
  `te` on cr is done (a te beside another smooth of its covariates stays refused: `gam.side`). Next: `t2`.
- Widening, `t2` on cr, first step (2026-11-20): the basis. `hgam.engine.gam_basis_t2` ports
  `smooth.construct.t2.smooth.spec`, `t2.model.matrix` (`full = FALSE`, no `ord`), `Predict.matrix.t2.smooth`, and
  smoothCon's scaling, sparse fit-constraint branch and `Cp` prediction constraint; `nat_param` gains type 3. Section
  27, `tests/test_t2_basis.py`: structure exact; margin P, raw X, C and Cp at 1.8e-14 up to column signs; X/S, Xp/Sp
  and PredictMat at 3.7e-14 up to an orthogonal map. `gam()` still refuses t2. Next: the REML fit.
- Widening, `t2` on cr, second step (2026-11-20): the REML fit. `t2(x, z, k=)` is parsed (`TensorCall(kind="t2")`) and
  built as `TermSpec(basis="t2")` in smoothCon's fit parameterisation; `G$P` is not ported (predictions agree:
  `[1, X]` and `[1, Xp]` span the same space, 1.1e-14), and `vcov`, `summary`, `k_check` refuse t2. Section 27b,
  `tests/test_t2_fit.py`: eta 2.4e-4 and 4.4e-4 (the second against mgcv's tight refit, PR #12 rule), se 3.1e-3,
  edf 3.5e-2 of the gate, predict 5.7e-4; fit at mgcv's sp 8.7e-12; structure exact. Next: discrete bam.
- Widening, `t2` on cr, third step (2026-11-20): discrete bam. `bam` calls `gam.setup(..., modCon = 2)`, so t2's fit
  constraint is `Cp` (dense, smoothCon's full QR branch; no `G$P`); `discrete.mf` compresses t2's covariates jointly
  (`uniquecombs`' text order ported with R's `as.character`, `r_double_string`) and the term is built on `mf0`
  (`t2_basis(frame=, mod_con=2)`). Section 27c, `tests/test_t2_discrete.py`: eta at most 6.3e-12, se 2.8e-11, predict
  5.6e-12, edf 4.8e-5 of the gate, |Cp| 5.9e-15, joint compress.df rows and index exact; structure exact. Next: `select=TRUE` with t2.
- Widening, `t2` on cr, fourth step (2026-11-21): `select=TRUE`. smoothCon's `null.space.penalty` (`need.full`, rank 3)
  is what `null_space_penalty` builds, at modCon 0 and 2 (section 27d1, 2.3e-14). Ported for discrete bam (27d4, 27d5:
  eta 1.4e-11, se 3.5e-11, edf 2.4e-5 of the gate, predict 7.2e-12). Still refused in a REML gam: on 27d3 the
  null-space sp's flat ridge leaves hgam's Newton search 1.3e-3 from mgcv's tight refit on prediction. Jonathan chose
  option (a) on PR #48. Next: follow-up (c), then (b) (step 3 above), then the bases (`tp` first).
- PR #48 follow-up (c) (2026-11-21): `control={"newton": {"conv.tol": ...}}` on `gam(method="REML")` (mgcv's
  `gam.control`, mgcv.r; default 1e-6 unchanged), through `fit_engine_gam(conv_tol=)` to the Newton search; every other
  `gam.control` entry and the option on discrete bam are refused. `tests/test_newton_conv_tol.py`: at 1e-12 against
  mgcv's tight refits, 27b2 eta 4.8e-10, predict 7.7e-10, se 2.6e-3; 27d3 (refusal bypassed in the test only) eta
  1.2e-9, predict 1.9e-9, se 2.5e-3, edf 1.4e-7 of the gate. Next: follow-up (b), the `newton` port.
- `tp`, one covariate (2026-11-22, measured, not ported): section 28 (`tests/test_tp_fit.py`). The moved
  `tp_basis` meets smoothCon's tp basis up to its gauge (design 6.2e-14, penalty 3.0e-14 in mgcv's basis); `s(x)` +
  cr, two tp smooths with an offset (Poisson) and `select=TRUE` meet every gate (eta at most 8.2e-12 on data and new
  data, se 3.2e-4 relative, edf 7.9e-9 of the gate). Still refused: the `max.knots` subsampling regime (more than 2000 rows and 2000 distinct values), tp in discrete bam, `by=` on tp, tp as the `fs`
  base, two-covariate tp. Next: `fs` on its default tp base.
- `fs` on its default tp base (2026-11-22): `s(x, f, bs="fs", k=)` with no `xt` (`smooth.construct.fs.smooth.spec`:
  base.bs `"tp"`). `fs_basis(base="tp")` builds the base with the moved `tp_basis` and runs the same `nat.param`; held to
  smoothCon at 7.9e-14 (X, S list, PredictMat) up to column signs and the null pair's order (section 29a). REML fits
  alone, beside `s(x)` and with `select=TRUE` meet every gate (eta at most 1.2e-12 on data and new data, se 3.0e-4
  relative, edf 1.9e-10 of the gate; section 29, `tests/test_fs_fit.py`). Still refused: fs on tp in discrete bam,
  `knots=` reaching a tp base, `xt="tp"` written out. Next: tp in discrete bam.
