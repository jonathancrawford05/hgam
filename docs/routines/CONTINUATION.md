# Continuation

Last updated: 2026-11-17, by the development run's twenty-third slice (run time: profile, README note, a question). M4 is
done for the exit fit. Read `docs/PRIORITIES.md` before
choosing work.

## State

M0 is done (engine in `src/hgam/engine/`, front end in `src/hgam/frontend/`; errors in `hgam.errors`). The six
mgcv-ported functions are held to fixtures in `tests/test_ported_functions.py`.

M1, outer smoothing-parameter search: measured, not ported (PR #5). `tests/test_gam_reml_fixtures.py` holds
`hgam.frontend.gam(method="REML")` to four `mgcv::gam` REML fixtures (sections 2, 5, 6 and 7 of
`oracle/make_fixtures.R`); all meet every gate, so mgcv's `newton` is not ported unless a later fixture fails a
gate on the search.

User-supplied knots: `knots={"x": [...]}` reach every cr smooth, ti margin and sz smooth of that variable
(`tests/test_cr_knots.py`, section 6).

The sz basis: `s(f, x, bs="sz", k=, xt=list(bs="cr"))` is accepted (factor first; other xt, `by=`, x-first order
and `select=TRUE` with sz are refused). Basis vs `smoothCon` at 1.5e-14; the section 7 fit
`p ~ s(x, bs="cr") + s(f, x, bs="sz", ...)` meets every gate (`tests/test_sz_basis.py`, section 7).

A factor main effect beside sz smooths of that factor (section 8, `tests/test_rank_deficient_sz.py`): structure,
eta, se, the uninvolved terms' edf and the summed edf of the terms sharing the redundant directions meet their gates.
The split of that sum between the factor and its sz smooths is not identified and not portable (DEFERRED). Jonathan
decided on PR #8 (option A): those terms are gated on their summed edf, at the per-term gate (0.05 absolute or 1%,
whichever is larger); the split is reported as not identified and is not gated. `gdiPK`'s rank decision is not
ported for this and the exit formula stays as it is. The matching `docs/PRIORITIES.md` line is Jonathan's change.

`select=TRUE` with sz smooths is accepted (section 9, `tests/test_sz_basis.py` and the section 9 case in
`tests/test_gam_reml_fixtures.py`): the null-space penalty agrees with `smoothCon(..., null.space.penalty=TRUE)` to
6e-14, and section 7's model with `select=TRUE` meets every gate. With a factor main effect beside its sz smooths
(section 11: sections 8a and 8b with `select=TRUE`, `tests/test_rank_deficient_sz.py`) the null-space penalties make
the fit full rank in mgcv and hgam, and every term meets the per-term edf gate.

A numeric `by` on cr, `s(x, by=z, bs="cr")` (section 10, `tests/test_cr_by_numeric.py`): basis vs `smoothCon` at
1.2e-14, no constraint (k coefficients, label `s(x):z`); a constant `by`, for which mgcv keeps the constraint, is
refused. 10c (with `select=TRUE`) meets every gate. 10b meets structure, edf and se against mgcv's default fit, and eta
at the estimated sp against mgcv refitted at `newton` `conv.tol = 1e-12` (1.9e-4; option B, Jonathan's decision on
PR #12): mgcv's default stops short of its own REML optimum there.

`bam(method="fREML", discrete=TRUE)` (section 12, `tests/test_bam_discrete_fixtures.py`): `gam(..., method="fREML",
discrete=True)` bins every numeric covariate a smooth reaches with `compress.df`, one covariate at a time, as
`discrete.mf` does for one-covariate margins (`hgam.frontend.discrete.discretise_frame`), then runs `bam_fitd`. sz is
accepted there. On section 5's cr + ti model and section 7's sz model, with and without `select=TRUE`, every key result
meets its gate (eta at most 3.7e-4); so does section 10's numeric-by model (12e, eta 1e-13). Section 13 adds
`select=TRUE` on a factor beside its sz smooth (13a, full rank, factor edf gated) and on the numeric by (13b): eta 1e-9
and 5e-14. Without binning hgam matched mgcv's `discrete = FALSE` bam instead (8e-9).

The exit formula itself (section 14, `bam_discrete_exit_formula`, section 3's frame with n = 4000 and `my_knots`,
response column p = DthCnt / ExposCnt, weights ExposCnt) meets every gate through
`hgam.frontend.gam(..., method="fREML", discrete=True, select=True)`: eta 1.1e-4, se 2.6e-4 relative, per-term edf
2.2e-3 including FaceSize, Smoke and FaceSize:Smoke, full rank (132 of 132) in mgcv and hgam. Structure exact.

The public entry points are wired: `hgam.gam` (`method="REML"` written out; mgcv's default `"GCV.Cp"` is refused) and `hgam.bam` (`method="fREML", discrete=True` only; bam's
default `discrete=FALSE` is refused by name) forward to `hgam.frontend.api.gam`, with `family=` as mgcv writes it or
`family="binomial", link="cloglog"`. A response `a / b` is evaluated as R's `model.frame` does (other left-hand
expressions are refused; a zero denominator is refused). **The exit test passes**: `tests/test_exit.py` fits
`hgam_formula` as written on section 14's data and knots through `hgam.bam` and meets every gate. Tests: 640 pass,
no xfail. What differs is in `docs/DEFERRED.md`.

Run time (M3's one benchmark, `docs/findings/0002-exit-benchmark.md`): on the exit model mgcv's `bam` takes a median
0.98 s; `hgam.bam` 2.2 to 2.4 s with OpenBLAS's default four threads and 0.42 s with `OPENBLAS_NUM_THREADS=1`. Inside
the ten-times rule, so nothing was tuned. Candidate for after M4 (Priorities item 3): cap BLAS threads for small
problems, or document it; this needs no new dependency only if done through an environment variable, so ask first if
it would need `threadpoolctl`.

`predict` on new data from a discrete fit (section 15, `tests/test_predict_bam.py`): `GamFit.predict` re-runs
`discretise_frame` on the new data, as `predict.bamd` re-runs `discrete.mf` (`discrete=False` predicts at the raw
covariates, as `predict.bam` falling through to `predict.gam`). On the exit fit, 15a (1500 rows, binned) eta 8.4e-5
and se 2.3e-4 relative; 15b (40 rows, not binned) eta 4.0e-5, se 2.2e-4. Without re-binning 15a missed by 3.4e-3.

`summary` of a discrete fit, first step (section 16, `tests/test_summary_bam.py`): `GamSummary.parametric` is
`summary.gam`'s `p.table` for a fixed scale (Estimate, Std. Error from Vp, z, `2 * pnorm`), under mgcv's coefficient
names (`GamFit.parametric_names`); `summary()` accepts a `discrete=True` fit (no REML score).

Discrete bam's constraint rows (section 17, `tests/test_discrete_constraint_rows.py`, ADR 0002): `gam.setup` builds the
smooths on `discrete.mf`'s padded frame `mf0`, so each cr and ti-margin constraint is `colMeans(X)` over its rows (unique
values, shuffled, padded with `sample()` under `temp.seed(8547)`). `hgam.frontend.r_rng.RRandom` ports R's
Mersenne-Twister and `sample.int` (Rejection), exact to R's draws; `discrete_model_frame` rebuilds `mf0` row for row
(5.7e-14); `fit_engine_bam` takes the constraints over it (`constraint_frame`; `unique_row_constraint` is gone). On the
exit fit: intercept 4.8e-7 (was 3.6e-3), every coefficient 9.8e-7, se 4.1e-6, eta 7.4e-8, predict 15a 6.0e-8. An offset
beside prior weights is refused in a discrete fit (its order in the parametric frame is not carried). Tests: 665 pass,
no xfail.

Smooth-term table, first step (section 16 and 18, `hgam.frontend.smooth_tests`, PR #23): `summary()` of a discrete fit
gives every fully penalised smooth (`null.space.dim == 0`) `reTest`'s Chi.sq and Ref.df (`reTest`, `recov` with no
random terms, `mroot` via SciPy's `dpstrf`); `bam_fitd` carries X'WX (`EngineGAMFit.xwx`). Exit fit: Chi.sq 1.7e-6
relative on all eight; Ref.df exact on five; the three sz terms give 7, 6, 5 vs mgcv's 11, 7, 6, reported and not gated
(Jonathan's option (a) on PR #23). p-values are `None`. A smooth with a `bs="re"` term beside it keeps `None` (`reTest` would take `recov`'s
`re` branch, not ported; PR #23 review). `null_space_dim` is mgcv's rule where it is known without the constructors (0
under select=TRUE and for re), else `None`. Tests: 678 pass, no xfail.

The smooth-term p-value (PR #25, then this run's PR): `davies.c` is ported line for line in `hgam.frontend.davies` (with
R's `revsort` and `log1pmx`); `psum_chisq` and `liu2` (mgcv.r) are in `hgam.frontend.smooth_tests`, held to
`psum.chisq(..., trace = TRUE)` on 19 cases (section 19, `tests/test_psum_chisq.py`: p to 4.4e-16, trace counts exact).
`re_test` returns `reTest`'s p-value (`psum_chisq(stat, ev)`, fixed scale) and `summary()` reports it: exit fit 3.2e-7 at
most against section 16's `s_table[:, 3]` (five of eight are 0), all eight terms including the three sz terms whose Ref.df
is not gated. `summary` is done for the exit fit. Tests: 727 pass, no xfail.

`vcov` (section 20, `tests/test_vcov_bam.py`, this run's PR): `GamFit.vcov(unconditional=, *, freq=, dispersion=,
sandwich=)` ports `vcov.gam` (mgcv.r): Vp by default, Ve with `freq=True` (`Sl.postproc`'s `F %*% Vp`, i.e.
`Vp X'WX Vp / sig2`; only for a discrete fit, which carries X'WX), `dispersion * vc / sig2`. `GamFit.coefficient_names` is
`names(b14$edf)`. Exit fit: Vp diagonal 8.3e-6 relative, Ve 8.7e-5 (gate 1%), names exact on 132. `unconditional=True`
on a discrete fit stays refused (mgcv's Vc there is Sl.postproc's sp-corrected one, up to 4.07x Vp; DEFERRED; the
fixture records it). `sandwich=True` refused. Section 20b (a gaussian REML gam fit) holds names and `dispersion = 2` off the
discrete path (3.1e-6). Tests: 739 pass, no xfail.

`k.check` (section 21, `tests/test_k_check_bam.py`, this run's PR; PR #27 merged): `GamFit.k_check(data, *, seed,
subsample=5000, n_rep=400)` in `hgam.frontend.gam_check` ports `k.check` (plots.r) for one-covariate smooths (cr, tp;
a numeric by is dropped, as mgcv does), with `deviance_residuals` (`residuals.gam`, deviance) and `Family.dev_resids`.
Smooths with a factor covariate (sz, re) are NA as in mgcv; a factor-by smooth is refused. `data` is the training
frame (mgcv's `b$model`, raw covariates, not the binned ones). Exit fit under `set.seed(21)`: k' exact, edf 1.4e-6,
residuals 1.8e-7, k-index 6.8e-10 relative, p-value exact on the three 1-D terms. k-index and p are not key results
(no gate in PRIORITIES); the test asserts the measured values. The `n > subsample` branch is held to
`k.check(b14, subsample = 1000, n.rep = 50)` (same section; k-index and p exact). Tests: 746 pass, no xfail.

k.check's tensor branch (PR #29, merged): a ti term is rescaled by `gr.f` (mean absolute finite difference of
`predict_term_design(term_state) @ beta`, step `diff(range)/1000`) and tested over each row's 3 nearest neighbours
(`cKDTree` for mgcv's `k_nn` kd-tree; a tie at the third neighbour, relative gap 1e-12, is refused, since mgcv breaks it in
its own order). Exit fit `ti(PolYear,AttdAge)`: k-index 1.3e-10 relative, p exact (0.7875) under `set.seed(21)`, and on the
subsample branch (0.88). `k.check` is done for the exit fit.

`gam.check` (section 22, `tests/test_gam_check_bam.py`, this run's second PR): `GamFit.gam_check(data, *, seed,
k_sample=5000, k_rep=200)` returns a `GamCheckReport` (`print()` it): method `fREML`, optimizer `perf chol`, `outer.info`
grad and Hessian (`bam_fitd` now keeps `Sl.fitChol`'s last `grad` and `hess`; `EngineGAMFit.outer_grad`/`outer_hess`),
`Model rank = 132 / 132` (exact), and the k table. `qq.gam` draws `s.rep = 10` permutations before `k.check`, so those
draws are made; the k table is then exact against the one gam.check computes (taken with `trace()` in the fixture). Plots
are not ported; a REML gam fit and `type != "deviance"` are refused. Tests: 756 pass, no xfail.

## Next slice: BLAS thread setting (Priorities item 3)

Jonathan decided on PR #31 (2026-11-17): option (A), as a setting. If PR #24 has merged first, follow its order instead
(its step 2 is to confirm the initial scope with Jonathan), and do this slice when that order reaches run time.

1. Read `bam` in bam.r (mgcv 1.9-4) for `nthreads`: its default (1), its two-element form, and where it is used.
2. Add `threadpoolctl` as an optional extra (`hgam[threads]`; Jonathan approved it on PR #31, so the extra itself is
   decided, but the PR still waits for his merge under SETTINGS condition 4).
3. `hgam.bam(..., nthreads=1)`: inside `bam_fitd`, `threadpool_limits(nthreads, user_api="blas")` when
   `threadpoolctl` imports. Without it, `nthreads=1` leaves BLAS as the process set it (the README note on
   `OPENBLAS_NUM_THREADS=1` stays), and `nthreads > 1` raises naming the missing extra, so the argument is never
   accepted and ignored. Forms of `nthreads` the port does not follow (for example the two-element vector) are refused.
   The docstring says the limit is process-wide for the length of the fit, and that mgcv's `nthreads` drives its own
   OpenMP code rather than BLAS, so this is a mapping onto mgcv's argument name, not a port of its threading.
4. Test: the fit's key results are unchanged with and without the cap (the exit test), and the limit is restored
   after the fit, including when the fit raises.
5. Rerun `docs/findings/exit_benchmark/time_hgam.py` with default threads and the extra installed; add one row to
   finding 0002.

Large data (about 30 million rows) is deferred: see `docs/DEFERRED.md`.

## Open items for Jonathan

Decided on PR #31 (2026-11-17): option (A) as a setting, `nthreads` with default 1 (next slice above); optimise for
large data (about 30m rows) later (DEFERRED).

Decided on PR #23 (2026-11-17): option (a). Ref.df is gated exact only where mgcv's rank is not set by rounding; on the
exit fit's s(FaceSize,AttdAge), s(Smoke,AttdAge) and s(FaceSize,PolYear) it is reported, not gated, and those terms are
gated through Chi.sq and the p-value. The matching `docs/PRIORITIES.md` row (with Q2's smooth-term gates)
is Jonathan's change.

Decided on PR #21 (2026-11-16): Q1 option (A): port mgcv's constraint rows for discrete bam, held to the
"Deterministic pieces ... constraints" gate (done 2026-11-17, ADR 0002; the strict xfail is lifted). Q2 option (a): smooth-term tests are gated
at Ref.df exact; Chi.sq 1% relative; p-value 1e-3 absolute or 1% relative, whichever is larger. The matching
`docs/PRIORITIES.md` row is Jonathan's change (a routine slice does not edit that file).

Decided on PR #12 (option B): where mgcv's default search stops short of its own optimum, a fitted quantity at
estimated smoothing parameters is gated against mgcv refitted at a tight `newton` `conv.tol`, at the unchanged gate.
The fixture records both mgcv fits and their REML scores, and the tight fit is the reference only where mgcv's own
tight score is not higher than its default score. mgcv's `newton` is not ported for this (option A not taken); the
moved Newton search stays. The rule is in `docs/PRIORITIES.md`.

(PR #10's question on SETTINGS condition 5 is decided: option (a). A refusal test may be retired when a
fixture-backed acceptance test of the same input replaces it; the wording is in `docs/routines/SETTINGS.md`.)

## Open questions

None.

(PR #15's nit on `s(x, f, bs="sz")` was already met: `api.py` refuses it by name and `tests/test_sz_basis.py` holds
the refusal. Removed from this list on 2026-11-17.)
