# Development session log

## 2026-11-28 (forty-seventh slice): `by=` on tp (PLAN step 3, bases)

Same run: PR #58 (tp in discrete bam) auto-merged after the Ready review of `db26679` (six conditions posted). Ported `by=` on tp: `build_python_tp_term(by=, group=)` (numeric by: no constraint, rows scaled; factor by: constraint on the whole column, one masked copy per level), the parser and `TermSpec` accept it, predict applies it. Section 31: 31a numeric by, 31b factor by, each as REML gam and discrete bam, with and without select.
Tests: 889 → 899 (+10 in `tests/test_tp_by.py`; `test_a_thin_plate_smooth_takes_no_by` replaced by `test_a_thin_plate_smooth_takes_a_by`; the TermSpec tp test's by row now asserts acceptance). `make lint` clean. Tier 1: eta 1.8e-11, se 6.5e-4 (REML; discrete 4e-13), edf 2.1e-6 of the gate, predict 9.7e-12 / 5.2e-4, sp 1.1e-9 (not gated).
Deferred: the REML se line gets section 31's number. Next: `ts`.

## 2026-11-28 (forty-sixth slice): `tp` in discrete bam (PLAN step 3, bases)

`bam(discrete=TRUE)` accepts one-covariate tp: `tp_term_basis` builds the basis on `discrete.mf`'s `mf0` (shift, Xu, column norms, scale.penalty, constraint over its rows) and evaluates `Predict.matrix.tprs.smooth` at the binned rows; the bam refusal is gone. Section 30 (28b, 28c, 28d as discrete bam).
Tests: 883 → 889 (+7 in `tests/test_tp_discrete.py`; `test_the_discrete_bam_path_refuses_a_thin_plate_smooth` retired, SETTINGS condition 5). `make lint` clean. Tier 1: eta 1.7e-8, se 4.6e-12, edf 1.7e-5 of the gate, predict 2.8e-8, shift/Xu 8.7e-16, select summary Chi.sq 4.2e-8, Ref.df exact.
Built on the data rows instead: eta still 1e-12 but shift 7.2e-2 off and sp 1.2e-2 relative. Nothing deferred. Next: `by=` on tp.

## 2026-11-28 (PR #55, Jonathan's decision)

Jonathan chose Q1 option (a) on PR #55: fs fits on a tp base stay refused (mgcv's own fit moves 1.6e-2 under a 1e-12 shift of x on tp, per PR #56's section 29e). Recorded in CONTINUATION, PLAN and DEFERRED. Review nit on `084fb08` done: `fs_basis` refuses knots on a tp base and a k that disagrees with cr's knots.
Tests: 882 → 883 (`test_fs_basis_refuses_arguments_its_base_would_ignore`). `make lint` clean. No key result changed. Next: tp in discrete bam.

## 2026-11-22 (forty-fifth slice): `fs` on its default tp base, engine only (PLAN step 3, bases)

Same run: PR #54 auto-merged after the Ready review of `74f9069` (six conditions posted; branch delete cut by the git proxy). Ported `tp_construct` and `fs_basis(base="tp")`; fits passed against local mgcv 1.9.1 (eta 1.8e-11) but CI's `oracle` (mgcv 1.9.4) failed: X 0.35, eta 2.2e-3 / 3.0e-2. Cause measured: the null pair's rotation is set by rounding and a 0.1 rad rotation moves eta 1.2e-3 (cr and tp). Front-end acceptance reverted; question for Jonathan on PR #55.
Tests: 881 → 882 (`test_fs_tp_basis_matches_smoothcon_up_to_the_null_pair`). `make lint` clean. Tier 1: range columns 2.5e-14, null pair as a subspace 4.5e-15, penalties 6.6e-15. PR #54's nit (`tp_predict`'s counterpart) done.
Deferred: the fs null-pair line. Next: depends on Jonathan's answer; otherwise tp in discrete bam.

## 2026-11-22 (PR #54 review follow-up, `ada3807`)

Should fix 1: tp's `max.knots` subsampling refusal raises `HgamNotImplementedError` (was `HgamValidationError`); engine test updated, `test_tp_max_knots_subsampling_is_refused_through_gam` added. Should fix 2: `tp_basis` and its module name `smooth.construct.tp.smooth.spec` / `construct_tprs`, mgcv 1.9-4; stale polaris-re references removed.
Nit: 28c predicts new data at a nonzero offset (fixture regenerated; eta new 4.7e-12). Tests: 880 → 881. `make lint` clean. No key result changed.

## 2026-11-22 (forty-fourth slice): the one-covariate `tp` basis, measured (PLAN step 3, bases)

No open PR at start (PR #53 merged). Section 28 fixtures: `smoothCon(s(x, bs="tp", k=8))`, `s(x) + s(z, bs="cr")` (binomial cloglog), two tp smooths with an offset (Poisson), and `select=TRUE`. No engine change: the moved `tp_basis` met every gate.
Tests: 876 → 880 (`tests/test_tp_fit.py`). `make lint` clean. Tier 1 (mgcv 1.9-1): basis in mgcv's gauge 6.2e-14 (design) and 3.0e-14 (penalty); eta ≤ 8.2e-12 on data and new data, se ≤ 3.2e-4 relative, edf ≤ 7.9e-9 of the gate; structure and coefficient names exact.
Deferred: se line extended with the tp numbers. Next: `fs` on its default tp base.

## 2026-11-22 (forty-third slice): the `newton` port is the default (PR #48 follow-up (b), second slice)

Jonathan chose PR #51 Q1 option (a) (like-for-like gating; the PRIORITIES text is PR #52, not this slice). `fit_engine_gam(newton_iteration=None)` now runs the port; `"hgam"` stays opt in. `_refuse_t2` no longer refuses `select=TRUE` in a REML gam. `summary()` reports the test the search ran (`EngineGAMFit.conv_tol`; mgcv's gradient relative to `5 * score.scale`).
Tests: 877 → 876 (`test_t2_refusals`' select case removed, replaced by `test_mgcv_newton`'s 27d3 default case, now through public `hgam.gam`). `make lint` clean. Against mgcv's default fits (tier 1): 27b2 eta 8.1e-11, 27d3 2.1e-10 (predict 3.2e-10, se 2.5e-3), by-numeric 1.2e-10, t2 1.96e-10; at conv.tol 1e-12 vs tight refits ≤ 9.6e-11.
Deferred: unchanged (inner P-IRLS conv.tol/100, quasipoisson log-scale coordinate, edge.correct). Next: the bases, `tp` first.

## 2026-11-21 (forty-second slice): mgcv's `newton` port, opt in (PR #48 follow-up (b), first slice)

PR #50 auto-merged (six conditions posted; branch delete cut by the git proxy). Ported `newton` (gam.fit3.r) as `newton_select_lambdas(iteration="mgcv")`, default unchanged; `mgcv_reml_score_offset` and `saturated_log_likelihood` (fix.family.ls) for `score.scale`. The convergence test alone regressed four select fits (paths diverge at the step logic), so the whole iteration is ported.
Tests: 866 → 877 (`tests/test_mgcv_newton.py`; fixture 27e). `make lint` clean. Against mgcv's default fits (tier 1): iteration counts equal on all five; eta 27b2 8.1e-11, 27d3 2.1e-10 (predict 3.2e-10), sz select 6.6e-5, fs select 1.8e-10, by-numeric 1.2e-10; at conv.tol 1e-12 vs tight refits 1.2e-10.
Deferred: quasipoisson `score.scale`, gamma != 1 offset unmeasured, sz select final score 8.8e-4. Question for Jonathan: Q1 (reference for ridge cases). Next: the port as default, refusal lifted.

## 2026-11-21 (PR #49 review nits)

Jonathan asked for PR #49's two nits after it merged. `control={}` (mgcv `control = list()`) returns the default `conv.tol` instead of a misleading validation error; the inner P-IRLS `conv.tol/100` tightening (gam.fit3.r:1308, not ported) is recorded in DEFERRED for follow-up (b).
Tests: 865 → 866 (one case inside `test_default_conv_tol_is_unchanged`; PR #50 review nit: `test_empty_control_on_discrete_bam_is_the_default`). `make lint` clean. No key result changes; no fixture change.
Next: follow-up (b), the `newton` port.

## 2026-11-21 (forty-first slice): `gam.control(newton = list(conv.tol = ))` (PR #48 follow-up (c))

Same run: PR #48 auto-merged after the Ready review of `b232be4` (six conditions posted; branch delete cut by the git proxy). `control={"newton": {"conv.tol": x}}` on `gam(method="REML")` and `hgam.gam`, threaded to `newton_select_lambdas`; default 1e-6; other entries and discrete bam refused; `summary().epsilon_rel` reports the tolerance used.
Tests: 855 → 865. `make lint` clean. At 1e-12 against mgcv's tight refits (tier 1, mgcv 1.9-1): 27b2 eta 4.8e-10, predict 7.7e-10, se 2.6e-3; 27d3 eta 1.2e-9, predict 1.9e-9, se 2.5e-3, edf 1.4e-7 of gate.
Deferred: hgam's Newton stop test is not mgcv's (27d3 reports converged=False at 1e-12 where mgcv converges). Next: follow-up (b), the `newton` port.

## 2026-11-21 (fortieth slice): `select=TRUE` with t2 (fourth step of `t2`, PLAN step 3)

No open PR at start (PR #47 merged). No engine change: `null_space_penalty` gives smoothCon's t2 null-space penalty at modCon 0 and 2 (27d1: 2.3e-14, rank 3). `select=TRUE` with t2 lifted for discrete bam only; a REML gam keeps a narrower refusal (no test removed).
Tests: 849 → 855. `make lint` clean. Section 27d (tier 1, mgcv 1.9-1 local): discrete bam eta 1.4e-11, se 3.5e-11, edf 2.4e-5 of gate, predict 7.2e-12 / se 3.3e-11, |Cp| 5.7e-15; four sp, structure exact.
Question for Jonathan: REML gam `p ~ f + t2(x, z)` with select misses predict by 1.3e-3 against mgcv's tight refit (hgam log sp 5.66; mgcv 4.85 default, 18.8 tight). At conv_tol 1e-12 hgam meets it (1.9e-9). Next: his answer, then `tp`.
PR #48: Jonathan chose (a). Follow-ups (c) `gam.control(newton=list(conv.tol=))` and then (b) the `newton` port are added to PLAN step 3. Next: (c).

## 2026-11-20 (thirty-ninth slice): `t2` on cr in discrete bam (third step of `t2`, PLAN step 3)

No open PR at start (PR #46 merged). bam's `modCon = 2` (t2 fit constraint = `Cp`, built on `mf0`), joint `compress.df` for t2's covariates, `uniquecombs`' multi-column text order with R's `as.character` (`r_double_string`); `tero` skips t2. Retired refusal case: `test_t2_refusals[discrete=True]`, replaced by `test_discrete_t2_meets_the_gates`.
Tests: 838 → 846. `make lint` clean. Section 27c (tier 1, mgcv 1.9-1 local): eta 6.3e-12, se 2.8e-11, edf 4.8e-5 of gate, predict 5.6e-12 / se 3.1e-11, |Cp| 5.9e-15, joint compress.df rows and index exact; structure exact.
Deferred: vcov/summary on a discrete t2 fit (refused, not measured); `r_double_string` above 1e15. Next: `select=TRUE` with t2.
PR #47 review (9820042): fixture 27c5 holds compress.df's joint grid branch (12000 rows, 4603 rows and index exact, values 2.7e-15; fit eta 1.3e-12). Tests 846 → 849.

## 2026-11-20 (thirty-eighth slice): `t2` on cr, the REML fit (second step of `t2`, PLAN step 3)

PR #45 auto-merged in this run (six conditions posted; the branch could not be deleted, the git proxy cuts the push). t2 wired into `gam(method="REML")` in smoothCon's fit parameterisation; `G$P` not ported (spans agree to 1.1e-14), so `vcov`/`summary`/`k_check` refuse t2. Retired refusal-only t2 cases (SETTINGS condition 5): `test_gam_still_refuses_t2`, the `t2()` formula refusal and `TermSpec(basis="t2")` refusal, replaced by `test_t2_meets_the_gates` (the formula cases now refuse `full=`, `ord=` and three margins).
Tests: 823 → 838. `make lint` clean. Section 27b (tier 1, mgcv 1.9-1): eta 2.4e-4 and 4.4e-4 (vs tight refit; 7.6e-4 vs default, reported), se 3.1e-3, edf 3.5e-2 of gate, predict 5.7e-4 / se 3.6e-3, fit at mgcv's sp 8.7e-12; structure exact.
Deferred: `G$P`; the f + t2 flat ridge (mgcv default vs tight 1.2e-3). Next: t2 in discrete bam (joint compression).
PR #46 review (742464d): unconditional se on a t2 fit refused (mgcv does not map `Vc` through `G$P`); nits done (`t2_raw_design` counterpart line, DEFERRED note on `fit.coef`). Tests stay 838.

## 2026-11-20 (thirty-seventh slice): `t2` on cr, the basis (first step of `t2`, PLAN step 3)

No open PR at start (PR #44 merged). Ported `smooth.construct.t2.smooth.spec`, `t2.model.matrix`, `Predict.matrix.t2.smooth`, `nat.param` type 3 and smoothCon's sparse fit constraint plus `Cp` prediction constraint (`hgam.engine.gam_basis_t2`); `gam()` still refuses t2. Section 27 added (two and three cr margins); every existing fixture regenerated unchanged.
Tests: 818 → 823. `make lint` clean. Section 27 (tier 1, mgcv 1.9-1): rank, null.space.dim, p.lab exact; margin P 8.0e-15, raw X 1.8e-14, C and Cp 5.1e-15 (up to column signs: one k = 3 margin column has the opposite sign); X/S 2.3e-14, Xp/Sp 2.5e-14, PredictMat 3.7e-14 (up to an orthogonal map).
Deferred: nothing new. Next: the t2 REML fit (gam.setup's `G$P`, fit to prediction parameterisation).

## 2026-11-20 (thirty-sixth slice): `select=TRUE` with te (fourth step of `te`, PLAN step 3)

PR #43 auto-merged in this run (six conditions posted). `select=TRUE` with te needed no engine change: `null_space_penalty` matches smoothCon's `need.full` third penalty; the refusal and its refusal-only test case are gone (SETTINGS condition 5, replaced by the four 26d fit cases). PR #43 nit: `fit_term_state` now passes only supplied knots to `te_basis`, so default knots come from the build rows in both builders.
Tests: 812 → 818. `make lint` clean. Section 26d (tier 1, mgcv 1.9-1): null-space penalty 1.6e-14, S 3.7e-15, rank [12, 10, 3]; REML gam eta 3.3e-5, se 2.7e-3, edf 9.1e-4 of gate; discrete bam eta 3.6e-13, se 4.7e-13, C Z 1.1e-15; predict within gate; structure, rank, names exact.
Deferred: nothing new. Next: `t2`.

## 2026-11-20 (thirty-fifth slice): `te` on cr in discrete bam (third step of `te`, PLAN step 3)

No open PR at start (PR #42 merged). te in `bam(discrete=TRUE)`: `_tero` keeps `kind` (it turned te into ti), `discrete.mf` margins include te, and `te_basis(frame=)` builds the term on `mf0` (knots, penalty scaling, constraint) with the design by `te_predict_design` at the binned rows. The discrete refusal case in `test_te_refusals` is retired (SETTINGS condition 5), replaced by the 26c cases of `test_te_meets_the_gates`.
Tests: 808 → 812. `make lint` clean. Section 26c (tier 1, mgcv 1.9-1): te(z,x) after tero eta 1.5e-13, se 7.5e-14, edf 6.1e-7 of gate, predict 6.2e-13; f + te(x,z) eta 8.3e-14, se 3.6e-14, edf 3.9e-6, predict 5.6e-14; C Z 1.1e-15 (data rows: 5.1e-2, 2.3e-1); structure, rank, names exact.
Deferred: nothing new. Next: `select=TRUE` with te.

## 2026-11-20 (thirty-fourth slice): `te` on cr, the REML fit (second step of `te`, PLAN step 3)

PR #41 (te basis) auto-merged in this run (six conditions posted). `te(x, z, bs="cr", k=)` parsed (`TensorCall(kind="te")`, bs defaults to cr) and built as `TermSpec(basis="te")` via `build_python_te_term`; predict through the stored `Z`. Refused by name: select=TRUE, discrete bam, a te beside another smooth of its covariates (gam.side not ported), k.check. Refusal-only te cases retired (SETTINGS condition 5), replaced by `tests/test_te_fit.py`.
Tests: 794 → 806, 808 after the review follow-up (k refusals). `make lint` clean. Section 26b (tier 1, mgcv 1.9-1): te(x,z) k=c(5,4) eta 3.0e-4, se 2.0e-3, edf 1.4e-2 of gate, predict eta 2.2e-4, se 1.9e-3; f + te(x,z) eta 4.9e-6, se 2.5e-3, edf 5.0e-4 of gate, predict 8.7e-6, se 2.3e-3; structure, rank, names exact.
Deferred: nothing new. Next: te in discrete bam (26c).

## 2026-11-20 (thirty-third slice): `te` on cr, the basis (first step of `te`, PLAN step 3)

No open PR at start (PR #40 had merged). `hgam.engine.gam_basis_te` ports `smooth.construct.tensor.smooth.spec` (te branch: margins by `smooth.construct`, cr's `noterp` skips the np reparameterisation), `tensor.prod.model.matrix`, `tensor.prod.penalties`, `Predict.matrix.tensor.smooth` and smoothCon's scaling and whole-term constraint, for any number of cr margins. `gam()` still refuses te.
Tests: 789 → 793 (four added), 794 after the review follow-up (constraint_rows test). `make lint` clean. Section 26 (tier 1, mgcv 1.9-1), structure (rank, null.space.dim, columns) exact; te(x,z) X 1.6e-14, S 1.7e-14, PredictMat 1.0e-14, knots 1.6e-15; te(x,z,w) X 1.5e-14, S 1.6e-14, PredictMat 8.5e-15.
Deferred: nothing new. Next: wire te into gam(REML) (section 26b).

## 2026-11-20 (thirty-second slice): `re` beside cr and fs (PLAN step 3)

PR #39 auto-merged in this run (six conditions posted). `s(g, bs="re")` measured, no code change: section 25 holds `s(x, bs="cr") + s(g, bs="re")` and `s(x, f, bs="fs") + s(g, bs="re")` as REML gam and discrete bam (`tests/test_re_fit.py`). PR #39 nits: hgam's fs select fit asserted equal to its no-select fit (exact); the `matrix_rank` vs `eps^0.66` tolerance is a DEFERRED line.
Tests: 785 → 789. `make lint` clean. Section 25 (tier 1, mgcv 1.9-1): REML cr+re eta 1.6e-5, se 4.6e-4; discrete cr+re eta 2.3e-8; REML fs+re eta 6.6e-7, se 7.2e-4; discrete fs+re eta 5.8e-14; edf at most 5.2e-3 of the gate width; predict on new data eta 2.7e-5, se 3.5e-4; structure, rank and names exact.
Deferred: nothing new. Next: `te` on cr.

## 2026-11-20 (thirty-first slice): `select=TRUE` with `fs` (fourth step of `fs`, PLAN step 3)

No open PR at start (PR #38 had merged). smoothCon's `null.space.penalty` branch read: fs takes `need.full`, its three penalties sum to full rank, so mgcv adds nothing; `null_space_penalty` agrees, and the `select` refusal for fs is dropped. The two select refusal cases in `test_fs_refusals` retired (SETTINGS condition 5), replaced by `test_fs_select_meets_the_gates` (section 24e).
Tests: 784 → 785 (three added, two refusal cases removed). `make lint` clean. Section 24e (tier 1, mgcv 1.9-1): REML gam eta 1.1e-4, se 4.3e-4, edf 1.3e-3 of gate; discrete bam eta 6.1e-14, se 3.7e-14; cr + fs eta 2.9e-5, se 4.4e-4, edf 1.6e-2 of gate; structure, rank, names exact; mgcv's fs fit equals its select=FALSE fit (eta difference 0).
Deferred: the fs null-pair sp swap beside s(x) with select (DEFERRED). Next: `re` beside fs and in discrete bam.

## 2026-11-17 (thirtieth slice): `fs` in discrete bam (third step of `fs`, PLAN step 3)

PR #37 auto-merged (six conditions posted; the branch delete was refused by the proxy). `bam(discrete=TRUE)` builds fs on `discrete.mf`'s `mf0`: `fs_basis(frame=)` takes `nat.param` and the penalty scaling over its rows (`fs_frame`, read through `constraint_frame`), and `TermSpec.margin_order` keeps the written order for compression. Refusal test `test_fs_refused_by_bam` retired (SETTINGS condition 5), replaced by `test_fs_discrete_bam_meets_the_gates`.
Tests: 783 → 784. `make lint` clean. Section 24d (tier 1, mgcv 1.9-1), both written orders: eta 6.1e-14, se 3.7e-14, P 1.3e-14, predict 6.3e-14, Ref.df exact, Chi.sq 3e-14; built on the data rows instead, eta misses by 8.1e-3.
Deferred: fs with select=TRUE (refused); the 24c sp nit from PR #37 recorded. Next: select=TRUE with fs.

## 2026-11-17 (twenty-ninth slice): the `fs` fit (second step of `fs`, PLAN step 3)

No open PR at start (PR #36 had merged). `s(x, f, bs="fs", xt="cr", k=)` wired into the REML gam path: parser, `TermSpec(basis="fs")`, `build_python_fs_term`, predict through `Predict.matrix.fs.interaction` with `P` kept in `TermState.fs_p`; no engine change to the fit itself. `test_fs_is_still_refused` retired (SETTINGS condition 5), replaced by `tests/test_fs_fit.py`.
Tests: 769 → 780 (eleven in `tests/test_fs_fit.py`, one parser case; one refusal test removed). `make lint` clean. Section 24b (tier 1, mgcv 1.9-1): eta 1.1e-4, se 4.3e-4 relative, edf 1.3e-3 of the gate width, predict on new data 1.1e-4; structure, names, rank exact.
Deferred: fs in discrete bam, fs with select=TRUE (refused). Next: fs in discrete bam.
Review follow-up (14e18a1): section 24c holds `s(x) + s(x, f, bs="fs")` and `f + s(x, f, bs="fs")` at every gate (eta 9.1e-5, 6.5e-7); k.check NA and null_space_dim 0 tested. Tests 780 → 783.

## 2026-11-17 (twenty-eighth slice): the `fs` basis (first step of `fs`, PLAN step 3)

PR #35 auto-merged (six conditions posted). `nat.param` (type 1) and `smooth.construct.fs.smooth.spec` on cr ported in `hgam.engine.gam_basis_fs`, with `Predict.matrix.fs.interaction`; `gam()` still refuses `bs="fs"`. The full fit would exceed the slice limit, so this is the basis only.
Tests: 767 → 769 (`tests/test_fs_basis.py`, two). `make lint` clean. X, P, S and PredictMat 2.4e-14 relative under one sign-and-null-order column map (section 24, tier 1, mgcv 1.9-1).
Deferred: the null-space order is set by rounding (`dsyevr` needed; mgcv's fit is invariant to the swap, measured 2.8e-14); unused factor levels (PR #35 nit). Next: the fs fit.

## 2026-11-17 (twenty-seventh slice): factor `by` on cr, first widening slice (PLAN step 3)

PR #34 had merged (Jonathan). Section 23 holds `s(x, by=f, bs="cr")` to `smoothCon`'s factor-by branch and to `f + s(x, by=f)` as a REML gam, a discrete bam and a discrete bam with `select=TRUE` plus predict on new data; the existing build already agreed, so no engine change. An ordered categorical is now refused by name.
Tests: 762 → 767 (`tests/test_cr_by_factor.py`, five). `make lint` clean. Basis 1.7e-14; eta 5.7e-7 (gam) and 1.3e-13 (bam); se 5.6e-4 relative; edf diff 2.3e-4 of the gate width; structure, names and rank exact (tier 1, mgcv 1.9-1).
Deferred: ordered factors (DEFERRED line). Next: `fs`.

## 2026-11-17 (twenty-sixth slice): `nthreads` as a BLAS cap (Priorities item 3, PR #31 option A)

PR #33 auto-merged (six conditions posted; the branch delete was refused by the proxy again). `hgam.bam(..., nthreads=1)` maps bam's `nthreads` (read in bam.r 1.9-4: scalar expanded to `rep(n, 2)`, non-finite or < 1 becomes 1) onto `threadpool_limits(n, user_api="blas")` for the length of the fit, through the new extra `hgam[threads]`; without it only `nthreads=1` is accepted; the two-element form is refused.
Tests: 756 → 762 (`tests/test_nthreads.py`, six). `make lint` clean. Exit fit meets its gates at `nthreads=1` and `4` (the two differ by 3.3e-8 in eta). Benchmark: median 0.49 s at the default, 1.98 s at `nthreads=4` (mgcv 0.98 s).
`threadpoolctl` also went into the `dev` extra so CI exercises the cap; that and the extra wait for Jonathan's merge (SETTINGS condition 4).

## 2026-11-17 (twenty-fifth slice): the unseen-level refusal (PR #32 review b087411, should-fix 1)

A factor level the fit did not see is now refused at predict time as not ported (`HgamNotImplementedError`, in `_FactorCoding.encode(at_predict=True)` and the engine's `_codes`), naming mgcv's behaviour (`predict.gam` warns and recodes to NA; `predict.bamd` zeroes the dummies, read in mgcv 1.9-4). The fit-time message for a `Factor` value outside its stated levels no longer claims mgcv refuses it.
Tests: 756 pass before and after, no xfail (assertions added inside existing tests: discrete=False predict and the fit-time `Factor` case). `make lint` clean. No key result changes; no fixture needed for a refusal.
Deferred: unseen-level handling itself (DEFERRED line updated).

## 2026-11-17 (twenty-fourth slice): initial-scope confirmation PR

PR #31 auto-merged (six SETTINGS conditions posted; the branch delete was refused by the proxy). PR #24 has merged, so I followed PLAN.md step 2: a PR that changes only routine files and asks Jonathan to confirm the initial scope, with a checklist of each item, its test, fixture and numbers.
Tests: 756 pass before and after, no xfail. `make lint` is clean. Pinned-image run (CI `oracle`, PR #31 head): 756 pass against the tier 3 fixtures.
The numbers quoted are tier 1 (mgcv 1.9-1). CI does not print the tier 3 margins, and its artifacts are blocked here.
Deferred: nothing new.
Review b087411: should-fix 1, the unseen-level refusal is a mislabelled `HgamValidationError`; it is moved out of the checklist's refusals, and a slice is scheduled in CONTINUATION. Should-fix 2: the exit row was remeasured (eta 7.5e-8, se 7.2e-7, edf 1.4e-6).
At Jonathan's request, CI's `oracle` job now prints the key-result margins (`margin` fixture, `HGAM_REPORT_MARGINS`). This found that eight M4 test modules read the committed tier 1 fixtures, not the oracle's. They now read `conftest.FIXTURES`. Locally: 756 pass; the margin table matches the checklist's (15b eta 2.2e-8, below the 4.0e-5 recorded before the constraint rows).
Pinned image (`oracle` 134048849560): 756 pass with every module on tier 3; margins in the PR #32 checklist. Jonathan confirmed the initial scope (option (a)); the next slice is the unseen-level refusal.

## 2026-11-17 (twenty-third slice, follow-up): Jonathan's answer on PR #31

Jonathan chose option (A) as a setting: `nthreads` (default 1) caps BLAS through `threadpoolctl` inside `bam_fitd`; large data (about 30m rows) is deferred. Recorded in CONTINUATION (next slice) and DEFERRED; the review nit's timing ranges are added to finding 0002. Docs only.
Tests: 756 pass before and after, no xfail. `make lint` is clean.

## 2026-11-17 (twenty-third slice): run time, profile and a question

No open development PR (PR #30 merged). Started Priorities item 3. The profile shows that on four threads OpenBLAS makes X'WX 4.5x slower and the penalty `eigh` 20x slower inside `bam_fitd`. A scoped cap needs a decision (threadpoolctl is "Always ask"), so this PR asks and adds only the README note and a finding addendum. The PR #15 sz nit was already handled.
Tests: 756 pass before and after (docs only), no xfail. `make lint` is clean.
Run time, this machine: default threads median 3.07 s (7 fits), `OPENBLAS_NUM_THREADS=1` 0.57 s; mgcv 0.98 s (finding 0002).
Deferred: nothing new.

## 2026-11-17 (twenty-second slice): gam.check's report

PR #29 (k.check's ti branch) auto-merged (six SETTINGS conditions posted); its branch delete was refused by the proxy. Ported `gam.check` (plots.r) without plots as `GamFit.gam_check(data, seed=)`: `outer.info` grad/hess carried from `bam_fitd`, model rank, `qq.gam`'s ten draws, then k.check; section 22, `tests/test_gam_check_bam.py`. PR #29 nit: tie test at relative 1e-12. Fixed a garbled heading in CONTINUATION from the previous slice.
Tests: 747 pass before; 756 pass, no xfail after (+6 gam.check, +3 tie grids). `make lint` is clean.
Tier 1, mgcv 1.9-1: method/optimizer/rank 132/132 exact; k table k' exact, edf 1.4e-6, k-index 1.3e-10, p exact; grad 1.1e-5, Hessian 6.2e-6 absolute (not key results).
Deferred: plots; R's print formatting of outer.info and printCoefmat's digit rules. M4 is done for the exit fit.

## 2026-11-17 (twenty-first slice): k.check's tensor branch

No open development PR (PR #28 merged). Ported k.check's ti branch (plots.r): `gr.f` finite-difference rescaling through `predict_term_design`, then 3 nearest neighbours via SciPy `cKDTree` in place of `k_nn`; ties at the third neighbour refused. `tests/test_ti_is_not_ported` replaced by `test_ti_tensor_branch` (fixture-backed) plus `test_ti_neighbour_tie_is_refused`.
Tests: 746 pass before; 747 pass, no xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1 (section 21, unchanged): ti(PolYear,AttdAge) k-index 1.3e-10 relative, p-value exact (0.7875; subsample branch 0.88).
Deferred: mgcv's kd-tree tie order (refused). Next: gam.check's printed report.

## 2026-11-17 (twentieth slice): k.check, gam.check's first step

PR #27 (vcov) auto-merged after both reviews (six SETTINGS conditions posted); its branch delete was refused by the proxy (403). Ported `k.check` (plots.r) as `GamFit.k_check(data, seed=)` with `deviance_residuals` (residuals.gam) and `Family.dev_resids`; section 21, `tests/test_k_check_bam.py`.
Tests: 739 pass before; 745 pass, no xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: labels and k' exact, edf 1.4e-6 at most, deviance residuals 1.8e-7, k-index 6.8e-10 relative, p-value exact (R's draws) on s(AttdAge), s(PolYear), s(AttdAge):StudyYear_C; sz NA as mgcv.
Deferred: the ti branch of k.check (PredictMat scaling and the k_nn kd-tree); gam.check's printed report.
Review 7c35e5c should-fix 1: section 21 adds `k.check(b14, subsample = 1000, n.rep = 50)`; the subsample branch gives k-index and p exactly as mgcv on the three 1-D terms. 746 pass.

## 2026-11-17 (nineteenth slice): vcov on the exit fit

Ported `vcov.gam` (mgcv.r): `GamFit.vcov(freq=, dispersion=)` gives Vp, Ve (`Sl.postproc`'s `F %*% Vp`) and the dispersion rescaling; `GamFit.coefficient_names` gives `names(b14$edf)`. Section 20, `tests/test_vcov_bam.py`. PR #26 had merged; PR #24 (Jonathan's planning PR) is open and not a development PR.
Tests: 727 pass before; 737 pass, no xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: Vp diagonal 8.3e-6 relative, Ve diagonal 8.7e-5, dispersion = 2 alike (gate 1%); names exact on 132 coefficients.
Deferred: `unconditional=True` on a discrete fit is refused (mgcv's Vc is Sl.postproc's sp-corrected one, up to 4.07x Vp); the off-diagonal is reported, not gated.
Review 4003b4e: should-fix 1, section 20b adds a gaussian REML fit: names exact, `dispersion = 2` diagonal 3.1e-6 (Vp 3.0e-6); should-fix 2, Vc is up to 4.07x Vp, not 3.1x. 739 pass.

## 2026-11-17 (eighteenth slice): the smooth-term p-value in summary()

PR #25 auto-merged (six SETTINGS conditions checked and posted). `re_test` now returns `reTest`'s p-value, `psum_chisq(stat, ev)` (mgcv.r), and `summary()` of a discrete fit reports it (and prints it). New test in `tests/test_summary_bam.py`; no new fixture (section 16 already holds `s_table[:, 3]`).
Tests: 726 pass before; 727 pass, no xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: p-value 3.2e-7 absolute at most on the eight exit smooths (gate 1e-3 or 1%); five are 0 in both.
Deferred: the estimated-scale p-value (`psum.chisq(0, c(ev, -stat/k), ...)`) stays refused with `sig2 != 1`.
Review 4fb2174 should-fix 1: the text report's footer depends on whether every p-value is filled (it said psum.chisq was not ported); both report tests updated. 727 pass.

## 2026-11-17 (seventeenth slice, review follow-up): PR #25 review b892122

Should-fix 1: three section 19 cases (`aux_lim`, `cfe_fail`, `cfe_clamp`) hold davies.c's auxiliary-loop `lim` exit (p = 2, ifault 0, as mgcv), `cfe`'s fail return and `axl` clamp to mgcv. Nit 1: DEFERRED's liu2 line says chi2 only and lists the remaining unreached branches.
Tests: 718 pass before; 724 pass, no xfail after (+6, the new cases in two parametrised tests). `make lint` is clean. Only `psum_chisq_davies.json` changed on regeneration.
Tier 1, mgcv 1.9-1: the new cases meet the existing assertions (p 1e-8, trace counts exact).
Merged with this run's own fix of the same review: a fourth case, `negative_nc` (davies.c's ifault = 3, p = NA as mgcv). Tests: 726 pass, no xfail.

## 2026-11-17 (seventeenth slice): psum.chisq and Davies' algorithm

Ported mgcv's `davies.c` (`hgam.frontend.davies`: errbd, ctff, truncation, findu, integrate, cfe, davies, with R's `revsort` and `log1pmx`) and `psum.chisq` / `liu2` (mgcv.r) into `smooth_tests`. New fixture section 19 (`psum_chisq_davies`, 15 cases plus liu2), `tests/test_psum_chisq.py`. Not yet wired into `summary()`.
Tests: 678 pass before; 718 pass, no xfail after (+40, all in `tests/test_psum_chisq.py`). `make lint` is clean.
Tier 1, mgcv 1.9-1: p-value 4.4e-16 absolute at most (gate 1e-3 or 1%); davies' trace 4.3e-15 relative, its term, integration and call counts and ifault exact on every case (aux integration, convergence factor, range exits, ties, liu2 fallback).
Deferred: Davies' ifault 2 branch has no fixture; liu2 uses SciPy's ncx2 for pnchisq.

## 2026-11-17 (sixteenth slice): summary.gam's smooth-term table, Chi.sq and Ref.df

PR #22 auto-merged (all six SETTINGS conditions checked and posted). Ported `reTest`, `recov` (no random terms) and `mroot` (mgcv.r) into `hgam.frontend.smooth_tests`; `summary()` of a discrete fit reports Chi.sq and Ref.df for fully penalised smooths; `bam_fitd` now carries X'WX. New fixture section 18 (`mroot_chol`); `tests/test_smooth_tests.py`; smooth-table tests in `tests/test_summary_bam.py`. PR #22's review nit (r_rng docstring) fixed.
Tests: 665 pass before; 678 pass, no xfail after (Jonathan's option (a): sz Ref.df reported, not gated) (review fix: smooths beside a `bs="re"` term keep None, `recov`'s re branch not ported; null.space.dim by mgcv's select/re rule). `make lint` is clean.
Tier 1, mgcv 1.9-1: Chi.sq 1.7e-6 relative on all eight exit smooths; null.space.dim exact; Ref.df exact on five, the three sz terms 7/6/5 vs 11/7/6 (rounding-set rank; p with hgam's eigenvalues 3e-6 from mgcv's): not gated, option (a). mroot 5.8e-15.
Deferred: p-value (`psum.chisq`, Davies), `testStat`, random-effect terms in `recov`, the `gam()` path.

## 2026-11-17 (fifteenth slice): discrete bam's constraint rows

Ported R's RNG (`set.seed`, Mersenne-Twister, `sample.int` Rejection: RNG.c, random.c) and `discrete.mf`'s padded frame `mf0` (bam.r, `discrete_model_frame`); cr and ti-margin constraints in `fit_engine_bam` are `colMeans` over it, as `smoothCon` takes them (ADR 0002). New fixture section 17, `tests/test_discrete_constraint_rows.py`; strict xfail lifted.
Tests: 647 pass and 1 strict xfail before; 665 pass, 0 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: R draws exact; `mf0` 5.7e-14; constraint directions 1.2e-15 (gate 1e-8). Exit fit: intercept 4.8e-7 (was 3.6e-3), eta 7.4e-8 (was 1.1e-4), se 7.0e-7, edf 1.3e-6.
Deferred: an offset beside weights is refused in a discrete fit; 12d and 13a eta moves 1e-9 to 3e-6 with the summation order alone; 12a's ti sp still 1.11 times mgcv's (eta 2e-13).

## 2026-11-16 (fourteenth slice): summary of a discrete fit, the parametric table

Ported `summary.gam`'s parametric table (mgcv.r): `GamSummary.parametric` (Estimate, Std. Error from Vp, z, 2 * pnorm), mgcv's coefficient names (`GamFit.parametric_names`); `summary()` now accepts a `discrete=True` fit. New fixture section 16 (`bam_discrete_exit_summary`), `tests/test_summary_bam.py`.
Tests: 644 pass before; 647 pass and 1 strict xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: contrasts 8.1e-7, se 1.8e-5 relative, edf 2.2e-3, names exact. Intercept 3.6e-3 (gate 1e-3): it is s(AttdAge)'s level, the constraint row set of discrete bam (strict xfail; question on the PR).
Deferred: smooth-term tests (Ref.df, Chi.sq, p: testStat / reTest / psum.chisq) to the next slice; estimated-scale p.table not ported.

## 2026-11-16 (thirteenth slice): predict on new data from a discrete fit

Ported `predict.bamd`'s re-discretisation (bam.r): `GamFit.predict` re-bins new data with `discretise_frame` for a `discrete=True` fit; `discrete=False` predicts at raw covariates. New fixture section 15 (`bam_discrete_exit_predict_a`, `_b`), `tests/test_predict_bam.py`. Also fixed review nit 1 of PR #19 in finding 0002.
Tests: 640 pass before; 644 pass after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 15a (binned) eta 8.4e-5, se 2.3e-4 relative; 15b eta 4.0e-5, se 2.2e-4. Without re-binning 15a missed by 3.4e-3.
Deferred: unseen factor levels refused (mgcv zeros them); `type="terms"` / `"lpmatrix"` not ported.

## 2026-11-16 (twelfth slice): the M3 benchmark of the exit model

Timed the exit model, `hgam.bam` vs `mgcv::bam(discrete = TRUE, select = TRUE)`, median of 15 after a warm-up (`docs/findings/0002-exit-benchmark.md`, scripts in `docs/findings/exit_benchmark/`). No engine change.
Tests: 640 pass before and after. `make lint` is clean.
mgcv 1.9-1 0.98 s; hgam 2.2 to 2.4 s at OpenBLAS's default 4 threads, 0.42 s on one thread. Inside the ten-times rule; nothing tuned.
Deferred: BLAS thread oversubscription on small problems, after M4.

## 2026-11-16 (eleventh slice): hgam.bam / hgam.gam wired; the exit test passes

`hgam.gam` (mgcv.r) and `hgam.bam` (bam.r) forward to the front end (`family=` + `link=`, unported methods refused); the response `a / b` (R's `model.frame`) is parsed and evaluated. `tests/test_exit.py` fits `hgam_formula` as written on section 14's data and knots, held to every gate; strict xfail lifted. New `tests/test_entry_points.py`.
Tests: 628 pass and 1 xfail before; 640 pass and 0 xfail after (review fix: hgam.gam defaults to mgcv's method="GCV.Cp" and refuses it). `make lint` is clean.
Tier 1, mgcv 1.9-1: the exit fit equals section 14's (eta 1.1e-4, se 2.6e-4 relative, edf 2.2e-3, structure exact vs sections 3 and 14). Also fixed PR #17's review nit (sp attribution in DEFERRED and the log).
Deferred: nothing new.

## 2026-11-16 (tenth slice): the exit formula on bam(discrete=TRUE, select=TRUE)

New fixture section 14 (`bam_discrete_exit_formula`: section 3's `b` refitted with response column p = DthCnt / ExposCnt, exact bam, rank and parametric edf beside it) held in `tests/test_bam_discrete_fixtures.py`; no engine change.
Tests: 627 pass and 1 xfail before; 628 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: eta 1.1e-4 (mgcv's own discrete vs exact gap 7.0e-3), se 2.6e-4 relative, per-term edf 2.2e-3 (FaceSize, Smoke, FaceSize:Smoke included), rank 132 of 132. Structure exact.
Deferred: sp of the ti term (6-14%), of s(Smoke,PolYear) (x1.96) and of s(FaceSize,PolYear) (x0.62; attribution corrected after the PR #17 review), not gated.

## 2026-11-16 (ninth slice): bam(discrete=TRUE, select=TRUE) on f + sz and on a numeric by

New fixture section 13 (13a: section 11a's `f + s(x) + s(f, x, sz)`, 13b: section 10's numeric by, both `bam(discrete = TRUE, select = TRUE)`, exact bam beside them) held in `tests/test_bam_discrete_fixtures.py`; no engine change.
Tests: 625 pass and 1 xfail before; 627 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 13a eta 1.0e-9, se 1.6e-7, edf (factor included) 3.1e-8, rank 20 of 20; 13b eta 4.7e-14, se 1.3e-13, edf 3.2e-7. Structure exact.
Deferred: mgcv's discrete = FALSE reference fits for 12d and 13a warn "algorithm did not converge" (informational only).

## 2026-11-16 (eighth slice): bam(discrete=TRUE) on cr + ti and on sz

Ported the one-covariate-margin subset of `discrete.mf` (bam.r) as `discretise_frame`; `fit_engine_bam` now fits on binned covariates and accepts sz. New fixture section 12 (12a cr + ti, 12b sz, 12c/12d with select=TRUE, 12e numeric by (review), each with mgcv's discrete = FALSE bam beside it), `tests/test_bam_discrete_fixtures.py`.
Tests: 619 pass and 1 xfail before; 625 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1, vs mgcv's discrete bam: 12e eta 1e-13; 12a eta 3.7e-4, se 1.2e-3, edf 7.1e-3; 12b eta 2e-6; 12c eta 3.6e-4, se 1.3e-3, edf 6.6e-3; 12d eta 1.5e-9. Structure exact. Before binning, 12a matched mgcv's exact bam (8e-9) and missed its discrete bam (4.0e-3).
Deferred: 12a ti sp ~10% off (constraints over data rows vs mgcv's padded frame); predict on new data does not re-bin.

## 2026-11-16 (seventh slice): select=TRUE on the factor + sz shapes

New fixture section 11 (8a and 8b with `select=TRUE`) and a test in `tests/test_rank_deficient_sz.py`; no engine change. The null-space penalties identify the shared directions: mgcv's rank equals the coefficient count and hgam pivots nothing.
Tests: 617 pass and 1 xfail before; 619 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 11a eta 8e-5, se 3.6e-4, per-term edf 8.5e-4; 11b eta 3.0e-4, se 9.2e-4, per-term edf 3.6e-3. Structure exact.
Deferred: 11a's near-linear s(x) sp (log10 7.40 vs 7.83, flat ridge); 11b's mgcv convergence message differs on the JSON round trip.

## 2026-11-16 (sixth slice): numeric by on a cr smooth

Held `s(x, by=z, bs="cr")` to mgcv (`smoothCon`'s numeric-by branch, smooth.r); constant `by` (mgcv keeps the constraint) is now refused. New fixture section 10 (10a smoothCon, 10b fit, 10c fit with select=TRUE), `tests/test_cr_by_numeric.py`.
Tests: 613 pass and 1 xfail before; 617 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: basis X 1.2e-14, S 2.1e-15 relative, C 0 rows, labels and 12 coef exact. 10c: eta 2.9e-5, edf 4e-4, se 2.1e-4. 10b: edf 0.029, se 0.79%, eta at mgcv's sp 1e-13, but eta at estimated sp 1.24e-3 (gate 1e-3): mgcv's Newton stops short of its own optimum (hgam is 1.9e-4 from mgcv at conv.tol 1e-12).
Jonathan chose option B: 10b now records mgcv at newton conv.tol 1e-12 (REML 2441.46465 vs default 2441.46481) and eta is gated against it, 1.9e-4 (se 9.7e-4, edf 3.5e-3).

## 2026-11-16 (fifth slice): select=TRUE with sz smooths

Accepted `select=TRUE` with sz (smoothCon's `null.space.penalty` need.full branch, smooth.r); the engine's `null_space_penalty` already built it, so only the front-end refusal went. New fixture section 9 (9a smoothCon S with null.space.penalty, 9b section 7's fit with select=TRUE).
Tests: 610 pass and 1 xfail before; 613 pass and 1 xfail after (the refusal test is replaced by an acceptance test). `make lint` is clean.
Tier 1, mgcv 1.9-1: sz null-space penalty 1.1e-14 / 5.7e-14 relative, ranks exact; fit: 6 sp and 18 coef exact, eta 9.2e-5, edf 1.5e-3, se 3.8e-4 relative.
Deferred: the near-linear s(x) penalty's sp (log10 7.29 vs 8.16, flat ridge).

## 2026-11-16 (fourth slice): a factor main effect beside sz smooths

New fixture section 8 (8a: `f + s(x) + s(f, x, sz)`; 8b: `f` with two sz smooths of a two-level factor) and `tests/test_rank_deficient_sz.py`; no engine change.
Tests: 608 pass and 1 xfail before; 610 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: eta 1.0e-4 / 3.6e-5, se 5.8e-4 / 3.7e-4 relative; edf of the involved terms summed 2.1e-3 / 1.2e-3; uninvolved terms inside the gate.
Not met: the per-term split inside the involved terms. mgcv's `gdiPK` choice depends on an arbitrary eigenvector basis (39 of 200 rotations change it): question for Jonathan, DEFERRED.

## 2026-11-16 (third slice): the sz basis

Accepted `s(f, x, bs="sz", k=, xt=list(bs="cr"))` in the front end (mgcv `smooth.construct.sz.smooth.spec` + `smoothCon`'s `XZKr`), knots shared as for cr; other xt, x-first order, `by=` and `select=TRUE` with sz are refused. New fixture section 7.
Tests: 596 pass and 1 xfail before; 608 pass and 1 xfail after (one refusal case in `test_formula.py` now names the still-refused no-xt form). `make lint` is clean.
sz basis vs `smoothCon` (2 and 3 levels): X 1.5e-14, S 1.6e-15 relative, ranks and bs.dim exact. sz fit (tier 1, mgcv 1.9-1): eta 1.0e-4, edf 2.1e-3, se 5.8e-4 relative; default knots would miss eta (1.7e-2).
Deferred: sp differences (log10 units; corrected after review); `f + sz` and two-sz-per-factor rank deficiency (per-term edf split differs from mgcv, next slice); the -2 lower bound on log10 lambda.

## 2026-11-16 (second slice): user-supplied cr knots

Added `knots=` to `hgam.frontend.gam` (mgcv `smooth.construct.cr.smooth.spec`): shared by a variable's cr smooths and ti margins, count must equal k, refused for tp/re and for unused names. New fixture section 6; section 2 now records `labels` (PR #5 nits).
Tests: 586 pass and 1 xfail before; 596 pass and 1 xfail after (incl. review follow-up: by-form knot tests). `make lint` is clean.
cr basis at supplied knots vs `smoothCon`: 1e-14 relative. Knots fit (tier 1): eta 3.3e-5, edf 2.8e-4, se 2.4e-4 relative; the same fit with default knots misses the eta gate (7.6e-3), so the knots are used.
Corrected PR #5's sp claim: 7e-5 on the Poisson fixture, not 1e-7.

## 2026-11-16: M1, outer REML search held to mgcv

Measured the moved Newton REML search against two `gam(method="REML")` fixtures (new: cr + ti binomial-cloglog with weights, section 5; existing: Poisson offset, section 2) instead of porting `newton`: it meets every gate, so it stays ("reuse before porting").
Tests: 583 pass and 1 xfail before; 586 pass and 1 xfail after. `make lint` is clean.
Largest differences (tier 1, mgcv 1.9-1): eta 5.7e-10 / 9.6e-7, edf 1.1e-8 / 2.9e-5, se 8.6e-4 / 5.9e-6 relative; sp agree to 1e-7.
Deferred: the REML score's normalising constant (not a key result) and the 8.6e-4 se difference (inside its gate).

## 2026-11-15: M0, move the engine

Moved 28 engine and front-end modules (about 10,800 lines) from polaris-re into `hgam.engine` and `hgam.frontend`, with no `polaris_re` imports left; dropped polars for NumPy columns; added mgcv counterpart lines to the six ported functions.
Tests: 5 pass and 1 xfail before; 564 pass and 1 xfail after. `make lint` is clean.
No key result was measured: this slice adds no fixture. 43 harness-bound tests stayed behind (`docs/DEFERRED.md`).
The tier-3 fixture artifact could not be downloaded: the network policy blocks its host.
After review: unported refusals raise `HgamNotImplementedError`; the six ports gained tier 1 fixtures (`tero`, `compress.df`, `initial.sp`, `Sl.fitChol`, `bgam.fitd` x2), all inside their gates (eta within 3e-9, edf within 3e-7). 583 pass.

## 2026-11-15: routines and handoff

Added the daily-development and review routines and their settings file, split the engine move into three
steps, and recorded in ADR 0001 that the project is the maintainer's own, with no employer sign-off pending.

## 2026-11-14: scaffold

Created the repository skeleton: package stub, GPL-2.0 licence text, CI, the fixture generator and
three tier 1 fixtures, the exit test as a strict xfail, scope, plan and ADR 0001.
Recorded the mgcv-rust check under `docs/findings/`.
