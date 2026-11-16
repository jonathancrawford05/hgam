# Continuation

Last updated: 2026-11-16, by the development run that held the REML search to mgcv. Read `docs/PRIORITIES.md`
before choosing work.

## State

M0 is done (engine in `src/hgam/engine/`, front end in `src/hgam/frontend/`; errors in `hgam.errors`). The six
mgcv-ported functions are held to fixtures in `tests/test_ported_functions.py`.

M1, outer smoothing-parameter search: measured, not ported. `tests/test_gam_reml_fixtures.py` holds
`hgam.frontend.gam(method="REML")` to two `mgcv::gam` REML fixtures (`gam_poisson_offset_reml`, section 2;
`gam_binomial_cloglog_ti_reml`, section 5: cr + ti, binomial cloglog, prior weights). Both meet every gate
(eta within 1e-6, edf within 3e-5, se within 8.6e-4 relative; sp agree to 1e-7), so the moved Newton search
stays and mgcv's `newton` is not ported unless a later fixture fails a gate on the search.

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 586 pass,
1 strict xfail. What did not move or differs is in `docs/DEFERRED.md`.

## Next slice: user-supplied knots for cr

The exit test passes `knots = my_knots` (cr margins for `AttdAge` and `PolYear`), and the front end has no
`knots=` argument. Add `knots: Mapping[str, Sequence[float]] | None` to `hgam.frontend.gam`, following
`smooth.construct.cr.smooth.spec` (smooth.r, mgcv 1.9-4): supplied knots are used as given when their count
equals `k`, otherwise refuse (mgcv errors). Hold the `s(x, bs="cr", k=6)` basis and penalty to the existing
`smoothcon_cr_k6_knots` fixture (1e-8 relative, raw and absorbed). Then extend section 5's fixture pattern with a
`gam(..., knots=)` fit on cr + ti and test eta/edf/se against the gates. Leave `knots=` refused for every other
basis.

After that, the exit test still needs: the `sz` basis (`smooth.construct.sz.smooth.spec`, refused today),
`xt=list(bs="cr")`, `select=TRUE` at the exit model, numeric `by`, and `bam(discrete=TRUE)` on `ti`.

## Open items for Jonathan

None.

## Open questions

None.
