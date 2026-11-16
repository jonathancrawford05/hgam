# Continuation

Last updated: 2026-11-16, by the development run's second slice (user-supplied cr knots). Read
`docs/PRIORITIES.md` before choosing work.

## State

M0 is done (engine in `src/hgam/engine/`, front end in `src/hgam/frontend/`; errors in `hgam.errors`). The six
mgcv-ported functions are held to fixtures in `tests/test_ported_functions.py`.

M1, outer smoothing-parameter search: measured, not ported (PR #5). `tests/test_gam_reml_fixtures.py` holds
`hgam.frontend.gam(method="REML")` to three `mgcv::gam` REML fixtures (sections 2, 5 and 6 of
`oracle/make_fixtures.R`); all meet every gate, so mgcv's `newton` is not ported unless a later fixture fails a
gate on the search.

User-supplied knots: `hgam.frontend.gam(..., knots={"x": [...]})` passes them to every cr smooth of that variable
and every ti margin on it; the count must equal k (mgcv stops otherwise). Refused for tp/re terms and for names
no smooth uses. Held by `tests/test_cr_knots.py` (basis at supplied knots vs `smoothCon`, 1e-14) and the
section 6 fit (eta 3.3e-5).

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 593 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice: the sz basis

The exit test has four `s(f, x, bs="sz", k=, xt=list(bs="cr"))` terms, and the front end refuses `bs="sz"` and
`xt=` (`src/hgam/frontend/formula.py`, `_REFUSED` "sz"). The engine already has `sz_basis` and an sz term
builder (`gam_term_build.py`), verified only at fixed sp. Port path: `smooth.construct.sz.smooth.spec`
(smooth.r, mgcv 1.9-4). Steps:
1. Fixture (new section 7): `smoothCon(s(f, x, bs="sz", k=6, xt=list(bs="cr")), knots=...)` X and S (absorbed),
   for a 2- and a 3-level factor, and a `gam(method="REML")` fit `y ~ f + s(x, bs="cr") + s(f, x, bs="sz", ...)`
   with sp, edf per term, eta, se and labels.
2. Hold `sz_basis` to the basis fixture at 1e-8; accept `bs="sz"` with `xt=list(bs="cr")` only (refuse other
   xt) in the parser and `_build_model`, with knots shared as for cr.
3. Test the fit against the gates.

After that, the exit test still needs: `select=TRUE` at the exit model, numeric `by`, and
`bam(discrete=TRUE)` on `ti`.

## Open items for Jonathan

None.

## Open questions

None.
