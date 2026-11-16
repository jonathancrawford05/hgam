# Continuation

Last updated: 2026-11-16, by the development run's third slice (the sz basis). Read `docs/PRIORITIES.md` before
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

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 608 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice: per-term edf with a factor main effect beside its sz smooth

The exit formula has `FaceSize + Smoke + ... + s(FaceSize, AttdAge, bs="sz", ...)`: the factor main effect sits
inside the sz smooth's null space (a cr basis sums to one), so the model is structurally rank deficient. The
same holds between two sz smooths of one factor (`s(FaceSize, AttdAge, ...)` and `s(FaceSize, PolYear, ...)`),
each of whose cr bases spans the factor's contrasts; the fixture should cover both shapes. hgam
already meets eta, se and total edf there, but splits per-term edf differently from mgcv (DEFERRED: mgcv keeps
`fb` with edf 1 and zeroes `fc`; hgam drops both `f` columns). Per-term edf is a gated key result. Steps:
1. Read how mgcv chooses what to drop: `gam.fit3` (gam.fit3.r) passes `rank.tol` to C `gdi2` (gdi.c), which
   pivots after `gam.reparam`; find which columns end up unidentified and how their edf is reported.
2. Fixture (section 8): section 7's data with `f +` in the formula, exporting per-coefficient edf and coef.
3. Make the engine drop the same direction (`hgam.engine.gam_rank_pivot`), and hold per-term edf to the gate.
   If after one honest attempt it cannot meet the gate, ask Jonathan (an "Always ask" item).

After that, the exit test still needs: `select=TRUE` with sz terms and at the exit model, numeric `by`, and
`bam(discrete=TRUE)` on `ti` and sz.

## Open items for Jonathan

None.

## Open questions

None.
