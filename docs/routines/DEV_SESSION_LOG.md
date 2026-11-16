# Development session log

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
