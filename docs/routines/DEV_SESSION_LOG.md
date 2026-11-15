# Development session log

## 2026-11-15: M0, move the engine

Moved 28 engine and front-end modules (about 10,800 lines) from polaris-re into `hgam.engine` and `hgam.frontend`, with no `polaris_re` imports left; dropped polars for NumPy columns; added mgcv counterpart lines to the six ported functions.
Tests: 5 pass and 1 xfail before; 564 pass and 1 xfail after. `make lint` is clean.
No key result was measured: this slice adds no fixture. 43 harness-bound tests stayed behind (`docs/DEFERRED.md`).
The tier-3 fixture artifact could not be downloaded: the network policy blocks its host.

## 2026-11-15: routines and handoff

Added the daily-development and review routines and their settings file, split the engine move into three
steps, and recorded in ADR 0001 that the project is the maintainer's own, with no employer sign-off pending.

## 2026-11-14: scaffold

Created the repository skeleton: package stub, GPL-2.0 licence text, CI, the fixture generator and
three tier 1 fixtures, the exit test as a strict xfail, scope, plan and ADR 0001.
Recorded the mgcv-rust check under `docs/findings/`.
