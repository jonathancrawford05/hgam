# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

A Python port of the hierarchical-GAM core of R's mgcv, under GPL-2.0-or-later. The goal is that a
hierarchical GAM written for `mgcv::gam` or `mgcv::bam` gives the same key results here, at least as fast.

**Read `docs/PRIORITIES.md` first.** Alignment on key results comes first; build time and attention come
second; everything else waits.

## Start every session here

1. Read `docs/PRIORITIES.md`, `docs/routines/CONTINUATION.md` (where the last session stopped) and `docs/PLAN.md` (milestones).
2. Run `make test` and confirm the baseline before changing anything.
3. At the end: append to `docs/routines/DEV_SESSION_LOG.md` and rewrite `CONTINUATION.md`.

## Ground rules

- **This is a port. Read the mgcv source and follow it.** The licence allows it (ADR 0001). Prefer
  mgcv's algorithm, constants and iteration order over a re-derivation: agreement with mgcv depends on
  following the same path. Source: https://github.com/cran/mgcv, at the version the oracle pins.
- **Every ported function names its counterpart.** The docstring's first lines give the mgcv function
  and file it ports (for example ``mgcv counterpart: bgam.fitd (bam.r)``) and the mgcv version read.
- **Mirror mgcv's function boundaries** so each port can be tested against its counterpart alone.
- **Stop at the gate.** Meet the gates in `docs/PRIORITIES.md`, then move on. Differences in anything that is
  not a key result go in `docs/DEFERRED.md` and do not block.
- **Alignment is measured, never asserted.** A ported function ships with a fixture produced by
  `oracle/make_fixtures.R` and a test against it. Fixtures record the mgcv version and tier:
  tier 1 is a local R (provisional), tier 3 is the pinned oracle image in CI.
- **Never edit `tests/fixtures/*.json` by hand.** Change `oracle/make_fixtures.R` and regenerate.
- **Refuse rather than approximate.** A term, family or option that is not ported raises
  `NotImplementedError` naming it. Never accept and ignore an argument.
- One slice per pull request, one open development pull request at a time.
- No real experience data in this repository. Synthetic data only.
- Core dependencies stay at NumPy and SciPy. Anything else goes behind an optional extra.
- Decisions that a later session might reopen get a numbered ADR in `docs/adr/`.

## Commands

```bash
uv sync --extra dev
make test
make lint
make format
make fixtures   # needs R with mgcv; CI runs it inside the pinned image
```

## Definition of done

Tests green with no unexpected xfail, `make lint` clean, new functions carry their mgcv counterpart
and a fixture test, and `CONTINUATION.md` and the session log are updated.
