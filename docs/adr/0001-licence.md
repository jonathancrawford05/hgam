# ADR 0001: GPL-2.0-or-later, and porting from mgcv source

Status: accepted, 2026-11-14.

## Context

mgcv is licensed GPL (>= 2). The GPL treats a translation of a program into another language as a work
based on it. The engine this project starts from was written inside an MIT-declared repository under a
rule against transcribing mgcv source, and an audit on 2026-11-14 found six functions (382 lines) that
were nevertheless translations of mgcv 1.9-1 source. The same record shows that agreement with
`bam(discrete = TRUE)` could only be reached by reproducing mgcv's code path.

## Decision

hgam is a port of mgcv and is distributed under GPL-2.0-or-later. Sessions may read mgcv source and
port it directly. Every ported function names its mgcv counterpart and the version read.

## Consequences

- Anyone distributing hgam, or a program that combines it, must do so under the GPL.
- A permissively licensed project can depend on hgam as an optional package but cannot bundle it and
  stay permissive as a whole.
- Ownership: this is the maintainer's personal project. He has confirmed the port is not owned by his
  employer (2026-11-15), so no employer sign-off gates development or publication.
