# hgam

Hierarchical generalized additive models in Python: a GPL port of the hierarchical-GAM core of R's
[mgcv](https://cran.r-project.org/package=mgcv), tested against mgcv itself.

**Status: pre-alpha.** `hgam.bam` fits with `method="fREML", discrete=True` only, and `hgam.gam` with
`method="REML"` only (it must be written out; mgcv's default `GCV.Cp` is refused). Every other method, term,
family or option is refused by name with `NotImplementedError`. The exit test, the target model fitted through
`hgam.bam`, meets every gate against mgcv. What is still to come is in [docs/PLAN.md](docs/PLAN.md).

## What it is for

Models of the form mgcv users write as `s(x) + s(x, by = f) + ti(x, z) + s(f, x, bs = "sz")`: smooths
that vary by group, with shared and group-level shapes. The acceptance test is one such model, fitted
exactly as written in R. See [docs/SCOPE.md](docs/SCOPE.md).

## Licence

hgam is a port of mgcv, which is licensed GPL (>= 2), so hgam is distributed under
**GPL-2.0-or-later**. mgcv is by Simon N. Wood. See [docs/adr/0001-licence.md](docs/adr/0001-licence.md).

## Development

```bash
uv sync --extra dev
make test      # pytest; the exit test is a strict xfail until the port reaches it
make lint
make fixtures  # regenerate mgcv reference fixtures (needs R with mgcv)
```

Reference numbers come from mgcv run inside the pinned
[R-Gam-base](https://github.com/jonathancrawford05/R-Gam-base) image and are committed under
`tests/fixtures`, so the test suite needs no R.
