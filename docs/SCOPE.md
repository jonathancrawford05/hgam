# Scope

Complete for hierarchical GAMs: roughly half of mgcv. mgcv 1.9-1 is about 33,000 lines of R and 19,500
of C with 21 smooth classes; this project does not target all of it.

## In

- `gam` (REML) and `bam` (fREML, with `discrete = TRUE`)
- Bases: `cr`, `cs`, `tp`, `ts`, `ps`, `cc`, `re`
- Constructions: `ti`, `te`, `t2`, `fs`, `sz`, `by` (numeric and factor)
- `select = TRUE`, user-supplied `knots`, offsets, prior weights, parametric terms and interactions
- Families: gaussian, binomial, poisson, Gamma, and the quasi families, with mgcv's links
- `predict`, `vcov`, `summary`, `gam.check`

## Out, at least initially

Location-scale and extended families, `gamm`, soap films, Markov random fields.

## What alignment means

The key results and their gates are in `docs/PRIORITIES.md`. Effective replication of what a user reads off
an mgcv fit, not bit-identical output.

## Exit test

The target model in `tests/test_exit.py` fits as written:

```r
bam(hgam_formula, family = binomial(link = "cloglog"), weights = ExposCnt,
    knots = my_knots, discrete = TRUE, select = TRUE)
```

On a synthetic frame mgcv gives 132 coefficients, 8 smooths and 21 smoothing parameters
(`tests/fixtures/exit_hgam_formula_structure.json`).
