# Finding 0001: mgcv-rust cannot express the target model

Date: 2026-11-14. Tested mgcv-rust 0.23.2 (latest on PyPI) against mgcv 1.9-1 (R 4.3.3) on two
synthetic 30,000-cell frames. Scripts are in `mgcv_rust_check/`. Measurements, tier 1.

## Rejected by the package

- `ti()`, the four `sz` terms and the `by` term: six of the eight smooths. Its formula parser accepts
  only `s(col, k=value)`; its bases are cr, bs, re and linear.
- `binomial(link = "cloglog")`: binomial is logit only.
- A proportion response with weights, for binomial and quasibinomial: it needs 0/1 rows.

## Accepted and ignored

`select=True`, `knots=` and `offset=` raise no error or warning and leave the fit unchanged.

## Agreement on the reduced model

Parametric block plus `s(AttdAge, k=13)` and `s(PolYear, k=6)`, logit link, max absolute difference on
the link scale over the 30,000 cells, frame 1 / frame 2:

| Comparison | Frame 1 | Frame 2 |
|---|---|---|
| mgcv-rust fREML vs mgcv `gam` REML (proportions) | 7.6e-6 | 2.4e-5 |
| mgcv-rust REML vs mgcv `gam` REML (0/1 rows) | 2.7e-3 | 6.7e-3 |
| mgcv `gam` REML vs mgcv `bam` fREML | 7.5e-4 | 4.2e-2 |
| mgcv-rust Poisson rates with weights vs mgcv Poisson offset | 2.4e-5 | 1.1e-5 |
| mgcv-rust quasi-Poisson fREML vs mgcv quasi-Poisson | 3.2e-4 | 1.2e-3 |

Under its default REML, mgcv-rust did not converge on quasi-Poisson in 200 outer iterations on either frame.

## Reading

A capable additive-model fitter, not a hierarchical one. Usable as an R-free cross-check on additive
Poisson sub-models. Limits: synthetic data, Python interface only, one version.
