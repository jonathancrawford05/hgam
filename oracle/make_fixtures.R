# Reference fixtures from mgcv. Usage: Rscript oracle/make_fixtures.R <out_dir>
# Each fixture records the mgcv and R versions that produced it. Fixtures made outside the
# pinned oracle image are tier 1 (provisional); CI regenerates them inside the image (tier 3).
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
out <- commandArgs(trailingOnly = TRUE)[1]; if (is.na(out)) out <- "tests/fixtures"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
meta <- list(mgcv = as.character(packageVersion("mgcv")), r = R.version.string,
             tier = if (nzchar(Sys.getenv("HGAM_ORACLE_DIGEST"))) 3 else 1,
             oracle_digest = Sys.getenv("HGAM_ORACLE_DIGEST"))
put <- function(name, x) write_json(c(list(meta = meta), x), file.path(out, paste0(name, ".json")),
                                    auto_unbox = TRUE, digits = NA, matrix = "rowmajor")

# 1. cr basis with supplied knots: design, penalty and constraint, before and after absorbing the constraint
set.seed(1); x <- sort(runif(40, 1, 21)); knots <- c(1, 2, 3, 5, 10, 21)
raw <- smoothCon(s(x, bs = "cr", k = 6), data.frame(x = x), knots = list(x = knots), absorb.cons = FALSE)[[1]]
abs <- smoothCon(s(x, bs = "cr", k = 6), data.frame(x = x), knots = list(x = knots), absorb.cons = TRUE)[[1]]
put("smoothcon_cr_k6_knots", list(x = x, knots = knots, X = raw$X, S = raw$S[[1]], C = raw$C,
                                  X_absorbed = abs$X, S_absorbed = abs$S[[1]]))

# 2. a small Poisson fit with an exposure offset, REML
set.seed(2); n <- 400
d <- data.frame(age = runif(n, 20, 90), dur = runif(n, 1, 21), expo = round(runif(n, 200, 5000)))
d$deaths <- rpois(n, d$expo * exp(-6 + 0.07 * (d$age - 40) - 0.4 * exp(-d$dur / 4)))
m <- gam(deaths ~ s(age, bs = "cr", k = 8) + s(dur, bs = "cr", k = 5) + offset(log(expo)),
         family = poisson, data = d, method = "REML")
put("gam_poisson_offset_reml", list(data = d, formula = "deaths ~ s(age, bs='cr', k=8) + s(dur, bs='cr', k=5) + offset(log(expo))",
    labels = sapply(m$smooth, function(s) s$label),
    family = "poisson", method = "REML", sp = unname(m$sp), edf = unname(summary(m)$edf), edf_total = sum(m$edf),
    n_coef = length(coef(m)), coef = unname(coef(m)), eta = as.numeric(predict(m, type = "link")),
    reml = as.numeric(m$gcv.ubre), se = as.numeric(predict(m, se.fit = TRUE)$se.fit)))

# 3. the exit test's structure: the target formula on a synthetic frame (structure only, small n)
set.seed(3); n <- 4000
f <- data.frame(AttdAge = runif(n, 1, 95), PolYear = runif(n, 1, 21), StudyYear_C = runif(n, -8.5, 8.5),
                FaceSize = factor(sample(c("Low", "High"), n, TRUE), levels = c("Low", "High")),
                Smoke = factor(sample(c("NS", "S"), n, TRUE, prob = c(.8, .2)), levels = c("NS", "S")),
                ExposCnt = round(runif(n, 200, 5000)))
eta <- -6.2 + 0.075 * (f$AttdAge - 40) + 1.2 * exp(-f$AttdAge / 3) - 0.5 * exp(-f$PolYear / 4) +
  0.1 * sin(f$AttdAge / 12) * cos(f$PolYear / 4) - 0.012 * f$StudyYear_C +
  (f$FaceSize == "High") * -0.2 + (f$Smoke == "S") * (0.6 - 0.007 * (f$AttdAge - 50))
f$DthCnt <- rbinom(n, f$ExposCnt, 1 - exp(-exp(eta)))
my_knots <- list(PolYear = c(1, 2, 3, 5, 10, 21), AttdAge = c(1, 2, 4, 7, 14, 18, 24, 35, 50, 70, 85, 90, 95))
hgam_formula <- (DthCnt / ExposCnt) ~ FaceSize + Smoke + FaceSize:Smoke +
  s(AttdAge, k = 13, bs = "cr") + s(PolYear, k = 6, bs = "cr") + ti(AttdAge, PolYear, k = c(13, 6), bs = "cr") +
  s(FaceSize, AttdAge, bs = "sz", k = 13, xt = list(bs = "cr")) + s(Smoke, AttdAge, bs = "sz", k = 13, xt = list(bs = "cr")) +
  s(FaceSize, PolYear, bs = "sz", k = 6, xt = list(bs = "cr")) + s(Smoke, PolYear, bs = "sz", k = 6, xt = list(bs = "cr")) +
  s(AttdAge, by = StudyYear_C, k = 13, bs = "cr")
b <- bam(hgam_formula, family = binomial(link = "cloglog"), weights = ExposCnt, data = f, knots = my_knots,
         discrete = TRUE, select = TRUE)
put("exit_hgam_formula_structure", list(n = n, n_coef = length(coef(b)), n_smooth = length(b$smooth), n_sp = length(b$sp),
    labels = sapply(b$smooth, function(s) s$label), term_coef = sapply(b$smooth, function(s) s$last.para - s$first.para + 1),
    edf_total = sum(b$edf)))
# 4. The six functions polaris-re ported from mgcv source and M0 moved here (PR #3).
# 4a. tero: bam(discrete = TRUE) moves the margin with the largest bs.dim last (the last one on ties).
tero_cases <- list(c(13, 6), c(6, 13), c(5, 5), c(7, 4, 5), c(7, 4, 7), c(4, 9, 3))
vars <- c("a", "b", "c")
put("tero_margin_order", list(cases = lapply(tero_cases, function(k) {
  v <- vars[seq_along(k)]
  sm <- eval(parse(text = sprintf("ti(%s, k = c(%s), bs = 'cr')", paste(v, collapse = ", "), paste(k, collapse = ", "))))
  out <- mgcv:::tero(sm)
  list(variables = v, k = k, term = out$term, k_out = sapply(out$margin, function(m) m$bs.dim), label = out$label)
})))

# 4b. compress.df: the unique covariate rows bam(discrete = TRUE) works on, and each data row's cell.
# Rows are shuffled by sample(), so only the partition and each row's discretised value are comparable.
cdf <- function(dat, m = NULL) {
  set.seed(8547); xu <- mgcv:::compress.df(dat, m = m); k <- attr(xu, "index")
  num <- function(col) if (is.factor(col)) as.integer(col) - 1L else col
  list(columns = lapply(dat, num), factors = names(dat)[sapply(dat, is.factor)], m = m,
       nr = nrow(xu), index = as.integer(k), row_values = lapply(xu, function(col) num(col)[k]))
}
set.seed(4)
put("compress_df_cases", list(cases = list(
  integer_identity = cdf(data.frame(age = sample(20:90, 3000, TRUE))),
  one_d_at_threshold = cdf(data.frame(x = sample(runif(1000)))),
  one_d_binned = cdf(data.frame(x = runif(1500, 3, 17))),
  two_d_binned = cdf(data.frame(x = runif(12000), z = rnorm(12000))),
  numeric_and_factor = cdf(data.frame(x = runif(2500), f = factor(sample(c("p", "q", "r"), 2500, TRUE)))),
  two_d_given_m = cdf(data.frame(x = runif(3000), z = runif(3000)), m = 400)
)))

# 4c. initial.sp with XX = TRUE: the starting smoothing parameters bgam.fitd takes from diag(X'WX).
set.seed(5); n <- 500
d4 <- data.frame(age = runif(n, 20, 90), dur = runif(n, 1, 21))
G <- gam(age ~ s(age, bs = "cr", k = 8) + s(dur, bs = "cr", k = 5) + ti(age, dur, bs = "cr", k = c(5, 4)),
         data = d4, fit = FALSE)
w <- runif(n, 0.5, 3); XX <- crossprod(G$X * sqrt(w))
pad <- function(S, off, p) { M <- matrix(0, p, p); i <- off:(off + ncol(S) - 1); M[i, i] <- S; M }
blocks <- lapply(seq_along(G$S), function(i) pad(G$S[[i]], G$off[i], ncol(G$X)))
put("initial_sp_xx", list(xx_diag = diag(XX), penalty_blocks = blocks,
    sp = mgcv:::initial.sp(XX, G$S, G$off, XX = TRUE)))

# 4d. Sl.fitChol at fixed log smoothing parameters, on bgam.fitd's own initial re-parameterisation;
# beta and PP are mapped back to the original coefficients as bgam.fitd does.
Sl <- mgcv:::Sl.setup(G)
z <- rnorm(n); f0 <- as.numeric(crossprod(G$X, w * z))
XXr <- mgcv:::Sl.initial.repara(Sl, XX, inverse = FALSE, both.sides = TRUE, cov = FALSE)
fr <- mgcv:::Sl.initial.repara(Sl, f0, inverse = FALSE, both.sides = TRUE, cov = FALSE)
rho <- log(c(2, 0.5, 8, 1.5))
fc <- mgcv:::Sl.fitChol(Sl, XXr, fr, rho = rho, tol = 0)
put("sl_fitchol", list(xx = XX, f = f0, penalty_blocks = blocks, rho = rho,
    beta = mgcv:::Sl.initial.repara(Sl, fc$beta, inverse = TRUE, both.sides = FALSE, cov = FALSE),
    grad = as.numeric(fc$grad), step = as.numeric(fc$step),
    PP = mgcv:::Sl.initial.repara(Sl, fc$PP, inverse = TRUE, both.sides = TRUE, cov = TRUE)))

# 4e. bgam.fitd: bam(method = "fREML", discrete = TRUE) on integer covariates, so the discretisation is
# the identity and the lpmatrix at the data is bam's own design. Exported with bam's own penalties.
bam_case <- function(b, y, offset, weights) {
  X <- predict(b, type = "lpmatrix"); p <- ncol(X)
  S <- unlist(lapply(b$smooth, function(sm) lapply(sm$S, pad, off = sm$first.para, p = p)), recursive = FALSE)
  list(X = X, penalty_blocks = S, y = y, offset = offset, weights = weights, sp = unname(b$sp),
       eta = as.numeric(b$linear.predictors), coef = unname(coef(b)),
       edf = sapply(b$smooth, function(sm) sum(b$edf[sm$first.para:sm$last.para])),
       term_first = sapply(b$smooth, function(sm) sm$first.para), term_last = sapply(b$smooth, function(sm) sm$last.para),
       se = as.numeric(predict(b, se.fit = TRUE)$se.fit), iter = b$iter)
}
set.seed(6); n <- 3000
d5 <- data.frame(age = sample(40:79, n, TRUE), dur = sample(1:15, n, TRUE), expo = runif(n, 5, 40))
d5$y <- rpois(n, d5$expo * exp(-5.5 + 0.05 * (d5$age - 40) - 0.3 * exp(-d5$dur / 3)))
b5 <- bam(y ~ s(age, bs = "cr", k = 8) + s(dur, bs = "cr", k = 5) + offset(log(expo)), family = poisson,
          data = d5, method = "fREML", discrete = TRUE)
put("bgam_fitd_poisson", bam_case(b5, d5$y, log(d5$expo), rep(1, n)))
d5$n <- sample(20:200, n, TRUE)
d5$p <- rbinom(n, d5$n, 1 - exp(-exp(-4 + 0.06 * (d5$age - 40) + 0.2 * sin(d5$dur / 2)))) / d5$n
b6 <- bam(p ~ s(age, bs = "cr", k = 8) + s(dur, bs = "cr", k = 5), family = binomial(link = "cloglog"),
          weights = n, data = d5, method = "fREML", discrete = TRUE)
put("bgam_fitd_binomial_cloglog", bam_case(b6, d5$p, rep(0, n), d5$n))

# 5. gam(method = "REML"): the outer Newton search (newton, gam.fit3.r) on a cr + ti binomial-cloglog model
# with prior weights. Per-term sp and edf, the fitted linear predictor and its standard errors.
gam_case <- function(m, d, formula, family) {
  list(data = d, formula = formula, family = family, method = "REML", sp = unname(m$sp),
       sp_labels = names(m$sp), labels = sapply(m$smooth, function(s) s$label),
       edf = sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para])), edf_total = sum(m$edf),
       n_coef = length(coef(m)), eta = as.numeric(predict(m, type = "link")),
       se = as.numeric(predict(m, se.fit = TRUE)$se.fit), reml = as.numeric(m$gcv.ubre),
       outer_iter = m$outer.info$iter, outer_conv = m$outer.info$conv)
}
set.seed(7); n <- 1500
d7 <- data.frame(age = runif(n, 20, 90), dur = runif(n, 1, 21), n = sample(20:200, n, TRUE))
d7$p <- rbinom(n, d7$n, 1 - exp(-exp(-4.5 + 0.05 * (d7$age - 40) - 0.4 * exp(-d7$dur / 4) +
                                     0.15 * sin(d7$age / 10) * cos(d7$dur / 5)))) / d7$n
f7 <- "p ~ s(age, bs='cr', k=8) + s(dur, bs='cr', k=5) + ti(age, dur, bs='cr', k=c(5, 4))"
m7 <- gam(as.formula(f7), family = binomial(link = "cloglog"), weights = n, data = d7, method = "REML")
put("gam_binomial_cloglog_ti_reml", gam_case(m7, d7, f7, "binomial(link='cloglog')"))

# 6. user-supplied knots (smooth.construct.cr.smooth.spec, smooth.r), as the exit test passes them: one knot
# vector per covariate, shared by its s() term and its ti() margin, so the ti margin k equals the s() k.
set.seed(8); n <- 1500
d8 <- data.frame(age = runif(n, 20, 90), dur = runif(n, 1, 21), n = sample(20:200, n, TRUE))
d8$p <- rbinom(n, d8$n, 1 - exp(-exp(-4.5 + 0.05 * (d8$age - 40) - 0.4 * exp(-d8$dur / 4) +
                                     0.15 * sin(d8$age / 10) * cos(d8$dur / 5)))) / d8$n
k8 <- list(age = c(20, 25, 30, 40, 50, 65, 80, 90), dur = c(1, 2, 4, 9, 21))
f8 <- "p ~ s(age, bs='cr', k=8) + s(dur, bs='cr', k=5) + ti(age, dur, bs='cr', k=c(8, 5))"
m8 <- gam(as.formula(f8), family = binomial(link = "cloglog"), weights = n, data = d8, knots = k8, method = "REML")
put("gam_binomial_cloglog_knots_reml", c(gam_case(m8, d8, f8, "binomial(link='cloglog')"), list(knots = k8)))

cat("fixtures written to", out, "with mgcv", meta$mgcv, "tier", meta$tier, "\n")
