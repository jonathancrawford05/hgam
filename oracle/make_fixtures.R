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

# 7. the sz basis (smooth.construct.sz.smooth.spec, smooth.r) with xt = list(bs = "cr"), as the exit test uses it:
# smoothCon's absorbed X and per-level S for a 2- and a 3-level factor at supplied knots, then a REML fit with an
# sz term beside a cr smooth of the same covariate, knots shared by both. No factor main effect: the cr null space
# already holds each level's constant, so adding f makes the model rank deficient (see docs/DEFERRED.md).
set.seed(9); n <- 300
sz_knots <- c(1, 3, 6, 10, 15, 21)
sz_case <- function(levels) {
  ds <- data.frame(x = runif(n, 1, 21), f = factor(sample(levels, n, TRUE), levels = levels))
  sm <- smoothCon(s(f, x, bs = "sz", k = 6, xt = list(bs = "cr")), ds, knots = list(x = sz_knots),
                  absorb.cons = TRUE)[[1]]
  list(x = ds$x, group = as.integer(ds$f) - 1L, n_levels = length(levels), knots = sz_knots, label = sm$label,
       X = sm$X, S = sm$S, rank = sm$rank, bs_dim = sm$bs.dim)
}
put("smoothcon_sz_cr", list(cases = list(two_levels = sz_case(c("a", "b")), three_levels = sz_case(c("a", "b", "c")))))
set.seed(10); n <- 1500
d10 <- data.frame(x = runif(n, 1, 21), f = factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c")),
                  n = sample(20:200, n, TRUE))
fe <- c(a = 0, b = 0.3, c = -0.2)[as.character(d10$f)]
fs <- c(a = 0.25, b = -0.2, c = 0)[as.character(d10$f)] * sin(d10$x / 3)
d10$p <- rbinom(n, d10$n, 1 - exp(-exp(-4 + 0.08 * (d10$x - 10) + fe + fs))) / d10$n
f10 <- "p ~ s(x, bs='cr', k=6) + s(f, x, bs='sz', k=6, xt=list(bs='cr'))"
m10 <- gam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10,
           knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_sz_reml", c(gam_case(m10, d10, f10, "binomial(link='cloglog')"), list(knots = list(x = sz_knots))))

# 8. a factor main effect beside sz smooths of that factor, the exit formula's shape: structurally rank deficient,
# because each level's constant lies in the cr null space of the sz term. 8a is section 7's model with f added; 8b
# has two sz smooths of one two-level factor, as the exit formula has for FaceSize. mgcv drops the unidentified
# directions in gdiPK (gdi.c), in the basis of totalPenaltySpace (gam.fit3.r); per coefficient edf and the
# coefficient names show which. gam_case's edf covers smooths only, so the factor's own edf is exported too.
rank_case <- function(m, d, formula, knots) {
  edf_param <- sapply(split(m$edf[seq_len(m$nsdf)], m$assign), sum)
  c(gam_case(m, d, formula, "binomial(link='cloglog')"),
    list(knots = knots, rank = m$rank, coef_names = names(coef(m)), coef = unname(coef(m)), edf_coef = unname(m$edf),
         param_labels = attr(m$pterms, "term.labels"), edf_param = unname(edf_param[-1])))
}
f8a <- "p ~ f + s(x, bs='cr', k=6) + s(f, x, bs='sz', k=6, xt=list(bs='cr'))"
m8a <- gam(as.formula(f8a), family = binomial(link = "cloglog"), weights = n, data = d10,
           knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_f_sz_reml", rank_case(m8a, d10, f8a, list(x = sz_knots)))
set.seed(11); n <- 2000
d11 <- data.frame(x = runif(n, 1, 21), z = runif(n, 1, 21), f = factor(sample(c("a", "b"), n, TRUE), levels = c("a", "b")),
                  n = sample(20:200, n, TRUE))
fb <- d11$f == "b"
d11$p <- rbinom(n, d11$n, 1 - exp(-exp(-4 + 0.08 * (d11$x - 10) - 0.3 * exp(-d11$z / 4) - 0.2 * fb +
                                       0.15 * fb * sin(d11$x / 3) - 0.1 * fb * exp(-d11$z / 3)))) / d11$n
f8b <- paste("p ~ f + s(x, bs='cr', k=6) + s(z, bs='cr', k=6) + s(f, x, bs='sz', k=6, xt=list(bs='cr')) +",
             "s(f, z, bs='sz', k=6, xt=list(bs='cr'))")
m8b <- gam(as.formula(f8b), family = binomial(link = "cloglog"), weights = n, data = d11,
           knots = list(x = sz_knots, z = sz_knots), method = "REML")
put("gam_binomial_cloglog_two_sz_reml", rank_case(m8b, d11, f8b, list(x = sz_knots, z = sz_knots)))

# 9. select = TRUE with an sz term (smoothCon's null.space.penalty, smooth.r): sz has one penalty per level, so
# smoothCon takes the need.full branch and appends U U' for the null space of the summed penalties. 9a: smoothCon's
# S list with null.space.penalty = TRUE for a 2- and a 3-level factor; 9b: section 7's model fitted with select = TRUE.
set.seed(12); n <- 300
sz_select_case <- function(levels) {
  ds <- data.frame(x = runif(n, 1, 21), f = factor(sample(levels, n, TRUE), levels = levels))
  sm <- smoothCon(s(f, x, bs = "sz", k = 6, xt = list(bs = "cr")), ds, knots = list(x = sz_knots),
                  absorb.cons = TRUE, null.space.penalty = TRUE)[[1]]
  list(x = ds$x, group = as.integer(ds$f) - 1L, n_levels = length(levels), knots = sz_knots, S = sm$S, rank = sm$rank)
}
put("smoothcon_sz_cr_select", list(cases = list(two_levels = sz_select_case(c("a", "b")),
                                                three_levels = sz_select_case(c("a", "b", "c")))))
m10s <- gam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10,
            knots = list(x = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_sz_select_reml", c(gam_case(m10s, d10, f10, "binomial(link='cloglog')"),
                                             list(knots = list(x = sz_knots), select = TRUE)))

# 10. a numeric by on a cr smooth (smoothCon, smooth.r), as the exit formula's s(AttdAge, by = StudyYear_C): the
# penalty is scaled on the un-multiplied X, then X is multiplied row-wise by the by variable; the sum-to-zero
# constraint is dropped because the by variable is not constant (sd(by) > mean(by) * eps * 1000), so the term keeps
# k coefficients. 10a: smoothCon's absorbed X and S at supplied knots; 10b: a REML fit with a bare cr smooth of x
# beside the by smooth, knots shared; 10c: 10b with select = TRUE.
set.seed(13); n <- 300
dz <- data.frame(x = runif(n, 1, 21), z = runif(n, -8.5, 8.5))
smz <- smoothCon(s(x, by = z, bs = "cr", k = 6), dz, knots = list(x = sz_knots), absorb.cons = TRUE)[[1]]
put("smoothcon_cr_by_numeric", list(x = dz$x, by = dz$z, knots = sz_knots, label = smz$label, X = smz$X,
                                    S = smz$S[[1]], C_rows = nrow(smz$C), rank = smz$rank, bs_dim = smz$bs.dim))
set.seed(14); n <- 1500
d14 <- data.frame(x = runif(n, 1, 21), z = runif(n, -8.5, 8.5), n = sample(20:200, n, TRUE))
d14$p <- rbinom(n, d14$n, 1 - exp(-exp(-4 + 0.08 * (d14$x - 10) - 0.012 * d14$z +
                                       0.004 * d14$z * sin(d14$x / 3)))) / d14$n
f14 <- "p ~ s(x, bs='cr', k=6) + s(x, by=z, bs='cr', k=6)"
m14 <- gam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14,
           knots = list(x = sz_knots), method = "REML")
# mgcv's default newton stops short of its own REML optimum on this fixture (the by smooth's sp is on a flat ridge),
# so the fit is repeated at a tight newton conv.tol. The tight fit is the reference for quantities at estimated sp
# only where its REML score is not higher than the default fit's (docs/PRIORITIES.md).
m14t <- gam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14,
            knots = list(x = sz_knots), method = "REML", control = gam.control(newton = list(conv.tol = 1e-12)))
put("gam_binomial_cloglog_by_numeric_reml", c(gam_case(m14, d14, f14, "binomial(link='cloglog')"),
                                              list(knots = list(x = sz_knots),
                                                   tight = c(gam_case(m14t, d14, f14, "binomial(link='cloglog')")[
                                                     c("sp", "edf", "eta", "se", "reml", "outer_iter", "outer_conv")],
                                                     list(conv_tol = 1e-12)))))
m14s <- gam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14,
            knots = list(x = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_by_numeric_select_reml", c(gam_case(m14s, d14, f14, "binomial(link='cloglog')"),
                                                     list(knots = list(x = sz_knots), select = TRUE)))

# 11. select = TRUE on section 8's rank-deficient shapes (8a: f + s(x) + s(f, x, sz); 8b: f with two sz smooths of a
# two-level factor). The null-space penalties (smoothCon's null.space.penalty, smooth.r) penalise the directions
# the factor shares with its sz smooths, so mgcv's fit is full rank (m$rank equals the coefficient count) and the
# per-term edf are identified.
m11a <- gam(as.formula(f8a), family = binomial(link = "cloglog"), weights = n, data = d10,
            knots = list(x = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_f_sz_select_reml", c(rank_case(m11a, d10, f8a, list(x = sz_knots)), list(select = TRUE)))
m11b <- gam(as.formula(f8b), family = binomial(link = "cloglog"), weights = n, data = d11,
            knots = list(x = sz_knots, z = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_two_sz_select_reml", c(rank_case(m11b, d11, f8b, list(x = sz_knots, z = sz_knots)),
                                                 list(select = TRUE)))

cat("fixtures written to", out, "with mgcv", meta$mgcv, "tier", meta$tier, "\n")
