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

# 12. bam(method = "fREML", discrete = TRUE) (bgam.fitd and discrete.mf, bam.r), the exit test's fitter, on section 5's
# cr + ti model (12a) and section 7's sz model (12b, knots shared). discrete.mf bins each smooth's (or ti margin's)
# covariates with compress.df, so mgcv's discrete fit sits ~1e-3 from its own exact fit; bam at discrete = FALSE is
# recorded beside it ("exact") so a difference can be placed on the binning or on the iteration.
bam_disc_case <- function(b, d, formula, family, extra = list()) {
  c(list(data = d, formula = formula, family = family, method = "fREML", discrete = TRUE, sp = unname(b$sp),
         sp_labels = names(b$sp), labels = sapply(b$smooth, function(s) s$label),
         edf = sapply(b$smooth, function(sm) sum(b$edf[sm$first.para:sm$last.para])), edf_total = sum(b$edf),
         n_coef = length(coef(b)), eta = as.numeric(b$linear.predictors),
         se = as.numeric(predict(b, se.fit = TRUE)$se.fit), iter = b$iter), extra)
}
bam_exact <- function(b) list(sp = unname(b$sp), edf = sapply(b$smooth, function(sm) sum(b$edf[sm$first.para:sm$last.para])),
                              eta = as.numeric(b$linear.predictors), se = as.numeric(predict(b, se.fit = TRUE)$se.fit))
b12a <- bam(as.formula(f7), family = binomial(link = "cloglog"), weights = n, data = d7, method = "fREML", discrete = TRUE)
b12ae <- bam(as.formula(f7), family = binomial(link = "cloglog"), weights = n, data = d7, method = "fREML")
put("bam_discrete_binomial_cloglog_ti", bam_disc_case(b12a, d7, f7, "binomial(link='cloglog')",
                                                       list(exact = bam_exact(b12ae))))
b12b <- bam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE)
b12be <- bam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
             method = "fREML")
put("bam_discrete_binomial_cloglog_sz", bam_disc_case(b12b, d10, f10, "binomial(link='cloglog')",
                                                       list(knots = list(x = sz_knots), exact = bam_exact(b12be))))

# 12c, 12d: 12a and 12b with select = TRUE, as the exit test fits.
b12c <- bam(as.formula(f7), family = binomial(link = "cloglog"), weights = n, data = d7, method = "fREML", discrete = TRUE,
            select = TRUE)
b12ce <- bam(as.formula(f7), family = binomial(link = "cloglog"), weights = n, data = d7, method = "fREML", select = TRUE)
put("bam_discrete_binomial_cloglog_ti_select", bam_disc_case(b12c, d7, f7, "binomial(link='cloglog')",
                                                              list(select = TRUE, exact = bam_exact(b12ce))))
b12d <- bam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE, select = TRUE)
b12de <- bam(as.formula(f10), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
             method = "fREML", select = TRUE)
put("bam_discrete_binomial_cloglog_sz_select", bam_disc_case(b12d, d10, f10, "binomial(link='cloglog')",
                                                              list(knots = list(x = sz_knots), select = TRUE,
                                                                   exact = bam_exact(b12de))))
# 12e: section 10's numeric-by model (f14 on d14, knots shared): discrete.mf compresses the by variable on its own,
# before the margin.
b12e <- bam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE)
b12ee <- bam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14, knots = list(x = sz_knots),
             method = "fREML")
put("bam_discrete_binomial_cloglog_by_numeric", bam_disc_case(b12e, d14, f14, "binomial(link='cloglog')",
                                                               list(knots = list(x = sz_knots), exact = bam_exact(b12ee))))
# 13. bam(method = "fREML", discrete = TRUE, select = TRUE) on the exit formula's remaining shapes: 13a section 11a's
# factor main effect beside its sz smooth (f8a on d10), 13b section 10's numeric by (f14 on d14). mgcv's rank and the
# parametric edf are exported as in section 8, so a rank-deficient fit would show.
bam_rank <- function(b) {
  list(rank = b$rank, coef_names = names(coef(b)), param_labels = attr(b$pterms, "term.labels"),
       edf_param = unname(sapply(split(b$edf[seq_len(b$nsdf)], b$assign), sum)[-1]))
}
b13a <- bam(as.formula(f8a), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE, select = TRUE)
b13ae <- bam(as.formula(f8a), family = binomial(link = "cloglog"), weights = n, data = d10, knots = list(x = sz_knots),
             method = "fREML", select = TRUE)
put("bam_discrete_binomial_cloglog_f_sz_select", bam_disc_case(b13a, d10, f8a, "binomial(link='cloglog')",
                                                                c(list(knots = list(x = sz_knots), select = TRUE,
                                                                       exact = bam_exact(b13ae)), bam_rank(b13a))))
b13b <- bam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE, select = TRUE)
b13be <- bam(as.formula(f14), family = binomial(link = "cloglog"), weights = n, data = d14, knots = list(x = sz_knots),
             method = "fREML", select = TRUE)
put("bam_discrete_binomial_cloglog_by_numeric_select", bam_disc_case(b13b, d14, f14, "binomial(link='cloglog')",
                                                                      list(knots = list(x = sz_knots), select = TRUE,
                                                                           exact = bam_exact(b13be))))
# 14. the exit formula itself: section 3's b (bam(discrete = TRUE, select = TRUE), my_knots), every key result. Two factors
# with sz smooths of two covariates each, their interaction, a ti and a numeric by together. The response is written
# as a column p = DthCnt / ExposCnt (prior weights ExposCnt), so the exported formula needs no arithmetic on the left.
# mgcv's bam at discrete = FALSE is recorded beside it, and its rank and parametric edf as in section 13.
f$p <- f$DthCnt / f$ExposCnt
f14x <- paste('p ~ FaceSize + Smoke + FaceSize:Smoke + s(AttdAge, k=13, bs="cr") + s(PolYear, k=6, bs="cr") +',
              'ti(AttdAge, PolYear, k=c(13, 6), bs="cr") + s(FaceSize, AttdAge, bs="sz", k=13, xt=list(bs="cr")) +',
              's(Smoke, AttdAge, bs="sz", k=13, xt=list(bs="cr")) + s(FaceSize, PolYear, bs="sz", k=6, xt=list(bs="cr")) +',
              's(Smoke, PolYear, bs="sz", k=6, xt=list(bs="cr")) + s(AttdAge, by=StudyYear_C, k=13, bs="cr")')
b14 <- bam(as.formula(f14x), family = binomial(link = "cloglog"), weights = ExposCnt, data = f, knots = my_knots,
           method = "fREML", discrete = TRUE, select = TRUE)
stopifnot(isTRUE(all.equal(unname(b14$linear.predictors), unname(b$linear.predictors))))
b14e <- bam(as.formula(f14x), family = binomial(link = "cloglog"), weights = ExposCnt, data = f, knots = my_knots,
            method = "fREML", select = TRUE)
put("bam_discrete_exit_formula", bam_disc_case(b14, f, f14x, "binomial(link='cloglog')",
                                               c(list(knots = my_knots, weights = "ExposCnt", select = TRUE,
                                                      exact = bam_exact(b14e)), bam_rank(b14))))
# 15. predict.bam on new data from section 14's discrete fit b14 (predict.bamd, bam.r): discrete.mf re-runs on newdata,
# so each smooth covariate of newdata is compressed on its own (compress.df, m = 1000) over newdata's own range.
# 15a: 1500 rows with more than 1000 unique values per numeric covariate, so they are binned; 15b: 40 rows, not binned.
# mgcv's prediction at the raw covariates (discrete = FALSE, predict.gam) is recorded beside each, to show the
# binning moves eta by more than its gate on 15a.
set.seed(15)
nd_frame <- function(n) data.frame(AttdAge = runif(n, 5, 90), PolYear = runif(n, 1.5, 20), StudyYear_C = runif(n, -8, 8),
                                   FaceSize = factor(sample(c("Low", "High"), n, TRUE), levels = c("Low", "High")),
                                   Smoke = factor(sample(c("NS", "S"), n, TRUE), levels = c("NS", "S")))
for (case in list(list(name = "a", n = 1500), list(name = "b", n = 40))) {
  nd <- nd_frame(case$n)
  pr <- predict(b14, newdata = nd, se.fit = TRUE)
  pe <- predict(b14, newdata = nd, se.fit = TRUE, discrete = FALSE)
  put(paste0("bam_discrete_exit_predict_", case$name),
      list(newdata = nd, eta = as.numeric(pr$fit), se = as.numeric(pr$se.fit),
           eta_exact = as.numeric(pe$fit), se_exact = as.numeric(pe$se.fit)))
}
# 16. summary.gam of section 14's discrete fit b14 (summary.gam, mgcv.r; bam fits use it): the parametric table
# (estimate, se, z, p; the scale is fixed) and the smooth table (edf, Ref.df, Chi.sq, p), with the scalars printed
# beside them. Ref.df, Chi.sq and p come from testStat / reTest (mgcv.r). The factor levels are exported: the frame's
# levels (Low, High) are not the order R would sort the strings in, and the coefficient names depend on them.
s14 <- summary(b14)
put("bam_discrete_exit_summary", list(
  p_names = rownames(s14$p.table), p_table = unname(s14$p.table), p_columns = colnames(s14$p.table),
  s_names = rownames(s14$s.table), s_table = unname(s14$s.table), s_columns = colnames(s14$s.table),
  residual_df = s14$residual.df, n = s14$n, dev_expl = s14$dev.expl, r_sq = s14$r.sq, scale = s14$scale,
  null_space_dim = sapply(b14$smooth, function(sm) sm$null.space.dim), levels = lapply(f[sapply(f, is.factor)], levels)))
# 17. the rows bam(discrete = TRUE) takes each sum-to-zero constraint over (discrete.mf, bam.r; smoothCon, smooth.r).
# discrete.mf, under temp.seed(8547), compresses and shuffles each covariate (compress.df) and pads every column to the
# longest with sample(); gam.setup then builds the smooths on that padded frame, so colMeans(X) is over its rows.
# 17a: R's own stream under set.seed(8547) (RNG.c, random.c), the draws discrete.mf makes. 17b: section 14's padded
# frame (captured from the gam.setup call by trace) and the unit constraint direction (qr.Q of the stored qrc) of
# s(AttdAge), s(PolYear) and both ti margins.
set.seed(8547); r_unif <- runif(5); r_perm <- sample.int(981, 981); r_rep <- sample.int(985, 50, replace = TRUE)
set.seed(8547); r_mixed <- c(sample.int(2, 2), sample.int(70000, 5, replace = TRUE), sample.int(1, 3, replace = TRUE))
put("r_rng_set_seed_8547", list(seed = 8547, unif = r_unif, perm_981 = r_perm, rep_985_50 = r_rep,
                                mixed = list(perm_2 = r_mixed[1:2], rep_70000_5 = r_mixed[3:7], rep_1_3 = r_mixed[8:10])))
mf17 <- NULL
trace(mgcv:::gam.setup, quote(assign("mf17", data, envir = globalenv())), print = FALSE, where = asNamespace("mgcv"))
b17 <- bam(as.formula(f14x), family = binomial(link = "cloglog"), weights = ExposCnt, data = f, knots = my_knots,
           method = "fREML", discrete = TRUE, select = TRUE)
untrace(mgcv:::gam.setup, where = asNamespace("mgcv"))
stopifnot(isTRUE(all.equal(unname(b17$linear.predictors), unname(b14$linear.predictors))))
con_dir <- function(sm) as.numeric(qr.Q(attr(sm, "qrc"))[, 1])
ti17 <- b17$smooth[[which(sapply(b17$smooth, function(sm) sm$label) == "ti(PolYear,AttdAge)")]]
put("bam_discrete_constraint_rows", list(
  columns = names(mf17), maxr = nrow(mf17),
  AttdAge = mf17$AttdAge, PolYear = mf17$PolYear, StudyYear_C = mf17$StudyYear_C, p = mf17$p,
  weights = mf17[["(weights)"]], FaceSize = as.integer(mf17$FaceSize) - 1L, Smoke = as.integer(mf17$Smoke) - 1L,
  n_unique = sapply(mf17, function(v) length(unique(v))),
  constraint = list(s_AttdAge = con_dir(b17$smooth[[1]]), s_PolYear = con_dir(b17$smooth[[2]]),
                    ti_PolYear = con_dir(ti17$margin[[1]]), ti_AttdAge = con_dir(ti17$margin[[2]])),
  margin_terms = sapply(ti17$margin, function(m) m$term)))
# 18. mroot (mgcv.r, method = "chol": chol(A, pivot = TRUE, tol = 0), LAPACK dpstrf), the square root reTest takes of
# Ve[m, m] and recov of the total penalty (summary.gam's smooth-term tests, section 16's s_table). Two cases: a full-rank
# positive definite 5 x 5 and a 4 x 4 with exact zero pivots (rank 2).
set.seed(18); M18 <- matrix(rnorm(25), 5, 5); A18a <- crossprod(M18) + diag(5) * 0.1
A18b <- matrix(c(4, 0, 2, 0,  0, 0, 0, 0,  2, 0, 5, 0,  0, 0, 0, 0), 4, 4)
put("mroot_chol", list(cases = lapply(list(full = A18a, deficient = A18b), function(A) {
  B <- mroot(A); list(A = A, B = B, rank = ncol(B))
})))
# 19. psum.chisq (mgcv.r) and Davies' algorithm (davies.c), the distribution summary.gam's smooth-term p-values come from
# (reTest: psum.chisq(stat, ev), or psum.chisq(0, c(ev, -stat/k), df = c(rep(1, length(ev)), k)) for an estimated scale).
# Each case records p, and davies' trace and ifault (trace = TRUE), so the port is held to mgcv's path as well as its
# value. The cases reach: both range exits (far_tail, zero_q), the convergence factor (multi_q, far_tail, skew), the
# auxiliary integration (dominant), ties in |lb| (ties), non-central terms with a normal term, df rounding, and the
# liu2 fallback (nlim = 5: ifault 1). liu2 is also recorded on its own.
set.seed(19)
psum19 <- list(
  central = list(q = 12, lb = c(3.1, 1.7, 0.9, 0.4, 0.05)),
  multi_q_a = list(q = 0.5, lb = c(2, 1, 0.5)), multi_q_b = list(q = 5, lb = c(2, 1, 0.5)),
  multi_q_c = list(q = 20, lb = c(2, 1, 0.5)), multi_q_d = list(q = 60, lb = c(2, 1, 0.5)),
  far_tail = list(q = 500, lb = c(2, 1, 0.5)), zero_q = list(q = 0, lb = c(2, 1, 0.5)),
  scale_est = list(q = 0, lb = c(2.5, 1.2, 0.3, -9.4 / 40), df = c(1, 1, 1, 40)),
  noncentral = list(q = 15, lb = c(1.5, -0.7, 2), df = c(2, 3, 1), nc = c(0.5, 1, 2), sigz = 0.8),
  ties = list(q = 1.5, lb = c(1, -1, 1, -1, 0.5)),
  dominant = list(q = 3, lb = c(10, rep(0.01, 30))),
  many_small = list(q = 1.2, lb = c(1, runif(40, 0, 0.02))),
  skew = list(q = 25, lb = c(1, 0.999, 1e-3), df = c(1, 1, 50)),
  df_round = list(q = 9, lb = c(1.2, 0.6), df = c(2.5, 3.4)),
  fallback = list(q = 12, lb = c(3.1, 1.7, 0.9), nlim = 5),
  # davies.c branches reached only here (PR #25 review): the auxiliary loop's `ntm > lim` exit, which
  # returns c = -1 with ifault = 0 so psum.chisq reports p = 2 without a warning; cfe's `fail` return;
  # cfe's `axl = axl2` clamp
  aux_lim = list(q = 3, lb = c(10, rep(0.01, 30)), nlim = 5),
  cfe_fail = list(q = 3, lb = c(10, rep(0.5, 200))),
  cfe_clamp = list(q = 12, lb = c(1, 0.05)),
  # a negative nc: davies.c's ifault = 3 exit; psum.chisq's fallback needs nc = 0, so p = NA
  negative_nc = list(q = 4, lb = c(2, 1), nc = c(-1, 0)))
put("psum_chisq_davies", list(
  cases = lapply(psum19, function(a) {
    a$df <- if (is.null(a$df)) rep(1, length(a$lb)) else a$df
    a$nc <- if (is.null(a$nc)) rep(0, length(a$lb)) else a$nc
    a$sigz <- if (is.null(a$sigz)) 0 else a$sigz
    a$nlim <- if (is.null(a$nlim)) 100000 else a$nlim
    p <- suppressWarnings(psum.chisq(a$q, a$lb, df = a$df, nc = a$nc, sigz = a$sigz, nlim = a$nlim, trace = TRUE))
    c(lapply(a, I), list(p = as.numeric(p), trace = attr(p, "trace"), ifault = attr(p, "ifault")))
  }),
  liu2 = list(x = c(-1, 0.5, 5, 30), lambda = c(2, 1, 0.5), h = c(1, 2, 1),
              p = mgcv:::liu2(c(-1, 0.5, 5, 30), c(2, 1, 0.5), h = c(1, 2, 1)))))
# 20. vcov.gam (mgcv.r) of section 14's discrete fit b14: Vp (the default), Ve (freq = TRUE), Vc (unconditional = TRUE;
# bgam.fitd's Sl.postproc sets it for a discrete fit), and a dispersion rescaling (dispersion * vc / sig2), under
# names(b14$edf) as dimnames.
put("bam_discrete_exit_vcov", list(
  names = rownames(vcov(b14)), sig2 = b14$sig2, vc_is_null = is.null(b14$Vc),
  Vp = unname(vcov(b14)), Ve = unname(vcov(b14, freq = TRUE)), Vc = unname(vcov(b14, unconditional = TRUE)),
  Vp_dispersion_2 = unname(vcov(b14, dispersion = 2)), Ve_dispersion_2 = unname(vcov(b14, freq = TRUE, dispersion = 2))))
# 20b. vcov.gam on a free-scale REML gam fit (gaussian): sig2 is estimated, so dispersion * Vp / sig2 is
# dispersion * (X'WX + S)^-1 whatever the scale estimator; the names cover a factor's treatment columns and a cr smooth.
set.seed(20); n20 <- 300
d20 <- data.frame(x = runif(n20), g = factor(sample(c("a", "b", "c"), n20, TRUE), levels = c("a", "b", "c")))
d20$y <- sin(2 * pi * d20$x) + c(a = 0, b = 0.4, c = -0.3)[as.character(d20$g)] + rnorm(n20, sd = 0.3)
m20 <- gam(y ~ g + s(x, bs = "cr", k = 10), data = d20, method = "REML")
put("gam_gaussian_vcov_reml", list(data = d20, formula = "y ~ g + s(x, bs='cr', k=10)", family = "gaussian",
  method = "REML", levels = list(g = levels(d20$g)), names = rownames(vcov(m20)), sig2 = m20$sig2,
  Vp = unname(vcov(m20)), Vp_dispersion_2 = unname(vcov(m20, dispersion = 2)),
  se = as.numeric(predict(m20, se.fit = TRUE)$se.fit)))
# 21. k.check (plots.r), gam.check's basis-dimension test, on section 14's discrete fit b14 under set.seed(21): per smooth
# k', edf, k-index and p-value. n = 4000 <= subsample, so no subsample is drawn; each 1-D or tensor term draws n.rep
# shuffles sample(1:nr, nr) in smooth order, terms with a factor (sz) get NA and draw nothing, and a by variable is
# dropped. residuals(b14) (deviance residuals, residuals.gam) are recorded beside it.
set.seed(21); kc21 <- k.check(b14)
put("bam_discrete_exit_k_check", list(seed = 21, n_rep = 400, subsample = 5000, names = rownames(kc21),
  columns = colnames(kc21), k_table = unname(kc21), residuals = as.numeric(residuals(b14)),
  # the n > subsample branch (sample(1:n, subsample) before the loop), reached at subsample = 1000
  subsampled = list(subsample = 1000, n_rep = 50, k_table = unname({set.seed(21); k.check(b14, subsample = 1000, n.rep = 50)}))))
# 22. gam.check (plots.r) on section 14's discrete fit b14, without the plots (drawn to a null device): what it prints
# for a bam fit, whose optimizer is c("perf", "chol") so outer.info (Sl.fitChol's grad and hess at the last
# smoothing-parameter update) is printed as a list; then "Model rank = b$rank / length(coef)" and k.check at
# k.sample = 5000, k.rep = 200 under set.seed(22). qq.gam (rep = 0) draws s.rep = 10 permutations sample(U, n) before
# k.check runs, so gam.check's k table is not k.check's after the same seed: the table gam.check computes is taken
# from inside it with trace().
pdf(NULL)
trace(k.check, exit = quote(assign("kc22", returnValue(), envir = globalenv())), print = FALSE,
      where = asNamespace("mgcv"))
set.seed(22); gc22 <- capture.output(gam.check(b14))
untrace(k.check, where = asNamespace("mgcv")); invisible(dev.off())
put("bam_discrete_exit_gam_check", list(seed = 22, k_sample = 5000, k_rep = 200, method = b14$method,
  optimizer = b14$optimizer, grad = as.numeric(b14$outer.info$grad), hess = unname(b14$outer.info$hess),
  rank = b14$rank, n_coef = length(b14$coefficients), names = rownames(kc22), k_table = unname(kc22), output = gc22))
# 23. a factor by on a cr smooth (smoothCon, smooth.r): one copy of the smooth per level of f, each X multiplied by
# that level's dummy and labelled s(x):f<level>. The sum-to-zero constraint is kept and taken on the whole,
# un-multiplied X (before the by branch), so every copy has k - 1 coefficients. 23a: smoothCon's absorbed X and S
# per level at supplied knots; 23b: a REML gam fit of f + s(x, by = f); 23c: the same with bam(discrete = TRUE);
# 23d: 23c with select = TRUE, with predict.bam on new data.
set.seed(23); n <- 1500
d23 <- data.frame(x = runif(n, 1, 21), f = factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c")),
                  n = sample(20:200, n, TRUE))
d23$p <- rbinom(n, d23$n, 1 - exp(-exp(-4 + 0.08 * (d23$x - 10) + c(0, 0.3, -0.2)[as.integer(d23$f)] +
                                       c(0.2, -0.1, 0.4)[as.integer(d23$f)] * sin(d23$x / 3)))) / d23$n
sm23 <- smoothCon(s(x, by = f, bs = "cr", k = 6), d23, knots = list(x = sz_knots), absorb.cons = TRUE)
put("smoothcon_cr_by_factor", list(x = d23$x, group = as.integer(d23$f) - 1L, knots = sz_knots,
  labels = sapply(sm23, function(s) s$label), by_levels = sapply(sm23, function(s) s$by.level),
  X = lapply(sm23, function(s) s$X), S = lapply(sm23, function(s) s$S[[1]]), rank = sapply(sm23, function(s) s$rank)))
f23 <- "p ~ f + s(x, by=f, bs='cr', k=6)"
m23 <- gam(as.formula(f23), family = binomial(link = "cloglog"), weights = n, data = d23,
           knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_by_factor_reml", c(gam_case(m23, d23, f23, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), rank = m23$rank, coef_names = names(coef(m23)))))
b23 <- bam(as.formula(f23), family = binomial(link = "cloglog"), weights = n, data = d23, knots = list(x = sz_knots),
           method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_by_factor", bam_disc_case(b23, d23, f23, "binomial(link='cloglog')",
  c(list(knots = list(x = sz_knots)), bam_rank(b23))))
b23s <- bam(as.formula(f23), family = binomial(link = "cloglog"), weights = n, data = d23, knots = list(x = sz_knots),
            method = "fREML", discrete = TRUE, select = TRUE)
set.seed(230); nd23 <- data.frame(x = runif(200, 1, 21),
                                  f = factor(sample(c("a", "b", "c"), 200, TRUE), levels = c("a", "b", "c")))
pr23 <- predict(b23s, nd23, se.fit = TRUE)
put("bam_discrete_binomial_cloglog_by_factor_select", bam_disc_case(b23s, d23, f23, "binomial(link='cloglog')",
  c(list(knots = list(x = sz_knots), select = TRUE, newdata = nd23, fit_new = as.numeric(pr23$fit),
         se_new = as.numeric(pr23$se.fit)), bam_rank(b23s))))
# 24. the fs factor-smooth interaction on a cr base (smooth.construct.fs.smooth.spec and nat.param, smooth.r): the base
# smooth re-parameterised by nat.param(type = 1), one uncentred copy per level, a range penalty and one penalty per
# null-space direction, each scaled by smoothCon. 24a: smoothCon's X, S list, rank and P at supplied knots, and
# PredictMat at new covariate values (Predict.matrix.fs.interaction), including values outside the knot range.
set.seed(24); n <- 400
d24 <- data.frame(x = runif(n, 1, 21), f = factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c")))
sm24 <- smoothCon(s(x, f, bs = "fs", xt = "cr", k = 6), d24, knots = list(x = sz_knots))[[1]]
nd24 <- data.frame(x = c(0.5, runif(30, 1, 21), 22), f = factor(sample(c("a", "b", "c"), 32, TRUE), levels = c("a", "b", "c")))
put("smoothcon_fs_cr", list(x = d24$x, group = as.integer(d24$f) - 1L, n_levels = 3L, knots = sz_knots, label = sm24$label,
  X = sm24$X, S = sm24$S, rank = sm24$rank, P = sm24$P, C_rows = nrow(sm24$C), null_space_dim = sm24$null.space.dim,
  new_x = nd24$x, new_group = as.integer(nd24$f) - 1L, X_new = PredictMat(sm24, nd24)))
# 24b. a REML gam fit with the fs term (no factor main effect, as mgcv's fs examples write it): fs smooths are not
# centred and side.constrain is FALSE, so the intercept lies in the span of the per-level null-space columns and only
# the null-space penalties make the fit identifiable; mgcv's rank is the full column count. predict.gam on new data
# (Predict.matrix.fs.interaction) with se.
set.seed(241); n <- 1500
d24b <- data.frame(x = runif(n, 1, 21), f = factor(sample(c("a", "b", "c"), n, TRUE), levels = c("a", "b", "c")),
                   n = sample(20:200, n, TRUE))
d24b$p <- rbinom(n, d24b$n, 1 - exp(-exp(-4 + 0.08 * (d24b$x - 10) + c(0, 0.3, -0.2)[as.integer(d24b$f)] +
                                         c(0.2, -0.1, 0.4)[as.integer(d24b$f)] * sin(d24b$x / 3)))) / d24b$n
f24b <- "p ~ s(x, f, bs='fs', xt='cr', k=6)"
m24b <- gam(as.formula(f24b), family = binomial(link = "cloglog"), weights = n, data = d24b,
            knots = list(x = sz_knots), method = "REML")
set.seed(242); nd24b <- data.frame(x = c(0.5, runif(60, 1, 21), 22),
                                   f = factor(sample(c("a", "b", "c"), 62, TRUE), levels = c("a", "b", "c")))
pr24b <- predict(m24b, nd24b, se.fit = TRUE)
put("gam_binomial_cloglog_fs_reml", c(gam_case(m24b, d24b, f24b, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), rank = m24b$rank, coef_names = names(coef(m24b)), newdata = nd24b,
       fit_new = as.numeric(pr24b$fit), se_new = as.numeric(pr24b$se.fit))))
# 24c. fs beside terms in the span of its null-space columns, on 24b's data, REML: (i) the global-plus-group-curves form
# s(x) + s(x, f, bs = "fs"), (ii) a factor main effect beside the fs term. side.constrain is FALSE for fs, so gam.side adds
# no constraint; only the penalties identify the fit. mgcv's rank is recorded.
f24c1 <- "p ~ s(x, bs='cr', k=6) + s(x, f, bs='fs', xt='cr', k=6)"
m24c1 <- gam(as.formula(f24c1), family = binomial(link = "cloglog"), weights = n, data = d24b,
             knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_cr_fs_reml", c(gam_case(m24c1, d24b, f24c1, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), rank = m24c1$rank, coef_names = names(coef(m24c1)))))
f24c2 <- "p ~ f + s(x, f, bs='fs', xt='cr', k=6)"
m24c2 <- gam(as.formula(f24c2), family = binomial(link = "cloglog"), weights = n, data = d24b,
             knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_factor_fs_reml", c(gam_case(m24c2, d24b, f24c2, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), rank = m24c2$rank, coef_names = names(coef(m24c2)))))
# 24d. fs in bam(method = "fREML", discrete = TRUE) on 24b's data. bam marks fs (tensor.possible) as a tensor.smooth.spec
# with one margin per written term, so discrete.mf compresses x and f in written order; gam.setup then builds the term on
# discrete.mf's padded frame mf0, so nat.param's P and smoothCon's penalty scaling are taken over mf0's rows, not the data.
# Both written orders (the padding draws differ), with P and S recorded, predict.bam on 24b's new data, and summary's
# s.table (fs has null.space.dim 0, so summary.gam takes reTest).
for (f24d in c("p ~ s(x, f, bs='fs', xt='cr', k=6)", "p ~ s(f, x, bs='fs', xt='cr', k=6)")) {
  b24d <- bam(as.formula(f24d), family = binomial(link = "cloglog"), weights = n, data = d24b,
              knots = list(x = sz_knots), method = "fREML", discrete = TRUE)
  pr24d <- predict(b24d, nd24b, se.fit = TRUE)
  put(if (grepl("s\\(x, f", f24d)) "bam_discrete_binomial_cloglog_fs" else "bam_discrete_binomial_cloglog_fs_factor_first",
      bam_disc_case(b24d, d24b, f24d, "binomial(link='cloglog')",
        c(list(knots = list(x = sz_knots), P = b24d$smooth[[1]]$P, S = b24d$smooth[[1]]$S, newdata = nd24b,
               fit_new = as.numeric(pr24d$fit), se_new = as.numeric(pr24d$se.fit),
               s_table = unname(summary(b24d)$s.table)), bam_rank(b24d))))
}
# 24e. select = TRUE with fs (smoothCon's null.space.penalty branch, smooth.r): fs has several penalties, so mgcv takes
# need.full, sums them and adds U U' only for eigenvalues below max * eps^0.66. The fs penalties sum to full rank, so the
# fs term gets no extra penalty and the fit equals the select = FALSE one (recorded as n_penalties and eta_vs_noselect).
# (i) 24b's REML fit, (ii) 24d's discrete bam in written order, (iii) 24c's s(x) + fs REML fit, where the cr smooth does
# get its null-space penalty.
m24e1 <- gam(as.formula(f24b), family = binomial(link = "cloglog"), weights = n, data = d24b,
             knots = list(x = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_fs_select_reml", c(gam_case(m24e1, d24b, f24b, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), select = TRUE, rank = m24e1$rank, coef_names = names(coef(m24e1)),
       n_penalties = sapply(m24e1$smooth, function(s) length(s$S)),
       eta_vs_noselect = max(abs(m24e1$linear.predictors - m24b$linear.predictors)))))
f24e2 <- "p ~ s(x, f, bs='fs', xt='cr', k=6)"
b24e2 <- bam(as.formula(f24e2), family = binomial(link = "cloglog"), weights = n, data = d24b,
             knots = list(x = sz_knots), method = "fREML", discrete = TRUE, select = TRUE)
b24e2n <- bam(as.formula(f24e2), family = binomial(link = "cloglog"), weights = n, data = d24b,
              knots = list(x = sz_knots), method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_fs_select", bam_disc_case(b24e2, d24b, f24e2, "binomial(link='cloglog')",
  c(list(knots = list(x = sz_knots), select = TRUE, n_penalties = sapply(b24e2$smooth, function(s) length(s$S)),
         eta_vs_noselect = max(abs(b24e2$linear.predictors - b24e2n$linear.predictors))), bam_rank(b24e2))))
m24e3 <- gam(as.formula(f24c1), family = binomial(link = "cloglog"), weights = n, data = d24b,
             knots = list(x = sz_knots), method = "REML", select = TRUE)
put("gam_binomial_cloglog_cr_fs_select_reml", c(gam_case(m24e3, d24b, f24c1, "binomial(link='cloglog')"),
  list(knots = list(x = sz_knots), select = TRUE, rank = m24e3$rank, coef_names = names(coef(m24e3)),
       n_penalties = sapply(m24e3$smooth, function(s) length(s$S)))))
# 25. the re random-effect term (smooth.construct.re.smooth.spec, smooth.r): model.matrix(~ g - 1), one identity
# penalty, null.space.dim 0, not centred. On 24b's frame with an eight-level factor g whose level effects enter the
# response. (a) beside a cr smooth as a REML gam; (b) the same as bam(method = "fREML", discrete = TRUE), where bam marks
# re tensor.possible and discrete.mf compresses g as a one-variable margin; (c) and (d) beside the fs term, REML gam and
# discrete bam. predict on new data (Predict.matrix.random.effect) with se; rank and coefficient names recorded.
set.seed(251)
d25 <- d24b[, c("x", "f", "n")]
d25$g <- factor(sample(paste0("g", 1:8), n, TRUE), levels = paste0("g", 1:8))
ge25 <- rnorm(8, 0, 0.3)
d25$p <- rbinom(n, d25$n, 1 - exp(-exp(-4 + 0.08 * (d25$x - 10) + c(0, 0.3, -0.2)[as.integer(d25$f)] +
                                       c(0.2, -0.1, 0.4)[as.integer(d25$f)] * sin(d25$x / 3) +
                                       ge25[as.integer(d25$g)]))) / d25$n
nd25 <- nd24b
nd25$g <- factor(sample(paste0("g", 1:8), nrow(nd25), TRUE), levels = paste0("g", 1:8))
re_extra <- function(m) {
  pr <- predict(m, nd25, se.fit = TRUE)
  list(knots = list(x = sz_knots), rank = m$rank, coef_names = names(coef(m)), newdata = nd25,
       fit_new = as.numeric(pr$fit), se_new = as.numeric(pr$se.fit))
}
f25a <- "p ~ s(x, bs='cr', k=6) + s(g, bs='re')"
f25c <- "p ~ s(x, f, bs='fs', xt='cr', k=6) + s(g, bs='re')"
m25a <- gam(as.formula(f25a), family = binomial(link = "cloglog"), weights = n, data = d25,
            knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_cr_re_reml", c(gam_case(m25a, d25, f25a, "binomial(link='cloglog')"), re_extra(m25a)))
b25b <- bam(as.formula(f25a), family = binomial(link = "cloglog"), weights = n, data = d25,
            knots = list(x = sz_knots), method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_cr_re", bam_disc_case(b25b, d25, f25a, "binomial(link='cloglog')",
                                                         c(re_extra(b25b), list(rank = b25b$rank))))
m25c <- gam(as.formula(f25c), family = binomial(link = "cloglog"), weights = n, data = d25,
            knots = list(x = sz_knots), method = "REML")
put("gam_binomial_cloglog_fs_re_reml", c(gam_case(m25c, d25, f25c, "binomial(link='cloglog')"), re_extra(m25c)))
b25d <- bam(as.formula(f25c), family = binomial(link = "cloglog"), weights = n, data = d25,
            knots = list(x = sz_knots), method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_fs_re", bam_disc_case(b25d, d25, f25c, "binomial(link='cloglog')",
                                                         c(re_extra(b25d), list(rank = b25d$rank))))
# 26. the te tensor product smooth on cr margins (te, smooth.construct.tensor.smooth.spec, tensor.prod.model.matrix,
# tensor.prod.penalties and Predict.matrix.tensor.smooth, smooth.r). Unlike ti, te builds each margin with
# smooth.construct (no margin constraint, unscaled penalty); cr margins set noterp, so the np = TRUE
# re-parameterisation is skipped; each margin penalty is divided by its largest eigenvalue; smoothCon then scales
# the tensor penalties and absorbs one sum-to-zero constraint on the whole product. 26a: two margins, x at supplied
# knots and z at default knots; 26a3: three margins at default knots. X, S, rank, null.space.dim, the margin knots
# and PredictMat at new values (outside the knot range too).
set.seed(26); n <- 300
d26 <- data.frame(x = runif(n, 1, 21), z = rnorm(n), w = runif(n, -2, 5))
nd26 <- data.frame(x = c(0.5, runif(25, 1, 21), 22), z = c(min(d26$z) - 0.3, rnorm(25), max(d26$z) + 0.2),
                   w = c(-2.5, runif(25, -2, 5), 5.5))
te_case <- function(sm, vars) list(data = d26[vars], newdata = nd26[vars], label = sm$label,
  margin_knots = lapply(sm$margin, function(mg) mg$xp), X = sm$X, S = sm$S, rank = sm$rank,
  null_space_dim = sm$null.space.dim, n_cons = attr(sm, "nCons"), X_new = PredictMat(sm, nd26))
sm26 <- smoothCon(te(x, z, bs = "cr", k = c(5, 4)), d26, knots = list(x = c(1, 3, 7, 12, 21)), absorb.cons = TRUE)[[1]]
put("smoothcon_te_cr", te_case(sm26, c("x", "z")))
sm26c <- smoothCon(te(x, z, w, bs = "cr", k = c(4, 4, 3)), d26, absorb.cons = TRUE)[[1]]
put("smoothcon_te_cr_three", te_case(sm26c, c("x", "z", "w")))
# 26b. te on cr in a REML gam fit (binomial cloglog, weights n): (i) p ~ te(x, z, k=c(5, 4)) with x's knots supplied;
# (ii) a factor main effect beside te(x, z) at default knots (k = 5 per margin, te's default). gam.side has
# nothing to do in either (no other smooth shares x or z). predict.gam on new data (Predict.matrix.tensor.smooth)
# with se, mgcv's rank and coefficient names.
set.seed(262); n <- 1500
d26b <- data.frame(x = runif(n, 1, 21), z = rnorm(n), f = factor(sample(c("a", "b", "c"), n, TRUE),
                   levels = c("a", "b", "c")), n = sample(20:200, n, TRUE))
d26b$p <- rbinom(n, d26b$n, 1 - exp(-exp(-4 + 0.08 * (d26b$x - 10) + 0.3 * sin(d26b$z) * cos(d26b$x / 4) +
                                         c(0, 0.25, -0.15)[as.integer(d26b$f)]))) / d26b$n
set.seed(263); nd26b <- data.frame(x = c(0.5, runif(60, 1, 21), 22), z = c(-3.5, rnorm(60), 3.5),
                                   f = factor(sample(c("a", "b", "c"), 62, TRUE), levels = c("a", "b", "c")))
te_knots <- list(x = c(1, 3, 7, 12, 21))
te_extra <- function(m, kn) {
  pr <- predict(m, nd26b, se.fit = TRUE)
  list(knots = kn, rank = m$rank, coef_names = names(coef(m)), newdata = nd26b,
       fit_new = as.numeric(pr$fit), se_new = as.numeric(pr$se.fit))
}
f26b1 <- "p ~ te(x, z, bs='cr', k=c(5, 4))"
m26b1 <- gam(as.formula(f26b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "REML")
put("gam_binomial_cloglog_te_reml", c(gam_case(m26b1, d26b, f26b1, "binomial(link='cloglog')"),
                                      te_extra(m26b1, te_knots)))
f26b2 <- "p ~ f + te(x, z)"
m26b2 <- gam(as.formula(f26b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML")
put("gam_binomial_cloglog_factor_te_reml", c(gam_case(m26b2, d26b, f26b2, "binomial(link='cloglog')"),
                                             te_extra(m26b2, NULL)))
# 26c. 26b's two models as bam(method = "fREML", discrete = TRUE). bam marks te tensor.smooth.spec: tero puts the
# margin with the largest bs.dim last (k = c(5, 4) becomes te(z,x), k = c(4, 5)), discrete.mf compresses each margin
# covariate on its own, and gam.setup builds the term on the padded frame mf0, so the penalty scaling and the
# sum-to-zero constraint are taken over mf0's rows. predict.bamd on new data, mgcv's rank and coefficient names, and
# the te term's constraint direction (con_dir, section 17), which over the data rows instead misses.
te_disc_extra <- function(b, kn) {
  te_sm <- b$smooth[[which(sapply(b$smooth, function(sm) inherits(sm, "tensor.smooth")))]]
  c(te_extra(b, kn), list(constraint = con_dir(te_sm)))
}
b26c1 <- bam(as.formula(f26b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_te", bam_disc_case(b26c1, d26b, f26b1, "binomial(link='cloglog')",
                                                      te_disc_extra(b26c1, te_knots)))
b26c2 <- bam(as.formula(f26b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "fREML",
             discrete = TRUE)
put("bam_discrete_binomial_cloglog_factor_te", bam_disc_case(b26c2, d26b, f26b2, "binomial(link='cloglog')",
                                                             te_disc_extra(b26c2, NULL)))
# 26d. select = TRUE with te. smoothCon's null.space.penalty branch takes need.full for a multi-penalty term: St is the
# sum of the (scaled, constrained) penalties (S[[1]] counted twice, which leaves the null space unchanged), and U U' is
# appended for the eigenvectors with eigenvalue below max * eps^0.66, rank sum(ind), null.space.dim 0. For te on two
# cr margins the null space of the sum is 2 x 2 less the constraint: three. 26d1: smoothCon on section 26's data with
# null.space.penalty = TRUE (the S list and rank). Then 26b's two models with select = TRUE as a REML gam (26d2, 26d3)
# and as a discrete bam (26d4, 26d5), each with predict on new data.
sm26d <- smoothCon(te(x, z, bs = "cr", k = c(5, 4)), d26, knots = list(x = c(1, 3, 7, 12, 21)), absorb.cons = TRUE,
                   null.space.penalty = TRUE)[[1]]
put("smoothcon_te_cr_select", te_case(sm26d, c("x", "z")))
m26d2 <- gam(as.formula(f26b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "REML", select = TRUE)
put("gam_binomial_cloglog_te_select_reml", c(gam_case(m26d2, d26b, f26b1, "binomial(link='cloglog')"),
                                             te_extra(m26d2, te_knots), list(select = TRUE)))
m26d3 <- gam(as.formula(f26b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML",
             select = TRUE)
put("gam_binomial_cloglog_factor_te_select_reml", c(gam_case(m26d3, d26b, f26b2, "binomial(link='cloglog')"),
                                                    te_extra(m26d3, NULL), list(select = TRUE)))
b26d4 <- bam(as.formula(f26b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "fREML", discrete = TRUE, select = TRUE)
put("bam_discrete_binomial_cloglog_te_select", bam_disc_case(b26d4, d26b, f26b1, "binomial(link='cloglog')",
                                                             c(te_disc_extra(b26d4, te_knots), list(select = TRUE))))
b26d5 <- bam(as.formula(f26b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "fREML",
             discrete = TRUE, select = TRUE)
put("bam_discrete_binomial_cloglog_factor_te_select",
    bam_disc_case(b26d5, d26b, f26b2, "binomial(link='cloglog')", c(te_disc_extra(b26d5, NULL), list(select = TRUE))))
# 27. the t2 tensor product smooth on cr margins, the basis (t2, smooth.construct.t2.smooth.spec, t2.model.matrix and
# Predict.matrix.t2.smooth, smooth.r). Each margin is built by smooth.construct (unconstrained, unscaled cr penalty) and
# re-parameterised by nat.param(type = 3, unit.fnorm = TRUE): penalty the identity on its range columns, the null space
# rotated so that the constant is its last column. t2.model.matrix forms the row products of range ("r") and null
# ("n") parts, one identity penalty per penalised block (names(S) is p.lab); full = FALSE. The fit constraint C sums the
# unpenalised (all-null) block only; smoothCon absorbs it by its sparse branch (indi, the nonzero columns), so the
# range columns are untouched. Cp = colSums(X) is the prediction constraint: smoothCon keeps Xp and Sp under it, and
# PredictMat uses it (attr qrc). 27a: two margins on section 26's data, x at supplied knots; 27a3: three margins at
# default knots. Before absorption (absorb.cons = FALSE) too: X, S, C and Cp.
t2_case <- function(sm, raw, vars) list(data = d26[vars], newdata = nd26[vars], label = sm$label,
  margin_knots = lapply(sm$margin, function(mg) mg$xp), margin_P = raw$P, p_lab = names(raw$S),
  X_raw = raw$X, S_raw = raw$S, C_raw = raw$C, Cp_raw = raw$Cp, rank_raw = raw$rank,
  null_space_dim_raw = raw$null.space.dim, X = sm$X, S = unname(sm$S), Xp = sm$Xp, Sp = unname(sm$Sp),
  rank = sm$rank, null_space_dim = sm$null.space.dim, n_cons = attr(sm, "nCons"), indi = attr(sm, "indi"),
  X_new = PredictMat(sm, nd26))
t2_kn <- list(x = c(1, 3, 7, 12, 21))
sm27 <- smoothCon(t2(x, z, bs = "cr", k = c(5, 4)), d26, knots = t2_kn, absorb.cons = TRUE)[[1]]
raw27 <- smoothCon(t2(x, z, bs = "cr", k = c(5, 4)), d26, knots = t2_kn, absorb.cons = FALSE)[[1]]
put("smoothcon_t2_cr", t2_case(sm27, raw27, c("x", "z")))
sm27c <- smoothCon(t2(x, z, w, bs = "cr", k = c(4, 4, 3)), d26, absorb.cons = TRUE)[[1]]
raw27c <- smoothCon(t2(x, z, w, bs = "cr", k = c(4, 4, 3)), d26, absorb.cons = FALSE)[[1]]
put("smoothcon_t2_cr_three", t2_case(sm27c, raw27c, c("x", "z", "w")))
# 27b. t2 on cr in a REML gam fit, on section 26b's data and new data (binomial cloglog, weights n): (i) p ~ t2(x, z,
# k=c(5, 4)) with x's knots supplied; (ii) a factor main effect beside t2(x, z) at default knots. gam fits in smoothCon's
# fit parameterisation (X, constraint C) and estimate.gam then maps coefficients, Vp and Ve to the prediction one (Xp,
# constraint Cp) with gam.setup's G$P; edf, eta and se come from before that map, and predict.gam uses Xp with the mapped
# coefficients. predict on new data with se, mgcv's rank and coefficient names.
f27b1 <- "p ~ t2(x, z, bs='cr', k=c(5, 4))"
m27b1 <- gam(as.formula(f27b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "REML")
put("gam_binomial_cloglog_t2_reml", c(gam_case(m27b1, d26b, f27b1, "binomial(link='cloglog')"),
                                      te_extra(m27b1, te_knots)))
f27b2 <- "p ~ f + t2(x, z)"
m27b2 <- gam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML")
# mgcv's default newton stops short of its own REML optimum here: the rn block's sp (x's range by z's null space) is on
# a flat ridge towards infinity (log sp 5.45 by default, 19.4 at a tight conv.tol, with a lower REML score), and the two
# mgcv fits differ by 1.2e-3 on eta. The fit is repeated at a tight newton conv.tol, the reference for quantities at
# estimated sp where its REML score is not higher than the default fit's (docs/PRIORITIES.md).
m27b2t <- gam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML",
              control = gam.control(newton = list(conv.tol = 1e-12)))
pr27b2t <- predict(m27b2t, nd26b, se.fit = TRUE)
put("gam_binomial_cloglog_factor_t2_reml", c(gam_case(m27b2, d26b, f27b2, "binomial(link='cloglog')"),
    te_extra(m27b2, NULL),
    list(tight = c(gam_case(m27b2t, d26b, f27b2, "binomial(link='cloglog')")[
      c("sp", "edf", "eta", "se", "reml", "outer_iter", "outer_conv")],
      list(conv_tol = 1e-12, fit_new = as.numeric(pr27b2t$fit), se_new = as.numeric(pr27b2t$se.fit))))))
# 27c. t2 on cr in bam(method = "fREML", discrete = TRUE). bam calls gam.setup(..., modCon = 2), so smoothCon sets the
# fit constraint to the prediction one (sm$C <- sm$Cp, colSums(X), dense: the full QR branch); fit and prediction
# parameterisations are then one and bam applies no G$P. A t2.smooth.spec is not a tensor.smooth.spec: tero leaves it
# alone and discrete.mf (nmarg = 1) compresses its covariates jointly, one compress.df call on (x, z) (uniquecombs orders
# multi-column rows by their text, paste0(x, "*", z), in the C locale). The term is built on the padded frame mf0.
# 27c1, 27c2: 27b's two models (every (x, z) pair unique, so compress.df returns the data unshuffled). 27c3: x on a
# 0.5 grid and z to one decimal, so pairs repeat: the unique pairs are shuffled and the columns padded. predict.bamd on
# new data, rank, coefficient names and the constraint direction (con_dir). 27c4: compress.df on two columns with
# repeated rows written in mixed notation, under set.seed(8547): the unique rows in mgcv's (shuffled) order.
t2_disc_extra <- function(b, kn, nd) {
  t2_sm <- b$smooth[[which(sapply(b$smooth, function(sm) inherits(sm, "t2.smooth")))]]
  pr <- predict(b, nd, se.fit = TRUE)
  list(knots = kn, rank = b$rank, coef_names = names(coef(b)), newdata = nd, fit_new = as.numeric(pr$fit),
       se_new = as.numeric(pr$se.fit), constraint = con_dir(t2_sm))
}
b27c1 <- bam(as.formula(f27b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "fREML", discrete = TRUE)
put("bam_discrete_binomial_cloglog_t2", bam_disc_case(b27c1, d26b, f27b1, "binomial(link='cloglog')",
                                                      t2_disc_extra(b27c1, te_knots, nd26b)))
b27c2 <- bam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "fREML",
             discrete = TRUE)
put("bam_discrete_binomial_cloglog_factor_t2", bam_disc_case(b27c2, d26b, f27b2, "binomial(link='cloglog')",
                                                             t2_disc_extra(b27c2, NULL, nd26b)))
set.seed(273); n <- 2000
d27c <- data.frame(x = round(runif(n, 1, 21) * 2) / 2, z = round(rnorm(n), 1), n = sample(20:200, n, TRUE))
d27c$p <- rbinom(n, d27c$n, 1 - exp(-exp(-4 + 0.08 * (d27c$x - 10) + 0.3 * sin(d27c$z) * cos(d27c$x / 4)))) / d27c$n
nd27c <- data.frame(x = c(0.5, runif(40, 1, 21), 22), z = c(-3.5, rnorm(40), 3.5))
f27c3 <- "p ~ t2(x, z, bs='cr', k=c(6, 5))"
b27c3 <- bam(as.formula(f27c3), family = binomial(link = "cloglog"), weights = n, data = d27c, method = "fREML",
             discrete = TRUE)
put("bam_discrete_binomial_cloglog_t2_repeated", bam_disc_case(b27c3, d27c, f27c3, "binomial(link='cloglog')",
    c(t2_disc_extra(b27c3, NULL, nd27c), list(n_unique_pairs = nrow(unique(d27c[c("x", "z")]))))))
set.seed(274)
dj <- data.frame(x = sample(c(1, 2.5, 10, 0.75, 1e5, 1e-4, -3, 12, 100, 0.001), 400, TRUE),
                 z = sample(c(-1.5, 0.2, 7, 30, 1e-5, 123456), 400, TRUE))
set.seed(8547); xu <- mgcv:::compress.df(dj)
put("compress_df_joint_order", list(columns = as.list(dj), nr = nrow(xu), x = xu$x, z = xu$z,
                                    index = as.integer(attr(xu, "index")), text = paste0(dj$x, "*", dj$z)))
# 27c5. more than 100^2 unique (x, z) pairs, so compress.df's grid branch fires on the joint term: each column is snapped
# to 100 points over its range before uniquecombs' text order and the shuffle (PR #47 review). The fit, and compress.df's
# shuffled rows and index under set.seed(8547).
set.seed(275); n <- 12000
d27e <- data.frame(x = runif(n, 1, 21), z = rnorm(n), n = sample(20:200, n, TRUE))
d27e$p <- rbinom(n, d27e$n, 1 - exp(-exp(-4 + 0.08 * (d27e$x - 10) + 0.3 * sin(d27e$z) * cos(d27e$x / 4)))) / d27e$n
f27c5 <- "p ~ t2(x, z, bs='cr', k=c(5, 4))"
b27c5 <- bam(as.formula(f27c5), family = binomial(link = "cloglog"), weights = n, data = d27e, method = "fREML",
             discrete = TRUE)
set.seed(8547); xu27e <- mgcv:::compress.df(d27e[c("x", "z")])
put("bam_discrete_binomial_cloglog_t2_binned", bam_disc_case(b27c5, d27e, f27c5, "binomial(link='cloglog')",
    c(t2_disc_extra(b27c5, NULL, nd27c), list(compress_nr = nrow(xu27e), compress_x = xu27e$x, compress_z = xu27e$z,
                                              compress_index = as.integer(attr(xu27e, "index"))))))
# 27d. select = TRUE with t2. smoothCon's null.space.penalty branch takes need.full for t2 (several penalties): St is
# the sum of the scaled, constrained penalties and U U' is appended over its eigenvectors with eigenvalue below
# max * eps^0.66 (rank 3 here: the 2 x 2 all-null block less the constraint), null.space.dim 0. The penalty is built in
# the fit parameterisation, so under gam's sparse C (modCon = 0) and under bam's dense Cp (modCon = 2); smoothCon adds
# none to Sp. 27d1: smoothCon on section 27's data at both modCon. Then 27b's two models with select = TRUE as a REML gam
# (27d2, 27d3; 27d3's rn sp is again on its flat ridge, so it is refitted at a tight newton conv.tol as in 27b) and as a
# discrete bam (27d4, 27d5), each with predict on new data.
t2s_case <- function(sm) list(data = d26[c("x", "z")], X = sm$X, S = unname(sm$S), rank = sm$rank,
                              null_space_dim = sm$null.space.dim)
put("smoothcon_t2_cr_select", list(
  mod_con_0 = t2s_case(smoothCon(t2(x, z, bs = "cr", k = c(5, 4)), d26, knots = t2_kn, absorb.cons = TRUE,
                                 null.space.penalty = TRUE)[[1]]),
  mod_con_2 = t2s_case(smoothCon(t2(x, z, bs = "cr", k = c(5, 4)), d26, knots = t2_kn, absorb.cons = TRUE,
                                 null.space.penalty = TRUE, modCon = 2)[[1]])))
m27d2 <- gam(as.formula(f27b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "REML", select = TRUE)
put("gam_binomial_cloglog_t2_select_reml", c(gam_case(m27d2, d26b, f27b1, "binomial(link='cloglog')"),
                                             te_extra(m27d2, te_knots), list(select = TRUE)))
m27d3 <- gam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML",
             select = TRUE)
m27d3t <- gam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "REML",
              select = TRUE, control = gam.control(newton = list(conv.tol = 1e-12)))
pr27d3t <- predict(m27d3t, nd26b, se.fit = TRUE)
put("gam_binomial_cloglog_factor_t2_select_reml", c(gam_case(m27d3, d26b, f27b2, "binomial(link='cloglog')"),
    te_extra(m27d3, NULL), list(select = TRUE),
    list(tight = c(gam_case(m27d3t, d26b, f27b2, "binomial(link='cloglog')")[
      c("sp", "edf", "eta", "se", "reml", "outer_iter", "outer_conv")],
      list(conv_tol = 1e-12, fit_new = as.numeric(pr27d3t$fit), se_new = as.numeric(pr27d3t$se.fit))))))
b27d4 <- bam(as.formula(f27b1), family = binomial(link = "cloglog"), weights = n, data = d26b, knots = te_knots,
             method = "fREML", discrete = TRUE, select = TRUE)
put("bam_discrete_binomial_cloglog_t2_select", bam_disc_case(b27d4, d26b, f27b1, "binomial(link='cloglog')",
    c(t2_disc_extra(b27d4, te_knots, nd26b), list(select = TRUE))))
b27d5 <- bam(as.formula(f27b2), family = binomial(link = "cloglog"), weights = n, data = d26b, method = "fREML",
             discrete = TRUE, select = TRUE)
put("bam_discrete_binomial_cloglog_factor_t2_select", bam_disc_case(b27d5, d26b, f27b2, "binomial(link='cloglog')",
    c(t2_disc_extra(b27d5, NULL, nd26b), list(select = TRUE))))
# 27e. mgcv's newton (gam.fit3.r) trajectory on REML gam fits whose stop depends on the search: the iteration count,
# the convergence message, the REML score after every iteration (outer.info$score.hist), the gradient at the stop and
# the log smoothing parameters. Data, formulas and default fits are in the named fixtures.
newton_case <- function(name, m) list(fixture = name, iter = m$outer.info$iter, conv = m$outer.info$conv,
                                      score_hist = m$outer.info$score.hist, grad = as.numeric(m$outer.info$grad),
                                      lsp = as.numeric(log(m$sp)))
put("gam_newton_trajectories", list(cases = list(
  newton_case("gam_binomial_cloglog_factor_t2_reml", m27b2),
  newton_case("gam_binomial_cloglog_factor_t2_select_reml", m27d3),
  newton_case("gam_binomial_cloglog_sz_select_reml", m10s),
  newton_case("gam_binomial_cloglog_cr_fs_select_reml", m24e3),
  newton_case("gam_binomial_cloglog_by_numeric_reml", m14))))
# 28. the thin-plate regression spline, one covariate (smooth.construct.tp.smooth.spec, smooth.r; construct_tprs, tprs.c),
# mgcv's default smooth. Fewer than max.knots = 2000 distinct values, so no subsampling. x is rounded to two decimals
# so values repeat (the basis is built on the unique values). 28a: smoothCon(s(x, bs = "tp", k = 8), absorb.cons = TRUE)
# on the data: X and S (compared up to the eigenvector gauge, tests/test_tp_fit.py). 28b: p ~ s(x) + s(z, bs = "cr",
# k = 6) as a REML gam (binomial cloglog, weights n), predict on new data with se, including x outside its range.
# 28c: two tp smooths at set k in a Poisson REML gam with an exposure offset. 28d: 28b with select = TRUE.
set.seed(28); n <- 1200
d28 <- data.frame(x = round(runif(n, 0, 10), 2), z = runif(n, -1, 1), n = sample(20:200, n, TRUE))
d28$p <- rbinom(n, d28$n, 1 - exp(-exp(-3.5 + 0.4 * sin(d28$x / 1.6) + 0.3 * d28$z^2))) / d28$n
nd28 <- data.frame(x = c(-0.5, 0, 2.345, 5, 7.77, 10, 10.5), z = c(0, -1, 0.3, 0.9, -0.4, 1, 0.2))
sc28 <- smoothCon(s(x, bs = "tp", k = 8), d28, absorb.cons = TRUE)[[1]]
put("smoothcon_tp_k8", list(x = d28$x, k = 8, X = sc28$X, S = sc28$S[[1]], null_space_dim = sc28$null.space.dim,
                            rank = sc28$rank))
f28b <- "p ~ s(x) + s(z, bs='cr', k=6)"
m28b <- gam(as.formula(f28b), family = binomial(link = "cloglog"), weights = n, data = d28, method = "REML")
pr28b <- predict(m28b, nd28, se.fit = TRUE)
put("gam_binomial_cloglog_tp_reml", c(gam_case(m28b, d28, f28b, "binomial(link='cloglog')"),
    list(rank = m28b$rank, coef_names = names(coef(m28b)), newdata = nd28,
         fit_new = as.numeric(pr28b$fit), se_new = as.numeric(pr28b$se.fit))))
d28$expo <- round(runif(n, 200, 5000)); d28$log_expo <- log(d28$expo)
d28$deaths <- rpois(n, d28$expo * exp(-6 + 0.3 * cos(d28$x / 2) - 0.4 * d28$z))
f28c <- "deaths ~ s(x, bs='tp', k=6) + s(z, bs='tp', k=5) + offset(log_expo)"
m28c <- gam(as.formula(f28c), family = poisson, data = d28, method = "REML")
pr28c <- predict(m28c, cbind(nd28, log_expo = 0), se.fit = TRUE)
put("gam_poisson_tp_reml", c(gam_case(m28c, d28, f28c, "poisson"),
    list(rank = m28c$rank, coef_names = names(coef(m28c)), newdata = nd28,
         fit_new = as.numeric(pr28c$fit), se_new = as.numeric(pr28c$se.fit))))
# 28d. 28b's model with select = TRUE: smoothCon's null.space.penalty adds a penalty on the tp term's null space (the
# linear function left after the constraint).
m28d <- gam(as.formula(f28b), family = binomial(link = "cloglog"), weights = n, data = d28, method = "REML",
            select = TRUE)
pr28d <- predict(m28d, nd28, se.fit = TRUE)
put("gam_binomial_cloglog_tp_select_reml", c(gam_case(m28d, d28, f28b, "binomial(link='cloglog')"),
    list(rank = m28d$rank, coef_names = names(coef(m28d)), newdata = nd28,
         fit_new = as.numeric(pr28d$fit), se_new = as.numeric(pr28d$se.fit))))
cat("fixtures written to", out, "with mgcv", meta$mgcv, "tier", meta$tier, "\n")
