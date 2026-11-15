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
cat("fixtures written to", out, "with mgcv", meta$mgcv, "tier", meta$tier, "\n")
