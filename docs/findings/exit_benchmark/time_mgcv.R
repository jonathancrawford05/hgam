# Time mgcv::bam(discrete = TRUE, select = TRUE) on the exit-test model: oracle/make_fixtures.R section 3's frame
# (n = 4000), knots and formula as written. Median of `reps` runs after one warm-up. Usage: Rscript time_mgcv.R [reps]
suppressPackageStartupMessages(library(mgcv))
reps <- as.integer(commandArgs(trailingOnly = TRUE)[1]); if (is.na(reps)) reps <- 15
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
fit <- function() bam(hgam_formula, family = binomial(link = "cloglog"), weights = ExposCnt, data = f,
                      knots = my_knots, discrete = TRUE, select = TRUE)
invisible(fit())
t <- replicate(reps, system.time(fit())[["elapsed"]])
cat(sprintf("mgcv %s, %s: median %.4f s, min %.4f s, max %.4f s over %d runs\n", packageVersion("mgcv"),
            R.version.string, median(t), min(t), max(t), reps))
