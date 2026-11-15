# Synthetic frame shaped like the hgam_formula target, plus mgcv reference fits.
# Output: hgam_synth.csv, ref_full.json, ref_reduced_*.csv/json
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
set.seed(20261114)
n <- 30000
AttdAge     <- runif(n, 1, 95)
PolYear     <- runif(n, 1, 21)
StudyYear_C <- runif(n, -8.5, 8.5)
FaceSize    <- factor(sample(c("Low", "High"), n, replace = TRUE, prob = c(0.6, 0.4)), levels = c("Low", "High"))
Smoke       <- factor(sample(c("NS", "S"), n, replace = TRUE, prob = c(0.8, 0.2)), levels = c("NS", "S"))
ExposCnt    <- round(runif(n, 50, 500))
fH <- as.numeric(FaceSize == "High"); sS <- as.numeric(Smoke == "S")
eta_true <- -4.5 + 0.03 * AttdAge - 0.02 * PolYear +
  0.15 * sin(AttdAge / 10) * cos(PolYear / 3) +                 # age x duration
  0.01 * StudyYear_C * (AttdAge - 50) / 50 +                    # improvement varying by age
  fH * (-0.25 + 0.004 * (AttdAge - 50) + 0.010 * (PolYear - 10)) +
  sS * ( 0.55 - 0.006 * (AttdAge - 50) + 0.015 * (PolYear - 10)) +
  0.10 * fH * sS
p_true <- 1 - exp(-exp(eta_true))
DthCnt <- rbinom(n, ExposCnt, p_true)
df <- data.frame(AttdAge, PolYear, StudyYear_C, FaceSize, Smoke, ExposCnt, DthCnt)
df$y <- df$DthCnt / df$ExposCnt
write.csv(df, "hgam_synth.csv", row.names = FALSE)

my_knots <- list(PolYear = c(1, 2, 3, 5, 10, 21),
                 AttdAge = c(1, 2, 4, 7, 14, 18, 24, 35, 50, 70, 85, 90, 95))
hgam_formula <- (DthCnt / ExposCnt) ~
  FaceSize + Smoke + FaceSize:Smoke +
  s(AttdAge, k = 13, bs = "cr") +
  s(PolYear, k = 6,  bs = "cr") +
  ti(AttdAge, PolYear, k = c(13, 6), bs = "cr") +
  s(FaceSize, AttdAge, bs = "sz", k = 13, xt = list(bs = "cr")) +
  s(Smoke,    AttdAge, bs = "sz", k = 13, xt = list(bs = "cr")) +
  s(FaceSize, PolYear, bs = "sz", k = 6,  xt = list(bs = "cr")) +
  s(Smoke,    PolYear, bs = "sz", k = 6,  xt = list(bs = "cr")) +
  s(AttdAge, by = StudyYear_C, k = 13, bs = "cr")

describe <- function(m, secs) list(
  n_coef = length(coef(m)), n_smooth = length(m$smooth), n_sp = length(m$sp),
  edf_total = sum(m$edf), secs = unname(secs),
  smooth_edf = setNames(as.list(summary(m)$edf), sapply(m$smooth, function(s) s$label)))

# --- full target, exactly as specified
t1 <- system.time(full_sel <- bam(hgam_formula, family = binomial(link = "cloglog"), weights = ExposCnt,
                                  data = df, knots = my_knots, discrete = TRUE, select = TRUE))["elapsed"]
t0 <- system.time(full_nosel <- bam(hgam_formula, family = binomial(link = "cloglog"), weights = ExposCnt,
                                    data = df, knots = my_knots, discrete = TRUE, select = FALSE))["elapsed"]
write_json(list(select_true = describe(full_sel, t1), select_false = describe(full_nosel, t0),
                mgcv = as.character(packageVersion("mgcv")), r = R.version.string),
           "ref_full.json", auto_unbox = TRUE, digits = 10, pretty = TRUE)
write.csv(data.frame(eta_full = as.numeric(predict(full_sel, type = "link"))), "ref_full_eta.csv", row.names = FALSE)

# --- reduced models: only what a one-smooth-per-covariate additive interface can express
reduced <- y ~ FaceSize * Smoke + s(AttdAge, bs = "cr", k = 13) + s(PolYear, bs = "cr", k = 6)
fit_reduced <- function(link, tag) {
  secs <- system.time(m <- gam(reduced, family = binomial(link = link), weights = ExposCnt, data = df, method = "REML"))["elapsed"]
  d <- describe(m, secs)
  d$sp <- as.list(m$sp); d$reml <- as.numeric(m$gcv.ubre); d$deviance <- deviance(m)
  d$pcoef <- as.list(coef(m)[1:4])
  d$knots_AttdAge <- m$smooth[[1]]$xp; d$knots_PolYear <- m$smooth[[2]]$xp
  write_json(d, paste0("ref_reduced_", tag, ".json"), auto_unbox = TRUE, digits = 12, pretty = TRUE)
  tm <- predict(m, type = "terms")
  out <- data.frame(eta = as.numeric(predict(m, type = "link")),
                    s_AttdAge = tm[, "s(AttdAge)"], s_PolYear = tm[, "s(PolYear)"])
  write.csv(out, paste0("ref_reduced_", tag, ".csv"), row.names = FALSE)
  m
}
m_logit   <- fit_reduced("logit", "logit")
m_cloglog <- fit_reduced("cloglog", "cloglog")

# how much the full target differs from the reduced cloglog model on the link scale
cat(sprintf("full target: %d coef, %d smooths, %d sp (select=TRUE), edf %.2f; select=FALSE: %d sp, edf %.2f\n",
            length(coef(full_sel)), length(full_sel$smooth), length(full_sel$sp), sum(full_sel$edf),
            length(full_nosel$sp), sum(full_nosel$edf)))
d_eta <- predict(full_sel, type = "link") - predict(m_cloglog, type = "link")
cat(sprintf("full vs reduced(cloglog): max |d eta| = %.4f, rms = %.4f\n", max(abs(d_eta)), sqrt(mean(d_eta^2))))
cat(sprintf("deviance: full %.1f, reduced cloglog %.1f, reduced logit %.1f\n", deviance(full_sel), deviance(m_cloglog), deviance(m_logit)))
