# Poisson-family references: counts with a log-exposure offset (the production form), and quasi-Poisson rates with weights.
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
df <- read.csv("hgam_synth.csv", stringsAsFactors = FALSE)
df$FaceSize <- factor(df$FaceSize, levels = c("Low", "High")); df$Smoke <- factor(df$Smoke, levels = c("NS", "S"))
df$rate <- df$DthCnt / df$ExposCnt
rhs <- "FaceSize * Smoke + s(AttdAge, bs = 'cr', k = 13) + s(PolYear, bs = 'cr', k = 6)"
save_fit <- function(m, tag, secs, shift = 0) {
  eta <- as.numeric(predict(m, newdata = df, type = "link")) - shift
  write.csv(data.frame(eta = eta), paste0("ref2_", tag, ".csv"), row.names = FALSE)
  write_json(list(n_coef = length(coef(m)), edf_total = sum(m$edf), secs = unname(secs), scale = m$sig2,
                  smooth_edf = setNames(as.list(summary(m)$edf), sapply(m$smooth, function(s) s$label)),
                  sp = as.list(m$sp), pcoef = as.list(coef(m)[1:4])),
             paste0("ref2_", tag, ".json"), auto_unbox = TRUE, digits = 12, pretty = TRUE)
  eta
}
t1 <- system.time(m_off <- gam(as.formula(paste("DthCnt ~", rhs, "+ offset(log(ExposCnt))")), family = poisson, data = df, method = "REML"))["elapsed"]
t2 <- system.time(m_qoff <- gam(as.formula(paste("DthCnt ~", rhs, "+ offset(log(ExposCnt))")), family = quasipoisson, data = df, method = "REML"))["elapsed"]
t3 <- system.time(m_qrate <- gam(as.formula(paste("rate ~", rhs)), family = quasipoisson, weights = ExposCnt, data = df, method = "REML"))["elapsed"]
e1 <- save_fit(m_off, "pois_offset_gam", t1, shift = log(df$ExposCnt))
e2 <- save_fit(m_qoff, "qpois_offset_gam", t2, shift = log(df$ExposCnt))
e3 <- save_fit(m_qrate, "qpois_rate_gam", t3)
cat(sprintf("mgcv Poisson offset: edf %.4f; quasi offset: edf %.4f scale %.4f; quasi rate-form: edf %.4f scale %.4f\n",
            sum(m_off$edf), sum(m_qoff$edf), m_qoff$sig2, sum(m_qrate$edf), m_qrate$sig2))
cat(sprintf("max |d log-rate|: quasi offset vs Poisson offset %.3g; quasi rate-form vs quasi offset %.3g\n",
            max(abs(e2 - e1)), max(abs(e3 - e2))))
