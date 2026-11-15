# mgcv references on the forms the Rust port accepts: 0/1 rows with weights, and Poisson rates.
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
df <- read.csv("hgam_synth.csv", stringsAsFactors = FALSE)
df$FaceSize <- factor(df$FaceSize, levels = c("Low", "High"))
df$Smoke <- factor(df$Smoke, levels = c("NS", "S"))
df$cell <- seq_len(nrow(df))

# two rows per cell: deaths (y = 1, weight = d) and survivors (y = 0, weight = E - d); zero-weight rows dropped
ex <- rbind(transform(df, y01 = 1, w01 = DthCnt), transform(df, y01 = 0, w01 = ExposCnt - DthCnt))
ex <- ex[ex$w01 > 0, ]
ex <- ex[order(ex$cell, -ex$y01), ]
write.csv(ex[, c("cell", "AttdAge", "PolYear", "StudyYear_C", "FaceSize", "Smoke", "y01", "w01")], "hgam_expanded.csv", row.names = FALSE)

reduced_rhs <- "FaceSize * Smoke + s(AttdAge, bs = 'cr', k = 13) + s(PolYear, bs = 'cr', k = 6)"
save_fit <- function(m, tag, secs, newdata = df, shift = 0) {
  eta <- as.numeric(predict(m, newdata = newdata, type = "link")) - shift
  tm <- predict(m, newdata = newdata, type = "terms")
  write.csv(data.frame(eta = eta, s_AttdAge = tm[, "s(AttdAge)"], s_PolYear = tm[, "s(PolYear)"]),
            paste0("ref2_", tag, ".csv"), row.names = FALSE)
  write_json(list(n_coef = length(coef(m)), edf_total = sum(m$edf), secs = unname(secs),
                  smooth_edf = setNames(as.list(summary(m)$edf), sapply(m$smooth, function(s) s$label)),
                  sp = as.list(m$sp), pcoef = as.list(coef(m)[1:4]), deviance = deviance(m),
                  knots_AttdAge = m$smooth[[1]]$xp, knots_PolYear = m$smooth[[2]]$xp),
             paste0("ref2_", tag, ".json"), auto_unbox = TRUE, digits = 12, pretty = TRUE)
  eta
}

# (1) proportions with weights (the target's response form) vs the same model on 0/1 rows: must be one model
t_p <- system.time(m_prop <- gam(as.formula(paste("y ~", reduced_rhs)), family = binomial, weights = ExposCnt, data = df, method = "REML"))["elapsed"]
t_e <- system.time(m_exp <- gam(as.formula(paste("y01 ~", reduced_rhs)), family = binomial, weights = w01, data = ex, method = "REML"))["elapsed"]
eta_prop <- save_fit(m_prop, "logit_prop_gam", t_p)
eta_exp  <- save_fit(m_exp, "logit_exp_gam", t_e)
cat(sprintf("proportion-form vs 0/1-form (mgcv gam REML): max |d eta| = %.3g; edf %.6f vs %.6f\n",
            max(abs(eta_prop - eta_exp)), sum(m_prop$edf), sum(m_exp$edf)))

# (2) bam fREML, exact and discrete, on the 0/1 rows
t_b <- system.time(b_exact <- bam(as.formula(paste("y01 ~", reduced_rhs)), family = binomial, weights = w01, data = ex, method = "fREML"))["elapsed"]
t_d <- system.time(b_disc <- bam(as.formula(paste("y01 ~", reduced_rhs)), family = binomial, weights = w01, data = ex, method = "fREML", discrete = TRUE))["elapsed"]
eta_b <- save_fit(b_exact, "logit_exp_bam", t_b); eta_d <- save_fit(b_disc, "logit_exp_bam_discrete", t_d)
cat(sprintf("mgcv bam fREML vs gam REML: max |d eta| = %.3g; bam discrete vs gam REML: %.3g\n",
            max(abs(eta_b - eta_exp)), max(abs(eta_d - eta_exp))))

# (3) Poisson-family references live in reference3.R (mgcv's plain Poisson REML returns NaN on non-integer rates)
cat(sprintf("rows: cells %d, expanded %d\n", nrow(df), nrow(ex)))
