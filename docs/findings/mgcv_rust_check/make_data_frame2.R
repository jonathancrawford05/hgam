# Second synthetic frame: a mortality-shaped age curve (Gompertz slope, childhood and young-adult features),
# a select-period duration effect, and sparse deaths at younger ages. Same columns as the first frame.
set.seed(20261115)
n <- 30000
AttdAge     <- runif(n, 1, 95)
PolYear     <- runif(n, 1, 21)
StudyYear_C <- runif(n, -8.5, 8.5)
FaceSize    <- factor(sample(c("Low", "High"), n, replace = TRUE, prob = c(0.6, 0.4)), levels = c("Low", "High"))
Smoke       <- factor(sample(c("NS", "S"), n, replace = TRUE, prob = c(0.8, 0.2)), levels = c("NS", "S"))
ExposCnt    <- round(runif(n, 200, 5000))
fH <- as.numeric(FaceSize == "High"); sS <- as.numeric(Smoke == "S")
eta_true <- -6.2 + 0.075 * (AttdAge - 40) + 1.2 * exp(-AttdAge / 3) + 0.35 * exp(-((AttdAge - 22) / 6)^2) +
  -0.5 * exp(-PolYear / 4) +                                        # select period
  0.10 * sin(AttdAge / 12) * cos(PolYear / 4) +                     # age x duration
  -0.012 * StudyYear_C * (1 - (AttdAge - 50) / 100) +               # improvement varying by age
  fH * (-0.20 + 0.003 * (AttdAge - 50) + 0.008 * (PolYear - 10)) +
  sS * ( 0.60 - 0.007 * (AttdAge - 50) + 0.012 * (PolYear - 10)) +
  0.08 * fH * sS
p_true <- 1 - exp(-exp(eta_true))
DthCnt <- rbinom(n, ExposCnt, p_true)
df <- data.frame(AttdAge, PolYear, StudyYear_C, FaceSize, Smoke, ExposCnt, DthCnt)
df$y <- df$DthCnt / df$ExposCnt
write.csv(df, "hgam_synth.csv", row.names = FALSE)
cat(sprintf("cells %d; total deaths %d; cells with zero deaths %.1f%%; crude rate %.5f\n",
            n, sum(DthCnt), 100 * mean(DthCnt == 0), sum(DthCnt) / sum(ExposCnt)))
