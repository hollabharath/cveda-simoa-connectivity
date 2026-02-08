# =============================================================================
# 04_run_clinical.R
# Clinical models: GFAP and NfL ~ Age*EXT + covariates (Gamma GLM, log link)
# Interaction plots, Johnson-Neyman, regression tables, post-hoc power.
# =============================================================================
#
# Requires: ggplot2, ggpubr, interactions, gtsummary
#   install.packages(c("ggplot2", "ggpubr", "interactions", "gtsummary"))
#
# =============================================================================

library(ggplot2)
library(ggpubr)
library(interactions)
library(gtsummary)

# -----------------------------------------------------------------------------
# Configuration: paths and column names (edit for your data)
# -----------------------------------------------------------------------------

path_data <- "data_clinical.csv"
path_fig  <- "figures"   # directory for saved plots (created if missing)
dir.create(path_fig, showWarnings = FALSE)

# Outcome columns (biomarker concentrations, e.g. pg/ml)
col_GFAP <- "GFAP.Conc..pg.ml"
col_NFL  <- "NF.L.Conc..pg.ml"

# Predictors and covariates (must match column names in data)
col_age   <- "Age"
col_ext   <- "EXT"
col_sex   <- "sex"
col_bmi   <- "BmiZscore"
col_cog   <- "gFc_T"
col_run   <- "Run"

# Johnson-Neyman moderator range (e.g. age range in years)
jn_range <- c(6, 23)

# Plot and save options
save_plots <- TRUE
fig_width_mm <- 180
fig_height_mm <- 100
fig_dpi <- 600

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

dat <- read.csv(path_data, stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------
# GFAP: Gamma GLM (log link), Age*EXT + covariates
# -----------------------------------------------------------------------------

form_GFAP <- as.formula(paste(col_GFAP, "~", col_age, "*", col_ext, "+", col_sex,
                              "+", col_bmi, "+", col_cog, "+", col_run))
m1 <- glm(form_GFAP, data = dat, family = Gamma(link = "log"))

# Interaction plot and Johnson-Neyman (do.call so column name strings are used, not NSE)
p1 <- do.call(interact_plot, list(
  model = m1, pred = col_age, modx = col_ext ,
  colors = c("blue", "red"), 
  rug = TRUE, interval = TRUE, main.title = "", y.label = "GFAP (pg/ml)"
)) + theme_bw() + theme(legend.position = "bottom")

jn1 <- do.call(johnson_neyman, list(
  model = m1, pred = col_ext , modx = col_age, alpha = 0.05,
  mod.range = jn_range, control.fdr = TRUE,
  title = "", y.label = "Conditional Effect of EXT on GFAP"
))
p2 <- jn1$plot + theme_bw() + guides(linetype = "none") +
  theme(legend.position = "bottom", legend.title = ggplot2::element_blank())

p_GFAP <- ggarrange(p1, p2, labels = c("A", "B"))
print(p_GFAP)
if (save_plots) {
  ggsave(file.path(path_fig, "GFAP_EXT_AGE_scatter.pdf"), p_GFAP,
         width = fig_width_mm, height = fig_height_mm, units = "mm", dpi = fig_dpi)
}

# Regression table (exponentiated = Mean Ratios)
m1 %>%
  tbl_regression(exponentiate = TRUE, add_estimate_to_reference_rows = TRUE) %>%
  modify_caption(paste0("Gamma GLM (log link) for GFAP concentration: ",
                        "Age by EXT interaction, adjusted for ", col_sex, ", ", col_bmi,
                        ", ", col_cog, ", and ", col_run, ". Coefficients exponentiated (Mean Ratios)."))
print(m1)

# Post-hoc power for EXT effect (two-tailed, alpha 0.05)
sm1 <- coef(summary(m1))
b_ext  <- sm1[col_ext, "Estimate"]
se_ext <- sm1[col_ext, "Std. Error"]
z_ext  <- abs(b_ext / se_ext)
power_EXT_GFAP <- pnorm(z_ext - qnorm(0.975))
message("Post-hoc power (EXT, GFAP model): ", round(power_EXT_GFAP, 3))

# -----------------------------------------------------------------------------
# NfL: Gamma GLM (log link), Age*EXT + covariates
# -----------------------------------------------------------------------------

form_NFL <- as.formula(paste(col_NFL, "~", col_age, "*", col_ext, "+", col_sex,
                             "+", col_bmi, "+", col_cog, "+", col_run))
m2 <- glm(form_NFL, data = dat, family = Gamma(link = "log"))

p3 <- do.call(interact_plot, list(
  model = m2, pred = col_age , modx = col_ext,
  colors = c("blue", "red"), rug = TRUE, interval = TRUE, int.width = 0.68,
  main.title = "", y.label = "NFL (pg/ml)"
)) + theme_bw() + theme(legend.position = "bottom")

jn2 <- do.call(johnson_neyman, list(
  model = m2, pred = col_ext, modx = col_age, alpha = 0.05,
  mod.range = jn_range, control.fdr = TRUE,
  title = "", y.label = "Conditional Effect of EXT on NFL"
))
p4 <- jn2$plot + theme_bw() + guides(linetype = "none") +
  theme(legend.position = "bottom", legend.title = ggplot2::element_blank())

p_NFL <- ggarrange(p3, p4, labels = c("C", "D"))
print(p_NFL)
if (save_plots) {
  ggsave(file.path(path_fig, "NFL_EXT_AGE.png"), p_NFL,
         width = 7, height = 4, dpi = fig_dpi)
}

m2 %>%
  tbl_regression(exponentiate = TRUE, add_estimate_to_reference_rows = TRUE) %>%
  modify_caption(paste0("Gamma GLM (log link) for NFL concentration: ",
                        "Age by EXT interaction, adjusted for ", col_sex, ", ", col_bmi,
                        ", ", col_cog, ", and ", col_run, ". Coefficients exponentiated (Mean Ratios)."))
print(m2)

sm2 <- coef(summary(m2))
power_EXT_NFL <- pnorm(abs(sm2[col_ext, "Estimate"] / sm2[col_ext, "Std. Error"]) - qnorm(0.975))
message("Post-hoc power (EXT, NFL model): ", round(power_EXT_NFL, 3))
