# =============================================================================
# 06_run_fMRI.R
# fMRI connectivity ~ Age*EXT + covariates: lm per edge, bootstrap CI, omega², heatmap
# =============================================================================
#
# For each between-network connectivity edge, fits lm(edge ~ formula).
# Uses bootstrap BCa for CIs, partial omega-squared for effect size.
# Plots heatmap of EXT main effect across edges.
#
# Requires: parameters, effectsize, tidyverse (dplyr, tidyr), ggplot2
#   install.packages(c("parameters", "effectsize", "tidyverse"))
#
# =============================================================================

library(parameters)
library(effectsize)
library(tidyverse)

# -----------------------------------------------------------------------------
# Configuration: paths and column names (edit for your data)
# -----------------------------------------------------------------------------

path_data   <- "data_connectivity_clinical.csv"
path_fig    <- "figures"
path_results <- "results"
dir.create(path_fig, showWarnings = FALSE)
dir.create(path_results, showWarnings = FALSE)

# Connectivity columns: first and last edge name (inclusive)
pred_first <- "VisualA_VisualB"
pred_last  <- "DefaultC_TemporalParietal"

# RHS of formula (no outcome); must match column names in data
# Example: "Age*EXT+Sex+mean_FD" or "Age * EXT + sex + mean_FD"
formula_rhs <- "Age*EXT+Sex+mean_FD"

# Bootstrap and model options
n_boot       <- 1000
ci_level     <- 0.95
alpha_sig    <- 0.05
save_results <- TRUE

# -----------------------------------------------------------------------------
# Load data and define connectivity edges
# -----------------------------------------------------------------------------

dat <- read.csv(path_data, stringsAsFactors = FALSE)
idx_first <- which(names(dat) == pred_first)
idx_last  <- which(names(dat) == pred_last)
if (length(idx_first) != 1 || length(idx_last) != 1)
  stop("pred_first and pred_last must each match exactly one column name.")
edge_cols <- names(dat)[idx_first:idx_last]

# -----------------------------------------------------------------------------
# Fit lm per edge; bootstrap BCa CI and partial omega-squared
# -----------------------------------------------------------------------------

results <- data.frame(
  edge           = character(),
  term           = character(),
  Estimate       = numeric(),
  CI_low         = numeric(),
  CI_high        = numeric(),
  p_value        = numeric(),
  omega_squared  = numeric(),
  stringsAsFactors = FALSE
)

for (edge in edge_cols) {
  formula_str <- paste(edge, "~", formula_rhs)
  formula     <- as.formula(formula_str)
  model       <- lm(formula, data = dat)

  boot_m <- model_parameters(
    model,
    ci_method = "bcai",
    bootstrap = TRUE,
    iteration = n_boot,
    standardize = "refit",
    robust = TRUE,
    ci = ci_level,
    p_digits = "scientific"
  )
  boot_m$p <- summary(model)[["coefficients"]][, "Pr(>|t|)"]

  omega_m <- omega_squared(model, partial = TRUE, ci = ci_level, alternative = "two.sided")

  boot_results <- as.data.frame(boot_m)
  boot_results$Parameter_for_match <- boot_results$Parameter
  boot_results$Parameter_for_match <- gsub("sexM", "sex", boot_results$Parameter_for_match)
  boot_results$omega_squared <- omega_m$Omega2_partial[match(boot_results$Parameter_for_match, omega_m$Parameter)]

  coef_df <- boot_results %>%
    dplyr::select(Parameter, Coefficient, CI_low, CI_high, p, omega_squared) %>%
    dplyr::rename(term = Parameter,
                  Estimate = Coefficient,
                  p_value = p) %>%
    dplyr::mutate(edge = edge)

  results <- dplyr::bind_rows(results, coef_df)
}

# -----------------------------------------------------------------------------
# Subsets by term (optional: save to CSV)
# -----------------------------------------------------------------------------

results_Age     <- dplyr::filter(results, term == "Age")
results_AgeEXT  <- dplyr::filter(results, term == "Age:EXT")
results_EXT     <- dplyr::filter(results, term == "EXT")

if (save_results) {
  write.csv(results,     file.path(path_results, "fMRI_lm_all_terms.csv"), row.names = FALSE)
  write.csv(results_EXT, file.path(path_results, "fMRI_lm_EXT_effect.csv"), row.names = FALSE)
}

# -----------------------------------------------------------------------------
# Heatmap: main effect of EXT across edges (half matrix, significance)
# -----------------------------------------------------------------------------

df_heat <- results_EXT %>%
  dplyr::mutate(pair = edge) %>%
  tidyr::separate(pair, into = c("network1", "network2"), sep = "_", remove = FALSE) %>%
  dplyr::mutate(sig = ifelse(p_value < alpha_sig, "*", ""))

network_order <- unique(c(df_heat$network1, df_heat$network2))
df_heat$network1 <- factor(df_heat$network1, levels = network_order)
df_heat$network2 <- factor(df_heat$network2, levels = network_order)

p_heat <- ggplot(df_heat, aes(x = network1, y = network2, fill = Estimate)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sig), size = 3) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  coord_fixed() +
  labs(title = "Main Effect of Externalizing Diagnosis\non Between-Network Connectivity",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_heat)
ggsave(file.path(path_fig, "EXT_effect_connectivity_heatmap.pdf"), p_heat,
       width = 8, height = 7, dpi = 300)
