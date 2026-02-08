# =============================================================================
# 05_run_sPLS.R
# Sparse Partial Least Squares (sPLS) for brain connectivity vs. biomarkers
# =============================================================================
#
# Methods (multivariate modeling):
#   - Separate sPLS regressions per outcome (e.g. GFAP, NfL).
#   - Predictors: brain connectivity deviation (Z) features; standardized.
#   - (1) Cross-validation (cv.spls) for optimal sparsity (eta) and K components.
#   - (2) Fit sPLS with optimized parameters.
#   - (3) Bootstrap (ci.spls, 10,000 resamples); retain predictors whose 95% CI
#         excludes zero (correct.spls).
#
# Requires: R package 'spls'
#   install.packages("spls")
#
# =============================================================================

library(spls)

# -----------------------------------------------------------------------------
# Configuration: paths and column names (edit for your data)
# -----------------------------------------------------------------------------

# Path to analysis dataset (relative to working directory or absolute).
# Dataset must contain:
#   - Outcome variable(s), e.g. GFAP and NfL (concentrations or log-transformed).
#   - Predictor columns: connectivity deviation (Z) scores (e.g. 136 BNFC edges),
#     in the same order as your template (e.g. VisualA_VisualB ... DefaultC_TemporalParietal).
path_data <- "data_connectivity_biomarkers.csv"

# Column name(s) of outcome(s) to model (one model per outcome).
outcome_names <- c("GFAP_n", "NFL_n")

# How to define predictor columns (connectivity features). Option A or B:
#   A) By first and last column names (inclusive).
pred_first <- "VisualA_VisualB"
pred_last  <- "DefaultC_TemporalParietal"

#   B) Or set pred_cols explicitly, e.g.:
# pred_cols <- setdiff(names(dat), c("sub_id", "age", "sex", outcome_names))

# Cross-validation search grid (sparsity and number of components).
eta_grid <- seq(0.1, 0.9, 0.1)
K_grid   <- 3:10

# Bootstrap for confidence intervals.
n_boot    <- 10000
ci_level  <- 0.95
set_seed  <- 1

# Multiple comparison correction over tested predictors (sPLS-selected only).
# "none" = no correction; "bonferroni" = FWER; "BH" = FDR (Benjamini-Hochberg).
# P-values are approximated from the bootstrap 95% CI (SE from CI width, then 2-tailed normal).
multi_correction <- "BH"

# -----------------------------------------------------------------------------
# Load data and define predictor matrix
# -----------------------------------------------------------------------------

dat <- read.csv(path_data, stringsAsFactors = FALSE)

# Predictor columns: connectivity features (between first and last, inclusive).
idx_first <- which(names(dat) == pred_first)
idx_last  <- which(names(dat) == pred_last)
if (length(idx_first) != 1 || length(idx_last) != 1)
  stop("pred_first and pred_last must each match exactly one column name.")
pred_cols <- names(dat)[idx_first:idx_last]

# Predictor matrix (standardized) and sample size
X_raw <- as.matrix(dat[, pred_cols])
X     <- scale(X_raw, center = TRUE, scale = TRUE)
if (is.null(colnames(X))) colnames(X) <- pred_cols

n <- nrow(X)
if (n != nrow(dat)) stop("Row mismatch between dat and X.")

# -----------------------------------------------------------------------------
# Run sPLS for each outcome
# -----------------------------------------------------------------------------

results <- list()

for (outcome_name in outcome_names) {

  if (!outcome_name %in% names(dat))
    stop("Outcome column not found: ", outcome_name)

  Y <- as.matrix(dat[[outcome_name]])
  if (any(is.na(Y))) stop("Outcome ", outcome_name, " contains NA; remove or impute.")

  message("\n========== Outcome: ", outcome_name, " ==========")

  # ---- Step 1: Cross-validation for eta and K ----
  set.seed(set_seed)
  cv_fit <- cv.spls(X, Y, eta = eta_grid, K = K_grid)
  eta_opt <- cv_fit$eta.opt
  K_opt   <- cv_fit$K.opt
  message("CV optimal: eta = ", eta_opt, ", K = ", K_opt)

  # ---- Step 2: Fit sPLS with optimal parameters ----
  fit <- spls::spls(X, Y, eta = eta_opt, K = K_opt)
  print(fit)

  # Variance explained (R²) and per-component R²
  Y_hat   <- predict(fit)
  Y_true  <- as.numeric(Y)
  R2      <- 1 - sum((Y_true - Y_hat)^2) / sum((Y_true - mean(Y_true))^2)
  message("Total R² = ", round(R2, 4))

  # Component scores and correlation with outcome (squared = variance explained per component)
  selected_vars <- rownames(fit$projection)
  X_sel         <- X[, selected_vars, drop = FALSE]
  X_centered    <- scale(X_sel, center = fit$meanx[selected_vars], scale = FALSE)
  W             <- fit$projection
  components    <- as.matrix(X_centered) %*% as.matrix(W)
  colnames(components) <- paste0("Component_", seq_len(fit$K))
  R2_components <- apply(components, 2, function(z) cor(z, Y_true)^2)
  message("Variance explained per component: ",
          paste(round(100 * R2_components, 2), "%", collapse = ", "))

  # ---- Step 3: Bootstrap CIs; retain only predictors whose CI excludes zero ----
  set.seed(set_seed)
  ci_fit <- ci.spls(fit, coverage = ci_level, B = n_boot,
                    plot.it = FALSE, plot.fix = "y", plot.var = TRUE,
                    K = fit$K, fit = fit$fit)
  # Corrected coefficients from bootstrap (beta = 0 for non-significant; rownames = predictor names)
  cf <- correct.spls(ci_fit)

  # Coefficient CIs (cibeta: matrix with rows = predictors, cols = lower, upper)
  cibeta <- ci_fit$cibeta
  if (is.list(cibeta)) cibeta <- cibeta[[1]]
  lb <- cibeta[, 1]
  ub <- cibeta[, 2]
  betas <- coef(fit)
  betas <- betas[betas != 0]

  # Variable names: same order as cibeta rows (use rownames if present, else names(betas))
  var_names <- if (!is.null(rownames(cibeta)) && length(rownames(cibeta)) == nrow(cibeta)) {
    rownames(cibeta)
  } else {
    names(betas)
  }
  # Significant predictors: 95% CI excludes zero (interval does not contain 0)
  sig_pred <- var_names[(lb > 0) | (ub < 0)]
  n_sig    <- length(sig_pred)
  n_total   <- ncol(X)
  n_selected <- length(selected_vars)

  # Optional multiple comparison correction (over n_selected tested predictors)
  sig_pred_final <- sig_pred
  n_sig_corrected <- n_sig
  if (multi_correction %in% c("bonferroni", "BH") && length(var_names) > 0) {
    beta_point <- cf[var_names, 1]
    if (is.null(names(beta_point))) names(beta_point) <- var_names
    na_beta <- is.na(beta_point)
    beta_point[na_beta] <- (lb[na_beta] + ub[na_beta]) / 2
    SE_approx <- (ub - lb) / (2 * 1.96)
    SE_approx[SE_approx <= 0] <- 1e-10
    p_approx  <- 2 * pnorm(-abs(beta_point) / SE_approx)
    p_adj     <- p.adjust(p_approx, method = if (multi_correction == "bonferroni") "bonferroni" else "BH")
    sig_pred_corrected <- var_names[p_adj < 0.05]
    n_sig_corrected    <- length(sig_pred_corrected)
    sig_pred_final     <- sig_pred_corrected
    message("After ", multi_correction, " correction: ", n_sig_corrected, " predictor(s) significant (uncorrected: ", n_sig, ").")
  }

  message("After bootstrap correction: ", n_sig, " predictor(s) with 95% CI excluding zero.")

  # Beta from corrected coefficients (cf); fallback to midpoint of CI if cf has no rownames/match
  beta_sig <- cf[sig_pred_final, 1]
  if (is.null(names(beta_sig))) names(beta_sig) <- sig_pred_final
  if (length(sig_pred_final) > 0 && any(is.na(beta_sig))) beta_sig[is.na(beta_sig)] <- (cibeta[sig_pred_final, 1][is.na(beta_sig)] + cibeta[sig_pred_final, 2][is.na(beta_sig)]) / 2

  # Variance explained using significant pairs only (after multi-test correction if applied)
  R2_sig <- NA_real_
  R2_components_sig <- NULL
  if (length(sig_pred_final) > 0) {
    sig_in_X <- intersect(sig_pred_final, colnames(X))
    if (length(sig_in_X) == 0) sig_in_X <- intersect(sig_pred_final, pred_cols)
    if (length(sig_in_X) > 0) {
      X_sig    <- X[, sig_in_X, drop = FALSE]
      beta_use <- beta_sig[sig_in_X]
      Y_hat_sig <- as.numeric(X_sig %*% beta_use)
      R2_sig   <- 1 - sum((Y_true - Y_hat_sig)^2) / sum((Y_true - mean(Y_true))^2)
      message("Variance explained (significant pairs only): R² = ", round(R2_sig, 4))
      # Per-predictor squared correlation with Y (marginal variance explained by each)
      R2_components_sig <- setNames(
        vapply(sig_in_X, function(v) cor(X_sig[, v] * beta_use[v], Y_true)^2, 0),
        sig_in_X
      )
    }
  }
  message("Correction summary: sPLS selected ", n_selected, " of ", n_total, " predictors; ",
          n_sig, " remained significant after bootstrap",
          if (multi_correction %in% c("bonferroni", "BH")) paste0("; ", n_sig_corrected, " after ", multi_correction) else "",
          ". R² (full sPLS) = ", round(100 * R2, 2), "%; R² (significant only) = ",
          if (length(sig_pred_final) > 0 && !is.na(R2_sig)) round(100 * R2_sig, 2) else "—", "%.")

  # Store for reporting / export
  res <- list(
    outcome_name     = outcome_name,
    eta_opt          = eta_opt,
    K_opt            = K_opt,
    R2               = R2,
    R2_sig           = R2_sig,
    R2_components    = R2_components,
    R2_components_sig = R2_components_sig,
    fit              = fit,
    coef             = betas,
    coef_corrected   = cf,
    cibeta           = cibeta,
    significant      = sig_pred,
    significant_final = sig_pred_final,
    n_sig_corrected  = n_sig_corrected,
    multi_correction = multi_correction,
    components       = components,
    Y_hat            = Y_hat,
    Y_true           = Y_true
  )
  results[[outcome_name]] <- res

  # Optional: add component scores and composite to dataset for this outcome
  comp_df <- as.data.frame(components)
  names(comp_df) <- paste0(outcome_name, "_", names(comp_df))
  n_obs <- nrow(X)
  # Composite = X_selected %*% coefs (same order: columns of X_sel match order of non-zero coefs from fit)
  coefs <- coef(fit)
  coefs <- coefs[coefs != 0]
  if (length(coefs) == ncol(X_sel)) {
    composite <- as.numeric(as.matrix(X_sel) %*% coefs)
  } else {
    # coef(fit) may be full-length; take coefficients for selected vars by position
    idx <- match(selected_vars, colnames(X))
    coefs_sel <- coef(fit)[idx]
    composite <- as.numeric(as.matrix(X_sel) %*% coefs_sel)
  }
  if (length(composite) != n_obs) composite <- rep(NA_real_, n_obs)
  dat[[paste0(outcome_name, "_Component_Composite")]] <- composite
  dat <- cbind(dat, comp_df)

  # Print results immediately (significant net pairs with beta and CI, by sign)
  cat("\n========== ", outcome_name, " ==========\n", sep = "")
  cat("eta = ", eta_opt, ", K = ", K_opt, "\n", sep = "")
  cat("Variables: ", n_selected, " selected by sPLS, ", n_sig, " significant after bootstrap", sep = "")
  if (multi_correction %in% c("bonferroni", "BH")) cat(" (", n_sig_corrected, " after ", multi_correction, ")", sep = "")
  cat(" (of ", n_total, " total).\n", sep = "")
  cat("R² (full sPLS) = ", round(100 * R2, 2), "%; R² (significant pairs only) = ", sep = "")
  if (length(sig_pred_final) > 0 && !is.na(R2_sig)) {
    cat(round(100 * R2_sig, 2), "%.\n", sep = "")
  } else {
    cat(round(100 * R2, 2), "%.\n", sep = "")
  }
  if (length(sig_pred_final) == 0) {
    cat("No significant connectivity features (95% CI excluding zero", if (multi_correction %in% c("bonferroni", "BH")) paste0("; ", multi_correction, " applied") else "", ").\n")
  } else {
    # Beta from correct.spls (bootstrap-corrected coefficients)
    ci_lo     <- cibeta[sig_pred_final, 1]
    ci_hi     <- cibeta[sig_pred_final, 2]
    beta_vals <- cf[sig_pred_final, 1]
    if (is.null(names(beta_vals))) names(beta_vals) <- sig_pred_final
    if (any(is.na(beta_vals))) beta_vals[is.na(beta_vals)] <- (ci_lo[is.na(beta_vals)] + ci_hi[is.na(beta_vals)]) / 2
    pos <- sig_pred_final[beta_vals > 0]
    neg <- sig_pred_final[beta_vals < 0]
    cat("\nSignificant net pairs (positive association with ", outcome_name, "):\n", sep = "")
    if (length(pos) > 0) {
      for (v in pos) {
        b  <- beta_vals[v]
        ci <- c(ci_lo[v], ci_hi[v])
        cat("  ", v, "  beta = ", round(b, 3), ", 95% CI [", round(ci[1], 3), ", ", round(ci[2], 3), "]\n", sep = "")
      }
    } else {
      cat("  (none)\n")
    }
    cat("\nSignificant net pairs (negative association with ", outcome_name, "):\n", sep = "")
    if (length(neg) > 0) {
      for (v in neg) {
        b  <- beta_vals[v]
        ci <- c(ci_lo[v], ci_hi[v])
        cat("  ", v, "  beta = ", round(b, 3), ", 95% CI [", round(ci[1], 3), ", ", round(ci[2], 3), "]\n", sep = "")
      }
    } else {
      cat("  (none)\n")
    }
  }
}

# -----------------------------------------------------------------------------
# Optional: write results to CSV (coefficients and CIs per outcome)
# -----------------------------------------------------------------------------
#

for (outcome_name in names(results)) {
  res  <- results[[outcome_name]]
  vars <- rownames(res$cibeta)
  if (is.null(vars) || length(vars) == 0) next
  beta_vals <- res$coef_corrected[vars, 1]
  if (is.null(names(beta_vals))) names(beta_vals) <- vars
  tbl  <- data.frame(
    outcome     = outcome_name,
    predictor   = vars,
    beta        = beta_vals,
    CI_lower    = res$cibeta[, 1],
    CI_upper    = res$cibeta[, 2],
    significant = vars %in% res$significant
  )
  write.csv(tbl, paste0("sPLS_", outcome_name, "_coefficients.csv"), row.names = FALSE)
}

write.csv(dat, "data_connectivity_biomarkers_with_sPLS.csv", row.names = FALSE)
