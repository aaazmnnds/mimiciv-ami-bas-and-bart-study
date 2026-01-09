################################################################################
# DATA SIMULATION SCRIPT (MCAR, MAR, MNAR)
#
# Logic:
# 1. Use TOP 4 variables with highest missingness as predictors.
# 2. Randomly assign β from {1.5, 1.0, 0.5, 0.1} to these 4 variables.
# 3. Match outcome ratios (0s/1s) from real datasets (MIMIC-III, MI).
#
# Process:
# 1. Load data.
# 2. Generate missing values (MCAR, MAR, MNAR).
# 3. Simulate outcome Y based on true betas.
# 4. Save complete datasets with missing indicators.
#
################################################################################

library(missMethods)

set.seed(123)

# ============================================================================
# CONFIGURATION
# ============================================================================

datasets_config <- list(
  MIMIC = list(
    name = "MIMIC-III",
    original = "cleaned.mi (mimiciii).csv",
    missforest = "cleaned.mimic_missForest_imputed.csv",
    response = "ICD9_CODE",
    output_prefix = "MIMIC"
  ),
  MI = list(
    name = "Myocardial Infarction",
    original = "cleaned.mi1 (myocardial infarction).csv",
    missforest = "cleaned.mi_missForest_imputed.csv",
    response = "ZSN",
    output_prefix = "MI"
  )
)

N_TOP_VARS <- 4
BETA_VALUES_POOL <- c(1.5, 1.0, 0.5, 0.1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Simulate binary outcome matching target proportion of 1s
simulate_outcome_with_ratio <- function(X_true, beta_true, target_prop_ones, dataset_name = "Dataset") {
  
  n <- nrow(X_true)
  X_matrix <- as.matrix(X_true)
  
  # Base linear predictor
  base_linear_pred <- X_matrix %*% beta_true
  
  # Function to calculate proportion of 1s given intercept
  calc_prop_ones <- function(intercept) {
    linear_pred <- intercept + base_linear_pred
    probs <- plogis(linear_pred)
    return(mean(probs))
  }
  
  # Optimization to find best intercept
  intercept_low <- -50
  intercept_high <- 50
  
  # Binary search
  for(iter in 1:50) {
    intercept_mid <- (intercept_low + intercept_high) / 2
    prop_mid <- calc_prop_ones(intercept_mid)
    
    if(abs(prop_mid - target_prop_ones) < 0.001) break
    
    if(prop_mid < target_prop_ones) {
      intercept_low <- intercept_mid
    } else {
      intercept_high <- intercept_mid
    }
  }
  
  best_intercept <- (intercept_low + intercept_high) / 2
  
  # Generate final outcome
  linear_pred_final <- best_intercept + base_linear_pred
  probs_final <- plogis(linear_pred_final)
  
  set.seed(456)
  y_sim <- rbinom(n, 1, probs_final)
  
  return(list(y = y_sim, intercept = best_intercept, prop_ones = mean(y_sim)))
}

# Create complete dataset with indicators
create_complete_dataset <- function(X_missing, Y, response_var_name) {
  missing_indicators <- as.data.frame(ifelse(is.na(X_missing), 1, 0))
  names(missing_indicators) <- paste0(names(X_missing), "_missing")
  total_missing_values <- rowSums(is.na(X_missing))
  
  complete_dataset <- data.frame(
    Y = Y,
    X_missing,
    missing_indicators,
    total_missing_values = total_missing_values
  )
  names(complete_dataset)[1] <- response_var_name
  return(complete_dataset)
}

# Reorder columns: Response, Var1, Var1_missing, Var2...
reorder_with_indicators <- function(df, response_var) {
  all_cols <- names(df)
  value_cols <- all_cols[!grepl("_missing|total_missing", all_cols) & all_cols != response_var]
  
  new_order <- c(response_var)
  for(var in value_cols) {
    new_order <- c(new_order, var)
    indicator_col <- paste0(var, "_missing")
    if(indicator_col %in% all_cols) {
      new_order <- c(new_order, indicator_col)
    }
  }
  new_order <- c(new_order, "total_missing_values")
  return(df[, new_order])
}


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

cat("Starting Simulation...\n")

for(dataset_name in names(datasets_config)) {
  config <- datasets_config[[dataset_name]]
  cat(sprintf("\nProcessing %s...\n", config$name))
  
  # 1. Load Data & Determine Target Ratio
  if (!file.exists(config$original)) {
      cat(sprintf("Warning: Original file %s not found. Skipping.\n", config$original))
      next
  }
  
  data_original <- read.csv(config$original)
  outcome_table <- table(data_original[[config$response]])
  target_prop_ones <- as.numeric(outcome_table["1"]) / sum(outcome_table)
  cat(sprintf("  Target 1s Ratio: %.2f%%\n", target_prop_ones * 100))
  
  # Load MissForest Data (Base for simulation)
  if (!file.exists(config$missforest)) {
      cat(sprintf("Warning: Input file %s not found. Skipping.\n", config$missforest))
      next
  }
  data_mf <- read.csv(config$missforest)
  
  # Extract value columns
  value_cols_mf <- !grepl("_missing|total_missing", names(data_mf)) & names(data_mf) != config$response
  X_full <- as.data.frame(data_mf[, value_cols_mf])
  original_miss_rate <- 0.1 # Approximate default if unknown, or calculate from original
  
  # Re-calculate miss rate from original to be precise
  value_cols_orig <- !grepl("_missing|total_missing", names(data_original)) & names(data_original) != config$response
  X_orig_vals <- data_original[, value_cols_orig]
  original_miss_rate <- mean(is.na(X_orig_vals))
  
  # 2. Create Missingness
  cat("  Generating MCAR, MAR, MNAR datasets...\n")
  p <- ncol(X_full)
  half_p <- floor(p / 2)
  
  models <- list()
  models$MCAR <- delete_MCAR(X_full, p = original_miss_rate, cols_mis = 1:p)
  models$MAR  <- delete_MAR_censoring(X_full, p = original_miss_rate, cols_mis = (half_p + 1):(2 * half_p), cols_ctrl = 1:half_p)
  models$MNAR <- delete_MNAR_censoring(X_full, p = original_miss_rate, cols_mis = 1:p)
  
  # 3. Simulate Outcome & Save
  for(mech in names(models)) {
    X_missing <- models[[mech]]
    
    # Identify variables with highest missingness
    miss_rates <- colMeans(is.na(X_missing))
    top_idx <- order(miss_rates, decreasing = TRUE)[1:N_TOP_VARS]
    top_vars <- names(X_missing)[top_idx]
    
    # Randomly assign betas
    set.seed(123 + which(names(models) == mech))
    selected_betas <- sample(BETA_VALUES_POOL, N_TOP_VARS, replace = FALSE)
    
    beta_true <- rep(0, p)
    names(beta_true) <- names(X_full)
    beta_true[top_idx] <- selected_betas
    
    # Simulate Y
    sim_res <- simulate_outcome_with_ratio(X_full[, top_idx], selected_betas, target_prop_ones)
    Y_sim <- sim_res$y
    
    # Create and Save Complete Dataset
    complete_data <- create_complete_dataset(X_missing, Y_sim, config$response)
    complete_data <- reorder_with_indicators(complete_data, config$response)
    
    filename <- paste0("complete_dataset_", config$output_prefix, "_", mech, ".csv")
    write.csv(complete_data, filename, row.names = FALSE)
    
    # Save Truth Info
    truth_info <- data.frame(
        variable = top_vars,
        beta_true = selected_betas,
        missingness_rate = miss_rates[top_idx]
    )
    truth_filename <- paste0(config$output_prefix, "_", mech, "_true_variables.csv")
    write.csv(truth_info, truth_filename, row.names = FALSE)
    
    cat(sprintf("  Saved %s and truth info.\n", filename))
  }
}

cat("\nSimulation Complete.\n")
