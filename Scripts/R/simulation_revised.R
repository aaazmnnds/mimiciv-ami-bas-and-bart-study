
library(missMethods)
library(missForest)

set.seed(123)

# = :
# CONFIGURATION

datasets_config <- list(
  MIMIC = list(
    name = "MIMIC-III",
    original = "../../Data/cleaned.mi (mimiciii).csv",
    missforest = "../../Data/cleaned.mimic_missForest_imputed.csv",
    response = "ICD9_CODE",
    output_prefix = "MIMIC"
  ),
  MI = list(
    name = "Myocardial Infarction",
    original = "../../Data/cleaned.mi (myocardial infarction).csv",
    missforest = "../../Data/cleaned.mi_missForest_imputed.csv",
    response = "ZSN",
    output_prefix = "MI"
  )
)

N_TOP_VARS <- 4
BETA_VALUES_POOL <- c(1.5, 1.0, 0.5, 0.1)
SELECTION_METHODS <- c("top", "independent", "mixed")

# HELPER FUNCTIONS

simulate_outcome_with_ratio <- function(X_true, beta_true, target_prop_ones) {
  n <- nrow(X_true)
  X_matrix <- as.matrix(X_true)
  base_linear_pred <- X_matrix %*% beta_true
  
  calc_prop_ones <- function(intercept) {
    linear_pred <- intercept + base_linear_pred
    probs <- plogis(linear_pred)
    return(mean(probs))
  }
  
  intercept_low <- -50
  intercept_high <- 50
  for(iter in 1:50) {
    intercept_mid <- (intercept_low + intercept_high) / 2
    prop_mid <- calc_prop_ones(intercept_mid)
    if(abs(prop_mid - target_prop_ones) < 0.001) break
    if(prop_mid < target_prop_ones) intercept_low <- intercept_mid else intercept_high <- intercept_mid
  }
  best_intercept <- (intercept_low + intercept_high) / 2
  linear_pred_final <- best_intercept + base_linear_pred
  probs_final <- plogis(linear_pred_final)
  y_sim <- rbinom(n, 1, probs_final)
  return(list(y = y_sim, intercept = best_intercept, prop_ones = mean(y_sim)))
}

create_complete_dataset <- function(X_missing, Y, response_var_name) {
  missing_indicators <- as.data.frame(ifelse(is.na(X_missing), 1, 0))
  names(missing_indicators) <- paste0(names(X_missing), "_missing")
  total_missing_values <- rowSums(is.na(X_missing))
  complete_dataset <- data.frame(Y = Y, X_missing, missing_indicators, total_missing_values = total_missing_values)
  names(complete_dataset)[1] <- response_var_name
  return(complete_dataset)
}

reorder_with_indicators <- function(df, response_var) {
  all_cols <- names(df)
  value_cols <- all_cols[!grepl("_missing|total_missing", all_cols) & all_cols != response_var]
  new_order <- c(response_var)
  for(var in value_cols) {
    new_order <- c(new_order, var); indicator_col <- paste0(var, "_missing")
    if(indicator_col %in% all_cols) new_order <- c(new_order, indicator_col)
  }
  new_order <- c(new_order, "total_missing_values")
  return(df[, new_order])
}

# MAIN PROCESSING

cat("Starting Revised Simulation...\n")

for(dataset_name in names(datasets_config)) {
  config <- datasets_config[[dataset_name]]
  cat(sprintf("\nProcessing %s...\n", config$name))
  
  if (!file.exists(config$original)) { cat(sprintf("  Warning: %s not found.\n", config$original)); next }
  
  data_original <- read.csv(config$original)
  outcome_table <- table(data_original[[config$response]])
  target_prop_ones <- as.numeric(outcome_table["1"]) / sum(outcome_table)
  
  if (file.exists(config$missforest)) {
      cat(sprintf("  Loading ground truth: %s\n", config$missforest))
      data_mf <- read.csv(config$missforest)
  } else {
      cat(sprintf("  Ground truth %s not found. Generating via missForest...\n", config$missforest))
      
      # Determine value columns for imputation
      value_cols_orig <- !grepl("_missing|total_missing", names(data_original)) & names(data_original) != config$response
      data_for_imp <- data_original[, value_cols_orig]
      
      # Handle character columns
      char_cols <- sapply(data_for_imp, is.character)
      if (any(char_cols)) {
        data_for_imp[, char_cols] <- lapply(data_for_imp[, char_cols], as.factor)
      }
      
      # Run missForest
      mf_result <- missForest(data_for_imp, verbose=FALSE)
      X_full_imputed <- mf_result$ximp
      
      # Reattach response variable
      data_mf <- cbind(X_full_imputed, data_original[config$response])
      
      # Save for future use
      write.csv(data_mf, config$missforest, row.names = FALSE)
      cat(sprintf("  Saved generated ground truth to: %s\n", config$missforest))
  }

  value_cols_mf <- !grepl("_missing|total_missing", names(data_mf)) & names(data_mf) != config$response
  X_full <- as.data.frame(data_mf[, value_cols_mf])
  original_miss_rate <- mean(is.na(data_original[, !grepl("_missing|total_missing", names(data_original)) & names(data_original) != config$response]))
  
  p <- ncol(X_full)
  half_p <- floor(p / 2)
  models <- list()
  models$MCAR <- delete_MCAR(X_full, p = original_miss_rate, cols_mis = 1:p)
  models$MAR  <- delete_MAR_censoring(X_full, p = original_miss_rate, cols_mis = (half_p + 1):(2 * half_p), cols_ctrl = 1:half_p)
  models$MNAR <- delete_MNAR_censoring(X_full, p = original_miss_rate, cols_mis = 1:p)

  for(method in SELECTION_METHODS) {
    cat(sprintf("  Running Selection Method: %s\n", method))
    
    for(mech in names(models)) {
      X_missing <- models[[mech]]
      miss_rates <- colMeans(is.na(X_missing))
      
      if(method == "top") {
        top_idx <- order(miss_rates, decreasing = TRUE)[1:N_TOP_VARS]
      } else if(method == "independent") {
        set.seed(42)
        top_idx <- sample(1:p, N_TOP_VARS)
      } else if(method == "mixed") {
        top_h <- order(miss_rates, decreasing = TRUE)[1:(N_TOP_VARS/2)]
        top_l <- order(miss_rates, decreasing = FALSE)[1:(N_TOP_VARS/2)]
        top_idx <- c(top_h, top_l)
      }
      
      top_vars <- names(X_missing)[top_idx]
      set.seed(123 + which(names(models) == mech))
      selected_betas <- sample(BETA_VALUES_POOL, N_TOP_VARS, replace = FALSE)
      
      sim_res <- simulate_outcome_with_ratio(X_full[, top_idx], selected_betas, target_prop_ones)
      Y_sim <- sim_res$y
      
      complete_data <- create_complete_dataset(X_missing, Y_sim, config$response)
      complete_data <- reorder_with_indicators(complete_data, config$response)
      
      # Save with Scenario Identifier
      dir.create(file.path("../../Data", method), showWarnings = FALSE)
      filename <- paste0("../../Data/", method, "/complete_dataset_", config$output_prefix, "_", mech, ".csv")
      write.csv(complete_data, filename, row.names = FALSE)
      
      truth_info <- data.frame(variable = top_vars, beta_true = selected_betas, missingness_rate = miss_rates[top_idx])
      truth_filename <- paste0("../../Data/", method, "/", config$output_prefix, "_", mech, "_true_variables.csv")
      write.csv(truth_info, truth_filename, row.names = FALSE)
    }
  }
}
cat("\nRevised Simulation Complete.\n")
