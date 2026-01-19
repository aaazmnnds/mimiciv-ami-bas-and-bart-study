################################################################################
# BAS.GLM ANALYSIS SCRIPT (Refactored)
#
# Generates evaluation artifacts for:
# 1. Bias (Beta Estimates)
# 2. Variable Selection (Selected Vars)
# 3. Prediction (Predictions & Log Probs)
#
# Conditions:
# - wMI (With Missing Indicators)
# - noMI (Without Missing Indicators)
#
# Outputs to: Results/
################################################################################

library(BAS)
library(dplyr)

set.seed(123)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

NUM_OF_FOLDS <- 10
TOP_NUM <- 20
ITER <- 1500

# M-values for MICE
M_VALUES <- c(3, 20) 

# Dataset Configs (Must match [Dataset]_[Mechanism]_[Method] naming for evaluation)
# Real datasets are excluded from Bias evaluation usually, but included here for completeness if needed.
# For simplicity, we focus on SIMULATED datasets as per evaluation scripts expectation.

configs <- list(
  
  # --- MIMIC MCAR ---
  list(name = "MIMIC_MCAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MCAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MCAR_MEAN.csv", 
       y_col = "ICD9_CODE"),
       
  list(name = "MIMIC_MCAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMIMIC_MCAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MCAR_MICE_%d.csv", 
       y_col = "ICD9_CODE"),
       
  list(name = "MIMIC_MCAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMIMIC_MCAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MCAR_missForest.csv", 
       y_col = "ICD9_CODE"),
  
  list(name = "MIMIC_MCAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MCAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MCAR_KNN.csv", 
       y_col = "ICD9_CODE"),

  # --- MIMIC MAR ---
  list(name = "MIMIC_MAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MAR_MEAN.csv", 
       y_col = "ICD9_CODE"),
  
  list(name = "MIMIC_MAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMIMIC_MAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MAR_MICE_%d.csv", 
       y_col = "ICD9_CODE"),

  list(name = "MIMIC_MAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMIMIC_MAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MAR_missForest.csv", 
       y_col = "ICD9_CODE"),

  list(name = "MIMIC_MAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MAR_KNN.csv", 
       y_col = "ICD9_CODE"),

  # --- MIMIC MNAR ---
  list(name = "MIMIC_MNAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MNAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MNAR_MEAN.csv", 
       y_col = "ICD9_CODE"),
  
  list(name = "MIMIC_MNAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMIMIC_MNAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MNAR_MICE_%d.csv", 
       y_col = "ICD9_CODE"),

  list(name = "MIMIC_MNAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMIMIC_MNAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MNAR_missForest.csv", 
       y_col = "ICD9_CODE"),

  list(name = "MIMIC_MNAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMIMIC_MNAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMIMIC_MNAR_KNN.csv", 
       y_col = "ICD9_CODE"),

  # --- MI MCAR ---
  list(name = "MI_MCAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMI_MCAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MCAR_MEAN.csv", 
       y_col = "ZSN"),
       
  list(name = "MI_MCAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMI_MCAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMI_MCAR_MICE_%d.csv", 
       y_col = "ZSN"),

  list(name = "MI_MCAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMI_MCAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMI_MCAR_missForest.csv", 
       y_col = "ZSN"),

  list(name = "MI_MCAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMI_MCAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MCAR_KNN.csv", 
       y_col = "ZSN"),

  # --- MI MAR ---
  list(name = "MI_MAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMI_MAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MAR_MEAN.csv", 
       y_col = "ZSN"),
       
  list(name = "MI_MAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMI_MAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMI_MAR_MICE_%d.csv", 
       y_col = "ZSN"),

  list(name = "MI_MAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMI_MAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMI_MAR_missForest.csv", 
       y_col = "ZSN"),

  list(name = "MI_MAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMI_MAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MAR_KNN.csv", 
       y_col = "ZSN"),

  # --- MI MNAR ---
  list(name = "MI_MNAR_MEAN", method = "MEAN", 
       train_pattern = "Data/%d_90train_dataMI_MNAR_MEAN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MNAR_MEAN.csv", 
       y_col = "ZSN"),
       
  list(name = "MI_MNAR_MICE", method = "MICE", 
       train_pattern = "Data/%d_90train_dataMI_MNAR_MICE_%d.csv", 
       test_pattern = "Data/%d_10test_dataMI_MNAR_MICE_%d.csv", 
       y_col = "ZSN"),

  list(name = "MI_MNAR_missForest", method = "missForest", 
       train_pattern = "Data/%d_90train_dataMI_MNAR_missForest.csv", 
       test_pattern = "Data/%d_10test_dataMI_MNAR_missForest.csv", 
       y_col = "ZSN"),

  list(name = "MI_MNAR_KNN", method = "KNN", 
       train_pattern = "Data/%d_90train_dataMI_MNAR_KNN.csv", 
       test_pattern = "Data/%d_10test_dataMI_MNAR_KNN.csv", 
       y_col = "ZSN")
)

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

get_data <- function(filename, use_mi) {
  if (!file.exists(filename)) return(NULL)
  data <- read.csv(filename)
  
  if (!use_mi) {
    # Remove missing indicators if noMI
    data <- data[, !grepl("_missing|total_missing", names(data))]
  }
  return(data)
}

run_bas_fold <- function(train_data, test_data, y_col, iter) {
  
  # Ensure Y is in data
  if (!y_col %in% names(train_data)) return(NULL)
  
  # BAS Model
  formula_str <- paste(y_col, "~ .")
  model <- tryCatch({
    bas.glm(as.formula(formula_str), 
            data = train_data,
            method = "MCMC", 
            MCMC.iterations = iter,
            betaprior = robust(151), 
            family = binomial(link = "logit"),
            modelprior = beta.binomial(1, 1))
  }, error = function(e) return(NULL))
  
  if (is.null(model)) return(NULL)
  
  # Extract Coefficients (BMA)
  coefs <- coef(model)
  beta_hat <- coefs$postmean
  
  # Extract Selected Variables (Top 20 by probne0)
  probs <- coefs$probne0
  # Remove Intercept
  probs <- probs[names(probs) != "Intercept"]
  sorted_vars <- names(sort(probs, decreasing = TRUE))
  top_n_vars <- sorted_vars[1:min(length(sorted_vars), TOP_NUM)]
  
  # Evaluate on Test Data (Log Probs + Predictions)
  # We need predictions for the specific top variables to match previous logic?
  # Or BMA predictions?
  # Evaluate_predictions wants "Best Model" based on avg_log_prob.
  # Let's calculate log_probs for Top 1..K variables.
  
  log_probs <- numeric(length(top_n_vars))
  for (k in 1:length(top_n_vars)) {
    current_vars <- top_n_vars[1:k]
    # Simple fitting on train set with these vars to get predictions on test?
    # Or use BAS BMA again?
    # Using simple GLM for speed and standard evaluation?
    # The original script reused bas.glm. Let's stick to bas.glm for consistency if performant.
    
    formula_sub <- as.formula(paste(y_col, "~", paste(current_vars, collapse = " + ")))
    model_sub <- bas.glm(formula_sub, 
                         data = train_data,
                         method = "MCMC", 
                         MCMC.iterations = iter, 
                         betaprior = robust(17), 
                         family = binomial(link = "logit"),
                         modelprior = beta.binomial(1, 1))
    
    preds <- predict(model_sub, newdata = test_data, type = "response")
    fit_probs <- preds$fit
    
    actual_y <- test_data[[y_col]]
    true_probs <- ifelse(actual_y == 1, fit_probs, 1 - fit_probs)
    true_probs[true_probs < 1e-10] <- 1e-10
    log_probs[k] <- mean(log(true_probs))
  }
  
  # Prediction for BEST model (Top 4 is standard truth, or best log prob)
  # We will save ALL predictions for the "Best by LogProb".
  best_k <- which.max(log_probs)
  best_vars <- top_n_vars[1:best_k]
  
  # Rerun prediction for best K to get raw probs for output
  formula_best <- as.formula(paste(y_col, "~", paste(best_vars, collapse = " + ")))
  model_best <- bas.glm(formula_best, data = train_data, method="MCMC", MCMC.iterations=iter, betaprior=robust(17), family=binomial(link="logit"), modelprior=beta.binomial(1,1))
  pred_best <- predict(model_best, newdata = test_data, type = "response")$fit
  
  return(list(
    beta_hat = beta_hat,
    top_vars = top_n_vars,
    log_probs = log_probs,
    preds_best = pred_best,
    best_k = best_k,
    true_y = test_data[[y_col]]
  ))
}

# ============================================================================
# 3. MAIN ANALYSIS LOOP
# ============================================================================

run_analysis <- function() {
  
  cat("STARTING BAS.GLM ANALYSIS (Results/)\n")
  
  for (cfg in configs) {
    for (use_mi in c(FALSE, TRUE)) {
      mi_tag <- if (use_mi) "wMI" else "noMI"
      full_name <- paste0(cfg$name, "_", mi_tag)
      
      cat(sprintf("\n--- Processing %s ---\n", full_name))
      
      m_list <- if (cfg$method == "MICE") M_VALUES else c(1)
      
      # Storage
      all_betas <- list()
      all_selected <- list()
      all_preds <- list()
      all_logprobs <- list()
      
      cnt_beta <- 1
      cnt_sel <- 1
      cnt_pred <- 1
      cnt_log <- 1
      
      for (m in m_list) {
         # Handle MICE suffix
         suffix <- if (cfg$method == "MICE") paste0("_m", m) else "_m1"
         
         if (cfg$method == "MICE") files_indices <- 1:m else files_indices <- c(1)
         
         for (fold in 1:NUM_OF_FOLDS) {
           cat(sprintf("    Fold %d (m=%d)...", fold, m))
           
           for (imp_idx in seq_along(files_indices)) {
             idx <- files_indices[imp_idx]
             
             # Construct file paths
             if (cfg$method == "MICE") {
                 train_file <- sprintf(cfg$train_pattern, fold, idx)
                 test_file <- sprintf(cfg$test_pattern, fold, idx)
             } else {
                 train_file <- sprintf(cfg$train_pattern, fold)
                 test_file <- sprintf(cfg$test_pattern, fold)
             }
             
             train_df <- get_data(train_file, use_mi)
             test_df <- get_data(test_file, use_mi)
             
             if (!is.null(train_df) && !is.null(test_df)) {
                
                res <- run_bas_fold(train_df, test_df, cfg$y_col, ITER)
                
                if (!is.null(res)) {
                  # 1. Betas
                  betas <- res$beta_hat
                  # Exclude intercept from clean list usually
                  betas <- betas[names(betas) != "Intercept"]
                  
                  if (length(betas) > 0) {
                     all_betas[[cnt_beta]] <- data.frame(
                       fold = fold,
                       imp_idx = imp_idx,
                       variable = names(betas),
                       beta_hat = as.numeric(betas)
                     )
                     cnt_beta <- cnt_beta + 1
                  }
                  
                  # 2. Selected Vars
                  all_selected[[cnt_sel]] <- data.frame(
                      fold = fold,
                      imp_idx = imp_idx,
                      variable = res$top_vars,
                      rank = 1:length(res$top_vars)
                  )
                  cnt_sel <- cnt_sel + 1
                  
                  # 3. Log Probs
                  all_logprobs[[cnt_log]] <- data.frame(
                      fold = fold,
                      imp_idx = imp_idx,
                      num_top = 1:length(res$log_probs),
                      log_prob = res$log_probs
                  )
                  cnt_log <- cnt_log + 1
                  
                  # 4. Predictions
                  all_preds[[cnt_pred]] <- data.frame(
                      fold = fold,
                      imp_idx = imp_idx,
                      true_label = res$true_y,
                      predicted_prob = res$preds_best,
                      num_top = res$best_k
                  )
                  cnt_pred <- cnt_pred + 1
                  
                  cat(" Done\n")
                } else {
                  cat(" Failed\n")
                }
             } else {
               cat(" Missing files\n")
             }
           }
         }
         
         # --- SAVE POOLED RESULTS PER M ---
         if (cfg$method == "MICE") {
             # For MICE we might want to pool over m now or later.
             # The evaluation scripts expect ONE file per condition.
             # "POOLED" implicitly means combined.
             # Logic implies we should combine all folds/imps into one file, 
             # and let evaluation script handle pooling (group by fold).
             # Actually evaluation scripts check for "POOLED" in filename for MICE.
         }
       }
       
       # --- WRITE FILES ---
       # 1. Beta Estimates
       if (length(all_betas) > 0) {
           df_beta <- do.call(rbind, all_betas)
           # If MICE, we need to Average Beta across imps for the same Fold?
           # Or evaluation script does it?
           # calculate_bias_all_folds expects "beta_pooled" column for MICE.
           if (cfg$method == "MICE") {
               df_beta <- df_beta %>% 
                   group_by(fold, variable) %>% 
                   summarise(beta_pooled = mean(beta_hat), .groups="drop")
               fname <- paste0("Results/", full_name, "_POOLED_beta_estimates.csv")
           } else {
               fname <- paste0("Results/", full_name, "_beta_estimates.csv")
           }
           write.csv(df_beta, fname, row.names=FALSE)
       }
       
       # 2. Selected Variables
       if (length(all_selected) > 0) {
           df_sel <- do.call(rbind, all_selected)
           # For MICE, pooling selection is complex. Evaluation script expects simple CSV.
           # Usually we vote or take union.
           # Or just append all and let evaluation filter?
           # calculate_selection_all_folds: read.csv -> filter(fold == f) -> pull(variable)
           # If MICE has multiple imps per fold, this will get duplicate variables!
           # We should probably take variables present in >50% imps or just union.
           # Let's simple Union for now (Frequency based pooling in BAS is implicit).
           if (cfg$method == "MICE") {
               df_sel <- df_sel %>% distinct(fold, variable) # Unique vars per fold
               fname <- paste0("Results/", full_name, "_POOLED_selected_variables.csv")
           } else {
               fname <- paste0("Results/", full_name, "_selected_variables.csv")
           }
           write.csv(df_sel, fname, row.names=FALSE)
       }
       
       # 3. Log Probs
       if (length(all_logprobs) > 0) {
           df_log <- do.call(rbind, all_logprobs)
           # If MICE, Average Log Prob across Imps
           if (cfg$method == "MICE") {
               df_log <- df_log %>% group_by(num_top) %>% # Averaging over folds? No, keep folds?
                   # Wait, evaluate_predictions: which.max(log_data$log_prob_pooled)
                   # It selects GLOBAL best num_top. So we average over all folds/imps.
                   summarise(log_prob_pooled = mean(log_prob), .groups="drop")
               fname <- paste0("Results/", full_name, "_POOLED_log_probabilities.csv")
           } else {
               # Single method: Average over folds to get global stability?
               df_log <- df_log %>% group_by(num_top) %>% summarise(avg_log_prob = mean(log_prob), .groups="drop")
               fname <- paste0("Results/", full_name, "_log_probabilities.csv")
           }
           write.csv(df_log, fname, row.names=FALSE)
       }
       
       # 4. Predictions
       if (length(all_preds) > 0) {
           df_pred <- do.call(rbind, all_preds)
           # If MICE, Average Probabilities
           if (cfg$method == "MICE") {
               df_pred <- df_pred %>% group_by(fold, true_label, num_top) %>% # Assuming rows align
                   summarise(predicted_prob_pooled = mean(predicted_prob), .groups="drop")
               # Note: this loses row-level alignment if test sets differ (they shouldn't for same fold).
               fname <- paste0("Results/", full_name, "_POOLED_predictions.csv")
           } else {
               fname <- paste0("Results/", full_name, "_predictions.csv")
           }
           write.csv(df_pred, fname, row.names=FALSE)
       }
    }
  }
}

run_analysis()
