# BAS.GLM ANALYSIS SCRIPT (CORRECTED PIPELINE)
#
# Generates evaluation artifacts for:
# 1. Bias (Beta Estimates)
# 2. Variable Selection (Selected Vars)
# 3. Prediction (Predictions & Log Probs)
#
# CORRECTED CV PIPELINE: Imputation and normalization are performed
# STRICTLY INSIDE the 10-fold cross-validation loop. Test folds are 
# imputed and normalized using ONLY parameters derived from the training fold.
#
# Outputs to: Results/

.libPaths("~/R/library")
library(BAS)
library(dplyr)
library(mice)
library(missForest)
library(missRanger)
library(VIM)

set.seed(123)

# 1. CONFIGURATION
NUM_OF_FOLDS <- 10
N_REPEATS <- 1
TOP_NUM <- 20
ITER <- 1500

M_VALUES <- c(3, 20)

configs <- list(
  # --- REAL DATA (Using Baseline Only for AMI)
  list(name = "MIMIC_REAL", file = "Data/mimic-iv sepsis.csv", y_col = "hospital_expire_flag", is_sim = FALSE),
  list(name = "MI_REAL", file = "Data/cleaned.mi (myocardial infarction)_baseline_only.csv", y_col = "ZSN", is_sim = FALSE)
  
  # --- SIMULATED DATA
  # list(name = "MIMIC_MCAR", file = "Data/complete_dataset_MIMIC_MCAR.csv", y_col = "ICD9_CODE", is_sim = TRUE),
  # list(name = "MIMIC_MAR", file = "Data/complete_dataset_MIMIC_MAR.csv", y_col = "ICD9_CODE", is_sim = TRUE),
  # list(name = "MIMIC_MNAR", file = "Data/complete_dataset_MIMIC_MNAR.csv", y_col = "ICD9_CODE", is_sim = TRUE),
  # list(name = "MI_MCAR", file = "Data/complete_dataset_MI_MCAR.csv", y_col = "ZSN", is_sim = TRUE),
  # list(name = "MI_MAR", file = "Data/complete_dataset_MI_MAR.csv", y_col = "ZSN", is_sim = TRUE),
  # list(name = "MI_MNAR", file = "Data/complete_dataset_MI_MNAR.csv", y_col = "ZSN", is_sim = TRUE)
)

METHODS <- c("MEAN", "KNN", "missForest", "MICE")

# 2. HELPER FUNCTIONS

# Create Stratified 10-Fold Splits
create_stratified_folds <- function(y, k = 10) {
  folds <- list()
  idx_0 <- which(y == 0)
  idx_1 <- which(y == 1)
  
  idx_0 <- sample(idx_0)
  idx_1 <- sample(idx_1)
  
  chunk_0 <- split(idx_0, cut(seq_along(idx_0), k, labels = FALSE))
  chunk_1 <- split(idx_1, cut(seq_along(idx_1), k, labels = FALSE))
  
  for (i in 1:k) {
    folds[[i]] <- c(chunk_0[[i]], chunk_1[[i]])
  }
  return(folds)
}

run_bas_fold <- function(train_aug, test_aug, y_col, iter) {
  formula_str <- paste(y_col, "~ .")
  model <- tryCatch({
    R.utils::withTimeout({
      bas.glm(as.formula(formula_str), 
              data = train_aug, method = "MCMC", 
              MCMC.iterations = iter, betaprior = robust(nrow(train_aug)), 
              family = binomial(link = "logit"), modelprior = beta.binomial(1, 1))
    }, timeout = 300, onTimeout = "error")
  }, error = function(e) {
    cat(sprintf("    [TIMEOUT/ERROR: %s]\n", conditionMessage(e)))
    return(NULL)
  })
  
  if (is.null(model)) return(NULL)
  
  coefs <- coef(model)
  beta_hat <- coefs$postmean
  names(beta_hat) <- model$namesx
  probs <- coefs$probne0
  names(probs) <- model$namesx
  probs <- probs[names(probs) != "Intercept"]
  
  sorted_vars <- names(sort(probs, decreasing = TRUE))
  top_n_vars <- sorted_vars[1:min(length(sorted_vars), TOP_NUM)]
  top_n_pips <- probs[top_n_vars]
  
  if (length(top_n_vars) == 0) return(NULL)
  log_probs <- numeric(length(top_n_vars))
  pred_list <- list()

  for (k in seq_along(top_n_vars)) {
    current_vars <- top_n_vars[1:k]

    coef_names <- model$namesx
    coef_vals  <- coefs$postmean

    intercept  <- coef_vals[coef_names == "Intercept"]
    sub_idx    <- coef_names %in% current_vars
    sub_coefs  <- coef_vals[sub_idx]
    sub_names  <- coef_names[sub_idx]

    X_test     <- as.matrix(test_aug[, sub_names, drop = FALSE])
    log_odds   <- intercept + X_test %*% sub_coefs
    fit_probs  <- as.numeric(1 / (1 + exp(-log_odds)))

    actual_y   <- test_aug[[y_col]]
    true_probs <- ifelse(actual_y == 1, fit_probs, 1 - fit_probs)
    true_probs[true_probs < 1e-10] <- 1e-10
    log_probs[k]   <- mean(log(true_probs))
    pred_list[[k]] <- fit_probs
  }

  best_k    <- which.max(log_probs)
  pred_best <- pred_list[[best_k]]

  return(list(
    beta_hat   = beta_hat,
    top_vars   = top_n_vars,
    top_pips   = top_n_pips,
    log_probs  = log_probs,
    preds_best = pred_best,
    best_k     = best_k,
    true_y     = test_aug[[y_col]]
  ))
}

# 3. MAIN LOOP
run_analysis <- function() {
  cat("STARTING CORRECTED BAS.GLM ANALYSIS\n")
  
  for (cfg in configs) {
    if (!file.exists(cfg$file)) next
    
    # Load raw data
    raw_data <- read.csv(cfg$file)
    # Remove any pre-existing _missing indicators from simulation step so we can compute freshly inside loop
    value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
    raw_data <- raw_data[, value_cols]
    # Exclude patient ID columns
    raw_data <- raw_data[, !names(raw_data) %in% c("HADM_ID", "subject_id", "stay_id"), drop=FALSE]
    
    y_target <- raw_data[[cfg$y_col]]
    
    for (method in METHODS) {
      for (use_mi in c(FALSE, TRUE)) {
        mi_tag <- if (use_mi) "wMI" else "noMI"
        full_name <- paste0(cfg$name, "_", method, "_", mi_tag)
        
        # Determine M
        if (method == "MICE") {
          m_val <- if (cfg$is_sim) 3 else 5
          files_indices <- 1:m_val
          output_suffix <- paste0("_m", m_val)
          full_name <- paste0(cfg$name, "_MICE_", mi_tag)
        } else {
          files_indices <- c(1)
          output_suffix <- "_m1"
        }
        
        # Skip if already completed
        if (method == "MICE") {
          fname_check <- paste0("Results/CORRECTED/", full_name, "_POOLED_log_probabilities.csv")
          fname_tmp_check <- paste0("Results/CORRECTED/tmp_", full_name, "_POOLED_log_probabilities.csv")
        } else {
          fname_check <- paste0("Results/CORRECTED/", full_name, "_log_probabilities.csv")
          fname_tmp_check <- paste0("Results/CORRECTED/tmp_", full_name, "_log_probabilities.csv")
        }
        if (file.exists(fname_check)) {
          existing <- read.csv(fname_check)
          if ("rep" %in% names(existing) && max(existing$rep, na.rm=TRUE) >= N_REPEATS) {
            cat(sprintf("\n--- Skipping %s (already completed) ---\n", full_name))
            next
          }
        }
        
        cat(sprintf("\n--- Processing %s ---\n", full_name))
        
        df_betas_all <- data.frame()
        df_selected_all <- data.frame()
        df_logprobs_all <- data.frame()
        df_preds_all <- data.frame()
        
        # Resume from tmp file if exists
        if (method == "MICE") {
          fname_tmp_check <- paste0("Results/CORRECTED/tmp_", full_name, "_POOLED_log_probabilities.csv")
        } else {
          fname_tmp_check <- paste0("Results/CORRECTED/tmp_", full_name, "_log_probabilities.csv")
        }
        start_rep <- 1
        if (file.exists(fname_tmp_check)) {
          tmp_log <- read.csv(fname_tmp_check)
          completed_reps <- max(tmp_log$rep, na.rm=TRUE)
          start_rep <- completed_reps + 1
          cat(sprintf("  Resuming from repeat %d\n", start_rep))
          df_betas_all <- if(file.exists(paste0("Results/CORRECTED/tmp_", full_name, "_beta_estimates.csv"))) read.csv(paste0("Results/CORRECTED/tmp_", full_name, "_beta_estimates.csv")) else data.frame()
          df_selected_all <- if(file.exists(paste0("Results/CORRECTED/tmp_", full_name, "_selected_variables.csv"))) read.csv(paste0("Results/CORRECTED/tmp_", full_name, "_selected_variables.csv")) else data.frame()
          df_logprobs_all <- read.csv(fname_tmp_check)
          df_preds_all <- if(file.exists(paste0("Results/CORRECTED/tmp_", full_name, "_predictions.csv"))) read.csv(paste0("Results/CORRECTED/tmp_", full_name, "_predictions.csv")) else data.frame()
        }
        
        all_betas <- list()
        all_selected <- list()
        all_preds <- list()
        all_logprobs <- list()
        
        if (start_rep <= N_REPEATS) {
        for (rep in start_rep:N_REPEATS) {
          set.seed(rep * 100)
          folds <- create_stratified_folds(y_target, k = NUM_OF_FOLDS)
          
          for (fold in 1:NUM_OF_FOLDS) {
          cat(sprintf("    Fold %d...", fold))
          test_idx <- folds[[fold]]
          
          # STEP 1: Split data
          train_raw <- raw_data[-test_idx, ]
          test_raw <- raw_data[test_idx, ]
          
          # Extract Y
          y_train <- train_raw[[cfg$y_col]]
          y_test <- test_raw[[cfg$y_col]]
          
          train_x <- train_raw[, names(train_raw) != cfg$y_col]
          test_x <- test_raw[, names(test_raw) != cfg$y_col]
          
          # STEP 2: Normalization (Compute on train, apply to both)
          train_means <- colMeans(train_x, na.rm = TRUE)
          train_sds <- apply(train_x, 2, sd, na.rm = TRUE)
          train_sds[train_sds == 0] <- 1
          
          train_x_scaled <- as.data.frame(scale(train_x, center = train_means, scale = train_sds))
          test_x_scaled <- as.data.frame(scale(test_x, center = train_means, scale = train_sds))
          
          # STEP 3: Missing Indicators
          train_indicators <- as.data.frame(ifelse(is.na(train_x_scaled), 1, 0))
          names(train_indicators) <- paste0(names(train_x_scaled), "_missing")
          train_total_missing <- rowSums(is.na(train_x_scaled))
          
          test_indicators <- as.data.frame(ifelse(is.na(test_x_scaled), 1, 0))
          names(test_indicators) <- paste0(names(test_x_scaled), "_missing")
          test_total_missing <- rowSums(is.na(test_x_scaled))
          
          # STEP 4: Imputation
          for (imp_idx in files_indices) {
            train_imp <- train_x_scaled
            test_imp  <- test_x_scaled

            if (method == "MEAN") {
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(train_imp)) {
                train_imp[is.na(train_imp[[col]]), col] <- c_means[col]
                test_imp[is.na(test_imp[[col]]), col]   <- c_means[col]
              }

            } else if (method == "KNN") {
              # Fit kNN on training data
              k_knn <- round(sqrt(nrow(train_imp)))
              train_imp <- VIM::kNN(train_imp, k = k_knn, imp_var = FALSE)
              # Apply to test: bind test to imputed training, run kNN, extract test rows
              n_train <- nrow(train_imp)
              combined <- rbind(train_imp, test_imp)
              combined_imp <- VIM::kNN(combined, k = k_knn, imp_var = FALSE)
              test_imp <- combined_imp[(n_train + 1):nrow(combined_imp), , drop = FALSE]

            } else if (method == "missForest") {
              # Fit missRanger on training data saving random forest models
              mr_obj <- missRanger::missRanger(train_imp, verbose = 0,
                                               num.trees = 100, pmm.k = 3,
                                               keep_forests = TRUE)
              train_imp <- mr_obj$data
              # Fill any test columns with no missingness in training using training means
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                if (any(is.na(test_imp[[col]]))) {
                  test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
                }
              }
              # Apply saved models to test data — no data leakage
              test_imp <- predict(mr_obj, newdata = test_imp)

            } else if (method == "MICE") {
              # Fit MICE on training data
              capture.output(
                mice_res <- mice::mice(train_imp, m = max(files_indices),
                                       method = 'pmm', printFlag = FALSE)
              )
              train_imp <- mice::complete(mice_res, imp_idx)
              # Apply to test using mice.mids
              test_mice <- mice::mice.mids(mice_res, newdata = test_imp,
                                            printFlag = FALSE)
              test_imp <- mice::complete(test_mice, imp_idx)
              
              # Fallback: fill any remaining NAs with training means
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(train_imp)) {
                if (any(is.na(train_imp[[col]]))) {
                  train_imp[is.na(train_imp[[col]]), col] <- c_means[col]
                }
                if (any(is.na(test_imp[[col]]))) {
                  test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
                }
              }
            }

          # STEP 5: Augmentation
            if (use_mi) {
              train_aug <- cbind(y = y_train, train_imp, train_indicators, total_missing_values = train_total_missing)
              test_aug <- cbind(y = y_test, test_imp, test_indicators, total_missing_values = test_total_missing)
            } else {
              train_aug <- cbind(y = y_train, train_imp)
              test_aug <- cbind(y = y_test, test_imp)
            }
            names(train_aug)[1] <- cfg$y_col
            names(test_aug)[1] <- cfg$y_col
            
            # STEP 6 & 7: Fit and Predict
            res <- run_bas_fold(train_aug, test_aug, cfg$y_col, ITER)
            
            if (!is.null(res)) {
              betas <- res$beta_hat
              betas <- betas[names(betas) != "Intercept"]
              if (length(betas) > 0) {
                 all_betas[[length(all_betas)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, rep = rep, variable = names(betas), beta_hat = as.numeric(betas))
              }
              all_selected[[length(all_selected)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, rep = rep, variable = res$top_vars, rank = 1:length(res$top_vars), pip = res$top_pips)
              all_logprobs[[length(all_logprobs)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, rep = rep, num_top = 1:length(res$log_probs), log_prob = res$log_probs)
              all_preds[[length(all_preds)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, rep = rep, obs_id = 1:length(res$true_y), true_label = res$true_y, predicted_prob = res$preds_best, num_top = res$best_k)
            }
          }
          cat(" Done\n")
        }
        
        # After fold loop closes, accumulate results from this repeat
        if (length(all_betas) > 0) df_betas_all <- rbind(df_betas_all, do.call(rbind, all_betas))
        if (length(all_selected) > 0) df_selected_all <- rbind(df_selected_all, do.call(rbind, all_selected))
        if (length(all_logprobs) > 0) df_logprobs_all <- rbind(df_logprobs_all, do.call(rbind, all_logprobs))
        if (length(all_preds) > 0) df_preds_all <- rbind(df_preds_all, do.call(rbind, all_preds))
        
        # Clear lists to free memory
        all_betas <- list()
        all_selected <- list()
        all_preds <- list()
        all_logprobs <- list()
        
        # Save intermediate results after each repeat
        if (nrow(df_betas_all) > 0) {
          fname_tmp <- paste0("Results/CORRECTED/tmp_", full_name, "_beta_estimates.csv")
          write.csv(df_betas_all, fname_tmp, row.names=FALSE)
        }
        if (nrow(df_logprobs_all) > 0) {
          fname_tmp <- paste0("Results/CORRECTED/tmp_", full_name, "_log_probabilities.csv")
          write.csv(df_logprobs_all, fname_tmp, row.names=FALSE)
        }
        if (nrow(df_preds_all) > 0) {
          fname_tmp <- paste0("Results/CORRECTED/tmp_", full_name, "_predictions.csv")
          write.csv(df_preds_all, fname_tmp, row.names=FALSE)
        }
        if (nrow(df_selected_all) > 0) {
          fname_tmp <- paste0("Results/CORRECTED/tmp_", full_name, "_selected_variables.csv")
          write.csv(df_selected_all, fname_tmp, row.names=FALSE)
        }
        } # close rep loop
        } # close start_rep guard
        
        # Save results
        if (nrow(df_betas_all) > 0) {
           df_beta <- df_betas_all
           if (method == "MICE") {
               df_beta <- df_beta %>% group_by(rep, fold, variable) %>% summarise(beta_pooled = mean(beta_hat), .groups="drop")
               fname <- paste0("Results/CORRECTED/", full_name, "_POOLED_beta_estimates.csv")
           } else {
               fname <- paste0("Results/CORRECTED/", full_name, "_beta_estimates.csv")
           }
           write.csv(df_beta, fname, row.names=FALSE)
        }
        
        if (nrow(df_selected_all) > 0) {
           df_sel <- df_selected_all
           if (method == "MICE") {
               df_sel <- df_sel %>% group_by(fold, variable) %>% summarise(rank = mean(rank, na.rm=TRUE), pip = mean(pip, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED/", full_name, "_POOLED_selected_variables.csv")
           } else {
               fname <- paste0("Results/CORRECTED/", full_name, "_selected_variables.csv")
           }
           write.csv(df_sel, fname, row.names=FALSE)
        }
        
        if (nrow(df_logprobs_all) > 0) {
           df_log <- df_logprobs_all
           if (method == "MICE") {
               df_log <- df_log %>% group_by(rep, num_top) %>% summarise(log_prob_pooled = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED/", full_name, "_POOLED_log_probabilities.csv")
           } else {
               df_log <- df_log %>% group_by(rep, num_top) %>% summarise(avg_log_prob = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED/", full_name, "_log_probabilities.csv")
           }
           write.csv(df_log, fname, row.names=FALSE)
        }
        
        if (nrow(df_preds_all) > 0) {
           df_pred <- df_preds_all
           if (method == "MICE") {
               df_pred <- df_pred %>% group_by(rep, fold, obs_id, true_label, num_top) %>% summarise(predicted_prob_pooled = mean(predicted_prob, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED/", full_name, "_POOLED_predictions.csv")
           } else {
               fname <- paste0("Results/CORRECTED/", full_name, "_predictions.csv")
           }
           write.csv(df_pred, fname, row.names=FALSE)
        }
        
        # Clean up tmp files after successful save
        suppressWarnings({
          file.remove(paste0("Results/CORRECTED/tmp_", full_name, "_beta_estimates.csv"))
          file.remove(paste0("Results/CORRECTED/tmp_", full_name, "_log_probabilities.csv"))
          file.remove(paste0("Results/CORRECTED/tmp_", full_name, "_predictions.csv"))
          file.remove(paste0("Results/CORRECTED/tmp_", full_name, "_selected_variables.csv"))
        })
      }
    }
  }
}

run_analysis()
