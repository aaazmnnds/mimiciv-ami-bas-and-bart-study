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

library(BAS)
library(dplyr)
library(mice)
library(missForest)
library(VIM)

set.seed(123)

# 1. CONFIGURATION
NUM_OF_FOLDS <- 10
# TOP_NUM removed for all_vars
ITER <- 1500


configs <- list(
  list(name = "MIMIC_REAL", file = "Data/cleaned.mi (mimiciii).csv", y_col = "ICD9_CODE", is_sim = FALSE),
  list(name = "MI_REAL", file = "Data/cleaned.mi (myocardial infarction)_baseline_only.csv", y_col = "ZSN", is_sim = FALSE)
)

METHODS <- c("MEAN", "KNN", "missForest", "MICE", "Z_only")

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
    }, timeout = 120, onTimeout = "error")
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
  top_n_vars <- sorted_vars[1:length(sorted_vars)]
  
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
    log_probs  = log_probs,
    preds_best = pred_best,
    best_k     = best_k,
    true_y     = test_aug[[y_col]]
  ))
}

# 3. MAIN LOOP
run_analysis <- function() {
  cat("STARTING CORRECTED BAS.GLM ANALYSIS\n")
  dir.create("Results/CORRECTED_ALL_VARS/", recursive = TRUE, showWarnings = FALSE)
  
  for (cfg in configs) {
    if (!file.exists(cfg$file)) next
    
    # Load raw data
    raw_data <- read.csv(cfg$file)
    # Remove any pre-existing _missing indicators from simulation step so we can compute freshly inside loop
    value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
    raw_data <- raw_data[, value_cols]
    
    y_target <- raw_data[[cfg$y_col]]
    folds <- create_stratified_folds(y_target, k = NUM_OF_FOLDS)
    
    for (method in METHODS) {
      for (use_mi in c(FALSE, TRUE)) {
        if (method == "Z_only" && !use_mi) next # Z_only implies we use indicators, no need to run without
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
        
        cat(sprintf("\n--- Processing %s ---\n", full_name))
        
        all_betas <- list()
        all_selected <- list()
        all_preds <- list()
        all_logprobs <- list()
        
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
          if (method == "MICE") {
            capture.output(mice_res <- mice::mice(train_x_scaled, m = max(files_indices), method = 'pmm', printFlag = FALSE))
          }
          
          for (imp_idx in files_indices) {
            
            train_imp <- train_x_scaled
            test_imp <- test_x_scaled
            
            if (method == "MEAN") {
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(train_imp)) {
                train_imp[is.na(train_imp[[col]]), col] <- c_means[col]
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "KNN") {
              # kNN on train
              train_knn <- VIM::kNN(train_imp, k=5, imp_var=FALSE)
              train_imp <- train_knn
              # Fallback for test: train column means
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "missForest") {
              mf_res <- missForest::missForest(train_imp, verbose = FALSE)
              train_imp <- mf_res$ximp
              # Fallback for test: train column means
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "MICE") {
              train_imp <- mice::complete(mice_res, imp_idx)
              # Fallback for test: train column means
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
            } else if (method == "Z_only") {
              # No imputation. We just use Z. We'll drop train_imp in Step 5.
            }
            
            # STEP 5: Augmentation
            if (method == "Z_only") {
              train_aug <- cbind(y = y_train, train_indicators)
              test_aug <- cbind(y = y_test, test_indicators)
            } else if (use_mi) {
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
                 all_betas[[length(all_betas)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, variable = names(betas), beta_hat = as.numeric(betas))
              }
              all_selected[[length(all_selected)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, variable = res$top_vars, rank = 1:length(res$top_vars))
              all_logprobs[[length(all_logprobs)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, num_top = 1:length(res$log_probs), log_prob = res$log_probs)
              all_preds[[length(all_preds)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, obs_id = 1:length(res$true_y), true_label = res$true_y, predicted_prob = res$preds_best, num_top = res$best_k)
            }
          }
          cat(" Done\n")
        }
        
        # Save results
        if (length(all_betas) > 0) {
           df_beta <- do.call(rbind, all_betas)
           if (method == "MICE") {
               df_beta <- df_beta %>% group_by(fold, variable) %>% summarise(beta_pooled = mean(beta_hat), .groups="drop")
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_POOLED_beta_estimates.csv")
           } else {
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_beta_estimates.csv")
           }
           write.csv(df_beta, fname, row.names=FALSE)
        }
        
        if (length(all_selected) > 0) {
           df_sel <- do.call(rbind, all_selected)
           if (method == "MICE") {
               df_sel <- df_sel %>% distinct(fold, variable)
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_POOLED_selected_variables.csv")
           } else {
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_selected_variables.csv")
           }
           write.csv(df_sel, fname, row.names=FALSE)
        }
        
        if (length(all_logprobs) > 0) {
           df_log <- do.call(rbind, all_logprobs)
           if (method == "MICE") {
               df_log <- df_log %>% group_by(num_top) %>% summarise(log_prob_pooled = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_POOLED_log_probabilities.csv")
           } else {
               df_log <- df_log %>% group_by(num_top) %>% summarise(avg_log_prob = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_log_probabilities.csv")
           }
           write.csv(df_log, fname, row.names=FALSE)
        }
        
        if (length(all_preds) > 0) {
           df_pred <- do.call(rbind, all_preds)
           if (method == "MICE") {
               df_pred <- df_pred %>% group_by(fold, obs_id, true_label, num_top) %>% summarise(predicted_prob_pooled = mean(predicted_prob, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_POOLED_predictions.csv")
           } else {
               fname <- paste0("Results/CORRECTED_ALL_VARS/", full_name, "_predictions.csv")
           }
           write.csv(df_pred, fname, row.names=FALSE)
        }
      }
    }
  }
}

run_analysis()
