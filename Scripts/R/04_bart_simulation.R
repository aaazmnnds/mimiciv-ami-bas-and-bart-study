# BART ANALYSIS SCRIPT (CORRECTED PIPELINE)
#
# Generates evaluation artifacts for:
# 1. Variable Selection (Selected Vars)
# 2. Prediction (Predictions & Log Probs)
#
# CORRECTED CV PIPELINE: Imputation and normalization are performed
# STRICTLY INSIDE the 10-fold cross-validation loop. Test folds are 
# imputed and normalized using ONLY parameters derived from the training fold.
#
# Outputs to: Results/

library(BART)
library(dplyr)
library(mice)
library(missForest)
library(VIM)

set.seed(123)

# 1. CONFIGURATION
NUM_OF_FOLDS <- 10
TOP_NUM <- 20

M_VALUES <- c(3, 20)

configs <- list(
  # --- REAL DATA (Using Baseline Only for AMI)
  # list(name = "MIMIC_REAL", file = "Data/cleaned.mi (mimiciii).csv", y_col = "ICD9_CODE", is_sim = FALSE),
  # list(name = "MI_REAL", file = "Data/cleaned.mi (myocardial infarction)_baseline_only.csv", y_col = "ZSN", is_sim = FALSE),
  
  # --- SIMULATED DATA
  list(name = "MIMIC_MCAR", file = "Data/complete_dataset_MIMIC_MCAR.csv", y_col = "ICD9_CODE", is_sim = TRUE),
  list(name = "MIMIC_MAR", file = "Data/complete_dataset_MIMIC_MAR.csv", y_col = "ICD9_CODE", is_sim = TRUE),
  list(name = "MIMIC_MNAR", file = "Data/complete_dataset_MIMIC_MNAR.csv", y_col = "ICD9_CODE", is_sim = TRUE)
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

get_bart_importance <- function(data, y_col) {
  target_idx <- which(names(data) == y_col)
  x.train <- as.matrix(data[, -target_idx])
  y.train <- data[[y_col]]
  
  bart_model <- tryCatch({
    R.utils::withTimeout({
      lbart(x.train = x.train, y.train = y.train, sparse = FALSE,
            ntree = 100L, ndpost = 500L, nskip = 100L, 
            printevery = 10000L, transposed = FALSE)
    }, timeout = 300, onTimeout = "error")
  }, error = function(e) {
    cat(sprintf("    [TIMEOUT/ERROR in importance fit: %s]\n", conditionMessage(e)))
    return(NULL)
  })
  if (is.null(bart_model)) return(NULL)
  
  varcount <- bart_model$varcount
  row_sums <- rowSums(varcount)
  normalized_matrix <- sweep(varcount, 1, row_sums, FUN = "/")
  normalized_sums <- colSums(normalized_matrix)
  normalized_sums <- normalized_sums / sum(normalized_sums)
  
  sorted_vars <- sort(normalized_sums, decreasing = TRUE)
  return(names(sorted_vars))
}

evaluate_top_vars_bart <- function(train_data, test_data, y_col, top_vars) {
  log_probs <- numeric(length(top_vars))
  all_k_preds <- list()
  
  for (k in 1:length(top_vars)) {
    current_vars <- top_vars[1:k]
    
    x.train <- as.matrix(train_data[, current_vars, drop=FALSE])
    y.train <- train_data[[y_col]]
    
    x.test <- as.matrix(test_data[, current_vars, drop=FALSE])
    y.test <- test_data[[y_col]]
    
    bart_mod <- tryCatch({
      R.utils::withTimeout({
        lbart(x.train = x.train, y.train = y.train, x.test = x.test,
              ntree = 100L, ndpost = 100L, nskip = 50L, printevery = 10000L)
      }, timeout = 120, onTimeout = "error")
    }, error = function(e) return(NULL))
    
    if (is.null(bart_mod)) {
      log_probs[k] <- NA
      all_k_preds[[k]] <- rep(NA, nrow(test_data))
      next
    }
    
    preds <- bart_mod$prob.test.mean
    all_k_preds[[k]] <- preds
    
    true_probs <- ifelse(y.test == 1, preds, 1 - preds)
    true_probs[true_probs < 1e-10] <- 1e-10
    
    log_probs[k] <- mean(log(true_probs))
  }
  
  best_k <- which.max(log_probs)
  if (length(best_k) == 0 || is.na(best_k)) {
    preds_best <- rep(NA, nrow(test_data))
    best_k <- NA
  } else {
    best_vars <- top_vars[1:best_k]
    x.train_best <- as.matrix(train_data[, best_vars, drop=FALSE])
    x.test_best <- as.matrix(test_data[, best_vars, drop=FALSE])
    y.train_best <- train_data[[y_col]]
    
    bart_mod_best <- tryCatch({
      R.utils::withTimeout({
        lbart(x.train = x.train_best, y.train = y.train_best, x.test = x.test_best,
              ntree = 100L, ndpost = 500L, nskip = 100L, printevery = 10000L)
      }, timeout = 300, onTimeout = "error")
    }, error = function(e) return(NULL))
    
    if (!is.null(bart_mod_best)) {
      preds_best <- bart_mod_best$prob.test.mean
    } else {
      preds_best <- all_k_preds[[best_k]]
    }
  }
  
  return(list(log_probs = log_probs, preds_best = preds_best, best_k = best_k))
}

# 3. MAIN LOOP
run_analysis <- function() {
  cat("STARTING CORRECTED BART ANALYSIS\n")
  
  for (cfg in configs) {
    if (!file.exists(cfg$file)) next
    
    raw_data <- read.csv(cfg$file)
    value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
    raw_data <- raw_data[, value_cols]
    
    y_target <- raw_data[[cfg$y_col]]
    folds <- create_stratified_folds(y_target, k = NUM_OF_FOLDS)
    
    for (method in METHODS) {
      for (use_mi in c(FALSE, TRUE)) {
        mi_tag <- if (use_mi) "wMI" else "noMI"
        full_name <- paste0(cfg$name, "_", method, "_", mi_tag)
        
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
        
        all_logprobs <- list()
        all_selected <- list()
        all_preds <- list()        
        for (fold in 1:NUM_OF_FOLDS) {
          cat(sprintf("    Fold %d...", fold))
          test_idx <- folds[[fold]]
          
          # STEP 1: Split data
          train_raw <- raw_data[-test_idx, ]
          test_raw <- raw_data[test_idx, ]
          
          y_train <- train_raw[[cfg$y_col]]
          y_test <- test_raw[[cfg$y_col]]
          
          train_x <- train_raw[, names(train_raw) != cfg$y_col]
          test_x <- test_raw[, names(test_raw) != cfg$y_col]
          
          # STEP 2: Normalization
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
            test_imp <- test_x_scaled
            
            if (method == "MEAN") {
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(train_imp)) {
                train_imp[is.na(train_imp[[col]]), col] <- c_means[col]
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "KNN") {
              train_knn <- VIM::kNN(train_imp, k=5, imp_var=FALSE)
              train_imp <- train_knn
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "missForest") {
              mf_res <- missForest::missForest(train_imp, verbose = FALSE)
              train_imp <- mf_res$ximp
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
              }
              
            } else if (method == "MICE") {
              capture.output(mice_res <- mice::mice(train_imp, m = max(files_indices), method = 'pmm', printFlag = FALSE))
              train_imp <- mice::complete(mice_res, imp_idx)
              c_means <- colMeans(train_imp, na.rm = TRUE)
              for (col in names(test_imp)) {
                test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
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
            ranked_vars <- get_bart_importance(train_aug, cfg$y_col)
            top_k_vars <- ranked_vars[1:min(length(ranked_vars), TOP_NUM)]
            
            res <- evaluate_top_vars_bart(train_aug, test_aug, cfg$y_col, top_k_vars)
            lps <- res$log_probs
            
            if (length(lps) < TOP_NUM) {
              lps <- c(lps, rep(NA, TOP_NUM - length(lps)))
            }
            
            all_logprobs[[length(all_logprobs)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, num_top = 1:length(lps), log_prob = lps)
            all_selected[[length(all_selected)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, variable = top_k_vars, rank = 1:length(top_k_vars))
            all_preds[[length(all_preds)+1]] <- data.frame(fold = fold, imp_idx = imp_idx, obs_id = 1:length(test_aug[[cfg$y_col]]), true_label = test_aug[[cfg$y_col]], predicted_prob = res$preds_best, num_top = res$best_k)
          }
          cat(" Done\n")
        }
        
        # Save results
        if (length(all_logprobs) > 0) {
           df_log <- do.call(rbind, all_logprobs)
           if (method == "MICE") {
               df_log <- df_log %>% group_by(num_top) %>% summarise(log_prob_pooled = mean(log_prob, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_POOLED_log_probabilities", output_suffix, ".csv")
           } else {
               df_log <- df_log %>% group_by(num_top) %>% summarise(avg_log_prob = mean(log_prob, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_log_probabilities", output_suffix, ".csv")
           }
           write.csv(df_log, fname, row.names=FALSE)
        }
        
        if (length(all_selected) > 0) {
           df_sel <- do.call(rbind, all_selected)
           if (method == "MICE") {
               df_sel <- df_sel %>% distinct(fold, variable)
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_POOLED_selected_variables", output_suffix, ".csv")
           } else {
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_selected_variables", output_suffix, ".csv")
           }
           write.csv(df_sel, fname, row.names=FALSE)
        }
        
        if (length(all_preds) > 0) {
           df_pred <- do.call(rbind, all_preds)
           if (method == "MICE") {
                df_pred <- df_pred %>% group_by(fold, obs_id, true_label, num_top) %>% summarise(predicted_prob_pooled = mean(predicted_prob, na.rm=TRUE), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_POOLED_predictions", output_suffix, ".csv")
           } else {
               fname <- paste0("Results/CORRECTED/SIMULATION/results_BART_", full_name, "_predictions", output_suffix, ".csv")
           }
           write.csv(df_pred, fname, row.names=FALSE)
        }
      }
    }
  }
}

run_analysis()
