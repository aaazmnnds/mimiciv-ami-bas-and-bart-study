# .libPaths("~/R/library")  # Ada only
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
library(missRanger)
library(VIM)
library(missMethods)

set.seed(123)

# 1. CONFIGURATION
NUM_OF_FOLDS <- 5
TOP_NUM <- 20
ITER <- 1000
N_MC_REPS <- 50

M_VALUES <- c(3, 20)
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
  top_n_vars <- sorted_vars[1:min(length(sorted_vars), TOP_NUM)]
  
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
    log_probs[k]  <- mean(log(true_probs))
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

generate_sim_dataset <- function(X_full, y_col, target_prop_ones, mechanism, miss_rate, seed) {
  set.seed(seed)
  p <- ncol(X_full)
  half_p <- floor(p / 2)
  
  # Select 4 true variables with highest missingness after imposing missingness
  if (mechanism == "MCAR") {
    X_miss <- missMethods::delete_MCAR(X_full, p = miss_rate, cols_mis = 1:p)
  } else if (mechanism == "MAR") {
    X_miss <- missMethods::delete_MAR_censoring(X_full, p = miss_rate, 
                cols_mis = (half_p + 1):(2 * half_p), cols_ctrl = 1:half_p)
  } else if (mechanism == "MNAR") {
    X_miss <- missMethods::delete_MNAR_censoring(X_full, p = miss_rate, cols_mis = 1:p)
  }
  
  miss_rates <- colMeans(is.na(X_miss))
  top_idx <- order(miss_rates, decreasing = TRUE)[1:4]
  top_vars <- names(X_miss)[top_idx]
  beta_values <- c(1.5, 1.0, 0.5, 0.1)
  
  # Generate outcome
  X_matrix <- scale(as.matrix(X_full[, top_idx]))
  base_pred <- X_matrix %*% beta_values
  
  intercept_low <- -50; intercept_high <- 50
  for(iter in 1:50) {
    intercept_mid <- (intercept_low + intercept_high) / 2
    prop <- mean(plogis(intercept_mid + base_pred))
    if(abs(prop - target_prop_ones) < 0.001) break
    if(prop < target_prop_ones) intercept_low <- intercept_mid else intercept_high <- intercept_mid
  }
  best_intercept <- (intercept_low + intercept_high) / 2
  y_sim <- rbinom(nrow(X_full), 1, plogis(best_intercept + base_pred))
  
  # Combine outcome with missing data
  df <- data.frame(y_sim, X_miss)
  names(df)[1] <- y_col
  
  return(list(data = df, true_vars = top_vars, true_betas = beta_values))
}

# Load base datasets for simulation
mimic_base <- read.csv("Data/mimic-iv sepsis.csv")
mimic_base <- mimic_base[, !grepl("_missing|total_missing", names(mimic_base))]
mimic_X_full <- mimic_base[, !names(mimic_base) %in% c("hospital_expire_flag", "HADM_ID")]
mimic_outcome_table <- table(mimic_base[["hospital_expire_flag"]])
mimic_target_prop <- as.numeric(mimic_outcome_table["1"]) / sum(mimic_outcome_table)
mimic_miss_rate <- mean(is.na(mimic_X_full))

cat("Imputing MIMIC-IV base dataset for simulation...\n")
if (file.exists("Data/mimic_iv_Xfull_imputed.csv")) {
  mimic_X_full_imp <- read.csv("Data/mimic_iv_Xfull_imputed.csv")
} else {
  mr_full <- missRanger::missRanger(mimic_X_full, verbose = 1, num.trees = 100)
  mimic_X_full_imp <- mr_full
  write.csv(mimic_X_full_imp, "Data/mimic_iv_Xfull_imputed.csv", row.names = FALSE)
}

mi_base <- read.csv("Data/cleaned.mi (myocardial infarction)_baseline_only.csv")
mi_base <- mi_base[, !grepl("_missing|total_missing", names(mi_base))]
mi_X_full <- mi_base[, !names(mi_base) %in% c("ZSN") & !grepl("_missing|total_missing", names(mi_base))]
mi_outcome_table <- table(mi_base[["ZSN"]])
mi_target_prop <- as.numeric(mi_outcome_table["1"]) / sum(mi_outcome_table)
mi_miss_rate <- mean(is.na(mi_X_full))

cat("Imputing AMI base dataset for simulation...\n")
if (file.exists("Data/mi_Xfull_imputed.csv")) {
  mi_X_full_imp <- read.csv("Data/mi_Xfull_imputed.csv")
} else {
  mr_full_mi <- missRanger::missRanger(mi_X_full, verbose = 1, num.trees = 100)
  mi_X_full_imp <- mr_full_mi
  write.csv(mi_X_full_imp, "Data/mi_Xfull_imputed.csv", row.names = FALSE)
}

configs <- list(
  list(name = "MIMIC_MCAR", X_full = mimic_X_full_imp, y_col = "hospital_expire_flag", 
       mechanism = "MCAR", miss_rate = mimic_miss_rate, target_prop = mimic_target_prop, is_sim = TRUE),
  list(name = "MIMIC_MAR", X_full = mimic_X_full_imp, y_col = "hospital_expire_flag", 
       mechanism = "MAR", miss_rate = mimic_miss_rate, target_prop = mimic_target_prop, is_sim = TRUE),
  list(name = "MIMIC_MNAR", X_full = mimic_X_full_imp, y_col = "hospital_expire_flag", 
       mechanism = "MNAR", miss_rate = mimic_miss_rate, target_prop = mimic_target_prop, is_sim = TRUE),
  list(name = "MI_MCAR", X_full = mi_X_full_imp, y_col = "ZSN", 
       mechanism = "MCAR", miss_rate = mi_miss_rate, target_prop = mi_target_prop, is_sim = TRUE),
  list(name = "MI_MAR", X_full = mi_X_full_imp, y_col = "ZSN", 
       mechanism = "MAR", miss_rate = mi_miss_rate, target_prop = mi_target_prop, is_sim = TRUE),
  list(name = "MI_MNAR", X_full = mi_X_full_imp, y_col = "ZSN", 
       mechanism = "MNAR", miss_rate = mi_miss_rate, target_prop = mi_target_prop, is_sim = TRUE)
)

# 3. MAIN LOOP
run_analysis <- function() {
  cat("STARTING CORRECTED BAS.GLM ANALYSIS\n")
  
  for (cfg in configs) {
    cat(sprintf("\nDataset: %s | Mechanism: %s\n", cfg$name, cfg$mechanism))
    
    for (method in METHODS) {
      for (use_mi in c(FALSE, TRUE)) {
        mi_tag <- if (use_mi) "wMI" else "noMI"
        full_name <- paste0(cfg$name, "_", method, "_", mi_tag)
        
        if (method == "MICE") {
          m_val <- 3
          files_indices <- 1:m_val
          full_name <- paste0(cfg$name, "_MICE_", mi_tag)
        } else {
          files_indices <- c(1)
        }
        
        fname_check <- if(method == "MICE") {
          paste0("Results/CORRECTED/SIMULATION/", full_name, "_POOLED_beta_estimates.csv")
        } else {
          paste0("Results/CORRECTED/SIMULATION/", full_name, "_beta_estimates.csv")
        }
        
        start_rep <- 1
        all_betas <- list()
        all_selected <- list()
        all_preds <- list()
        all_logprobs <- list()
        
        if (file.exists(fname_check)) {
          existing <- read.csv(fname_check)
          if ("mc_rep" %in% names(existing)) {
            start_rep <- max(existing$mc_rep, na.rm=TRUE) + 1
            if(start_rep <= N_MC_REPS) {
              all_betas[[1]] <- existing
              
              # Load other existing files to append to
              fname_sel <- sub("beta_estimates", "selected_variables", fname_check)
              if (file.exists(fname_sel)) all_selected[[1]] <- read.csv(fname_sel)
              
              fname_log <- sub("beta_estimates", "log_probabilities", fname_check)
              if (file.exists(fname_log)) all_logprobs[[1]] <- read.csv(fname_log)
              
              fname_pred <- sub("beta_estimates", "predictions", fname_check)
              if (file.exists(fname_pred)) all_preds[[1]] <- read.csv(fname_pred)
            }
          }
        }
        
        if (start_rep > N_MC_REPS) {
          cat(sprintf("--- Skipping %s (already completed) ---\n", full_name))
          next
        }
        
        cat(sprintf("\n--- Processing %s ---\n", full_name))
        
        # Setup parallel backend
        library(doParallel)
        cores_to_use <- min(32, max(1, parallel::detectCores() - 2))
        registerDoParallel(cores=cores_to_use)
        
        results_list <- foreach(mc_rep = start_rep:N_MC_REPS, .packages = c("BAS", "dplyr", "mice", "missForest", "missRanger", "VIM", "missMethods")) %dopar% {
          cat(sprintf("  MC Rep %d/%d\n", mc_rep, N_MC_REPS))
          
          all_betas_iter <- list()
          all_selected_iter <- list()
          all_logprobs_iter <- list()
          all_preds_iter <- list()          
          sim_result <- generate_sim_dataset(
            X_full = cfg$X_full,
            y_col = cfg$y_col,
            target_prop_ones = cfg$target_prop,
            mechanism = cfg$mechanism,
            miss_rate = cfg$miss_rate,
            seed = mc_rep * 1000 + which(sapply(configs, function(x) identical(x$name, cfg$name)))
          )
          raw_data <- sim_result$data
          true_vars <- sim_result$true_vars
          true_vars_str <- paste(true_vars, collapse=",")
          
          value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
          raw_data <- raw_data[, value_cols]
          y_target <- raw_data[[cfg$y_col]]
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
                  test_imp[is.na(test_imp[[col]]), col]   <- c_means[col]
                }
  
              } else if (method == "KNN") {
                k_knn <- round(sqrt(nrow(train_imp)))
                train_imp <- VIM::kNN(train_imp, k = k_knn, imp_var = FALSE)
                n_train <- nrow(train_imp)
                combined <- rbind(train_imp, test_imp)
                combined_imp <- VIM::kNN(combined, k = k_knn, imp_var = FALSE)
                test_imp <- combined_imp[(n_train + 1):nrow(combined_imp), , drop = FALSE]
  
              } else if (method == "missForest") {
                mr_obj <- missRanger::missRanger(train_imp, verbose = 0,
                                                 num.trees = 100, pmm.k = 3,
                                                 keep_forests = TRUE)
                train_imp <- mr_obj$data
                c_means <- colMeans(train_imp, na.rm = TRUE)
                for (col in names(test_imp)) {
                  if (any(is.na(test_imp[[col]]))) {
                    test_imp[is.na(test_imp[[col]]), col] <- c_means[col]
                  }
                }
                test_imp <- predict(mr_obj, newdata = test_imp)
  
              } else if (method == "MICE") {
                capture.output(
                  mice_res <- mice::mice(train_imp, m = max(files_indices),
                                         method = 'pmm', printFlag = FALSE)
                )
                train_imp <- mice::complete(mice_res, imp_idx)
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
                   all_betas_iter[[length(all_betas_iter)+1]] <- data.frame(mc_rep = mc_rep, fold = fold, imp_idx = imp_idx, true_vars = true_vars_str, variable = names(betas), beta_hat = as.numeric(betas))
                }
                all_selected_iter[[length(all_selected_iter)+1]] <- data.frame(mc_rep = mc_rep, fold = fold, imp_idx = imp_idx, true_vars = true_vars_str, variable = res$top_vars, rank = 1:length(res$top_vars))
                all_logprobs_iter[[length(all_logprobs_iter)+1]] <- data.frame(mc_rep = mc_rep, fold = fold, imp_idx = imp_idx, true_vars = true_vars_str, num_top = 1:length(res$log_probs), log_prob = res$log_probs)
                all_preds_iter[[length(all_preds_iter)+1]] <- data.frame(mc_rep = mc_rep, fold = fold, imp_idx = imp_idx, obs_id = 1:length(res$true_y), true_vars = true_vars_str, true_label = res$true_y, predicted_prob = res$preds_best, num_top = res$best_k)
              }
            }
          }
          list(betas = all_betas_iter, selected = all_selected_iter, logprobs = all_logprobs_iter, preds = all_preds_iter)
        }
        
        cat(" Done\n")
        
        # Unpack foreach results
        for(res in results_list) {
          if (length(res$betas) > 0) all_betas <- c(all_betas, res$betas)
          if (length(res$selected) > 0) all_selected <- c(all_selected, res$selected)
          if (length(res$logprobs) > 0) all_logprobs <- c(all_logprobs, res$logprobs)
          if (length(res$preds) > 0) all_preds <- c(all_preds, res$preds)
        }
        
        # Save results at the end of foreach loop
        if (length(all_betas) > 0) {
           df_beta <- do.call(rbind, all_betas)
           if (method == "MICE") {
               df_beta <- df_beta %>% group_by(mc_rep, fold, true_vars, variable) %>% summarise(beta_pooled = mean(beta_hat), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_POOLED_beta_estimates.csv")
           } else {
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_beta_estimates.csv")
           }
           write.csv(df_beta, fname, row.names=FALSE)
        }
        
        if (length(all_selected) > 0) {
           df_sel <- do.call(rbind, all_selected)
           if (method == "MICE") {
               df_sel <- df_sel %>% distinct(mc_rep, fold, true_vars, variable)
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_POOLED_selected_variables.csv")
           } else {
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_selected_variables.csv")
           }
           write.csv(df_sel, fname, row.names=FALSE)
        }
        
        if (length(all_logprobs) > 0) {
           df_log <- do.call(rbind, all_logprobs)
           if (method == "MICE") {
               df_log <- df_log %>% group_by(mc_rep, true_vars, num_top) %>% summarise(log_prob_pooled = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_POOLED_log_probabilities.csv")
           } else {
               df_log <- df_log %>% group_by(mc_rep, true_vars, num_top) %>% summarise(avg_log_prob = mean(log_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_log_probabilities.csv")
           }
           write.csv(df_log, fname, row.names=FALSE)
        }
        
        if (length(all_preds) > 0) {
           df_pred <- do.call(rbind, all_preds)
           if (method == "MICE") {
               df_pred <- df_pred %>% group_by(mc_rep, fold, obs_id, true_vars, true_label, num_top) %>% summarise(predicted_prob_pooled = mean(predicted_prob), .groups="drop")
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_POOLED_predictions.csv")
           } else {
               fname <- paste0("Results/CORRECTED/SIMULATION/", full_name, "_predictions.csv")
           }
           write.csv(df_pred, fname, row.names=FALSE)
        }
      }
    }
  }
  # Parallel backend will close automatically on script exit
}

run_analysis()
