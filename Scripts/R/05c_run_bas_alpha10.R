.libPaths("~/R/library")
# BAS.GLM ANALYSIS SCRIPT (ALPHA10 SENSITIVITY PIPELINE)
#
# Generates evaluation artifacts for:
# 1. Bias (Beta Estimates)
# 2. Variable Selection (Selected Vars)
# 3. Prediction (Predictions & Log Probs)
#
# Runs on ALPHA10 datasets where outcomes are driven by missing indicators
# Imputation and normalization are performed inside the 10-fold CV loop.
#
# Outputs to: Results/CORRECTED/ALPHA10/

library(BAS)
library(dplyr)
library(mice)
library(missForest)
library(VIM)

set.seed(123)

# 1. CONFIGURATION
NUM_OF_FOLDS <- 10
N_REPEATS <- 1
TOP_NUM <- 20
ITER <- 1500

M_VALUES <- c(3)

DATASETS <- list(
  MIMIC = list(prefix = "MIMIC", y_col = "hospital_expire_flag"),
  MI    = list(prefix = "MI",    y_col = "ZSN")
)

MECHANISMS <- c("MCAR", "MAR", "MNAR")
METHODS <- c("MEAN", "KNN", "missForest", "MICE")
MI_CONDITIONS <- c(FALSE, TRUE)

# Build configuration list
configs <- list()
for (d_name in names(DATASETS)) {
  d_cfg <- DATASETS[[d_name]]
  for (mech in MECHANISMS) {
    cfg_name <- paste0(d_cfg$prefix, "_", mech, "_ALPHA10")
    file_path <- paste0("Data/top/complete_dataset_", d_cfg$prefix, "_", mech, "_ALPHA10.csv")
    configs[[length(configs) + 1]] <- list(
      name = cfg_name,
      file = file_path,
      y_col = d_cfg$y_col,
      is_sim = TRUE
    )
  }
}

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
  
  if (length(top_n_vars) == 0) return(NULL)
  log_probs <- numeric(length(top_n_vars))
  pred_list <- list()

  for (k in seq_along(top_n_vars)) {
    current_vars <- top_n_vars[1:k]
    current_formula <- as.formula(paste(y_col, "~", paste(current_vars, collapse = " + ")))
    
    sub_model <- tryCatch({
      glm(current_formula, data = train_aug, family = binomial(link = "logit"))
    }, error = function(e) NULL)
    
    if (is.null(sub_model)) {
      log_probs[k] <- -Inf
      next
    }
    
    preds <- predict(sub_model, newdata = test_aug, type = "response")
    preds <- pmin(pmax(preds, 1e-15), 1 - 1e-15)
    
    y_test <- test_aug[[y_col]]
    log_probs[k] <- mean(y_test * log(preds) + (1 - y_test) * log(1 - preds))
    pred_list[[k]] <- preds
  }
  
  best_k <- which.max(log_probs)
  
  return(list(
    beta_estimates = beta_hat,
    inclusion_probs = probs,
    selected_variables = top_n_vars,
    log_probabilities = log_probs,
    predictions = pred_list[[best_k]],
    best_k = best_k
  ))
}

# 3. MAIN LOOP
run_analysis <- function() {
  cat("STARTING BAS.GLM ALPHA10 SENSITIVITY ANALYSIS\n")
  dir.create("Results/CORRECTED/ALPHA10/", recursive = TRUE, showWarnings = FALSE)
  
  for (cfg in configs) {
    if (!file.exists(cfg$file)) {
      cat(sprintf("Skipping missing file: %s\n", cfg$file))
      next
    }
    
    raw_data <- read.csv(cfg$file)
    value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
    raw_data <- raw_data[, value_cols]
    
    y_target <- raw_data[[cfg$y_col]]
    folds <- create_stratified_folds(y_target, k = NUM_OF_FOLDS)
    
    for (method in METHODS) {
      for (use_mi in MI_CONDITIONS) {
        mi_tag <- if (use_mi) "wMI" else "noMI"
        
        if (method == "MICE") {
          m_val <- 3
          files_indices <- 1:m_val
          output_suffix <- paste0("_m", m_val)
          full_name <- paste0(cfg$name, "_MICE_", mi_tag)
        } else {
          files_indices <- c(1)
          output_suffix <- "_m1"
          full_name <- paste0(cfg$name, "_", method, "_", mi_tag)
        }
        
        # Check skip logic (if predictions file already exists)
        if (method == "MICE") {
          final_file_check <- paste0("Results/CORRECTED/ALPHA10/", full_name, "_POOLED_predictions.csv")
        } else {
          final_file_check <- paste0("Results/CORRECTED/ALPHA10/", full_name, "_predictions.csv")
        }
        
        if (file.exists(final_file_check)) {
          cat(sprintf("Skipping %s (Already completed)\n", full_name))
          next
        }
        
        cat(sprintf("\n--- Processing %s ---\n", full_name))
        
        all_betas <- list()
        all_selected <- list()
        all_preds <- list()
        all_logprobs <- list()
        
        for (fold in 1:NUM_OF_FOLDS) {
          cat(sprintf("    Fold %d...", fold))
          test_idx <- folds[[fold]]
          
          train_raw <- raw_data[-test_idx, ]
          test_raw  <- raw_data[test_idx, ]
          
          # Pure missing indicators from un-imputed sets
          train_Z <- as.data.frame(ifelse(is.na(train_raw[, names(train_raw) != cfg$y_col]), 1, 0))
          test_Z  <- as.data.frame(ifelse(is.na(test_raw[, names(test_raw) != cfg$y_col]), 1, 0))
          names(train_Z) <- paste0(names(train_Z), "_missing")
          names(test_Z)  <- paste0(names(test_Z), "_missing")
          
          # IMPUTATION
          if (method == "MEAN") {
            train_imp_list <- list(train_raw)
            test_imp_list  <- list(test_raw)
            
            for (col in names(train_raw)) {
              if (col == cfg$y_col) next
              if (is.numeric(train_raw[[col]])) {
                mean_val <- mean(train_raw[[col]], na.rm = TRUE)
                train_imp_list[[1]][[col]][is.na(train_imp_list[[1]][[col]])] <- mean_val
                test_imp_list[[1]][[col]][is.na(test_imp_list[[1]][[col]])]   <- mean_val
              }
            }
          } else if (method == "KNN") {
            knn_k <- 5
            combined <- rbind(train_raw, test_raw)
            imp_combined <- VIM::kNN(combined, k = knn_k, imp_var = FALSE)
            train_imp_list <- list(imp_combined[1:nrow(train_raw), ])
            test_imp_list  <- list(imp_combined[(nrow(train_raw) + 1):nrow(combined), ])
          } else if (method == "missForest") {
            combined <- rbind(train_raw, test_raw)
            features_combined <- combined[, names(combined) != cfg$y_col]
            mf_out <- missForest::missForest(features_combined, maxiter = 5, ntree = 20)
            imp_feats <- mf_out$ximp
            imp_all <- cbind(imp_feats, combined[cfg$y_col])
            
            train_imp_list <- list(imp_all[1:nrow(train_raw), ])
            test_imp_list  <- list(imp_all[(nrow(train_raw) + 1):nrow(combined), ])
          } else if (method == "MICE") {
            imp_train <- mice::mice(train_raw, m = m_val, maxit = 5, printFlag = FALSE)
            train_imp_list <- lapply(1:m_val, function(i) mice::complete(imp_train, i))
            
            test_imp_list <- list()
            for (m in 1:m_val) {
              imp_test_m <- test_raw
              for (col in names(test_raw)) {
                if (col == cfg$y_col) next
                if (any(is.na(imp_test_m[[col]]))) {
                  donor_vals <- train_imp_list[[m]][[col]]
                  imp_test_m[[col]][is.na(imp_test_m[[col]])] <- sample(donor_vals[!is.na(donor_vals)], sum(is.na(imp_test_m[[col]])), replace = TRUE)
                }
              }
              test_imp_list[[m]] <- imp_test_m
            }
          }
          
          # RUN BAS ACROSS M DATASETS
          fold_betas <- list()
          fold_selected <- list()
          fold_preds <- list()
          fold_logprobs <- list()
          
          for (m in files_indices) {
            tr_imp <- train_imp_list[[m]]
            ts_imp <- test_imp_list[[m]]
            
            # Normalization
            for (col in names(tr_imp)) {
              if (col == cfg$y_col) next
              if (is.numeric(tr_imp[[col]])) {
                col_sd <- sd(tr_imp[[col]], na.rm = TRUE)
                col_mean <- mean(tr_imp[[col]], na.rm = TRUE)
                if (col_sd > 0) {
                  tr_imp[[col]] <- (tr_imp[[col]] - col_mean) / col_sd
                  ts_imp[[col]] <- (ts_imp[[col]] - col_mean) / col_sd
                }
              }
            }
            
            # Augment with Missing Indicators
            if (use_mi) {
              tr_aug <- cbind(tr_imp, train_Z)
              ts_aug <- cbind(ts_imp, test_Z)
            } else {
              tr_aug <- tr_imp
              ts_aug <- ts_imp
            }
            
            res <- run_bas_fold(tr_aug, ts_aug, cfg$y_col, iter = ITER)
            
            if (!is.null(res)) {
              fold_betas[[m]] <- res$beta_estimates
              fold_selected[[m]] <- res$selected_variables
              fold_preds[[m]] <- res$predictions
              fold_logprobs[[m]] <- res$log_probabilities
            }
          }
          
          all_betas[[fold]] <- fold_betas
          all_selected[[fold]] <- fold_selected
          all_preds[[fold]] <- fold_preds
          all_logprobs[[fold]] <- fold_logprobs
          cat("Done\n")
        }
        
        # SAVE ARTIFACTS
        for (m in files_indices) {
          # 1. Betas
          betas_df <- bind_rows(lapply(1:NUM_OF_FOLDS, function(f) {
            b <- tryCatch(all_betas[[f]][[m]], error = function(e) NULL)
            if (is.null(b)) return(NULL)
            data.frame(fold = f, variable = names(b), beta = as.numeric(b))
          }))
          
          # 2. Selected Variables
          sel_df <- bind_rows(lapply(1:NUM_OF_FOLDS, function(f) {
            s <- tryCatch(all_selected[[f]][[m]], error = function(e) NULL)
            if (is.null(s)) return(NULL)
            data.frame(fold = f, rank = 1:length(s), variable = s)
          }))
          
          # 3. Log Probs
          logp_matrix <- do.call(cbind, lapply(1:NUM_OF_FOLDS, function(f) {
            val <- tryCatch(all_logprobs[[f]][[m]], error = function(e) NULL)
            if (is.null(val)) rep(NA, 20) else val
          }))
          if (!is.null(logp_matrix) && ncol(logp_matrix) > 0) {
            avg_logp <- rowMeans(logp_matrix, na.rm = TRUE)
            logp_df <- data.frame(rep = 1, num_top = 1:length(avg_logp), avg_logp = avg_logp)
          } else {
            logp_df <- data.frame()
          }
          
          # 4. Predictions
          preds_df <- bind_rows(lapply(1:NUM_OF_FOLDS, function(f) {
            p <- tryCatch(all_preds[[f]][[m]], error = function(e) NULL)
            t_idx <- folds[[f]]
            if (is.null(p)) return(NULL)
            data.frame(fold = f, test_idx = t_idx, y_true = y_target[t_idx], predicted_prob = p)
          }))
          
          # Filename writing
          if (method == "MICE") {
            write.csv(betas_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_beta_estimates.csv"), row.names = FALSE)
            write.csv(sel_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_selected_variables.csv"), row.names = FALSE)
            write.csv(logp_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_log_probabilities.csv"), row.names = FALSE)
            write.csv(preds_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_predictions.csv"), row.names = FALSE)
          } else {
            write.csv(betas_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_beta_estimates.csv"), row.names = FALSE)
            write.csv(sel_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_selected_variables.csv"), row.names = FALSE)
            write.csv(logp_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_log_probabilities.csv"), row.names = FALSE)
            write.csv(preds_df, paste0("Results/CORRECTED/ALPHA10/", full_name, "_predictions.csv"), row.names = FALSE)
          }
        }
        
        # MICE POOLING
        if (method == "MICE") {
          cat("    Pooling MICE results...\n")
          # Check if any imputation files exist and are non-empty
          pred_files_exist <- all(sapply(1:m_val, function(m) {
            f <- paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_predictions.csv")
            if (!file.exists(f)) return(FALSE)
            tryCatch({
              df <- read.csv(f)
              nrow(df) > 0
            }, error = function(e) FALSE)
          }))
          logp_files_exist <- all(sapply(1:m_val, function(m) {
            f <- paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_log_probabilities.csv")
            file.exists(f) && file.info(f)$size > 0
          }))
          if (!pred_files_exist || !logp_files_exist) {
            cat("    [SKIPPING MICE POOLING: empty or missing imputation files]\n")
          } else {
            # 1. Pool Predictions
            all_m_preds <- lapply(1:m_val, function(m) {
              read.csv(paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_predictions.csv"))
            })
            pooled_preds <- all_m_preds[[1]]
            prob_cols <- do.call(cbind, lapply(all_m_preds, function(df) {
              if (nrow(df) == nrow(pooled_preds)) df$predicted_prob else rep(NA, nrow(pooled_preds))
            }))
            pooled_preds$predicted_prob <- rowMeans(prob_cols, na.rm = TRUE)
            write.csv(pooled_preds, paste0("Results/CORRECTED/ALPHA10/", full_name, "_POOLED_predictions.csv"), row.names = FALSE)
            
            # 2. Pool Log Probs
            all_m_logp <- lapply(1:m_val, function(m) {
              read.csv(paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_log_probabilities.csv"))
            })
            pooled_logp <- all_m_logp[[1]]
            logp_cols <- do.call(cbind, lapply(all_m_logp, function(df) df$avg_logp))
            pooled_logp$avg_logp <- rowMeans(logp_cols, na.rm = TRUE)
            write.csv(pooled_logp, paste0("Results/CORRECTED/ALPHA10/", full_name, "_POOLED_log_probabilities.csv"), row.names = FALSE)
            
            # 3. Pool Selected Variables
            all_m_sel <- bind_rows(lapply(1:m_val, function(m) {
              read.csv(paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_selected_variables.csv"))
            }))
            pooled_sel <- all_m_sel %>%
              group_by(variable) %>%
              summarise(mean_rank = mean(rank), count = n(), .groups = "drop") %>%
              arrange(mean_rank)
            write.csv(pooled_sel, paste0("Results/CORRECTED/ALPHA10/", full_name, "_POOLED_selected_variables.csv"), row.names = FALSE)
            
            # 4. Pool Betas
            all_m_betas <- bind_rows(lapply(1:m_val, function(m) {
              read.csv(paste0("Results/CORRECTED/ALPHA10/", full_name, "_m", m, "_beta_estimates.csv"))
            }))
            pooled_betas <- all_m_betas %>%
              group_by(variable) %>%
              summarise(mean_beta = mean(beta), sd_beta = sd(beta), .groups = "drop")
            write.csv(pooled_betas, paste0("Results/CORRECTED/ALPHA10/", full_name, "_POOLED_beta_estimates.csv"), row.names = FALSE)
          }
        }
      }
    }
  }
  cat("\nBAS ALPHA10 SENSITIVITY ANALYSIS COMPLETE.\n")
}

if (!interactive()) {
  run_analysis()
}
