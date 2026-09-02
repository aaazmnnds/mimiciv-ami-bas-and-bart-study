.libPaths("~/R/library")
# BART ANALYSIS SCRIPT (ALPHA10 SENSITIVITY PIPELINE)
#
# Generates evaluation artifacts for:
# 1. Variable Selection (Selected Vars)
# 2. Prediction (Predictions & Log Probs)
#
# Runs on ALPHA10 datasets where outcomes are driven by missing indicators
# Imputation and normalization are performed strictly inside the 10-fold CV loop.
#
# Outputs to: Results/CORRECTED/ALPHA10/

library(BART)
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
M_VAL <- 3

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

dir.create("Results/CORRECTED/ALPHA10", recursive = TRUE, showWarnings = FALSE)

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
  
  # Adaptive settings for high dimensional vs standard
  p_dim <- ncol(x.train)
  ntree_val <- if (p_dim > 100) 50L else 100L
  ndpost_val <- if (p_dim > 100) 200L else 500L
  
  bart_model <- tryCatch({
    R.utils::withTimeout({
      # Remove any columns with all-NA or zero variance
      x.train <- x.train[, apply(x.train, 2, function(col) !all(is.na(col)) && sd(col, na.rm=TRUE) > 0), drop=FALSE]
      x.train <- as.matrix(x.train)
      # Replace any remaining NAs with 0
      x.train[is.na(x.train)] <- 0
      x.train[!is.finite(x.train)] <- 0
      lbart(
        x.train = x.train, y.train = y.train, sparse = FALSE,
        ntree = ntree_val, ndpost = ndpost_val, nskip = 100L, printevery = 10000L, transposed = FALSE
      )
    }, timeout = 300, onTimeout = "error")
  }, error = function(e) {
    cat(sprintf("    [TIMEOUT/ERROR in var importance: %s]\n", conditionMessage(e)))
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
        x.train <- as.matrix(x.train)
        x.test <- as.matrix(x.test)
        mode(x.train) <- "numeric"
        mode(x.test) <- "numeric"
        x.train[is.na(x.train)] <- 0
        x.test[is.na(x.test)] <- 0
        x.train[!is.finite(x.train)] <- 0
        x.test[!is.finite(x.test)] <- 0
        y.train <- as.numeric(as.character(y.train))
        y.train[is.na(y.train) | !is.finite(y.train)] <- 0
        valid_cols <- apply(x.train, 2, function(col) var(col, na.rm=TRUE) > 1e-10)
        if (sum(valid_cols) == 0) stop("No valid columns")
        x.train <- x.train[, valid_cols, drop=FALSE]
        x.test <- x.test[, valid_cols, drop=FALSE]
        lbart(x.train = x.train, y.train = y.train, x.test = x.test,
              sparse = FALSE, transposed = FALSE,
              ntree = 100L, ndpost = 100L, nskip = 50L, printevery = 10000L)
      }, timeout = 300, onTimeout = "error")
    }, error=function(e) return(NULL))
    
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
  
  valid_lp <- which(!is.na(log_probs))
  if (length(valid_lp) == 0) {
    preds_best <- rep(NA, nrow(test_data))
    best_k <- NA
  } else {
    best_k <- valid_lp[which.max(log_probs[valid_lp])]
    preds_best <- all_k_preds[[best_k]]
  }
  
  return(list(log_probs = log_probs, preds_best = preds_best, best_k = best_k))
}

# 3. MAIN PIPELINE
run_analysis <- function() {
  cat("STARTING BART ALPHA10 SENSITIVITY ANALYSIS\n")
  
  for (cfg in configs) {
    if (!file.exists(cfg$file)) {
      cat(sprintf("Config file missing: %s\n", cfg$file))
      next
    }
    
    raw_data <- read.csv(cfg$file)
    value_cols <- names(raw_data)[!grepl("_missing|total_missing", names(raw_data))]
    raw_data <- raw_data[, value_cols, drop=FALSE]
    # Exclude patient ID columns
    raw_data <- raw_data[, !names(raw_data) %in% c("HADM_ID", "subject_id", "stay_id"), drop=FALSE]
    
    y_target <- raw_data[[cfg$y_col]]
    
    for (method in METHODS) {
      for (use_mi in MI_CONDITIONS) {
        mi_tag <- if (use_mi) "wMI" else "noMI"
        full_name <- paste0(cfg$name, "_", method, "_", mi_tag)
        
        if (method == "MICE") {
          m_val <- M_VAL
          files_indices <- 1:m_val
          output_suffix <- paste0("_m", m_val)
        } else {
          m_val <- 1
          files_indices <- c(1)
          output_suffix <- "_m1"
        }
        
        # Skip logic check
        if (method == "MICE") {
          fname_check <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_POOLED_log_probabilities", output_suffix, ".csv")
        } else {
          fname_check <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_log_probabilities", output_suffix, ".csv")
        }
        if (file.exists(fname_check)) {
          existing <- read.csv(fname_check)
          if (nrow(existing) >= TOP_NUM) {
            cat(sprintf("Skipping completed condition: %s\n", full_name))
            next
          }
        }
        
        cat(sprintf("\nProcessing: %s (Method: %s, MI: %s, m: %d)\n", cfg$name, method, mi_tag, m_val))
        
        df_logprobs_all <- data.frame()
        df_preds_all <- data.frame()
        df_selected_all <- data.frame()
        
        for (r in 1:N_REPEATS) {
          set.seed(123 + r * 1000)
          folds <- create_stratified_folds(y_target, k = NUM_OF_FOLDS)
          
          all_logprobs <- list()
          all_selected <- list()
          all_preds <- list()
          
          for (f in 1:NUM_OF_FOLDS) {
            cat(sprintf("  Repeat %d/%d - Fold %d/%d...\n", r, N_REPEATS, f, NUM_OF_FOLDS))
            test_idx <- folds[[f]]
            train_idx <- setdiff(1:nrow(raw_data), test_idx)
            
            train_raw <- raw_data[train_idx, , drop=FALSE]
            test_raw  <- raw_data[test_idx, , drop=FALSE]
            
            # Missing Indicators
            train_mi <- as.data.frame(lapply(train_raw, function(x) as.numeric(is.na(x))))
            test_mi  <- as.data.frame(lapply(test_raw, function(x) as.numeric(is.na(x))))
            names(train_mi) <- paste0(names(train_raw), "_missing")
            names(test_mi)  <- paste0(names(test_raw), "_missing")
            
            # Remove constant indicators
            valid_mi <- sapply(train_mi, function(col) length(unique(col)) > 1)
            train_mi <- train_mi[, valid_mi, drop=FALSE]
            test_mi  <- test_mi[, valid_mi, drop=FALSE]
            
            # Perform Method-Specific Imputation strictly inside fold
            train_imp_list <- list()
            test_imp_list  <- list()
            
            if (method == "MEAN") {
              means <- sapply(train_raw, function(x) if(is.numeric(x)) mean(x, na.rm=TRUE) else NA)
              tr_imp <- train_raw
              te_imp <- test_raw
              for (col in names(tr_imp)) {
                if (col == cfg$y_col) next
                if (is.numeric(tr_imp[[col]])) {
                  m_val_col <- means[col]
                  if (is.na(m_val_col)) m_val_col <- 0
                  tr_imp[[col]][is.na(tr_imp[[col]])] <- m_val_col
                  te_imp[[col]][is.na(te_imp[[col]])] <- m_val_col
                }
              }
              train_imp_list[[1]] <- tr_imp
              test_imp_list[[1]]  <- te_imp
              
            } else if (method == "KNN") {
              combined_fold <- rbind(train_raw, test_raw)
              suppressWarnings({
                imp_combined <- VIM::kNN(combined_fold, k=5, imp_var=FALSE)
              })
              train_imp_list[[1]] <- imp_combined[1:nrow(train_raw), , drop=FALSE]
              test_imp_list[[1]]  <- imp_combined[(nrow(train_raw)+1):nrow(combined_fold), , drop=FALSE]
              
            } else if (method == "missForest") {
              suppressWarnings({
                tr_rf <- missRanger::missRanger(train_raw, pmm.k=3, verbose=0, num.trees=50)
                te_rf <- missRanger::missRanger(test_raw, pmm.k=3, verbose=0, num.trees=50)
              })
              train_imp_list[[1]] <- tr_rf
              test_imp_list[[1]]  <- te_rf
              
            } else if (method == "MICE") {
              suppressWarnings({
                tr_mice_obj <- mice::mice(train_raw, m=m_val, method="pmm", printFlag=FALSE, maxit=5)
                te_mice_obj <- mice::mice(test_raw, m=m_val, method="pmm", printFlag=FALSE, maxit=5)
              })
              for (m in 1:m_val) {
                train_imp_list[[m]] <- mice::complete(tr_mice_obj, m)
                test_imp_list[[m]]  <- mice::complete(te_mice_obj, m)
              }
            }
            
            # Process each imputation
            for (m in files_indices) {
              tr_data <- train_imp_list[[m]]
              te_data <- test_imp_list[[m]]
              
              if (use_mi && ncol(train_mi) > 0) {
                y_tr <- tr_data[[cfg$y_col]]
                y_te <- te_data[[cfg$y_col]]
                tr_data[[cfg$y_col]] <- NULL
                te_data[[cfg$y_col]] <- NULL
                
                tr_data <- cbind(tr_data, train_mi)
                te_data <- cbind(te_data, test_mi)
                
                tr_data[[cfg$y_col]] <- y_tr
                te_data[[cfg$y_col]] <- y_te
              }
              
              # Normalize based on training fold
              pred_names <- setdiff(names(tr_data), cfg$y_col)
              for (col in pred_names) {
                if (is.numeric(tr_data[[col]])) {
                  col_mean <- mean(tr_data[[col]], na.rm=TRUE)
                  col_sd   <- sd(tr_data[[col]], na.rm=TRUE)
                  if (!is.na(col_sd) && col_sd > 0) {
                    tr_data[[col]] <- (tr_data[[col]] - col_mean) / col_sd
                    te_data[[col]] <- (te_data[[col]] - col_mean) / col_sd
                  }
                }
              }
              
              # Run BART variable importance
              top_vars <- get_bart_importance(tr_data, cfg$y_col)
              
              if (is.null(top_vars) || length(top_vars) == 0) {
                cat("    [WARN: Variable importance failed]\n")
                next
              }
              
              eval_len <- min(TOP_NUM, length(top_vars))
              top_vars_eval <- top_vars[1:eval_len]
              
              all_selected[[length(all_selected) + 1]] <- data.frame(
                rep = r, fold = f, m = m,
                rank = 1:eval_len,
                variable = top_vars_eval
              )
              
              # Evaluate Top Variables
              res_eval <- evaluate_top_vars_bart(tr_data, te_data, cfg$y_col, top_vars_eval)
              
              all_logprobs[[length(all_logprobs) + 1]] <- data.frame(
                rep = r, fold = f, m = m,
                num_top = 1:eval_len,
                log_prob = res_eval$log_probs
              )
              
              all_preds[[length(all_preds) + 1]] <- data.frame(
                rep = r, fold = f, m = m,
                obs_id = test_idx,
                true_label = te_data[[cfg$y_col]],
                num_top = res_eval$best_k,
                predicted_prob = res_eval$preds_best
              )
            } # close m loop
          } # close fold loop
          
          if (length(all_selected) > 0) df_selected_all <- rbind(df_selected_all, do.call(rbind, all_selected))
          if (length(all_logprobs) > 0) df_logprobs_all <- rbind(df_logprobs_all, do.call(rbind, all_logprobs))
          if (length(all_preds) > 0) df_preds_all <- rbind(df_preds_all, do.call(rbind, all_preds))
        } # close rep loop
        
        # Save results to Results/CORRECTED/ALPHA10/
        if (nrow(df_logprobs_all) > 0) {
          df_log <- df_logprobs_all
          if (method == "MICE") {
            df_log <- df_log %>% group_by(rep, num_top) %>% summarise(log_prob_pooled = mean(log_prob, na.rm=TRUE), .groups="drop")
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_POOLED_log_probabilities", output_suffix, ".csv")
          } else {
            df_log <- df_log %>% group_by(rep, num_top) %>% summarise(avg_log_prob = mean(log_prob, na.rm=TRUE), .groups="drop")
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_log_probabilities", output_suffix, ".csv")
          }
          write.csv(df_log, fname, row.names=FALSE)
        }
        
        if (nrow(df_selected_all) > 0) {
          df_sel <- df_selected_all
          if (method == "MICE") {
            df_sel <- df_sel %>% group_by(fold, variable) %>% summarise(rank = mean(rank, na.rm=TRUE), .groups="drop")
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_POOLED_selected_variables", output_suffix, ".csv")
          } else {
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_selected_variables", output_suffix, ".csv")
          }
          write.csv(df_sel, fname, row.names=FALSE)
        }
        
        if (nrow(df_preds_all) > 0) {
          df_pred <- df_preds_all
          if (method == "MICE") {
            df_pred <- df_pred %>% group_by(rep, fold, obs_id, true_label, num_top) %>% summarise(predicted_prob_pooled = mean(predicted_prob, na.rm=TRUE), .groups="drop")
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_POOLED_predictions", output_suffix, ".csv")
          } else {
            fname <- paste0("Results/CORRECTED/ALPHA10/results_BART_", full_name, "_predictions", output_suffix, ".csv")
          }
          write.csv(df_pred, fname, row.names=FALSE)
        }
        
      } # close mi loop
    } # close method loop
  } # close config loop
  cat("\nBART ALPHA10 SENSITIVITY ANALYSIS COMPLETE.\n")
}

if (!interactive()) {
  run_analysis()
}
