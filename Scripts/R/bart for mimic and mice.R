################################################################################
# BART ANALYSIS SCRIPT
#
# Datasets: 
# 1. Real Data: MIMIC-III, MI
# 2. Simulated Data: MIMIC/MI x (MCAR, MAR, MNAR)
#
# Imputation Methods: Mean, MICE (m=3, m=20), missForest, KNN
#
# Method: 
# - Bayesian Additive Regression Trees (BART) for classification (lbart)
# - Variable Selection based on inclusion counts
# - Evaluation of Average Log Predicted Probabilities for Top 1 to Top 20 variables
#
################################################################################

library(BART)
library(dplyr)

set.seed(123)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

NUM_OF_FOLDS <- 10
TOP_NUM <- 20

# M-values for MICE
M_VALUES <- c(3, 20) 

# Dataset Configurations
# Adjust 'file_pattern' to match exactly what your preprocessing pipeline outputs.
# Pattern: [Fold]_90train_data[Tag].csv
# CRITICAL: 'y_col' must match the response column name in your CSVs.

configs <- list(
  
  # ==========================================================================
  # REAL DATASETS (MIMIC-III & MI)
  # ==========================================================================
  
  # --- MIMIC ---
  list(name = "MIMIC_MEAN", method = "MEAN", 
       train_pattern = "%d_90train_datamimiciii_mean_imputed.csv", 
       test_pattern = "%d_10test_datamimiciii_mean_imputed.csv", 
       y_col = "ICD9_CODE"),
       
  list(name = "MIMIC_MICE", method = "MICE", 
       train_pattern = "%d_90train_datamimiciii_mice_imputed_%d.csv", 
       test_pattern = "%d_10test_datamimiciii_mice_imputed_%d.csv", 
       y_col = "ICD9_CODE"),
       
  # --- MI ---
  list(name = "MI_MEAN", method = "MEAN", 
       train_pattern = "%d_90train_dataMI.meanimp.csv", 
       test_pattern = "%d_10test_dataMI.meanimp.csv", 
       y_col = "ZSN"),
  
  list(name = "MI_MICE", method = "MICE", 
       train_pattern = "%d_90train_dataMI1_%d.csv", 
       test_pattern = "%d_10test_dataMI1_%d.csv", 
       y_col = "ZSN"),


  # ==========================================================================
  # SIMULATED DATASETS (Original Beta Case)
  # ==========================================================================
  
  # --- MIMIC MCAR ---
  list(name = "Sim_MIMIC_MCAR_MEAN", method = "MEAN", 
       train_pattern = "%d_90train_dataMIMIC_MCAR_MEAN.csv", 
       test_pattern = "%d_10test_dataMIMIC_MCAR_MEAN.csv", 
       y_col = "ICD9_CODE"),
       
  list(name = "Sim_MIMIC_MCAR_MICE", method = "MICE", 
       train_pattern = "%d_90train_dataMIMIC_MCAR_MICE_%d.csv", 
       test_pattern = "%d_10test_dataMIMIC_MCAR_MICE_%d.csv", 
       y_col = "ICD9_CODE"),
       
  list(name = "Sim_MIMIC_MCAR_missForest", method = "MF", 
       train_pattern = "%d_90train_dataMIMIC_MCAR_missForest.csv", 
       test_pattern = "%d_10test_dataMIMIC_MCAR_missForest.csv", 
       y_col = "ICD9_CODE"),
  
  list(name = "Sim_MIMIC_MCAR_KNN", method = "KNN", 
       train_pattern = "%d_90train_dataMIMIC_MCAR_KNN.csv", 
       test_pattern = "%d_10test_dataMIMIC_MCAR_KNN.csv", 
       y_col = "ICD9_CODE"),

  # --- MI MCAR ---
  list(name = "Sim_MI_MCAR_MEAN", method = "MEAN", 
       train_pattern = "%d_90train_dataMI_MCAR_MEAN.csv", 
       test_pattern = "%d_10test_dataMI_MCAR_MEAN.csv", 
       y_col = "ZSN"),
       
  list(name = "Sim_MI_MCAR_MICE", method = "MICE", 
       train_pattern = "%d_90train_dataMI_MCAR_MICE_%d.csv", 
       test_pattern = "%d_10test_dataMI_MCAR_MICE_%d.csv", 
       y_col = "ZSN"),

  list(name = "Sim_MI_MCAR_missForest", method = "MF", 
       train_pattern = "%d_90train_dataMI_MCAR_missForest.csv", 
       test_pattern = "%d_10test_dataMI_MCAR_missForest.csv", 
       y_col = "ZSN"),

  list(name = "Sim_MI_MCAR_KNN", method = "KNN", 
       train_pattern = "%d_90train_dataMI_MCAR_KNN.csv", 
       test_pattern = "%d_10test_dataMI_MCAR_KNN.csv", 
       y_col = "ZSN")
       
  # (Note: Assuming MAR/MNAR files follow similar naming conventions)
)


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

# Read and preprocess data
get_data <- function(filename) {
  if (!file.exists(filename)) {
    return(NULL) 
  }
  data <- read.csv(filename)
  # Remove indicators automatically
  data <- data[, !grepl("_missing", names(data))]
  return(data)
}

# Run BART on Data to get Variable Importance
get_bart_importance <- function(data, y_col) {
  
  # Prepare Matrices
  target_idx <- which(names(data) == y_col)
  if (length(target_idx) == 0) stop(paste("Column", y_col, "not found"))
  
  x.train <- as.matrix(data[, -target_idx])
  y.train <- data[[y_col]]
  
  # Parameters matched to original script
  bart_model <- lbart(
    x.train = x.train,
    y.train = y.train,
    sparse = FALSE,
    ntree = 200L,
    ndpost = 1000L,
    nskip = 100L,
    printevery = 10000L, # reduce spam
    transposed = FALSE
  )
  
  varcount <- bart_model$varcount
  row_sums <- rowSums(varcount)
  # Normalize
  normalized_matrix <- sweep(varcount, 1, row_sums, FUN = "/")
  normalized_sums <- colSums(normalized_matrix)
  normalized_sums <- normalized_sums / sum(normalized_sums)
  
  # Sort and return
  sorted_vars <- sort(normalized_sums, decreasing = TRUE)
  return(names(sorted_vars))
}

# Evaluate Top K variables
evaluate_top_vars_bart <- function(train_data, test_data, y_col, top_vars) {
  
  log_probs <- numeric(length(top_vars))
  
  target_idx_train <- which(names(train_data) == y_col)
  target_idx_test <- which(names(test_data) == y_col)
  
  for (k in 1:length(top_vars)) {
    current_vars <- top_vars[1:k]
    
    # Subset Data
    x.train <- as.matrix(train_data[, current_vars, drop=FALSE])
    y.train <- train_data[[y_col]]
    
    x.test <- as.matrix(test_data[, current_vars, drop=FALSE])
    y.test <- test_data[[y_col]]
    
    # Fit BART
    bart_mod <- lbart(
      x.train = x.train,
      y.train = y.train,
      x.test = x.test,
      ntree = 200L,
      ndpost = 1000L,
      nskip = 100L,
      printevery = 10000L
    )
    
    # Predictions (prob.test.mean is Mean posterior probability)
    preds <- bart_mod$prob.test.mean
    
    # Calculate True Prob (likelihood of actual class)
    true_probs <- ifelse(y.test == 1, preds, 1 - preds)
    true_probs[true_probs < 1e-10] <- 1e-10 # Log safety
    
    log_probs[k] <- mean(log(true_probs))
  }
  return(log_probs)
}


# ============================================================================
# 3. MAIN ANALYSIS LOOP
# ============================================================================

run_analysis <- function() {
  
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("STARTING BART ANALYSIS\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  
  for (cfg in configs) {
    
    cat(sprintf("\n--- Processing Config: %s ---\n", cfg$name))
    
    m_list <- if (cfg$method == "MICE") M_VALUES else c(1)
    
    for (m in m_list) {
       
       if (cfg$method == "MICE") {
           cat(sprintf("  > MICE Run (m=%d)\n", m))
           files_indices <- 1:m
           output_suffix <- paste0("_m", m)
       } else {
           files_indices <- c(1)
           output_suffix <- "_m1"
       }
       
       all_folds_log_probs <- list()
       
       for (fold in 1:NUM_OF_FOLDS) {
         cat(sprintf("    Fold %d/%d ", fold, NUM_OF_FOLDS))

         m_log_probs <- matrix(NA, nrow = length(files_indices), ncol = TOP_NUM)
         valid_count <- 0
         
         for (imp_idx in seq_along(files_indices)) {
             idx <- files_indices[imp_idx]
             
             if (cfg$method == "MICE") {
                 train_file <- sprintf(cfg$train_pattern, fold, idx)
                 test_file <- sprintf(cfg$test_pattern, fold, idx)
             } else {
                 train_file <- sprintf(cfg$train_pattern, fold)
                 test_file <- sprintf(cfg$test_pattern, fold)
             }
             
             train_df <- get_data(train_file)
             test_df <- get_data(test_file)
             
             if (!is.null(train_df) && !is.null(test_df)) {
               
                 cat(".") # Progress dot
                 tryCatch({
                     # 1. Get Variable Importance
                     ranked_vars <- get_bart_importance(train_df, cfg$y_col)
                     top_k_vars <- ranked_vars[1:min(length(ranked_vars), TOP_NUM)]
                     
                     # 2. Evaluate Top K
                     lps <- evaluate_top_vars_bart(train_df, test_df, cfg$y_col, top_k_vars)
                     
                     # Pad if needed
                     if (length(lps) < TOP_NUM) {
                         lps <- c(lps, rep(NA, TOP_NUM - length(lps)))
                     }
                     m_log_probs[imp_idx, ] <- lps
                     valid_count <- valid_count + 1
                     
                 }, error = function(e) {
                     # cat(sprintf(" [Err: %s] ", e$message))
                 })
             }
         }
         
         if (valid_count > 0) {
             avg_probs <- colMeans(m_log_probs[1:valid_count, , drop=FALSE], na.rm = TRUE)
             all_folds_log_probs[[fold]] <- avg_probs
             cat(" OK\n")
         } else {
             cat(" Skip\n")
         }
       } 
       
       # Aggregate Results over Folds
       if (length(all_folds_log_probs) > 0) {
           results_mat <- do.call(rbind, all_folds_log_probs)
           final_avg <- colMeans(results_mat, na.rm = TRUE)
           
           results_df <- data.frame(
               Top_Var_Count = 1:length(final_avg),
               Avg_Log_Prob = final_avg
           )
           
           fname <- paste0("results_BART_", cfg$name, output_suffix, ".csv")
           write.csv(results_df, fname, row.names = FALSE)
           cat(sprintf("\n  SAVED: %s\n", fname))
       }
    }
  }
  cat("\nALL DONE.\n")
}

run_analysis()
