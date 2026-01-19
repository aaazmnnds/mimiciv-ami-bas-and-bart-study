################################################################################
# IMPUTATION METHODS: MICE, MEAN, MISSFOREST, KNN
#
# Processed Datasets (Total 12):
# 1. Original Beta Case (6 datasets):
#    - MIMIC: MCAR, MAR, MNAR
#    - MI: MCAR, MAR, MNAR
# 2. Extended Beta Case (6 datasets):
#    - MIMIC: MCAR, MAR, MNAR (alpha=10 case)
#    - MI: MCAR, MAR, MNAR (alpha=10 case)
#
# Output:
# - MICE (m=3)
# - Mean Imputation
# - missForest
# - KNN
################################################################################

# ============================================================================
# 1. SETUP & LIBRARIES
# ============================================================================

library(mice)       # For MICE
library(missForest) # For missForest
library(VIM)        # For KNN
library(dplyr)      # For data manipulation

set.seed(123)

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

# Parameters
M_IMPUTATIONS <- 3    # MICE: Number of imputed datasets
MAXIT_MICE <- 50      # MICE: Max iterations
MAXIT_MF <- 10        # missForest: Max iterations
NTREE_MF <- 100       # missForest: Number of trees

# Define all 12 datasets to process
datasets <- list(
  # --- ORIGINAL BETA CASE (6) ---
  list(name = "MIMIC_MCAR", file = "Data/complete_dataset_MIMIC_MCAR.csv", response = "ICD9_CODE"),
  list(name = "MIMIC_MAR",  file = "Data/complete_dataset_MIMIC_MAR.csv",  response = "ICD9_CODE"),
  list(name = "MIMIC_MNAR", file = "Data/complete_dataset_MIMIC_MNAR.csv", response = "ICD9_CODE"),
  
  list(name = "MI_MCAR", file = "Data/complete_dataset_MI_MCAR.csv", response = "ZSN"),
  list(name = "MI_MAR",  file = "Data/complete_dataset_MI_MAR.csv",  response = "ZSN"),
  list(name = "MI_MNAR", file = "Data/complete_dataset_MI_MNAR.csv", response = "ZSN"),
  
  # --- EXTENDED BETA CASE (6) ---
  # Filenames end with _NEW_alpha10.csv based on previous script output
  list(name = "MIMIC_MCAR_EXT", file = "Data/complete_dataset_MIMIC_MCAR_NEW_alpha10.csv", response = "ICD9_CODE"),
  list(name = "MIMIC_MAR_EXT",  file = "Data/complete_dataset_MIMIC_MAR_NEW_alpha10.csv",  response = "ICD9_CODE"),
  list(name = "MIMIC_MNAR_EXT", file = "Data/complete_dataset_MIMIC_MNAR_NEW_alpha10.csv", response = "ICD9_CODE"),
  
  list(name = "MI_MCAR_EXT", file = "Data/complete_dataset_MI_MCAR_NEW_alpha10.csv", response = "ZSN"),
  list(name = "MI_MAR_EXT",  file = "Data/complete_dataset_MI_MAR_NEW_alpha10.csv",  response = "ZSN"),
  list(name = "MI_MNAR_EXT", file = "Data/complete_dataset_MI_MNAR_NEW_alpha10.csv", response = "ZSN")
)


# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

# Function: Separate VALUE columns from indicators
separate_columns <- function(data, response_var) {
  # Identify column types
  # We want to impute only actual value columns, not indicators or totals
  value_cols <- !grepl("_missing|total_missing", names(data))
  indicator_cols <- grepl("_missing", names(data)) & !grepl("total_missing", names(data))
  total_missing_col <- grepl("^total_missing", names(data))
  
  result <- list(
    value_data = data[, value_cols],
    indicators = data[, indicator_cols, drop = FALSE],
    total_missing = data[, total_missing_col, drop = FALSE]
  )
  return(result)
}

# Function: Recombine imputed data with indicators
recombine_data <- function(imputed_values, indicators, total_missing) {
  complete_data <- cbind(imputed_values, indicators, total_missing)
  return(complete_data)
}

# Function: Mean Imputation
mean_impute <- function(data) {
  for (col in colnames(data)) {
    if (is.numeric(data[[col]])) {
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    }
  }
  return(data)
}

# Function: Prepare data for missForest (Handle factors/missingness limits)
prepare_for_missforest <- function(data) {
  # Convert character columns to factors
  char_cols <- sapply(data, is.character)
  if (any(char_cols)) {
    data[, char_cols] <- lapply(data[, char_cols], function(x) {
      x <- factor(x)
      if (length(levels(x)) <= 1) return(NULL)  # skip useless columns
      return(x)
    })
    data <- data[, !sapply(data, is.null)]
  }
  
  # Remove columns with >80% missing (missForest constraint)
  na_props <- colMeans(is.na(data))
  if (any(na_props > 0.8)) {
    cat(sprintf("  ⚠️  Removing %d columns with >80%% missing for missForest\n", sum(na_props > 0.8)))
    data <- data[, na_props <= 0.8]
  }
  
  return(data)
}


# ============================================================================
# 4. MAIN PROCESSING LOOP
# ============================================================================

cat(paste(rep("=", 80), collapse=""), "\n")
cat("STARTING IMPUTATION PIPELINE - 12 DATASETS\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

for (dataset_idx in seq_along(datasets)) {
  
  ds <- datasets[[dataset_idx]]
  
  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat(sprintf("PROCESSING %d/12: %s\n", dataset_idx, ds$name))
  cat(sprintf("File: %s\n", ds$file))
  cat(paste(rep("=", 80), collapse=""), "\n\n")
  
  # --- CHECK FILE EXISTENCE ---
  if (!file.exists(ds$file)) {
    cat(sprintf("⚠️  File not found: %s\n", ds$file))
    cat("Skipping...\n")
    next
  }
  
  # --- LOAD DATA ---
  # Use stringsAsFactors=FALSE for compatibility
  data_full <- read.csv(ds$file, stringsAsFactors = FALSE)
  
  # Separate logical components
  separated <- separate_columns(data_full, ds$response)
  data_values <- separated$value_data  # This is what we impute
  indicators <- separated$indicators
  total_missing <- separated$total_missing
  
  cat(sprintf("Dimensions to impute: %d rows x %d cols\n", nrow(data_values), ncol(data_values)))
  cat(sprintf("Missing values: %d (%.2f%%)\n\n", sum(is.na(data_values)), mean(is.na(data_values))*100))
  
  
  # --------------------------------------------------------------------------
  # A. MICE IMPUTATION (m=3)
  # --------------------------------------------------------------------------
  cat("--- Running MICE (m=3) ---\n")
  
  tryCatch({
    mice_out <- mice(data = data_values, method = "pmm", m = M_IMPUTATIONS, maxit = MAXIT_MICE, print = FALSE)
    
    for (i in 1:M_IMPUTATIONS) {
      imputed_part <- complete(mice_out, action = i)
      final_mice <- recombine_data(imputed_part, indicators, total_missing)
      
      filename <- paste0("Data/", ds$name, "_MICE_", i, ".csv")
      write.csv(final_mice, filename, row.names = FALSE)
      cat(sprintf("  Saved: %s\n", filename))
    }
  }, error = function(e) {
    cat(sprintf("  ERROR in MICE: %s\n", e$message))
  })
  cat("\n")
  
  
  # --------------------------------------------------------------------------
  # B. MEAN IMPUTATION
  # --------------------------------------------------------------------------
  cat("--- Running MEAN Imputation ---\n")
  
  tryCatch({
    mean_part <- mean_impute(data_values)
    final_mean <- recombine_data(mean_part, indicators, total_missing)
    
    filename <- paste0("Data/", ds$name, "_MEAN.csv")
    write.csv(final_mean, filename, row.names = FALSE)
    cat(sprintf("  Saved: %s\n", filename))
  }, error = function(e) {
    cat(sprintf("  ERROR in Mean Imputation: %s\n", e$message))
  })
  cat("\n")
  
  
  # --------------------------------------------------------------------------
  # C. MISSFOREST IMPUTATION
  # --------------------------------------------------------------------------
  cat("--- Running missForest ---\n")
  
  tryCatch({
    mf_data <- prepare_for_missforest(data_values)
    mf_out <- missForest(mf_data, maxiter = MAXIT_MF, ntree = NTREE_MF, verbose = FALSE, parallelize = 'no')
    
    final_mf <- recombine_data(mf_out$ximp, indicators, total_missing)
    
    filename <- paste0("Data/", ds$name, "_missForest.csv")
    write.csv(final_mf, filename, row.names = FALSE)
    cat(sprintf("  Saved: %s (NRMSE: %.4f)\n", filename, mf_out$OOBerror))
  }, error = function(e) {
      cat(sprintf("  ERROR in missForest: %s\n", e$message))
  })
  cat("\n")
  
  
  # --------------------------------------------------------------------------
  # D. KNN IMPUTATION
  # --------------------------------------------------------------------------
  cat("--- Running KNN ---\n")
  
  tryCatch({
    # Prepare data for KNN (exclude response variable from imputation set if present in data_values, 
    # though separate_columns usually keeps response in data_values. 
    # Logic: KNN often shouldn't use response for imputation, but here we treat all as features?
    # Original script separated response out. We will follow suit.)
    
    response_col_data <- data_values[[ds$response]]
    vars_to_impute <- setdiff(names(data_values), ds$response)
    knn_input <- data_values[, vars_to_impute]
    
    k_val <- round(sqrt(nrow(knn_input)))
    cat(sprintf("  k = %d\n", k_val))
    
    knn_out <- kNN(knn_input, k = k_val, imp_var = FALSE)
    
    # Reattach response
    # Need to verify if kNN reordered rows? (Usually no)
    knn_combined_values <- cbind(data.frame(TEMP_RESP = response_col_data), knn_out)
    # Rename response column back
    names(knn_combined_values)[1] <- ds$response
    
    final_knn <- recombine_data(knn_combined_values, indicators, total_missing)
    
    filename <- paste0("Data/", ds$name, "_KNN.csv")
    write.csv(final_knn, filename, row.names = FALSE)
    cat(sprintf("  Saved: %s\n", filename))
    
  }, error = function(e) {
    cat(sprintf("  ERROR in KNN: %s\n", e$message))
  })
  
  cat("\nDone with dataset.\n")
}

cat("\n")
cat(paste(rep("=", 80), collapse=""), "\n")
cat("ALL PROCESSES COMPLETED.\n")
cat(paste(rep("=", 80), collapse=""), "\n")
