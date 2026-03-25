# MASTER IMPUTATION SCRIPT (Scenario Analysis)

library(mice)
library(dplyr)

# Configuration
SELECTION_METHODS <- c("top", "independent", "mixed")
DATASETS <- list(
  MIMIC = "MIMIC",
  MI = "MI"
)
MECHANISMS <- c("MCAR", "MAR", "MNAR")

# Helper: Mean Impute
mean_impute <- function(data) {
  for (col in colnames(data)) {
    if (is.numeric(data[[col]])) {
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    }
  }
  return(data)
}

cat("Starting Imputation for all scenarios...\n")

for (method in SELECTION_METHODS) {
  cat(sprintf("\n--- Scenario: %s ---\n", method))
  for (dataset in names(DATASETS)) {
    for (mech in MECHANISMS) {
      file_path <- paste0("../../Data/", method, "/complete_dataset_", dataset, "_", mech, ".csv")
      
      if (!file.exists(file_path)) {
        cat(sprintf("  Warning: %s not found.\n", file_path))
        next
      }
      
      cat(sprintf("  Imputing %s / %s...\n", dataset, mech))
      df <- read.csv(file_path)
      
      # Determine Response Var
      resp_var <- if(dataset == "MIMIC") "ICD9_CODE" else "ZSN"
      
      # 1. Mean Imputation
      # We ONLY impute the original variables, not indicators
      all_cols <- names(df)
      orig_vars <- all_cols[!grepl("_missing|total_missing", all_cols) & all_cols != resp_var]
      
      df_mean <- df
      df_mean[, orig_vars] <- mean_impute(df[, orig_vars])
      write.csv(df_mean, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MEAN.csv"), row.names = FALSE)
      
      # 2. MICE Imputation (m=3)
      # We use mice on the same original columns
      # The simulation script includes Y and indicators in complete_dataset,
      # but we should only impute the feature columns.
      imp_data <- df[, c(resp_var, orig_vars)]
      imp <- mice(imp_data, m=3, method="pmm", maxit=50, printFlag=FALSE)
      
      for (i in 1:3) {
        df_imp_vals <- complete(imp, i)
        # Reconstruct full df with indicators and total_missing
        df_full_imp <- df
        df_full_imp[, orig_vars] <- df_imp_vals[, orig_vars]
        write.csv(df_full_imp, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MICE_", i, ".csv"), row.names = FALSE)
      }
    }
  }
}

cat("\nImputation Complete.\n")
