# MASTER IMPUTATION SCRIPT (Scenario Analysis)

library(mice)
library(dplyr)
library(VIM)
library(missForest)

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
      
      # 1. Mean Imputation (COMMENTED OUT)
      # df_mean <- df
      # df_mean[, orig_vars] <- mean_impute(df[, orig_vars])
      # write.csv(df_mean, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MEAN.csv"), row.names = FALSE)
      
      # 2. MICE Imputation (m=3) (COMMENTED OUT)
      # imp_data <- df[, c(resp_var, orig_vars)]
      # imp <- mice(imp_data, m=3, method="pmm", maxit=50, printFlag=FALSE)
      # for (i in 1:3) {
      #   df_imp_vals <- complete(imp, i)
      #   df_full_imp <- df
      #   df_full_imp[, orig_vars] <- df_imp_vals[, orig_vars]
      #   write.csv(df_full_imp, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MICE_", i, ".csv"), row.names = FALSE)
      # }

      # 3. KNN Imputation
      cat("    Running KNN Imputation...\n")
      df_knn <- df
      knn_imp <- VIM::kNN(df[, orig_vars], k = 5, imp_var = FALSE)
      df_knn[, orig_vars] <- knn_imp
      write.csv(df_knn, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_KNN.csv"), row.names = FALSE)

      # 4. missForest Imputation
      cat("    Running missForest Imputation...\n")
      df_mf <- df
      mf_imp <- missForest::missForest(df[, orig_vars])
      df_mf[, orig_vars] <- mf_imp$ximp
      write.csv(df_mf, paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_missForest.csv"), row.names = FALSE)
    }
  }
}

cat("\nImputation Complete.\n")
