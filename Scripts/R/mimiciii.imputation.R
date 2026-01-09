# ==============================================================================
# IMPUTATION SCRIPT: MICE & MEAN
#
# Datasets:
# 1. MIMIC-III: cleaned.mi (mimiciii).csv
# 2. Myocardial Infarction (MI): cleaned.mi (myocardial infarction).csv
#
# Methods:
# - MICE (PMM, m=3)
# - Mean Imputation
# ==============================================================================

library(mice)

# Helper Function: Simple Mean Imputation
mean_impute <- function(data) {
  for (col in colnames(data)) {
    if (is.numeric(data[[col]])) {
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    }
  }
  return(data)
}

# ==============================================================================
# 1. MIMIC-III DATASET
# ==============================================================================

cat("\nProcessing MIMIC-III Dataset...\n")
mimic_file <- "cleaned.mi (mimiciii).csv"

if(file.exists(mimic_file)) {
    mimiciii <- read.csv(mimic_file)
    
    # --- A. MICE Imputation ---
    cat("  Running MICE (m=3)...\n")
    imputation_model_mimic <- mice(data = mimiciii, 
                                   method = "pmm",
                                   m = 3,    
                                   maxit = 50, 
                                   print = FALSE)
    
    for (i in 1:3) {
      imputed_data <- complete(imputation_model_mimic, action = i)
      file_name <- paste0("mimiciii_mice_imputed_", i, ".csv")
      write.csv(imputed_data, file_name, row.names = FALSE)
      cat("  Saved:", file_name, "\n")
    }
    
    # --- B. Mean Imputation ---
    cat("  Running Mean Imputation...\n")
    imputed_mimic_mean <- mean_impute(mimiciii)
    write.csv(imputed_mimic_mean, "mimiciii_mean_imputed.csv", row.names = FALSE)
    cat("  Saved: mimiciii_mean_imputed.csv\n")
    
} else {
    cat("  Warning: MIMIC file not found:", mimic_file, "\n")
}

# ==============================================================================
# 2. MYOCARDIAL INFARCTION (MI) DATASET
# ==============================================================================

cat("\nProcessing Myocardial Infarction Dataset...\n")
# Note: Filename taken from folder structure `cleaned.mi (myocardial infarction).csv`
# (Original script had `cleaned.mi1...` which seemed like a typo or versioning)
mi_file <- "cleaned.mi (myocardial infarction).csv" 

if(file.exists(mi_file)) {
    myocardial <- read.csv(mi_file)
    
    # --- A. MICE Imputation ---
    cat("  Running MICE (m=3)...\n")
    imputation_model_mi <- mice(data = myocardial, 
                                method = "pmm",
                                m = 3,    
                                maxit = 50, 
                                print = FALSE)
    
    for (i in 1:3) {
      imputed_data <- complete(imputation_model_mi, action = i)
      file_name <- paste0("myocardial_mice_imputed_", i, ".csv")
      write.csv(imputed_data, file_name, row.names = FALSE)
      cat("  Saved:", file_name, "\n")
    }
    
    # --- B. Mean Imputation ---
    cat("  Running Mean Imputation...\n")
    imputed_mi_mean <- mean_impute(myocardial)
    write.csv(imputed_mi_mean, "myocardial_mean_imputed.csv", row.names = FALSE)
    cat("  Saved: myocardial_mean_imputed.csv\n")
    
} else {
     cat("  Warning: MI file not found:", mi_file, "\n")
}

cat("\nImputation tasks completed.\n")
