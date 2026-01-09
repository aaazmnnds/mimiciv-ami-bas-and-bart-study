################################################################################
# CREATE ALL SUMMARY TABLES
#
# Generates:
# 1. TABLE1_classification_performance.csv (AUC, F1, Type I Error, Sensitivity)
# 2. TABLE2_bias_analysis.csv (Bias, MSE, RMSE)
# 3. TABLE3_predictive_performance.csv (Prediction detailed metrics)
# 4. TABLE4_type1_error.csv (Type I error detailed analysis)
# 5. BEST_PERFORMERS_summary.csv
# 6. MASTER_SUMMARY_TABLE.xlsx (All combined)
#
################################################################################

library(dplyr)
library(openxlsx)

cat("\n", rep("=", 80), "\n", sep = "")
cat("CREATING MASTER SUMMARY TABLES AND EXCEL REPORT\n")
cat(rep("=", 80), "\n\n", sep = "")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

load_csv <- function(filename) {
  if (file.exists(filename)) {
    cat(sprintf("  ✓ Loaded: %s\n", filename))
    return(read.csv(filename))
  } else {
    cat(sprintf("  ⚠️  File not found: %s\n", filename))
    return(NULL)
  }
}

bias_data   <- load_csv("BIAS_SUMMARY_by_combination.csv")
varsel_data <- load_csv("VARIABLE_SELECTION_summary_with_SD.csv")
type1_data  <- load_csv("TYPE1_ERROR_summary.csv") # Often same as varsel_data
pred_data   <- load_csv("PREDICTION_PERFORMANCE_metrics.csv")

cat("\n")

# ============================================================================
# 2. PREPARE MASTER TABLE
# ============================================================================

# Initialize with Bias Data structure or create from scratch if missing
if (!is.null(bias_data)) {
  master <- bias_data %>% select(dataset, mechanism, method, mi_condition, n_vars, mean_bias, mse, rmse)
} else {
  # Fallback skeleton
  master <- expand.grid(
    dataset = c("MIMIC", "MI"),
    mechanism = c("MCAR", "MAR", "MNAR"),
    method = c("MICE", "MEAN", "missForest", "KNN"),
    mi_condition = c("noMI", "wMI"),
    stringsAsFactors = FALSE
  )
}

# Join VarSel Data
if (!is.null(varsel_data)) {
  master <- master %>% left_join(varsel_data, by = c("dataset", "mechanism", "method", "mi_condition"))
}

# Join Prediction Data
if (!is.null(pred_data)) {
  pred_subset <- pred_data %>% select(dataset, mechanism, method, mi_condition, auc, f1_pred = f1_score, avg_log_lik = avg_log_likelihood)
  master <- master %>% left_join(pred_subset, by = c("dataset", "mechanism", "method", "mi_condition"))
}

# Join Type I Error (if separate file, otherwise it's in varsel_data)
if (!is.null(type1_data) && !("type1_error_pct_mean" %in% names(master))) {
   type1_subset <- type1_data %>% select(dataset, mechanism, method, mi_condition, type1_error_pct_mean, type1_error_pct_sd)
   master <- master %>% left_join(type1_subset, by = c("dataset", "mechanism", "method", "mi_condition"))
}

# Format Missing Indicator column
master$missing_indicator <- ifelse(master$mi_condition == "wMI", "Yes", "No")

# ============================================================================
# 3. GENERATE TABLE 1: CLASSIFICATION PERFORMANCE
# ============================================================================

cat("Generating Table 1...\n")

table1 <- master %>%
  mutate(
    sens_fmt = sprintf("%.3f (%.3f)", mean_sens, sd_sens),
    prec_fmt = sprintf("%.3f (%.3f)", mean_prec, sd_prec),
    f1_sel_fmt = sprintf("%.3f (%.3f)", mean_f1, sd_f1),
    type1_fmt = ifelse(is.na(type1_error_pct_mean), "NA", sprintf("%.2f%% (%.2f%%)", type1_error_pct_mean, type1_error_pct_sd))
  ) %>%
  select(
    Dataset = dataset,
    Mechanism = mechanism,
    Method = method,
    MI = missing_indicator,
    AUC = auc,
    `F1 (Pred)` = f1_pred,
    `Log-Likelihood` = avg_log_lik,
    `Sensitivity (SD)` = sens_fmt,
    `Precision (SD)` = prec_fmt,
    `F1 VarSel (SD)` = f1_sel_fmt,
    `Type I Error` = type1_fmt
  )

write.csv(table1, "TABLE1_classification_performance.csv", row.names = FALSE)

# ============================================================================
# 4. GENERATE TABLE 2: BIAS ANALYSIS
# ============================================================================

cat("Generating Table 2...\n")

if (!is.null(bias_data)) {
  table2 <- master %>%
    select(dataset, mechanism, method, missing_indicator, mean_bias, mse, rmse, relative_bias_pct, relative_mse_pct) %>%
    mutate(across(where(is.numeric), ~ round(., 4))) %>%
    arrange(dataset, mechanism, rmse)
  
  write.csv(table2, "TABLE2_bias_analysis.csv", row.names = FALSE)
}

# ============================================================================
# 5. GENERATE TABLE 3: PREDICTION PERFORMANCE
# ============================================================================

cat("Generating Table 3...\n")

if (!is.null(pred_data)) {
  table3 <- master %>%
    select(dataset, mechanism, method, missing_indicator, auc, f1_pred, avg_log_lik) %>%
    mutate(across(where(is.numeric), ~ round(., 4))) %>%
    arrange(dataset, mechanism, desc(auc))
  
  write.csv(table3, "TABLE3_predictive_performance.csv", row.names = FALSE)
}

# ============================================================================
# 6. GENERATE TABLE 4: TYPE I ERROR
# ============================================================================

cat("Generating Table 4...\n")

if ("type1_error_pct_mean" %in% names(master)) {
  table4 <- master %>%
    select(dataset, mechanism, method, missing_indicator, type1_error_pct_mean, type1_error_pct_sd) %>%
    mutate(
      Status = case_when(
        type1_error_pct_mean < 5 ~ "Good",
        type1_error_pct_mean < 10 ~ "Moderate",
        TRUE ~ "High"
      )
    ) %>%
    mutate(across(where(is.numeric), ~ round(., 2))) %>%
    arrange(dataset, mechanism, type1_error_pct_mean)
  
  write.csv(table4, "TABLE4_type1_error.csv", row.names = FALSE)
}

# ============================================================================
# 7. GENERATE EXCEL REPORT
# ============================================================================

cat("Generating Excel Report...\n")

wb <- createWorkbook()

addWorksheet(wb, "Table1_Classification")
writeData(wb, "Table1_Classification", table1)

if (exists("table2")) {
  addWorksheet(wb, "Table2_Bias")
  writeData(wb, "Table2_Bias", table2)
}

if (exists("table3")) {
  addWorksheet(wb, "Table3_Prediction")
  writeData(wb, "Table3_Prediction", table3)
}

if (exists("table4")) {
  addWorksheet(wb, "Table4_Type1Error")
  writeData(wb, "Table4_Type1Error", table4)
}

# Add Raw Data sheet for reference
addWorksheet(wb, "Master_Raw_Data")
writeData(wb, "Master_Raw_Data", master)

# Style
headerStyle <- createStyle(textDecoration = "bold", fgFill = "#4F81BD", fontColour = "white")
for (sheet in names(wb)) {
  addStyle(wb, sheet, headerStyle, rows = 1, cols = 1:ncol(read.xlsx(wb, sheet = sheet)), gridExpand = TRUE)
  setColWidths(wb, sheet, cols = 1:ncol(read.xlsx(wb, sheet = sheet)), widths = "auto")
}

saveWorkbook(wb, "MASTER_SUMMARY_TABLE.xlsx", overwrite = TRUE)
cat("  ✓ Saved MASTER_SUMMARY_TABLE.xlsx\n")

# ============================================================================
# 8. BEST PERFORMERS
# ============================================================================

cat("Identifying Best Performers...\n")

best_performers <- master %>%
  group_by(dataset, mechanism) %>%
  summarise(
    Best_AUC_Method = method[which.max(auc)],
    Best_AUC_Value = max(auc, na.rm=TRUE),
    
    Best_MSE_Method = method[which.min(mse)],
    Best_MSE_Value = min(mse, na.rm=TRUE),
    
    Lowest_Type1_Method = method[which.min(type1_error_pct_mean)],
    Lowest_Type1_Value = min(type1_error_pct_mean, na.rm=TRUE),
    .groups = "drop"
  )

write.csv(best_performers, "BEST_PERFORMERS_summary.csv", row.names = FALSE)
cat("  ✓ Saved BEST_PERFORMERS_summary.csv\n")

cat("\nDONE.\n")
