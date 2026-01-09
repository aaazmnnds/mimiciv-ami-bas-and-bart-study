################################################################################
# EVALUATE VARIABLE SELECTION (Fold-by-Fold)
#
# Metrics:
# 1. Sensitivity (Recall): TP / n_true_vars
# 2. Precision: TP / (TP + FP)
# 3. F1 Score: 2 * (Prec * Sens) / (Prec + Sens)
#
# Process:
# - Calculates metrics for EACH fold individually.
# - Reports Mean +/- SD across 10 folds.
#
################################################################################

library(dplyr)
library(ggplot2)

load("evaluation_config_4VAR.RData")

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")
N_FOLDS <- 10

cat("\n================================================================================\n")
cat("VARIABLE SELECTION EVALUATION (Fold-by-Fold)\n")
cat("================================================================================\n\n")

# ============================================================================
# FUNCTION: Calculate Selection Metrics for One Combination
# ============================================================================

calculate_selection_all_folds <- function(dataset, mechanism, method, mi_condition) {
  
  key <- paste0(dataset, "_", mechanism)
  true_vars <- true_variables[[key]]
  n_true_vars <- length(true_vars)
  
  if (is.null(true_vars)) return(NULL)
  
  # Determine Total Variables (P) by reading header of a data file
  # (Only need to do this once per Dataset/Mechanism really, but doing it here is safe)
  data_file <- paste0("complete_dataset_", dataset, "_", mechanism, ".csv")
  if (!file.exists(data_file)) {
    # Try alternate naming or skip type 1 error calc
    # Assuming standard names from simulation script
    p_total <- NA
  } else {
    # Read headers only
    headers <- names(read.csv(data_file, nrows = 1))
    # Exclude Y, missing indicators, total_missing
    # Pattern: Not Y, not ends with _missing, not total_missing
    # We need to know Response Var Name?
    # Evaluation config doesn't store response name, but we can guess or specificy
    resp_var <- if(dataset == "MIMIC") "ICD9_CODE" else "ZSN"
    
    predictor_cols <- headers[!grepl("_missing|total_missing", headers) & headers != resp_var]
    p_total <- length(predictor_cols)
  }
  
  # File name
  if (method == "MICE") {
    file_name <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_selected_variables.csv")
  } else {
    file_name <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_selected_variables.csv")
  }
  
  if (!file.exists(file_name)) return(NULL)
  
  selected_data <- read.csv(file_name)
  
  fold_results <- list()
  
  for (f in 1:N_FOLDS) {
    # Get vars selected in this fold
    selected_in_fold <- selected_data %>% 
      filter(fold == f) %>% 
      pull(variable)
    
    # Exclude indicator columns from "selected variables" count if present
    selected_in_fold <- selected_in_fold[!grepl("_missing|total_missing", selected_in_fold)]
    
    TP <- sum(true_vars %in% selected_in_fold)
    FP <- sum(!(selected_in_fold %in% true_vars))
    FN <- n_true_vars - TP
    
    # Type I Error Rate = FP / (Total Negatives) = FP / (P - True)
    if (!is.na(p_total)) {
        type1_err <- FP / (p_total - n_true_vars)
        type1_err_pct <- type1_err * 100
    } else {
        type1_err_pct <- NA
    }
    
    sens <- TP / n_true_vars
    prec <- if ((TP + FP) > 0) TP / (TP + FP) else 0
    f1 <- if ((prec + sens) > 0) 2 * (prec * sens) / (prec + sens) else 0
    
    fold_results[[f]] <- data.frame(
      dataset = dataset,
      mechanism = mechanism,
      method = method,
      mi_condition = mi_condition,
      fold = f,
      TP = TP, FP = FP, FN = FN,
      sensitivity = sens,
      precision = prec,
      f1_score = f1,
      type1_error_pct = type1_err_pct
    )
  }
  
  return(bind_rows(fold_results))
}

# ============================================================================
# 1. PROCESS ALL COMBINATIONS
# ============================================================================

all_fold_res <- list()
counter <- 1

cat("Processing combinations...\n")

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- calculate_selection_all_folds(d, m, meth, mi)
        if (!is.null(res)) {
          all_fold_res[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

fold_data <- bind_rows(all_fold_res)
cat(sprintf("✓ Calculated metrics for %d total fold-outcomes.\n\n", nrow(fold_data)))

# ============================================================================
# 2. SUMMARY STATISTICS (Mean +/- SD)
# ============================================================================

summary_stats <- fold_data %>%
  group_by(dataset, mechanism, method, mi_condition) %>%
  summarise(
    mean_sens = mean(sensitivity, na.rm = TRUE), sd_sens = sd(sensitivity, na.rm = TRUE),
    mean_prec = mean(precision, na.rm = TRUE),   sd_prec = sd(precision, na.rm = TRUE),
    mean_f1 = mean(f1_score, na.rm = TRUE),      sd_f1 = sd(f1_score, na.rm = TRUE),
    # Type I Error
    type1_error_pct_mean = mean(type1_error_pct, na.rm = TRUE),
    type1_error_pct_sd = sd(type1_error_pct, na.rm = TRUE),
    FP_mean = mean(FP, na.rm = TRUE),
    sensitivity = mean(sensitivity, na.rm = TRUE), # Redundant but kept for compatibility
    .groups = "drop"
  )

# ============================================================================
# 3. SAVE OUTPUTS
# ============================================================================

write.csv(fold_data, "VARIABLE_SELECTION_fold_by_fold.csv", row.names = FALSE)
write.csv(summary_stats, "VARIABLE_SELECTION_summary_with_SD.csv", row.names = FALSE)
write.csv(summary_stats, "TYPE1_ERROR_summary.csv", row.names = FALSE)

cat("Saved: VARIABLE_SELECTION_fold_by_fold.csv\n")
cat("Saved: VARIABLE_SELECTION_summary_with_SD.csv\n")
cat("Saved: TYPE1_ERROR_summary.csv\n")

# ============================================================================
# 4. PLOTTING
# ============================================================================

cat("Generating plots...\n")

# Plot 1: Mean F1 +/- SD
p1 <- ggplot(summary_stats, aes(x = method, y = mean_f1, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = pmax(0, mean_f1 - sd_f1), ymax = pmin(1, mean_f1 + sd_f1)),
                position = position_dodge(0.9), width = 0.25) +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "Variable Selection F1 Score (Mean ± SD)", x = NULL, y = "F1 Score") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("VARSEL_PLOT_mean_f1.png", p1, width = 10, height = 8)

# Plot 2: Boxplot of Sensitivity Distribution
# (Shows variability across folds more clearly than just SD)
p2 <- ggplot(fold_data, aes(x = method, y = sensitivity, fill = mi_condition)) +
  geom_boxplot() +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "Sensitivity Distribution across 10 Folds", x = NULL, y = "Sensitivity") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("VARSEL_PLOT_sensitivity_boxplot.png", p2, width = 10, height = 8)

cat("Saved plots.\n")
cat("\nDONE.\n")
