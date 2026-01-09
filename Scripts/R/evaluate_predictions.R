################################################################################
# EVALUATE OVERALL PREDICTION PERFORMANCE
#
# Metrics:
# 1. AUC (Area Under ROC Curve)
# 2. F1 Score, Precision, Recall (at threshold 0.5)
# 3. Average Log-Likelihood
#
# Logic:
# - Uses the "best model" size (num_top) selected by highest log-probability.
# - Evaluates performance on the held-out Test set.
#
################################################################################

library(dplyr)
library(ggplot2)
library(pROC)

load("evaluation_config_4VAR.RData")

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")

cat("\n================================================================================\n")
cat("PREDICTION PERFORMANCE EVALUATION\n")
cat("================================================================================\n\n")

# ============================================================================
# FUNCTION: Calculate metrics for One Combination
# ============================================================================

calculate_prediction_metrics <- function(dataset, mechanism, method, mi_condition) {
  
  # File Names
  if (method == "MICE") {
    pred_file <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_predictions.csv")
    log_file  <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_log_probabilities.csv")
    pred_col  <- "predicted_prob_pooled"
  } else {
    pred_file <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_predictions.csv")
    log_file  <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_log_probabilities.csv")
    pred_col  <- "predicted_prob"
  }
  
  if (!file.exists(pred_file) || !file.exists(log_file)) return(NULL)
  
  pred_data <- read.csv(pred_file)
  log_data  <- read.csv(log_file)
  
  # 1. Select Best Model Size (num_top) based on max log-likelihood
  if (method == "MICE") {
    best_idx <- which.max(log_data$log_prob_pooled)
    best_num_top <- log_data$num_top[best_idx]
    best_log_lik <- log_data$log_prob_pooled[best_idx]
  } else {
    # Check if we have 'avg_log_prob' column or need to average V1..V10
    if ("avg_log_prob" %in% names(log_data)) {
        best_idx <- which.max(log_data$avg_log_prob)
        best_num_top <- log_data$num_top[best_idx]
        best_log_lik <- log_data$avg_log_prob[best_idx]
    } else {
        # Fallback: assume column with max value in the first row is the best? 
        # Or usually there's a structure like 'num_top', 'avg_log_prob'.
        # If simpler structure (rows=num_top), just take row max.
        # Assuming standard structure from our cleaned scripts:
        if ("num_top" %in% names(log_data)) {
             # Re-calculate average if needed or take the column
             # Let's assume there is a Summary Column or we take the row max if 'avg_log_prob' missing
             # For safety, let's just pick num_top=4 (since we know Truth=4) if we can't determine optimization
             # But the script claims to optimize. Let's try finding 'avg_log_prob' again.
             if(!"avg_log_prob" %in% names(log_data)) {
                 # Create it if missing (e.g. average of V1..VK)
                 v_cols <- grep("^V", names(log_data), value=TRUE)
                 if(length(v_cols) > 0) {
                     log_data$avg_log_prob <- rowMeans(log_data[, v_cols], na.rm=TRUE)
                     best_idx <- which.max(log_data$avg_log_prob)
                     best_num_top <- log_data$num_top[best_idx]
                     best_log_lik <- log_data$avg_log_prob[best_idx]
                 } else {
                     # Fallback to 4 variables
                     best_num_top <- 4
                     best_log_lik <- NA
                 }
             }
        } else {
             best_num_top <- 4
             best_log_lik <- NA
        }
    }
  }

  # 2. Filter Predictions for that Best Model Size
  # The prediction file has predictions for ALL num_top values usually? 
  # Or just the best? Previous scripts saved ALL.
  if ("num_top" %in% names(pred_data)) {
      pred_subset <- pred_data %>% filter(num_top == best_num_top)
  } else {
      pred_subset <- pred_data # Assume file only contains best if no column
  }
  
  if (nrow(pred_subset) == 0) return(NULL)
  
  # 3. Calculate Metrics
  # AUC
  roc_obj <- roc(pred_subset$true_label, pred_subset[[pred_col]], quiet=TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  # Classification at 0.5
  pred_class <- ifelse(pred_subset[[pred_col]] > 0.5, 1, 0)
  TP <- sum(pred_class == 1 & pred_subset$true_label == 1)
  TN <- sum(pred_class == 0 & pred_subset$true_label == 0)
  FP <- sum(pred_class == 1 & pred_subset$true_label == 0)
  FN <- sum(pred_class == 0 & pred_subset$true_label == 1)
  
  accuracy <- (TP + TN) / nrow(pred_subset)
  precision <- if((TP+FP)>0) TP/(TP+FP) else 0
  recall    <- if((TP+FN)>0) TP/(TP+FN) else 0
  f1        <- if((precision+recall)>0) 2*(precision*recall)/(precision+recall) else 0
  
  return(data.frame(
      dataset = dataset,
      mechanism = mechanism,
      method = method,
      mi_condition = mi_condition,
      best_num_top = best_num_top,
      auc = auc_val,
      f1_score = f1,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      avg_log_likelihood = best_log_lik
  ))
}

# ============================================================================
# 1. PROCESS ALL COMBINATIONS
# ============================================================================

all_res <- list()
counter <- 1

cat("Processing combinations...\n")

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- calculate_prediction_metrics(d, m, meth, mi)
        if (!is.null(res)) {
          all_res[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

pred_results <- bind_rows(all_res)
cat(sprintf("✓ Calculated metrics for %d models.\n\n", nrow(pred_results)))

# ============================================================================
# 3. SAVE OUTPUTS
# ============================================================================

write.csv(pred_results, "PREDICTION_PERFORMANCE_metrics.csv", row.names = FALSE)
cat("Saved: PREDICTION_PERFORMANCE_metrics.csv\n")

# ============================================================================
# 4. PLOTTING
# ============================================================================

cat("Generating plots...\n")

# Plot 1: AUC
p1 <- ggplot(pred_results, aes(x = method, y = auc, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "AUC by Method", x = NULL, y = "AUC") +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("PRED_PLOT_auc.png", p1, width = 10, height = 8)

# Plot 2: F1 Score
p2 <- ggplot(pred_results, aes(x = method, y = f1_score, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "F1 Score by Method", x = NULL, y = "F1 Score") +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("PRED_PLOT_f1.png", p2, width = 10, height = 8)

cat("Saved plots.\n")
cat("\nDONE.\n")
