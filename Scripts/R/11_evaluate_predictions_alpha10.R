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

library(dplyr)
library(ggplot2)
library(pROC)

MODELS <- c("BAS", "BART")
METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")

cat("\n================================================================================\n")
cat("PREDICTION PERFORMANCE EVALUATION (ALPHA10)\n")
cat("================================================================================\n\n")

# FUNCTION: Calculate metrics for One Combination

calculate_prediction_metrics <- function(model, dataset, mechanism, method, mi_condition) {
  
  # File Names
  if (model == "BAS") {
    if (method == "MICE") {
      pred_file <- paste0("Results/CORRECTED/ALPHA10/", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_predictions.csv")
      log_file  <- paste0("Results/CORRECTED/ALPHA10/", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_log_probabilities.csv")
    } else {
      pred_file <- paste0("Results/CORRECTED/ALPHA10/", dataset, "_", mechanism, "_ALPHA10_", method, "_", mi_condition, "_predictions.csv")
      log_file  <- paste0("Results/CORRECTED/ALPHA10/", dataset, "_", mechanism, "_ALPHA10_", method, "_", mi_condition, "_log_probabilities.csv")
    }
  } else if (model == "BART") {
    if (method == "MICE") {
      pred_file <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_predictions_m3.csv")
      log_file  <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_log_probabilities_m3.csv")
      # Fallback to _m5 if _m3 not found
      if (!file.exists(pred_file) && file.exists(paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_predictions_m5.csv"))) {
        pred_file <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_predictions_m5.csv")
        log_file  <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_MICE_", mi_condition, "_POOLED_log_probabilities_m5.csv")
      }
    } else {
      pred_file <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_", method, "_", mi_condition, "_predictions_m1.csv")
      log_file  <- paste0("Results/CORRECTED/ALPHA10/results_BART_", dataset, "_", mechanism, "_ALPHA10_", method, "_", mi_condition, "_log_probabilities_m1.csv")
    }
  }
  
  if (!file.exists(pred_file) || !file.exists(log_file)) return(NULL)
  
  pred_data <- read.csv(pred_file)
  log_data  <- read.csv(log_file)
  
  # Detect Prediction and Label Columns
  pred_col <- if ("predicted_prob_pooled" %in% names(pred_data)) "predicted_prob_pooled" else "predicted_prob"
  if (!("true_label" %in% names(pred_data)) && ("y_true" %in% names(pred_data))) {
    pred_data$true_label <- pred_data$y_true
  }
  
  # 1. Select Best Model Size (num_top) based on max log-likelihood
  log_col <- if ("log_prob_pooled" %in% names(log_data)) {
    "log_prob_pooled"
  } else if ("avg_log_prob" %in% names(log_data)) {
    "avg_log_prob"
  } else if ("avg_logp" %in% names(log_data)) {
    "avg_logp"
  } else {
    NULL
  }
  
  if (!is.null(log_col) && "num_top" %in% names(log_data) && nrow(log_data) > 0) {
    best_idx <- which.max(log_data[[log_col]])
    best_num_top <- log_data$num_top[best_idx]
    best_log_lik <- log_data[[log_col]][best_idx]
  } else {
    best_num_top <- 4
    best_log_lik <- NA
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
  
  # 3. Calculate Metrics Per Fold
  if (!("fold" %in% names(pred_subset))) {
    pred_subset$fold <- 1
  }
  
  fold_metrics <- pred_subset %>%
    group_by(fold) %>%
    summarise(
      auc_val = {
        if(length(unique(true_label)) > 1) {
            roc_obj <- suppressMessages(roc(true_label, .data[[pred_col]], quiet=TRUE))
            as.numeric(auc(roc_obj))
        } else { NA }
      },
      f1_val = {
        pred_class <- ifelse(.data[[pred_col]] > 0.5, 1, 0)
        TP <- sum(pred_class == 1 & true_label == 1)
        FP <- sum(pred_class == 1 & true_label == 0)
        FN <- sum(pred_class == 0 & true_label == 1)
        prec <- if((TP+FP)>0) TP/(TP+FP) else 0
        rec  <- if((TP+FN)>0) TP/(TP+FN) else 0
        if((prec+rec)>0) 2*(prec*rec)/(prec+rec) else 0
      },
      accuracy = {
        pred_class <- ifelse(.data[[pred_col]] > 0.5, 1, 0)
        sum(pred_class == true_label) / length(true_label)
      },
      precision_val = {
        pred_class <- ifelse(.data[[pred_col]] > 0.5, 1, 0)
        TP <- sum(pred_class == 1 & true_label == 1)
        FP <- sum(pred_class == 1 & true_label == 0)
        if((TP+FP)>0) TP/(TP+FP) else 0
      },
      recall_val = {
        pred_class <- ifelse(.data[[pred_col]] > 0.5, 1, 0)
        TP <- sum(pred_class == 1 & true_label == 1)
        FN <- sum(pred_class == 0 & true_label == 1)
        if((TP+FN)>0) TP/(TP+FN) else 0
      },
      alpp_val = {
        true_probs <- ifelse(true_label == 1, .data[[pred_col]], 1 - .data[[pred_col]])
        true_probs[true_probs < 1e-10] <- 1e-10
        mean(log(true_probs))
      },
      .groups="drop"
    )
    
  auc_mean <- mean(fold_metrics$auc_val, na.rm=TRUE)
  auc_sd   <- sd(fold_metrics$auc_val, na.rm=TRUE)
  f1_mean  <- mean(fold_metrics$f1_val, na.rm=TRUE)
  f1_sd    <- sd(fold_metrics$f1_val, na.rm=TRUE)
  alpp_mean <- mean(fold_metrics$alpp_val, na.rm=TRUE)
  alpp_sd  <- sd(fold_metrics$alpp_val, na.rm=TRUE)
  
  acc_mean <- mean(fold_metrics$accuracy, na.rm=TRUE)
  prec_mean <- mean(fold_metrics$precision_val, na.rm=TRUE)
  rec_mean <- mean(fold_metrics$recall_val, na.rm=TRUE)
  
  auc_str  <- sprintf("%.3f (%.3f)", auc_mean, if(is.na(auc_sd)) 0 else auc_sd)
  f1_str   <- sprintf("%.3f (%.3f)", f1_mean, if(is.na(f1_sd)) 0 else f1_sd)
  alpp_str <- sprintf("%.3f (%.3f)", alpp_mean, if(is.na(alpp_sd)) 0 else alpp_sd)
  
  return(data.frame(
      model = model,
      dataset = dataset,
      mechanism = mechanism,
      method = method,
      mi_condition = mi_condition,
      best_num_top = best_num_top,
      auc = auc_mean,
      auc_sd = auc_sd,
      auc_formatted = auc_str,
      f1_score = f1_mean,
      f1_sd = f1_sd,
      f1_formatted = f1_str,
      accuracy = acc_mean,
      precision = prec_mean,
      recall = rec_mean,
      avg_log_likelihood = alpp_mean,
      alpp_sd = alpp_sd,
      alpp_formatted = alpp_str
  ))
}

# 1. PROCESS ALL COMBINATIONS

all_res <- list()
counter <- 1

cat("Processing combinations...\n")

for (mod in MODELS) {
  for (d in DATASETS) {
    for (m in MECHANISMS) {
      for (meth in METHODS) {
        for (mi in MI_CONDITIONS) {
          res <- calculate_prediction_metrics(mod, d, m, meth, mi)
          if (!is.null(res)) {
            all_res[[counter]] <- res
            counter <- counter + 1
          }
        }
      }
    }
  }
}

pred_results <- bind_rows(all_res)
cat(sprintf(" Calculated metrics for %d models.\n\n", nrow(pred_results)))

# 3. SAVE OUTPUTS

write.csv(pred_results, "Results/CORRECTED/ALPHA10/PREDICTION_PERFORMANCE_metrics_alpha10.csv", row.names = FALSE)
cat("Saved: PREDICTION_PERFORMANCE_metrics.csv\n")

# 4. PLOTTING

cat("Generating plots...\n")

# Plot 1: AUC
p1 <- ggplot(pred_results, aes(x = method, y = auc, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "AUC by Method", x = NULL, y = "AUC") +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw(base_size = 18) + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    axis.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12)
  )

ggsave("Results/CORRECTED/ALPHA10/PRED_PLOT_auc_alpha10.png", p1, width = 10, height = 8)

# Plot 2: F1 Score
p2 <- ggplot(pred_results, aes(x = method, y = f1_score, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "F1 Score by Method", x = NULL, y = "F1 Score") +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw(base_size = 18) + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    axis.title = element_text(size = 16, face = "bold"),
    strip.text = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12)
  )

ggsave("Results/CORRECTED/ALPHA10/PRED_PLOT_f1_alpha10.png", p2, width = 10, height = 8)

cat("Saved plots.\n")
cat("\nDONE.\n")
