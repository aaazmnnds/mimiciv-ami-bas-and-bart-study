.libPaths("~/R/library")

library(dplyr)
library(pROC)

results_dir <- "Results/CORRECTED/"

# Helper to calculate metrics per fold
compute_metrics_per_fold <- function(df, pred_col) {
  folds <- unique(df$fold)
  fold_metrics <- lapply(folds, function(f) {
    df_f <- df[df$fold == f, ]
    prob <- df_f[[pred_col]]
    prob <- pmax(pmin(prob, 1 - 1e-10), 1e-10)
    true_label <- df_f$true_label
    auc_val <- tryCatch(as.numeric(pROC::auc(pROC::roc(true_label, prob, quiet=TRUE))), error=function(e) NA)
    pred_label <- ifelse(prob >= 0.5, 1, 0)
    tp <- sum(pred_label == 1 & true_label == 1)
    fp <- sum(pred_label == 1 & true_label == 0)
    fn <- sum(pred_label == 0 & true_label == 1)
    precision <- ifelse((tp+fp)==0, 0, tp/(tp+fp))
    recall <- ifelse((tp+fn)==0, 0, tp/(tp+fn))
    f1_val <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
    log_probs <- log(ifelse(true_label==1, prob, 1-prob))
    alpp_val <- mean(log_probs, na.rm=TRUE)
    data.frame(fold=f, AUC=auc_val, F1=f1_val, ALPP=alpp_val)
  })
  fold_df <- do.call(rbind, fold_metrics)
  data.frame(
    AUC_mean=mean(fold_df$AUC, na.rm=TRUE), AUC_sd=sd(fold_df$AUC, na.rm=TRUE),
    F1_mean=mean(fold_df$F1, na.rm=TRUE), F1_sd=sd(fold_df$F1, na.rm=TRUE),
    ALPP_mean=mean(fold_df$ALPP, na.rm=TRUE), ALPP_sd=sd(fold_df$ALPP, na.rm=TRUE)
  )
}

# Collect all files
all_files <- list.files(results_dir, pattern = "predictions", full.names = TRUE)

bas_files <- all_files[grepl("(MIMIC|MI)_REAL", all_files) & !grepl("results_BART", all_files)]
bart_files <- all_files[grepl("results_BART_(MIMIC|MI)_REAL", all_files)]

results_list <- list()

# Process BAS
for (f in bas_files) {
  df <- read.csv(f)
  pred_col <- if("predicted_prob_pooled" %in% names(df)) "predicted_prob_pooled" else "predicted_prob"
  
  if ("fold" %in% names(df) && "num_top" %in% names(df)) {
    df_best <- df %>% group_by(fold) %>% filter(num_top == num_top[which.max(get(pred_col))][1]) %>% ungroup()
  } else {
    df_best <- df
  }
  
  metrics <- compute_metrics_per_fold(df_best, pred_col)
  
  results_list[[length(results_list) + 1]] <- data.frame(
    model = "BAS",
    dataset = ifelse(grepl("MIMIC", f), "MIMIC", "MI"),
    method = ifelse(grepl("MICE", f), "MICE", ifelse(grepl("KNN", f), "KNN", ifelse(grepl("MEAN", f), "MEAN", "missForest"))),
    mi = ifelse(grepl("wMI", f), "wMI", "noMI"),
    AUC_mean = metrics$AUC_mean,
    AUC_sd = metrics$AUC_sd,
    F1_mean = metrics$F1_mean,
    F1_sd = metrics$F1_sd,
    ALPP_mean = metrics$ALPP_mean,
    ALPP_sd = metrics$ALPP_sd
  )
}

# Process BART
for (f in bart_files) {
  df <- read.csv(f)
  pred_col <- if("predicted_prob_pooled" %in% names(df)) "predicted_prob_pooled" else "predicted_prob"
  
  if ("fold" %in% names(df) && "num_top" %in% names(df)) {
    df_best <- df %>% group_by(fold) %>% filter(num_top == num_top[which.max(get(pred_col))][1]) %>% ungroup()
  } else {
    df_best <- df
  }
  
  metrics <- compute_metrics_per_fold(df_best, pred_col)
  
  results_list[[length(results_list) + 1]] <- data.frame(
    model = "BART",
    dataset = ifelse(grepl("MIMIC", f), "MIMIC", "MI"),
    method = ifelse(grepl("MICE", f), "MICE", ifelse(grepl("KNN", f), "KNN", ifelse(grepl("MEAN", f), "MEAN", "missForest"))),
    mi = ifelse(grepl("wMI", f), "wMI", "noMI"),
    AUC_mean = metrics$AUC_mean,
    AUC_sd = metrics$AUC_sd,
    F1_mean = metrics$F1_mean,
    F1_sd = metrics$F1_sd,
    ALPP_mean = metrics$ALPP_mean,
    ALPP_sd = metrics$ALPP_sd
  )
}

final_results <- bind_rows(results_list)

output_file <- file.path(results_dir, "real_data_performance_summary.csv")
write.csv(final_results, output_file, row.names = FALSE)
cat("Successfully wrote performance summary to:", output_file, "\n")
