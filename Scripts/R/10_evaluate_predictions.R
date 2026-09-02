.libPaths("~/R/library")
# EVALUATE PREDICTIONS - Monte Carlo Version
# Reads predictions from simulation output files (mc_rep level)
# Metrics: AUC, F1, ALPP

library(pROC)

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")
SIM_DIR <- "Results/CORRECTED/SIMULATION/"
OUT_DIR <- "Results/"

cat("PREDICTION EVALUATION (Monte Carlo)\n\n")

all_results <- list()

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        
        full_name <- if (meth == "MICE") {
          paste0(d, "_", m, "_MICE_", mi, "_POOLED")
        } else {
          paste0(d, "_", m, "_", meth, "_", mi)
        }
        
        pred_file <- paste0(SIM_DIR, full_name, "_predictions.csv")
        if (!file.exists(pred_file)) next
        
        pred_data <- read.csv(pred_file)
        if (!"mc_rep" %in% names(pred_data)) next
        
        pred_col <- if ("predicted_prob_pooled" %in% names(pred_data)) "predicted_prob_pooled" else "predicted_prob"
        if (!pred_col %in% names(pred_data)) next
        
        # Use only first 50 MC replications per sensei guidance
        pred_data <- pred_data[pred_data$mc_rep <= 50, ]
        mc_reps <- unique(pred_data$mc_rep)
        
        rep_results <- lapply(mc_reps, function(r) {
          rep_data <- pred_data[pred_data$mc_rep == r, ]
          folds <- unique(rep_data$fold)
          
          fold_res <- lapply(folds, function(f) {
            df_f <- rep_data[rep_data$fold == f, ]
            prob <- df_f[[pred_col]]
            prob <- pmax(pmin(prob, 1 - 1e-10), 1e-10)
            true_label <- df_f$true_label
            
            auc_val <- tryCatch(as.numeric(pROC::auc(pROC::roc(true_label, prob, quiet=TRUE))), error=function(e) NA)
            
            pred_label <- ifelse(prob >= 0.5, 1, 0)
            tp <- sum(pred_label == 1 & true_label == 1)
            fp <- sum(pred_label == 1 & true_label == 0)
            fn <- sum(pred_label == 0 & true_label == 1)
            
            prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
            rec  <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
            f1_val <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
            
            log_probs <- log(ifelse(true_label == 1, prob, 1 - prob))
            alpp_val <- mean(log_probs, na.rm=TRUE)
            
            data.frame(mc_rep=r, fold=f, AUC=auc_val, F1=f1_val, ALPP=alpp_val)
          })
          
          fold_df <- do.call(rbind, fold_res)
          data.frame(
            mc_rep = r,
            AUC = mean(fold_df$AUC, na.rm=TRUE),
            F1 = mean(fold_df$F1, na.rm=TRUE),
            ALPP = mean(fold_df$ALPP, na.rm=TRUE)
          )
        })
        
        rep_df <- do.call(rbind, rep_results)
        if (is.null(rep_df) || nrow(rep_df) == 0) next
        
        final <- data.frame(
          dataset = d,
          mechanism = m,
          method = meth,
          mi = mi,
          AUC_mean = mean(rep_df$AUC, na.rm=TRUE),
          AUC_sd = sd(rep_df$AUC, na.rm=TRUE),
          F1_mean = mean(rep_df$F1, na.rm=TRUE),
          F1_sd = sd(rep_df$F1, na.rm=TRUE),
          ALPP_mean = mean(rep_df$ALPP, na.rm=TRUE),
          ALPP_sd = sd(rep_df$ALPP, na.rm=TRUE),
          n_reps = nrow(rep_df)
        )
        
        all_results[[length(all_results) + 1]] <- final
        cat(sprintf("Done: %s\n", full_name))
      }
    }
  }
}

results_df <- do.call(rbind, all_results)
write.csv(results_df, paste0(OUT_DIR, "SIM_PREDICTION_MC_summary.csv"), row.names=FALSE)
cat("\nSaved: SIM_PREDICTION_MC_summary.csv\n")
cat("DONE.\n")
