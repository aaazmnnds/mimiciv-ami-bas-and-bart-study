.libPaths("~/R/library")
# EVALUATE VARIABLE SELECTION - Monte Carlo Version
# Reads true_vars from simulation output files (mc_rep level)
# Metrics: Sensitivity, Precision, F1, Type I Error, Type II Error

library(dplyr)

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")
SIM_DIR <- "Results/CORRECTED/SIMULATION/"
OUT_DIR <- "Results/"

cat("VARIABLE SELECTION EVALUATION (Monte Carlo)\n\n")

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
        
        sel_file <- paste0(SIM_DIR, full_name, "_selected_variables.csv")
        if (!file.exists(sel_file)) next
        
        sel_data <- read.csv(sel_file)
        if (!"true_vars" %in% names(sel_data)) next
        
        # Use only first 50 MC replications per sensei guidance
        sel_data <- sel_data[sel_data$mc_rep <= 50, ]
        mc_reps <- unique(sel_data$mc_rep)
        
        # Get total number of non-indicator predictors from selected variables
        all_vars <- unique(sel_data$variable)
        all_vars <- all_vars[!grepl("_missing|total_missing", all_vars)]
        p_total <- length(all_vars)
        
        rep_results <- lapply(mc_reps, function(r) {
          rep_data <- sel_data[sel_data$mc_rep == r, ]
          
          # Get true vars for this rep
          true_vars <- unique(unlist(strsplit(rep_data$true_vars[1], ",")))
          true_vars <- trimws(true_vars)
          n_true <- length(true_vars)
          
          folds <- unique(rep_data$fold)
          
          fold_res <- lapply(folds, function(f) {
            fold_data <- rep_data[rep_data$fold == f, ]
            selected <- unique(fold_data$variable)
            selected <- selected[!grepl("_missing|total_missing", selected)]
            
            TP <- sum(true_vars %in% selected)
            FP <- sum(!(selected %in% true_vars))
            FN <- n_true - TP
            TN <- max(0, length(selected) - TP)
            
            sens <- TP / n_true
            prec <- if ((TP + FP) > 0) TP / (TP + FP) else 0
            f1 <- if ((prec + sens) > 0) 2 * prec * sens / (prec + sens) else 0
            type2 <- FN / n_true
            # Type I error: FP / total noise variables
            # Total variables = all unique variables selected across folds (approximation)
            type1 <- if ((p_total - n_true) > 0) FP / (p_total - n_true) else NA
            data.frame(mc_rep=r, fold=f, TP=TP, FP=FP, FN=FN,
                      sensitivity=sens, precision=prec, f1=f1, 
                      type2_error=type2, type1_error=type1)
          })
          bind_rows(fold_res)
        })
        
        rep_df <- bind_rows(rep_results)
        
        # Average across folds within each rep, then across reps
        rep_summary <- rep_df %>%
          group_by(mc_rep) %>%
          summarise(
            sens = mean(sensitivity, na.rm=TRUE),
            prec = mean(precision, na.rm=TRUE),
            f1 = mean(f1, na.rm=TRUE),
            type2 = mean(type2_error, na.rm=TRUE),
            type1 = mean(type1_error, na.rm=TRUE),
            fp_mean = mean(FP, na.rm=TRUE),
            .groups="drop"
          )
        
        final <- data.frame(
          dataset = d, mechanism = m, method = meth, mi = mi,
          sensitivity_mean = mean(rep_summary$sens),
          sensitivity_sd = sd(rep_summary$sens),
          precision_mean = mean(rep_summary$prec),
          precision_sd = sd(rep_summary$prec),
          f1_mean = mean(rep_summary$f1),
          f1_sd = sd(rep_summary$f1),
          type2_error_mean = mean(rep_summary$type2),
          type2_error_sd = sd(rep_summary$type2),
          type1_error_mean = mean(rep_summary$type1, na.rm=TRUE),
          type1_error_sd = sd(rep_summary$type1, na.rm=TRUE),
          n_reps = length(mc_reps)
        )
        
        all_results[[length(all_results)+1]] <- final
        cat(sprintf("Done: %s\n", full_name))
      }
    }
  }
}

results_df <- bind_rows(all_results)
write.csv(results_df, paste0(OUT_DIR, "VARIABLE_SELECTION_MC_summary.csv"), row.names=FALSE)
cat("\nSaved: VARIABLE_SELECTION_MC_summary.csv\n")
cat("DONE.\n")
