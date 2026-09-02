.libPaths("~/R/library")
# EVALUATE BIAS - Monte Carlo Version
# Reads true_vars and beta_true from simulation beta_estimates files
# Metrics: Relative Bias (%), Relative MSE (%)

library(dplyr)

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")
SIM_DIR <- "Results/CORRECTED/SIMULATION/"
OUT_DIR <- "Results/"

# True beta values used in simulation (fixed across all reps)
BETA_TRUE <- c(1.5, 1.0, 0.5, 0.1)

cat("BIAS EVALUATION (Monte Carlo)\n\n")

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
        
        beta_file <- paste0(SIM_DIR, full_name, "_beta_estimates.csv")
        if (!file.exists(beta_file)) next
        
        beta_data <- read.csv(beta_file)
        if (!"true_vars" %in% names(beta_data)) next
        
        # Standardize beta column name
        if ("beta_pooled" %in% names(beta_data)) {
          beta_data$beta_hat <- beta_data$beta_pooled
        }
        if (!"beta_hat" %in% names(beta_data)) next
        
        # Use only first 50 MC replications per sensei guidance
        beta_data <- beta_data[beta_data$mc_rep <= 50, ]
        mc_reps <- unique(beta_data$mc_rep)
        
        rep_results <- lapply(mc_reps, function(r) {
          rep_data <- beta_data[beta_data$mc_rep == r, ]
          
          # Get true vars for this rep
          true_vars <- trimws(unlist(strsplit(as.character(rep_data$true_vars[1]), ",")))
          
          # Create true beta mapping (sorted by beta value descending)
          beta_mapping <- data.frame(
            variable = true_vars,
            beta_true = sort(BETA_TRUE, decreasing = TRUE),
            stringsAsFactors = FALSE
          )
          
          # Get beta estimates for true vars only using base R
          rep_true <- rep_data[rep_data$variable %in% true_vars, ]
          rep_true <- merge(rep_true, beta_mapping, by = "variable")
          
          if (nrow(rep_true) == 0) return(NULL)
          
          # Average beta_hat across folds and imputations using base R
          rep_avg <- aggregate(beta_hat ~ variable + beta_true, data=rep_true, FUN=mean)
          
          # Compute metrics
          rel_bias <- mean((rep_avg$beta_hat - rep_avg$beta_true) / rep_avg$beta_true) * 100
          rel_mse <- sum((rep_avg$beta_hat - rep_avg$beta_true)^2) / sum(rep_avg$beta_true^2) * 100
          
          data.frame(mc_rep=r, rel_bias=rel_bias, rel_mse=rel_mse)
        })
        
        rep_df <- bind_rows(rep_results)
        if (is.null(rep_df) || nrow(rep_df) == 0) next
        
        final <- data.frame(
          dataset = d, mechanism = m, method = meth, mi = mi,
          rel_bias_mean = mean(rep_df$rel_bias, na.rm=TRUE),
          rel_bias_sd = sd(rep_df$rel_bias, na.rm=TRUE),
          rel_mse_mean = mean(rep_df$rel_mse, na.rm=TRUE),
          rel_mse_sd = sd(rep_df$rel_mse, na.rm=TRUE),
          n_reps = nrow(rep_df)
        )
        
        all_results[[length(all_results)+1]] <- final
        cat(sprintf("Done: %s\n", full_name))
      }
    }
  }
}

results_df <- bind_rows(all_results)
write.csv(results_df, paste0(OUT_DIR, "BIAS_MC_summary.csv"), row.names=FALSE)
cat("\nSaved: BIAS_MC_summary.csv\n")
cat("DONE.\n")
