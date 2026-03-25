# BAS EVALUATION SCRIPT (Scenario Analysis)
#
# Goal: Run BAS on full datasets for all scenarios and extract
# Posterior Inclusion Probabilities (PIP).

library(BAS)
library(dplyr)

set.seed(456)

# Configuration
SELECTION_METHODS <- c("top", "independent", "mixed")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")
ITER <- 5000 # Enough for snapshot

# Helper: Run BAS on one file
run_bas_snapshot <- function(data_file, y_col, true_vars, dataset_name) {
  if (!file.exists(data_file)) return(NULL)
  
  df <- read.csv(data_file)
  
  # Dataset-specific robust prior
  n_robust <- if(dataset_name == "MIMIC") 151 else 1529
  
  # BAS.glm
  formula_str <- paste(y_col, "~ .")
  model <- bas.glm(as.formula(formula_str), 
                   data = df,
                   method = "MCMC", 
                   MCMC.iterations = ITER,
                   betaprior = robust(n_robust), 
                   family = binomial(link = "logit"),
                   modelprior = beta.binomial(1, 1))
  
  summ <- summary(model)
  # Extract P(B != 0 | Y)
  prob_col <- "P(B != 0 | Y)"
  if(prob_col %in% colnames(summ)){
      probs <- summ[, prob_col]
  } else {
      probs <- summ[, 1] 
  }
  
  # Create a clean PIP dataframe
  pip_df <- data.frame(
    variable = names(probs),
    pip = as.numeric(probs)
  ) %>% filter(variable != "Intercept")
  
  # Label True vs False
  pip_df$is_true <- pip_df$variable %in% true_vars
  
  return(pip_df)
}

results_all <- list()

for (method in SELECTION_METHODS) {
  cat(sprintf("\nEvaluating Scenario: %s\n", method))
  for (dataset in DATASETS) {
    y_col <- if(dataset == "MIMIC") "ICD9_CODE" else "ZSN"
    for (mech in MECHANISMS) {
      cat(sprintf("  Dataset: %s | Mech: %s\n", dataset, mech))
      
      # Load True Vars
      truth_file <- paste0("../../Data/", method, "/", dataset, "_", mech, "_true_variables.csv")
      if(!file.exists(truth_file)) next
      true_vars_df <- read.csv(truth_file)
      true_vars <- true_vars_df$variable
      
      # Run for MEAN (wMI)
      cat("    Running MEAN (wMI)...\n")
      file_mean <- paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MEAN.csv")
      res_mean <- run_bas_snapshot(file_mean, y_col, true_vars, dataset)
      if(!is.null(res_mean)) {
        res_mean$method_scenario <- method
        res_mean$dataset <- dataset
        res_mean$mechanism <- mech
        res_mean$imputation <- "MEAN"
        results_all[[length(results_all) + 1]] <- res_mean
      }
      
      # Run for MICE (m=1 only for snapshot, wMI)
      cat("    Running MICE m=1 (wMI)...\n")
      file_mice <- paste0("../../Data/", method, "/imputed_", dataset, "_", mech, "_MICE_1.csv")
      res_mice <- run_bas_snapshot(file_mice, y_col, true_vars, dataset)
      if(!is.null(res_mice)) {
        res_mice$method_scenario <- method
        res_mice$dataset <- dataset
        res_mice$mechanism <- mech
        res_mice$imputation <- "MICE"
        results_all[[length(results_all) + 1]] <- res_mice
      }
    }
  }
}

# Combine and Save
final_results <- bind_rows(results_all)
write.csv(final_results, "../../Results/SCENARIO_ANALYSIS_BAS_PIP_SNAPSHOT.csv", row.names = FALSE)

# Summarize Performance by Scenario
summary_perf <- final_results %>%
  group_by(method_scenario, dataset, mechanism, imputation) %>%
  summarise(
    avg_pip_true = mean(pip[is_true]),
    max_pip_false = max(pip[!is_true]),
    num_vars_pip_high = sum(pip > 0.5),
    num_true_pip_high = sum(pip > 0.5 & is_true),
    .groups = "drop"
  )

write.csv(summary_perf, "../../Results/SCENARIO_ANALYSIS_SUMMARY.csv", row.names = FALSE)

cat("\nEvaluation Complete. Results saved to Results/SCENARIO_ANALYSIS_SUMMARY.csv\n")
