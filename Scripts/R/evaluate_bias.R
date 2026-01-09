################################################################################
# EVALUATE BIAS
#
# Evaluates coefficient bias (estimates vs true betas).
# Computes both Absolute and Relative metrics (per Supervisor's Section 2.6.4).
#
# Metrics:
# 1. Mean Bias: mean(beta_hat - beta_true)
# 2. MSE: sum(bias^2) across 4 variables
# 3. Relative Bias (%): (1/4) * Sum[ (1/10) * Sum(Relative Error) ]
# 4. Relative MSE (%): ||beta_hat - beta_true||^2 / ||beta_true||^2
#
################################################################################

library(dplyr)
library(ggplot2)

# Load shared configuration
load("evaluation_config_4VAR.RData")

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")

cat("\n================================================================================\n")
cat("BIAS EVALUATION (Absolute & Relative)\n")
cat("================================================================================\n\n")

# ============================================================================
# FUNCTION: Calculate Bias for One Combination
# ============================================================================

calculate_bias_all_folds <- function(dataset, mechanism, method, mi_condition) {
  
  key <- paste0(dataset, "_", mechanism)
  true_vars <- true_variables[[key]]
  beta_true_vals <- beta_true_values[[key]]
  
  if (is.null(true_vars)) {
    cat(sprintf("⚠️  No true variables found for %s\n", key))
    return(NULL)
  }
  
  # Construct file name
  if (method == "MICE") {
    file_name <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_beta_estimates.csv")
  } else {
    file_name <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_beta_estimates.csv")
  }
  
  if (!file.exists(file_name)) {
    # cat("  (File not found:", file_name, ")\n") 
    # Quiet error to reduce spam if files are missing
    return(NULL)
  }
  
  beta_data <- read.csv(file_name)
  
  # Normalize MICE/Single column names
  if (method == "MICE") {
    beta_estimates <- beta_data %>%
      filter(variable %in% true_vars) %>%
      select(fold, variable, beta_hat = beta_pooled)
  } else {
    beta_estimates <- beta_data %>%
      filter(variable %in% true_vars) %>%
      select(fold, variable, beta_hat)
  }
  
  # Map true betas
  beta_mapping <- data.frame(variable = true_vars, beta_true = beta_true_vals)
  
  result <- beta_estimates %>%
    left_join(beta_mapping, by = "variable") %>%
    mutate(
      dataset = dataset,
      mechanism = mechanism,
      method = method,
      mi_condition = mi_condition,
      
      # Metrics
      bias = beta_hat - beta_true,
      abs_bias = abs(beta_hat - beta_true),
      squared_bias = (beta_hat - beta_true)^2,
      relative_error = (beta_hat - beta_true) / beta_true
    )
  
  return(result)
}

# ============================================================================
# 1. PROCESS ALL COMBINATIONS
# ============================================================================

all_results <- list()
counter <- 1

cat("Processing combinations...\n")

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- calculate_bias_all_folds(d, m, meth, mi)
        if (!is.null(res)) {
          all_results[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

beta_all <- bind_rows(all_results)
cat(sprintf("✓ Calculated bias for %d total estimates.\n\n", nrow(beta_all)))

# ============================================================================
# 2. COMPUTE SUMMARY STATISTICS
# ============================================================================

# A. Relative Bias (%) per Supervisor Formula
# (1/4) * Sum_j [ (1/10) * Sum_k (beta_hat - beta_true)/beta_true ] * 100
relative_bias_summary <- beta_all %>%
  group_by(dataset, mechanism, method, mi_condition, variable) %>%
  summarise(mean_rel_error_var = mean(relative_error, na.rm = TRUE), .groups = "drop") %>%
  group_by(dataset, mechanism, method, mi_condition) %>%
  summarise(relative_bias_pct = mean(mean_rel_error_var, na.rm = TRUE) * 100, .groups = "drop")

# B. Relative MSE (%) per Supervisor Formula
# ||beta_hat - beta_true||^2 / ||beta_true||^2 * 100
relative_mse_summary <- beta_all %>%
  group_by(dataset, mechanism, method, mi_condition) %>%
  summarise(
    sum_sq_error = sum(squared_bias, na.rm = TRUE),
    sum_sq_true = sum(beta_true^2, na.rm = TRUE),
    relative_mse_pct = (sum_sq_error / sum_sq_true) * 100,
    .groups = "drop"
  )

# C. Absolute Metrics
absolute_summary <- beta_all %>%
  group_by(dataset, mechanism, method, mi_condition) %>%
  summarise(
    mean_bias = mean(bias, na.rm = TRUE),
    mean_abs_bias = mean(abs_bias, na.rm = TRUE),
    mse = sum(squared_bias, na.rm = TRUE), # Sum of squared biases across all vars/folds
    rmse = sqrt(sum(squared_bias, na.rm = TRUE)),
    .groups = "drop"
  )

# Combine
final_summary <- absolute_summary %>%
  left_join(relative_bias_summary, by = c("dataset", "mechanism", "method", "mi_condition")) %>%
  left_join(relative_mse_summary %>% select(-sum_sq_error, -sum_sq_true), 
            by = c("dataset", "mechanism", "method", "mi_condition"))

# ============================================================================
# 3. SAVE OUTPUTS
# ============================================================================

write.csv(beta_all, "BIAS_DETAILED_all_variables.csv", row.names = FALSE)
write.csv(final_summary, "BIAS_SUMMARY_by_combination.csv", row.names = FALSE)
write.csv(final_summary %>% select(dataset, mechanism, method, mi_condition, relative_bias_pct, relative_mse_pct),
          "BIAS_RELATIVE_for_manuscript.csv", row.names = FALSE)

cat("Saved: BIAS_DETAILED_all_variables.csv\n")
cat("Saved: BIAS_SUMMARY_by_combination.csv\n")
cat("Saved: BIAS_RELATIVE_for_manuscript.csv\n")

# ============================================================================
# 4. PLOTTING
# ============================================================================

cat("Generating plots...\n")

# Plot 1: Relative Bias
p1 <- ggplot(final_summary, aes(x = method, y = relative_bias_pct, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ dataset + mechanism) +
  labs(title = "Relative Bias (%) by Method", x = NULL, y = "Relative Bias (%)") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("BIAS_PLOT_relative_bias.png", p1, width = 10, height = 8)

# Plot 2: Relative MSE
p2 <- ggplot(final_summary, aes(x = method, y = relative_mse_pct, fill = mi_condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ dataset + mechanism, scales = "free_y") +
  labs(title = "Relative MSE (%) by Method", x = NULL, y = "Relative MSE (%)") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("BIAS_PLOT_relative_mse.png", p2, width = 10, height = 8)

cat("Saved plots.\n")
cat("\nDONE.\n")
