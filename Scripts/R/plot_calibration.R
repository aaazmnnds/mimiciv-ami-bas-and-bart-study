################################################################################
# PLOT CALIBRATION CURVES
#
# Generates calibration plots for the "Best Model" (selected by num_top).
# Compares Observed vs Predicted probabilities.
#
# Logic:
# 1. Load predictions.
# 2. Bin predicted probabilities (e.g., 10 bins).
# 3. Calculate mean predicted probability and mean observed outcome rate per bin.
# 4. Plot Calibration Curve (Diagonal = Perfect Calibration).
#
################################################################################

library(dplyr)
library(ggplot2)
library(gridExtra)

load("evaluation_config_4VAR.RData")

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("noMI", "wMI")

cat("\n", rep("=", 80), "\n", sep = "")
cat("PLOTTING CALIBRATION CURVES\n")
cat(rep("=", 80), "\n\n", sep = "")

# ============================================================================
# FUNCTION: Extract Calibration Data
# ============================================================================

get_calibration_data <- function(dataset, mechanism, method, mi_condition) {
  
  # File Names (Predictions and LogProbs to find best num_top)
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
  
  # Find Best Num Top
  if (method == "MICE") {
    best_idx <- which.max(log_data$log_prob_pooled)
    best_num_top <- log_data$num_top[best_idx]
  } else {
    if ("avg_log_prob" %in% names(log_data)) {
        best_idx <- which.max(log_data$avg_log_prob)
        best_num_top <- log_data$num_top[best_idx]
    } else {
        # Fallback if structure differs
        best_num_top <- 4 
    }
  }
  
  # Filter Predictions
  pred_subset <- pred_data %>% 
    filter(num_top == best_num_top) %>%
    select(true_label, predicted_prob = all_of(pred_col))
  
  if (nrow(pred_subset) == 0) return(NULL)
  
  return(pred_subset %>% mutate(dataset=dataset, mechanism=mechanism, method=method, mi_condition=mi_condition))
}

# ============================================================================
# 1. AGGREGATE CALIBRATION DATA
# ============================================================================

cat("Extracting prediction data...\n")
all_preds <- list()
counter <- 1

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- get_calibration_data(d, m, meth, mi)
        if (!is.null(res)) {
          all_preds[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

combined_preds <- bind_rows(all_preds)
cat(sprintf("✓ Loaded %d prediction records.\n\n", nrow(combined_preds)))

# ============================================================================
# 2. CREATE CALIBRATION BINS
# ============================================================================

cat("Calculating calibration bins...\n")

# compute bins
calibration_summary <- combined_preds %>%
  mutate(bin = ntile(predicted_prob, 10)) %>% # Deciles
  group_by(dataset, mechanism, method, mi_condition, bin) %>%
  summarise(
    mean_predicted = mean(predicted_prob),
    observed_proportion = mean(true_label),
    n = n(),
    .groups = "drop"
  )

# ============================================================================
# 3. PLOT
# ============================================================================

cat("Creating plots...\n")

for (ds in DATASETS) {
  
  ds_data <- calibration_summary %>% filter(dataset == ds)
  
  # Plot: Compare Methods (averaged or separate facets)
  # Let's facet by Mechanism and Color by Method
  # Separate noMI and wMI
  
  for (mic in MI_CONDITIONS) {
    
    plot_data <- ds_data %>% filter(mi_condition == mic)
    
    p <- ggplot(plot_data, aes(x = mean_predicted, y = observed_proportion, color = method, group = method)) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
      geom_point(size = 2) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ mechanism) +
      labs(
        title = paste0("Calibration Plot: ", ds, " (", mic, ")"),
        x = "Mean Predicted Probability",
        y = "Observed Proportion"
      ) +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
      theme_bw() +
      theme(legend.position = "bottom")
    
    filename <- paste0("CALIBRATION_PLOT_", ds, "_", mic, ".png")
    ggsave(filename, p, width = 10, height = 6)
    cat(sprintf("  ✓ Saved: %s\n", filename))
  }
}

cat("\nDONE.\n")
