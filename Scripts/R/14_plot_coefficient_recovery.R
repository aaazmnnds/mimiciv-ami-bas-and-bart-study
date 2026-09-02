.libPaths(c("~/R/library", .libPaths()))

# PLOT BETA ESTIMATES
#
# Logic:
# - Visualizes Beta estimates (Mean +/- SD across Monte Carlo replications & folds).
# - Compares "wMI" (With Missing indicators) vs "noMI" (Without).
# - Reads true_vars directly from simulation CSV outputs.
# - Outputs high-resolution comparison plots directly into Submission_v2.0.
#

library(ggplot2)
library(gridExtra)
library(grid)

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("wMI", "noMI")
BETA_TRUE_VALUES <- c(0.1, 0.5, 1.0, 1.5)

output_dir <- "/Users/nazu.ds/Documents/Research Collections/Scientific_Reports_Submission_Package/For resubmission to another journal/Resubmission_Package/Submission_v2.0/"

cat("\n================================================================================\n")
cat("PLOTTING BETA ESTIMATES (Comparison: wMI vs noMI)\n")
cat("================================================================================\n\n")

# FUNCTION: Extract Beta Estimates for a Single Combination

extract_fold_betas <- function(dataset, mechanism, method, mi_condition) {
  
  if (method == "MICE") {
    file_name <- paste0("Results/CORRECTED/SIMULATION/", dataset, "_", mechanism, "_MICE_", mi_condition, "_POOLED_beta_estimates.csv")
  } else {
    file_name <- paste0("Results/CORRECTED/SIMULATION/", dataset, "_", mechanism, "_", method, "_", mi_condition, "_beta_estimates.csv")
  }
  
  if (!file.exists(file_name)) return(NULL)
  
  beta_data <- read.csv(file_name, stringsAsFactors = FALSE)
  
  # Filter to 50 Monte Carlo replications if mc_rep exists
  if ("mc_rep" %in% names(beta_data)) {
    beta_data <- beta_data[beta_data$mc_rep <= 50, ]
  }
  
  if (nrow(beta_data) == 0) return(NULL)
  
  # Determine true variables from the true_vars column (using mode across rows)
  if ("true_vars" %in% names(beta_data)) {
    mode_str <- names(sort(table(beta_data$true_vars), decreasing = TRUE))[1]
    true_vars <- trimws(strsplit(mode_str, ",")[[1]])
  } else {
    warning(paste("No true_vars column found in", file_name))
    return(NULL)
  }
  
  # Normalize beta estimate column
  if (method == "MICE") {
    beta_data$beta_hat <- beta_data$beta_pooled
  }
  
  # Filter rows for the 4 true variables
  subset_data <- beta_data[beta_data$variable %in% true_vars, ]
  if (nrow(subset_data) == 0) return(NULL)
  
  # Map true variables to beta true values (1.5, 1.0, 0.5, 0.1) in original order
  beta_mapping <- data.frame(
    variable = true_vars,
    beta_true = c(1.5, 1.0, 0.5, 0.1),
    stringsAsFactors = FALSE
  )
  
  result <- merge(subset_data, beta_mapping, by = "variable")
  result$dataset <- dataset
  result$mechanism <- mechanism
  result$method <- method
  result$mi_condition <- mi_condition
  
  keep_cols <- c("dataset", "mechanism", "method", "mi_condition", "variable", "beta_true", "beta_hat")
  return(result[, keep_cols])
}

# 1. EXTRACT ALL DATA

cat("Extracting simulation data...\n")
all_data_list <- list()
counter <- 1

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- extract_fold_betas(d, m, meth, mi)
        if (!is.null(res) && nrow(res) > 0) {
          all_data_list[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

if (length(all_data_list) == 0) {
  stop("No simulation data could be extracted. Please check input paths.")
}

all_fold_data <- do.call(rbind, all_data_list)
cat(sprintf(" Extracted %d estimates across combinations.\n\n", nrow(all_fold_data)))

# 2. AGGREGATE (MEAN +/- SD) USING BASE R

plot_data <- aggregate(
  beta_hat ~ dataset + mechanism + method + mi_condition + beta_true,
  data = all_fold_data,
  FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE))
)

# Unpack matrix columns from aggregate
plot_data_df <- data.frame(
  dataset = plot_data$dataset,
  mechanism = plot_data$mechanism,
  method = factor(plot_data$method, levels = c("KNN", "MEAN", "MICE", "missForest")),
  mi_condition = factor(plot_data$mi_condition, levels = c("wMI", "noMI")),
  beta_true = plot_data$beta_true,
  mean_beta = plot_data$beta_hat[, "mean"],
  sd_beta = plot_data$beta_hat[, "sd"],
  stringsAsFactors = FALSE
)

# 3. CREATE PLOTS

cat("Creating plots...\n")

for (ds in DATASETS) {
  cat(sprintf("  Plotting %s...\n", ds))
  
  ds_data <- plot_data_df[plot_data_df$dataset == ds, ]
  beta_values <- sort(unique(ds_data$beta_true))
  
  plots <- list()
  
  for (i in seq_along(beta_values)) {
    beta_val <- beta_values[i]
    sub_data <- ds_data[ds_data$beta_true == beta_val, ]
    
    # Calculate y-limits for better styling
    y_min <- min(sub_data$mean_beta - sub_data$sd_beta, na.rm = TRUE)
    y_max <- max(sub_data$mean_beta + sub_data$sd_beta, na.rm = TRUE)
    # Ensure true beta is included
    y_min <- min(y_min, beta_val)
    y_max <- max(y_max, beta_val)
    
    # Add padding
    rng <- y_max - y_min
    if (rng == 0) rng <- 1
    y_lims <- c(y_min - 0.1 * rng, y_max + 0.1 * rng)

    p <- ggplot(sub_data, aes(x = method, y = mean_beta, 
                              color = mechanism, shape = mechanism, linetype = mi_condition)) +
      geom_point(size = 3, position = position_dodge(width = 0.5)) +
      geom_errorbar(aes(ymin = mean_beta - sd_beta, ymax = mean_beta + sd_beta), 
                    width = 0.2, position = position_dodge(width = 0.5)) +
      geom_hline(yintercept = beta_val, linetype = "dotted", color = "black", linewidth = 0.8) +
      scale_color_manual(values = c("MCAR" = "#0066CC", "MAR" = "#CC0000", "MNAR" = "#006600")) +
      scale_shape_manual(values = c("MCAR" = 16, "MAR" = 17, "MNAR" = 15)) +
      scale_linetype_manual(values = c("wMI" = "solid", "noMI" = "dashed"), 
                            labels = c("wMI" = "With MI", "noMI" = "No MI")) +
      labs(title = bquote(beta[true] == .(beta_val)), x = NULL, y = "Estimate") +
      theme_bw(base_size = 18) +
      theme(
        legend.position = "none",
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
        axis.text.y = element_text(size = 16),
        axis.title = element_text(size = 18, face = "bold"),
        plot.title = element_text(size = 18, face = "bold"),
        strip.text = element_text(size = 18, face = "bold")
      ) +
      coord_cartesian(ylim = y_lims)
    
    plots[[i]] <- p
  }
  
  # Extract legend from dummy plot
  dummy_p <- plots[[1]] + theme(legend.position = "right")
  g <- ggplotGrob(dummy_p)
  legend_idx <- which(sapply(g$grobs, function(x) x$name) == "guide-box")
  if (length(legend_idx) > 0) {
    legend <- g$grobs[[legend_idx]]
  } else {
    legend <- NULL
  }
  
  # Combine subplots
  combined_plot <- grid.arrange(
    grobs = plots,
    ncol = 2,
    top = textGrob(paste(ifelse(ds == "MIMIC", "MIMIC-IV", "AMI"), "Beta Estimates (Comparing wMI vs noMI)"), 
                   gp = gpar(fontsize = 20, fontface = "bold")),
    right = legend
  )
  
  # Output path matching LaTeX include
  fname <- file.path(output_dir, paste0("nb_BETA_PLOT_", ds, "_ALL_wMI_vs_noMI.png"))
  ggsave(fname, combined_plot, width = 14, height = 10, bg = "white", dpi = 300)
  cat(sprintf("  Saved: %s\n", fname))
}

cat("\nDONE.\n")
