################################################################################
# PLOT BETA ESTIMATES
#
# Logic:
# - Visualizes Beta estimates (Mean ± SD across 10 folds).
# - Compares "wMI" (With Missing indicators) vs "noMI" (Without).
# - Uses a single figure per dataset, with subplots for each True Beta value.
#
################################################################################

library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

load("evaluation_config_4VAR.RData")

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("wMI", "noMI")

cat("\n================================================================================\n")
cat("PLOTTING BETA ESTIMATES (Comparison: wMI vs noMI)\n")
cat("================================================================================\n\n")

# ============================================================================
# FUNCTION: Extract Beta Estimates
# ============================================================================

extract_fold_betas <- function(dataset, mechanism, method, mi_condition) {
  
  key <- paste0(dataset, "_", mechanism)
  true_vars <- true_variables[[key]]
  beta_true_vals <- beta_true_values[[key]]
  
  if (is.null(true_vars)) return(NULL)
  
  # File name
  if (method == "MICE") {
    file_name <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_beta_estimates.csv")
  } else {
    file_name <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_beta_estimates.csv")
  }
  
  if (!file.exists(file_name)) return(NULL)
  
  beta_data <- read.csv(file_name)
  
  # Normalize column names
  if (method == "MICE") {
    beta_estimates <- beta_data %>% filter(variable %in% true_vars) %>% select(fold, variable, beta_hat = beta_pooled)
  } else {
    beta_estimates <- beta_data %>% filter(variable %in% true_vars) %>% select(fold, variable, beta_hat)
  }
  
  beta_mapping <- data.frame(variable = true_vars, beta_true = beta_true_vals)
  
  result <- beta_estimates %>%
    left_join(beta_mapping, by = "variable") %>%
    mutate(
      dataset = dataset,
      mechanism = mechanism,
      method = method,
      mi_condition = mi_condition
    )
  
  return(result)
}

# ============================================================================
# 1. EXTRACT ALL DATA
# ============================================================================

cat("Extracting data...\n")
all_data_list <- list()
counter <- 1

for (d in DATASETS) {
  for (m in MECHANISMS) {
    for (meth in METHODS) {
      for (mi in MI_CONDITIONS) {
        res <- extract_fold_betas(d, m, meth, mi)
        if (!is.null(res)) {
          all_data_list[[counter]] <- res
          counter <- counter + 1
        }
      }
    }
  }
}

all_fold_data <- bind_rows(all_data_list)
cat(sprintf("✓ Extracted %d estimates.\n\n", nrow(all_fold_data)))

# ============================================================================
# 2. AGGREGATE (MEAN +/- SD)
# ============================================================================

plot_data <- all_fold_data %>%
  group_by(dataset, mechanism, method, mi_condition, beta_true) %>%
  summarise(
    mean_beta = mean(beta_hat, na.rm = TRUE),
    sd_beta = sd(beta_hat, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================================
# 3. CREATE PLOTS
# ============================================================================

cat("Creating plots...\n")

for (ds in DATASETS) {
  cat(sprintf("  Plotting %s...\n", ds))
  
  ds_data <- plot_data %>% filter(dataset == ds)
  beta_values <- sort(unique(ds_data$beta_true))
  
  plots <- list()
  
  for (i in seq_along(beta_values)) {
    beta_val <- beta_values[i]
    sub_data <- ds_data %>% filter(beta_true == beta_val)
    
    # Calculate y-limits for better styling
    y_min <- min(sub_data$mean_beta - sub_data$sd_beta, na.rm=TRUE)
    y_max <- max(sub_data$mean_beta + sub_data$sd_beta, na.rm=TRUE)
    # Ensure true beta is included
    y_min <- min(y_min, beta_val)
    y_max <- max(y_max, beta_val)
    
    # Add padding
    rng <- y_max - y_min
    if(rng == 0) rng <- 1
    y_lims <- c(y_min - 0.1*rng, y_max + 0.1*rng)

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
      theme_bw() +
      theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1)
      ) +
      coord_cartesian(ylim = y_lims)
    
    plots[[i]] <- p
  }
  
  # Extract legend from a dummy plot
  dummy_p <- plots[[1]] + theme(legend.position = "right")
  legend <- cowplot::get_legend(dummy_p) 
  # If cowplot not installed, use standard grid extraction
  if(is.null(legend)) {
      # Fallback manual extraction
      g <- ggplotGrob(dummy_p)
      legend <- g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  }
  
  # Combine
  combined_plot <- grid.arrange(
    grobs = plots,
    ncol = 2,
    top = textGrob(paste(ds, "Beta Estimates (Comparing wMI vs noMI)"), gp=gpar(fontsize=15, fontface="bold")),
    right = legend
  )
  
  ggsave(paste0("BETA_PLOT_", ds, "_wMI_vs_noMI.png"), combined_plot, width = 12, height = 10, bg="white")
}

cat("Saved plots.\n")
cat("\nDONE.\n")
