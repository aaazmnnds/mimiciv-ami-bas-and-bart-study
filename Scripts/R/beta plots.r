################################################################################
# BETA PLOTS - USING FOLD-BY-FOLD DATA (FIXED Y-AXIS LABELS)
# - Mean ± SD calculated across 10 folds
# - Beta_true value shown AS Y-AXIS LABEL (not floating)
# - Single merged legend
# - FIXED: Better spacing for beta labels on y-axis
################################################################################

library(dplyr)
library(ggplot2)
library(gridExtra)

cat("\n", rep("=", 80), "\n", sep = "")
cat("BETA PLOTS - FOLD-BY-FOLD CALCULATION (FIXED Y-AXIS SPACING)\n")
cat("Beta_true shown on Y-axis with better spacing, single legend\n")
cat(rep("=", 80), "\n\n", sep = "")

# ============================================================================
# FUNCTION: Load true variables
# ============================================================================

load_true_variables <- function(dataset, mechanism) {
  info_file <- paste0(dataset, "_", mechanism, "_true_variables.csv")
  if (!file.exists(info_file)) return(NULL)
  info <- read.csv(info_file)
  return(list(variables = info$variable, betas = info$beta_true))
}

# ============================================================================
# FUNCTION: Extract fold-by-fold beta estimates
# ============================================================================

extract_fold_betas <- function(dataset, mechanism, method, mi_condition) {
  
  true_info <- load_true_variables(dataset, mechanism)
  if (is.null(true_info)) return(NULL)
  
  true_vars <- true_info$variables
  beta_true_vals <- true_info$betas
  
  if (method == "MICE") {
    file_name <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_beta_estimates.csv")
  } else {
    file_name <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_beta_estimates.csv")
  }
  
  if (!file.exists(file_name)) {
    cat("  ⚠️  File not found:", file_name, "\n")
    return(NULL)
  }
  
  beta_data <- read.csv(file_name)
  beta_data_filtered <- beta_data %>% filter(variable %in% true_vars)
  
  if (method == "MICE") {
    beta_data_filtered <- beta_data_filtered %>%
      select(fold, variable, beta_hat = beta_pooled)
  } else {
    beta_data_filtered <- beta_data_filtered %>%
      select(fold, variable, beta_hat)
  }
  
  beta_mapping <- data.frame(variable = true_vars, beta_true = beta_true_vals)
  
  result <- beta_data_filtered %>%
    left_join(beta_mapping, by = "variable") %>%
    mutate(dataset = dataset, mechanism = mechanism,
           method = method, mi_condition = mi_condition)
  
  return(result)
}

# ============================================================================
# MAIN: Extract all fold-by-fold data
# ============================================================================

cat("Extracting fold-by-fold beta estimates...\n\n")

all_fold_data <- list()
counter <- 1

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")

for (dataset in DATASETS) {
  for (mechanism in MECHANISMS) {
    for (method in METHODS) {
      cat("Processing:", dataset, mechanism, method, "wMI ... ")
      result <- extract_fold_betas(dataset, mechanism, method, "wMI")
      if (!is.null(result)) {
        all_fold_data[[counter]] <- result
        counter <- counter + 1
        cat("✓ (", nrow(result), " rows)\n", sep = "")
      } else {
        cat("✗\n")
      }
    }
  }
}

fold_data_all <- bind_rows(all_fold_data)

cat("\n✓ Total fold-by-fold data:", nrow(fold_data_all), "rows\n\n")

# ============================================================================
# CALCULATE MEAN ± SD ACROSS FOLDS
# ============================================================================

plot_data <- fold_data_all %>%
  group_by(dataset, mechanism, method, beta_true) %>%
  summarise(
    mean_beta = mean(beta_hat, na.rm = TRUE),
    sd_beta = sd(beta_hat, na.rm = TRUE),
    n_folds = n(),
    .groups = "drop"
  )

cat("Plot data prepared:", nrow(plot_data), "combinations\n\n")

# ============================================================================
# CREATE PLOTS
# ============================================================================

for (ds in c("MIMIC", "MI")) {
  
  cat(rep("-", 80), "\n", sep = "")
  cat("Creating plot for", ds, "dataset...\n\n")
  
  ds_data <- plot_data %>% filter(dataset == ds)
  beta_values <- sort(unique(ds_data$beta_true))
  
  cat("  Beta_true values:", paste(beta_values, collapse = ", "), "\n\n")
  
  plots <- list()
  
  for (i in 1:length(beta_values)) {
    beta_val <- beta_values[i]
    beta_data <- ds_data %>% filter(beta_true == beta_val)
    
    # FIXED: Create better y-axis breaks with more spacing
    y_range <- range(c(beta_data$mean_beta - beta_data$sd_beta,
                       beta_data$mean_beta + beta_data$sd_beta,
                       beta_val))
    
    # Expand range slightly for better spacing
    y_range_expanded <- y_range + c(-0.1, 0.1) * diff(y_range)
    
    # Create more breaks for better spacing
    y_breaks <- pretty(y_range_expanded, n = 6)
    
    # For MIMIC dataset with small beta values, ensure breaks don't crowd around 0
    if (ds == "MIMIC" && abs(beta_val) < 0.6) {
      # Create custom breaks that give more space around beta_val
      if (beta_val < 0.3) {
        # For β=0.1, ensure breaks are well-spaced
        y_min <- min(y_range_expanded)
        y_max <- max(y_range_expanded)
        y_breaks <- seq(from = floor(y_min/10)*10, 
                        to = ceiling(y_max/10)*10, 
                        by = 10)
        if (length(y_breaks) < 4) {
          y_breaks <- seq(from = floor(y_min/5)*5, 
                          to = ceiling(y_max/5)*5, 
                          by = 5)
        }
      } else {
        # For β=0.5, use appropriate spacing
        y_min <- min(y_range_expanded)
        y_max <- max(y_range_expanded)
        y_breaks <- seq(from = floor(y_min/100)*100, 
                        to = ceiling(y_max/100)*100, 
                        by = 200)
      }
    }
    
    # Ensure beta_val is included in breaks, but avoid crowding
    if (!any(abs(y_breaks - beta_val) < 0.001)) {
      # Find nearest break to beta_val
      nearest_idx <- which.min(abs(y_breaks - beta_val))
      nearest_val <- y_breaks[nearest_idx]
      
      # Only add beta_val if it's sufficiently far from nearest break
      spacing <- if(length(y_breaks) > 1) min(diff(y_breaks)) else 1
      if (abs(nearest_val - beta_val) > spacing * 0.3) {
        y_breaks <- sort(c(y_breaks, beta_val))
      } else {
        # Replace nearest break with beta_val
        y_breaks[nearest_idx] <- beta_val
      }
    }
    
    # Create labels with beta_val highlighted in bold
    y_labels <- sapply(y_breaks, function(val) {
      if (abs(val - beta_val) < 0.001) {
        paste0("bold(β==", beta_val, ")")
      } else {
        as.character(round(val, 1))
      }
    })
    
    p <- ggplot(beta_data, aes(x = method, y = mean_beta, 
                               color = mechanism, 
                               shape = mechanism)) +
      geom_point(size = 4, alpha = 0.9, 
                 position = position_dodge(width = 0.3)) +
      geom_errorbar(aes(ymin = mean_beta - sd_beta, 
                        ymax = mean_beta + sd_beta),
                    width = 0.2, linewidth = 1, alpha = 0.7,
                    position = position_dodge(width = 0.3)) +
      geom_hline(yintercept = beta_val, linetype = "dashed", 
                 linewidth = 0.8, color = "black") +
      scale_y_continuous(breaks = y_breaks, labels = parse(text = y_labels)) +
      scale_color_manual(
        name = "Missing data\nMechanism",
        values = c("MCAR" = "#0066CC", "MAR" = "#CC0000", "MNAR" = "#006600"),
        breaks = c("MCAR", "MAR", "MNAR")
      ) +
      scale_shape_manual(
        name = "Mechanism",
        values = c("MCAR" = 16, "MAR" = 17, "MNAR" = 15),
        breaks = c("MCAR", "MAR", "MNAR")
      ) +
      guides(color = guide_legend(override.aes = list(shape = c(16, 17, 15))),
             shape = "none") +
      labs(
        title = bquote(beta[true] == .(beta_val)),
        x = "Imputation Method",
        y = expression(paste("Predicted ", beta, " (mean ± SD across 10 folds)"))
      ) +
      theme_bw(base_size = 18) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
        axis.text.y = element_text(size = 16, face = "plain"),
        axis.title = element_text(size = 18),
        legend.position = "right",
        legend.title = element_text(size = 18, face = "bold"),
        legend.text = element_text(size = 16),
        legend.key.width = unit(1.5, "cm"),
        panel.grid.minor = element_blank()
      )
    
    plots[[i]] <- p
  }
  
  combined <- grid.arrange(
    grobs = plots,
    ncol = 2,
    nrow = 2,
    top = grid::textGrob(
      paste(ds, "Dataset: Beta Coefficients (mean ± SD across 10 folds)"),
      gp = grid::gpar(fontsize = 22, fontface = "bold")
    )
  )
  
  filename <- paste0("BETA_PLOT_", ds, "_fold_by_fold.png")
  ggsave(filename, combined, width = 14, height = 12, dpi = 300, bg = "white")
  
  cat("  ✓ Saved:", filename, "\n\n")
}

# ============================================================================
# SUMMARY
# ============================================================================

cat(rep("=", 80), "\n", sep = "")
cat("COMPLETE!\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("Files created:\n")
cat("  - BETA_PLOT_MIMIC_fold_by_fold.png\n")
cat("  - BETA_PLOT_MI_fold_by_fold.png\n\n")

cat("Corrections applied:\n")
cat("  ✓ Beta_true value appears ON THE Y-AXIS (highlighted in BOLD)\n")
cat("  ✓ Better spacing for y-axis breaks (no crowding/overlap)\n")
cat("  ✓ MIMIC dataset: Special handling for small beta values (0.1, 0.5)\n")
cat("  ✓ Single merged legend (color + shape combined)\n")
cat("  ✓ Dashed line at beta_true value\n")
cat("  ✓ Mean ± SD calculated across 10 folds\n\n")

cat("✅ Corrected plots with readable y-axis labels created!\n\n")


####################

################################################################################
# BETA PLOTS - COMPARING wMI vs noMI
# - Mean ± SD calculated across 10 folds
# - Beta_true value shown AS Y-AXIS LABEL (not floating)
# - Comparing WITH and WITHOUT missing indicators
################################################################################

library(dplyr)
library(ggplot2)
library(gridExtra)

cat("\n", rep("=", 80), "\n", sep = "")
cat("BETA PLOTS - COMPARING wMI vs noMI\n")
cat("Beta_true shown on Y-axis, comparing missing indicator conditions\n")
cat(rep("=", 80), "\n\n", sep = "")

# ============================================================================
# FUNCTION: Load true variables
# ============================================================================

load_true_variables <- function(dataset, mechanism) {
  info_file <- paste0(dataset, "_", mechanism, "_true_variables.csv")
  if (!file.exists(info_file)) return(NULL)
  info <- read.csv(info_file)
  return(list(variables = info$variable, betas = info$beta_true))
}

# ============================================================================
# FUNCTION: Extract fold-by-fold beta estimates
# ============================================================================

extract_fold_betas <- function(dataset, mechanism, method, mi_condition) {
  
  true_info <- load_true_variables(dataset, mechanism)
  if (is.null(true_info)) return(NULL)
  
  true_vars <- true_info$variables
  beta_true_vals <- true_info$betas
  
  if (method == "MICE") {
    file_name <- paste0(dataset, "_", mechanism, "_", mi_condition, "_POOLED_beta_estimates.csv")
  } else {
    file_name <- paste0(dataset, "_", mechanism, "_", method, "_", mi_condition, "_beta_estimates.csv")
  }
  
  if (!file.exists(file_name)) {
    cat("  ⚠️  File not found:", file_name, "\n")
    return(NULL)
  }
  
  beta_data <- read.csv(file_name)
  beta_data_filtered <- beta_data %>% filter(variable %in% true_vars)
  
  if (method == "MICE") {
    beta_data_filtered <- beta_data_filtered %>%
      select(fold, variable, beta_hat = beta_pooled)
  } else {
    beta_data_filtered <- beta_data_filtered %>%
      select(fold, variable, beta_hat)
  }
  
  beta_mapping <- data.frame(variable = true_vars, beta_true = beta_true_vals)
  
  result <- beta_data_filtered %>%
    left_join(beta_mapping, by = "variable") %>%
    mutate(dataset = dataset, mechanism = mechanism,
           method = method, mi_condition = mi_condition)
  
  return(result)
}

# ============================================================================
# MAIN: Extract all fold-by-fold data (BOTH wMI and noMI)
# ============================================================================

cat("Extracting fold-by-fold beta estimates...\n\n")

all_fold_data <- list()
counter <- 1

METHODS <- c("MICE", "MEAN", "missForest", "KNN")
DATASETS <- c("MIMIC", "MI")
MECHANISMS <- c("MCAR", "MAR", "MNAR")
MI_CONDITIONS <- c("wMI", "noMI")  # ← NOW INCLUDING BOTH

for (dataset in DATASETS) {
  for (mechanism in MECHANISMS) {
    for (method in METHODS) {
      for (mi_cond in MI_CONDITIONS) {  # ← NEW LOOP
        cat("Processing:", dataset, mechanism, method, mi_cond, "... ")
        result <- extract_fold_betas(dataset, mechanism, method, mi_cond)
        if (!is.null(result)) {
          all_fold_data[[counter]] <- result
          counter <- counter + 1
          cat("✓ (", nrow(result), " rows)\n", sep = "")
        } else {
          cat("✗\n")
        }
      }
    }
  }
}

fold_data_all <- bind_rows(all_fold_data)

cat("\n✓ Total fold-by-fold data:", nrow(fold_data_all), "rows\n\n")

# ============================================================================
# CALCULATE MEAN ± SD ACROSS FOLDS
# ============================================================================

plot_data <- fold_data_all %>%
  group_by(dataset, mechanism, method, mi_condition, beta_true) %>%  # ← ADDED mi_condition
  summarise(
    mean_beta = mean(beta_hat, na.rm = TRUE),
    sd_beta = sd(beta_hat, na.rm = TRUE),
    n_folds = n(),
    .groups = "drop"
  )

cat("Plot data prepared:", nrow(plot_data), "combinations\n\n")

# ============================================================================
# CREATE COMBINED PLOTS (ALL 4 BETAS IN ONE FIGURE)
# ============================================================================

cat(rep("-", 80), "\n", sep = "")
cat("Creating combined plots with all betas...\n\n")

for (ds in c("MIMIC", "MI")) {
  
  ds_data <- plot_data %>% filter(dataset == ds)
  beta_values <- sort(unique(ds_data$beta_true))
  
  plots <- list()
  
  for (i in 1:length(beta_values)) {
    beta_val <- beta_values[i]
    beta_data <- ds_data %>% filter(beta_true == beta_val)
    
    y_range <- range(c(beta_data$mean_beta - beta_data$sd_beta,
                       beta_data$mean_beta + beta_data$sd_beta,
                       beta_val))
    y_range_expanded <- y_range + c(-0.1, 0.1) * diff(y_range)
    y_breaks <- pretty(y_range_expanded, n = 5)
    
    if (!any(abs(y_breaks - beta_val) < 0.001)) {
      nearest_idx <- which.min(abs(y_breaks - beta_val))
      y_breaks[nearest_idx] <- beta_val
    }
    
    y_labels <- sapply(y_breaks, function(val) {
      if (abs(val - beta_val) < 0.001) {
        paste0("bold(β==", beta_val, ")")
      } else {
        as.character(round(val, 1))
      }
    })
    
    p <- ggplot(beta_data, aes(x = method, y = mean_beta, 
                               color = mechanism, 
                               shape = mechanism)) +
      geom_point(size = 2.5, alpha = 0.9, 
                 position = position_dodge(width = 0.3)) +
      geom_errorbar(aes(ymin = mean_beta - sd_beta, 
                        ymax = mean_beta + sd_beta),
                    width = 0.2, linewidth = 0.6, alpha = 0.7,
                    position = position_dodge(width = 0.3)) +
      geom_hline(yintercept = beta_val, linetype = "dashed", 
                 linewidth = 0.6, color = "black") +
      facet_wrap(~ mi_condition, nrow = 1, 
                 labeller = labeller(mi_condition = c(
                   "wMI" = "With MI",
                   "noMI" = "Without MI"
                 ))) +
      scale_y_continuous(breaks = y_breaks, labels = parse(text = y_labels)) +
      scale_color_manual(
        name = "Mechanism",
        values = c("MCAR" = "#0066CC", "MAR" = "#CC0000", "MNAR" = "#006600"),
        breaks = c("MCAR", "MAR", "MNAR")
      ) +
      scale_shape_manual(
        name = "Mechanism",
        values = c("MCAR" = 16, "MAR" = 17, "MNAR" = 15),
        breaks = c("MCAR", "MAR", "MNAR")
      ) +
      guides(color = guide_legend(override.aes = list(shape = c(16, 17, 15))),
             shape = "none") +
      labs(
        title = bquote(beta[true] == .(beta_val)),
        x = "Method",
        y = expression(paste(beta, " (mean ± SD)"))
      ) +
      theme_bw(base_size = 18) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
        axis.text.y = element_text(size = 16),
        axis.title = element_text(size = 18),
        legend.position = "right",
        legend.title = element_text(size = 18, face = "bold"),
        legend.text = element_text(size = 16),
        legend.key.size = unit(0.6, "cm"),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 18, face = "bold"),
        strip.background = element_rect(fill = "grey90")
      )
    
    plots[[i]] <- p
  }
  
  combined <- grid.arrange(
    grobs = plots,
    ncol = 2,
    nrow = 2,
    top = grid::textGrob(
      paste(ds, "Dataset: Beta Coefficients - Comparing wMI vs noMI"),
      gp = grid::gpar(fontsize = 20, fontface = "bold")
    )
  )
  
  # Filename matched to LaTeX: nb_Prefix
  filename_combined <- paste0("nb_BETA_PLOT_", ds, "_ALL_wMI_vs_noMI.png")
  ggsave(filename_combined, combined, width = 18, height = 16, dpi = 300, bg = "white")
  
  cat("  ✓ Saved:", filename_combined, "\n\n")
}

# ============================================================================
# SUMMARY
# ============================================================================

cat(rep("=", 80), "\n", sep = "")
cat("COMPLETE!\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("Files created:\n")
cat("  - BETA_PLOT_MIMIC_ALL_wMI_vs_noMI.png\n")
cat("  - BETA_PLOT_MI_ALL_wMI_vs_noMI.png\n\n")

cat("Features:\n")
cat("  ✓ Beta_true value on Y-AXIS (highlighted in BOLD)\n")
cat("  ✓ Comparison of wMI vs noMI side-by-side\n")
cat("  ✓ Three mechanisms (MCAR, MAR, MNAR) with different colors and shapes\n")
cat("  ✓ Mean ± SD calculated across 10 folds\n")
cat("  ✓ All 4 beta values in one comprehensive figure per dataset\n\n")

cat("✅ Beta plots with wMI vs noMI comparison created!\n\n")

