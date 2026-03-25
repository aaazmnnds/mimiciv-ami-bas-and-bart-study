# Calculate correlation matrices for MIMIC-III and AMI predictors
# For Reviewer 3 Major Comment 1

# Define relative paths
mimic_path <- "Data/mimic_septic_shock_tabular.csv"
ami_path <- "Data/cleaned.mi (myocardial infarction).csv"

# Function to calculate and save correlation matrix
generate_correlation_report <- function(path, name, output_csv) {
  if (!file.exists(path)) {
    stop(paste("File not found:", path))
  }
  data <- read.csv(path)
  
  # Filter numeric columns and those with < 95% missingness
  missing_pct <- colMeans(is.na(data))
  valid_cols <- names(data)[missing_pct <= 0.95]
  numeric_data <- data[, valid_cols]
  numeric_data <- numeric_data[, sapply(numeric_data, is.numeric)]
  
  # Calculate correlation matrix (pairwise complete observations)
  cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
  
  # Save to CSV
  write.csv(cor_matrix, output_csv)
  cat(paste("\n---", name, "Correlation Matrix saved to", output_csv, "---\n"))
  
  # Optional: Simple heatmap if library(ggplot2) and library(reshape2) are available
  # library(reshape2)
  # library(ggplot2)
  # melted_cor <- melt(cor_matrix)
  # ggplot(data = melted_cor, aes(x=Var1, y=Var2, fill=value)) +
  #   geom_tile() +
  #   scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  #   theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust = 1))
}

# Run for both
generate_correlation_report(mimic_path, "MIMIC-III", "correlation_matrix_mimic.csv")
generate_correlation_report(ami_path, "AMI", "correlation_matrix_ami.csv")
