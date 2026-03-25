# Calculate missingness percentage for each variable in MIMIC-III and AMI datasets
# For Reviewer 3 Major Comment 3

# Define relative paths (assuming script runs from Replication_Package/)
mimic_path <- "Data/mimic_septic_shock_tabular.csv"
ami_path <- "Data/cleaned.mi (myocardial infarction).csv"

# Function to calculate missingness
calculate_missing_pct <- function(path, name) {
  if (!file.exists(path)) {
    stop(paste("File not found:", path))
  }
  data <- read.csv(path)
  missing_pct <- colMeans(is.na(data)) * 100
  results <- data.frame(
    Variable = names(missing_pct),
    Missing_Pct = as.numeric(missing_pct)
  )
  results <- results[order(-results$Missing_Pct), ]
  
  cat(paste("\n---", name, "Missingness ---\n"))
  print(head(results, 20)) # Print top indices
  return(results)
}

# Run for both datasets
mimic_results <- calculate_missing_pct(mimic_path, "MIMIC-III")
ami_results <- calculate_missing_pct(ami_path, "AMI")

# Save detailed results to CSV if needed
# write.csv(mimic_results, "variable_missingness_mimic.csv", row.names = FALSE)
# write.csv(ami_results, "variable_missingness_ami.csv", row.names = FALSE)
