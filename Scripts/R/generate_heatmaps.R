library(corrplot)

# Use relative paths for data loading

# Function to generate and save heatmap
generate_heatmap <- function(file_path, output_name, title) {
  # Load data
  df <- read.csv(file_path)
  
  # Select numeric columns only
  df_numeric <- df[, sapply(df, is.numeric)]
  
  # Remove columns with zero variance
  df_numeric <- df_numeric[, apply(df_numeric, 2, function(x) var(x, na.rm=TRUE) > 0)]
  
  # Calculate correlation matrix
  cor_mat <- cor(df_numeric, use="pairwise.complete.obs")
  
  # Save plot
  png(output_name, width=1200, height=1200, res=150)
  corrplot(cor_mat, method="color", type="upper", order="hclust", 
           tl.col="black", tl.srt=45, tl.cex=0.6,
           title=title, mar=c(0,0,1,0))
  dev.off()
  
  # Also save the matrix as CSV for the replication package
  write.csv(cor_mat, paste0(gsub(".png", "", output_name), ".csv"))
}

# Generate MIMIC-III heatmap
generate_heatmap("../../Data/cleaned.mi (mimiciii).csv", 
                 "mimic_correlation_heatmap.png", 
                 "MIMIC-III Predictor Correlations")

# Generate AMI heatmap
generate_heatmap("../../Data/cleaned.mi (myocardial infarction).csv", 
                 "ami_correlation_heatmap.png", 
                 "AMI Predictor Correlations")
