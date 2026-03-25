import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define paths
base_dir = "."
data_dir = os.path.join(base_dir, "Data")
output_dir = base_dir

mimic_path = os.path.join(data_dir, "mimic_septic_shock_tabular.csv")
ami_path = os.path.join(data_dir, "cleaned.mi (myocardial infarction).csv")

def generate_correlation_assets(file_path, name, short_name):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Filter numeric columns and those with < 95% missingness
    missing_pct = df.isnull().mean()
    valid_cols = missing_pct[missing_pct <= 0.95].index
    df_filtered = df[valid_cols]
    
    # Keep only numeric
    df_numeric = df_filtered.select_dtypes(include=[np.number])
    
    # Drop columns with zero variance
    df_numeric = df_numeric.loc[:, df_numeric.std() > 0]
    
    # Calculate correlation matrix
    cor_matrix = df_numeric.corr()
    
    # Save CSV
    csv_name = f"correlation_matrix_{short_name}.csv"
    cor_matrix.to_csv(os.path.join(output_dir, csv_name))
    print(f"Saved {csv_name}")
    
    # Generate Heatmap
    plt.figure(figsize=(12, 10))
    # Use clustering to group correlated variables
    sns.clustermap(cor_matrix.fillna(0), cmap='coolwarm', center=0, 
                   annot=False, figsize=(15, 15))
    
    plt.title(f"{name} Predictor Correlations")
    png_name = f"{short_name}_correlation_heatmap.png"
    plt.savefig(os.path.join(output_dir, png_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_name}")

# Run for both
generate_correlation_assets(mimic_path, "MIMIC-III", "mimic")
generate_correlation_assets(ami_path, "AMI", "ami")
