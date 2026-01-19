import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import os
import glob
import re

# ==========================================
# Configuration
# ==========================================
DATA_DIR = "Data"
MISSING_THRESHOLD = 0.95
SPLIT_FOLDS = 10
TEST_SIZE = 0.10
RANDOM_STATE = 42

# Target Columns
TARGET_MAP = {
    "MIMIC": "ICD9_CODE",
    "MI": "ZSN"
}

# ==========================================
# Helper Functions
# ==========================================

def normalize_data(df, target_col):
    """
    Applies Z-score normalization to all columns except the target column.
    """
    if target_col not in df.columns:
        # Try to find target column specifically for this file
        # Check if it was renamed to Y or something?
        # Assuming R script kept original names.
        print(f"    Warning: Target column {target_col} not found.")
        return df

    target_data = df[target_col]
    data_to_normalize = df.drop(columns=[target_col])
    
    # Drop non-numeric columns before z-score
    numeric_data = data_to_normalize.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < data_to_normalize.shape[1]:
        print("    Note: Non-numeric columns excluded from normalization.")

    normalized_data = numeric_data.apply(zscore)
    
    # Add target back
    normalized_data[target_col] = target_data
    
    return normalized_data

def filter_high_missing_columns(df, threshold=0.95):
    """
    Drops columns that have more than <threshold> missing values.
    """
    missing_percentages = df.isnull().mean()
    cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"    Dropping {len(cols_to_drop)} columns > {threshold*100}% missing.")
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df

def generate_splits(df, output_tag, folds=10, test_size=0.10):
    """
    Generates random train/test splits and saves them to CSVs.
    """
    print(f"    Generating {folds} splits...")
    
    for i in range(1, folds + 1):
        # Use simple random split (bootstrap-like or just random seed variation?)
        # For K-Fold, usually we use KFold splitter.
        # But this function implements N random splits (Monte Carlo interpretation).
        # We vary the random_state by fold index to get different splits.
        current_seed = RANDOM_STATE + i
        
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=current_seed)
        
        # Naming convention: [Fold]_90train_data[Tag].csv
        train_filename = os.path.join(DATA_DIR, f"{i}_{int((1-test_size)*100)}train_data{output_tag}.csv")
        test_filename = os.path.join(DATA_DIR, f"{i}_{int(test_size*100)}test_data{output_tag}.csv")
        
        train_data.to_csv(train_filename, index=False)
        test_data.to_csv(test_filename, index=False)

def get_target_col(filename):
    if "MIMIC" in filename:
        return TARGET_MAP["MIMIC"]
    elif "MI" in filename:
        # Be careful to distinguish MI (Myocardial Infarction) from MIMIC
        if "MIMIC" not in filename: 
            return TARGET_MAP["MI"]
    return None

def process_file(filepath):
    filename = os.path.basename(filepath)
    
    # Determine Output Tag based on Filename
    # Imputation script outputs: [Dataset]_[Method].csv or [Dataset]_[Method]_[i].csv
    # Example: MIMIC_MCAR_MEAN.csv
    # We want output_tag = MIMIC_MCAR_MEAN
    
    name_no_ext = os.path.splitext(filename)[0]
    output_tag = name_no_ext
    
    target_col = get_target_col(output_tag)
    
    if not target_col:
        print(f"Skipping {filename}: Could not determine target column (MIMIC/MI).")
        return

    print(f"Processing {filename} (Target: {target_col})...")
    
    try:
        df = pd.read_csv(filepath)
        
        # 1. Normalize
        df_norm = normalize_data(df, target_col)
        
        # 2. Filter High Missing (Likely 0 in imputed data, but good check)
        df_clean = filter_high_missing_columns(df_norm, MISSING_THRESHOLD)
        
        # 3. Generate Splits
        generate_splits(df_clean, output_tag, folds=SPLIT_FOLDS, test_size=TEST_SIZE)
        
    except Exception as e:
        print(f"    Error processing {filename}: {e}")

# ==========================================
# Main Execution
# ==========================================
def main():
    print("Starting Batch Preprocessing Pipeline...")
    print(f"Looking for imputed files in {DATA_DIR}...")
    
    # Pattern to match Imputation Output Files
    # They usually end with _MEAN.csv, _MICE_x.csv, _missForest.csv, _KNN.csv
    patterns = ["*_MEAN.csv", "*_MICE_*.csv", "*_missForest.csv", "*_KNN.csv"]
    
    files_to_process = []
    for p in patterns:
        files_to_process.extend(glob.glob(os.path.join(DATA_DIR, p)))
    
    # Remove duplicates if any
    files_to_process = list(set(files_to_process))
    files_to_process.sort()
    
    if not files_to_process:
        print("No imputed files found. Run the imputation script first.")
        return

    print(f"Found {len(files_to_process)} files.")
    
    for fp in files_to_process:
        process_file(fp)
        
    print("\nBatch Processing Completed.")

if __name__ == "__main__":
    main()
