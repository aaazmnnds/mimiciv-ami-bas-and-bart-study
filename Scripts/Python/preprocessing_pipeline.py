import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import os
import sys

# ==========================================
# Configuration & Constants
# ==========================================
MI_TARGET_COL = 'ZSN'
MIMIC_TARGET_COL = 'ICD9_CODE'
MISSING_THRESHOLD = 0.95
SPLIT_FOLDS = 10
TEST_SIZE = 0.10

# ==========================================
# Helper Functions
# ==========================================

def load_and_clean_raw_mi(filepath):
    """
    Loads raw MI dataset, replaces '?' with NaN, and saves intermediate cleaned files.
    """
    print(f"Processing raw MI file: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df.replace("?", np.nan, inplace=True)
        
        # Save intermediate cleaned versions
        df.to_csv(f"cleaned.{filepath}", index=False)
        df.to_csv(f"mi with NA (myocardial infarction).csv", index=False) # Preserving original naming convention
        
        return df
    except FileNotFoundError:
        print(f"Warning: Raw file {filepath} not found. Skipping initial MI processing.")
        return None

def add_missing_indicators(df):
    """
    Adds missing indicator columns (_missing) and a total_missing_values column.
    Reorders columns to place missing indicators next to their original variables.
    """
    print("Generating missing indicators...")
    # Create missing indicator dummy variables
    missing_indicator_df = df.apply(lambda x: pd.isnull(x).astype(int), axis=0)
    missing_indicator_df.columns = [col + '_missing' for col in missing_indicator_df.columns]

    # Calculate total missing values per row
    total_missing_values = df.isnull().sum(axis=1)
    total_missing_values = total_missing_values.rename("total_missing_values")

    # Concatenate everything
    df_combined = pd.concat([df, missing_indicator_df, total_missing_values], axis=1)

    # Reorder columns to interleave original and _missing cols
    new_order = []
    for col in df.columns:
        new_order.append(col)
        new_order.append(col + '_missing')
    
    new_order.append('total_missing_values')
    
    # Select only columns that exist (some might not have been created if logic changed, but logic above creates all)
    final_cols = [c for c in new_order if c in df_combined.columns]
    
    return df_combined[final_cols]

def normalize_data(df, target_col):
    """
    Applies Z-score normalization to all columns except the target column.
    """
    print(f"Normalizing data (preserving target: {target_col})...")
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found in dataset. Skipping normalization.")
        return df

    target_data = df[target_col]
    data_to_normalize = df.drop(columns=[target_col])
    
    # Apply z-score, dealing with potential errors if std is 0 (returns NaN, which we might want to fill or handle)
    # The original code just used zscore directly.
    normalized_data = data_to_normalize.apply(zscore)
    
    # Add target back
    normalized_data[target_col] = target_data
    
    # Move target to first column
    cols = [target_col] + [c for c in normalized_data.columns if c != target_col]
    return normalized_data[cols]

def filter_high_missing_columns(df, threshold=0.95):
    """
    Drops columns that have more than <threshold> missing values.
    Also drops the corresponding _missing indicators for those columns.
    """
    print(f"Filtering columns with > {threshold*100}% missing values...")
    missing_percentages = df.isnull().mean()
    cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
    
    # Also find corresponding _missing columns if they exist
    # (The logic in the notebook implies dropping the _missing version if the parent is dropped)
    # However, if the _missing column itself is full of 0s/1s, it won't have NaNs. 
    # But the notebook logic explicitly finds corresponding _missing cols.
    
    additional_drops = []
    for col in cols_to_drop:
        if not col.endswith('_missing'):
            missing_ver = col + '_missing'
            if missing_ver in df.columns:
                additional_drops.append(missing_ver)
                
    all_drops = list(set(cols_to_drop + additional_drops))
    
    if all_drops:
        print(f"Dropping {len(all_drops)} columns.")
        df_dropped = df.drop(columns=all_drops, errors='ignore')
        return df_dropped
    return df

def generate_splits(df, output_tag, folds=10, test_size=0.10):
    """
    Generates random train/test splits and conserves them to CSVs.
    """
    print(f"Generating {folds} splits for {output_tag}...")
    for i in range(1, folds + 1):
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=None) # Random state None means random every time
        
        train_filename = f"{i}_{int((1-test_size)*100)}train_data{output_tag}.csv"
        test_filename = f"{i}_{int(test_size*100)}test_data{output_tag}.csv"
        
        train_data.to_csv(train_filename, index=False)
        test_data.to_csv(test_filename, index=False)

def process_imputed_workflow(input_file, target_col, output_tag, create_nomi=False):
    """
    Standard workflow for an imputed file:
    Load -> Normalize -> Filter High Missing -> Save Normalized -> Split -> Save Splits.
    Optionally does the same for a version with all Missing Indicators removed (step 'NoMI').
    """
    if not os.path.exists(input_file):
        print(f"Skipping {input_file} (not found)")
        return

    print(f"=== Processing Imputed File: {input_file} ===")
    
    # 1. Load
    df = pd.read_csv(input_file)
    
    # 2. Normalize
    df_norm = normalize_data(df, target_col)
    
    # 3. Filter High Missing
    df_clean = filter_high_missing_columns(df_norm, MISSING_THRESHOLD)
    
    # 4. Save Normalized Cleaned
    norm_filename = f"normalized_{output_tag}.csv"
    if 'imputed' in input_file:
         norm_filename = f"normalized_{input_file}" # Stick to original naming if possible or use tag? 
         # The notebook used 'normalized_myocardial_imputed_1F.csv' for the cleaned version.
         # Let's use a standardized name based on input to be safe and clean.
    
    df_clean.to_csv(norm_filename, index=False)
    print(f"Saved normalized file: {norm_filename}")
    
    # 5. Generate Splits (Full version)
    generate_splits(df_clean, output_tag)
    
    # 6. Optional: Create No-Missing-Indicators version
    if create_nomi:
        print("Creating dataset version without missing indicators...")
        cols_to_remove = [col for col in df_clean.columns if '_missing' in col or col == 'total_missing_values']
        df_nomi = df_clean.drop(columns=cols_to_remove)
        
        nomi_filename = f"normalized_{output_tag}_nomi.csv"
        df_nomi.to_csv(nomi_filename, index=False)
        
        generate_splits(df_nomi, f"no{output_tag}")

# ==========================================
# Main Execution
# ==========================================
def main():
    print("Starting Preprocessing Pipeline...")
    
    # ---------------------------------------------------------
    # PART 1: Raw Data Processing (Initial Cleaning & Indicators)
    # ---------------------------------------------------------
    
    # A. MI Dataset
    raw_mi = load_and_clean_raw_mi('mi (myocardial infarction).csv')
    if raw_mi is not None:
        mi_with_ind = add_missing_indicators(raw_mi)
        # Assuming we just print/inspect it here as per notebook.
    
    # B. MIMIC Dataset
    mimic_file = 'mimiciii_reshaped_row100.csv'
    if os.path.exists(mimic_file):
        print(f"Processing raw MIMIC file: {mimic_file}")
        raw_mimic = pd.read_csv(mimic_file)
        mimic_with_ind = add_missing_indicators(raw_mimic)
        mimic_with_ind.to_csv('cleaned.mi (mimiciii).csv', index=False)
        print("Saved cleaned MIMIC file clean.mi (mimiciii).csv")
    else:
        print(f"Warning: {mimic_file} not found.")

    # ---------------------------------------------------------
    # PART 2: Imputed Data Processing (Normalize -> Clean -> Split)
    # ---------------------------------------------------------
    # To process different datasets, modify the variables below.
    # IMPORTANT: Ensure 'output_tag' is unique for each dataset to avoid overwriting files.
    
    # EXAMPLE 1: Processing an MI Imputed File
    input_file_mi = 'myocardial_imputed_1.csv'   # CHANGE THIS to your input filename
    output_tag_mi = 'MI1'                        # CHANGE THIS to a unique tag (e.g., 'MI2', 'MI_Mean')
    
    process_imputed_workflow(
        input_file=input_file_mi, 
        target_col=MI_TARGET_COL, 
        output_tag=output_tag_mi, 
        create_nomi=True  # Set to False if you don't need the 'No Missing Indicator' version
    )

    # EXAMPLE 2: Processing a MIMIC Imputed File
    input_file_mimic = 'mimiciii_mean_imputed.csv'  # CHANGE THIS to your input filename
    output_tag_mimic = 'mimiciii_mean_imputed'      # CHANGE THIS to a unique tag
    
    process_imputed_workflow(
        input_file=input_file_mimic, 
        target_col=MIMIC_TARGET_COL, 
        output_tag=output_tag_mimic, 
        create_nomi=False
    )
    
    print("\nPipeline Completed Successfully.")

if __name__ == "__main__":
    main()
