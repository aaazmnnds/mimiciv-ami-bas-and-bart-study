import pandas as pd
import sys

def preprocess_mimiciii_data():
    """
    Loads, merges, cleans, and pivots MIMIC-III data (LABEVENTS and DIAGNOSES_ICD).
    """
    print("Starting MIMIC-III Data Preprocessing...")

    # Define file paths
    lab_file = 'LABEVENTS.csv'
    diagnoses_file = 'DIAGNOSES_ICD.csv'
    output_file = 'mimiciii_final_preprocessed.csv'

    # 1. Load Data
    print(f"Loading {lab_file} and {diagnoses_file}...")
    try:
        # Using low_memory=False to suppress DtypeWarnings if files are large
        labevents = pd.read_csv(lab_file, low_memory=False)
        diagnoses_icd = pd.read_csv(diagnoses_file, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files are in the current directory.")
        sys.exit(1)

    # 2. Merge DataFrames
    print("Merging DataFrames on SUBJECT_ID...")
    merged_df = pd.merge(labevents, diagnoses_icd, on='SUBJECT_ID', how='inner')

    # 3. Clean Duplicates and Filter Observations
    print("Cleaning duplicates and filtering observations...")
    
    # Function to handle duplicates logic: prioritize maintaining key ITEMIDs if present
    def select_first_observation(group):
        # Specific logic derived from notebook analysis: 
        # Check if 78552 is in group ICD9 (though logic seemed mixed, this is the most robust interpretation)
        if 78552 in group['ICD9_CODE'].values:
            return group
        else:
            return group.drop_duplicates(subset='ITEMID', keep='first')

    # Apply the duplicate handling logic
    # Note: Using group_keys=False to prevent index explosion
    merged_df = merged_df.groupby('ICD9_CODE', group_keys=False).apply(select_first_observation)

    # 4. Drop Unnecessary Columns
    columns_to_remove = [
        'ROW_ID_x', 'HADM_ID_x', 'CHARTTIME', 'VALUENUM', 'VALUEUOM',
        'ROW_ID_y', 'HADM_ID_y', 'SEQ_NUM'
    ]
    # Only drop columns that actually exist to avoid errors
    existing_cols_to_drop = [c for c in columns_to_remove if c in merged_df.columns]
    merged_df = merged_df.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")

    # 5. Clean VALUE and FLAG columns
    print("Cleaning VALUE and FLAG columns...")
    # Keep only rows where VALUE is numeric/valid
    merged_df = merged_df[pd.to_numeric(merged_df['VALUE'], errors='coerce').notnull()]
    
    # Fill missing FLAGs and standardize values
    if 'FLAG' in merged_df.columns:
        merged_df['FLAG'].fillna('normal', inplace=True)
        merged_df['FLAG'] = merged_df['FLAG'].replace({'normal': 0, 'abnormal': 1})

    # 6. Pivot to Wide Format
    print("Pivoting data to wide format (Features as Columns)...")
    # Pivot so each ITEMID becomes a column for VALUE and FLAG
    pivot_df = merged_df.pivot_table(
        index=['SUBJECT_ID', 'ICD9_CODE'], 
        columns='ITEMID', 
        values=['VALUE', 'FLAG'], 
        aggfunc='first'
    )
    
    # Flatten the hierarchical columns
    pivot_df = pivot_df.reset_index()
    pivot_df.columns = ['_'.join(map(str, col)) if col[1] != '' else col[0] for col in pivot_df.columns]

    # Rearrange columns for readability (Group VALUE_x and FLAG_x together)
    item_ids = merged_df['ITEMID'].unique()
    columns_reordered = []
    # Only try to add columns that actually exist after pivoting
    for item_id in item_ids:
        val_col = f'VALUE_{item_id}'
        flag_col = f'FLAG_{item_id}'
        if val_col in pivot_df.columns:
            columns_reordered.append(val_col)
        if flag_col in pivot_df.columns:
            columns_reordered.append(flag_col)
            
    final_df = pivot_df[['SUBJECT_ID', 'ICD9_CODE'] + columns_reordered]

    # 7. Final Polish: Remove columns with 100% missing values
    print("Removing empty columns...")
    final_df = final_df.dropna(axis=1, how='all')

    print(f"Final Data Shape: {final_df.shape}")

    # 8. Save Output
    print(f"Saving cleaned data to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_mimiciii_data()
