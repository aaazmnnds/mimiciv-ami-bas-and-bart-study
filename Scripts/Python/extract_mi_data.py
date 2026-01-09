import pandas as pd
from ucimlrepo import fetch_ucirepo
import os
import sys

def fetch_and_process_mi_data():
    """
    Fetches the Myocardial Infarction Complications dataset from the UCI Machine Learning Repository,
    processes the features and targets, and saves variable descriptions.
    """
    print("Fetching Myocardial Infarction Complications dataset (ID: 579)...")
    try:
        # Fetch the dataset
        mi_complications = fetch_ucirepo(id=579)
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        sys.exit(1)

    # Extract features and targets
    X = mi_complications.data.features
    y = mi_complications.data.targets
    
    print("Dataset fetched successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # Display Metadata
    print("\nDataset Metadata:")
    print(mi_complications.metadata)

    # Variable Information
    variables = mi_complications.variables
    
    # Save variable descriptions to CSV
    desc_filename = 'mi_variable_descriptions.csv'
    print(f"\nSaving variable descriptions to '{desc_filename}'...")
    
    if isinstance(variables, dict):
        variable_info = pd.DataFrame.from_dict(variables, orient='index', columns=['Description'])
        variable_info.to_csv(desc_filename, index_label='Variable')
    else:
        # Assuming variables is a DataFrame or list of dicts as per ucimlrepo structure
        variable_info = pd.DataFrame(variables)
        variable_info.to_csv(desc_filename, index=False)
        
    print("Variable descriptions saved.")

    # List Key Variables
    print("\nFeature Variables:")
    print(X.columns.tolist())

    print("\nTarget Variables:")
    print(y.columns.tolist())

    # NOTE: The original notebook contained a local PostgreSQL connection block 
    # with hardcoded credentials. This has been removed for security and portability.
    # If database connectivity is required, please configure environment variables:
    # DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
    
    # Example of how to connect securely if needed:
    # import psycopg2
    # conn = psycopg2.connect(
    #     dbname=os.getenv("DB_NAME", "mimic-iii"),
    #     user=os.getenv("DB_USER"),
    #     password=os.getenv("DB_PASSWORD"),
    #     host=os.getenv("DB_HOST", "localhost"),
    #     port=os.getenv("DB_PORT", "5432")
    # )

    print("\nExtraction and initial processing complete.")

if __name__ == "__main__":
    fetch_and_process_mi_data()
