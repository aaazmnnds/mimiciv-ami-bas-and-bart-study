#!/opt/anaconda3/bin/python
import pandas as pd
import os

# --- CONFIG ---
# Sepsis-3 / Septic Shock ICD Codes
# ICD-9: 785.52 (Septic Shock)
# ICD-10: A41.* (Sepsis), R65.21 (Severe Sepsis with Septic Shock)
SEPSIS_ICD9 = ["78552"]
SEPSIS_ICD10_PREFIXES = ["A41", "R6521"]

# MIMIC-IV Lab ItemIDs matching existing dataset columns
LAB_ITEMS = {
    # Blood Gas
    50802: "Base_Excess",
    50804: "Calculated_Total_CO2",
    50808: "Free_Calcium",
    50809: "Glucose",
    50812: "Intubated",
    50813: "Lactate",
    50816: "Oxygen",
    50817: "Oxygen_Saturation",
    50818: "pCO2",
    50819: "PEEP",
    50820: "pH",
    50821: "pO2",
    50825: "Temperature",
    50826: "Tidal_Volume",
    50827: "Ventilation_Rate",
    # Chemistry
    50861: "Alanine_Aminotransferase_(ALT)",
    50862: "Albumin",
    50863: "Alkaline_Phosphatase",
    50868: "Anion_Gap",
    50878: "Asparate_Aminotransferase_(AST)",
    50882: "Bicarbonate",
    50885: "Bilirubin_Total",
    50893: "Calcium_Total",
    50902: "Chloride",
    50912: "Creatinine",
    50931: "Glucose_2",
    50954: "Lactate_Dehydrogenase_(LD)",
    50960: "Magnesium",
    50970: "Phosphate",
    50971: "Potassium_2",
    50983: "Sodium",
    51006: "Urea_Nitrogen",
    # Hematology
    51146: "Basophils_5",
    51200: "Eosinophils_6",
    51221: "Hematocrit",
    51222: "Hemoglobin_2",
    51237: "INR(PT)",
    51244: "Lymphocytes_5",
    51248: "MCH",
    51249: "MCHC",
    51250: "MCV",
    51254: "Monocytes_4",
    51256: "Neutrophils",
    51265: "Platelet_Count",
    51274: "PT",
    51275: "PTT",
    51277: "RDW",
    51279: "Red_Blood_Cells",
    51301: "White_Blood_Cells",
}

# MIMIC-IV root directory
ROOT_DIR = "/Users/nazu.ds/Documents/Research Collections/Dr. Zhang/mimic-iv-2.2"
OUTPUT_DIR = "/Users/nazu.ds/Documents/Research Collections/Scientific_Reports_Submission_Package/For resubmission to another journal/mimic iv"

def extract_mimic_iv_sepsis():
    print("--- Extracting MIMIC-IV Septic Shock Cohort ---")

    # Paths
    path_diag = os.path.join(ROOT_DIR, "hosp", "diagnoses_icd.csv.gz")
    path_lab = os.path.join(ROOT_DIR, "hosp", "labevents.csv.gz")
    path_adm = os.path.join(ROOT_DIR, "hosp", "admissions.csv.gz")
    path_icu = os.path.join(ROOT_DIR, "icu", "icustays.csv.gz")

    chunksize = 1000000

    # 1. Identify Sepsis Cohort
    print("Scanning diagnoses for septic shock codes...")
    sepsis_hadm_ids = set()
    for chunk in pd.read_csv(path_diag, chunksize=chunksize):
        mask9 = (chunk['icd_version'] == 9) & (chunk['icd_code'].isin(SEPSIS_ICD9))
        mask10 = (chunk['icd_version'] == 10) & (
            chunk['icd_code'].str.startswith(tuple(SEPSIS_ICD10_PREFIXES), na=False)
        )
        sepsis_hadm_ids.update(chunk[mask9 | mask10]['hadm_id'].unique())
    print(f"Found {len(sepsis_hadm_ids)} septic shock admissions.")

    # 2. Get ICU stay times
    print("Loading ICU stay times...")
    df_icu = pd.read_csv(path_icu, usecols=['hadm_id', 'intime', 'outtime'])
    df_icu = df_icu[df_icu['hadm_id'].isin(sepsis_hadm_ids)]
    df_icu['intime'] = pd.to_datetime(df_icu['intime'])
    df_icu['outtime'] = pd.to_datetime(df_icu['outtime'])
    # Keep first ICU stay per admission
    df_icu = df_icu.sort_values('intime').groupby('hadm_id').first().reset_index()
    df_icu['window_end'] = df_icu['intime'] + pd.Timedelta(hours=24)
    print(f"ICU stays found: {len(df_icu)}")

    # 3. Get outcome — in-hospital mortality
    print("Loading admissions for outcome...")
    df_adm = pd.read_csv(path_adm, usecols=['hadm_id', 'hospital_expire_flag'])
    df_adm = df_adm[df_adm['hadm_id'].isin(sepsis_hadm_ids)]

    # 4. Extract labs within first 24 hours of ICU admission
    print("Extracting labs within first 24h of ICU admission...")
    lab_chunks = []
    for chunk in pd.read_csv(
        path_lab, chunksize=chunksize,
        usecols=['hadm_id', 'itemid', 'valuenum', 'charttime']
    ):
        chunk = chunk.dropna(subset=['hadm_id', 'valuenum', 'charttime'])
        chunk = chunk[chunk['hadm_id'].isin(df_icu['hadm_id'].values)]
        chunk = chunk[chunk['itemid'].isin(LAB_ITEMS.keys())]
        if not chunk.empty:
            chunk['charttime'] = pd.to_datetime(chunk['charttime'])
            chunk = chunk.merge(
                df_icu[['hadm_id', 'intime', 'window_end']], on='hadm_id', how='left'
            )
            chunk = chunk[
                (chunk['charttime'] >= chunk['intime']) &
                (chunk['charttime'] <= chunk['window_end'])
            ]
            if not chunk.empty:
                lab_chunks.append(chunk[['hadm_id', 'itemid', 'valuenum']])

    if not lab_chunks:
        print("No labs found within 24h window.")
        return

    df_labs = pd.concat(lab_chunks)

    # Clinical plausibility filters to remove erroneous values
    VALID_RANGES = {
        50802: (-30, 30),      # Base_Excess mmol/L
        50804: (5, 50),        # Calculated_Total_CO2 mmol/L
        50808: (0.5, 3.5),     # Free_Calcium mmol/L
        50809: (30, 600),      # Glucose blood gas mg/dL
        50812: (0, 1),         # Intubated binary
        50813: (0.3, 30),      # Lactate mmol/L
        50816: (21, 100),      # Oxygen %
        50817: (50, 100),      # Oxygen_Saturation %
        50818: (10, 150),      # pCO2 mmHg
        50819: (0, 30),        # PEEP cmH2O
        50820: (6.8, 7.8),     # pH
        50821: (20, 700),      # pO2 mmHg
        50825: (30, 42),       # Temperature Celsius
        50826: (100, 1000),    # Tidal_Volume mL
        50827: (0, 40),        # Ventilation_Rate breaths/min
        50861: (0, 5000),      # ALT U/L
        50862: (1, 6),         # Albumin g/dL
        50863: (0, 2000),      # Alkaline_Phosphatase U/L
        50868: (1, 30),        # Anion_Gap mEq/L
        50878: (0, 5000),      # AST U/L
        50882: (5, 50),        # Bicarbonate mEq/L
        50885: (0, 50),        # Bilirubin_Total mg/dL
        50893: (4, 15),        # Calcium_Total mg/dL
        50902: (70, 130),      # Chloride mEq/L
        50912: (0.1, 30),      # Creatinine mg/dL
        50931: (50, 1000),     # Glucose_2 mg/dL chemistry
        50954: (0, 5000),      # Lactate_Dehydrogenase U/L
        50960: (0.5, 5),       # Magnesium mg/dL
        50970: (0.5, 10),      # Phosphate mg/dL
        50971: (2, 8),         # Potassium_2 mEq/L
        50983: (110, 160),     # Sodium mEq/L
        51006: (1, 200),       # Urea_Nitrogen mg/dL
        51146: (0, 5),         # Basophils %
        51200: (0, 20),        # Eosinophils %
        51221: (15, 65),       # Hematocrit %
        51222: (3, 20),        # Hemoglobin g/dL
        51237: (0.5, 20),      # INR
        51244: (0, 80),        # Lymphocytes %
        51248: (15, 45),       # MCH pg
        51249: (25, 40),       # MCHC g/dL
        51250: (50, 130),      # MCV fL
        51254: (0, 30),        # Monocytes %
        51256: (0, 95),        # Neutrophils %
        51265: (10, 1500),     # Platelet_Count K/uL
        51274: (9, 100),       # PT seconds
        51275: (15, 200),      # PTT seconds
        51277: (10, 25),       # RDW %
        51279: (2, 7),         # Red_Blood_Cells M/uL
        51301: (0.5, 50),      # White_Blood_Cells K/uL
    }

    # Apply filters
    def filter_row(row):
        itemid = row['itemid']
        val = row['valuenum']
        if itemid in VALID_RANGES:
            lo, hi = VALID_RANGES[itemid]
            return lo <= val <= hi
        return True

    df_labs = df_labs[df_labs.apply(filter_row, axis=1)]
    print(f"After filtering: {len(df_labs)} lab values retained")
    df_labs['item_name'] = df_labs['itemid'].map(LAB_ITEMS)

    # Aggregate — mean per admission per lab
    df_pivot = df_labs.pivot_table(
        index='hadm_id', columns='item_name',
        values='valuenum', aggfunc='mean'
    )
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'hadm_id': 'HADM_ID'}, inplace=True)

    # 5. Merge with outcome
    df_adm.rename(columns={'hadm_id': 'HADM_ID'}, inplace=True)
    df_final = df_pivot.merge(df_adm[['HADM_ID', 'hospital_expire_flag']], on='HADM_ID', how='left')

    # 6. Save
    out_path = os.path.join(OUTPUT_DIR, "mimic-iv sepsis.csv")
    df_final.to_csv(out_path, index=False)
    print(f"Saved {len(df_final)} patients to {out_path}")
    print(f"Outcome distribution:\n{df_final['hospital_expire_flag'].value_counts()}")
    print(f"Missing data summary:\n{df_final.isnull().sum()}")

if __name__ == "__main__":
    extract_mimic_iv_sepsis()
