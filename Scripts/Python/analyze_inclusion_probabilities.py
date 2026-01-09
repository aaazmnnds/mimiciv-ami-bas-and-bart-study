import re
import pandas as pd
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# List of files to process (as found in the original notebook)
TARGET_FILES = [
    "bas.glm_full_nomi_missForest_imputed.R",
    "bas.glm_full_wmi_missForest_imputed.R",
    "lbart_full_top1to20_nomi_missForest_imputed.R"
]

def parse_inclusion_probabilities(file_path):
    """
    Parses an R output file to extract Top variables and their inclusion probabilities.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None

    print(f"Processing: {file_path}...")
    
    results = []
    current_top = None
    current_vars = []
    current_probs = []
    reading_vars = False

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Detect "Top K variables:" block start
            match = re.match(r'^Top (\d+) variables:', line)
            if match:
                # Save previous block if it exists
                if current_top is not None:
                    results.append({
                        "Top": current_top,
                        "Variables": ', '.join(current_vars),
                        "InclusionProbability": ', '.join([f"{p:.6f}" for p in current_probs])
                    })
                
                # Reset for new block
                current_top = int(match.group(1))
                current_vars = []
                current_probs = []
                reading_vars = True
                continue

            # Stop reading at end of block marker
            if "Average Log Predicted Probability:" in line:
                reading_vars = False
                continue

            # Skip header lines inside block
            if reading_vars and line.startswith("Variable"):
                continue

            # Parse Variable and Probability lines
            if reading_vars and line:
                # line format example: "   AGE   0.969333"
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 2:
                    try:
                        # Sometimes var is in the second to last position, prob is last
                        raw_var = parts[-2].strip()
                        # specific cleaning if needed
                        var = raw_var.split()[-1]
                        prob = float(parts[-1])
                        
                        current_vars.append(var)
                        current_probs.append(prob)
                    except ValueError:
                        continue

        # Append the very last block found
        if current_top is not None and current_vars:
            results.append({
                "Top": current_top,
                "Variables": ', '.join(current_vars),
                "InclusionProbability": ', '.join([f"{p:.6f}" for p in current_probs])
            })

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

    return pd.DataFrame(results)

def main():
    print("="*80)
    print("ANALYZING INCLUSION PROBABILITIES")
    print("="*80)

    for r_file in TARGET_FILES:
        df = parse_inclusion_probabilities(r_file)
        
        if df is not None and not df.empty:
            # Generate output filename
            # e.g. "bas.glm_full_nomi_missForest_imputed.R" -> "topvar_bas_glm_full_nomi_missForest.csv"
            base_name = os.path.splitext(r_file)[0]
            # Simple sanitization
            safe_name = base_name.replace("imputed", "").strip("_.")
            output_csv = f"topvar_{safe_name}.csv"
            
            df.to_csv(output_csv, index=False)
            print(f"✓ Saved: {output_csv}")
            print(df.head())
            print("-" * 40)
        else:
            print(f"Skipping {r_file} (No data found or file missing).")

    print("\nDONE.")

if __name__ == "__main__":
    main()
