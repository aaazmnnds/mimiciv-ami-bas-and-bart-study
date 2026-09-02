import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SCRIPT 20: PLOT ALL VARIABLES (FIGURES D.1 & E.1)
# Exact layout matching manuscript captions for Figure D.1 and E.1:
# - 5 Subplots (Panels): MICE, Mean, KNN, missForest (top row/grid) and Z (indicators only)
# - Inside EACH subplot:
#     * BAS solid red: BAS w/ missing indicators (wMI)
#     * BAS solid black: BAS w/o missing indicators (noMI)
#     * BART dashed blue: BART w/ missing indicators (wMI)
#     * BART dashed black: BART w/o missing indicators (noMI)
# - X-axis: 10% to 100% of variables (deciles)
# - Outputs: table_of_bas_bart_MIMIC_all_vars.png & table_of_bas_bart_MI_all_vars.png
# ==============================================================================

RESULTS_DIR = "Results/CORRECTED_ALL_VARS"
OUTPUT_DIR = "../Resubmission_Package/Submission_v2.0"

DATASETS = {
    "MIMIC": {
        "prefix": "MIMIC_REAL",
        "title": "MIMIC-IV Dataset (All Variables Decile Evaluation: 10%—100%)",
        "outfile": "table_of_bas_bart_MIMIC_all_vars.png"
    },
    "MI": {
        "prefix": "MI_REAL",
        "title": "AMI Dataset (All Variables Decile Evaluation: 10%—100%)",
        "outfile": "table_of_bas_bart_MI_all_vars.png"
    }
}

PERCENTILE_LABELS = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
PANELS = [
    {"key": "MICE",       "title": "MICE Imputation"},
    {"key": "MEAN",       "title": "Mean Imputation"},
    {"key": "KNN",        "title": "KNN Imputation"},
    {"key": "missForest", "title": "missForest Imputation"},
    {"key": "Z_only",     "title": r"Missing Indicators Only: $Z$"}
]

def load_deciles(filepath, col_name):
    """
    Reads log_probabilities CSV and returns 10 decile points (10% to 100%).
    Handles both dense evaluation (all num_top present) and sparse decile evaluation (BART).
    """
    if not os.path.exists(filepath):
        return None, 0
    
    df = pd.read_csv(filepath)
    if col_name not in df.columns:
        if "log_prob_pooled" in df.columns:
            col_name = "log_prob_pooled"
        elif "avg_log_prob" in df.columns:
            col_name = "avg_log_prob"
        else:
            return None, 0
            
    total_vars = len(df)
    if total_vars == 0:
        return None, 0
    
    # 1. If sparse evaluation (e.g. BART with NAs for non-evaluated ranks), extract non-null deciles
    df_clean = df.dropna(subset=[col_name])
    if len(df_clean) == 10:
        return df_clean[col_name].values, total_vars
    elif len(df_clean) > 0 and len(df_clean) < total_vars:
        # Interpolate or sample 10 points evenly from available non-null evaluation points
        indices = np.linspace(0, len(df_clean) - 1, 10, dtype=int)
        return df_clean[col_name].iloc[indices].values, total_vars

    # 2. Dense evaluation (BAS): sample exactly at cumulative decile indices
    indices = [max(0, int(np.round(total_vars * p / 10.0)) - 1) for p in range(1, 11)]
    decile_values = df[col_name].iloc[indices].values
    return decile_values, total_vars

def generate_figure(dataset_key):
    cfg = DATASETS[dataset_key]
    prefix = cfg["prefix"]
    
    # 2 rows x 3 columns layout (5 panels + 1 panel for unified legend / layout)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs_flat = axs.flatten()
    
    x_ticks = np.arange(len(PERCENTILE_LABELS))
    
    for idx, panel in enumerate(PANELS):
        ax = axs_flat[idx]
        method = panel["key"]
        
        # 1. BAS Files
        if method == "MICE":
            bas_nomi_file = os.path.join(RESULTS_DIR, f"{prefix}_MICE_noMI_POOLED_log_probabilities.csv")
            bas_wmi_file  = os.path.join(RESULTS_DIR, f"{prefix}_MICE_wMI_POOLED_log_probabilities.csv")
            bas_nomi_data, _ = load_deciles(bas_nomi_file, "log_prob_pooled")
            bas_wmi_data,  _ = load_deciles(bas_wmi_file,  "log_prob_pooled")
        elif method == "Z_only":
            bas_nomi_data = None
            bas_wmi_file  = os.path.join(RESULTS_DIR, f"{prefix}_Z_only_wMI_log_probabilities.csv")
            bas_wmi_data,  _ = load_deciles(bas_wmi_file, "avg_log_prob")
        else:
            bas_nomi_file = os.path.join(RESULTS_DIR, f"{prefix}_{method}_noMI_log_probabilities.csv")
            bas_wmi_file  = os.path.join(RESULTS_DIR, f"{prefix}_{method}_wMI_log_probabilities.csv")
            bas_nomi_data, _ = load_deciles(bas_nomi_file, "avg_log_prob")
            bas_wmi_data,  _ = load_deciles(bas_wmi_file,  "avg_log_prob")
            
        # 2. BART Files
        if method == "MICE":
            bart_nomi_file = os.path.join(RESULTS_DIR, f"results_BART_{prefix}_MICE_noMI_POOLED_log_probabilities_m5.csv")
            bart_wmi_file  = os.path.join(RESULTS_DIR, f"results_BART_{prefix}_MICE_wMI_POOLED_log_probabilities_m5.csv")
            bart_nomi_data, _ = load_deciles(bart_nomi_file, "log_prob_pooled")
            bart_wmi_data,  _ = load_deciles(bart_wmi_file,  "log_prob_pooled")
        elif method == "Z_only":
            bart_nomi_data = None
            bart_wmi_file  = os.path.join(RESULTS_DIR, f"results_BART_{prefix}_Z_only_wMI_log_probabilities_m1.csv")
            bart_wmi_data,  _ = load_deciles(bart_wmi_file, "avg_log_prob")
        else:
            bart_nomi_file = os.path.join(RESULTS_DIR, f"results_BART_{prefix}_{method}_noMI_log_probabilities_m1.csv")
            bart_wmi_file  = os.path.join(RESULTS_DIR, f"results_BART_{prefix}_{method}_wMI_log_probabilities_m1.csv")
            bart_nomi_data, _ = load_deciles(bart_nomi_file, "avg_log_prob")
            bart_wmi_data,  _ = load_deciles(bart_wmi_file,  "avg_log_prob")
            
        # Plot Curves
        # BAS: Solid lines (Black = noMI, Red = wMI)
        if bas_nomi_data is not None:
            ax.plot(x_ticks, bas_nomi_data, color="black", linestyle="-", marker="o", markersize=4, label="BAS (w/o mi)")
        if bas_wmi_data is not None:
            ax.plot(x_ticks, bas_wmi_data, color="red", linestyle="-", marker="s", markersize=4, label="BAS (w/ mi)")
            
        # BART: Dashed lines (Black = noMI, Blue = wMI)
        if bart_nomi_data is not None:
            ax.plot(x_ticks, bart_nomi_data, color="black", linestyle="--", marker="o", markersize=4, label="BART (w/o mi)")
        if bart_wmi_data is not None:
            ax.plot(x_ticks, bart_wmi_data, color="blue", linestyle="--", marker="s", markersize=4, label="BART (w/ mi)")
            
        ax.set_title(panel["title"], fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(PERCENTILE_LABELS, rotation=45, fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("Top variables", fontsize=10)
        ax.set_ylabel("Avg log predicted prob", fontsize=10)
            
    # Subplot 6 (Bottom right): Legend and summary box
    ax_legend = axs_flat[5]
    ax_legend.axis("off")
    
    handles, labels = axs_flat[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", fontsize=11, frameon=True, title="Model & Indicator Condition", title_fontsize=12)
    
    fig.suptitle(cfg["title"], fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, cfg["outfile"])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("Generating Figures D.1 and E.1 matching manuscript caption (BAS & BART combined per panel)...")
    for ds_key in DATASETS:
        generate_figure(ds_key)
    print("Done.")

if __name__ == "__main__":
    main()
