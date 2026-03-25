import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# CONFIGURATION
sns.set_context("talk", font_scale=1.2)
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

METHODS = ["MICE", "MEAN", "missForest", "KNN"]
DATASETS = ["MIMIC", "MI"]
MECHANISMS = ["MCAR", "MAR", "MNAR"]
MI_CONDITIONS = ["noMI", "wMI"]

def get_calibration_data(dataset, mechanism, method, mi_condition):
    """
    Reads predictions and log probabilities, filters for the best num_top,
    and returns a DataFrame with predictions and true labels.
    """
    
    # Construct filenames based on naming convention
    if method == "MICE":
        pred_file = f"{dataset}_{mechanism}_{mi_condition}_POOLED_predictions.csv"
        log_file = f"{dataset}_{mechanism}_{mi_condition}_POOLED_log_probabilities.csv"
        pred_col = "predicted_prob_pooled"
        log_col = "log_prob_pooled"
    else:
        pred_file = f"{dataset}_{mechanism}_{method}_{mi_condition}_predictions.csv"
        log_file = f"{dataset}_{mechanism}_{method}_{mi_condition}_log_probabilities.csv"
        pred_col = "predicted_prob"
        # Handle potential column name variations if necessary, but assuming clean standard
        log_col = "avg_log_prob"

    # Check existence
    if not os.path.exists(pred_file) or not os.path.exists(log_file):
        return None

    try:
        pred_data = pd.read_csv(pred_file)
        log_data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading files for {dataset} {mechanism} {method}: {e}")
        return None

    # Find Best Num Top based on Logic
    if log_col in log_data.columns:
        best_idx = log_data[log_col].idxmax()
        best_num_top = log_data.loc[best_idx, "num_top"]
    else:
        # Fallback
        best_num_top = 4

    # Filter Predictions
    pred_subset = pred_data[pred_data["num_top"] == best_num_top].copy()
    
    if pred_subset.empty:
        return None
    
    # Select and Rename Columns
    result = pred_subset[["true_label", pred_col]].rename(columns={pred_col: "predicted_prob"})
    
    # Add Metadata
    result["dataset"] = dataset
    result["mechanism"] = mechanism
    result["method"] = method
    result["mi_condition"] = mi_condition
    
    return result

def plot_calibration():
    print("="*80)
    print("GENERATING CALIBRATION CURVES (PYTHON)")
    print("="*80)

    all_data = []

    # 1. Aggregate Data
    print("Loading data...")
    for ds in DATASETS:
        for mech in MECHANISMS:
            for meth in METHODS:
                for mi in MI_CONDITIONS:
                    res = get_calibration_data(ds, mech, meth, mi)
                    if res is not None:
                        all_data.append(res)
    
    if not all_data:
        print("No data found. Ensure CSV files are in the current directory.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} prediction records.")

    # 2. Binning and Calculation
    # We will use equal-sized bins (quantiles) to match the R 'ntile' logic
    print("Calculating calibration bins...")
    
    # Create a 'bin' column based on deciles within each group
    # Note: pd.qcut can drop bins if duplicates exist, so we use rank method for stability
    def assign_bins(x):
        try:
            return pd.qcut(x, 10, labels=False, duplicates='drop') + 1
        except:
             # Fallback for too few unique values
            return 1

    combined_df['bin'] = combined_df.groupby(['dataset', 'mechanism', 'method', 'mi_condition'])['predicted_prob'].transform(assign_bins)

    calibration_summary = combined_df.groupby(['dataset', 'mechanism', 'method', 'mi_condition', 'bin']).agg(
        mean_predicted=('predicted_prob', 'mean'),
        observed_proportion=('true_label', 'mean'),
        n=('true_label', 'count')
    ).reset_index()

    # 3. Plotting
    print("Creating plots...")
    sns.set_context("talk") # This increases base font size significantly
    sns.set_style("whitegrid")

    plt.rcParams.update({
        'legend.title_fontsize': 'large',
        'legend.fontsize': 'medium',
        'axes.labelsize': 'large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium'
    })

    for ds in DATASETS:
        ds_data = calibration_summary[calibration_summary["dataset"] == ds]
        
        for mi in MI_CONDITIONS:
            plot_data = ds_data[ds_data["mi_condition"] == mi]
            
            if plot_data.empty:
                continue

            # Since seaborn relplot handles its own figure creation:
            g = sns.relplot(
                data=plot_data,
                x="mean_predicted",
                y="observed_proportion",
                hue="method",
                style="method",
                col="mechanism",
                kind="line",
                markers=True,
                height=6,
                aspect=0.9,
                facet_kws={'sharey': True, 'sharex': True}
            )
            
            # Add diagonal reference line to each facet
            for ax in g.axes.flat:
                ax.plot([0, 1], [0, 1], ls="--", c="gray", alpha=0.5)
                ax.set_xlabel("Mean Predicted Probability", fontweight='bold')
                ax.set_ylabel("Observed Proportion", fontweight='bold')

            g.fig.subplots_adjust(top=0.88)
            g.fig.suptitle(f"Calibration Plot: {ds} ({mi})", fontsize=18, fontweight='bold')
            
            filename = f"CALIBRATION_PLOT_PYTHON_{ds}_{mi}.png"
            g.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filename}")

    print("\nDONE.")

if __name__ == "__main__":
    plot_calibration()
