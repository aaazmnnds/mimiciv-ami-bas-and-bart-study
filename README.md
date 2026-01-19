# Analysis of Imputation Methods and Missing Indicators for Bayesian Variable Selection in Clinical Data

**Author:** Azman Nads (azmannads@msutawi-tawi.edu.ph)  
**License:** MIT License  
**Repository URL:** [https://github.com/aaazmnnds/mimiciv-ami-bas-and-bart-study](https://github.com/aaazmnnds/mimiciv-ami-bas-and-bart-study)

This repository contains the code and data to replicate the findings of the manuscript.

## Repository Structure

*   `Data/`: Contains the processed datasets used for simulation and analysis.
    *   `cleaned.mi (mimiciii).csv`: Processed MIMIC-III cohort.
    *   `cleaned.mi (myocardial infarction).csv`: Processed Myocardial Infarction dataset.
*   `Scripts/`:
    *   `R/`: All R scripts for data simulation, imputation, modeling, and evaluation.
    *   `Python/`: Helper scripts for data extraction and additional plotting.
*   `Results/`: Destination folder for generated plots and tables.

## Requirements

*   **R** (v4.0+)
    *   Packages: `missMethods`, `dplyr`, `ggplot2`, `pROC`, `openxlsx`, `BART`, `BAS`
*   **Python** (v3.8+) (Optional, for alternative plotting)
    *   Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`

## Replication Workflow

### 1. Data Simulation
Run the simulation script to generate datasets under MCAR, MAR, and MNAR mechanisms.
```bash
Rscript Scripts/R/simulation_mcar_mar_mnar.R
```

### 2. Imputation
Apply the four imputation methods (MICE, MEAN, KNN, missForest) to the simulated datasets.
```bash
Rscript Scripts/R/imputation_methods.R
```

### 3. Model Analysis (BAS & BART)
Run the Bayesian Variable Selection models.
```bash
Rscript Scripts/R/bas.glm_mimic_mice.R
Rscript Scripts/R/bart_mimic_mice.R
```

### 4. Evaluation
Calculate bias, variable selection metrics, and prediction performance.
```bash
Rscript Scripts/R/evaluation_configuration.R  # Generates config
Rscript Scripts/R/evaluate_bias.R
Rscript Scripts/R/evaluate_variable_selection.R
Rscript Scripts/R/evaluate_predictions.R
```

### 5. Generate Summary Tables and Plots
Create the master summary Excel file and all plots.
```bash
Rscript Scripts/R/create_summary_tables.R
Rscript Scripts/R/plot_beta_estimates.R
Rscript Scripts/R/plot_calibration.R
```

## Citation

If you use the code or data in this repository, please cite:

> Nads, A. (2024). *Analysis of Imputation Methods and Missing Indicators for Bayesian Variable Selection in Clinical Data*. [Manuscript submitted for publication].

## Contact
Azman Nads (azmannads@msutawi-tawi.edu.ph)
