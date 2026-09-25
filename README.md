# Imputation Strategies and Missing Indicators for Bayesian Variable Selection in Clinical Data: A Comparative Study of BAS and BART

**Authors:** A. Nads (azmannads@msutawi-tawi.edu.ph), D. Andrade (andrade@hiroshima-u.ac.jp)  
**License:** MIT License  
**Repository:** https://github.com/aaazmnnds/mimiciv-ami-bas-and-bart-study  
**Submission:** BMC Medical Research Methodology (under review)

## Overview

This repository contains the analysis scripts to replicate the findings of the manuscript. The study evaluates four imputation methods (mean imputation, MICE, KNN, missForest) combined with Bayesian variable selection methods (BAS and BART) on two clinical datasets: MIMIC-IV (septic shock, 24.27% missingness, N=10,990) and AMI (chronic heart failure, 4.63% missingness, N=1,699), under three missingness mechanisms (MCAR, MAR, MNAR).

## Repository Structure
Scripts/

├── R/

│   ├── 01_bas_real_data.R           # BAS analysis on real clinical data

│   ├── 02_bart_real_data.R          # BART analysis on real clinical data

│   ├── 03_bas_simulation.R          # BAS analysis on simulated data

│   ├── 04_bart_simulation.R         # BART analysis on simulated data

│   ├── 05_generate_scenario_datasets.R  # Generate scenario analysis datasets

│   ├── 05c_run_bas_alpha10.R        # BAS sensitivity analysis with amplified signal

│   ├── 05d_run_bart_alpha10.R       # BART sensitivity analysis with amplified signal

│   ├── 06_impute_scenarios.R        # Impute scenario datasets

│   ├── 07_evaluate_scenarios.R      # Evaluate scenario analysis (BAS)

│   ├── 08_bas_all_variables.R       # BAS full variable set analysis

│   ├── 09_bart_all_variables.R      # BART full variable set analysis

│   ├── 10_evaluate_predictions.R    # Evaluate prediction performance

│   ├── 11_evaluate_predictions_alpha10.R  # Evaluate amplified signal predictions

│   ├── 12_evaluate_variable_selection.R   # Evaluate variable selection metrics

│   ├── 13_evaluate_bias.R           # Evaluate coefficient bias

│   ├── 14_plot_coefficient_recovery.R     # Plot coefficient recovery figures

│   ├── 15_calculate_missingness.R   # Calculate missingness statistics

│   ├── 16_calculate_correlations.R  # Calculate correlation matrices

│   ├── 17_extract_top_variables.R   # Extract top-ranked variables by PIP and inclusion proportion

│   ├── 18_evaluate_real_data.R      # Evaluate real data models and metrics

│   └── Archive/                     # Deprecated and exploratory scripts

├── Python/

│   ├── 20_plot_all_variables.py     # Plot full variable set performance

│   ├── extract_mimic_iv_sepsis.py   # Extract and preprocess MIMIC-IV sepsis cohort

│   └── Archive/                     # Deprecated scripts

## Requirements

**R (v4.0+)**
BAS, BART, mice, missForest, VIM, dplyr, ggplot2, pROC, missMethods

**Python (v3.8+)**
pandas, numpy, matplotlib, seaborn

## Replication Workflow

### 1. Real Data Analysis
```bash
Rscript Scripts/R/01_bas_real_data.R
Rscript Scripts/R/02_bart_real_data.R
```

### 2. Simulation Analysis
```bash
Rscript Scripts/R/03_bas_simulation.R
Rscript Scripts/R/04_bart_simulation.R
```

### 3. Scenario Analysis
```bash
Rscript Scripts/R/05_generate_scenario_datasets.R
Rscript Scripts/R/06_impute_scenarios.R
Rscript Scripts/R/07_evaluate_scenarios.R
```

### 4. Sensitivity Analysis
```bash
Rscript Scripts/R/05c_run_bas_alpha10.R
Rscript Scripts/R/05d_run_bart_alpha10.R
```

### 5. Full Variable Set Analysis
```bash
Rscript Scripts/R/08_bas_all_variables.R
Rscript Scripts/R/09_bart_all_variables.R
```

### 6. Evaluation and Plotting
```bash
Rscript Scripts/R/10_evaluate_predictions.R
Rscript Scripts/R/12_evaluate_variable_selection.R
Rscript Scripts/R/13_evaluate_bias.R
Rscript Scripts/R/14_plot_coefficient_recovery.R
```

## Important Notes

- All preprocessing, imputation, normalization, and model fitting are performed **strictly within each cross-validation fold** using only training data parameters to prevent data leakage.
- For MICE, test fold imputation uses column means derived from the MICE-imputed training fold.
- The MIMIC-IV dataset requires PhysioNet credentialing: https://physionet.org/content/mimiciv/
- The AMI dataset is the UCI Myocardial Infarction Complications dataset.

## Data Availability

- **MIMIC-IV:** Raw data available through PhysioNet (https://physionet.org/content/mimiciv/) following completion of required training and data use agreements.
- **AMI (Myocardial Infarction Complications):** Raw data available through UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications).
- Processed datasets used in this study can be requested from the corresponding author (andrade@hiroshima-u.ac.jp).

## Citation

If you use this code, please cite:

> Nads, A., & Andrade, D. (2026). Imputation Strategies and Missing Indicators for Bayesian Variable Selection in Clinical Data: A Comparative Study of BAS and BART. BMC Medical Research Methodology (under review).

## Contact

- A. Nads: azmannads@msutawi-tawi.edu.ph  
- D. Andrade: andrade@hiroshima-u.ac.jp
