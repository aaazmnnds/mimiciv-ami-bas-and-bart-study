# Analysis of Imputation Methods and Missing Indicators for Bayesian Variable Selection in Clinical Data

**Authors:** A. Nads (azmannads@msutawi-tawi.edu.ph), D. Andrade (andrade@hiroshima-u.ac.jp)  
**License:** MIT License  
**Repository:** https://github.com/aaazmnnds/mimiciv-ami-bas-and-bart-study  
**Submission:** Scientific Reports (Round 2 Revision)

## Overview

This repository contains the analysis scripts to replicate the findings of the manuscript. The study evaluates four imputation methods (mean imputation, MICE, KNN, missForest) combined with Bayesian variable selection methods (BAS and BART) on two clinical datasets: MIMIC-III (septic shock, 64% missingness) and AMI (chronic heart failure, 4.63% missingness), under three missingness mechanisms (MCAR, MAR, MNAR).

## Repository Structure
Scripts/

├── R/

│   ├── 01_bas_real_data.R           # BAS analysis on real clinical data

│   ├── 02_bart_real_data.R          # BART analysis on real clinical data

│   ├── 03_bas_simulation.R          # BAS analysis on simulated data

│   ├── 04_bart_simulation.R         # BART analysis on simulated data

│   ├── 05_generate_scenario_datasets.R  # Generate scenario analysis datasets

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

│   └── Archive/                     # Deprecated and exploratory scripts

├── Python/

│   ├── 01_preprocess_mimic.py       # MIMIC-III data preprocessing

│   ├── 02_preprocessing_pipeline.py # General preprocessing pipeline

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

### 4. Full Variable Set Analysis
```bash
Rscript Scripts/R/08_bas_all_variables.R
Rscript Scripts/R/09_bart_all_variables.R
```

### 5. Evaluation and Plotting
```bash
Rscript Scripts/R/10_evaluate_predictions.R
Rscript Scripts/R/12_evaluate_variable_selection.R
Rscript Scripts/R/13_evaluate_bias.R
Rscript Scripts/R/14_plot_coefficient_recovery.R
```

## Important Notes

- All preprocessing, imputation, normalization, and model fitting are performed **strictly within each cross-validation fold** using only training data parameters to prevent data leakage.
- For MICE, test fold imputation uses column means derived from the MICE-imputed training fold.
- The MIMIC-III dataset requires PhysioNet credentialing: https://physionet.org/content/mimiciii/
- The AMI dataset is the UCI Myocardial Infarction Complications dataset.

## Data Availability

- **MIMIC-III:** Available through PhysioNet (https://physionet.org/content/mimiciii/) following completion of required training and data use agreements.
- **AMI:** Available through UCI Machine Learning Repository.

## Citation

If you use this code, please cite:

> Nads, A., & Andrade, D. (2026). Analysis of Imputation Methods and Missing Indicators for Bayesian Variable Selection in Clinical Data. *Scientific Reports* (under review).

## Contact

- A. Nads: azmannads@msutawi-tawi.edu.ph  
- D. Andrade: andrade@hiroshima-u.ac.jp
