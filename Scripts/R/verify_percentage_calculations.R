# Verify percentage improvements and inflation claims
# For Reviewer 3 Major Comment 5

# Define calculation functions
calc_rel_improvement <- function(new_val, old_val) {
  return((new_val - old_val) / old_val * 100)
}

calc_rel_mse_inflation <- function(rel_mse) {
  # If rel_mse is 872.35%, calculation is (8.72 - 1.0)/1.0 * 100
  return(rel_mse - 100)
}

# 1. MIMIC-III AUC: 0.69 -> 1.00
mimic_auc_imp <- calc_rel_improvement(1.00, 0.69)
cat(sprintf("MIMIC-III AUC Improvement (1.00 vs 0.69): %.2f%%\n", mimic_auc_imp))

# 2. MIMIC-III Sophisticated vs Mean: 0.995 vs 0.69
mimic_soph_imp <- calc_rel_improvement(0.995, 0.69)
cat(sprintf("MIMIC-III Sophisticated vs Mean (0.995 vs 0.69): %.2f%%\n", mimic_soph_imp))

# 3. AMI F1: 0.22 vs 0.29
ami_f1_imp <- calc_rel_improvement(0.2855, 0.2245)
cat(sprintf("AMI F1 Improvement (0.2855 vs 0.2245): %.2f%%\n", ami_f1_imp))

# 4. AMI MAR vs MCAR AUC: 0.48 vs 0.91
ami_mar_drop <- calc_rel_improvement(0.48, 0.91)
cat(sprintf("AMI MAR vs MCAR AUC Drop (0.48 vs 0.91): %.2f%%\n", ami_mar_drop))

# 5. AMI Coefficient Inflation: 872.35%
ami_coeff_infl <- calc_rel_mse_inflation(872.35)
cat(sprintf("AMI Coefficient Inflation (Rel MSE 872.35%%): %.2f%%\n", ami_coeff_infl))

# 6. AMI MNAR Precision (w/ Ind vs w/o Ind): 0.38 vs 0.20
ami_mnar_prec <- calc_rel_improvement(0.38, 0.20)
cat(sprintf("AMI MNAR Precision Improvement (0.38 vs 0.20): %.2f%%\n", ami_mnar_prec))
