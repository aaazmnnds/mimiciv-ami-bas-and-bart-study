# Response to Reviewers
Manuscript ID: d09478c6-df6e-4897-b833-d217bac44f2f
Title: Analysis of imputation methods and missing indicators for Bayesian variable selection in clinical data

Dear Editor Nojiri and Reviewers,

We appreciate the constructive and insightful comments on our manuscript. We have carefully addressed each point raised by the reviewers and the editor. Below is our point-by-point response detailing the revisions made.

## Response to Reviewer 1

### Major Comment 1 (Section 2.4 "Missing indicators and experimental design")
**Reviewer Comment:** "In Section 2.4 "Missing indicators and experimental design", the discussion of missing-indicator methods would benefit from tighter integration with the manuscript’s core goals of variable selection and prediction. The current content is the basic insight about missing indicator, but it remains somewhat generic relative to the paper’s own results and to prior evidence (Donders et al., 2006; Van der Heijden et al., 2006; Groenwold et al., 2012)."

**Response:** We have entirely rewritten Section 2.4 "Missing indicators and experimental design" to provide a deeper technical rationale for using missing indicators in the context of variable selection and prediction. We now explicitly discuss the trade-off where indicators improve predictive metrics by acting as informative surrogate features but simultaneously alter model selection behavior and potentially increase false discoveries. We have integrated the recommended citations (Donders et al., 2006; Van der Heijden et al., 2006; Groenwold et al., 2012) to support this discussion.

**Revisions in Manuscript:** See Section 2.4 "Missing indicators and experimental design," specifically the sub-subsection "Rationale for including missing indicators" (Section 2.4.3 "Rationale for including missing indicators").

### Major Comment 2 (Simulation design)
**Reviewer Comment:** "In Section 2.7 (Simulation design), one setup choice has a large impact on the results: the 'true' predictors are chosen as the variables with the most missing values. This means the important predictors are also the hardest to observe. This is a useful test case, but it is only one type of situation. It can make methods that use missingness information (such as adding missing indicators) look stronger, because the missing pattern itself carries extra signal. As a result, the reported gains in prediction and the variable-selection results (Type I/II, sensitivity, precision) should be interpreted as specific to this setup, not as a general conclusion for all datasets. The simulation section would be stronger if it also included additional scenarios: (1) true predictors chosen independently of missingness level, (2) a mixed case where some true predictors have high missingness and others have low missingness, and (3) a case where many non-true variables have high missingness."

**Response:** We agree that the simulation design should accommodate different patterns of true predictors. We have now implemented and evaluated three distinct scenarios for selecting true predictors: (1) Top Missingness (original base case), (2) Independent Selection (randomly chosen regardless of missingness), and (3) Mixed Selection (a combination of high- and low-missingness variables). Our analysis across 18 unique simulation conditions (3 scenarios × 3 mechanisms × 2 datasets) confirms that our primary findings on mechanism dominance and method-indicator trade-offs hold. For instance, in the Independent Selection case for MIMIC-III (MCAR), PIPs for true variables remained near 0.88, consistent with the Top Missingness case. We have updated Sections 2.7 "Simulation study design" and 3 "Results" of the manuscript to include these additional scenarios and sensitivity results.

**Revisions in Manuscript:**
- Updated Section 2.7 "Simulation study design" to describe the three selection scenarios.
- Updated Section 3 "Results" to include a summary of the sensitivity findings.
- Added Appendix I with detailed sensitivity analysis results (Table I.6).

### Major Comment 3 (Overgeneralization of "Mechanism Matters")
**Reviewer Comment:** "In Section 3 and Discussion 4.1, the statement that 'mechanism matters more than method' and that MAR leads to large performance loss is an important message, and the results do show a clear mechanism effect. At the same time, this conclusion appears to depend on the specific data settings used in this paper. The two applications are quite different (n/p, missingness level, and variability across folds), with MIMIC-III showing much higher missingness than AMI (64% vs 6.15%). Because of this, the size of the MAR effect may reflect both the missingness mechanism and the data structure in each setting. A small wording adjustment would make the conclusion more precise (e.g., 'in the settings studied, mechanism had a larger impact than method')."

**Response:** We agree that the "Mechanism matters" conclusion should be presented with appropriate context. We have added qualifiers to our main conclusion statements and inserted a sensitivity summary in the Discussion. We also added a caveat in the Limitations section acknowledging that mechanism effects are confounded with dataset-specific factors like $n/p$ ratios and missingness levels.

**Revisions in Manuscript:**
- Added qualifiers "in the settings studied" to Section 3.2.5 "Validation of real data findings" (Point 6) and Section 4.1 "Interpretation of simulation findings".
- Added sensitivity analysis summary to Section 4.4 "Practical considerations".
- Updated Section 4.5 "Limitations and future directions" to discuss confounding factors and generalizability.

### Major Comment 4 (Prediction-Selection Trade-off in Section 4.2 "Implications for method selection")
**Reviewer Comment:** "In Section 4.2 (Implications for method selection), the conclusion that mean imputation with missing indicators performs well for prediction is supported by several reported results. However, the simulation results also show higher false-positive selection in some scenarios. Since the paper is about both Bayesian variable selection and prediction, this trade-off should be stated more directly in the main interpretation. In practical terms, the same approach can improve prediction accuracy but also select extra variables that may not be truly important."

**Response:** We have revised Section 4.2 "Implications for method selection" to explicitly state the prediction-selection trade-off. We now clearly distinguish between applications prioritizing risk stratification (where the trade-off may be acceptable) and those prioritizing unbiased inference (where it is not). We have added direct decision guidance: Prediction $\rightarrow$ Mean with Indicators; Inference $\rightarrow$ MICE.

**Revisions in Manuscript:**
- Inserted a dedicated trade-off paragraph in Section 4.2 "Implications for method selection".
- Added explicit decision guidance to the introduction of Section 4.2 "Implications for method selection".
- Added a summary sentence to Section 3.2.5 "Validation of real data findings" incorporating both benefits and costs.

### Minor Comment 1 (Duplicate Sentence)
**Reviewer Comment:** "Page 12: In Section 2.7.1, 'For MCAR, we deleted values uniformly at random across all variables.' appears duplicated consecutively. Please remove the repeated sentence."

**Response:** We have removed the duplicate sentence: "For MCAR, we deleted values uniformly at random across all variables." from Section 2.7 "Simulation study design".

### Minor Comment 2 (Figure Font Size and Readability)
**Reviewer Comment:** "The figure text is difficult to read at the current size (e.g., Figure 1). Please increase the font size of axis labels, tick labels, legends, and in-panel annotations to improve readability and interpretation."

**Response:** We have addressed this by applying two types of refinements:
1.  **Manuscript Display Size**: For Figures 1 and 2 (Average Log Predicted Probabilities), we have increased the display width in the manuscript from 0.8\textwidth to 1.1\textwidth to maximize space. For Figures 3 and 4 (Coefficient Recovery), we have increased the width to 1.1\textwidth.
2.  **Font Size Adjustments**: For Figures 3 and 4 (Coefficient Recovery), we have significantly increased the base font size to **18** in the corresponding plotting script (`plot_beta_estimates.R`). This includes increasing axis labels, tick labels, legends, and sub-panel titles (annotations) to be clearly legible. We have also ensured that high-resolution outputs (300 DPI) are generated with these larger text elements.

**Revisions in Manuscript:** See updated Figure 1, 2, 3, and 4 in the main text.

---

## Response to Reviewer 2

We thank Reviewer 2 for the thorough evaluation and constructive feedback. We have revised the manuscript to improve clarity, precision, and the interpretation of our results.

### Major Comment 1 (Reviewer Point 1: Section 2.4.3 "Rationale for including missing indicators")
**Reviewer Comment:** "In Section 2.4.3 Rationale for including missing indicators, can the authors clarify the following points?
1) Is the missing indicator of one variable independent of its imputed values?
2) If the missing indicator of one variable depends on its imputed values or other covariates, will including them in a logistic model lead to estimation instability?
3) Did the authors consider including the interaction terms between the missing indicators and the imputed covariates in logistic regression models?
4) If one missing indicator is associated with more of the other imputed covariates, is it reasonable to treat this missing indicator equally with the other missing indicators?"

**Response:** We thank the reviewer for these technical points. We have added a paragraph to Section 2.4.3 "Rationale for including missing indicators" specifically addressing these structural considerations:
1.  **Independence:** Indicators $Z$ are constructed prior to imputation; while stochastically distinct from imputed values $X^{\text{imp}}$, their utility stems from correlations with underlying true values (MNAR) or observed covariates (MAR).
2.  **Instability:** We acknowledge the potential for collinearity/instability when including both features. We rely on the shrinkage and selection properties of BAS and BART to mitigate this; explicit stability analysis was not performed.
3.  **Interactions:** We restricted our model to main effects ($\beta X + \gamma Z$) as shown in Equation 18 to maintain model parsimony and avoid overwhelming the variable selection algorithm.
4.  **Equal Treatment:** All candidate variables (indicators and covariates) are treated equally by the variable selection algorithms, with their inclusion driven by the data and Bayesian priors.

**Revisions in Manuscript:** See expanded technical discussion in Section 2.4.3 "Rationale for including missing indicators".

### Minor Comment 1 (Reviewer Point 2: Section 2.7 "Simulation study design")
**Reviewer Comment:** "In Step 1 - Missingness imposition of Section 2.7.1 Data generation, the sentence 'For MCAR, we deleted values uniformly at random across all variables.' is repeated."

**Response:** We have removed the duplicate sentence: "For MCAR, we deleted values uniformly at random across all variables." from Section 2.7 "Simulation study design".

**Revisions in Manuscript:** See Section 2.7 "Simulation study design".

### Major Comment 2 (Reviewer Point 3: Section 3.1.1 "BAS performance")
**Reviewer Comment:** "In Section 3.1.1 BAS performance, the authors have an interesting finding: 'Without indicators, sophisticated methods (MICE, KNN, missForest) substantially outperformed mean imputation (44% better). With indicators, this gap disappeared completely.' Can the authors provide any interpretation or explanation?"

**Response:** We have added the requested justification in Section 3.1.1 "BAS performance". In the high-missingness regimes studied, missing indicators provide a higher-order signal that sophisticated imputation cannot recover from the available covariates alone. Once this "missingness signal" is captured by the indicator, the incremental predictive gain from the precise imputed value is diminished. This aligns with recent benchmarks (e.g., Paterakis et al. 2024) which suggest that mean imputation with indicators is often sufficient for competitive prediction in clinical data.

**Revisions in Manuscript:** See expanded discussion in Section 3.1.1 "BAS performance".

### Major Comment 3 (Reviewer Point 4: Figures 3 and 4 - Descriptive Captions)
**Reviewer Comment:** "For Fig. 3, can the authors provide more explanations or evidence to support the description given in the last sentence 'Missing indicators improve recovery under MNAR but provide minimal benefit under MCAR and cannot recover MAR failures.'? For instance, is this observation consistent across all the methods and scenarios?"

**Response:** We have revised the captions for Figures 3 and 4 to provide a clearer explanation of the observed performance trends. We have added specific qualifications noting that while patterns for MCAR and MAR are consistent across datasets, the benefit of missing indicators under MNAR is subject to sample-size and dimensionality constraints. We also added Section 4.4 "Practical considerations" to the Discussion to explicitly state the settings where indicators provide maximal benefit (e.g., MNAR in larger datasets) versus limited benefit (e.g., higher-dimensional regimes like MIMIC-III).

**Revisions in Manuscript:** Updated captions for Figure 3 (Coefficient recovery for MIMIC-III dataset) and Figure 4 (Coefficient recovery for AMI dataset); added Section 4.4 "Practical considerations".

### Major Comment 4 (Reviewer Point 5: Figures 3 and 4 - Trend Inconsistencies in MIMIC-III)
**Reviewer Comment:** "Similarly, for Fig. 4, can the authors provide more explanations or evidence to support the description given in the last sentence 'Missing indicators substantially improve MNAR recovery, show minimal MCAR impact, and cannot overcome MAR information loss.'? For instance, is this observation consistent across all the methods and scenarios?"

**Response:** We thank the reviewer for identifying these discrepancies. We have added text acknowledging that in high-missingness/low-sample-size regimes like MIMIC-III where the variable-to-observation ratio is high ($p/n \approx 0.6$), the inclusion of indicators for every variable significantly expands the feature space. This can introduce noise and overwhelm the variable selection algorithms, potentially hurting sensitivity despite the precision improvements typically provided by indicators. This explains the dataset-specific variations observed under MNAR.

**Revisions in Manuscript:** See revised Figure 3 and 4 captions and expanded text in Section 3.2.2 "Comprehensive performance evaluation."

### Major Comment 5 (Reviewer Point 6: Section 3.2.2 "Comprehensive performance evaluation")
**Reviewer Comment:** "In Section 3.2.2 Comprehensive performance evaluation, the last sentence at the bottom of page 20 needs to be reorganized; for example, the phrase should be 'minimal benefit under MNAR'? Moreover, this trend is not limited to MNAR, but also MAR and MCAR for MIMIC-III data. For AMI data, it may help readers better understand the results if the authors can describe which imputation methods are beneficial from including missing indicators."

**Response:** We have reorganized the corresponding paragraph in Section 3.2.2 "Comprehensive performance evaluation" to reflect that the observed trends in MIMIC-III (minimal benefit from indicators) are consistent across MCAR, MAR, and MNAR mechanisms due to sample size constraints ($p/n \approx 0.6$). We also explicitly clarify that for the larger AMI dataset, indicators substantially improve recovery under MNAR, particularly when combined with KNN and missForest imputation.

**Revisions in Manuscript:** See reorganized Section 3.2.2 "Comprehensive performance evaluation".

### Major Comment 6 (Reviewer Point 7: 770% Calculation)
**Reviewer Comment:** "At the top of page 23, how did the authors arrive at the number 770%? More explanations are needed."

**Response:** We have clarified this figure in the manuscript. The 770% refers to the Relative MSE of the best-performing method (KNN) in the AMI MCAR scenario (Table 7: Comprehensive Simulation Results for AMI Dataset, KNN MCAR w/o mi, Rel MSE = 872.35%), which indicates that coefficient error magnitude significantly exceeds the true effect size. We have added a parenthetical reference to Section 3.2.2 "Comprehensive performance evaluation" to make this calculation transparent.

**Revisions in Manuscript:** Added parenthetical calculation reference to Section 3.2.2 "Comprehensive performance evaluation".

### Major Comment 7 (Reviewer Point 8: Comparison Benchmarks)
**Reviewer Comment:** "In the same paragraph at the top of page 23, the authors stated 'MAR caused complete failure (47% worse AUC, sensitivity dropping from 80-100% to 0-7.5%). MNAR restored strong performance with missing indicators improving precision by 90%. Mean imputation with indicators achieved best coefficient recovery under MNAR (15% better relative MSE).' Do these statements and the reported numbers come from a comparison with the results under MCAR? The specification is needed. Meanwhile, the authors may need to clarify what benchmarks were used for comparisons that led to the given statement."

**Response:** We have clarified the comparison benchmarks in the manuscript. Specifically: (1) mechanism-based comparisons (e.g., "47% worse AUC") are relative to the near-perfect performance under the MCAR baseline for the same dataset, and (2) indicator-based benefits (e.g., "improving precision by 90%") are compared directly to models without indicators within the same missingness mechanism. We have added parenthetical clarifications to Section 3.2.2 "Comprehensive performance evaluation" to ensure these reference points are explicit.

**Revisions in Manuscript:** Added baseline clarifications to Section 3.2.2 "Comprehensive performance evaluation".

### Major Comment 8 (Reviewer Point 9: Section 3.2.4 "Error rate analysis")
**Reviewer Comment:** "In Section 3.2.4 Error rate analysis, the authors stated 'MCAR and MNAR enabled near-perfect true variable identification.' However, it contradicts with the observation that type II error under MCAR and MNAR looks uncontrolled."

**Response:** We agree that our previous characterization was overgeneralized. We have revised Section 3.2.4 "Error rate analysis" to distinguish between datasets: while the larger AMI dataset achieves near-perfect identification (Sensitivity 0.87-1.00) under MCAR and MNAR, the high-missingness MIMIC-III regime shows significantly higher Type II errors (Sensitivity 0.40-0.60). We have also removed the redundant/contradictory sentence identified in Section 3.2.4 "Error rate analysis".

**Revisions in Manuscript:** See updated Section 3.2.4 "Error rate analysis".

### Major Comment 9 (Reviewer Point 10: 400-fold Calculation)
**Reviewer Comment:** "At the bottom of page 24, how did the authors get 400-fold higher bias? Please clarify."

**Response:** We have added the requested calculation to the manuscript to clarify this figure. The "400-fold" refers to the ratio between Mean Imputation Relative Bias (41,416.90%) and MICE Relative Bias (99.22%) in the MIMIC-III MNAR scenario (Table 6: Comprehensive Simulation Results for MIMIC-III Dataset). We have added a parenthetical comparison (Mean 41,416% vs MICE 99%) to Section 3.2.5 "Validation of real data findings" to ensure this is explicit.

**Revisions in Manuscript:** Added parenthetical calculation to Section 3.2.5 "Validation of real data findings".

### Major Comment 10 (Reviewer Point 11: Definitions of MAR and MNAR)
**Reviewer Comment:** "In '6. Mechanism matters more than method' at page 25, the authors stated 'MAR caused large drops in predictive performance across all methods, demonstrating that no imputation approach can fully compensate when missingness depends on observed values.' This statement contradicts good predictive performance under MNAR, which depicts the missingness also dependent on observed values. The same comment applies to the statement 'Under MAR, substantial performance degradation across all methods demonstrates that no imputation approach can fully compensate when missingness depends on observed values.' in Section 4.1 Interpretation of simulation findings."

**Response:** We have corrected the terminology throughout the manuscript and Appendix A. We now precisely define MAR as dependency on other observed covariates (not the missing value itself) and MNAR as dependency on the unobserved value of the missing variable itself.

**Revisions in Manuscript:** See Sections 2.4.3 "Rationale for including missing indicators", 3.2.1 "Coefficient recovery", 3.2.2 "Comprehensive performance evaluation", 3.2.4 "Error rate analysis", 4.1 "Interpretation of simulation findings", and Appendix A.

### Minor Comment 2 (Reviewer Point 12: Section 4.3 "Context-specific recommendations")
**Reviewer Comment:** "At the top of page 27, the point is not clearly conveyed in the sentence 'Achieving strong predictive performance on both datasets, the simulation’s elevated Type I errors do not preclude prediction applications in Bayesian variable selection frameworks.' The authors may consider rephrasing it."

**Response:** We have rephrased the sentence in Section 4.3 "Context-specific recommendations" to clarify that while elevated Type I errors indicate a risk of false-positive variable selection, this does not preclude the use of these models in prediction-focused applications where overall accuracy is prioritized.

**Revisions in Manuscript:** See Section 4.3 "Context-specific recommendations".

---

## Response to Reviewer 3

We thank Reviewer 3 for the detailed and helpful feedback. We have addressed the technical and notational points raised to improve the precision of our manuscript.

### Major Comment 1 (Correlation structure and matrices)
**Reviewer Comment:** "Variable selection is one of the important aspects of this study. The author could discuss the correlation structure of predictor variables by providing correlation matrix for each dataset. On page 13, the authors indicated that there were only four predictors with highest missingness rates for each missing data mechanism and 48 noise variables in the MIMIC-III data. It would be helpful if the correlation structure between the four variables with the highest missingness and the 48 noise variables were presented. This would help readers understand how noise variables were determined. The same clarification would be helpful for the AMI data."

**Response:** We have addressed this comment in two parts:
1. **Simulation Analysis:** We have added a technical clarification to Section 2.7 "Simulation study design" (Subsection 2.7.1 "Data generation"). "Noise variables" are defined as predictors with $\beta_{true} = 0$, generated as independent Gaussian random vectors. Consequently, their correlation with the signal variables and the outcome is near-zero ($r \approx 0$) by design, ensuring they represent purely irrelevant features.
2. **Real Data Correlations:** We have added Appendix H ("Correlation structure of clinical predictors") with a summary table of the strongest pairwise correlations ($|r| \ge 0.70$) for both datasets (Table H.1). This quantitative approach directly addresses the reviewer's request regarding the relationship between high-missingness variables and noise predictors while ensuring maximum readability. The appendix discusses observed dependencies, including physiological clusters in MIMIC-III (e.g., Hematocrit and Hemoglobin, $r=0.95$) and hemodynamic dependencies in AMI. Full correlation matrices are provided in the GitHub repository as CSV files.

**Revisions in Manuscript:** See Section 2.7 "Simulation study design" (Subsection 2.7.1 "Data generation") and Appendix H (Table H.1).

### Major Comment 2 (Section 2.2.1 "Bayesian adaptive sampling (BAS)")
**Reviewer Comment:** "On page 4, the authors wrote that in BAS, the linear predictor for observation i is given as z_i= β_(0 )+ ∑_(jϵ S)▒β_j x_ij, an equation (1) in the manuscript. It is unclear if z_i denotes the indicator variable for subject i for covariate j although z_i does not have a subscript j. The authors could specify if the linear predictor corresponds to a created indicator variable subsequently used in the study. Did they mean yi instead of zi?"

**Response:** We thank the reviewer for identifying this notational ambiguity. We have changed the notation for the linear predictor from $z_i$ to $\eta_i$ to clearly distinguish it from the missing indicator matrix $Z$ used later in the study. We have also added a clarifying sentence to explicitly define $\eta_i$ as the log-odds of the outcome $y_i=1$.

**Revisions in Manuscript:** See updated Equation 1 and the accompanying text in Section 2.2 "Bayesian variable selection methods." We have also standardized the notation across the manuscript, including updating simulation variables in Appendix F from $x_i$ to $x_i$ to avoid any overlap with the missing indicator matrix $Z$.

### Major Comment 3 (Variable-level missingness)
**Reviewer Comment:** "The authors reported that the datasets MIMIC-III and AMI have 64% and 6.15% overall missingness, respectively. However, the paper did not provide details regarding the percentage of missingness for each individual variable. The authors could provide a table with varying percentage of missingness for each of the predictors in Supplement."

**Response:** We have added a new Appendix (Appendix G: "Variable-level missingness statistics") containing Table G.4 and Table G.5. These tables provide the precise missingness percentage for every predictor included in the analysis for both the MIMIC-III and AMI datasets. We have ensured these tables strictly adhere to our inclusion criteria by excluding variables with >95\% missingness (e.g., CPK and Heredity in the AMI dataset), resulting in a finalized set of 121 analyzed covariates for the AMI dataset.

**Revisions in Manuscript:** See new Appendix G and Tables G.4 and G.5.

### Major Comment 4 (Bayesian vs. Frequentist context)
**Reviewer Comment:** "On page 26, section 4.2, the authors stated that imputation with Bayesian variable selection was not considered in the past while performing missing value imputation and only frequentist approaches have been adopted instead. Here are some references the authors may want to consult. The authors may want to consider comparing the Bayesian variable selection method with a frequentist approach, i.e., stepwise variable selection techniques, using Lasso etc."

**Response:** We thank the reviewer for this insightful comment. We have addressed it by:
1. **Qualifying Methodological Novelty:** We have revised Section 4.2 "Implications for method selection" to correctly qualify our contribution, focusing on the systematic evaluation of these methods specifically within the Bayesian variable selection framework rather than claiming total novelty.
2. **Comparison with Frequentist Approaches:** We agree that a direct comparison between Bayesian (BAS, BART) and frequentist (LASSO, stepwise selection) approaches is highly valuable. While a full side-by-side evaluation is beyond the current scope of this study, we have added this comparison as a primary direction for future research in Section 4.5 "Limitations and future directions". We have specifically noted that while LASSO provides "hard" binary selections, BVS provides continuous Posterior Inclusion Probabilities (PIPs) that naturally quantify selection uncertainty.

**Revisions in Manuscript:** See updated Section 4.2 "Implications for method selection" and Section 4.5 "Limitations and future directions".

### Major Comment 5 (Calculation transparency)
**Reviewer Comment:** "On page 18, the authors showed that for MIMIC-III data with missing indicators mean imputation achieved perfect AUC, a 45% improvement over mean alone while without indicators MICE, missForest, and KNN outperformed mean imputation by 44%. For AMI data, mean with indicators had a 27% better F1 score. The authors could add a few sentences here to clarify how they achieved those performance percentage changes amongst various imputation methods."

**Response:** We have added parenthetical calculation details for all relative improvement and inflation percentages mentioned in the results and discussion (including 45%, 44%, 27%, 770%, 47%, and 90%). We explicitly provide the starting and ending values (e.g., AUC: 1.00 vs 0.69) and the calculation formula used to arrive at the reported percentage, ensuring full transparency of our quantitative claims.

**Revisions in Manuscript:** See updated parenthetical details in Section 3.1.1 "BAS performance" and Section 3.2.3 "Sensitivity Analysis: Amplified Missingness Effect ($\alpha=10$)".

### Major Comment 6 (Appendix B - Choice of k)
**Reviewer Comment:** "The authors could elaborate how k =13 for MIMIC-III and k= 41 for AMI data were chosen for KNN (Appendix B). When KNN is adopted for imputation purposes, it is imperative that the choice of k be justified."

**Response:** We have added the requested justification to Appendix B. The choices of $k=13$ (MIMIC-III) and $k=41$ (AMI) were derived from the standard heuristic $k = \sqrt{n}$, where $n$ is the total number of observations in each dataset ($n=168$ and $n=1,699$, respectively). This heuristic is widely used to balance the trade-off between local bias (small $k$) and global variance (large $k$). We chose this consistent rule to ensure reproducibility and provide a stable baseline for our cross-method comparison.

**Revisions in Manuscript:** See expanded text following Table B.1 in Appendix B "Implementation details."

### Minor Comment 1 (Section 4.5 "Limitations and future directions")
**Reviewer Comment:** "The authors focused solely on missing predictors and used complete-case analysis for the outcomes. They should discuss this limitation and the potential for joint imputation of predictors and outcomes in future work. The authors stated that in clinical studies the outcome variables are not affected with missing values. This may not always be true in practice. Therefore, they could make a comment on how in a situation (outcome variable with missing values) they would be handled along with predictors with missing values."

**Response:** We have addressed this by expanding Section 4.5 "Limitations and future directions". We acknowledge that while our specific clinical datasets (MIMIC-III and AMI) had complete outcome data, this represents a limitation that may not generalize to all clinical studies. We have added a comment on established statistical approaches for handling missing outcomes, including joint multiple imputation of both predictors and outcomes, inverse probability weighting (IPW) to adjust for selection bias, and the conditions of mechanism-specific validity for complete-case analysis. We have identified these as critical directions for future work to extend the Bayesian variable selection framework to more complex data dependencies.

**Revisions in Manuscript:** See expanded Section 4.5 "Limitations and future directions".

### Minor Comment 2 (Section 2.3 "Imputation methods")
**Reviewer Comment:** "While mean imputation was conducted, the authors mentioned that the missing values were replaced by the column means for the continuous predictors. It is not clear whether there were significant categorical predictors. If so, then the authors should clarify how they handled imputations for those predictors."

**Response:** We clarified this in Section 2.3 "Imputation methods" by adding a "Mean imputation" subsection. MIMIC-III uses continuous variables; AMI includes binary and categorical predictors. For mean imputation, categorical variables were coded as numeric (0/1) and missing values replaced with column means.

**Revisions in Manuscript:** See new "Mean imputation" subsection in Section 2.3 "Imputation methods".

### Minor Comment 3 (Section 2.3 "Imputation methods")
**Reviewer Comment:** "The authors evaluated four missing imputation techniques (mean, missForest, MICE, and KNN) on Bayesian variable selection. This reviewer did not find justification in the manuscript as to why these four methods were chosen, please clarify."

**Response:** We added a paragraph to Section 2.3 "Imputation methods" justifying the selection of these four methods. We chose a spectrum of approaches ranging from simple baselines (Mean) to parametric gold standards (MICE), machine learning alternatives (missForest), and instance-based similarity (KNN). This selection enables comparison across different statistical frameworks with Bayesian variable selection.

**Revisions in Manuscript:** See new introductory paragraph in Section 2.3 "Imputation methods".

### Minor Comment 4 (Section 2.2.1 "Bayesian adaptive sampling (BAS)")
**Reviewer Comment:** "The authors did not explicitly specify what robust g-prior does as opposed to some informative or non-informative priors do. They could clarify how robust g-prior excels in achieving better posterior predictive probabilities than some other priors."

**Response:** We have expanded the description and technical justification of the robust $g$-prior in Section 2.2.1 "Bayesian adaptive sampling (BAS)". We now explicitly state that the robust $g$-prior—a mixture of $g$-priors—is used to ensure consistency in variable selection and improve posterior calibration by adapting to the signal-to-noise ratio in the data. We also highlight its advantages over fixed $g$-priors or non-informative uniform priors, specifically its ability to guard against prior-data conflict in high-dimensional settings without the subjectivity of informative priors.

**Revisions in Manuscript:** See expanded text in Section 2.2.1 "Bayesian adaptive sampling (BAS)".

**Reviewer Comment:** "On page 24, point 2 under section 3.2.5, the authors claimed that mean imputation is better for the prediction of risk stratification than the other imputation methods. This statement is strong in the sense that it provides reliable estimates for the prediction of the missing values. However, mean imputation is known to produce large variance for the estimates. Did the authors consider addressing this aspect of large variance?"

While this compression of variance and the resulting biased coefficients are acceptable for empirical risk stratification focused solely on predictive discrimination (AUC), they lead to underestimated standard errors and anti-conservative inference that are fundamentally unsuitable for effect estimation. We have reinforced our recommendation in Sections 4.2 "Implications for method selection" and 4.3 "Context-specific recommendations" that Multiple Imputation (MICE) should be the preferred approach when the goal is identifying true predictors or obtaining unbiased parameter estimates.

**Revisions in Manuscript:** See updated discussion in Section 4.2 "Implications for method selection".

### Minor Comment 6 (Section 3.2.5 "Validation of real data findings")
**Reviewer Comment:** "On page 24, point 6 under section 3.2.5, the authors reported that no imputation approach can fully compensate when missingness occurs at random (MAR). The authors can suggest that as far as the imputation approaches, they adopted in this study and the clinical datasets they have, MAR cannot be fully compensated—having said that MAR can be managed properly by appropriate imputation techniques which may be out of scope of this study."

**Response:** We qualified this in Sections 4.1 "Interpretation of simulation findings", 4.3 "Context-specific recommendations", and 4.4 "Practical considerations". Our findings on MAR failure are limited to the methods and datasets evaluated. While specialized MAR techniques exist, they were beyond the scope of this study.

**Revisions in Manuscript:** See updated text in Section 3.2.5 "Validation of real data findings" (Point 6) and Sections 4.1 "Interpretation of simulation findings" and 4.3 "Context-specific recommendations".

---

Sincerely,

Nads A. and Andrade D.
