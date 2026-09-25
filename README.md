# Cervical Cancer Biopsy Prediction - [UM5BM753](https://moodle-sciences-26.sorbonne-universite.fr/course/view.php?id=25 "UM5BM753 - Supervised and Unsupervised Learning for Biological Data") Supervised Learning Project

## Project overview

This project investigates supervised machine-learning approaches for predicting the **`Biopsy`** outcome in the UCI **Cervical Cancer (Risk Factors)** dataset.

The objective was to:

-   construct a reproducible preprocessing and modelling workflow;
-   compare several supervised-learning methods;
-   investigate the most relevant approaches in greater depth;
-   justify the final analytical strategy using predictive performance, stability, complexity and biological interpretability.

A central methodological choice was to distinguish two prediction settings:

1.  **Selected risk factors only** : prediction from upstream patient characteristics and risk factors.
2.  **Selected + diagnostics** : prediction after adding screening variables such as `Hinselmann`, `Schiller` and `Citology`.

These scenarios answer different biological questions and are therefore kept separate throughout the project.

------------------------------------------------------------------------

## Dataset

The project uses the UCI **Cervical Cancer (Risk Factors)** dataset.

### Initial data

-   **858 observations**
-   **36 variables**

### After preprocessing

-   **835 observations**
-   **32 variables**

The preprocessing workflow included:

-   conversion of `?` to missing values;
-   numeric conversion;
-   removal of **23 exact duplicate rows**;
-   removal of two variables with approximately 92% missing values:
    -   `STDs: Time since first diagnosis`
    -   `STDs: Time since last diagnosis`
-   removal of two constant variables:
    -   `STDs:cervical condylomatosis`
    -   `STDs:AIDS`
-   median imputation for selected continuous variables;
-   zero imputation for several binary exposure variables;
-   creation of missingness indicators for selected variables.

The zero-imputation strategy is retained as a limitation because an unknown exposure value may not always be equivalent to a true absence of exposure.

------------------------------------------------------------------------

## Prediction target and class imbalance

The target is:

``` text
Biopsy
```

After preprocessing:

-   **781 negative biopsies**
-   **54 positive biopsies**
-   positive prevalence ≈ **6.5%**

The dataset is therefore strongly imbalanced.

A classifier predicting every patient as negative would already reach more than 93% accuracy. For this reason, **accuracy is not used as the main model-selection metric**.

The primary metric is **Average Precision (AP)**, stored in the notebooks under the name `PR_AUC`.

Complementary metrics include:

-   ROC-AUC
-   Recall
-   Precision
-   F1-score
-   Balanced accuracy
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Train / test strategy

A stratified **70/30 train-test split** was created before final model evaluation.

Approximate sizes:

-   **Training set:** 584 observations
-   **Held-out test set:** 251 observations

The final test set contains:

-   **235 negative biopsies**
-   **16 positive biopsies**

The held-out test set is not used during:

-   feature selection,
-   benchmark comparison,
-   hyperparameter tuning,
-   final model selection,
-   decision-threshold selection.

It is used only once after all modelling decisions have been frozen.

------------------------------------------------------------------------

# Prediction scenarios

## 1. Selected risk factors only

Diagnostic screening variables are excluded in order to ask:

> Can biopsy outcome be predicted from upstream patient characteristics and risk factors alone ?

After feature selection, 13 predictors are retained:

``` text
Smokes
Hormonal Contraceptives
IUD
STDs (number)
STDs:syphilis
STDs:HIV
Dx:CIN
Dx:HPV
Dx
smokes_missing
hormonal_contraceptives_missing
number_of_sexual_partners_missing
num_of_pregnancies_missing
```

Training shape:

``` text
584 × 13 predictors
```

------------------------------------------------------------------------

## 2. Selected + diagnostics

Screening variables are allowed in order to ask:

> How accurately can biopsy outcome be predicted when diagnostic information is already available?

After feature selection, 6 predictors are retained:

``` text
STDs:syphilis
Dx:CIN
Hinselmann
Schiller
Citology
hormonal_contraceptives_missing
```

Training shape:

``` text
584 × 6 predictors
```

The smaller feature set is expected because `Schiller`, `Hinselmann` and `Citology` already contain strong information related to the biopsy outcome.

------------------------------------------------------------------------

# Feature selection

Feature selection is performed on the **training data only**.

The workflow includes :

-   low-variance filtering,
-   correlation analysis,
-   mutual information,
-   RFECV,
-   multicollinearity / VIF analysis,
-   FDR-based statistical filtering.

One limitation is that RFECV uses Logistic Regression as its estimator. The resulting subsets may therefore partly favour linear models. This is explicitly considered when comparing Logistic Regression with Random Forest.

------------------------------------------------------------------------

# Class imbalance strategy

Two approaches were considered:

-   class weighting,
-   SMOTE.

Class weighting performed better in the preliminary analysis and was retained.

For the main models:

``` python
class_weight="balanced"
```

is used.

------------------------------------------------------------------------

# Supervised model benchmark

The benchmark compared :

-   Dummy classifier,
-   Logistic Regression,
-   Decision Tree,
-   Random Forest,
-   Support Vector Machine,
-   Gaussian Naive Bayes,
-   XGBoost

The benchmark was used to identify methods worth studying in depth rather than to maximize the number of tested algorithms.

Two models were retained:

-   **Logistic Regression,**
-   **Random Forest**

These methods are complementary:

-   Logistic Regression is simple, regularized and directly interpretable through its coefficients.
-   Random Forest can represent nonlinear relationships and interactions without explicitly specifying them.

------------------------------------------------------------------------

# Logistic Regression deep dive

The Logistic Regression analysis investigates:

-   `StandardScaler` inside a sklearn `Pipeline`,
-   L1 versus L2 regularization,
-   regularization strength `C`,
-   class weighting,
-   nested cross-validation,
-   coefficient stability,
-   decision-threshold optimization

Standardization is included inside the pipeline so that scaling is learned only from each training fold and does not introduce data leakage.

The tested regularization strengths are:

``` python
np.logspace(-3, 2, 6)
```

corresponding to:

``` text
0.001, 0.01, 0.1, 1, 10, 100
```

With the scikit-learn version used in the project:

``` text
l1_ratio = 0.0 → L2
l1_ratio = 1.0 → L1
```

The diagnostic-inclusive analysis consistently favours **L2 regularization**.

Coefficient analysis shows that:

-   `Schiller`
-   `Hinselmann`
-   `Citology`

have consistently positive predictive contributions, with `Schiller` having the strongest coefficient in the final diagnostic-inclusive model.

These coefficients are interpreted as **predictive associations**, not causal effects.

------------------------------------------------------------------------

# Random Forest deep dive

The Random Forest analysis focuses on parameters controlling model complexity:

-   `max_depth`
-   `min_samples_leaf`
-   `max_features`

The number of trees is fixed at:

``` python
n_estimators = 500
```

because increasing the number of trees beyond a sufficiently large value mainly improves stability rather than model flexibility.

A randomized hyperparameter search is used to avoid an unnecessarily large exhaustive grid.

Permutation importance is used for interpretation.

In the diagnostic-inclusive scenario, `Schiller` strongly dominates the Random Forest prediction, while `Hinselmann`, `Citology` and the remaining variables provide smaller and less stable contributions.

------------------------------------------------------------------------

# Nested cross-validation

Hyperparameter optimization and performance estimation are separated using **nested cross-validation**:

-   **inner CV** → hyperparameter selection;
-   **outer CV** → unbiased performance estimation.

The outer procedure uses:

``` text
5 folds × 5 repeats = 25 evaluations
```

This reduces optimistic bias from tuning and evaluating on the same folds.

------------------------------------------------------------------------

# Optimized model comparison

## Risk factors only

| Metric            | Logistic Regression |     Random Forest |
|-------------------|--------------------:|------------------:|
| AP / PR-AUC       |   **0.129 ± 0.035** |     0.119 ± 0.034 |
| ROC-AUC           |   **0.613 ± 0.096** |     0.605 ± 0.098 |
| Recall            |   **0.437 ± 0.181** |     0.331 ± 0.183 |
| Precision         |       0.102 ± 0.039 | **0.123 ± 0.060** |
| F1                |       0.161 ± 0.060 |     0.161 ± 0.059 |
| Balanced accuracy |   **0.572 ± 0.095** |     0.556 ± 0.062 |
| MCC               |       0.083 ± 0.101 |     0.084 ± 0.083 |

Both models perform poorly. The small differences between algorithms are much less important than the overall lack of predictive signal.

Increasing model complexity does not compensate for the limited information available in the selected upstream risk factors.

------------------------------------------------------------------------

## Selected + diagnostics

| Metric            | Logistic Regression |     Random Forest |
|-------------------|--------------------:|------------------:|
| AP / PR-AUC       |   **0.699 ± 0.171** |     0.685 ± 0.167 |
| ROC-AUC           |   **0.944 ± 0.065** |     0.943 ± 0.054 |
| Recall            |   **0.905 ± 0.097** |     0.869 ± 0.112 |
| Precision         |       0.545 ± 0.099 | **0.627 ± 0.118** |
| F1                |       0.675 ± 0.087 | **0.722 ± 0.095** |
| Balanced accuracy |   **0.925 ± 0.049** |     0.915 ± 0.056 |
| MCC               |       0.674 ± 0.090 | **0.713 ± 0.101** |

Both models perform substantially better once diagnostic screening variables are included.

Logistic Regression has:

-   slightly higher AP / PR-AUC,
-   slightly higher ROC-AUC,
-   higher recall,
-   higher balanced accuracy

Random Forest has:

-   higher precision,
-   higher F1,
-   higher MCC

The performance differences remain modest relative to cross-validation variability.

------------------------------------------------------------------------

# Final model selection

**Logistic Regression is retained for both scenarios.**

The final choice is based primarily on Average Precision, together with:

-   recall,
-   stability,
-   complexity,
-   interpretability

For the diagnostic-inclusive scenario, Logistic Regression has a slightly higher mean AP / PR-AUC than Random Forest (**0.699 vs 0.685**) while remaining substantially simpler and easier to interpret.

The Random Forest does not provide a sufficiently large predictive advantage to justify its additional complexity.

This is a **parsimony-based analytical choice**, not a claim that Random Forest is statistically inferior.

------------------------------------------------------------------------

# Decision-threshold selection

After the final Logistic Regression configurations are frozen, the classification thresholds are selected using **out-of-fold predictions from the training data only**.

Final frozen thresholds:

``` text
Risk factors only       → 0.52
Selected + diagnostics  → 0.58
```

These thresholds are then applied unchanged to the held-out test set.

The thresholds maximizing MCC are statistical operating points, not clinically validated decision thresholds.

------------------------------------------------------------------------

# Final held-out test evaluation

## Risk factors only — Logistic Regression

Final test metrics:

| Metric            | Value |
|-------------------|------:|
| Threshold         |  0.52 |
| AP / PR-AUC       | 0.336 |
| ROC-AUC           | 0.743 |
| Recall            | 0.625 |
| Precision         | 0.167 |
| F1                | 0.263 |
| Balanced accuracy | 0.706 |
| MCC               | 0.236 |

Final confusion matrix:

``` text
TN = 185
FP = 50
FN = 6
TP = 10
```

The test result is better than the nested-CV average, but precision remains low and the model generates many false positives.

Because only 16 positive biopsies are present in the test set, this relatively favourable test score should be interpreted cautiously.

The repeated nested-CV results remain the stronger evidence that prediction from risk factors alone is weak and unstable.

------------------------------------------------------------------------

## Selected + diagnostics — Logistic Regression

Final test metrics:

| Metric            |     Value |
|-------------------|----------:|
| Threshold         |      0.58 |
| AP / PR-AUC       | **0.630** |
| ROC-AUC           | **0.932** |
| Recall            | **0.875** |
| Precision         | **0.583** |
| F1                | **0.700** |
| Balanced accuracy | **0.916** |
| MCC               | **0.692** |

Final confusion matrix:

``` text
TN = 225
FP = 10
FN = 2
TP = 14
```

The final model therefore detects:

``` text
14 / 16 positive biopsies
```

while producing only 10 false-positive predictions.

The held-out results are close to the nested-CV estimates:

| Metric            |     Nested CV | Final test |
|-------------------|--------------:|-----------:|
| AP / PR-AUC       | 0.699 ± 0.171 |      0.630 |
| ROC-AUC           | 0.944 ± 0.065 |      0.932 |
| Recall            | 0.905 ± 0.097 |      0.875 |
| Precision         | 0.545 ± 0.099 |      0.583 |
| F1                | 0.675 ± 0.087 |      0.700 |
| Balanced accuracy | 0.925 ± 0.049 |      0.916 |
| MCC               | 0.674 ± 0.090 |      0.692 |

This supports good **internal generalization** of the diagnostic-inclusive Logistic Regression.

------------------------------------------------------------------------

# Main scientific conclusion

> **Adding diagnostically relevant information has a much larger effect on predictive performance than increasing algorithmic complexity.**

Risk factors alone provide weak and unstable prediction for both Logistic Regression and Random Forest.

When diagnostic variables are included, both methods improve strongly.

The final analytical approach is therefore :

> **L2-regularized, class-weighted Logistic Regression using the selected diagnostic-inclusive feature set.**

This approach provides an appropriate compromise between :

-   predictive performance,
-   high sensitivity,
-   stability,
-   simplicity,
-   interpretability.

The diagnostic-inclusive model should not be interpreted as a model of upstream cervical-cancer risk. It predicts biopsy outcome partly from screening tests that are already clinically close to that outcome.

------------------------------------------------------------------------

# Repository structure

``` text
UM5BM753_Python_Project/
│
├── .venv/                         # local only; ignored by Git
│
├── 01_DATA/
│   ├── raw/
│   ├── processed/
│   └── selected/
│
├── 02_NOTEBOOKS/
│   ├── preprocessing notebooks
│   ├── 03_target_class_imbalance_python.ipynb
│   ├── 04_feature_selection_python.ipynb
│   ├── 05_benchmark_models_python.ipynb
│   ├── 06_logistic_regression_deep_dive.ipynb
│   ├── 07_random_forest_deep_dive.ipynb
│   └── 08_final_model_comparison.ipynb
│
├── 04_FIGURES/
├── 05_RESULTS/
│
├── README.md
├── requirements.txt
└── .gitignore
```

A dedicated `src/` directory is not required as this is a notebook-centred analysis.

------------------------------------------------------------------------

# Reproducibility

## Create the virtual environment

``` bash
python -m venv .venv
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

Windows:

``` bash
.venv\Scripts\activate
```

The project was developed with **Python 3.13**.

------------------------------------------------------------------------

## Install dependencies

``` bash
pip install -r requirements.txt
```

The working environment can be exported with:

``` bash
pip freeze > requirements.txt
```

------------------------------------------------------------------------

## Run the workflow

The notebooks should be executed sequentially because later stages depend on files generated earlier.

``` text
Preprocessing
→ class imbalance
→ feature selection
→ benchmark
→ Logistic Regression deep dive
→ Random Forest deep dive
→ final comparison
```

In numbered form:

``` text
03 → 04 → 05 → 06 → 07 → 08
```

after the preprocessing notebooks have been run.

------------------------------------------------------------------------

# Generated outputs

Figures are stored in:

``` text
04_FIGURES/
```

Numerical results are stored in:

``` text
05_RESULTS/
```

Examples:

``` text
logistic_nested_cv_risk.csv
logistic_nested_cv_diag.csv
logistic_best_hyperparameters.json

random_forest_nested_cv_risk.csv
random_forest_nested_cv_diag.csv
random_forest_best_hyperparameters.json

final_nested_cv_model_comparison.csv
final_model_selection.csv
final_held_out_test_results.csv
final_cv_vs_test_risk.csv
final_cv_vs_test_diag.csv
```

------------------------------------------------------------------------

# Methodological safeguards

The workflow includes several safeguards against optimistic bias and leakage:

-   stratified train/test split before final evaluation,
-   feature selection on training data only,
-   scaling inside sklearn pipelines,
-   nested cross-validation for tuning and performance estimation,
-   class weighting inside the fitted estimators,
-   threshold selection from out-of-fold training predictions,
-   final model selection before accessing the test set,
-   one-time held-out test evaluation,
-   no post-test retuning.

------------------------------------------------------------------------

# Limitations

### Small number of positive cases

Only 54 positive biopsies are available in the full cleaned dataset, and only 16 are present in the held-out test set.

Performance estimates therefore remain sensitive to sampling variability.

### Strong class imbalance

The positive class represents approximately 6.5% of observations. Accuracy is therefore misleading and is not used as the primary metric.

### Missing-value assumptions

Some missing binary exposures were imputed as zero, potentially conflating unknown exposure with true absence.

### Feature-selection dependence

RFECV used Logistic Regression as its estimator and may partly favour a feature subset suited to linear modelling.

### Diagnostic proximity to the outcome

`Schiller`, `Hinselmann` and `Citology` are much closer to biopsy outcome than upstream epidemiological risk factors.

The strong diagnostic-inclusive performance must therefore not be interpreted as equally strong prediction from risk factors alone.

### Predictive rather than causal interpretation

Logistic Regression coefficients and Random Forest permutation importance describe model behaviour, not causal biological effects.

### Statistical rather than clinical threshold

The selected probability thresholds maximize MCC on training OOF predictions. They are not clinically validated operating thresholds.

### No external validation

The held-out test set provides an internal hold-out evaluation only.

Independent external validation would be required before considering clinical application.

------------------------------------------------------------------------

# Key take-home message

> **The main limitation is the information available to the model, not the lack of algorithmic complexity.**

A simple, regularized Logistic Regression performs as well as the more complex Random Forest once informative screening variables are available.

The final diagnostic-inclusive Logistic Regression achieves strong held-out discrimination while remaining interpretable and parsimonious.

------------------------------------------------------------------------

## Authors

Supervised Learning project for **UM5BM753 - Supervised and Unsupervised Learning for Biological Data (M2 AIDA - 2026/2027)**.

Made by Mariam, Claire and Azzedine.
