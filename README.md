# Cervical Cancer Risk Factors — Supervised Learning Project

## From Risk Factors to Diagnosis: Predicting Cervical Biopsy Outcome with Supervised Machine Learning

**UM5BM753 — M2 AIDA 2026/2027**

This project investigates whether **cervical biopsy outcome** can be predicted from upstream risk factors alone, or whether most predictive performance comes from screening information closer to diagnosis.

## Research question

> **Can biopsy outcome be predicted from risk factors alone, or is performance mainly driven by screening information?**

Two scenarios were studied:

1. **Risk factors only**
2. **Risk factors + screening tests** (`Hinselmann`, `Schiller`, `Citology`)

## Dataset

UCI **Cervical Cancer (Risk Factors)** dataset:

- **858 patients**
- **36 variables**
- **55 positive biopsies** (~6.4%)

Because the positive class is rare, **accuracy is misleading**.  
The primary metric is **Average Precision (PR-AUC)**, with Recall, Precision, F1, Balanced Accuracy, ROC-AUC and MCC as complementary metrics.

## Feature selection

After preprocessing:

```text
32 variables
→ 26 after low-variance filtering
→ consensus across 4 selection methods
→ 10 final features after collinearity pruning
```

Methods used:

- Fisher exact / Mann–Whitney tests
- Mutual Information
- L1-regularized Logistic Regression
- Random Forest permutation importance

Final selected features:

```text
Age
Age 1st intercourse
Hormonal Contraceptives
Hormonal Contraceptives (years)
STDs (number)
Dx
Dx:CIN
Hinselmann
Schiller
Citology
```

## Class imbalance

Class weighting and SMOTE were compared.

| Scenario | Class weighting | SMOTE |
|---|---:|---:|
| Risk factors only | **0.15 ± 0.09** | 0.12 ± 0.08 |
| Risk + screening | **0.72 ± 0.16** | 0.63 ± 0.16 |

**Class weighting was retained**.

## Model benchmark

Models tested:

- Dummy classifier
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- XGBoost

For deeper analysis, **Logistic Regression** and **Random Forest** were retained because they represent two complementary assumptions:

- **Logistic Regression:** linear and interpretable
- **Random Forest:** nonlinear and interaction-aware

## Nested cross-validation

- **Inner CV:** hyperparameter tuning
- **Outer repeated CV:** performance estimation
- **5 folds × 5 repeats** for the outer evaluation

### Logistic Regression

```text
StandardScaler
class_weight = "balanced"
solver = "liblinear"
max_iter = 5000
regularization = L1 or L2
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

### Random Forest

```text
n_estimators = 500
bootstrap = True
criterion = "gini"
class_weight = "balanced"

max_depth = [None, 3, 5, 8, 12]
min_samples_leaf = [1, 2, 4, 8, 12]
max_features = ["sqrt", 0.5, 0.75, 1.0]
```

## Main results

### Scenario 1 — Risk factors only

| Model | PR-AUC |
|---|---:|
| Logistic Regression | **0.138** |
| Random Forest | 0.132 |

Both models remain weak: increasing model complexity does not recover a strong predictive signal.

### Scenario 2 — Risk factors + screening tests

| Model | PR-AUC |
|---|---:|
| Logistic Regression | **0.705** |
| Random Forest | 0.704 |

Performance improves strongly once screening information is added, while the two models remain almost equivalent.

> **Main result: predictor information matters more than algorithmic complexity.**

## Final model

The final model is an **L2-regularized Logistic Regression** using risk factors + screening tests.

Final settings:

```text
class_weight = "balanced"
regularization = L2
decision threshold = 0.61
```

### Held-out test

| Metric | Value |
|---|---:|
| PR-AUC | **0.720** |
| Recall | **0.875** |
| Precision | **0.583** |
| MCC | **0.692** |

The model detected **14 of 16 positive biopsies**, with **2 false negatives**.

## Biological interpretation

Risk factors alone contain limited information for predicting current biopsy status.

`Hinselmann`, `Schiller` and `Citology` are much closer to cervical abnormalities and therefore provide a much stronger predictive signal.

The screening-inclusive model should **not** be interpreted as an early-risk prediction model based only on epidemiological factors.

## Limitations

- strong class imbalance
- small number of positive biopsies
- assumptions in missing-value handling
- screening variables are clinically close to the biopsy outcome
- predictive, not causal, interpretation
- statistically optimized threshold, not clinically validated
- no external validation cohort

## Reproducibility

```bash
source MLproject.venv/bin/activate
python -m pip install -r requirements.txt
```

Workflow:

```text
preprocessing
→ class imbalance
→ feature selection
→ benchmark
→ Logistic Regression deep dive
→ Random Forest deep dive
→ final comparison
→ held-out test
```

## Authors

Mariam H. · Claire D. · Azzedine A.  
M2 AIDA — UM5BM753 — 2026/2027
