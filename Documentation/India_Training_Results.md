# India Model Training Results

## Overview
This document contains the comprehensive results from training machine learning models on the India COVID-19 prediction dataset. Two datasets were used: the original imbalanced dataset (375 samples) and a SMOTE-balanced dataset (600 samples).

---

## Dataset 1: Original Dataset (375 Samples)

### Dataset Description
- **Total Samples**: 375 (300 negatives + 75 positives)
- **Class Distribution**: Imbalanced (1:4 ratio - 20% positive, 80% negative)
- **Train/Test Split**: 80:20 with stratification (300 train, 75 test)
- **Original Features**: 16 features initially present in raw data
- **Dropped Features**: 
  - First column (index/ID)
  - 'Others' column
  - **'Gender'** (demographic - not used for modeling)
  - **'Age'** (demographic - not used for modeling)
- **Final Feature Count**: 14 blood test parameters only

### Features Used in Modeling
The following 14 blood test features were available for SelectKBest feature selection:

1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Basophils(%)
8. Total RBC Count(millions/Cu)
9. HCT(%)
10. MCV(f L)
11. MCH(pg)
12. MCHC(gms%)
13. RDWCV(%)
14. Platelet Count(Lakh / Cumm)

### Grid Search Configuration
- **Feature Selection**: SelectKBest with f_classif scoring
- **k values tested**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] (all possible values)
- **Cross-Validation**: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- **Scoring Metric**: Accuracy
- **Hyperparameter Grids**:
  - **XGBoost**: 
    - n_estimators: [50, 100]
    - max_depth: [3, 5, 7]
    - learning_rate: [0.01, 0.1]
  - **AdaBoost**:
    - n_estimators: [50, 100]
    - learning_rate: [0.01, 0.1, 1.0]
  - **Random Forest**:
    - n_estimators: [50, 100]
    - max_depth: [3, 5, 7]
  - **Decision Tree**:
    - max_depth: [3, 5, 7, None]
    - criterion: ['gini', 'entropy']

---

## Model Results: Original Dataset (375 Samples)

### 1. XGBoost
**Best Hyperparameters**:
- select__k: 12
- model__learning_rate: 0.1
- model__max_depth: 5
- model__n_estimators: 100

**Selected Features (12)**:
1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Total RBC Count(millions/Cu)
8. MCV(f L)
9. MCH(pg)
10. MCHC(gms%)
11. RDWCV(%)
12. Platelet Count(Lakh / Cumm)

**Performance Metrics**:
- Train CV Accuracy: 0.9667 (96.67%)
- Test Accuracy: 0.9733 (97.33%)
- Sensitivity: 0.9333 (93.33%)
- Specificity: 0.9833 (98.33%)
- AUC Score: 0.9956 (99.56%)

---

### 2. AdaBoost
**Best Hyperparameters**:
- select__k: 5
- model__learning_rate: 0.1
- model__n_estimators: 100

**Selected Features (5)**:
1. Total WBC Count(/Cumm)
2. Neutrophils(%)
3. Lymphocytes(%)
4. Eosinophils(%)
5. MCHC(gms%)

**Performance Metrics**:
- Train CV Accuracy: 0.9567 (95.67%)
- Test Accuracy: 0.9733 (97.33%)
- Sensitivity: 0.8667 (86.67%)
- Specificity: 1.0000 (100.00%)
- AUC Score: 0.9744 (97.44%)

---

### 3. Random Forest
**Best Hyperparameters**:
- select__k: 11
- model__max_depth: 7
- model__n_estimators: 50

**Selected Features (11)**:
1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Total RBC Count(millions/Cu)
8. MCV(f L)
9. MCH(pg)
10. MCHC(gms%)
11. Platelet Count(Lakh / Cumm)

**Performance Metrics**:
- Train CV Accuracy: 0.9567 (95.67%)
- Test Accuracy: 0.9733 (97.33%)
- Sensitivity: 0.8667 (86.67%)
- Specificity: 1.0000 (100.00%)
- AUC Score: 0.9856 (98.56%)

---

### 4. Decision Tree
**Best Hyperparameters**:
- select__k: 14 (all features)
- model__criterion: entropy
- model__max_depth: 5

**Selected Features (14)** - All available features:
1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Basophils(%)
8. Total RBC Count(millions/Cu)
9. HCT(%)
10. MCV(f L)
11. MCH(pg)
12. MCHC(gms%)
13. RDWCV(%)
14. Platelet Count(Lakh / Cumm)

**Performance Metrics**:
- Train CV Accuracy: 0.9533 (95.33%)
- Test Accuracy: 0.9333 (93.33%)
- Sensitivity: 0.7333 (73.33%)
- Specificity: 0.9833 (98.33%)
- AUC Score: 0.8889 (88.89%)

---

### Confusion Matrix - Original Dataset
**XGBoost Model** (Best performing model):
```
Confusion Matrix:
[[59  1]
 [ 1 14]]

Actual values:
- True Negatives (TN): 59 (correctly predicted negatives)
- False Positives (FP): 1 (negatives incorrectly predicted as positive)
- False Negatives (FN): 1 (positives incorrectly predicted as negative)
- True Positives (TP): 14 (correctly predicted positives)

Total Test Samples: 75 (60 negatives, 15 positives)
Correctly Classified: 73/75 (97.33%)
Misclassified: 2/75 (2.67%)
```

---

## Dataset 2: SMOTE Balanced Dataset (600 Samples)

### Dataset Description
- **Total Samples**: 600 (300 negatives + 300 positives)
- **Class Distribution**: Balanced (1:1 ratio - 50% positive, 50% negative)
- **Generation Method**: SMOTE (Synthetic Minority Over-sampling Technique) with random_state=42
- **Train/Test Split**: 80:20 with stratification (480 train, 120 test)
- **Final Feature Count**: 14 blood test parameters (same as original dataset)

### Features Used in Modeling
Same 14 blood test features as the original dataset (listed above).

### Grid Search Configuration
Identical configuration to the original dataset:
- **Feature Selection**: SelectKBest with f_classif scoring
- **k values tested**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
- **Cross-Validation**: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- **Scoring Metric**: Accuracy
- **Hyperparameter Grids**: Same as original dataset (listed above)

---

## Model Results: SMOTE Dataset (600 Samples)

### 1. XGBoost
**Best Hyperparameters**:
- select__k: 8
- model__learning_rate: 0.1
- model__max_depth: 3
- model__n_estimators: 100

**Selected Features (8)**:
1. Total WBC Count(/Cumm)
2. Neutrophils(%)
3. Lymphocytes(%)
4. Eosinophils(%)
5. Monocytes(%)
6. Total RBC Count(millions/Cu)
7. MCH(pg)
8. MCHC(gms%)

**Performance Metrics**:
- Train CV Accuracy: 0.9604 (96.04%)
- Test Accuracy: 0.9833 (98.33%)
- Sensitivity: 0.9833 (98.33%)
- Specificity: 0.9833 (98.33%)
- AUC Score: 0.9994 (99.94%)

---

### 2. AdaBoost
**Best Hyperparameters**:
- select__k: 7
- model__learning_rate: 1.0
- model__n_estimators: 50

**Selected Features (7)**:
1. Total WBC Count(/Cumm)
2. Neutrophils(%)
3. Lymphocytes(%)
4. Eosinophils(%)
5. Monocytes(%)
6. Total RBC Count(millions/Cu)
7. MCHC(gms%)

**Performance Metrics**:
- Train CV Accuracy: 0.9625 (96.25%)
- Test Accuracy: 0.9833 (98.33%)
- Sensitivity: 0.9833 (98.33%)
- Specificity: 0.9833 (98.33%)
- AUC Score: 0.9961 (99.61%)

---

### 3. Random Forest
**Best Hyperparameters**:
- select__k: 11
- model__max_depth: 7
- model__n_estimators: 50

**Selected Features (11)**:
1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Basophils(%)
8. Total RBC Count(millions/Cu)
9. MCH(pg)
10. MCHC(gms%)
11. RDWCV(%)

**Performance Metrics**:
- Train CV Accuracy: 0.9625 (96.25%)
- Test Accuracy: 0.9750 (97.50%)
- Sensitivity: 0.9667 (96.67%)
- Specificity: 0.9833 (98.33%)
- AUC Score: 0.9983 (99.83%)

---

### 4. Decision Tree
**Best Hyperparameters**:
- select__k: 14 (all features)
- model__criterion: entropy
- model__max_depth: None (no limit)

**Selected Features (14)** - All available features:
1. Haemoglobin(gms%)
2. Total WBC Count(/Cumm)
3. Neutrophils(%)
4. Lymphocytes(%)
5. Eosinophils(%)
6. Monocytes(%)
7. Basophils(%)
8. Total RBC Count(millions/Cu)
9. HCT(%)
10. MCV(f L)
11. MCH(pg)
12. MCHC(gms%)
13. RDWCV(%)
14. Platelet Count(Lakh / Cumm)

**Performance Metrics**:
- Train CV Accuracy: 0.9438 (94.38%)
- Test Accuracy: 0.9500 (95.00%)
- Sensitivity: 0.9333 (93.33%)
- Specificity: 0.9667 (96.67%)
- AUC Score: 0.9500 (95.00%)

---

### Confusion Matrix - SMOTE Dataset
**XGBoost Model** (Best performing model):
```
Confusion Matrix:
[[59  1]
 [ 1 59]]

Actual values:
- True Negatives (TN): 59 (correctly predicted negatives)
- False Positives (FP): 1 (negatives incorrectly predicted as positive)
- False Negatives (FN): 1 (positives incorrectly predicted as negative)
- True Positives (TP): 59 (correctly predicted positives)

Total Test Samples: 120 (60 negatives, 60 positives)
Correctly Classified: 118/120 (98.33%)
Misclassified: 2/120 (1.67%)
```

---

## Key Observations

### Feature Selection Patterns

#### Original Dataset (375 samples):
1. **XGBoost** selected **12 features**, achieving the best AUC (99.56%) with a balanced approach
2. **AdaBoost** was the most selective with only **5 features**, yet achieved 97.33% test accuracy and 100% specificity
3. **Random Forest** used **11 features**, achieving perfect specificity (100%)
4. **Decision Tree** utilized all **14 features** but had the lowest sensitivity (73.33%)

#### SMOTE Dataset (600 samples):
1. **XGBoost** selected **8 features**, achieving exceptional performance (98.33% accuracy, 99.94% AUC)
2. **AdaBoost** selected **7 features**, matching XGBoost's test accuracy (98.33%)
3. **Random Forest** used **11 features**, slightly lower accuracy (97.50%) but still excellent AUC (99.83%)
4. **Decision Tree** used all **14 features** with the lowest performance (95.00% accuracy)

### Important Features Across Models

#### Original Dataset:
The most frequently selected features were:
- **Total WBC Count(/Cumm)** - Selected by all 4 models
- **Neutrophils(%)** - Selected by all 4 models
- **Lymphocytes(%)** - Selected by all 4 models
- **Eosinophils(%)** - Selected by all 4 models
- **MCHC(gms%)** - Selected by all 4 models
- **Monocytes(%)** - Selected by 3/4 models (XGBoost, RandomForest, DecisionTree)
- **Total RBC Count(millions/Cu)** - Selected by 3/4 models (XGBoost, RandomForest, DecisionTree)

#### SMOTE Dataset:
Core features consistently selected:
- **Total WBC Count(/Cumm)** - Selected by all 4 models
- **Neutrophils(%)** - Selected by all 4 models
- **Lymphocytes(%)** - Selected by all 4 models
- **Eosinophils(%)** - Selected by all 4 models
- **MCHC(gms%)** - Selected by all 4 models
- **Monocytes(%)** - Selected by all 4 models
- **Total RBC Count(millions/Cu)** - Selected by all 4 models

**Clinical Insight**: The white blood cell differential counts (Neutrophils, Lymphocytes, Eosinophils, Monocytes) and MCHC are the most discriminative features for COVID-19 prediction across both datasets.

### Performance Comparison: Original vs SMOTE

| Metric | Original Dataset (Best: XGBoost/AdaBoost/RF) | SMOTE Dataset (Best: XGBoost/AdaBoost) |
|--------|----------------------------------------------|----------------------------------------|
| Test Accuracy | 97.33% | 98.33% |
| Sensitivity | 93.33% (XGBoost) | 98.33% (XGBoost, AdaBoost) |
| Specificity | 100% (AdaBoost, RF) | 98.33% (XGBoost, AdaBoost, RF) |
| AUC | 99.56% (XGBoost) | 99.94% (XGBoost) |

**Key Findings**:
1. **SMOTE improved sensitivity** from 93.33% to 98.33%, better detecting positive COVID cases
2. **AdaBoost** achieved perfect specificity (100%) on the original dataset with only 5 features
3. **XGBoost** performed best overall on both datasets with exceptional AUC scores
4. **Balanced data** (SMOTE) resulted in more balanced sensitivity/specificity trade-offs
5. **Feature economy**: AdaBoost showed that excellent performance is possible with just 5 carefully selected features

---

## Reproducibility Information
- **Random State**: 42 (used across all random processes)
- **Python Version**: 3.8.10
- **Key Libraries**:
  - scikit-learn: 1.3.2
  - xgboost: 2.0.3
  - imblearn: 0.12.4
  - pandas: 2.0.3
  - numpy: 1.24.3

---

## Files Generated
1. **Models Directory**: 
   - `best_XGBoost_original.pkl`
   - `best_AdaBoost_original.pkl`
   - `best_RandomForest_original.pkl`
   - `best_DecisionTree_original.pkl`
   - `best_XGBoost_smote.pkl`
   - `best_AdaBoost_smote.pkl`
   - `best_RandomForest_smote.pkl`
   - `best_DecisionTree_smote.pkl`

2. **Results Directory**:
   - `India375.csv` - Model summaries for original dataset
   - `IndiaSmote600.csv` - Model summaries for SMOTE dataset
   - `training_metadata_375_original.csv` - Training metadata with hyperparameters
   - `training_metadata_600_smote.csv` - Training metadata for SMOTE models

---

## Notes
- All metric values have been populated from the actual notebook execution.
- The SMOTE dataset showed improved sensitivity (98.33% vs 93.33%) while maintaining high overall accuracy.
- XGBoost achieved the best AUC scores on both datasets (99.56% original, 99.94% SMOTE).
- AdaBoost demonstrated excellent feature efficiency, achieving 97.33% accuracy with only 5 features on the original dataset.
- Complete confusion matrices and detailed metrics are available in the generated CSV files in the Results directory.
- Both training metadata CSV files contain the complete hyperparameter configurations and selected features for reproducibility.

---

**Document Version**: 2.0  
**Last Updated**: February 14, 2026  
**Status**: Complete - All results populated from actual notebook execution
