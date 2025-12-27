# India Model Implementation Documentation

## Overview
This document details the implementation of the machine learning pipeline for COVID-19 prediction using the `India.ipynb` notebook. The workflow covers data preprocessing, class balancing, feature selection, model training, evaluation, model persistence, and results export. The approach is designed for reproducibility, transparency, and extensibility.

---

## 1. Data Loading and Preprocessing

- **Data Sources:**
  - `Edited_bitsM.csv`: Contains 300 negatives and 36 positives from BITSM.
  - `ESIdf.csv`: Contains 75 positives from ESI.
- **External Validation:**
  - Positives from BITSM are set aside for external validation.
- **Training Data Construction:**
  - Only negatives from BITSM and all positives from ESI are used for training (total 375 samples).
- **Label Encoding:**
  - The `Result` and `Gender` columns are label-encoded for model compatibility.
- **Feature Dropping:**
  - The first column (likely an index) and the `Others` column are dropped.
  - For modeling, `Gender` and `Age` are also dropped.

---

## 2. Class Balancing with SMOTE

- **SMOTE (Synthetic Minority Over-sampling Technique):**
  - Applied to the training data to balance the classes, resulting in 600 samples (equal positives and negatives).
  - Post-processing ensures `Age` is integer and `Haemoglobin(gms%)` is rounded to 1 decimal.

---

## 3. Feature Selection

- **SelectKBest:**
  - Integrated into the model pipeline using `f_classif` as the scoring function.
  - The number of features (`k`) is treated as a hyperparameter and tuned via grid search (`[5, 8, 'all']`).

---

## 4. Model Training and Hyperparameter Tuning

- **Models Used:**
  - XGBoost
  - AdaBoost
  - Random Forest
  - Decision Tree
- **Pipeline:**
  - Each model is wrapped in a `Pipeline` with `SelectKBest` for feature selection.
- **GridSearchCV:**
  - Used for hyperparameter tuning and cross-validation (5-fold).
  - Hyperparameters for each model (e.g., `n_estimators`, `max_depth`, `learning_rate`, `criterion`) are included in the grid.
  - `random_state=42` is set for all models and data splits for reproducibility.
- **Training Datasets:**
  - Models are trained and evaluated on both the original (375) and SMOTE-balanced (600) datasets.

---

## 5. Model Persistence

- **Saving Models:**
  - Best models for each algorithm and dataset are saved as `.pkl` files in the `Models` directory (e.g., `best_XGBoost_original.pkl`, `best_XGBoost_smote.pkl`).
- **Loading Models:**
  - If a model file exists, it is loaded instead of retraining, ensuring efficiency and reproducibility.

---

## 6. Evaluation and Reporting


- **Metrics:**
  - Training cross-validation accuracy and test accuracy are reported for each model.
- **Feature Reporting:**
  - The features selected by `SelectKBest` in the best pipeline are printed and included in the results.
- **Results Export:**
  - Results for both datasets are saved as CSV files in the `Results` directory (`India375.csv`, `IndiaSmote600.csv`).

### Confusion Matrices
For both the original and SMOTE datasets, confusion matrices are generated for the XGBoost models. These matrices provide a detailed breakdown of true positives, true negatives, false positives, and false negatives, allowing for a comprehensive assessment of model performance.

### Metrics Tables
For all models (XGBoost, AdaBoost, Random Forest, Decision Tree), the following metrics are calculated and presented in tables for both datasets:
- **Sensitivity (Recall):** $\frac{TP}{TP + FN}$
- **Specificity:** $\frac{TN}{TN + FP}$
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **AUC Score:** Area under the ROC curve, calculated using predicted probabilities.
- **Number of Features Selected:** The number of features chosen by SelectKBest in the best pipeline.

These metrics are displayed in a summary table for each dataset, making it easy to compare model performance across different algorithms and feature sets.

---

## 7. Reproducibility and Best Practices

- **Random State:**
  - All random processes (splitting, SMOTE, model training) use `random_state=42`.
- **Directory Management:**
  - The code ensures the `Models` and `Results` directories exist before saving files.
- **Efficiency:**
  - Model training is skipped if a saved model is found, reducing unnecessary computation.
- **Extensibility:**
  - The pipeline structure allows easy addition of new models, features, or preprocessing steps.

---

## 8. Usage Instructions

1. **Run all cells in order** to preprocess data, train models, and save results/models.
2. **Check the `Models` folder** for saved model files and the `Results` folder for CSV summaries.
3. **To retrain a model**, delete the corresponding `.pkl` file from the `Models` directory and rerun the notebook.

---



## 9. Dependencies

```
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)]
sklearn 1.3.2
xgboost 2.0.3
imblearn 0.12.4
pandas 2.0.3
numpy 1.24.3
matplotlib 3.7.5
seaborn 0.13.2
joblib 1.3.2
lightgbm 4.5.0
```

---

## 10. Notes
- The notebook is designed for clarity and reproducibility, suitable for research and publication.
- All key steps (data handling, modeling, evaluation, saving/loading) are automated and robust to reruns.
- For further customization, adjust the hyperparameter grids, feature selection strategy, or add new models as needed.

---

## 11. File Structure
- `Codes/India.ipynb` — Main notebook
- `Models/` — Saved model pipelines
- `Results/` — Output CSVs with model results
- `Data/` — Input datasets

---

## 12. Contact
For questions or contributions, please contact the project maintainer or refer to the repository documentation.
