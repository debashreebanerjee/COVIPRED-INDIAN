# Italy Model Implementation Documentation

## Overview
This document summarizes the implementation of the cross-dataset evaluation pipeline in `Codes/Italy.ipynb`. The notebook trains an Italy-based XGBoost classifier and evaluates it on Indian and Brazilian cohorts after aligning all datasets to a common feature schema.

---

## 1. Data Sources

- `Data/Dataset-2a.csv`: Italy training dataset.
- `Data/Edited_bitsM.csv`: India BITSM dataset.
- `Data/ESIdf.csv`: India ESI positive dataset.
- `Data/smoteesimedc.csv`: SMOTE-balanced India dataset.
- `Data/3-fourteen-feature.csv`: Brazil dataset used for external testing.

### Dataset Roles
- **Italy model training**: All 1,388 Italy samples are used to fit the model.
- **India external test set (375)**: Constructed from 300 BITSM negatives and 75 ESI positives.
- **India SMOTE test set (600)**: Loaded from the saved SMOTE-balanced India file.
- **Brazil external test set (11,916)**: Used as the second external validation cohort.

---

## 2. Feature Harmonization and Preprocessing

### India 375 Reference Set
- BITSM positives are removed so only 300 negatives remain from `Edited_bitsM.csv`.
- The 75 ESI positives are concatenated to form a 375-sample Indian reference dataset.
- The index column and `Others` column are dropped.
- `Result` and `Gender` are label-encoded.
- A fixed 11-feature order is applied.
- Z-score standardization is applied to the Indian reference features.

### Italy Training Set
- Demographic, chemistry, symptom, and absolute-count columns are removed.
- Remaining columns are renamed to the shared naming scheme.
- `Neutrophils(%)` is dropped so the Italy schema matches the notebook's shared feature set.

### Brazil Test Set
- Columns are renamed to match the India/Italy naming scheme.
- The Brazil frame is restricted to the same ordered feature list used for the Italy model.

### Shared Feature Set
The final Italy model uses these 11 laboratory features:

1. `Total WBC Count(/Cumm)`
2. `Haemoglobin(gms%)`
3. `HCT(%)`
4. `MCV(f L)`
5. `MCH(pg)`
6. `MCHC(gms%)`
7. `Platelet Count(Lakh / Cumm)`
8. `Lymphocytes(%)`
9. `Monocytes(%)`
10. `Eosinophils(%)`
11. `Basophils(%)`

---

## 3. Model Training

- **Model**: `XGBClassifier()`
- **Training strategy**: The classifier is fit on the full Italy dataset without an internal train/test split.
- **Persistence**: The fitted model is saved to `Models/italy_model.pkl`.
- **Reload behavior**: If `Models/italy_model.pkl` exists, it is loaded instead of being recreated.

---

## 4. Evaluation Workflow

The trained Italy model is evaluated in three settings:

1. **Italy on India 375**
2. **Italy on India 600**
3. **Italy on Brazil**

For each setting, the notebook reports:

- Accuracy
- Classification report
- Confusion matrix
- Sensitivity
- Specificity
- Balanced accuracy
- `F2 Score` column in the notebook output
- ROC AUC
- Brier score

### Metric Implementation Notes
- The notebook's `F2 Score` column is computed with `fbeta_score(beta=2)`.
- ROC AUC is computed from predicted probabilities using `predict_proba`.
- Brier score is computed from predicted probabilities.
- The helper `computemetrics()` function also prints threshold-specific values at `0.15` and `0.5`.

---

## 5. Output Files

- `Models/italy_model.pkl` - Saved Italy XGBoost model
- `Results/Italy_metrics.csv` - Summary metrics across the three evaluation cohorts
- `Results/2aIndiaItaly375.csv` - Saved `y_true` and `y_proba` for the India 375 evaluation
- `Results/2bIndiaItaly600.csv` - Saved `y_true` and `y_proba` for the India 600 evaluation
- `Results/2cBrazilItaly.csv` - Saved `y_true` and `y_proba` for the Brazil evaluation
- `Results/SimilarityItaly.csv` - Average KS-based similarity scores

---

## 6. Reproducibility Notes

- The notebook evaluates a persisted model if `Models/italy_model.pkl` is already present.
- The India-derived evaluation sets are standardized inside the notebook.
- The Brazil evaluation set is aligned by renaming and column restriction, but not standardized in the notebook.
- The Brazil evaluation block now recomputes `roc_auc` and `brier_score` before appending the `Italy on Brazil` row to `Results/Italy_metrics.csv`, so the exported summary matches the notebook output.

---

## 7. File Structure

- `Codes/Italy.ipynb` - Main notebook
- `Models/` - Saved model artifact
- `Results/` - Metric summaries and prediction exports
- `Data/` - Source datasets
