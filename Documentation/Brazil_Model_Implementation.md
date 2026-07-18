# Brazil Model Implementation Documentation

## Overview
This document summarizes the implementation of the cross-dataset evaluation pipeline in `Codes/Brazil.ipynb`. The notebook trains a Brazil-based XGBoost classifier and evaluates it on Indian and Italian cohorts after feature harmonization.

---

## 1. Data Sources

- `Data/3-fourteen-feature.csv`: Brazil training dataset.
- `Data/Edited_bitsM.csv`: India BITSM dataset.
- `Data/ESIdf.csv`: India ESI positive dataset.
- `Data/smoteesimedc.csv`: SMOTE-balanced India dataset.
- `Data/Dataset-2a.csv`: Italy dataset used for external testing.

### Dataset Roles
- **Brazil model training**: All 11,916 Brazil samples are used to fit the model.
- **India external test set (375)**: Constructed from 300 BITSM negatives and 75 ESI positives.
- **India SMOTE test set (600)**: Constructed from the saved SMOTE-balanced India file.
- **Italy external test set (1,388)**: Used as a second out-of-country evaluation cohort.

---

## 2. Feature Harmonization and Preprocessing

### India 375 Reference Set
- BITSM positives are removed so only 300 negatives remain from `Edited_bitsM.csv`.
- The 75 ESI positives are concatenated to create a 375-sample Indian reference set.
- The index column and `Others` column are dropped.
- `Result` and `Gender` are label-encoded.
- A fixed feature order is applied before testing.
- Z-score standardization is applied to the Indian reference features.

### Brazil Training Set
- `MPV` and `Probability` are dropped.
- Columns are renamed to match the India naming scheme.
- `RDWCV(%)` and `Total RBC Count(millions/Cu)` are dropped so the feature schema matches the other cohorts used in the notebook.

### Italy Test Set
- Chemistry, symptom, demographic, and absolute-count columns are removed.
- Remaining columns are renamed to the shared naming scheme.
- `Neutrophils(%)` is dropped to align with the Brazil model feature set.

### Shared Feature Set
The final model operates on these 11 laboratory features:

1. `HCT(%)`
2. `Haemoglobin(gms%)`
3. `Platelet Count(Lakh / Cumm)`
4. `Lymphocytes(%)`
5. `MCHC(gms%)`
6. `Total WBC Count(/Cumm)`
7. `Basophils(%)`
8. `MCH(pg)`
9. `Eosinophils(%)`
10. `MCV(f L)`
11. `Monocytes(%)`

---

## 3. Model Training

- **Model**: `XGBClassifier()`
- **Training strategy**: The notebook fits the classifier on the full Brazil dataset without an internal train/test split.
- **Persistence**: The fitted model is saved to `Models/brazil_model.pkl`.
- **Reload behavior**: If `Models/brazil_model.pkl` already exists, it is loaded instead of writing a new model artifact.

---

## 4. Evaluation Workflow

The trained Brazil model is evaluated in three settings:

1. **Brazil on India 375**
2. **Brazil on India SMOTE 600**
3. **Brazil on Italy**

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

- `Models/brazil_model.pkl` - Saved Brazil XGBoost model
- `Results/Brazil_metrics.csv` - Summary metrics across the three evaluation cohorts
- `Results/1aIndiaBrazil375.csv` - Saved `y_true` and `y_proba` for the India 375 evaluation
- `Data/1bIndiaBrazil600.csv` - Saved `y_true` and `y_proba` for the India 600 SMOTE evaluation
- `Results/1cItalyBrazil.csv` - Saved `y_true` and `y_proba` for the Italy evaluation
- `Results/SimilarityBrazil.csv` - Average KS-based similarity scores
- `Results/merged_brit.csv` - Concatenated Brazil and Italy metric summaries

---

## 6. Reproducibility Notes

- The notebook evaluates a persisted model if `Models/brazil_model.pkl` is already present.
- The India-derived evaluation sets are standardized inside the notebook.
- The Italy evaluation set is aligned by column selection and renaming, but not standardized in the notebook.
- The India 600 prediction export is written under `Data/` rather than `Results/`, which is worth keeping in mind when locating artifacts.

---

## 7. File Structure

- `Codes/Brazil.ipynb` - Main notebook
- `Models/` - Saved model artifact
- `Results/` - Metric summaries and prediction exports
- `Data/` - Source datasets and one saved prediction file for the India 600 evaluation
