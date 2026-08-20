# Italy Model Implementation Documentation

## Overview
This document summarizes the implementation of the cross-dataset evaluation pipeline in `Codes/Italy.ipynb`. The notebook internally validates and tunes an Italy-based XGBoost classifier, trains the final model on the full Italy cohort, and evaluates it on Indian and Brazilian cohorts after aligning all datasets to a common feature schema.

---

## 1. Data Sources

- `Data/Dataset-2a.csv`: Italy training dataset.
- `Data/Edited_bitsM.csv`: India BITSM dataset.
- `Data/ESIdf.csv`: India ESI positive dataset.
- `Data/smoteesimedc.csv`: SMOTE-balanced India dataset.
- `Data/3-fourteen-feature.csv`: Brazil dataset used for external testing.

### Dataset Roles
- **Italy model training**: The 1,388 Italy samples are split 64/16/20 for hyperparameter selection and internal validation; the final model is then refit on all 1,388 samples.
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

- **Model**: `XGBClassifier(eval_metric='logloss', random_state=42)`

### Split Design

A stratified three-way split separates hyperparameter selection from the set the internal metrics are reported on, without using k-fold cross-validation:

| Split | Fraction | Samples | Role |
|---|---:|---:|---|
| Inner train | 64% | 888 | Fits each candidate configuration |
| Validation | 16% | 222 | Selects hyperparameters (single holdout) |
| Test | 20% | 278 | Scored once, with the selected configuration |

It is produced by two nested `train_test_split` calls, both stratified on `Result` with `random_state=42`: an outer 80:20 split, then an 80:20 split of the resulting dev set.

### Hyperparameter Search

- The materialized train/validation pair is presented to `GridSearchCV` through `PredefinedSplit` (`-1` = always train, `0` = validation), so selection uses one holdout rather than k-fold cross-validation.
- **Grid (27 candidates)**: `n_estimators [50, 100, 200]` x `max_depth [3, 5, 7]` x `learning_rate [0.01, 0.1, 0.3]`. `scale_pos_weight` is not searched because the Italy cohort is close to balanced at 623:765.
- **Scoring**: accuracy, balanced accuracy, ROC AUC, and F2 are recorded for every candidate; `refit='accuracy'` selects the winner.
- `refit=True` then refits the winning configuration on the full 80% dev set. That estimator is `grid.best_estimator_`, used for the internal metrics.

### Internal Validation

The 80%-trained model is scored once on the untouched 20% test set, using the same metric block applied to every external cohort: accuracy, classification report, confusion matrix, sensitivity, specificity, balanced accuracy, F2, ROC AUC, Brier score, plus the `computemetrics()` threshold analysis at `0.15` and `0.5`.

### Final Model

The selected hyperparameters are refit on 100% of the Italy cohort. This is the model used for every external evaluation in the notebook.

- **Persistence**: The final model is written to `Models/italy_model.pkl` and the 80%-trained model to `Models/italy_model_holdout.pkl`.
- **Write behavior**: Both files are written unconditionally on every run. The previous try-load / except-dump pattern reloaded the stale artifact whenever the file already existed, which silently discarded the newly fitted model.

---

## 4. Evaluation Workflow

The final Italy model is evaluated in three external settings:

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

- `Models/italy_model.pkl` - Final Italy XGBoost model, fitted on 100% of the cohort
- `Models/italy_model_holdout.pkl` - Same configuration fitted on the 80% dev set, retained so the internal metrics stay reproducible
- `Results/Italy_internal_metrics.csv` - Internal validation metrics on the held-out 20%
- `Results/Italy_internal_test_preds.csv` - Saved `y_true` and `y_proba` for the internal 20% test set
- `Results/Italy_tuning_results.csv` - All 27 candidate configurations ranked by validation accuracy, with balanced accuracy, ROC AUC, and F2 recorded for each
- `Results/Italy_metrics.csv` - Summary metrics across the three external evaluation cohorts
- `Results/2aIndiaItaly375.csv` - Saved `y_true` and `y_proba` for the India 375 evaluation
- `Results/2bIndiaItaly600.csv` - Saved `y_true` and `y_proba` for the India 600 evaluation
- `Results/2cBrazilItaly.csv` - Saved `y_true` and `y_proba` for the Brazil evaluation
- `Results/SimilarityItaly.csv` - Average KS-based similarity scores

---

## 6. Reproducibility Notes

- Every split, the grid search, and both model fits use `random_state=42`, so a rerun reproduces the saved metrics exactly.
- The notebook always retrains and overwrites `Models/italy_model.pkl`; it no longer short-circuits to a persisted artifact.
- The internal metrics describe the 80%-trained model, while external validation uses the 100%-trained model. This is standard practice, but it means the internal numbers are a slightly conservative estimate of the final model's in-domain performance.
- `Data/Dataset-2a.csv` arrives already z-scored across the full Italy cohort, so the train/validation/test splits inherit scaling statistics computed over all rows. This is a mild optimistic bias that cannot be removed without the raw, unstandardized values.
- The validation set holds only 222 samples, so single-holdout hyperparameter selection is noisier here than for Brazil; the top five configurations sit within `0.01` accuracy of each other. The grid was kept coarse at 27 candidates to limit selection on noise.
- Four of the eleven features (`Lymphocytes(%)`, `Monocytes(%)`, `Eosinophils(%)`, `Basophils(%)`) contain 14 missing values each, passed through to XGBoost, which handles them natively.
- The India-derived evaluation sets are standardized inside the notebook.
- The Brazil evaluation set is aligned by renaming and column restriction, but not standardized in the notebook.
- The Brazil evaluation block now recomputes `roc_auc` and `brier_score` before appending the `Italy on Brazil` row to `Results/Italy_metrics.csv`, so the exported summary matches the notebook output.

---

## 7. File Structure

- `Codes/Italy.ipynb` - Main notebook
- `Models/` - Saved model artifact
- `Results/` - Metric summaries and prediction exports
- `Data/` - Source datasets
