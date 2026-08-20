# Brazil Model Implementation Documentation

## Overview
This document summarizes the implementation of the cross-dataset evaluation pipeline in `Codes/Brazil.ipynb`. The notebook internally validates and tunes a Brazil-based XGBoost classifier, trains the final model on the full Brazil cohort, and evaluates it on Indian and Italian cohorts after feature harmonization.

---

## 1. Data Sources

- `Data/3-fourteen-feature.csv`: Brazil training dataset.
- `Data/Edited_bitsM.csv`: India BITSM dataset.
- `Data/ESIdf.csv`: India ESI positive dataset.
- `Data/smoteesimedc.csv`: SMOTE-balanced India dataset.
- `Data/Dataset-2a.csv`: Italy dataset used for external testing.

### Dataset Roles
- **Brazil model training**: The 11,916 Brazil samples are split 64/16/20 for hyperparameter selection and internal validation; the final model is then refit on all 11,916 samples.
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

- **Model**: `XGBClassifier(eval_metric='logloss', random_state=42)`

### Split Design

A stratified three-way split separates hyperparameter selection from the set the internal metrics are reported on, without using k-fold cross-validation:

| Split | Fraction | Samples | Role |
|---|---:|---:|---|
| Inner train | 64% | 7,625 | Fits each candidate configuration |
| Validation | 16% | 1,907 | Selects hyperparameters (single holdout) |
| Test | 20% | 2,384 | Scored once, with the selected configuration |

It is produced by two nested `train_test_split` calls, both stratified on `Result` with `random_state=42`: an outer 80:20 split, then an 80:20 split of the resulting dev set.

### Hyperparameter Search

- The materialized train/validation pair is presented to `GridSearchCV` through `PredefinedSplit` (`-1` = always train, `0` = validation), so selection uses one holdout rather than k-fold cross-validation.
- **Grid (54 candidates)**: `n_estimators [50, 100, 200]` x `max_depth [3, 5, 7]` x `learning_rate [0.01, 0.1, 0.3]` x `scale_pos_weight [1, 2.788]`. `scale_pos_weight` is searched because the Brazil cohort is imbalanced at 8,771:3,145.
- **Scoring**: accuracy, balanced accuracy, ROC AUC, and F2 are recorded for every candidate; `refit='accuracy'` selects the winner.
- `refit=True` then refits the winning configuration on the full 80% dev set. That estimator is `grid.best_estimator_`, used for the internal metrics.

### Internal Validation

The 80%-trained model is scored once on the untouched 20% test set, using the same metric block applied to every external cohort: accuracy, classification report, confusion matrix, sensitivity, specificity, balanced accuracy, F2, ROC AUC, Brier score, plus the `computemetrics()` threshold analysis at `0.15` and `0.5`.

### Final Model

The selected hyperparameters are refit on 100% of the Brazil cohort. This is the model used for every external evaluation in the notebook.

- **Persistence**: The final model is written to `Models/brazil_model.pkl` and the 80%-trained model to `Models/brazil_model_holdout.pkl`.
- **Write behavior**: Both files are written unconditionally on every run. The previous try-load / except-dump pattern reloaded the stale artifact whenever the file already existed, which silently discarded the newly fitted model.

---

## 4. Evaluation Workflow

The final Brazil model is evaluated in three external settings:

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

- `Models/brazil_model.pkl` - Final Brazil XGBoost model, fitted on 100% of the cohort
- `Models/brazil_model_holdout.pkl` - Same configuration fitted on the 80% dev set, retained so the internal metrics stay reproducible
- `Results/Brazil_internal_metrics.csv` - Internal validation metrics on the held-out 20%
- `Results/Brazil_internal_test_preds.csv` - Saved `y_true` and `y_proba` for the internal 20% test set
- `Results/Brazil_tuning_results.csv` - All 54 candidate configurations ranked by validation accuracy, with balanced accuracy, ROC AUC, and F2 recorded for each
- `Results/Brazil_metrics.csv` - Summary metrics across the three external evaluation cohorts
- `Results/1aIndiaBrazil375.csv` - Saved `y_true` and `y_proba` for the India 375 evaluation
- `Results/1bIndiaBrazil600.csv` - Saved `y_true` and `y_proba` for the India 600 SMOTE evaluation
- `Results/1cItalyBrazil.csv` - Saved `y_true` and `y_proba` for the Italy evaluation
- `Results/SimilarityBrazil.csv` - Average KS-based similarity scores
- `Results/merged_brit.csv` - Concatenated Brazil and Italy metric summaries

---

## 6. Reproducibility Notes

- Every split, the grid search, and both model fits use `random_state=42`, so a rerun reproduces the saved metrics exactly.
- The notebook always retrains and overwrites `Models/brazil_model.pkl`; it no longer short-circuits to a persisted artifact.
- The internal metrics describe the 80%-trained model, while external validation uses the 100%-trained model. This is standard practice, but it means the internal numbers are a slightly conservative estimate of the final model's in-domain performance.
- `Data/3-fourteen-feature.csv` arrives already z-scored across the full Brazil cohort, so the train/validation/test splits inherit scaling statistics computed over all rows. This is a mild optimistic bias that cannot be removed without the raw, unstandardized values.
- Hyperparameters are selected on a single 1,907-sample validation holdout rather than by cross-validation, which is a deliberate design choice and is noisier than a k-fold estimate would be.
- The India-derived evaluation sets are standardized inside the notebook.
- The Italy evaluation set is aligned by column selection and renaming, but not standardized in the notebook.
- All six prediction exports are written under `Results/`; `Data/` holds source datasets only. The India 600 export previously landed in `Data/`, and a stale duplicate of it sat in `Results/`; both were consolidated to the single current file at `Results/1bIndiaBrazil600.csv`.

---

## 7. File Structure

- `Codes/Brazil.ipynb` - Main notebook
- `Models/` - Saved model artifact
- `Results/` - Metric summaries and prediction exports
- `Data/` - Source datasets
