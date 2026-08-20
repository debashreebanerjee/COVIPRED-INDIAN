# Brazil Model Training Results

## Overview
This document records the results produced by `Codes/Brazil.ipynb`. The notebook now performs internal validation on the Brazil cohort before training the final model, then evaluates that model on Indian and Italian cohorts after aligning the feature schema.

---

## Training Dataset

- **Source**: `Data/3-fourteen-feature.csv`
- **Total Samples**: 11,916
- **Class Distribution**: 8,771 negatives and 3,145 positives (prevalence `0.264`)
- **Training Strategy**: Stratified 64/16/20 three-way split for hyperparameter selection and internal validation, followed by a final fit on 100% of the cohort
- **Final Feature Count**: 11 shared laboratory features

### Features Used

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

## Internal Validation

### Split Composition

| Split | Samples | Positives | Prevalence | Role |
|---|---:|---:|---:|---|
| Inner train | 7,625 | 2,013 | 0.264 | Fits each candidate configuration |
| Validation | 1,907 | 503 | 0.264 | Selects hyperparameters (single holdout) |
| Test | 2,384 | 629 | 0.264 | Scored once, with the selected configuration |

All three splits are stratified on `Result` with `random_state=42`. The winning configuration is refit on the combined 80% (inner train + validation) before being scored on the 20% test set.

### Hyperparameter Search

- **Candidates evaluated**: 54
- **Grid**: `n_estimators [50, 100, 200]` x `max_depth [3, 5, 7]` x `learning_rate [0.01, 0.1, 0.3]` x `scale_pos_weight [1, 2.788]`
- **Selection objective**: accuracy, matching `Codes/India.ipynb`
- **Validation accuracy of the winner**: `0.7981`

**Selected hyperparameters**

| Parameter | Selected | XGBoost default (previous model) |
|---|---:|---:|
| `n_estimators` | 100 | 100 |
| `max_depth` | 3 | 6 |
| `learning_rate` | 0.1 | 0.3 |
| `scale_pos_weight` | 1 | 1 |

The full ranking of all 54 candidates, scored on accuracy, balanced accuracy, ROC AUC, and F2, is saved to `Results/Brazil_tuning_results.csv`.

### Internal Metrics (held-out 20%)

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brazil internal (held-out 20%) | 2,384 | 0.7957 | 0.4452 | 0.9214 | 0.6833 | 0.4772 | 0.8075 | 0.1456 |

```
[[1617  138]
 [ 349  280]]
```

- True Negatives: 1,617
- False Positives: 138
- False Negatives: 349
- True Positives: 280
- Test Set Composition: 1,755 negatives, 629 positives

---

## Evaluation Summary

All external evaluations use the final model, which is the selected configuration refit on 100% of the Brazil cohort.

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brazil on India 375 | 375 | 0.7947 | 0.0667 | 0.9767 | 0.5217 | 0.0801 | 0.5923 | 0.1601 |
| Brazil on India SMOTE 600 | 600 | 0.5433 | 0.1033 | 0.9833 | 0.5433 | 0.1254 | 0.6576 | 0.3184 |
| Brazil on Italy | 1,388 | 0.6441 | 0.4157 | 0.9246 | 0.6701 | 0.4642 | 0.7932 | 0.2312 |

\* `F2 Score` is computed with `fbeta_score(beta=2)`.

\** ROC AUC is computed from predicted probabilities using `predict_proba`.

---

## Confusion Matrices

### Brazil on India 375

```
[[293   7]
 [ 70   5]]
```

- True Negatives: 293
- False Positives: 7
- False Negatives: 70
- True Positives: 5
- Test Set Composition: 300 negatives, 75 positives

### Brazil on India SMOTE 600

```
[[295   5]
 [269  31]]
```

- True Negatives: 295
- False Positives: 5
- False Negatives: 269
- True Positives: 31
- Test Set Composition: 300 negatives, 300 positives

### Brazil on Italy

```
[[576  47]
 [447 318]]
```

- True Negatives: 576
- False Positives: 47
- False Negatives: 447
- True Positives: 318
- Test Set Composition: 623 negatives, 765 positives

---

## Key Observations

- **The internal result establishes that the Brazil model is genuinely discriminative.** With a held-out ROC AUC of `0.8075` and balanced accuracy of `0.6833`, the model learns real signal from the 11-feature panel. This is the reference point the previous full-dataset fit could not provide, and it is what makes the external results interpretable as domain shift rather than as an undertrained model.
- **Transfer to Italy is nearly as strong as the in-domain result in ranking terms.** External ROC AUC on Italy is `0.7932` against an internal `0.8075`, a drop of only `0.015`. The Brazil model's ordering of patients survives the move to the Italian cohort almost intact.
- **Transfer to India collapses.** ROC AUC falls to `0.5923` (India 375) and `0.6576` (India SMOTE 600), and balanced accuracy sits at `0.5217` and `0.5433` — close to chance. Since the same model scores `0.81` internally and `0.79` on Italy, the failure is specific to the Indian cohort rather than a property of the model.
- **Accuracy-based selection produced a specificity-heavy operating point.** The search chose `scale_pos_weight=1` over `2.788`, because on a 2.8:1 imbalanced cohort accuracy rewards conservative positive calls. Internal sensitivity is therefore only `0.4452` at the `0.5` threshold, and on India 375 it drops to `0.0667` with specificity at `0.9767`. Sensitivity-oriented thresholds are reported separately by `computemetrics()` at `0.15`.
- **Tuning improved every external ROC AUC relative to the previous default-hyperparameter model**: India 375 `0.5471 -> 0.5923`, India SMOTE 600 `0.5900 -> 0.6576`, Italy `0.7458 -> 0.7932`. Threshold-dependent metrics moved in both directions because the selected model is more conservative than the default.

---

## Files Generated

- `Models/brazil_model.pkl` - final model, fitted on 100% of the Brazil cohort
- `Models/brazil_model_holdout.pkl` - same configuration fitted on the 80% dev set, retained so the internal metrics stay reproducible
- `Results/Brazil_internal_metrics.csv`
- `Results/Brazil_internal_test_preds.csv`
- `Results/Brazil_tuning_results.csv`
- `Results/Brazil_metrics.csv`
- `Results/1aIndiaBrazil375.csv`
- `Results/1bIndiaBrazil600.csv`
- `Results/1cItalyBrazil.csv`
- `Results/SimilarityBrazil.csv`
- `Results/merged_brit.csv`

---

## Notes

- All values above were taken from the saved notebook outputs and exported prediction files.
- Confusion matrices in this document were reconstructed from the saved `y_true` and `y_proba` files using the notebook's default `0.5` decision threshold.
- The internal metrics describe the model trained on the 80% dev set. The model shipped for external validation is the same configuration refit on 100% of the cohort, which is standard practice but means the internal metrics are a slightly conservative estimate of the final model's in-domain performance.
- `Data/3-fourteen-feature.csv` arrives already z-scored across the full Brazil cohort, so the train/validation/test splits inherit scaling statistics computed over all rows. This is a mild optimistic bias that cannot be removed without the raw, unstandardized values.
- Hyperparameters are selected on a single validation holdout rather than by cross-validation. With 1,907 validation samples this is reasonably stable for Brazil, but the selection is still noisier than a k-fold estimate would be.
