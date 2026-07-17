# Brazil Model Training Results

## Overview
This document records the results produced by `Codes/Brazil.ipynb`. The notebook trains an XGBoost model on the full Brazil cohort and evaluates it on Indian and Italian cohorts after aligning the feature schema.

---

## Training Dataset

- **Source**: `Data/3-fourteen-feature.csv`
- **Total Samples**: 11,916
- **Class Distribution**: 8,771 negatives and 3,145 positives
- **Training Strategy**: Full-dataset fitting with no internal train/test split
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

## Evaluation Summary

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brazil on India 375 | 375 | 0.7067 | 0.1467 | 0.8467 | 0.4967 | 0.1815 | 0.4967 | 0.2050 |
| Brazil on India SMOTE 600 | 600 | 0.5183 | 0.1767 | 0.8600 | 0.5183 | 0.3897 | 0.5183 | 0.3498 |
| Brazil on Italy | 1,388 | 0.6189 | 0.4209 | 0.8620 | 0.6414 | 0.6717 | 0.6414 | 0.2605 |

\* The notebook labels this metric as `F2 Score`, but the code computes `fbeta_score(beta=0.5)`.

\** The notebook summary uses hard predictions for ROC AUC.

---

## Confusion Matrices

### Brazil on India 375

```
[[254  46]
 [ 64  11]]
```

- True Negatives: 254
- False Positives: 46
- False Negatives: 64
- True Positives: 11
- Test Set Composition: 300 negatives, 75 positives

### Brazil on India SMOTE 600

```
[[258  42]
 [247  53]]
```

- True Negatives: 258
- False Positives: 42
- False Negatives: 247
- True Positives: 53
- Test Set Composition: 300 negatives, 300 positives

### Brazil on Italy

```
[[537  86]
 [443 322]]
```

- True Negatives: 537
- False Positives: 86
- False Negatives: 443
- True Positives: 322
- Test Set Composition: 623 negatives, 765 positives

---

## Key Observations

- The Brazil-trained model transfers best to the Italy cohort, where it reaches the highest balanced accuracy (`0.6414`) and the highest sensitivity (`0.4209`).
- Performance on both Indian cohorts is limited by very low sensitivity, even though specificity remains near `0.85`.
- The SMOTE-balanced Indian set slightly improves balanced accuracy versus the 375-sample Indian cohort, but positive-class recall remains weak.
- The Italy evaluation produces the strongest overall trade-off among the three external tests.

---

## Files Generated

- `Models/brazil_model.pkl`
- `Results/Brazil_metrics.csv`
- `Results/1aIndiaBrazil375.csv`
- `Data/1bIndiaBrazil600.csv`
- `Results/1cItalyBrazil.csv`
- `Results/SimilarityBrazil.csv`
- `Results/merged_brit.csv`

---

## Notes

- All values above were taken from the saved notebook outputs and exported prediction files.
- Confusion matrices in this document were reconstructed from the saved `y_true` and `y_proba` files using the notebook's default `0.5` decision threshold.
