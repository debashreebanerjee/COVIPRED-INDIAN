# Italy Model Training Results

## Overview
This document records the results produced by `Codes/Italy.ipynb`. The notebook trains an XGBoost model on the full Italy cohort and evaluates it on Indian and Brazilian cohorts after feature alignment.

---

## Training Dataset

- **Source**: `Data/Dataset-2a.csv`
- **Total Samples**: 1,388
- **Class Distribution**: 623 negatives and 765 positives
- **Training Strategy**: Full-dataset fitting with no internal train/test split
- **Final Feature Count**: 11 shared laboratory features

### Features Used

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

## Evaluation Summary

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Italy on India 375 | 375 | 0.4987 | 0.4400 | 0.5133 | 0.4767 | 0.3445 | 0.4767 | 0.3637 |
| Italy on India 600 | 600 | 0.5433 | 0.5167 | 0.5700 | 0.5433 | 0.5222 | 0.5433 | 0.3345 |
| Italy on Brazil | 11,916 | 0.6490 | 0.7272 | 0.6209 | 0.6740 | 0.6286 | 0.6740 | 0.2747 |

\* `F2 Score` is computed with `fbeta_score(beta=2)`.

\** The notebook summary uses hard predictions for ROC AUC.

---

## Confusion Matrices

### Italy on India 375

```
[[154 146]
 [ 42  33]]
```

- True Negatives: 154
- False Positives: 146
- False Negatives: 42
- True Positives: 33
- Test Set Composition: 300 negatives, 75 positives

### Italy on India 600

```
[[171 129]
 [145 155]]
```

- True Negatives: 171
- False Positives: 129
- False Negatives: 145
- True Positives: 155
- Test Set Composition: 300 negatives, 300 positives

### Italy on Brazil

```
[[5446 3325]
 [ 858 2287]]
```

- True Negatives: 5,446
- False Positives: 3,325
- False Negatives: 858
- True Positives: 2,287
- Test Set Composition: 8,771 negatives, 3,145 positives

---

## Key Observations

- The Italy-trained model transfers best to the Brazil cohort, where it achieves the strongest accuracy (`0.6490`), sensitivity (`0.7272`), and balanced accuracy (`0.6740`).
- The original Indian 375 cohort is the hardest transfer setting for this model, with accuracy below `0.50`.
- Moving from India 375 to India 600 improves both sensitivity and specificity, suggesting the balanced evaluation set is more favorable to the Italy-trained classifier.
- On Brazil, the model favors positive detection more strongly than on the Indian cohorts, at the cost of a larger false-positive count.

---

## Files Generated

- `Models/italy_model.pkl`
- `Results/Italy_metrics.csv`
- `Results/2aIndiaItaly375.csv`
- `Results/2bIndiaItaly600.csv`
- `Results/2cBrazilItaly.csv`
- `Results/SimilarityItaly.csv`

---

## Notes

- The current `Results/Italy_metrics.csv` is aligned with the corrected notebook logic, including the `Italy on Brazil` row.
- After the rerun, `Results/2bIndiaItaly600.csv` and `Results/2cBrazilItaly.csv` changed only in very small floating-point tails of some `y_proba` values; these changes did not alter thresholded predictions or standardized net benefit at the notebook's `0.15` and `0.5` thresholds.
- Confusion matrices in this document were reconstructed from the saved `y_true` and `y_proba` files using the notebook's default `0.5` decision threshold.
