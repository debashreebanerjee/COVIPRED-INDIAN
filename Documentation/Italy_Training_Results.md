# Italy Model Training Results

## Overview
This document records the results produced by `Codes/Italy.ipynb`. The notebook now performs internal validation on the Italy cohort before training the final model, then evaluates that model on Indian and Brazilian cohorts after feature alignment.

---

## Training Dataset

- **Source**: `Data/Dataset-2a.csv`
- **Total Samples**: 1,388
- **Class Distribution**: 623 negatives and 765 positives (prevalence `0.551`)
- **Training Strategy**: Stratified 64/16/20 three-way split for hyperparameter selection and internal validation, followed by a final fit on 100% of the cohort
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

## Internal Validation

### Split Composition

| Split | Samples | Positives | Prevalence | Role |
|---|---:|---:|---:|---|
| Inner train | 888 | 490 | 0.552 | Fits each candidate configuration |
| Validation | 222 | 122 | 0.550 | Selects hyperparameters (single holdout) |
| Test | 278 | 153 | 0.550 | Scored once, with the selected configuration |

All three splits are stratified on `Result` with `random_state=42`. The winning configuration is refit on the combined 80% (inner train + validation) before being scored on the 20% test set.

### Hyperparameter Search

- **Candidates evaluated**: 27
- **Grid**: `n_estimators [50, 100, 200]` x `max_depth [3, 5, 7]` x `learning_rate [0.01, 0.1, 0.3]`
- **Selection objective**: accuracy, matching `Codes/India.ipynb`
- **Validation accuracy of the winner**: `0.7117`

`scale_pos_weight` is not searched for Italy because the cohort is close to balanced at 623:765.

**Selected hyperparameters**

| Parameter | Selected | XGBoost default (previous model) |
|---|---:|---:|
| `n_estimators` | 100 | 100 |
| `max_depth` | 5 | 6 |
| `learning_rate` | 0.1 | 0.3 |

The full ranking of all 27 candidates, scored on accuracy, balanced accuracy, ROC AUC, and F2, is saved to `Results/Italy_tuning_results.csv`.

### Internal Metrics (held-out 20%)

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Italy internal (held-out 20%) | 278 | 0.7374 | 0.8170 | 0.6400 | 0.7285 | 0.7992 | 0.7910 | 0.1884 |

```
[[ 80  45]
 [ 28 125]]
```

- True Negatives: 80
- False Positives: 45
- False Negatives: 28
- True Positives: 125
- Test Set Composition: 125 negatives, 153 positives

---

## Evaluation Summary

All external evaluations use the final model, which is the selected configuration refit on 100% of the Italy cohort.

| Evaluation Set | Samples | Accuracy | Sensitivity | Specificity | Balanced Accuracy | `F2 Score`* | ROC AUC** | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Italy on India 375 | 375 | 0.5280 | 0.4267 | 0.5533 | 0.4900 | 0.3433 | 0.4946 | 0.2946 |
| Italy on India 600 | 600 | 0.5017 | 0.4900 | 0.5133 | 0.5017 | 0.4923 | 0.5429 | 0.2959 |
| Italy on Brazil | 11,916 | 0.6537 | 0.7307 | 0.6260 | 0.6784 | 0.6328 | 0.7403 | 0.2419 |

\* `F2 Score` is computed with `fbeta_score(beta=2)`.

\** ROC AUC is computed from predicted probabilities using `predict_proba`.

---

## Confusion Matrices

### Italy on India 375

```
[[166 134]
 [ 43  32]]
```

- True Negatives: 166
- False Positives: 134
- False Negatives: 43
- True Positives: 32
- Test Set Composition: 300 negatives, 75 positives

### Italy on India 600

```
[[154 146]
 [153 147]]
```

- True Negatives: 154
- False Positives: 146
- False Negatives: 153
- True Positives: 147
- Test Set Composition: 300 negatives, 300 positives

### Italy on Brazil

```
[[5491 3280]
 [ 847 2298]]
```

- True Negatives: 5,491
- False Positives: 3,280
- False Negatives: 847
- True Positives: 2,298
- Test Set Composition: 8,771 negatives, 3,145 positives

---

## Key Observations

- **The internal result establishes that the Italy model is genuinely discriminative.** A held-out ROC AUC of `0.7910` and balanced accuracy of `0.7285` on 278 unseen samples confirms the 11-feature panel carries real signal in the Italian cohort. The previous full-dataset fit provided no such evidence.
- **Transfer to Brazil retains most of that discrimination.** External ROC AUC on Brazil is `0.7403` against an internal `0.7910`, a drop of `0.051` — meaningful but modest, and balanced accuracy holds at `0.6784`.
- **Transfer to India is at chance.** ROC AUC is `0.4946` on India 375 and `0.5429` on India 600, with balanced accuracy of `0.4900` and `0.5017`. The India 375 AUC is fractionally below `0.5`, meaning the model's ranking carries no usable information on that cohort. Because the same model reaches `0.79` internally, this is a property of the India transfer, not of the model.
- **The Italy model is sensitivity-leaning where the Brazil model is specificity-leaning.** Internal sensitivity is `0.8170` against specificity `0.6400`, a direct consequence of the near-balanced Italian cohort. On Brazil this produces `0.7307` sensitivity at the cost of 3,280 false positives.
- **Effect of tuning on external metrics was mixed.** ROC AUC improved on Brazil (`0.7314 -> 0.7403`) but moved slightly against the previous default-hyperparameter model on the Indian cohorts (`0.5172 -> 0.4946` on India 375, `0.5787 -> 0.5429` on India 600). Both the old and new values sit close enough to `0.5` that the difference is within the noise of a chance-level transfer; neither supports a usable India prediction. Brier scores improved substantially across all three external sets, reflecting the better-calibrated shallower model.

---

## Files Generated

- `Models/italy_model.pkl` - final model, fitted on 100% of the Italy cohort
- `Models/italy_model_holdout.pkl` - same configuration fitted on the 80% dev set, retained so the internal metrics stay reproducible
- `Results/Italy_internal_metrics.csv`
- `Results/Italy_internal_test_preds.csv`
- `Results/Italy_tuning_results.csv`
- `Results/Italy_metrics.csv`
- `Results/2aIndiaItaly375.csv`
- `Results/2bIndiaItaly600.csv`
- `Results/2cBrazilItaly.csv`
- `Results/SimilarityItaly.csv`

---

## Notes

- All values above were taken from the saved notebook outputs and exported prediction files.
- Confusion matrices in this document were reconstructed from the saved `y_true` and `y_proba` files using the notebook's default `0.5` decision threshold.
- The internal metrics describe the model trained on the 80% dev set. The model shipped for external validation is the same configuration refit on 100% of the cohort, which is standard practice but means the internal metrics are a slightly conservative estimate of the final model's in-domain performance.
- `Data/Dataset-2a.csv` arrives already z-scored across the full Italy cohort, so the train/validation/test splits inherit scaling statistics computed over all rows. This is a mild optimistic bias that cannot be removed without the raw, unstandardized values.
- **The validation set holds only 222 samples**, so single-holdout hyperparameter selection is noticeably noisier here than for Brazil. The top five configurations sit within `0.01` accuracy of each other (see `Results/Italy_tuning_results.csv`), which is inside the resolution of a 222-sample estimate. The grid was deliberately kept coarse at 27 candidates to limit selection on noise, but this remains the main statistical limitation of the Italy internal validation.
- Four of the eleven features (`Lymphocytes(%)`, `Monocytes(%)`, `Eosinophils(%)`, `Basophils(%)`) contain 14 missing values each. These are passed through to XGBoost, which handles them natively via its default-direction split learning.
