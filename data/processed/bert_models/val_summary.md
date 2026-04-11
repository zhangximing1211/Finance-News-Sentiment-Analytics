# Val

- Split: `val`
- Rows: 484
- Accuracy: 0.9091
- Macro F1: 0.9016
- Weighted F1: 0.9097
- Expected Calibration Error: 0.0325
- Multiclass Brier Score: 0.1411
- Low-confidence threshold: 0.66
- Coverage after abstain: 0.9421
- Abstain rate: 0.0579
- Retained accuracy: 0.932
- Review queue size: 28
- Neutral boundary sample count: 6

## Per-class Metrics

| label | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8906 | 0.9344 | 0.9120 | 61 |
| neutral | 0.9495 | 0.9164 | 0.9326 | 287 |
| positive | 0.8392 | 0.8824 | 0.8602 | 136 |

## Confusion Matrix

| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 57 | 3 | 1 |
| neutral | 2 | 263 | 22 |
| positive | 5 | 11 | 120 |

## Decision Confusion Matrix

| actual \ decision | negative | neutral | positive | abstain |
| --- | ---: | ---: | ---: | ---: |
| negative | 55 | 3 | 0 | 3 |
| neutral | 2 | 256 | 12 | 17 |
| positive | 4 | 10 | 114 | 8 |

## Threshold Selection

Threshold chosen to improve retained accuracy while keeping at least 70% coverage.

## Notes

Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.
Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.