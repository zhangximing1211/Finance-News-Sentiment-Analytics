# Bert Test

- Split: `test`
- Rows: 484
- Accuracy: 0.907
- Macro F1: 0.8975
- Weighted F1: 0.9072
- Expected Calibration Error: 0.0444
- Multiclass Brier Score: 0.1562
- Low-confidence threshold: 0.66
- Coverage after abstain: 0.9649
- Abstain rate: 0.0351
- Retained accuracy: 0.9143
- Review queue size: 17
- Neutral boundary sample count: 1

## Per-class Metrics

| label | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8406 | 0.9508 | 0.8923 | 61 |
| neutral | 0.9359 | 0.9164 | 0.9261 | 287 |
| positive | 0.8806 | 0.8676 | 0.8741 | 136 |

## Confusion Matrix

| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 58 | 2 | 1 |
| neutral | 9 | 263 | 15 |
| positive | 2 | 16 | 118 |

## Decision Confusion Matrix

| actual \ decision | negative | neutral | positive | abstain |
| --- | ---: | ---: | ---: | ---: |
| negative | 55 | 2 | 1 | 3 |
| neutral | 9 | 258 | 12 | 8 |
| positive | 2 | 14 | 114 | 6 |

## Threshold Selection

Threshold chosen to improve retained accuracy while keeping at least 70% coverage.

## Notes

Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.
Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.