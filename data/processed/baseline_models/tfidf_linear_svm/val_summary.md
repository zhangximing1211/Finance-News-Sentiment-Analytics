# Val

- Split: `val`
- Rows: 484
- Accuracy: 0.7645
- Macro F1: 0.686
- Weighted F1: 0.7488
- Expected Calibration Error: 0.0663
- Multiclass Brier Score: 0.3393
- Low-confidence threshold: 0.5
- Coverage after abstain: 0.9298
- Abstain rate: 0.0702
- Retained accuracy: 0.7933
- Review queue size: 45
- Neutral boundary sample count: 33

## Per-class Metrics

| label | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8235 | 0.4590 | 0.5895 | 61 |
| neutral | 0.7599 | 0.9373 | 0.8393 | 287 |
| positive | 0.7604 | 0.5368 | 0.6293 | 136 |

## Confusion Matrix

| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 28 | 27 | 6 |
| neutral | 1 | 269 | 17 |
| positive | 5 | 58 | 73 |

## Decision Confusion Matrix

| actual \ decision | negative | neutral | positive | abstain |
| --- | ---: | ---: | ---: | ---: |
| negative | 23 | 23 | 4 | 11 |
| neutral | 0 | 265 | 9 | 13 |
| positive | 3 | 54 | 69 | 10 |

## Threshold Selection

Threshold chosen to improve retained accuracy while keeping at least 70% coverage.

## Notes

Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.
Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.