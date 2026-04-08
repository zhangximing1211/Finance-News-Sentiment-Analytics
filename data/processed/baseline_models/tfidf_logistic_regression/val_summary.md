# Val

- Split: `val`
- Rows: 484
- Accuracy: 0.7521
- Macro F1: 0.6549
- Weighted F1: 0.7285
- Expected Calibration Error: 0.0722
- Multiclass Brier Score: 0.3562
- Low-confidence threshold: 0.5
- Coverage after abstain: 0.936
- Abstain rate: 0.064
- Retained accuracy: 0.777
- Review queue size: 46
- Neutral boundary sample count: 31

## Per-class Metrics

| label | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.7812 | 0.4098 | 0.5376 | 61 |
| neutral | 0.7340 | 0.9617 | 0.8326 | 287 |
| positive | 0.8289 | 0.4632 | 0.5943 | 136 |

## Confusion Matrix

| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 25 | 32 | 4 |
| neutral | 2 | 276 | 9 |
| positive | 5 | 68 | 63 |

## Decision Confusion Matrix

| actual \ decision | negative | neutral | positive | abstain |
| --- | ---: | ---: | ---: | ---: |
| negative | 23 | 27 | 3 | 8 |
| neutral | 1 | 272 | 5 | 9 |
| positive | 4 | 61 | 57 | 14 |

## Threshold Selection

Threshold chosen to improve retained accuracy while keeping at least 70% coverage.

## Notes

Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.
Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.