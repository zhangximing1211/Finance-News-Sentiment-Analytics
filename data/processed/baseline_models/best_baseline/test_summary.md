# Test

- Split: `test`
- Rows: 484
- Accuracy: 0.7872
- Macro F1: 0.7391
- Weighted F1: 0.7791
- Expected Calibration Error: 0.0717
- Multiclass Brier Score: 0.308
- Low-confidence threshold: 0.5
- Coverage after abstain: 0.9112
- Abstain rate: 0.0888
- Retained accuracy: 0.8209
- Review queue size: 49
- Neutral boundary sample count: 32

## Per-class Metrics

| label | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.8222 | 0.6066 | 0.6981 | 61 |
| neutral | 0.7874 | 0.9164 | 0.8470 | 287 |
| positive | 0.7714 | 0.5956 | 0.6722 | 136 |

## Confusion Matrix

| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 37 | 18 | 6 |
| neutral | 6 | 263 | 18 |
| positive | 2 | 53 | 81 |

## Decision Confusion Matrix

| actual \ decision | negative | neutral | positive | abstain |
| --- | ---: | ---: | ---: | ---: |
| negative | 34 | 15 | 5 | 7 |
| neutral | 0 | 255 | 16 | 16 |
| positive | 1 | 42 | 73 | 20 |

## Threshold Selection

Threshold chosen to improve retained accuracy while keeping at least 70% coverage.

## Notes

Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.
Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.