# Data Description

## Raw Data Ingestion

- Source file: `/Users/zhangximing/Desktop/Finance News Sentiment Analytics/data/raw/all-data.csv`
- Encoding: `ISO-8859-1`
- Reader strategy: python csv.reader after CR/LF normalization
- Raw rows: 4846

## Explicit Label Mapping

| raw_label | label | label_id |
| --- | --- | ---: |
| negative | negative | 0 |
| neutral | neutral | 1 |
| positive | positive | 2 |

## Cleaning Rules

- Rows with encoding/markup artifacts: 405
- Rows with control characters: 17
- Rows with quote normalization: 303
- Representative fixes:
  - `+Æ -> ä`
  - `+_ -> å`
  - `+¦ -> ö`
  - `+\x88EUR TM s -> 's`
  - `-\x93 s -> 's`
  - `+â -> EUR`

## Deduplication Policy

- Exact duplicate rows removed: 6
- Conflicting duplicate rows removed: 4
- Conflicting duplicate groups removed: 2
- Final rows after cleaning and deduplication: 4836

Exact duplicates with the same label were collapsed to one sample. Texts that appeared with multiple labels were removed from all splits to avoid leakage and label ambiguity.

## Split Strategy

- Stratified split with `random_state=42`.
- Ratio: 80% train / 10% val / 10% test.

| split | rows | negative | neutral | positive |
| --- | ---: | ---: | ---: | ---: |
| train | 3868 | 482 | 2297 | 1089 |
| val | 484 | 61 | 287 | 136 |
| test | 484 | 61 | 287 | 136 |

## Class Imbalance

- Counts: {'negative': 604, 'neutral': 2871, 'positive': 1361}
- Ratios: {'negative': 0.1249, 'neutral': 0.5937, 'positive': 0.2814}
- Majority/minority ratio: 4.753
- Suggested balanced class weights: {'negative': 2.6689, 'neutral': 0.5615, 'positive': 1.1844}

Neutral is the majority class, so naive accuracy can be misleading. Any downstream trainer should at least report macro-F1 and consider balanced class weights or resampling.

## Sampled Label-Noise Candidates

The following rows are high-confidence disagreements from 5-fold out-of-fold predictions. They are not auto-relabeled, but they should be reviewed before trusting leaderboard-level metrics.

| sample_id | label | predicted | confidence | excerpt | manual review note |
| ---: | --- | --- | ---: | --- | --- |
| 1861 | positive | neutral | 0.862 | The iTunes-based material will be accessible on Windows-based or Macintosh computers and transferable to portable device... | 更像 neutral。只是可访问性/可转移性描述，没有明显业绩改善或事件催化。 |
| 4572 | neutral | negative | 0.855 | Operating profit for the 12-month period decreased from EUR9 .6 m while net sales increased from EUR69 .0 m , as compare... | 更像 negative。利润下降通常比销售增长更主导情绪判断。 |
| 1954 | positive | neutral | 0.854 | In complying with the European Water Framework Directive requirements , the pre-treatment unit will be fully renovated ,... | 更像 neutral。属于项目改造说明，积极语义不强。 |
| 3523 | neutral | negative | 0.826 | As a result , the number of personnel in Finland will be reduced by 158 .... | 更像 negative。人员减少 158 人通常应归为负面。 |
| 711 | positive | neutral | 0.810 | Both loans will be used to finance strategic investments such as shopping center redevelopment projects and refinancing ... | 偏 neutral。融资用途是常规资本安排，正面幅度有限。 |
| 658 | positive | negative | 0.804 | Unit costs for flight operations fell by 6.4 percent .... | 更像 positive。运营成本下降通常是明确利好。 |
| 2286 | positive | negative | 0.796 | Operating profit totaled EUR 17.7 mn compared to EUR 17.6 mn in the corresponding period in 2007 .... | 偏 positive 但很弱。利润同比微增，值得保留人工复核。 |
| 471 | positive | neutral | 0.794 | After the transaction , Alma Media raised its stake in Talentum to 30.65 % of the shares and some 31.12 % of voting righ... | 更像 positive。提升持股比例通常是偏正面的公司动作。 |
| 1473 | positive | neutral | 0.792 | " The CHF is a great product .... | 疑似噪声或截断样本。文本过短，不适合稳定标注。 |
| 3563 | neutral | negative | 0.788 | Earnings per share EPS in the first quarter amounted to a loss of EUR0 .20 .... | 更像 negative。EPS 为亏损通常不应标为 neutral。 |
| 4082 | positive | negative | 0.785 | Cash flow from business operations totalled EUR 0.4 mn compared to a negative EUR 15.5 mn in the first half of 2008 .... | 更像 positive。经营现金流从大幅负值转为正值，是明显改善。 |
| 4610 | neutral | negative | 0.783 | - Among other Finnish shares traded in the US , Stora Enso closed 0.33 pct lower at 12.11 eur , UPM-Kymmene was up 0.28 ... | 更像 neutral。市场收盘摘要混合涨跌，不应简单归为 negative。 |

## Notes

- Processed CSVs are UTF-8 encoded.
- `text` is the cleaned training text.
- `text_raw` preserves the original source string for traceability.
- `sample_id` is stable within this processed dataset version.
