from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_predict


BASE_DIR = Path(__file__).resolve().parents[4]
UTILS_DIR = BASE_DIR / "packages" / "utils" / "python"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from finance_utils import build_sentiment_pipeline


RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "all-data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "eda" / "data_audit.ipynb"
DATA_DESCRIPTION_PATH = PROCESSED_DIR / "data_description.md"
AUDIT_SUMMARY_PATH = PROCESSED_DIR / "audit_summary.json"

RAW_LABEL_TO_LABEL = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

MANUAL_NOISE_REVIEW_NOTES = {
    1861: "更像 neutral。只是可访问性/可转移性描述，没有明显业绩改善或事件催化。",
    4572: "更像 negative。利润下降通常比销售增长更主导情绪判断。",
    1954: "更像 neutral。属于项目改造说明，积极语义不强。",
    3523: "更像 negative。人员减少 158 人通常应归为负面。",
    711: "偏 neutral。融资用途是常规资本安排，正面幅度有限。",
    658: "更像 positive。运营成本下降通常是明确利好。",
    2286: "偏 positive 但很弱。利润同比微增，值得保留人工复核。",
    471: "更像 positive。提升持股比例通常是偏正面的公司动作。",
    1473: "疑似噪声或截断样本。文本过短，不适合稳定标注。",
    3563: "更像 negative。EPS 为亏损通常不应标为 neutral。",
    4082: "更像 positive。经营现金流从大幅负值转为正值，是明显改善。",
    4610: "更像 neutral。市场收盘摘要混合涨跌，不应简单归为 negative。",
}

TEXT_REPLACEMENTS = [
    ("\r\n", "\n"),
    ("\r", "\n"),
    ("+\x88  x201a -õ", " EUR "),
    ("+â", " EUR "),
    ("+\x88EUR TM s", "'s"),
    ("+\x88 s", "'s"),
    ("-\x93 s", "'s"),
    ("-\x93", "'"),
    ("-\x8b ;", " "),
    ("-\x8b", " "),
    ("+Æ", "ä"),
    ("+_", "å"),
    ("+¦", "ö"),
    ("+\x97", "ê"),
    ("+è", "á"),
    ("£", ""),
    ("ð", "u"),
    ("``", '"'),
    ("''", '"'),
]

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
NORMALIZE_FOR_DEDUP_PATTERN = re.compile(r"\s+")
ARTIFACT_MARKER_PATTERN = re.compile(r"\+Æ|\+_|\+¦|\+\x88|-\x93|-\x8b|\+\x97|\+è|\+â|£|ð|``|''")


def read_raw_csv(path: str | Path = RAW_DATA_PATH, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    raw_path = Path(path)
    decoded = raw_path.read_bytes().decode(encoding)
    normalized_newlines = decoded.replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(io.StringIO(normalized_newlines))
    rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {raw_path}.")

    width_set = {len(row) for row in rows}
    if width_set != {2}:
        raise ValueError(f"Unexpected CSV shape in {raw_path}: observed row widths {sorted(width_set)}")

    frame = pd.DataFrame(rows, columns=["label_raw", "text_raw"])
    frame.insert(0, "source_row_id", range(len(frame)))
    return frame


def clean_text(text: str) -> tuple[str, dict[str, bool]]:
    value = str(text)
    flags = {
        "had_artifact_marker": bool(ARTIFACT_MARKER_PATTERN.search(value)),
        "had_control_char": bool(CONTROL_CHAR_PATTERN.search(value)),
        "had_quote_normalization": ("``" in value) or ("''" in value),
    }

    cleaned = value
    for source, target in TEXT_REPLACEMENTS:
        cleaned = cleaned.replace(source, target)

    cleaned = CONTROL_CHAR_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned, flags


def build_dedup_key(text: str) -> str:
    lowered = text.casefold().strip()
    lowered = NORMALIZE_FOR_DEDUP_PATTERN.sub(" ", lowered)
    return lowered


def prepare_dataset(
    path: str | Path = RAW_DATA_PATH,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df = read_raw_csv(path)

    unknown_labels = sorted(set(raw_df["label_raw"]) - set(RAW_LABEL_TO_LABEL))
    if unknown_labels:
        raise ValueError(f"Unknown labels in raw dataset: {unknown_labels}")

    raw_df["label"] = raw_df["label_raw"].map(RAW_LABEL_TO_LABEL)
    raw_df["label_id"] = raw_df["label"].map(LABEL_TO_ID)

    cleaned_texts: list[str] = []
    artifact_flags: list[bool] = []
    control_flags: list[bool] = []
    quote_flags: list[bool] = []

    for value in raw_df["text_raw"]:
        cleaned, flags = clean_text(value)
        cleaned_texts.append(cleaned)
        artifact_flags.append(flags["had_artifact_marker"])
        control_flags.append(flags["had_control_char"])
        quote_flags.append(flags["had_quote_normalization"])

    working_df = raw_df.copy()
    working_df["text"] = cleaned_texts
    working_df["had_artifact_marker"] = artifact_flags
    working_df["had_control_char"] = control_flags
    working_df["had_quote_normalization"] = quote_flags
    working_df["char_length"] = working_df["text"].str.len()
    working_df["word_count"] = working_df["text"].str.split().map(len)
    working_df["dedup_key"] = working_df["text"].map(build_dedup_key)

    working_df = working_df[working_df["text"] != ""].copy()

    exact_duplicate_mask = working_df.duplicated(subset=["label", "dedup_key"], keep="first")
    exact_duplicates_removed = working_df.loc[exact_duplicate_mask].copy()
    deduped_df = working_df.loc[~exact_duplicate_mask].copy()

    conflicting_mask = deduped_df.groupby("dedup_key")["label"].transform("nunique") > 1
    conflicting_rows = deduped_df.loc[conflicting_mask].copy()
    clean_df = deduped_df.loc[~conflicting_mask].copy()

    clean_df = clean_df.sort_values("source_row_id").reset_index(drop=True)
    clean_df.insert(0, "sample_id", range(len(clean_df)))

    train_df, test_df = train_test_split(
        clean_df,
        test_size=0.10,
        random_state=random_state,
        stratify=clean_df["label"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=(1.0 / 9.0),
        random_state=random_state,
        stratify=train_df["label"],
    )

    split_frames = {
        "train": train_df.copy(),
        "val": val_df.copy(),
        "test": test_df.copy(),
    }

    processed_frames: dict[str, pd.DataFrame] = {}
    split_stats: dict[str, Any] = {}
    for split_name, frame in split_frames.items():
        ordered = frame.sort_values("sample_id").copy()
        ordered["split"] = split_name
        ordered = ordered[
            [
                "sample_id",
                "source_row_id",
                "split",
                "label_raw",
                "label",
                "label_id",
                "text",
                "text_raw",
                "had_artifact_marker",
                "had_control_char",
                "had_quote_normalization",
                "char_length",
                "word_count",
            ]
        ].reset_index(drop=True)
        processed_frames[split_name] = ordered
        split_stats[split_name] = {
            "rows": int(len(ordered)),
            "label_distribution": {
                label: int(count) for label, count in ordered["label"].value_counts().sort_index().items()
            },
        }

    split_df = pd.concat(processed_frames.values(), ignore_index=True)

    audit_summary = {
        "raw": {
            "path": str(Path(path)),
            "rows": int(len(raw_df)),
            "label_distribution": {
                label: int(count) for label, count in raw_df["label"].value_counts().sort_index().items()
            },
            "encoding": "ISO-8859-1",
            "csv_reader": "python csv.reader after CR/LF normalization",
        },
        "label_mapping": {
            "raw_to_label": RAW_LABEL_TO_LABEL,
            "label_to_id": LABEL_TO_ID,
        },
        "cleaning": {
            "artifact_rows": int(working_df["had_artifact_marker"].sum()),
            "control_char_rows": int(working_df["had_control_char"].sum()),
            "quote_normalized_rows": int(working_df["had_quote_normalization"].sum()),
            "replacement_rules": [{source: target} for source, target in TEXT_REPLACEMENTS],
        },
        "deduplication": {
            "exact_duplicates_removed": int(len(exact_duplicates_removed)),
            "conflicting_duplicate_rows_removed": int(len(conflicting_rows)),
            "conflicting_duplicate_groups": int(conflicting_rows["dedup_key"].nunique()),
            "rows_after_cleaning": int(len(working_df)),
            "rows_after_deduplication": int(len(clean_df)),
        },
        "splits": split_stats,
        "text_length": {
            "char_min": int(clean_df["char_length"].min()),
            "char_p50": float(clean_df["char_length"].quantile(0.50)),
            "char_p95": float(clean_df["char_length"].quantile(0.95)),
            "char_max": int(clean_df["char_length"].max()),
            "word_min": int(clean_df["word_count"].min()),
            "word_p50": float(clean_df["word_count"].quantile(0.50)),
            "word_p95": float(clean_df["word_count"].quantile(0.95)),
            "word_max": int(clean_df["word_count"].max()),
        },
        "class_imbalance": compute_class_imbalance(clean_df),
        "conflicting_examples": conflicting_rows[
            ["label_raw", "text_raw"]
        ].to_dict(orient="records"),
        "exact_duplicate_examples": exact_duplicates_removed[
            ["label_raw", "text_raw"]
        ].head(10).to_dict(orient="records"),
    }

    return split_df, audit_summary


def compute_class_imbalance(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["label"].value_counts().sort_index()
    majority = int(counts.max())
    minority = int(counts.min())
    ratios = {label: round(int(count) / len(frame), 4) for label, count in counts.items()}
    class_weights = {
        label: round(len(frame) / (len(counts) * int(count)), 4) for label, count in counts.items()
    }
    return {
        "counts": {label: int(count) for label, count in counts.items()},
        "ratios": ratios,
        "majority_to_minority_ratio": round(majority / minority, 3),
        "suggested_balanced_class_weights": class_weights,
    }


def sample_label_noise_candidates(frame: pd.DataFrame, top_k: int = 12) -> list[dict[str, Any]]:
    pipeline = build_sentiment_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        pipeline,
        frame["text"],
        frame["label"],
        cv=cv,
        method="predict_proba",
    )
    classes = list(pipeline.fit(frame["text"], frame["label"]).named_steps["classifier"].classes_)

    prob_df = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])
    scored = pd.concat([frame.reset_index(drop=True), prob_df], axis=1)

    prob_columns = [f"prob_{label}" for label in classes]
    scored["predicted_label"] = scored[prob_columns].idxmax(axis=1).str.replace("prob_", "", regex=False)
    scored["predicted_confidence"] = scored[prob_columns].max(axis=1)
    scored["margin"] = scored.apply(
        lambda row: row[f"prob_{row['predicted_label']}"] - row[f"prob_{row['label']}"], axis=1
    )

    candidates = (
        scored[scored["predicted_label"] != scored["label"]]
        .sort_values(["predicted_confidence", "margin"], ascending=False)
        .head(top_k)
    )

    candidate_records = candidates[
        [
            "sample_id",
            "label",
            "predicted_label",
            "predicted_confidence",
            "margin",
            "text",
        ]
    ].to_dict(orient="records")

    for item in candidate_records:
        item["manual_review_note"] = MANUAL_NOISE_REVIEW_NOTES.get(
            item["sample_id"],
            "需要人工复核，当前无预置备注。",
        )

    return candidate_records


def save_processed_splits(split_df: pd.DataFrame) -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for split_name in ["train", "val", "test"]:
        split_path = PROCESSED_DIR / f"{split_name}.csv"
        subset = split_df[split_df["split"] == split_name].copy()
        subset.to_csv(split_path, index=False, encoding="utf-8")
        output_paths[split_name] = split_path
    return output_paths


def save_audit_summary(summary: dict[str, Any]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return AUDIT_SUMMARY_PATH


def save_data_description(summary: dict[str, Any], noise_candidates: list[dict[str, Any]]) -> Path:
    class_imbalance = summary["class_imbalance"]
    cleaning = summary["cleaning"]
    deduplication = summary["deduplication"]
    splits = summary["splits"]

    lines = [
        "# Data Description",
        "",
        "## Raw Data Ingestion",
        "",
        f"- Source file: `{summary['raw']['path']}`",
        f"- Encoding: `{summary['raw']['encoding']}`",
        f"- Reader strategy: {summary['raw']['csv_reader']}",
        f"- Raw rows: {summary['raw']['rows']}",
        "",
        "## Explicit Label Mapping",
        "",
        "| raw_label | label | label_id |",
        "| --- | --- | ---: |",
    ]
    for raw_label, label in summary["label_mapping"]["raw_to_label"].items():
        lines.append(f"| {raw_label} | {label} | {summary['label_mapping']['label_to_id'][label]} |")

    lines.extend(
        [
            "",
            "## Cleaning Rules",
            "",
            f"- Rows with encoding/markup artifacts: {cleaning['artifact_rows']}",
            f"- Rows with control characters: {cleaning['control_char_rows']}",
            f"- Rows with quote normalization: {cleaning['quote_normalized_rows']}",
            "- Representative fixes:",
            "  - `+Æ -> ä`",
            "  - `+_ -> å`",
            "  - `+¦ -> ö`",
            "  - `+\\x88EUR TM s -> 's`",
            "  - `-\\x93 s -> 's`",
            "  - `+â -> EUR`",
            "",
            "## Deduplication Policy",
            "",
            f"- Exact duplicate rows removed: {deduplication['exact_duplicates_removed']}",
            f"- Conflicting duplicate rows removed: {deduplication['conflicting_duplicate_rows_removed']}",
            f"- Conflicting duplicate groups removed: {deduplication['conflicting_duplicate_groups']}",
            f"- Final rows after cleaning and deduplication: {deduplication['rows_after_deduplication']}",
            "",
            "Exact duplicates with the same label were collapsed to one sample. Texts that appeared with multiple labels were removed from all splits to avoid leakage and label ambiguity.",
            "",
            "## Split Strategy",
            "",
            "- Stratified split with `random_state=42`.",
            "- Ratio: 80% train / 10% val / 10% test.",
            "",
            "| split | rows | negative | neutral | positive |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ["train", "val", "test"]:
        dist = splits[split_name]["label_distribution"]
        lines.append(
            f"| {split_name} | {splits[split_name]['rows']} | {dist.get('negative', 0)} | {dist.get('neutral', 0)} | {dist.get('positive', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Class Imbalance",
            "",
            f"- Counts: {class_imbalance['counts']}",
            f"- Ratios: {class_imbalance['ratios']}",
            f"- Majority/minority ratio: {class_imbalance['majority_to_minority_ratio']}",
            f"- Suggested balanced class weights: {class_imbalance['suggested_balanced_class_weights']}",
            "",
            "Neutral is the majority class, so naive accuracy can be misleading. Any downstream trainer should at least report macro-F1 and consider balanced class weights or resampling.",
            "",
            "## Sampled Label-Noise Candidates",
            "",
            "The following rows are high-confidence disagreements from 5-fold out-of-fold predictions. They are not auto-relabeled, but they should be reviewed before trusting leaderboard-level metrics.",
            "",
            "| sample_id | label | predicted | confidence | excerpt | manual review note |",
            "| ---: | --- | --- | ---: | --- | --- |",
        ]
    )

    for item in noise_candidates:
        excerpt = item["text"][:120].replace("|", " ")
        lines.append(
            f"| {item['sample_id']} | {item['label']} | {item['predicted_label']} | {item['predicted_confidence']:.3f} | {excerpt}... | {item['manual_review_note']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Processed CSVs are UTF-8 encoded.",
            "- `text` is the cleaned training text.",
            "- `text_raw` preserves the original source string for traceability.",
            "- `sample_id` is stable within this processed dataset version.",
        ]
    )

    DATA_DESCRIPTION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DATA_DESCRIPTION_PATH


def build_notebook(summary: dict[str, Any], noise_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def markdown_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    def code_cell(source: str, output_text: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True),
                }
            ],
            "source": source.splitlines(keepends=True),
        }

    noise_table_lines = [
        f"{item['sample_id']:>4} | {item['label']:<8} | {item['predicted_label']:<8} | {item['predicted_confidence']:.3f} | {item['text'][:80]} | {item['manual_review_note']}"
        for item in noise_candidates
    ]
    noise_table = "sample_id | label    | predicted | confidence | text | manual_review_note\n" + "\n".join(noise_table_lines)

    notebook = {
        "cells": [
            markdown_cell(
                "# Data Audit\n\n"
                "This notebook documents raw-data ingestion, cleaning, deduplication, split generation, class imbalance, and label-noise sampling."
            ),
            code_cell(
                "import json\nfrom pathlib import Path\nsummary = json.loads(Path('../../data/processed/audit_summary.json').read_text())\nsummary['raw']",
                json.dumps(summary["raw"], ensure_ascii=False, indent=2),
            ),
            code_cell(
                "summary['label_mapping']",
                json.dumps(summary["label_mapping"], ensure_ascii=False, indent=2),
            ),
            code_cell(
                "summary['cleaning']",
                json.dumps(
                    {
                        "artifact_rows": summary["cleaning"]["artifact_rows"],
                        "control_char_rows": summary["cleaning"]["control_char_rows"],
                        "quote_normalized_rows": summary["cleaning"]["quote_normalized_rows"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            code_cell(
                "summary['deduplication']",
                json.dumps(summary["deduplication"], ensure_ascii=False, indent=2),
            ),
            code_cell(
                "summary['splits']",
                json.dumps(summary["splits"], ensure_ascii=False, indent=2),
            ),
            code_cell(
                "summary['class_imbalance']",
                json.dumps(summary["class_imbalance"], ensure_ascii=False, indent=2),
            ),
            code_cell(
                "print('sample_id | label | predicted | confidence | text')",
                noise_table,
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def save_notebook(summary: dict[str, Any], noise_candidates: list[dict[str, Any]]) -> Path:
    notebook = build_notebook(summary, noise_candidates)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return NOTEBOOK_PATH


def run_full_data_audit(path: str | Path = RAW_DATA_PATH) -> dict[str, Any]:
    split_df, summary = prepare_dataset(path=path)
    noise_candidates = sample_label_noise_candidates(
        pd.concat(
            [
                split_df[split_df["split"] == "train"],
                split_df[split_df["split"] == "val"],
                split_df[split_df["split"] == "test"],
            ],
            ignore_index=True,
        ),
        top_k=12,
    )

    split_paths = save_processed_splits(split_df)
    save_audit_summary(summary)
    save_data_description(summary, noise_candidates)
    save_notebook(summary, noise_candidates)

    return {
        "split_paths": {key: str(value) for key, value in split_paths.items()},
        "audit_summary_path": str(AUDIT_SUMMARY_PATH),
        "data_description_path": str(DATA_DESCRIPTION_PATH),
        "notebook_path": str(NOTEBOOK_PATH),
        "noise_candidates": noise_candidates,
        "summary": summary,
    }
