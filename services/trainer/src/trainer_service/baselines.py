from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


BASE_DIR = Path(__file__).resolve().parents[4]
UTILS_DIR = BASE_DIR / "packages" / "utils" / "python"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BASELINE_DIR = PROCESSED_DIR / "baseline_models"
LABEL_ORDER = ["negative", "neutral", "positive"]
DECISION_LABEL_ORDER = LABEL_ORDER + ["abstain"]
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
NEUTRAL_BOUNDARY_MARGIN = 0.08
THRESHOLD_CANDIDATES = [round(value / 100, 2) for value in range(50, 91, 2)]
CLASS_IMBALANCE_STRATEGY = {
    "type": "class_weight",
    "value": "balanced",
    "note": "Both Logistic Regression and Linear SVM are trained with class_weight='balanced'.",
}

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from finance_utils import MODEL_BUILDERS, build_calibrated_sentiment_pipeline


def load_processed_split(split_name: str, processed_dir: str | Path = PROCESSED_DIR) -> pd.DataFrame:
    split_path = Path(processed_dir) / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Processed split not found: {split_path}")
    return pd.read_csv(split_path)


def _softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _predict_probabilities(model: Any, frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    predictions = list(model.predict(frame["text"]))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame["text"])
        classes = list(getattr(model, "classes_", LABEL_ORDER))
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(frame["text"])
        rows: list[list[float]] = []
        for raw in raw_scores:
            if hasattr(raw, "__len__") and not isinstance(raw, str):
                rows.append(_softmax([float(value) for value in raw]))
            else:
                rows.append([1.0])
        probabilities = rows
        classes = list(getattr(model, "classes_", LABEL_ORDER[: len(rows[0])]))
    else:
        classes = LABEL_ORDER
        probabilities = [
            [1.0 if prediction == label else 0.0 for label in classes]
            for prediction in predictions
        ]

    probability_df = pd.DataFrame(probabilities, columns=classes)
    for label in LABEL_ORDER:
        if label not in probability_df.columns:
            probability_df[label] = 0.0
    probability_df = probability_df[LABEL_ORDER]

    return predictions, probability_df


def _compute_multiclass_brier_score(labels: list[str], probability_df: pd.DataFrame) -> float:
    actual = pd.get_dummies(pd.Categorical(labels, categories=LABEL_ORDER)).astype(float)
    return float(((actual.values - probability_df.values) ** 2).sum(axis=1).mean())


def _compute_calibration(labels: list[str], predictions: list[str], probability_df: pd.DataFrame) -> dict[str, Any]:
    confidence_series = probability_df.max(axis=1)
    correctness = [actual == predicted for actual, predicted in zip(labels, predictions)]
    calibration_rows: list[dict[str, Any]] = []
    expected_calibration_error = 0.0

    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        if index == 9:
            mask = (confidence_series >= lower) & (confidence_series <= upper)
        else:
            mask = (confidence_series >= lower) & (confidence_series < upper)

        bucket_size = int(mask.sum())
        if bucket_size == 0:
            calibration_rows.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_confidence = float(confidence_series[mask].mean())
        bucket_accuracy = float(pd.Series(correctness)[mask].mean())
        gap = abs(bucket_accuracy - avg_confidence)
        expected_calibration_error += (bucket_size / len(confidence_series)) * gap
        calibration_rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": bucket_size,
                "avg_confidence": round(avg_confidence, 4),
                "accuracy": round(bucket_accuracy, 4),
                "gap": round(gap, 4),
            }
        )

    return {
        "mean_confidence": round(float(confidence_series.mean()), 4),
        "expected_calibration_error": round(float(expected_calibration_error), 4),
        "multiclass_brier_score": round(_compute_multiclass_brier_score(labels, probability_df), 4),
        "bins": calibration_rows,
    }


def _select_low_confidence_threshold(labels: list[str], predictions: list[str], confidence_series: pd.Series) -> dict[str, Any]:
    base_accuracy = float(accuracy_score(labels, predictions))
    evaluations: list[dict[str, Any]] = []

    for threshold in THRESHOLD_CANDIDATES:
        keep_mask = confidence_series >= threshold
        coverage = float(keep_mask.mean())
        kept = int(keep_mask.sum())
        if kept == 0:
            retained_accuracy = 0.0
            retained_macro_f1 = 0.0
        else:
            retained_labels = [label for label, keep in zip(labels, keep_mask) if keep]
            retained_predictions = [label for label, keep in zip(predictions, keep_mask) if keep]
            retained_accuracy = float(accuracy_score(retained_labels, retained_predictions))
            retained_macro_f1 = float(
                f1_score(
                    retained_labels,
                    retained_predictions,
                    labels=LABEL_ORDER,
                    average="macro",
                    zero_division=0,
                )
            )

        evaluations.append(
            {
                "threshold": threshold,
                "coverage": round(coverage, 4),
                "abstain_rate": round(1.0 - coverage, 4),
                "retained_accuracy": round(retained_accuracy, 4),
                "retained_macro_f1": round(retained_macro_f1, 4),
                "kept_rows": kept,
            }
        )

    eligible = [
        item
        for item in evaluations
        if item["coverage"] >= 0.70 and item["retained_accuracy"] >= round(base_accuracy + 0.02, 4)
    ]

    if eligible:
        best = max(eligible, key=lambda item: (item["coverage"], item["retained_macro_f1"], item["retained_accuracy"]))
    else:
        best = max(evaluations, key=lambda item: (item["retained_macro_f1"], item["coverage"], item["retained_accuracy"]))

    best["selection_note"] = (
        "Threshold chosen to improve retained accuracy while keeping at least 70% coverage."
        if eligible
        else "No threshold satisfied the stricter retained-accuracy constraint; selected by best retained macro-F1."
    )
    return {
        "base_accuracy": round(base_accuracy, 4),
        "low_confidence_threshold": best["threshold"],
        "selection_note": best["selection_note"],
        "candidates": evaluations,
    }


def _build_decision_confusion_matrix(actual_labels: list[str], decision_labels: list[str]) -> dict[str, Any]:
    actual = pd.Categorical(actual_labels, categories=LABEL_ORDER)
    decisions = pd.Categorical(decision_labels, categories=DECISION_LABEL_ORDER)
    table = pd.crosstab(actual, decisions, dropna=False)
    table = table.reindex(index=LABEL_ORDER, columns=DECISION_LABEL_ORDER, fill_value=0)
    return {
        "labels": DECISION_LABEL_ORDER,
        "rows": table.values.tolist(),
    }


def _build_neutral_boundary_analysis(
    frame: pd.DataFrame,
    predictions: list[str],
    probability_df: pd.DataFrame,
    confidence_series: pd.Series,
    margin: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        ranked = sorted(
            [(label, float(probability_df.iloc[idx][label])) for label in LABEL_ORDER],
            key=lambda item: item[1],
            reverse=True,
        )
        top_label, top_probability = ranked[0]
        second_label, second_probability = ranked[1]
        neutral_in_top_two = "neutral" in {top_label, second_label}
        gap = round(top_probability - second_probability, 4)

        if neutral_in_top_two and gap <= margin:
            records.append(
                {
                    "sample_id": int(row.get("sample_id", idx)),
                    "actual_label": row["label"],
                    "predicted_label": predictions[idx],
                    "confidence": round(float(confidence_series.iloc[idx]), 4),
                    "top_label": top_label,
                    "second_label": second_label,
                    "top_probability": round(top_probability, 4),
                    "second_probability": round(second_probability, 4),
                    "neutral_probability": round(float(probability_df.iloc[idx]["neutral"]), 4),
                    "margin_to_decision": gap,
                    "text": row["text"],
                }
            )

    boundary_df = pd.DataFrame(records)
    if boundary_df.empty:
        return {
            "margin_threshold": margin,
            "boundary_sample_count": 0,
            "actual_label_distribution": {},
            "predicted_label_distribution": {},
            "samples": [],
        }

    boundary_df = boundary_df.sort_values(["margin_to_decision", "confidence"], ascending=[True, True])
    return {
        "margin_threshold": margin,
        "boundary_sample_count": int(len(boundary_df)),
        "actual_label_distribution": {
            key: int(value) for key, value in boundary_df["actual_label"].value_counts().sort_index().items()
        },
        "predicted_label_distribution": {
            key: int(value) for key, value in boundary_df["predicted_label"].value_counts().sort_index().items()
        },
        "samples": boundary_df.head(50).to_dict(orient="records"),
    }


def _build_review_queue(
    frame: pd.DataFrame,
    predictions: list[str],
    probability_df: pd.DataFrame,
    confidence_series: pd.Series,
    threshold: float,
    neutral_boundary_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    boundary_lookup = {
        int(item["sample_id"]): item for item in neutral_boundary_analysis.get("samples", [])
    }
    review_items: list[dict[str, Any]] = []

    for idx, row in frame.reset_index(drop=True).iterrows():
        sample_id = int(row.get("sample_id", idx))
        reasons: list[str] = []
        confidence = float(confidence_series.iloc[idx])
        if confidence < threshold:
            reasons.append("low_confidence")
        if sample_id in boundary_lookup:
            reasons.append("neutral_boundary")
        if not reasons:
            continue

        priority = "high" if "low_confidence" in reasons and confidence < (threshold - 0.05) else "medium"
        review_items.append(
            {
                "sample_id": sample_id,
                "priority": priority,
                "predicted_label": predictions[idx],
                "confidence": round(confidence, 4),
                "reasons": ",".join(reasons),
                "top_negative_probability": round(float(probability_df.iloc[idx]["negative"]), 4),
                "top_neutral_probability": round(float(probability_df.iloc[idx]["neutral"]), 4),
                "top_positive_probability": round(float(probability_df.iloc[idx]["positive"]), 4),
                "text": row["text"],
            }
        )

    return review_items


def _build_error_analysis(
    frame: pd.DataFrame,
    predictions: list[str],
    confidence_series: pd.Series,
    review_queue: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    queue_lookup = {item["sample_id"]: item for item in review_queue}
    error_rows: list[dict[str, Any]] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        if row["label"] == predictions[idx]:
            continue
        sample_id = int(row.get("sample_id", idx))
        review_reasons = queue_lookup.get(sample_id, {}).get("reasons", "")
        error_rows.append(
            {
                "sample_id": sample_id,
                "actual_label": row["label"],
                "predicted_label": predictions[idx],
                "confidence": round(float(confidence_series.iloc[idx]), 4),
                "below_threshold": float(confidence_series.iloc[idx]) < threshold,
                "review_reasons": review_reasons,
                "text": row["text"],
            }
        )

    error_df = pd.DataFrame(error_rows)
    if error_df.empty:
        return {
            "misclassified_count": 0,
            "top_confusion_pairs": [],
            "samples": [],
        }

    confusion_pairs = (
        error_df.groupby(["actual_label", "predicted_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    return {
        "misclassified_count": int(len(error_df)),
        "top_confusion_pairs": confusion_pairs.to_dict(orient="records"),
        "samples": error_df.sort_values("confidence", ascending=False).head(100).to_dict(orient="records"),
    }


def evaluate_classifier(
    model: Any,
    frame: pd.DataFrame,
    split_name: str,
    low_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    predictions, probability_df = _predict_probabilities(model, frame)
    confidence_series = probability_df.max(axis=1)

    report = classification_report(
        frame["label"],
        predictions,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(frame["label"], predictions, labels=LABEL_ORDER)
    calibration = _compute_calibration(frame["label"].tolist(), predictions, probability_df)
    threshold_selection = _select_low_confidence_threshold(frame["label"].tolist(), predictions, confidence_series)
    threshold = low_confidence_threshold if low_confidence_threshold is not None else threshold_selection["low_confidence_threshold"]

    abstained = confidence_series < threshold
    decision_labels = [prediction if not abstain else "abstain" for prediction, abstain in zip(predictions, abstained)]
    coverage = round(float((~abstained).mean()), 4)
    abstain_rate = round(float(abstained.mean()), 4)

    if (~abstained).sum() > 0:
        kept_labels = frame.loc[~abstained, "label"].tolist()
        kept_predictions = [label for label, abstain in zip(predictions, abstained) if not abstain]
        retained_accuracy = round(float(accuracy_score(kept_labels, kept_predictions)), 4)
        retained_macro_f1 = round(
            float(
                f1_score(
                    kept_labels,
                    kept_predictions,
                    labels=LABEL_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        )
    else:
        retained_accuracy = 0.0
        retained_macro_f1 = 0.0

    neutral_boundary_analysis = _build_neutral_boundary_analysis(
        frame=frame,
        predictions=predictions,
        probability_df=probability_df,
        confidence_series=confidence_series,
        margin=NEUTRAL_BOUNDARY_MARGIN,
    )
    review_queue = _build_review_queue(
        frame=frame,
        predictions=predictions,
        probability_df=probability_df,
        confidence_series=confidence_series,
        threshold=threshold,
        neutral_boundary_analysis=neutral_boundary_analysis,
    )
    error_analysis = _build_error_analysis(
        frame=frame,
        predictions=predictions,
        confidence_series=confidence_series,
        review_queue=review_queue,
        threshold=threshold,
    )

    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in LABEL_ORDER
    }

    return {
        "split": split_name,
        "rows": int(len(frame)),
        "accuracy": round(float(accuracy_score(frame["label"], predictions)), 4),
        "macro_f1": round(float(f1_score(frame["label"], predictions, labels=LABEL_ORDER, average="macro")), 4),
        "weighted_f1": round(float(f1_score(frame["label"], predictions, labels=LABEL_ORDER, average="weighted")), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "rows": matrix.tolist(),
        },
        "decision_confusion_matrix": _build_decision_confusion_matrix(frame["label"].tolist(), decision_labels),
        "calibration": calibration,
        "abstain_policy": {
            "low_confidence_threshold": round(float(threshold), 4),
            "coverage": coverage,
            "abstain_rate": abstain_rate,
            "retained_accuracy": retained_accuracy,
            "retained_macro_f1": retained_macro_f1,
            "threshold_selection": threshold_selection,
        },
        "class_imbalance_strategy": CLASS_IMBALANCE_STRATEGY,
        "review_queue_summary": {
            "queue_size": int(len(review_queue)),
            "high_priority": int(sum(item["priority"] == "high" for item in review_queue)),
            "medium_priority": int(sum(item["priority"] == "medium" for item in review_queue)),
        },
        "neutral_boundary_analysis": neutral_boundary_analysis,
        "error_analysis": error_analysis,
        "predictions": predictions,
        "probabilities": probability_df.to_dict(orient="records"),
        "confidences": [round(float(value), 4) for value in confidence_series.tolist()],
        "review_queue": review_queue,
    }


def _save_metrics_files(
    output_dir: Path,
    prefix: str,
    metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = output_dir / f"{prefix}_metrics.json"
    confusion_csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    decision_confusion_csv_path = output_dir / f"{prefix}_decision_confusion_matrix.csv"
    per_class_csv_path = output_dir / f"{prefix}_per_class_metrics.csv"
    calibration_bins_csv_path = output_dir / f"{prefix}_calibration_bins.csv"
    threshold_candidates_csv_path = output_dir / f"{prefix}_threshold_candidates.csv"
    error_analysis_csv_path = output_dir / f"{prefix}_error_analysis.csv"
    neutral_boundary_csv_path = output_dir / f"{prefix}_neutral_boundary_samples.csv"
    review_queue_csv_path = output_dir / f"{prefix}_review_queue.csv"
    summary_md_path = output_dir / f"{prefix}_summary.md"

    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "probabilities", "confidences", "review_queue"}
    }
    metrics_json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"]["rows"],
        index=metrics["confusion_matrix"]["labels"],
        columns=metrics["confusion_matrix"]["labels"],
    )
    confusion_df.to_csv(confusion_csv_path, encoding="utf-8")

    decision_confusion_df = pd.DataFrame(
        metrics["decision_confusion_matrix"]["rows"],
        index=LABEL_ORDER,
        columns=metrics["decision_confusion_matrix"]["labels"],
    )
    decision_confusion_df.to_csv(decision_confusion_csv_path, encoding="utf-8")

    per_class_df = pd.DataFrame(metrics["per_class_metrics"]).T.reset_index(names="label")
    per_class_df.to_csv(per_class_csv_path, index=False, encoding="utf-8")

    pd.DataFrame(metrics["calibration"]["bins"]).to_csv(calibration_bins_csv_path, index=False, encoding="utf-8")
    pd.DataFrame(metrics["abstain_policy"]["threshold_selection"]["candidates"]).to_csv(
        threshold_candidates_csv_path,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(metrics["error_analysis"]["samples"]).to_csv(error_analysis_csv_path, index=False, encoding="utf-8")
    pd.DataFrame(metrics["neutral_boundary_analysis"]["samples"]).to_csv(
        neutral_boundary_csv_path,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(metrics["review_queue"]).to_csv(review_queue_csv_path, index=False, encoding="utf-8")

    matrix_lines = ["| actual \\ predicted | " + " | ".join(LABEL_ORDER) + " |", "| --- | ---: | ---: | ---: |"]
    for label, row in zip(LABEL_ORDER, metrics["confusion_matrix"]["rows"]):
        matrix_lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")

    decision_matrix_lines = [
        "| actual \\ decision | " + " | ".join(DECISION_LABEL_ORDER) + " |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in zip(LABEL_ORDER, metrics["decision_confusion_matrix"]["rows"]):
        decision_matrix_lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")

    lines = [
        f"# {prefix.replace('_', ' ').title()}",
        "",
        f"- Split: `{metrics['split']}`",
        f"- Rows: {metrics['rows']}",
        f"- Accuracy: {metrics['accuracy']}",
        f"- Macro F1: {metrics['macro_f1']}",
        f"- Weighted F1: {metrics['weighted_f1']}",
        f"- Expected Calibration Error: {metrics['calibration']['expected_calibration_error']}",
        f"- Multiclass Brier Score: {metrics['calibration']['multiclass_brier_score']}",
        f"- Low-confidence threshold: {metrics['abstain_policy']['low_confidence_threshold']}",
        f"- Coverage after abstain: {metrics['abstain_policy']['coverage']}",
        f"- Abstain rate: {metrics['abstain_policy']['abstain_rate']}",
        f"- Retained accuracy: {metrics['abstain_policy']['retained_accuracy']}",
        f"- Review queue size: {metrics['review_queue_summary']['queue_size']}",
        f"- Neutral boundary sample count: {metrics['neutral_boundary_analysis']['boundary_sample_count']}",
        "",
        "## Per-class Metrics",
        "",
        "| label | precision | recall | f1 | support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABEL_ORDER:
        item = metrics["per_class_metrics"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
        )
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "\n".join(matrix_lines),
            "",
            "## Decision Confusion Matrix",
            "",
            "\n".join(decision_matrix_lines),
            "",
            "## Threshold Selection",
            "",
            metrics["abstain_policy"]["threshold_selection"]["selection_note"],
            "",
            "## Notes",
            "",
            CLASS_IMBALANCE_STRATEGY["note"],
            "Neutral boundary samples are defined as rows where neutral is in the top-two probabilities and the probability gap is at most 0.08.",
        ]
    )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "metrics_json": str(metrics_json_path),
        "confusion_matrix_csv": str(confusion_csv_path),
        "decision_confusion_matrix_csv": str(decision_confusion_csv_path),
        "per_class_metrics_csv": str(per_class_csv_path),
        "calibration_bins_csv": str(calibration_bins_csv_path),
        "threshold_candidates_csv": str(threshold_candidates_csv_path),
        "error_analysis_csv": str(error_analysis_csv_path),
        "neutral_boundary_csv": str(neutral_boundary_csv_path),
        "review_queue_csv": str(review_queue_csv_path),
        "summary_md": str(summary_md_path),
    }


def train_baseline_candidates(processed_dir: str | Path = PROCESSED_DIR) -> dict[str, Any]:
    train_df = load_processed_split("train", processed_dir)
    val_df = load_processed_split("val", processed_dir)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    candidate_results: dict[str, Any] = {}

    for model_name in MODEL_BUILDERS:
        model = build_calibrated_sentiment_pipeline(
            model_name=model_name,
            calibration_method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV,
        )
        model.fit(train_df["text"], train_df["label"])
        val_metrics = evaluate_classifier(model, val_df, split_name="val")
        comparison_rows.append(
            {
                "model_name": model_name,
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "weighted_f1": val_metrics["weighted_f1"],
                "ece": val_metrics["calibration"]["expected_calibration_error"],
                "abstain_rate": val_metrics["abstain_policy"]["abstain_rate"],
                "retained_accuracy": val_metrics["abstain_policy"]["retained_accuracy"],
            }
        )

        candidate_dir = BASELINE_DIR / model_name
        metric_paths = _save_metrics_files(candidate_dir, "val", val_metrics)
        candidate_results[model_name] = {
            "validation_metrics": {
                key: value
                for key, value in val_metrics.items()
                if key not in {"predictions", "probabilities", "confidences", "review_queue"}
            },
            "validation_artifacts": metric_paths,
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["macro_f1", "ece", "weighted_f1", "accuracy"],
        ascending=[False, True, False, False],
    )
    comparison_path = BASELINE_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")

    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_threshold = float(
        candidate_results[best_model_name]["validation_metrics"]["abstain_policy"]["low_confidence_threshold"]
    )
    best_model = build_calibrated_sentiment_pipeline(
        model_name=best_model_name,
        calibration_method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
    )
    best_model.fit(train_val_df["text"], train_val_df["label"])

    best_model_path = BASELINE_DIR / "best_baseline.joblib"
    best_metadata_path = BASELINE_DIR / "best_baseline_metadata.json"

    joblib.dump(best_model, best_model_path)
    best_metadata = {
        "model_name": best_model_name,
        "model_family": "capability_module_v0",
        "selection_metric": "macro_f1 on validation split, with ECE as tie-breaker",
        "calibration": {
            "method": CALIBRATION_METHOD,
            "cv": CALIBRATION_CV,
            "validation_ece": candidate_results[best_model_name]["validation_metrics"]["calibration"]["expected_calibration_error"],
        },
        "abstain_policy": {
            "low_confidence_threshold": best_threshold,
            "selection_note": candidate_results[best_model_name]["validation_metrics"]["abstain_policy"]["threshold_selection"]["selection_note"],
        },
        "review_queue_policy": {
            "primary_trigger": "low_confidence",
            "secondary_trigger": "neutral_boundary",
        },
        "neutral_boundary": {
            "margin_threshold": NEUTRAL_BOUNDARY_MARGIN,
        },
        "class_imbalance_strategy": CLASS_IMBALANCE_STRATEGY,
        "comparison": comparison_rows,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_val_rows": int(len(train_val_df)),
        "artifact_path": str(best_model_path),
    }
    best_metadata_path.write_text(json.dumps(best_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "best_model_name": best_model_name,
        "best_model_path": str(best_model_path),
        "best_metadata_path": str(best_metadata_path),
        "comparison_csv": str(comparison_path),
        "candidate_results": candidate_results,
    }


def evaluate_saved_model(
    split_name: str = "test",
    model_path: str | Path | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_path = Path(model_path) if model_path else BASELINE_DIR / "best_baseline.joblib"
    resolved_metadata_path = Path(metadata_path) if metadata_path else BASELINE_DIR / "best_baseline_metadata.json"

    model = joblib.load(resolved_model_path)
    split_df = load_processed_split(split_name, processed_dir)

    low_confidence_threshold = None
    if resolved_metadata_path.exists():
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
        low_confidence_threshold = metadata.get("abstain_policy", {}).get("low_confidence_threshold")

    metrics = evaluate_classifier(
        model,
        split_df,
        split_name=split_name,
        low_confidence_threshold=low_confidence_threshold,
    )

    model_label = resolved_model_path.stem
    output_dir = BASELINE_DIR / model_label
    artifact_paths = _save_metrics_files(output_dir, split_name, metrics)

    return {
        "model_path": str(resolved_model_path),
        "metadata_path": str(resolved_metadata_path) if resolved_metadata_path.exists() else None,
        "split": split_name,
        "artifacts": artifact_paths,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"predictions", "probabilities", "confidences", "review_queue"}
        },
    }
