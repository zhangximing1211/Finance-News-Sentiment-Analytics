from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
TRAINER_SRC = BASE_DIR / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service import evaluate_saved_model


def main() -> None:
    parser = ArgumentParser(description="Evaluate the saved baseline sentiment model.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print the full evaluation payload.")
    args = parser.parse_args()

    result = evaluate_saved_model(split_name=args.split, model_path=args.model_path)
    if args.verbose:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    metrics = result["metrics"]
    compact_result = {
        "model_path": result["model_path"],
        "metadata_path": result["metadata_path"],
        "split": result["split"],
        "artifacts": result["artifacts"],
        "metrics": {
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class_metrics": metrics["per_class_metrics"],
            "confusion_matrix": metrics["confusion_matrix"],
            "calibration": {
                "mean_confidence": metrics["calibration"]["mean_confidence"],
                "expected_calibration_error": metrics["calibration"]["expected_calibration_error"],
                "multiclass_brier_score": metrics["calibration"]["multiclass_brier_score"],
            },
            "abstain_policy": {
                "low_confidence_threshold": metrics["abstain_policy"]["low_confidence_threshold"],
                "coverage": metrics["abstain_policy"]["coverage"],
                "abstain_rate": metrics["abstain_policy"]["abstain_rate"],
                "retained_accuracy": metrics["abstain_policy"]["retained_accuracy"],
                "retained_macro_f1": metrics["abstain_policy"]["retained_macro_f1"],
                "selection_note": metrics["abstain_policy"]["threshold_selection"]["selection_note"],
            },
            "class_imbalance_strategy": metrics["class_imbalance_strategy"],
            "review_queue_summary": metrics["review_queue_summary"],
            "neutral_boundary_analysis": {
                "margin_threshold": metrics["neutral_boundary_analysis"]["margin_threshold"],
                "boundary_sample_count": metrics["neutral_boundary_analysis"]["boundary_sample_count"],
                "actual_label_distribution": metrics["neutral_boundary_analysis"]["actual_label_distribution"],
                "predicted_label_distribution": metrics["neutral_boundary_analysis"]["predicted_label_distribution"],
            },
            "error_analysis": {
                "misclassified_count": metrics["error_analysis"]["misclassified_count"],
                "top_confusion_pairs": metrics["error_analysis"]["top_confusion_pairs"][:10],
            },
        },
    }
    print(json.dumps(compact_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
