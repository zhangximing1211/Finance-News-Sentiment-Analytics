from __future__ import annotations

from pathlib import Path
import json
import sys


BASE_DIR = Path(__file__).resolve().parents[3]
TRAINER_SRC = BASE_DIR / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service import train_baseline_candidates


def main() -> None:
    result = train_baseline_candidates()
    compact_result = {
        "best_model_name": result["best_model_name"],
        "best_model_path": result["best_model_path"],
        "best_metadata_path": result["best_metadata_path"],
        "comparison_csv": result["comparison_csv"],
        "candidate_results": {
            model_name: {
                "validation_summary": {
                    "accuracy": payload["validation_metrics"]["accuracy"],
                    "macro_f1": payload["validation_metrics"]["macro_f1"],
                    "weighted_f1": payload["validation_metrics"]["weighted_f1"],
                    "ece": payload["validation_metrics"]["calibration"]["expected_calibration_error"],
                    "low_confidence_threshold": payload["validation_metrics"]["abstain_policy"]["low_confidence_threshold"],
                    "coverage": payload["validation_metrics"]["abstain_policy"]["coverage"],
                    "abstain_rate": payload["validation_metrics"]["abstain_policy"]["abstain_rate"],
                    "review_queue_size": payload["validation_metrics"]["review_queue_summary"]["queue_size"],
                    "neutral_boundary_sample_count": payload["validation_metrics"]["neutral_boundary_analysis"]["boundary_sample_count"],
                    "misclassified_count": payload["validation_metrics"]["error_analysis"]["misclassified_count"],
                },
                "validation_artifacts": payload["validation_artifacts"],
            }
            for model_name, payload in result["candidate_results"].items()
        },
    }
    print(json.dumps(compact_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
