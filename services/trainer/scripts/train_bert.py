"""Fine-tune FinBERT and evaluate on test split.

Usage::

    python services/trainer/scripts/train_bert.py
    python services/trainer/scripts/train_bert.py --epochs 5 --batch-size 8
    python services/trainer/scripts/train_bert.py --evaluate-only --split test
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[3]
TRAINER_SRC = BASE_DIR / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service.train_bert import evaluate_bert_model, train_finbert


def main() -> None:
    parser = ArgumentParser(description="Fine-tune FinBERT for sentiment classification.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training; only evaluate saved model.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.evaluate_only:
        print(f"[evaluate] split={args.split}")
        result = evaluate_bert_model(split_name=args.split)
        if args.verbose:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            metrics = result["metrics"]
            print(json.dumps({
                "model_path": result["model_path"],
                "split": result["split"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "per_class_metrics": metrics["per_class_metrics"],
                "calibration_ece": metrics["calibration"]["expected_calibration_error"],
                "abstain_policy": {
                    "low_confidence_threshold": metrics["abstain_policy"]["low_confidence_threshold"],
                    "coverage": metrics["abstain_policy"]["coverage"],
                    "retained_accuracy": metrics["abstain_policy"]["retained_accuracy"],
                    "retained_macro_f1": metrics["abstain_policy"]["retained_macro_f1"],
                },
            }, ensure_ascii=False, indent=2))
        return

    hyperparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    print("[train_bert] Starting FinBERT fine-tuning ...")
    result = train_finbert(hyperparams=hyperparams)

    print(f"\n[train_bert] Best epoch: {result['best_epoch']}")
    print(f"[train_bert] Best val macro-F1: {result['best_val_macro_f1']}")
    print(f"[train_bert] Model saved to: {result['model_path']}")
    print(f"[train_bert] Metadata saved to: {result['metadata_path']}")

    # ---- evaluate on test set ---------------------------------------------
    print(f"\n[train_bert] Evaluating on test split ...")
    test_result = evaluate_bert_model(split_name="test")
    test_metrics = test_result["metrics"]

    print(json.dumps({
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_per_class": test_metrics["per_class_metrics"],
        "test_calibration_ece": test_metrics["calibration"]["expected_calibration_error"],
        "test_abstain": {
            "threshold": test_metrics["abstain_policy"]["low_confidence_threshold"],
            "coverage": test_metrics["abstain_policy"]["coverage"],
            "retained_accuracy": test_metrics["abstain_policy"]["retained_accuracy"],
            "retained_macro_f1": test_metrics["abstain_policy"]["retained_macro_f1"],
        },
    }, ensure_ascii=False, indent=2))

    # ---- comparison summary -----------------------------------------------
    print("\n[train_bert] Training complete. Run the following to compare with baseline:")
    print("  make evaluate-baseline   # baseline (TF-IDF + SVM)")
    print("  make evaluate-bert       # FinBERT")


if __name__ == "__main__":
    main()
