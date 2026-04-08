from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINER_SRC = ROOT / "services" / "trainer" / "src"

sys.path.insert(0, str(TRAINER_SRC))

from trainer_service.baselines import LABEL_ORDER, evaluate_classifier


class DummyModel:
    def __init__(self, predictions: list[str]) -> None:
        self._predictions = predictions

    def predict(self, texts: pd.Series) -> list[str]:
        return self._predictions[: len(texts)]


class BaselineEvaluationTests(unittest.TestCase):
    def test_evaluate_classifier_returns_confusion_matrix_and_per_class_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d", "e", "f"],
                "label": ["negative", "neutral", "positive", "neutral", "positive", "negative"],
            }
        )
        model = DummyModel(["negative", "negative", "positive", "neutral", "neutral", "negative"])

        metrics = evaluate_classifier(model, frame, split_name="test")

        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["confusion_matrix"]["labels"], LABEL_ORDER)
        self.assertEqual(len(metrics["confusion_matrix"]["rows"]), 3)
        self.assertEqual(set(metrics["per_class_metrics"].keys()), set(LABEL_ORDER))
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)


if __name__ == "__main__":
    unittest.main()
