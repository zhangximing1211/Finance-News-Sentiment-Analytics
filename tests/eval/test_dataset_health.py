from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAINER_SRC = ROOT / "services" / "trainer" / "src"

if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from trainer_service import TrainerService


class DatasetHealthTests(unittest.TestCase):
    def test_dataset_contains_three_sentiment_labels(self) -> None:
        trainer = TrainerService()
        summary = trainer.dataset_summary()
        self.assertEqual(set(summary["labels"].keys()), {"negative", "neutral", "positive"})
        self.assertGreater(summary["rows"], 4000)


if __name__ == "__main__":
    unittest.main()
