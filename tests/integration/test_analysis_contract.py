from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVING_SRC = ROOT / "services" / "model-serving" / "src"

if str(MODEL_SERVING_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVING_SRC))

from model_serving import AnalysisService


class AnalysisContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = AnalysisService()

    def test_response_contains_expected_top_level_keys(self) -> None:
        result = self.service.analyze_text(
            "Tesla (NASDAQ: TSLA) said it will host an investor event next month to discuss factory plans."
        )
        self.assertEqual(
            set(result.keys()),
            {
                "input_text",
                "context",
                "sentiment",
                "event",
                "entities",
                "explanation",
                "risk_alert",
                "review_queue_item",
                "secondary_explanation",
                "metadata",
            },
        )
        self.assertIn("confidence", result["sentiment"])
        self.assertIn("historical_announcements", result["context"])
        self.assertIn("decision_label", result["sentiment"])
        self.assertIn("abstained", result["sentiment"])
        self.assertIn("matched_signals", result["event"])
        self.assertIn("needs_human_review", result["risk_alert"])
        self.assertIn("provider", result["secondary_explanation"])
        self.assertIn("capability_module", result["metadata"])


if __name__ == "__main__":
    unittest.main()
