from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVING_SRC = ROOT / "services" / "model-serving" / "src"

if str(MODEL_SERVING_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVING_SRC))

from model_serving import AnalysisService


class AnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = AnalysisService()

    def test_positive_guidance_signal(self) -> None:
        result = self.service.analyze_text(
            "Apple Inc. (NASDAQ: AAPL) raised its full-year revenue guidance after signing a new supply agreement."
        )
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["event"]["type"], "guidance")
        self.assertIn("AAPL", result["entities"]["tickers"])

    def test_negative_layoff_signal(self) -> None:
        result = self.service.analyze_text(
            "The company announced layoffs after profits fell and orders weakened across its main manufacturing unit."
        )
        self.assertEqual(result["sentiment"]["label"], "negative")
        self.assertEqual(result["event"]["type"], "layoffs")

    def test_negative_lay_off_phrase_maps_to_layoffs_event(self) -> None:
        result = self.service.analyze_text(
            "Amer Sports said it has decided to lay off 370 workers from its Salomon division in France."
        )
        self.assertEqual(result["sentiment"]["label"], "negative")
        self.assertEqual(result["event"]["type"], "layoffs")

    def test_context_override_enriches_entity_event_and_history(self) -> None:
        result = self.service.analyze_text(
            "Management said demand trends were mixed.",
            context={
                "source_name": "Reuters Breakingviews",
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "event_type": "guidance",
                "historical_announcements": [
                    {
                        "announced_at": "2025-12-01",
                        "event_type": "guidance",
                        "summary": "Company raised full-year guidance.",
                    }
                ],
            },
        )
        self.assertEqual(result["context"]["source_name"], "Reuters Breakingviews")
        self.assertEqual(result["event"]["type"], "guidance")
        self.assertTrue(result["entities"]["companies"][0].startswith("Apple Inc"))
        self.assertIn("AAPL", result["entities"]["tickers"])
        self.assertEqual(len(result["context"]["historical_announcements"]), 1)


if __name__ == "__main__":
    unittest.main()
