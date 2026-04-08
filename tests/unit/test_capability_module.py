from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVING_SRC = ROOT / "services" / "model-serving" / "src"

if str(MODEL_SERVING_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVING_SRC))

from model_serving.capability import CapabilityDecision, build_review_queue_item
from model_serving.capability import load_capability_policy
from model_serving.llm_reviewer import LLMReviewer
from model_serving.secondary_explainer import SecondaryExplainer


class CapabilityModuleTests(unittest.TestCase):
    def test_low_confidence_prediction_abstains_and_enters_review_queue(self) -> None:
        policy = CapabilityDecision(low_confidence_threshold=0.62, neutral_boundary_margin=0.08)
        decision = policy.decide({"negative": 0.24, "neutral": 0.41, "positive": 0.35})

        self.assertTrue(decision["abstained"])
        self.assertEqual(decision["decision_label"], "abstain")
        self.assertIn("low_confidence", decision["review_reasons"])
        self.assertTrue(decision["neutral_boundary"])

        review_item = build_review_queue_item(
            input_text="Example finance text.",
            entities={"companies": ["Example Corp"], "tickers": [], "industry": "technology", "industry_zh": "科技"},
            event={"type": "guidance"},
            decision=decision,
        )
        self.assertIsNotNone(review_item)
        self.assertEqual(review_item["queue"], "human_review")

    def test_capability_policy_supports_environment_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"LOW_CONFIDENCE_THRESHOLD_OVERRIDE": "0.7", "NEUTRAL_BOUNDARY_MARGIN_OVERRIDE": "0.11"},
            clear=False,
        ):
            policy = load_capability_policy(None)

        self.assertEqual(policy.low_confidence_threshold, 0.7)
        self.assertEqual(policy.neutral_boundary_margin, 0.11)

    def test_secondary_explainer_returns_fallback_payload(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            explainer = SecondaryExplainer()
            payload = explainer.generate(
                input_text="Tesla said it will update production targets.",
                sentiment={"label": "neutral", "label_zh": "中性", "confidence": 0.58},
                event={"type": "guidance", "type_zh": "指引更新"},
                entities={"companies": ["Tesla"], "tickers": ["TSLA"], "industry": "technology", "industry_zh": "科技"},
                risk_alert={"message": "建议人工复核：置信度偏低。", "needs_human_review": True, "reasons": ["置信度偏低"]},
                capability_decision={
                    "decision_label": "abstain",
                    "top_label": "neutral",
                    "confidence": 0.58,
                    "review_reasons": ["low_confidence"],
                    "neutral_boundary": True,
                },
            )
        self.assertEqual(payload["provider"], "template_fallback")
        self.assertTrue(payload["llm_ready"])
        self.assertIn("Tesla", payload["summary"])
        self.assertEqual(payload["fallback_reason"], "openai_not_configured")

    @patch("model_serving.secondary_explainer.requests.post")
    def test_secondary_explainer_uses_openai_when_configured(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"summary":"真实二次解释","review_note":"需要人工复核","rationale":"低置信度且存在边界歧义。"}'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"}, clear=False):
            explainer = SecondaryExplainer()
            payload = explainer.generate(
                input_text="Tesla said it will update production targets.",
                sentiment={"label": "neutral", "label_zh": "中性", "confidence": 0.58},
                event={"type": "guidance", "type_zh": "指引更新"},
                entities={"companies": ["Tesla"], "tickers": ["TSLA"], "industry": "technology", "industry_zh": "科技"},
                risk_alert={"message": "建议人工复核：置信度偏低。", "needs_human_review": True, "reasons": ["置信度偏低"]},
                capability_decision={
                    "decision_label": "abstain",
                    "top_label": "neutral",
                    "confidence": 0.58,
                    "review_reasons": ["low_confidence"],
                    "neutral_boundary": True,
                },
            )

        self.assertEqual(payload["provider"], "openai_responses_api")
        self.assertTrue(payload["used_external_llm"])
        self.assertIsNone(payload["fallback_reason"])

    @patch("model_serving.llm_reviewer.requests.post")
    def test_llm_reviewer_uses_openai_when_configured(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"reviewed_label":"negative","reviewed_confidence":0.82,"should_override":true,"review_summary":"LLM 复判为负面","review_rationale":"亏损信号明确。"}'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"}, clear=False):
            reviewer = LLMReviewer()
            payload = reviewer.review(
                input_text="The company reported a sharp loss.",
                sentiment={"label": "neutral", "label_zh": "中性", "confidence": 0.48},
                event={"type": "earnings", "type_zh": "财报"},
                entities={"companies": ["Example Corp"], "tickers": [], "industry": "technology", "industry_zh": "科技"},
                risk_alert={"message": "建议人工复核：置信度偏低。", "needs_human_review": True, "reasons": ["置信度偏低"]},
                capability_decision={
                    "decision_label": "abstain",
                    "top_label": "neutral",
                    "confidence": 0.48,
                    "review_reasons": ["low_confidence"],
                    "neutral_boundary": False,
                },
            )

        self.assertEqual(payload["provider"], "openai_responses_api")
        self.assertTrue(payload["used_external_llm"])
        self.assertTrue(payload["should_override"])


if __name__ == "__main__":
    unittest.main()
