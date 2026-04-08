from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKER_SRC = ROOT / "services" / "worker" / "src"

if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from worker_service import ReviewQueueRepository, ReviewQueueWorker


class FakeExplainer:
    external_llm_enabled = True

    def generate(self, **_: object) -> dict[str, object]:
        return {
            "provider": "openai_responses_api",
            "template_path": "fake-template",
            "summary": "二次解释已生成",
            "review_note": "保留在人工复核队列",
            "rationale": "低置信度且靠近 neutral 边界。",
            "llm_ready": True,
            "used_external_llm": True,
            "prompt_available": True,
            "input_excerpt": "Example finance text.",
            "risk_message": "建议人工复核：置信度偏低。",
            "fallback_reason": None,
        }


class ReviewQueueWorkerTests(unittest.TestCase):
    def test_enqueue_and_process_review_queue_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = ReviewQueueRepository(Path(temp_dir) / "review_queue.sqlite3")
            analysis_result = {
                "input_text": "Example finance text.",
                "sentiment": {
                    "label": "neutral",
                    "label_zh": "中性",
                    "confidence": 0.48,
                    "decision_label": "abstain",
                },
                "event": {"type": "guidance", "type_zh": "指引更新"},
                "entities": {"companies": ["Example Corp"], "tickers": [], "industry": "technology", "industry_zh": "科技"},
                "risk_alert": {
                    "needs_human_review": True,
                    "message": "建议人工复核：置信度偏低。",
                    "reasons": ["置信度偏低"],
                },
                "review_queue_item": {
                    "queue": "human_review",
                    "priority": "high",
                    "primary_entity": "Example Corp",
                    "event_type": "guidance",
                    "predicted_label": "neutral",
                    "decision_label": "abstain",
                    "confidence": 0.48,
                    "review_reasons": ["low_confidence", "neutral_boundary"],
                    "recommended_action": "Escalate to analyst review before auto-publishing this classification.",
                    "text_excerpt": "Example finance text.",
                },
                "secondary_explanation": {
                    "provider": "template_fallback",
                    "template_path": "fake-template",
                    "summary": "模板解释",
                    "review_note": "需要人工复核",
                    "rationale": "模板兜底",
                    "llm_ready": True,
                    "used_external_llm": False,
                    "prompt_available": True,
                    "input_excerpt": "Example finance text.",
                    "risk_message": "建议人工复核：置信度偏低。",
                    "fallback_reason": "openai_not_configured",
                },
            }

            persisted = repository.enqueue_analysis(analysis_result)
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted["status"], "pending")

            worker = ReviewQueueWorker(repository=repository, explainer=FakeExplainer())
            result = worker.process_pending(limit=10)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["failed"], 0)

            records = repository.list_items(limit=10)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "ready_for_review")
            self.assertEqual(records[0]["secondary_explanation"]["provider"], "openai_responses_api")


if __name__ == "__main__":
    unittest.main()
